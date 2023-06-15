from flask import Flask, render_template, request
import numpy as np
from scipy.fftpack import fft, ifft
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import os
import base64

app = Flask(__name__)

def ofdm_transmitter(data, fft_size, cyclic_prefix):
    # Pad the data to match the FFT size
    padded_data = np.concatenate((data, np.zeros(fft_size - len(data))))
    
    # Perform the FFT
    frequency_domain = fft(padded_data)
    
    # Add a cyclic prefix
    prefix = frequency_domain[-cyclic_prefix:]
    frequency_domain_with_prefix = np.concatenate((prefix, frequency_domain))
    
    # Perform the inverse FFT
    time_domain = ifft(frequency_domain_with_prefix)
    
    # Return the time domain signal
    return time_domain

def encrypt_signal(signal, key):
    # Generate a random initialization vector (IV)
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    # Convert the signal to bytes
    signal_bytes = signal.tobytes()

    # Pad the signal to match the block size of AES
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(signal_bytes) + padder.finalize()

    # Encrypt the padded signal
    encrypted_signal = encryptor.update(padded_data) + encryptor.finalize()

    # Combine the IV and encrypted signal
    encrypted_signal_with_iv = iv + encrypted_signal

    # Base64 encode the result for safe transmission
    encoded_signal = base64.b64encode(encrypted_signal_with_iv)
    return encoded_signal.decode()

def decrypt_signal(encoded_signal, key):
    # Decode the Base64 encoded signal
    encrypted_signal_with_iv = base64.b64decode(encoded_signal)

    # Extract the IV from the encrypted signal
    iv = encrypted_signal_with_iv[:16]

    # Create a cipher object with the provided key and IV
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    # Decrypt the encrypted signal
    decrypted_signal = decryptor.update(encrypted_signal_with_iv[16:]) + decryptor.finalize()

    # Remove padding from the decrypted signal
    unpadder = padding.PKCS7(128).unpadder()
    unpadded_signal = unpadder.update(decrypted_signal) + unpadder.finalize()

    # Convert the signal back to a numpy array
    signal = np.frombuffer(unpadded_signal, dtype=np.float64)

    return signal

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         data = np.array(list(map(int, request.form['data'].split(','))))
#         fft_size = int(request.form['fft_size'])
#         cyclic_prefix = int(request.form['cyclic_prefix'])

#         transmitted_signal = ofdm_transmitter(data, fft_size, cyclic_prefix)
#         encrypted_signal = encrypt_signal(transmitted_signal, key)

#         return render_template('result.html', encrypted_signal=encrypted_signal)

#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = np.array(list(map(int, request.form['data'].split(','))))
        fft_size = int(request.form['fft_size'])
        cyclic_prefix = int(request.form['cyclic_prefix'])
        key = b'16-byte secret!!'  # The key should be 16, 24, or 32 bytes long

        transmitted_signal = ofdm_transmitter(data, fft_size, cyclic_prefix)
        encrypted_signal = encrypt_signal(transmitted_signal, key)
        decrypted_signal = decrypt_signal(encrypted_signal, key)
        print(transmitted_signal)
        print(encrypted_signal)
        print(decrypted_signal)

        return render_template('result.html', transmitted_signal=transmitted_signal, encrypted_signal=encrypted_signal, decrypted_signal=decrypted_signal)

    return render_template('index.html')

if __name__ == '__main__':
    # key = b'16-byte secret!!'  # The key should be 16, 24, or 32 bytes long
    app.run(debug=True)
