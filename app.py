from flask import Flask, render_template, request
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use Agg backend for non-blocking GUI operations

import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
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

def add_awgn_noise(signal, snr_dB):
    # Calculate the signal power
    signal_power = np.sum(np.abs(signal)**2) / len(signal)

    # Calculate the noise power
    noise_power = signal_power / (10**(snr_dB / 10))

    # Generate AWGN noise
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))

    # Add noise to the signal
    noisy_signal = signal + noise

    return noisy_signal

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

    # Convert the signal back to a complex numpy array
    signal = np.frombuffer(unpadded_signal, dtype=np.complex128)

    return signal

def ofdm_receiver(received_signal, fft_size, cyclic_prefix, data):
    # Perform the FFT on the received signal
    frequency_domain_with_prefix = fft(received_signal)
    
    # Remove the cyclic prefix
    frequency_domain = frequency_domain_with_prefix[cyclic_prefix:]
    
    # Perform the inverse FFT
    time_domain = ifft(frequency_domain)
    
    # Discard the padding
    transmitted_data = np.round(time_domain.real[:len(data)]).astype(int)
    
    # Return the transmitted data
    return transmitted_data

def add_awgn_noise_to_encrypted_signal(encrypted_signal, snr_dB):
    # Decode the Base64 encoded encrypted signal
    encrypted_signal_with_iv = base64.b64decode(encrypted_signal)

    # Extract the IV from the encrypted signal
    iv = encrypted_signal_with_iv[:16]

    # Create a cipher object with a dummy key and IV
    cipher = Cipher(algorithms.AES(b'16-byte secret!!'), modes.CBC(iv), backend=default_backend())
    dummy_decryptor = cipher.decryptor()

    # Decrypt the encrypted signal without removing padding
    decrypted_signal_with_padding = dummy_decryptor.update(encrypted_signal_with_iv[16:]) + dummy_decryptor.finalize()

    # Convert the signal back to a complex numpy array
    signal = np.frombuffer(decrypted_signal_with_padding, dtype=np.complex128)

    # Add AWGN noise to the signal
    noisy_signal = add_awgn_noise(signal, snr_dB)

    # Pad the noisy signal to match the block size of AES
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(noisy_signal.tobytes()) + padder.finalize()

    # Encrypt the padded noisy signal
    encryptor = cipher.encryptor()
    encrypted_noisy_signal = encryptor.update(padded_data) + encryptor.finalize()

    # Combine the IV and encrypted noisy signal
    encrypted_noisy_signal_with_iv = iv + encrypted_noisy_signal

    # Base64 encode the result for safe transmission
    encoded_noisy_signal = base64.b64encode(encrypted_noisy_signal_with_iv)

    return encoded_noisy_signal.decode()

def decrypt_noisy_signal(noisy_encrypted_signal, key):
    # Decode the Base64 encoded noisy encrypted signal
    encrypted_noisy_signal_with_iv = base64.b64decode(noisy_encrypted_signal)

    # Extract the IV from the noisy encrypted signal
    iv = encrypted_noisy_signal_with_iv[:16]

    # Create a cipher object with the provided key and IV
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    # Decrypt the noisy encrypted signal
    decrypted_noisy_signal = decryptor.update(encrypted_noisy_signal_with_iv[16:]) + decryptor.finalize()

    # Remove padding from the decrypted noisy signal
    unpadder = padding.PKCS7(128).unpadder()
    unpadded_noisy_signal = unpadder.update(decrypted_noisy_signal) + unpadder.finalize()

    # Convert the noisy signal back to a complex numpy array
    signal = np.frombuffer(unpadded_noisy_signal, dtype=np.complex128)

    return signal


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve form data
        data_str = request.form['data'].strip()
        data = np.array([float(val) for val in data_str.split(',')])

        fft_size = int(request.form['fft_size'])
        cyclic_prefix = int(request.form['cyclic_prefix'])
        # key = Fernet.generate_key()
        key = b'16-byte secret!!' 
        snr_dB = int(request.form['snr_dB'])
        noise_length = len(data)  # Length of the noise (same as the data length)
        noise_power = 1 / (10 ** (snr_dB / 10))
        noise = np.sqrt(noise_power/2) * (np.random.randn(noise_length) + 1j * np.random.randn(noise_length))


        transmitted_signal = ofdm_transmitter(data, fft_size, cyclic_prefix)
        encrypted_signal = encrypt_signal(transmitted_signal, key)
        noisy_encrypted_signal = add_awgn_noise_to_encrypted_signal(encrypted_signal, snr_dB)
        decrypted_noisy_signal = decrypt_noisy_signal(noisy_encrypted_signal, key)
        received_data = ofdm_receiver(decrypted_noisy_signal, fft_size, cyclic_prefix,data)

        # Plotting the signals and data
        fig, axs = plt.subplots(4, 2, figsize=(10, 10))
        fig.suptitle('OFDM Signal Transmission')

        # Transmitted Signal
        axs[0, 0].plot(np.real(transmitted_signal), label='Real')
        axs[0, 0].plot(np.imag(transmitted_signal), label='Imaginary')
        axs[0, 0].set_title('Transmitted Signal')
        axs[0, 0].legend()

        # Encrypted Signal
        axs[0, 1].plot(np.real(decrypt_signal(encrypted_signal, key)), label='Real')
        axs[0, 1].plot(np.imag(decrypt_signal(encrypted_signal, key)), label='Imaginary')
        axs[0, 1].set_title('Encrypted Signal')
        axs[0, 1].legend()

        # Noisy Encrypted Signal
        axs[1, 0].plot(np.real(decrypt_signal(noisy_encrypted_signal, key)), label='Real')
        axs[1, 0].plot(np.imag(decrypt_signal(noisy_encrypted_signal, key)), label='Imaginary')
        axs[1, 0].set_title('Noisy Encrypted Signal')
        axs[1, 0].legend()

        # Decrypted Noisy Signal
        axs[1, 1].plot(np.real(decrypted_noisy_signal), label='Real')
        axs[1, 1].plot(np.imag(decrypted_noisy_signal), label='Imaginary')
        axs[1, 1].set_title('Decrypted Noisy Signal')
        axs[1, 1].legend()

        # Received Data
        axs[2, 0].stem(received_data)
        axs[2, 0].set_title('Received Data')

        # Original Data
        axs[2, 1].stem(data)
        axs[2, 1].set_title('Original Data')

        # AWGN Noise
        axs[3, 0].plot(np.real(noise), label='Real')
        axs[3, 0].plot(np.imag(noise), label='Imaginary')
        axs[3, 0].set_title('AWGN Noise')
        axs[3, 0].legend()


        plot_path = 'static/plot.png'
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        return render_template('result.html',
                               plot_path=plot_path,
                               transmitted_signal=transmitted_signal,
                               noisy_signal=noisy_encrypted_signal,
                               noise=noise,
                               encrypted_signal=encrypted_signal,
                               decrypted_signal=decrypted_noisy_signal,
                               received_data=received_data,
                               transmitted_data=data)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
