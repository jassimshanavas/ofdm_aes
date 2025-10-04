from flask import Flask, render_template, request, jsonify, send_file
import json
import sys
from io import StringIO, BytesIO
import traceback
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Store workflow state
workflows = {}
execution_contexts = {}

@app.route('/')
def index():
    return render_template('workflow.html')

@app.route('/api/execute_node', methods=['POST'])
def execute_node():
    """Execute a single node's code and return output"""
    try:
        data = request.json
        node_id = data.get('node_id')
        code = data.get('code', '')
        workflow_id = data.get('workflow_id', 'default')
        input_data = data.get('input_data', {})
        
        # Initialize workflow context if not exists
        if workflow_id not in execution_contexts:
            execution_contexts[workflow_id] = {}
        
        context = execution_contexts[workflow_id]
        
        # Add input data to context
        for key, value in input_data.items():
            context[key] = value
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        # Capture matplotlib plots
        plt.clf()
        plt.close('all')
        
        try:
            # Execute code in context
            exec(code, context)
            
            # Get output
            output_text = captured_output.getvalue()
            
            # Check if matplotlib figure exists
            plot_data = None
            if plt.get_fignums():
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                plot_data = base64.b64encode(buf.read()).decode('utf-8')
                plt.close('all')
            
            # Get variables created (excluding built-ins and imports)
            output_vars = {}
            for key, value in context.items():
                if not key.startswith('_') and key not in ['__builtins__', '__name__', '__doc__']:
                    try:
                        # Try to convert to string representation
                        if hasattr(value, '__module__'):
                            # Skip modules and functions
                            if value.__module__ in ['builtins', 'matplotlib', 'pandas', 'yfinance', 'datetime']:
                                continue
                        
                        str_value = str(value)
                        # Limit string length
                        if len(str_value) > 1000:
                            str_value = str_value[:1000] + '...'
                        output_vars[key] = str_value
                    except:
                        pass
            
            return jsonify({
                'success': True,
                'output': output_text,
                'variables': output_vars,
                'plot': plot_data,
                'node_id': node_id
            })
            
        except Exception as e:
            error_trace = traceback.format_exc()
            return jsonify({
                'success': False,
                'error': str(e),
                'traceback': error_trace,
                'node_id': node_id
            })
        finally:
            sys.stdout = old_stdout
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/save_workflow', methods=['POST'])
def save_workflow():
    """Save workflow configuration"""
    try:
        data = request.json
        workflow_id = data.get('workflow_id', 'default')
        workflows[workflow_id] = data
        return jsonify({'success': True, 'workflow_id': workflow_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/load_workflow/<workflow_id>', methods=['GET'])
def load_workflow(workflow_id):
    """Load workflow configuration"""
    try:
        if workflow_id in workflows:
            return jsonify({'success': True, 'workflow': workflows[workflow_id]})
        else:
            return jsonify({'success': False, 'error': 'Workflow not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/reset_context', methods=['POST'])
def reset_context():
    """Reset execution context"""
    try:
        data = request.json
        workflow_id = data.get('workflow_id', 'default')
        if workflow_id in execution_contexts:
            del execution_contexts[workflow_id]
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/execute_workflow', methods=['POST'])
def execute_workflow():
    """Execute entire workflow in order"""
    try:
        data = request.json
        workflow_id = data.get('workflow_id', 'default')
        nodes = data.get('nodes', [])
        edges = data.get('edges', [])
        
        # Reset context
        execution_contexts[workflow_id] = {}
        
        # Build execution order using topological sort
        execution_order = topological_sort(nodes, edges)
        
        results = []
        for node_id in execution_order:
            node = next((n for n in nodes if n['id'] == node_id), None)
            if node:
                # Get input data from connected nodes
                input_data = {}
                for edge in edges:
                    if edge['target'] == node_id:
                        # Input from source node
                        source_result = next((r for r in results if r['node_id'] == edge['source']), None)
                        if source_result:
                            input_data.update(source_result.get('variables', {}))
                
                # Execute node
                result = execute_node_internal(node_id, node.get('data', {}).get('code', ''), workflow_id, input_data)
                results.append(result)
                
                if not result['success']:
                    break
        
        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

def execute_node_internal(node_id, code, workflow_id, input_data):
    """Internal function to execute a node"""
    try:
        # Initialize workflow context if not exists
        if workflow_id not in execution_contexts:
            execution_contexts[workflow_id] = {}
        
        context = execution_contexts[workflow_id]
        
        # Add input data to context
        for key, value in input_data.items():
            context[key] = value
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        # Capture matplotlib plots
        plt.clf()
        plt.close('all')
        
        try:
            # Execute code in context
            exec(code, context)
            
            # Get output
            output_text = captured_output.getvalue()
            
            # Check if matplotlib figure exists
            plot_data = None
            if plt.get_fignums():
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                plot_data = base64.b64encode(buf.read()).decode('utf-8')
                plt.close('all')
            
            # Get variables created
            output_vars = {}
            for key, value in context.items():
                if not key.startswith('_') and key not in ['__builtins__', '__name__', '__doc__']:
                    try:
                        if hasattr(value, '__module__'):
                            if value.__module__ in ['builtins', 'matplotlib', 'pandas', 'yfinance', 'datetime']:
                                continue
                        
                        str_value = str(value)
                        if len(str_value) > 1000:
                            str_value = str_value[:1000] + '...'
                        output_vars[key] = str_value
                    except:
                        pass
            
            return {
                'success': True,
                'output': output_text,
                'variables': output_vars,
                'plot': plot_data,
                'node_id': node_id
            }
            
        except Exception as e:
            error_trace = traceback.format_exc()
            return {
                'success': False,
                'error': str(e),
                'traceback': error_trace,
                'node_id': node_id
            }
        finally:
            sys.stdout = old_stdout
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'node_id': node_id
        }

def topological_sort(nodes, edges):
    """Topological sort for node execution order"""
    from collections import defaultdict, deque
    
    # Build adjacency list
    graph = defaultdict(list)
    in_degree = {node['id']: 0 for node in nodes}
    
    for edge in edges:
        graph[edge['source']].append(edge['target'])
        in_degree[edge['target']] = in_degree.get(edge['target'], 0) + 1
    
    # Find nodes with no incoming edges
    queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
    result = []
    
    while queue:
        node_id = queue.popleft()
        result.append(node_id)
        
        for neighbor in graph[node_id]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
