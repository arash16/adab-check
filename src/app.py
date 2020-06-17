from flask import Flask, request, jsonify
from check import check

app = Flask(__name__) 

# hangs untill it's up and running
check('asshole')

# once running, it returns 200
@app.route('/health')
def health():
  return 'health'

@app.route('/') 
def adabCheck():
  return str(check(request.args.get('text')))

if __name__ == "__main__":
  app.run(host ='0.0.0.0', port = 80, debug = True)
