from flask import Flask, request, jsonify
from check import check

check('asshole')
app = Flask(__name__) 

@app.route('/') 
def hello():
  return str(check(request.args.get('user')))

if __name__ == "__main__":
  app.run(host ='0.0.0.0', port = 80, debug = True)
