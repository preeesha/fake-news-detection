from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from joblib import load

# Accuracy score: 0.9865985576923076
predictionModel = load('model.joblib')


app = Flask(__name__)
cors = CORS(app)


@app.route("/", methods=["POST"])
@cross_origin()
def helloWorld():
    body = request.get_json()
    print(body)
    print(type(body))
    if "content" not in body:
        return jsonify({"message": "content is missing"}), 400
    content = body["content"]
    

    isReal = predictionModel.predict([content])[0] == 0
    return jsonify({"real": isReal}), 200


if __name__ == "__main__":
    app.run(debug=True)
