class State {
  init(x: Int) {
    number = x
    print("in class") // break here
  }

  var number:Int
}

func f() {
  print("in function") // break here
}

f()
State(x: 20)
