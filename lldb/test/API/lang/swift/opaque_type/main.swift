protocol P {}

class C: P {
  let i = 23
}

func f(_ p : some P) {
  print("break here")
}

f(C())
