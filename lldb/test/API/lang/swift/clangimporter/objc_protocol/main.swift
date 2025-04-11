import P

class ImplementsP : P {}

class C {
  init(p: P) { x = p }
  weak var x : P?
  
  func f() {
    print("break here")
  }
}

C(p: ImplementsP()).f()
