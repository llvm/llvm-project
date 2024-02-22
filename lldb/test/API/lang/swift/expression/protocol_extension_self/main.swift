class C : CP {
  let f: Int = 12345
}

protocol CP : class {}

extension CP {
  func foo() {
    print(self)  // break here
  }
}

C().foo()
