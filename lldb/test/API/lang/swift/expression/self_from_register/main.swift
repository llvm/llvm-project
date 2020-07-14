class C : CP {
  let f: Int = 12345
}

protocol CP : class {}

extension CP {
  func foo() {
    print(self)  //% self.expect('e f', substrs=[' = 12345'])
  }
}

C().foo()
