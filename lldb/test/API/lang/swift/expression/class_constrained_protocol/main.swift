class C : CP {
  let f: Int = 12345

  func foo() {
    print(self)  // Break here in method
    let fn = { [weak self] in
      //
      _ = self  // Break here for method weak self
    }

    fn()
  }
}

protocol CP : class {}

extension CP {
  func bar() {
    print(self)  // Break here in class protocol
    let fn = { [weak self] in
      //
      _ = self // Break here for weak self
    }

    fn()
  }
}

C().foo()
C().bar()
