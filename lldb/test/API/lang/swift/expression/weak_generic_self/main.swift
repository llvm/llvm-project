class GenericClass<T> {
  let t: T

  init(t: T) {
    self.t = t
  }

  func foo() {
    { [weak self] in
      print(self) // break here
    }()
  }
}

let instance = GenericClass<Int>(t: 42)
instance.foo()

