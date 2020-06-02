class Base {
  init() {
    print("Base: c = \(c), v = \(v)") // break here
  }
  func show() -> Self {
    return self
  }
  let c = 100
  var v = 200
}

func use<T>(_ t: T) {}

class Child : Base {
  override init() {
    super.init()
    v += 10
    print("Child: c = \(c), v = \(v)") // break here
  }
  override func show() -> Self {
    v += 10
    use((self.c, self.v)) // break here
    return self
  }
}

var child = Child()
child.show()
