class Base {
  func show() -> Self {
    return self
  }
  let c = 100
  var v = 200
}

func use<T>(_ t: T) {}

class Child : Base {
  override func show() -> Self {
    v += 10
    use((self.c, self.v)) // break here
    return self
  }
}

var child = Child()
child.show()
