class Container<A, B> {
  var value = 42

  // No generic parameters of its own, so every parameter in scope belongs to
  // the outermost type and the unbound path applies.
  func method() {
    print(value) // break in method
  }

  // Generic parameters of its own, which live at depth 1.
  func genericMethod<X>(_ x: X) {
    print(value) // break in generic method
  }
}

let container = Container<Int, String>()
container.method()
container.genericMethod(0)
