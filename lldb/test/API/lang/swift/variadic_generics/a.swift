public struct A {
  var i: Int
}
public struct B {
  var d: Double
}

public func variadic_function<each T>(args: repeat each T) {
  print("break here")
}

let a = A(i: 23)
let b = B(d: 2.71)
variadic_function(args: a, b)
