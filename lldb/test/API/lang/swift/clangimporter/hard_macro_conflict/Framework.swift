@_implementationOnly import Foo

func use<T>(_ t: T) {}

public func f() {
  let foo = Foo(j: 23)
  use(foo) // break here
}
