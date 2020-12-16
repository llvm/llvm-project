class C : Foo {
}

protocol Foo: class {
  func foo()
}

extension Foo {
  func foo() {
    print(777) // break here
  }
}

let c = C()
c.foo()