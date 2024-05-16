import Foo

class A {
  let bridged : BridgedPtr? = nil
}

class B : A {}

func use<T>(_ t: T) {}

func main() {
  let a = A()
  let b = B()
  use(a)
  use(b) // break here
}

main()
