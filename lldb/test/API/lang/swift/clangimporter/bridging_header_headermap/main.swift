import dylib

func use<T>(_ t: T) {}

class Wrapper {
  let foo = Foo(i: 42)
}

func main() {
  let c = C<Wrapper>(Wrapper())
  c.f() // break here
}

main()
