import ReturnsClass

func main() {
  let fooClass = foo.CxxClass()
  let fooInherited = foo.InheritedCxxClass()

  let barClass = bar.CxxClass()
  let barInherited = bar.InheritedCxxClass()

  let bazClass = bar.baz.CxxClass()
  let bazInherited = bar.baz.InheritedCxxClass()

  print(1) // Set breakpoint here
}
main()

