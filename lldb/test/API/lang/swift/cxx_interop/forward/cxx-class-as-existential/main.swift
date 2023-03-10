import ReturnsClass

protocol P {}

extension CxxClass: P{}

extension InheritedCxxClass: P{}

func main() {
  let x: P = CxxClass()
  let y:P = InheritedCxxClass()
  print(1) // Set breakpoint here
}
main()
