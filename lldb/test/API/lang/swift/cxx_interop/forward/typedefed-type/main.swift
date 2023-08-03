import ReturnsClass

typealias TypeAliased = CxxClass

func main() {
  let typedef = TypedefedCxxClass()
  let using = UsingCxxClass()
  let typealiased = TypeAliased()
  print(1) // Set breakpoint here
}
main()
