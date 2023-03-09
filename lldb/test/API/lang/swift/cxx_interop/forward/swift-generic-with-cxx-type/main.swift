import ReturnsClass

struct Wrapper<T> {
  let t: T
}
func main() {
  let classWrapper = Wrapper(t: CxxClass())
  let subclassWrapper = Wrapper(t: CxxSubclass())
  print(1) // Set breakpoint here
}
main()
