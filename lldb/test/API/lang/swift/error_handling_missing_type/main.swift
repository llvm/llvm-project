func use<T>(_ t: T) {}
func main() {
  var object = (1, FromC(i: 23), 2)
  use(object) // break here
}
main()
