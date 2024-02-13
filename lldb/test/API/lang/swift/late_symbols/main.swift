func use<T>(_ t: T) {}
func breakpoint(_ object: FromC) {
  print("stop")
}

func main() {
  var object = FromC(i: 23)
  breakpoint(object)
}

main()
