import Library

func use<T>(_ t: T) {}

func main() {
  let x = getObject()
  let y = getConformingObject()
  use((x, y)) // break here
}

main()
