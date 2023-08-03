import Bar
import Framework

func use<T>(_ t: T) {}

func main() {
  let bar = Bar(i: 42)
  f() // break here
  use(bar)
}

main()
