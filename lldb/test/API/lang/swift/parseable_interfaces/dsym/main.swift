import AA

func use<T>(_ : T) {}

func foo() {
  let x = MyPoint(x: 10, y: 0)
  let y = MyPoint(x: -2, y: 20)

  use(x.magnitudeSquared) // break here
  use(y.magnitudeSquared)
}

foo()
