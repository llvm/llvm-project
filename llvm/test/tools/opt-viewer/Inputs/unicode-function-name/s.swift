infix operator •: AdditionPrecedence

func • (a: Int, b: Int) -> Int {
  return a * b
}

@inline(never)
func g(a: Int) -> Int{
  return a + 1
}

let i = g(a: 1 • 2)
