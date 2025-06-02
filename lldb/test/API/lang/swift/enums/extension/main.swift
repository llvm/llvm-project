enum EmptyBase {}

extension EmptyBase {
  static var svar = 32
  struct S { let i = 16 }
}

enum Base {
  case baseCase
}
protocol P {}
extension Base : P {}

func getP() -> P { return Base.baseCase }


func main() {
  let s = EmptyBase.S()
  let e = getP()
  print("break here")
}

main()
