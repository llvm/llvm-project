public class PublicBase {}

private class PrivateSubclass: PublicBase, CustomStringConvertible {
  var description = "Easy as pie"
}

func makeIt() -> PublicBase {
  return PrivateSubclass()
}

func main() {
  let x = makeIt()
  print("break here \(x)")
}

main()
