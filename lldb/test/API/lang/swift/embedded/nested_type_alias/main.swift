// A generic class with a nested closure type alias.
final class Storage<Element> {
  typealias Handler = @Sendable (Element) -> Void
  struct State {
    var handler: Handler? = nil
    var flag: Bool = false
  }
  var state = State()
  init() {}
}

func main() {
  let storage = Storage<Int>()
  print("break here")
  _ = storage
}

main()
