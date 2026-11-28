protocol Subcommand {
  static var helpText: String { get }

  static func parse(_ args: ArraySlice<String>) throws(ArgumentParseError) -> Self

  func run()
}

extension Subcommand {
  static func run(_ args: ArraySlice<String>) {
    do {
      try parse(args).run()
    } catch {
      switch error {
      case .failure(let message):
        print(message)
      case .unexpectedArgument(let arg):
        print("Unexpected argument: \(arg)")
      }
      print(helpText)
    }
  }
}
