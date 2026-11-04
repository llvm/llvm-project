protocol Subcommand {
  static var helpText: String { get }

  static func parse(_ args: ArraySlice<String>) throws(ArgumentParseError) -> Self

  func run()
}
