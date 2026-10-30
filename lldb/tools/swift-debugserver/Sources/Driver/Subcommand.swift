protocol Subcommand {
  static func parse(_ args: ArraySlice<String>) throws(ArgumentParseError) -> Self
  func run()
}
