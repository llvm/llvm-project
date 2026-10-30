struct Version: Subcommand {
  static let helpText = """
    Usage:
      swift-debugserver version

    Displays the version of swift-debugserver.
    """

  static func parse(_ args: ArraySlice<String>)
      throws(ArgumentParseError) -> Version {
    if let extra = args.first {
      throw .unexpectedArgument(extra)
    }
    return Version()
  }

  func run() {
    print("swift-debugserver Version 0.0.1")
  }
}
