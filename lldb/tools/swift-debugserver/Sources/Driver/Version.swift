internal struct Version {
  static let helpText = """
    Usage:
      swift-debugserver version

    Displays the version of swift-debugserver.
    """

  init() {
  }

  static func parse(_ args: borrowing [String])
      throws(ArgumentParseError) -> Version {
    guard args.isEmpty else {
      throw .failure("Unexpected argument \(args[0])")
    }
    return Version()
  }

  func run() {
    print("swift-debugserver Version 0.0.1")
  }
}
