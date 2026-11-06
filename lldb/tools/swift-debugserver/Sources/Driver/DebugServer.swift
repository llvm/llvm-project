@main
private struct DebugServer {
  static func main() {
    execute(CommandLine.arguments)
  }

  static let helpText = """
    Usage:
      swift-debugserver <subcommand>

    Subcommands:
      [v]ersion   Display version information.
  """

  static func execute(_ args: consuming [String]) {
    guard args.count > 1 else {
      print(helpText)
      return
    }

    let subcommandName = args[1]
    let subcommandArgs = args.dropFirst(2)

    switch subcommandName {
    case "v", "version":
      version(subcommandArgs)
    default:
      print(helpText)
    }
  }

  static func version(_ args: ArraySlice<String>) {
    do {
      try Version.parse(args).run()
    } catch {
      switch error {
      case .failure(let message):
        print(message)
      case .unexpectedArgument(let arg):
        print("Unexpected argument: \(arg)")
      }
      print(Version.helpText)
    }
  }
}
