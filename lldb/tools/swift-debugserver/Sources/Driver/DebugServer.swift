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
      [g]dbserver Run in gdb-remote server mode.
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
      run(VersionCommand.self, subcommandArgs)
    case "g", "gdbserver":
      run(GDBServerCommand.self, subcommandArgs)
    default:
      print(helpText)
    }
  }

  static func run<C: Subcommand>(_: C.Type, _ args: ArraySlice<String>) {
    do {
      try C.parse(args).run()
    } catch {
      switch error {
      case .failure(let message):
        print(message)
      case .unexpectedArgument(let arg):
        print("Unexpected argument: \(arg)")
      }
      print(C.helpText)
    }
  }
}
