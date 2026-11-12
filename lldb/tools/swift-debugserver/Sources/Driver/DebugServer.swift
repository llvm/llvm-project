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
      VersionCommand.run(subcommandArgs)
    case "g", "gdbserver":
      GDBServerCommand.run(subcommandArgs)
    default:
      print(helpText)
    }
  }
}
