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
    var subcommandArgs = consume args
    subcommandArgs.removeFirst(2)

    switch subcommandName {
      case "v", "version":
        return version(subcommandArgs)
      default:
        print(helpText)
    }
  }

  static func version(_ args: consuming [String]) {
    var versionCommand: Version
    do {
      versionCommand = try Version.parse(args)
    } catch {
      print(Version.helpText)
      return
    }
    versionCommand.run()
  }
}
