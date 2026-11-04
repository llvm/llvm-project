struct GDBServerCommand: Subcommand {
  static let helpText = """
    Usage:
      swift-debugserver gdbserver [--fd <fd>] [--native-regs] [--setsid]

    Runs swift-debugserver in gdb-remote server mode.

    Options:
      --fd <fd>       Communicate over the given file descriptor.
      --native-regs   Use the native register layout.
      --setsid        Run debugserver in a new session.
    """

  var fd: Int32?
  var nativeRegs = false
  var setsid = false

  static func parse(_ args: ArraySlice<String>)
      throws(ArgumentParseError) -> GDBServerCommand {
    var server = GDBServerCommand()
    var remaining = args[...]
    while let arg = remaining.popFirst() {
      // Options may be spelled `--name value` or `--name=value`.
      let parts = arg.split(
        separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
      let name = String(parts[0])
      let inlineValue = parts.count > 1 ? String(parts[1]) : nil

      func value() throws(ArgumentParseError) -> String {
        if let inlineValue { return inlineValue }
        guard let next = remaining.popFirst() else {
          throw .failure("Missing value for \(name)")
        }
        return next
      }

      switch name {
      case "--native-regs" where inlineValue == nil:
        server.nativeRegs = true
      case "--setsid" where inlineValue == nil:
        server.setsid = true
      case "--fd":
        server.fd = try parseFileDescriptor(try value())
      default:
        throw .unexpectedArgument(arg)
      }
    }
    return server
  }

  private static func parseFileDescriptor(_ value: String)
      throws(ArgumentParseError) -> Int32 {
    guard let fd = Int32(value), fd >= 0 else {
      throw .failure("Invalid file descriptor: \(value)")
    }
    return fd
  }

  func run() {
  }
}
