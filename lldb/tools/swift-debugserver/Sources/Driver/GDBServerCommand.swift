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

  var fd: CInt?
  var nativeRegs = false
  var setsid = false

  static func parse(_ args: ArraySlice<String>)
      throws(ArgumentParseError) -> GDBServerCommand {
    var server = GDBServerCommand()
    var remaining = args
    while let arg = remaining.popFirst() {
      // Options may be spelled `--name value` or `--name=value`.
      let parts = arg.split(separator: "=", omittingEmptySubsequences: false)
      guard parts.count <= 2 else {
        throw .failure("Invalid argument: \(arg)")
      }
      let name = String(parts[0])
      let storage = parts.count > 1 ? String(parts[1]) : nil

      func value() throws(ArgumentParseError) -> String {
        if let storage { return storage }
        guard let next = remaining.popFirst() else {
          throw .failure("Missing value for \(name)")
        }
        return next
      }

      switch name {
      case "--native-regs" where storage == nil:
        server.nativeRegs = true
      case "--setsid" where storage == nil:
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
      throws(ArgumentParseError) -> CInt {
    guard let fd = CInt(value), fd >= 0 else {
      throw .failure("Invalid file descriptor: \(value)")
    }
    return fd
  }

  func run() {
  }
}
