include "llvm/Option/OptParser.td"

class F<string name>: Flag<["--", "-"], name>;
class R<list<string> prefixes, string name>
  : Option<prefixes, name, KIND_REMAINING_ARGS>;

multiclass SJ<string name, string help> {
  def NAME: Separate<["--", "-"], name>,
    HelpText<help>;
  def NAME # _eq: Joined<["--", "-"], name # "=">,
    Alias<!cast<Separate>(NAME)>;
}

def grp_connect : OptionGroup<"connection">, HelpText<"CONNECTION OPTIONS">;

defm listen: SJ<"listen", "Host and port to listen on. Format: [host]:port or protocol://[host]:port (e.g., tcp://localhost:1234, unix:///path/to/socket). Short form: -L">,
  MetaVarName<"<[host]:port>">,
  Group<grp_connect>;
def: Separate<["-"], "L">, Alias<listen>,
  Group<grp_connect>;

defm socket_file: SJ<"socket-file", "Write listening socket information (port number for TCP or path for Unix domain sockets) to the specified file. Short form: -f">,
  MetaVarName<"<path>">,
  Group<grp_connect>;
def: Separate<["-"], "f">, Alias<socket_file>,
  Group<grp_connect>;

defm gdbserver_port: SJ<"gdbserver-port", "Port to use for spawned gdbserver instances. If 0 or unspecified, a port will be chosen automatically. Short form: -P">,
  MetaVarName<"<port>">,
  Group<grp_connect>;
def: Separate<["-"], "P">, Alias<gdbserver_port>,
  Group<grp_connect>;

defm child_platform_fd: SJ<"child-platform-fd", "File descriptor for communication with parent platform process (internal use only).">,
  MetaVarName<"<fd>">,
  Group<grp_connect>,
  Flags<[HelpHidden]>;

def grp_general : OptionGroup<"general options">, HelpText<"GENERAL OPTIONS">;

def server: F<"server">,
  HelpText<"Run in server mode, accepting multiple client connections sequentially. Without this flag, the server exits after handling the first connection.">,
  Group<grp_general>;

defm log_channels: SJ<"log-channels", "Channels to log. A colon-separated list of entries. Each entry starts with a channel followed by a space-separated list of categories. Common channels: lldb, gdb-remote, platform, process. Short form: -c">,
  MetaVarName<"<channel1 categories...:channel2 categories...>">,
  Group<grp_general>;
def: Separate<["-"], "c">, Alias<log_channels>,
  Group<grp_general>;

defm log_file: SJ<"log-file", "Destination file to log to. If empty, log to stderr. Short form: -l">,
  MetaVarName<"<file>">,
  Group<grp_general>;
def: Separate<["-"], "l">, Alias<log_file>,
  Group<grp_general>;

def debug: F<"debug">,
  HelpText<"(Unused, kept for backward compatibility)">,
  Group<grp_general>,
  Flags<[HelpHidden]>;

def verbose: F<"verbose">,
  HelpText<"(Unused, kept for backward compatibility)">,
  Group<grp_general>,
  Flags<[HelpHidden]>;

def help: F<"help">, 
  HelpText<"Display this help message and exit.">,
  Group<grp_general>;
def: Flag<["-"], "h">, Alias<help>,
  Group<grp_general>;

def REM : R<["--"], "">, 
  HelpText<"Arguments to pass to launched gdbserver instances.">,
  MetaVarName<"program args">;
