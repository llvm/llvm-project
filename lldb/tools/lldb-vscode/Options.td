include "llvm/Option/OptParser.td"

class F<string name>: Flag<["--", "-"], name>;
class S<string name>: Separate<["--", "-"], name>;
class R<list<string> prefixes, string name>
  : Option<prefixes, name, KIND_REMAINING_ARGS>;

def help: F<"help">,
  HelpText<"Prints out the usage information for the LLDB VSCode tool.">;
def: Flag<["-"], "h">,
  Alias<help>,
  HelpText<"Alias for --help">;

def wait_for_debugger: F<"wait-for-debugger">,
  HelpText<"Pause the program at startup.">;
def: Flag<["-"], "g">,
  Alias<wait_for_debugger>,
  HelpText<"Alias for --wait-for-debugger">;

def port: Separate<["--", "-"], "port">,
  MetaVarName<"<port>">,
  HelpText<"Communicate with the lldb-vscode tool over the defined port.">;
def: Separate<["-"], "p">,
  Alias<port>,
  HelpText<"Alias for --port">;

def launch_target: Separate<["--", "-"], "launch-target">,
  MetaVarName<"<target>">,
  HelpText<"Launch a target for the launchInTerminal request. Any argument "
    "provided after this one will be passed to the target. The parameter "
    "--comm-file must also be specified.">;

def comm_file: Separate<["--", "-"], "comm-file">,
  MetaVarName<"<file>">,
  HelpText<"The fifo file used to communicate the with the debug adaptor "
    "when using --launch-target.">;

def debugger_pid: Separate<["--", "-"], "debugger-pid">,
  MetaVarName<"<pid>">,
  HelpText<"The PID of the lldb-vscode instance that sent the launchInTerminal "
    "request when using --launch-target.">;

def repl_mode: S<"repl-mode">,
  MetaVarName<"<mode>">,
  HelpText<"The mode for handling repl evaluation requests, supported modes: variable, command, auto.">;
