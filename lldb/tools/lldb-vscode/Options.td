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
