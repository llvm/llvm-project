#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBDebugger.h"

using namespace lldb;

static SBCommandReturnObject subcommand(SBDebugger &dbg, const char *cmd) {
  SBCommandReturnObject Result;
  dbg.GetCommandInterpreter().HandleCommand(cmd, Result);
  return Result;
}

class CommandCrasher : public SBCommandPluginInterface {
public:
  bool DoExecute(SBDebugger dbg, char **command,
                 SBCommandReturnObject &result) {
    // Test assignment from a different SBCommandReturnObject instance.
    result = subcommand(dbg, "help");
    // Test also whether self-assignment is handled correctly.
    result = result;
    return result.Succeeded();
  }
};

int main() {
  SBDebugger::Initialize();
  SBDebugger dbg = SBDebugger::Create(false /*source_init_files*/);
  SBCommandInterpreter interp = dbg.GetCommandInterpreter();
  static CommandCrasher crasher;
  interp.AddCommand("crasher", &crasher, nullptr /*help*/);
  SBCommandReturnObject Result;
  dbg.GetCommandInterpreter().HandleCommand("crasher", Result);
}
