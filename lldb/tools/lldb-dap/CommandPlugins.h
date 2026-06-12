//===-- CommandPlugins.h --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_COMMANDPLUGINS_H
#define LLDB_TOOLS_LLDB_DAP_COMMANDPLUGINS_H

#include "DAP.h"
#include "lldb/API/SBCommandInterpreter.h"

namespace lldb_dap {

struct StartDebuggingCommand : public lldb::SBCommandPluginInterface {
  DAP &dap;
  explicit StartDebuggingCommand(DAP &d) : dap(d) {};
  bool DoExecute(lldb::SBDebugger debugger, char **command,
                 lldb::SBCommandReturnObject &result) override;
};

struct ReplModeCommand : public lldb::SBCommandPluginInterface {
  DAP &dap;
  explicit ReplModeCommand(DAP &d) : dap(d) {};
  bool DoExecute(lldb::SBDebugger debugger, char **command,
                 lldb::SBCommandReturnObject &result) override;
};

struct SendEventCommand : public lldb::SBCommandPluginInterface {
  DAP &dap;
  explicit SendEventCommand(DAP &d) : dap(d) {};
  bool DoExecute(lldb::SBDebugger debugger, char **command,
                 lldb::SBCommandReturnObject &result) override;
};

} // namespace lldb_dap

#endif
