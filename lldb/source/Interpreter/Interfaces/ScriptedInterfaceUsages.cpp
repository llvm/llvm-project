//===-- ScriptedInterfaceUsages.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/Interfaces/ScriptedInterfaceUsages.h"

using namespace lldb;
using namespace lldb_private;

void ScriptedInterfaceUsages::Dump(Stream &s, UsageKind kind) const {
  s.IndentMore();
  s.Indent();
  llvm::StringRef usage_kind =
      (kind == UsageKind::CommandInterpreter) ? "Command Interpreter" : "API";
  s << usage_kind << " Usages:";
  const std::vector<llvm::StringRef> &usages =
      (kind == UsageKind::CommandInterpreter) ? GetCommandInterpreterUsages()
                                              : GetSBAPIUsages();
  if (usages.empty())
    s << " None\n";
  else if (usages.size() == 1)
    s << " " << usages.front() << '\n';
  else {
    s << '\n';
    for (llvm::StringRef usage : usages) {
      s.IndentMore();
      s.Indent();
      s << usage << '\n';
      s.IndentLess();
    }
  }
  s.IndentLess();
}
