//===-- CommandObjectScript.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_INTERPRETER_COMMANDOBJECTSCRIPT_H
#define LLDB_SOURCE_INTERPRETER_COMMANDOBJECTSCRIPT_H

#include "lldb/Interpreter/CommandObject.h"

namespace lldb_private {

class CommandObjectScript : public CommandObjectRaw {
public:
  CommandObjectScript(CommandInterpreter &interpreter);
  ~CommandObjectScript() override;

protected:
  bool DoExecute(llvm::StringRef command, CommandReturnObject &result) override;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_INTERPRETER_COMMANDOBJECTSCRIPT_H
