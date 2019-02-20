//===-- CommandObjectScript.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectScript_h_
#define liblldb_CommandObjectScript_h_

#include "lldb/Interpreter/CommandObject.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectScript
//-------------------------------------------------------------------------

class CommandObjectScript : public CommandObjectRaw {
public:
  CommandObjectScript(CommandInterpreter &interpreter,
                      lldb::ScriptLanguage script_lang);

  ~CommandObjectScript() override;

protected:
  bool DoExecute(llvm::StringRef command, CommandReturnObject &result) override;
};

} // namespace lldb_private

#endif // liblldb_CommandObjectScript_h_
