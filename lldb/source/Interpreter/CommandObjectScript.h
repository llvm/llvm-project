//===-- CommandObjectScript.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectScript_h_
#define liblldb_CommandObjectScript_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectScript
//-------------------------------------------------------------------------

class CommandObjectScript : public CommandObjectRaw
{
public:

    CommandObjectScript (CommandInterpreter &interpreter,
                         lldb::ScriptLanguage script_lang);

    virtual
    ~CommandObjectScript ();

protected:
    virtual bool
    DoExecute (const char *command, CommandReturnObject &result);
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectScript_h_
