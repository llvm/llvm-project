//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_CPLUSPLUS_COMMANDOBJECTCPLUSPLUS_H
#define LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_CPLUSPLUS_COMMANDOBJECTCPLUSPLUS_H

#include "lldb/Interpreter/CommandObjectMultiword.h"
namespace lldb_private {

class CommandObjectCPlusPlusDemangle : public CommandObjectParsed {
public:
  CommandObjectCPlusPlusDemangle(CommandInterpreter &interpreter);

protected:
  void DoExecute(Args &command, CommandReturnObject &result) override;
};

class CommandObjectCPlusPlus : public CommandObjectMultiword {
public:
  CommandObjectCPlusPlus(CommandInterpreter &interpreter);
};

} // namespace lldb_private

#endif
