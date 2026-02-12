//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectCPlusPlus.h"

#include "lldb/Core/Mangled.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

CommandObjectCPlusPlusDemangle::CommandObjectCPlusPlusDemangle(
    CommandInterpreter &interpreter)
    : CommandObjectParsed(interpreter, "demangle",
                          "Demangle a C++ mangled name.",
                          "language cplusplus demangle [<mangled-name> ...]") {
  AddSimpleArgumentList(eArgTypeSymbol, eArgRepeatPlus);
}

void CommandObjectCPlusPlusDemangle::DoExecute(Args &command,
                                               CommandReturnObject &result) {
  bool demangled_any = false;
  bool error_any = false;
  for (auto &entry : command.entries()) {
    if (entry.ref().empty())
      continue;

    // the actual Mangled class should be strict about this, but on the
    // command line if you're copying mangled names out of 'nm' on Darwin,
    // they will come out with an extra underscore - be willing to strip this
    // on behalf of the user.   This is the moral equivalent of the -_/-n
    // options to c++filt
    auto name = entry.ref();
    if (name.starts_with("__Z"))
      name = name.drop_front();

    Mangled mangled(name);
    if (mangled.GuessLanguage() == lldb::eLanguageTypeC_plus_plus) {
      ConstString demangled(mangled.GetDisplayDemangledName());
      demangled_any = true;
      result.AppendMessageWithFormat("%s ---> %s\n", entry.c_str(),
                                     demangled.GetCString());
    } else {
      error_any = true;
      result.AppendErrorWithFormat("%s is not a valid C++ mangled name\n",
                                   entry.ref().str().c_str());
    }
  }

  result.SetStatus(
      error_any ? lldb::eReturnStatusFailed
                : (demangled_any ? lldb::eReturnStatusSuccessFinishResult
                                 : lldb::eReturnStatusSuccessFinishNoResult));
}

CommandObjectCPlusPlus::CommandObjectCPlusPlus(CommandInterpreter &interpreter)
    : CommandObjectMultiword(
          interpreter, "cplusplus",
          "Commands for operating on the C++ language runtime.",
          "cplusplus <subcommand> [<subcommand-options>]") {
  LoadSubCommand("demangle", CommandObjectSP(new CommandObjectCPlusPlusDemangle(
                                 interpreter)));
}
