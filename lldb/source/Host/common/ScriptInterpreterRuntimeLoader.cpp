//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/ScriptInterpreterRuntimeLoader.h"
#include "lldb/Host/Config.h"
#include "llvm/Support/ErrorHandling.h"

namespace lldb_private {

ScriptInterpreterRuntimeLoader::~ScriptInterpreterRuntimeLoader() = default;

#if !LLDB_ENABLE_PYTHON
llvm::Expected<ScriptInterpreterRuntimeLoader &>
ScriptInterpreterRuntimeLoader::Get(lldb::ScriptLanguage language) {
  switch (language) {
  case lldb::eScriptLanguagePython:
    return llvm::createStringError(
        "this build of LLDB does not include Python support");
  case lldb::eScriptLanguageLua:
  case lldb::eScriptLanguageNone:
  case lldb::eScriptLanguageUnknown:
    return llvm::createStringError(
        "no runtime loader for the requested script language");
  }
  llvm_unreachable("unhandled ScriptLanguage");
}
#endif

} // namespace lldb_private
