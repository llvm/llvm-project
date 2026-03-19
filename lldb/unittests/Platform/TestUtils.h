//===-- TestUtils.h -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UNITTESTS_PLATFORM_TESTUTILS
#define LLDB_UNITTESTS_PLATFORM_TESTUTILS

#include "lldb/Interpreter/ScriptInterpreter.h"

namespace lldb_private {
class Debugger;

class MockScriptInterpreterPython : public ScriptInterpreter {
public:
  MockScriptInterpreterPython(Debugger &debugger);
  ~MockScriptInterpreterPython() override = default;

  bool ExecuteOneLine(llvm::StringRef command, CommandReturnObject *,
                      const ExecuteScriptOptions &) override {
    return false;
  }

  void ExecuteInterpreterLoop() override {}

  static void Initialize();

  static void Terminate();

  bool IsReservedWord(const char *word) override {
    return llvm::is_contained({"import", "mykeyword_1_1_1"},
                              llvm::StringRef(word));
  }

  static lldb::ScriptInterpreterSP CreateInstance(Debugger &debugger) {
    return std::make_shared<MockScriptInterpreterPython>(debugger);
  }

  static llvm::StringRef GetPluginNameStatic() {
    return "MockScriptInterpreterPython";
  }

  static llvm::StringRef GetPluginDescriptionStatic() {
    return "MockScriptInterpreterPython";
  }

  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }
};

std::string CreateFile(llvm::StringRef filename,
                       llvm::SmallString<128> parent_dir);

} // namespace lldb_private

#endif // LLDB_UNITTESTS_PLATFORM_TESTUTILS_H_IN
