//===-- ScriptInterpreterPython.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_SCRIPTINTERPRETERPYTHON_H
#define LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_SCRIPTINTERPRETERPYTHON_H

#include "lldb/Breakpoint/BreakpointOptions.h"
#include "lldb/Core/IOHandler.h"
#include "lldb/Core/StructuredDataImpl.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/lldb-private.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace lldb_private {
/// Abstract interface for the Python script interpreter.
class ScriptInterpreterPython : public ScriptInterpreter,
                                public IOHandlerDelegateMultiline {
public:
  class CommandDataPython : public BreakpointOptions::CommandData {
  public:
    CommandDataPython() : BreakpointOptions::CommandData() {
      interpreter = lldb::eScriptLanguagePython;
    }
    CommandDataPython(StructuredData::ObjectSP extra_args_sp)
        : BreakpointOptions::CommandData(),
          m_extra_args(std::move(extra_args_sp)) {
      interpreter = lldb::eScriptLanguagePython;
    }
    StructuredDataImpl m_extra_args;
  };

  ScriptInterpreterPython(Debugger &debugger)
      : ScriptInterpreter(debugger, lldb::eScriptLanguagePython),
        IOHandlerDelegateMultiline("DONE") {}

  llvm::Expected<std::string>
  ExtensionToImportPath(lldb::ScriptedExtension extension) override;
  StructuredData::DictionarySP GetInterpreterInfo() override;
  llvm::Expected<FileSpec>
  GenerateExtensionTemplate(const std::string &name,
                            std::vector<ExtensionTemplateRequest> &extensions,
                            bool generate_non_abstract_methods,
                            std::string output_file) override;

  static void Initialize();
  static void Terminate();
  static llvm::StringRef GetPluginNameStatic() { return "script-python"; }
  static llvm::StringRef GetPluginDescriptionStatic();
  static FileSpec GetPythonDir();
  static void SharedLibraryDirectoryHelper(FileSpec &this_file);

protected:
  llvm::Error
  ParseExtensionSchema(Stream &s, llvm::StringRef output_script_prefix,
                       const llvm::SmallVector<llvm::StringRef> &extension_path,
                       bool generate_non_abstract_methods,
                       std::set<std::string> &typing_imports);
  llvm::Expected<StructuredData::ObjectSP>
  GetExtensionSchema(const llvm::SmallVector<llvm::StringRef> &extension_path);

  static void ComputePythonDirForApple(llvm::SmallVectorImpl<char> &path);
  static void ComputePythonDir(llvm::SmallVectorImpl<char> &path);
};
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_SCRIPTINTERPRETERPYTHON_H
