//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDFRAMEPROVIDERPYTHONINTERFACE_H
#define LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDFRAMEPROVIDERPYTHONINTERFACE_H

#include "lldb/Host/Config.h"

#if LLDB_ENABLE_PYTHON

#include "ScriptedPythonInterface.h"
#include "lldb/Interpreter/Interfaces/ScriptedFrameProviderInterface.h"
#include <optional>

namespace lldb_private {
class ScriptedFrameProviderPythonInterface
    : public ScriptedFrameProviderInterface,
      public ScriptedPythonInterface {
public:
  ScriptedFrameProviderPythonInterface(
      ScriptInterpreterPythonImpl &interpreter);

  llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(llvm::StringRef class_name,
                     lldb::StackFrameListSP input_frames,
                     StructuredData::DictionarySP args_sp) override;

  llvm::SmallVector<AbstractMethodRequirement>
  GetAbstractMethodRequirements() const override {
    return llvm::SmallVector<AbstractMethodRequirement>(
        {{"get_frame_at_index"}});
  }

  StructuredData::ObjectSP GetFrameAtIndex(uint32_t index) override;
};
} // namespace lldb_private

#endif // LLDB_ENABLE_PYTHON
#endif // LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDFRAMEPROVIDERPYTHONINTERFACE_H
