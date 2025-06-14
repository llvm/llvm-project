//===-- JITLoaderPython.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_JITLoaderPython_h_
#define liblldb_JITLoaderPython_h_

#include "lldb/Host/Config.h"

#if LLDB_ENABLE_PYTHON

#include "lldb/Target/JITLoader.h"
#include "lldb/Utility/StructuredData.h"

namespace lldb_private {
class ScriptInterpreter;
}

class JITLoaderPython : public lldb_private::JITLoader {
public:
  JITLoaderPython(lldb_private::Process *process,
                  const lldb_private::FileSpec &python_module_path);

  ~JITLoaderPython() override;

  // Static Functions
  static lldb::JITLoaderSP CreateInstance(lldb_private::Process *process, 
                                          bool force);

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "python"; }

  static llvm::StringRef GetPluginDescriptionStatic();

  // lldb_private::PluginInterface Methods
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  // lldb_private::JITLoader Methods
  void DidAttach() override;
  void DidLaunch() override;
  void ModulesDidLoad(lldb_private::ModuleList &module_list) override;

protected:
  bool IsValid() const {
    return m_script_object_sp && m_script_object_sp->IsValid();
  }

  lldb_private::ScriptInterpreter *m_interpreter = nullptr;
  lldb::JITLoaderInterfaceSP m_interface_sp;
  lldb_private::StructuredData::GenericSP m_script_object_sp;
};

#endif // LLDB_ENABLE_PYTHON

#endif // liblldb_JITLoaderPython_h_
