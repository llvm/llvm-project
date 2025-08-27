//===-- JITLoaderPythonInterface.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_JITLOADERPYTHONINTERFACE_H
#define LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_JITLOADERPYTHONINTERFACE_H

#include "lldb/Host/Config.h"
#include "lldb/Interpreter/Interfaces/JITLoaderInterface.h"

#if LLDB_ENABLE_PYTHON

#include "ScriptedThreadPythonInterface.h"

#include <optional>

/// Defines a JITLoader interface for Python.
/// 
/// Users can implement a JIT loader in python by implementing a class that
/// named "JITLoaderPlugin" in a module. The path to this module is specified
/// in the settings using:
///
///     (lldb) setting set target.process.python-jit-loader-path <path>
///
/// When the process starts up it will load this module and call methods on the
/// python class. The python class must implement the following methods:
///
/// #---------------------------------------------------------------------------
/// # The class must be named "JITLoaderPlugin" in the python file.
/// #---------------------------------------------------------------------------
/// class JITLoaderPlugin:
///     #-----------------------------------------------------------------------
///     # Construct this object with reference to the process that owns this 
///     # JIT loader.
///     #-----------------------------------------------------------------------
///     def __init__(self, process: lldb.SBProcess):
///         self.process = process
///
///     #-----------------------------------------------------------------------
///     # Called when attaching is completed.
///     #-----------------------------------------------------------------------
///     def did_attach(self):
///         pass
///
///     #-----------------------------------------------------------------------
///     # Called when launching is completed.
///     #-----------------------------------------------------------------------
///     def did_launch(self):
///         pass
///
///     #-----------------------------------------------------------------------
///     # Called once for each module that is loaded into the debug sessions. 
///     # This allows clients to search to symbols or references to JIT'ed 
///     # functions in each module as it gets loaded. Note that this function
///     # can be called prior to did_attach() or did_launch() being called as
///     # libraries get loaded during the attach or launch.
///     #-----------------------------------------------------------------------
///     def module_did_load(self, module: lldb.SBModule):
///         pass
///

namespace lldb_private {
class JITLoaderPythonInterface
    : virtual public JITLoaderInterface,
      virtual public ScriptedPythonInterface,
      public PluginInterface {
public:
  JITLoaderPythonInterface(ScriptInterpreterPythonImpl &interpreter);

  llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(llvm::StringRef class_name, 
                     ExecutionContext &exe_ctx) override;


  llvm::SmallVector<AbstractMethodRequirement>
  GetAbstractMethodRequirements() const override {
    return llvm::SmallVector<AbstractMethodRequirement>(
        {{"did_attach"},
         {"did_launch"},
         {"module_did_load", 1}});
  }
          
  void DidAttach() override;
  void DidLaunch() override;
  void ModulesDidLoad(lldb_private::ModuleList &module_list) override;
                   
  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() {
    return "JITLoaderPythonInterface";
  }

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }
};
} // namespace lldb_private

#endif // LLDB_ENABLE_PYTHON
#endif // LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_JITLOADERPYTHONINTERFACE_H
