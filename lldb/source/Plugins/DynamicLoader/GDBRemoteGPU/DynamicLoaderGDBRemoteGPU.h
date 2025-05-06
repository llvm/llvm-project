//===-- DynamicLoaderGDBRemoteGPU.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_DYNAMICLOADER_STATIC_DYNAMICLOADERGDBREMOTEGPU_H
#define LLDB_SOURCE_PLUGINS_DYNAMICLOADER_STATIC_DYNAMICLOADERGDBREMOTEGPU_H

#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/UUID.h"

/// A dynamic loader class for lldb-server GPU plug-ins.
///
/// GPUs have special requirements for loading and unloading shared libraries
/// and this class implements the DynamicLoader interface to support these
/// targets. The lldb-server GPU plug-ins implement functions that return the
/// information needed to load and unload shared libraries by handling the
/// "jGPUPluginGetDynamicLoaderLibraryInfo" packet. Many GPUs have drivers that
/// coordinate the loading and unloading of shared libraries, but they don't use
/// the standard method of setting a breakpoint in the target and handle the
/// breakpoint callback in the dynamic loader plug-in. Instead, the drivers 
/// have callbacks or notifications that tell the lldb-server GPU plug-in when
/// a shared library is loaded or unloaded. 
class DynamicLoaderGDBRemoteGPU : public lldb_private::DynamicLoader {
public:
  DynamicLoaderGDBRemoteGPU(lldb_private::Process *process);

  // Static Functions
  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "gdb-remote-gpu"; }

  static llvm::StringRef GetPluginDescriptionStatic();

  static lldb_private::DynamicLoader *
  CreateInstance(lldb_private::Process *process, bool force);

  /// Called after attaching a process.
  ///
  /// Allow DynamicLoader plug-ins to execute some code after
  /// attaching to a process.
  void DidAttach() override;

  void DidLaunch() override;

  bool HandleStopReasonDynammicLoader() override;
  
  lldb::ThreadPlanSP GetStepThroughTrampolinePlan(lldb_private::Thread &thread,
                                                  bool stop_others) override;

  lldb_private::Status CanLoadImage() override;

  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

private:
  /// Load all modules by sending a "jGPUPluginGetDynamicLoaderLibraryInfo"
  /// packet to the GDB server.
  ///
  /// \param[in] full
  ///     If true, load all modules. If false, load or unload only new modules.
  ///
  /// \returns True if the GDB server supports the packet named 
  ///     "jGPUPluginGetDynamicLoaderLibraryInfo", false otherwise.
  bool LoadModulesFromGDBServer(bool full);
};

#endif // LLDB_SOURCE_PLUGINS_DYNAMICLOADER_STATIC_DYNAMICLOADERGDBREMOTEGPU_H
