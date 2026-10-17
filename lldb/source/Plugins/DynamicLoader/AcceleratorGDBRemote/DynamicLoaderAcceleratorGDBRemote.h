//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_DYNAMICLOADER_ACCELERATORGDBREMOTE_DYNAMICLOADERACCELERATORGDBREMOTE_H
#define LLDB_SOURCE_PLUGINS_DYNAMICLOADER_ACCELERATORGDBREMOTE_DYNAMICLOADERACCELERATORGDBREMOTE_H

#include "lldb/Target/DynamicLoader.h"

/// Dynamic loader for accelerator (e.g. GPU) targets.
///
/// Accelerators don't set a rendezvous breakpoint the way SVR4 loaders do;
/// their runtime tells the lldb-server plugin when libraries load or unload.
/// This loader asks the server for that list via
/// "jAcceleratorPluginGetDynamicLoaderLibraryInfo".
class DynamicLoaderAcceleratorGDBRemote : public lldb_private::DynamicLoader {
public:
  DynamicLoaderAcceleratorGDBRemote(lldb_private::Process *process);

  static void Initialize();
  static void Terminate();
  static llvm::StringRef GetPluginNameStatic() {
    return "accelerator-gdb-remote";
  }
  static llvm::StringRef GetPluginDescriptionStatic();
  static lldb_private::DynamicLoader *
  CreateInstance(lldb_private::Process *process, bool force);

  void DidAttach() override;
  void DidLaunch() override;
  lldb::ThreadPlanSP GetStepThroughTrampolinePlan(lldb_private::Thread &thread,
                                                  bool stop_others) override;
  lldb_private::Status CanLoadImage() override;

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

private:
  /// Returns true if the server answered the packet.
  bool LoadModulesFromGDBServer(bool full);
};

#endif // LLDB_SOURCE_PLUGINS_DYNAMICLOADER_ACCELERATORGDBREMOTE_DYNAMICLOADERACCELERATORGDBREMOTE_H
