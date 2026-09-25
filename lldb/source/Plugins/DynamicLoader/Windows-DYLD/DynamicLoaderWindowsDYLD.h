//===-- DynamicLoaderWindowsDYLD.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_DYNAMICLOADER_WINDOWS_DYLD_DYNAMICLOADERWINDOWSDYLD_H
#define LLDB_SOURCE_PLUGINS_DYNAMICLOADER_WINDOWS_DYLD_DYNAMICLOADERWINDOWSDYLD_H

#include "lldb/Target/DynamicLoader.h"
#include "lldb/lldb-forward.h"

namespace lldb_private {

class DynamicLoaderWindowsDYLD : public DynamicLoader {
public:
  DynamicLoaderWindowsDYLD(Process *process);

  ~DynamicLoaderWindowsDYLD() override;

  static void Initialize();
  static void Terminate();
  static llvm::StringRef GetPluginNameStatic() { return "windows-dyld"; }
  static llvm::StringRef GetPluginDescriptionStatic();

  static DynamicLoader *CreateInstance(Process *process, bool force);

  void OnLoadModule(lldb::ModuleSP module_sp, const ModuleSpec module_spec,
                    lldb::addr_t module_addr);
  void OnUnloadModule(lldb::addr_t module_addr);

  void DidAttach() override;
  void DidLaunch() override;
  Status CanLoadImage() override;
  lldb::ThreadPlanSP GetStepThroughTrampolinePlan(Thread &thread,
                                                  bool stop) override;

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

protected:
  /// Returns the load address for the given executable module.
  ///
  /// The lookup proceeds in two stages:
  ///
  /// 1. **Cache lookup** – \c m_loaded_modules is scanned for an existing
  ///    entry whose \c ModuleSP matches \p executable. Because the same
  ///    \c ModuleSP can be inserted more than once under different base
  ///    addresses (e.g. a DLL loaded into several processes, or a module
  ///    that was unloaded and reloaded at a different address), the scan
  ///    returns the *first* valid (non-LLDB_INVALID_ADDRESS) entry it
  ///    finds.
  ///
  /// 2. **Process / platform query** – If no cached entry is found,
  ///    \c Process::GetFileLoadAddress is called. On a remote target the
  ///    remote platform is responsible for resolving the address. A
  ///    successful result is inserted into \c m_loaded_modules so that
  ///    subsequent calls hit the cache.
  ///
  /// \param executable  The module whose load address is requested.
  /// \return            The load address, or \c LLDB_INVALID_ADDRESS if it
  ///                    could not be determined.
  lldb::addr_t GetLoadAddress(lldb::ModuleSP executable);

  /// Maps load addresses to their corresponding modules.
  ///
  /// Weak pointers are used intentionally: on Windows, a Module holds a
  /// memory-mapped view of the DLL file, and an open memory mapping locks
  /// the file on disk. Holding a strong reference (ModuleSP) here would
  /// prevent the mapping from being released even after the Target drops
  /// its own reference, keeping the file locked and blocking recompilation
  /// during an active debug session.
  llvm::DenseMap<lldb::addr_t, lldb::ModuleWP> m_loaded_modules;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_DYNAMICLOADER_WINDOWS_DYLD_DYNAMICLOADERWINDOWSDYLD_H
