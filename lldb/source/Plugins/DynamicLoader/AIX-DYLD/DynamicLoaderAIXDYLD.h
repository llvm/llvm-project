//===-- DynamicLoaderAIXDYLD.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_DYNAMICLOADER_AIX_DYLD_DYNAMICLOADERAIXDYLD_H
#define LLDB_SOURCE_PLUGINS_DYNAMICLOADER_AIX_DYLD_DYNAMICLOADERAIXDYLD_H

#include "lldb/Target/DynamicLoader.h"
#include "lldb/lldb-forward.h"

#include <map>

namespace lldb_private {

class DynamicLoaderAIXDYLD : public DynamicLoader {
public:
  DynamicLoaderAIXDYLD(Process *process);

  ~DynamicLoaderAIXDYLD() override;

  static void Initialize();
  static void Terminate();
  static llvm::StringRef GetPluginNameStatic() { return "aix-dyld"; }
  static llvm::StringRef GetPluginDescriptionStatic();

  static DynamicLoader *CreateInstance(Process *process, bool force);

  void OnLoadModule(lldb::ModuleSP module_sp, const ModuleSpec module_spec,
                    lldb::addr_t module_addr);
  void OnUnloadModule(lldb::addr_t module_addr);

  void FillCoreLoaderData(lldb_private::DataExtractor &data,
          uint64_t loader_offset, uint64_t loader_size); 

  void DidAttach() override;
  void DidLaunch() override;
  Status CanLoadImage() override;
  lldb::ThreadPlanSP GetStepThroughTrampolinePlan(Thread &thread,
                                                  bool stop) override;

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  static bool NotifyBreakpointHit(void *baton, StoppointCallbackContext *context, lldb::user_id_t break_id, lldb::user_id_t break_loc_id);

protected:
  lldb::addr_t GetLoadAddress(lldb::ModuleSP executable);

  /// Loads Module from inferior process.
  void ResolveExecutableModule(lldb::ModuleSP &module_sp);

  /// Returns true if the process is for a core file.
  bool IsCoreFile() const;

private:
  std::map<lldb::ModuleSP, lldb::addr_t> m_loaded_modules;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_DYNAMICLOADER_AIX_DYLD_DYNAMICLOADERWAIXDYLD_H
