//===-- DynamicLoaderDumpWithModuleList.h --------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_DYNAMICLOADER_MODULELIST_DYLD_DYNAMICLOADERDUMPWITHMODULELIST_H
#define LLDB_SOURCE_PLUGINS_DYNAMICLOADER_MODULELIST_DYLD_DYNAMICLOADERDUMPWITHMODULELIST_H

#include "lldb/Core/ModuleList.h"
#include "lldb/Target/DynamicLoader.h"

/**
 * Dynamic loader for dump process with module list available.
 * For example, some coredump files have NT_FILE note section available
 * so can directly provide the module list without main executable's dynamic
 * section.
 */
class DynamicLoaderDumpWithModuleList : public lldb_private::DynamicLoader {
public:
  DynamicLoaderDumpWithModuleList(lldb_private::Process *process);

  ~DynamicLoaderDumpWithModuleList() override;

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() {
    return "dump-modulelist-dyld";
  }

  static llvm::StringRef GetPluginDescriptionStatic();

  static lldb_private::DynamicLoader *
  CreateInstance(lldb_private::Process *process, bool force);

  // DynamicLoader protocol

  void DidAttach() override;

  void DidLaunch() override {
    llvm_unreachable(
        "DynamicLoaderDumpWithModuleList::DidLaunch shouldn't be called");
  }

  lldb::ThreadPlanSP GetStepThroughTrampolinePlan(lldb_private::Thread &thread,
                                                  bool stop_others) override {
    llvm_unreachable("DynamicLoaderDumpWithModuleList::"
                     "GetStepThroughTrampolinePlan shouldn't be called");
  }

  lldb_private::Status CanLoadImage() override;

  lldb::addr_t GetThreadLocalData(const lldb::ModuleSP module,
                                  const lldb::ThreadSP thread,
                                  lldb::addr_t tls_file_addr) override {
    // TODO: how to implement this?
    return LLDB_INVALID_ADDRESS;
  }

  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

private:
  DynamicLoaderDumpWithModuleList(const DynamicLoaderDumpWithModuleList &) =
      delete;
  const DynamicLoaderDumpWithModuleList &
  operator=(const DynamicLoaderDumpWithModuleList &) = delete;
};

#endif // LLDB_SOURCE_PLUGINS_DYNAMICLOADER_MODULELIST_DYLD_DYNAMICLOADERDUMPWITHMODULELIST_H
