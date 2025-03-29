//===-- ProcessAIXCore.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Notes about AIX Process core dumps:
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_AIX_CORE_PROCESSAIXCORE_H
#define LLDB_SOURCE_PLUGINS_PROCESS_AIX_CORE_PROCESSAIXCORE_H

#include <list>
#include <vector>

#include "lldb/Target/PostMortemProcess.h"
#include "lldb/Utility/Status.h"
#include "lldb/Target/Process.h"
#include "AIXCore.h"
#include "ThreadAIXCore.h"

struct ThreadData;

class ProcessAIXCore : public lldb_private::PostMortemProcess {
public:
  // Constructors and Destructors
  static lldb::ProcessSP
  CreateInstance(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp,
                 const lldb_private::FileSpec *crash_file_path,
                 bool can_connect);

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "aix-core"; }

  static llvm::StringRef GetPluginDescriptionStatic();

  // Constructors and Destructors
  ProcessAIXCore(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp,
                 const lldb_private::FileSpec &core_file);

  ~ProcessAIXCore() override;

  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  // Process Control
  lldb_private::Status DoDestroy() override;

  lldb_private::Status WillResume() override {
    return lldb_private::Status::FromErrorStringWithFormatv(
        "error: {0} does not support resuming processes", GetPluginName());
  }

  bool WarnBeforeDetach() const override { return false; }

  lldb_private::ArchSpec GetArchitecture();
  
  bool CanDebug(lldb::TargetSP target_sp,
          bool plugin_specified_by_name) override;
  
  // Creating a new process, or attaching to an existing one
  lldb_private::Status DoLoadCore() override;
 
  bool DoUpdateThreadList(lldb_private::ThreadList &old_thread_list,
          lldb_private::ThreadList &new_thread_list) override;

  lldb_private::Status
  DoGetMemoryRegionInfo(lldb::addr_t load_addr,
              lldb_private::MemoryRegionInfo &region_info) override;

  void RefreshStateAfterStop() override;

  lldb_private::DynamicLoader *GetDynamicLoader() override;

  // Process Memory
  size_t ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                    lldb_private::Status &error) override;

  size_t DoReadMemory(lldb::addr_t addr, void *buf, size_t size,
          lldb_private::Status &error) override; 

  void ParseAIXCoreFile();


private:
  lldb::ModuleSP m_core_module_sp;
  std::string m_dyld_plugin_name;

  // True if m_thread_contexts contains valid entries
  bool m_thread_data_valid = false;
  AIXCORE::AIXCore64Header m_aixcore_header;

  std::vector<ThreadData> m_thread_data;
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_AIX_CORE_PROCESSAIXCORE_H
