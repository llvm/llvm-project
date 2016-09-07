//===-- ProcessWinMiniDump.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessWinMiniDump_h_
#define liblldb_ProcessWinMiniDump_h_

#include <list>
#include <vector>

#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Target/Process.h"

#include "Plugins/Process/Windows/Common/ProcessWindows.h"

struct ThreadData;

class ProcessWinMiniDump : public lldb_private::ProcessWindows {
public:
  static lldb::ProcessSP
  CreateInstance(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp,
                 const lldb_private::FileSpec *crash_file_path);

  static void Initialize();

  static void Terminate();

  static lldb_private::ConstString GetPluginNameStatic();

  static const char *GetPluginDescriptionStatic();

  ProcessWinMiniDump(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp,
                     const lldb_private::FileSpec &core_file);

  virtual ~ProcessWinMiniDump();

  bool CanDebug(lldb::TargetSP target_sp,
                bool plugin_specified_by_name) override;

  lldb_private::Error DoLoadCore() override;

  lldb_private::DynamicLoader *GetDynamicLoader() override;

  lldb_private::ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

  lldb_private::Error DoDestroy() override;

  void RefreshStateAfterStop() override;

  bool IsAlive() override;

  bool WarnBeforeDetach() const override;

  size_t ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                    lldb_private::Error &error) override;

  size_t DoReadMemory(lldb::addr_t addr, void *buf, size_t size,
                      lldb_private::Error &error) override;

  lldb_private::ArchSpec GetArchitecture();

  lldb_private::Error
  GetMemoryRegionInfo(lldb::addr_t load_addr,
                      lldb_private::MemoryRegionInfo &range_info) override;

protected:
  void Clear();

  bool UpdateThreadList(lldb_private::ThreadList &old_thread_list,
                        lldb_private::ThreadList &new_thread_list) override;

private:
  // Keep Windows-specific types out of this header.
  class Impl;
  std::unique_ptr<Impl> m_impl_up;
};

#endif // liblldb_ProcessWinMiniDump_h_
