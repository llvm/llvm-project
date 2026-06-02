//===-- ProcessEZH.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessEZH_h_
#define liblldb_ProcessEZH_h_

#include "../gdb-remote/ProcessGDBRemote.h"
#include "lldb/lldb-private.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

class ProcessEZH : public lldb_private::process_gdb_remote::ProcessGDBRemote {
public:
  ProcessEZH(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp);

  ~ProcessEZH() override;

  lldb_private::Status
  EnableBreakpointSite(lldb_private::BreakpointSite *bp_site) override;

  lldb_private::Status
  DisableBreakpointSite(lldb_private::BreakpointSite *bp_site) override;

  llvm::Error
  UpdateBreakpointSites(const lldb_private::Process::BreakpointSiteToActionMap &site_to_action) override;

  lldb::addr_t GetBaseAddress() const;

  lldb_private::Status
  DoConnectRemote(llvm::StringRef remote_url) override;

  void WillPublicStop() override;

  void DidAttach(lldb_private::ArchSpec &process_arch) override;

  void InvalidateMemoryCache() { m_memory_cache.Clear(); }
  size_t DoReadMemoryDirect(lldb::addr_t addr, void *buf, size_t size, lldb_private::Status &error) { return DoReadMemory(addr, buf, size, error); }

  lldb_private::ArchSpec GetSystemArchitecture() override;

  static void Initialize();

  static void DebuggerInitialize(lldb_private::Debugger &debugger);

  static void Terminate();

  static lldb::ProcessSP
  CreateInstance(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp,
                 const lldb_private::FileSpec *crash_file_path,
                 bool can_connect);

  std::shared_ptr<lldb_private::process_gdb_remote::ThreadGDBRemote>
  CreateThread(lldb::tid_t tid) override;

  bool DoUpdateThreadList(lldb_private::ThreadList &old_thread_list,
                          lldb_private::ThreadList &new_thread_list) override;

  lldb_private::Status
  DoResume(lldb::RunDirection direction) override;

  lldb_private::Status
  DoHalt(bool &caused_stop) override;

  lldb_private::Status
  DoDetach(bool keep_stopped) override;

  lldb_private::Status
  DoDestroy() override;

  void RefreshStateAfterStop() override;

  static llvm::StringRef GetPluginNameStatic() { return "ezh-remote"; }

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  void ModulesDidLoad(lldb_private::ModuleList &module_list) override;

  lldb_private::Status WriteEZHRegister(lldb::addr_t offset, uint32_t value);
  lldb_private::Status ReadEZHRegister(lldb::addr_t offset, uint32_t &value);

  lldb::addr_t GetDebugFrameAddr();
  lldb::addr_t GetDebugSoftwareBreakpointAddr(uint32_t slot);
  lldb::addr_t GetActiveSoftwareBreakpointAddr(uint32_t slot) const {
    return (slot < 16) ? m_active_sw_breakpoints[slot] : LLDB_INVALID_ADDRESS;
  }

private:
  void PollingThread();

  std::thread m_polling_thread;
  std::atomic<bool> m_destroy_polling_thread{false};
  std::atomic<bool> m_is_stepping{false};
  std::atomic<bool> m_halt_requested{false};
  std::condition_variable m_polling_cv;
  std::mutex m_polling_mutex;

  lldb::addr_t m_debug_frame_addr = LLDB_INVALID_ADDRESS;
  lldb::addr_t m_debug_sw_bp_addrs[16];
  lldb::addr_t m_active_sw_breakpoints[16];
  lldb::addr_t m_step_bp_addr = LLDB_INVALID_ADDRESS;
  uint32_t m_step_bp_original_op = 0;
  int m_step_bp_slot = -1;
};

#endif // liblldb_ProcessEZH_h_
