//===-- ProcessWindowsLive.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_Live_ProcessWindowsLive_H_
#define liblldb_Plugins_Process_Windows_Live_ProcessWindowsLive_H_

// C Includes

// C++ Includes
#include <memory>
#include <queue>

// Other libraries and framework includes
#include "ForwardDecl.h"
#include "IDebugDelegate.h"
#include "lldb/Core/Error.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Target/Process.h"
#include "lldb/lldb-forward.h"

#include "llvm/Support/Mutex.h"

#include "plugins/Process/Windows/Common/ProcessWindows.h"

class ProcessMonitor;

namespace lldb_private {
class HostProcess;
class ProcessWindowsData;

class ProcessWindowsLive : public lldb_private::ProcessWindows,
                           public lldb_private::IDebugDelegate {
public:
  //------------------------------------------------------------------
  // Static functions.
  //------------------------------------------------------------------
  static lldb::ProcessSP CreateInstance(lldb::TargetSP target_sp,
                                        lldb::ListenerSP listener_sp,
                                        const lldb_private::FileSpec *);

  static void Initialize();

  static void Terminate();

  static lldb_private::ConstString GetPluginNameStatic();

  static const char *GetPluginDescriptionStatic();

  //------------------------------------------------------------------
  // Constructors and destructors
  //------------------------------------------------------------------
  ProcessWindowsLive(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp);

  ~ProcessWindowsLive();

  // lldb_private::Process overrides
  lldb_private::ConstString GetPluginName() override;
  uint32_t GetPluginVersion() override;

  lldb_private::Error
  EnableBreakpointSite(lldb_private::BreakpointSite *bp_site) override;
  lldb_private::Error
  DisableBreakpointSite(lldb_private::BreakpointSite *bp_site) override;

  lldb_private::Error DoDetach(bool keep_stopped) override;
  lldb_private::Error
  DoLaunch(lldb_private::Module *exe_module,
           lldb_private::ProcessLaunchInfo &launch_info) override;
  lldb_private::Error DoAttachToProcessWithID(
      lldb::pid_t pid,
      const lldb_private::ProcessAttachInfo &attach_info) override;
  lldb_private::Error DoResume() override;
  lldb_private::Error DoDestroy() override;
  lldb_private::Error DoHalt(bool &caused_stop) override;

  void DidLaunch() override;
  void DidAttach(lldb_private::ArchSpec &arch_spec) override;

  void RefreshStateAfterStop() override;

  bool CanDebug(lldb::TargetSP target_sp,
                bool plugin_specified_by_name) override;
  bool DestroyRequiresHalt() override { return false; }
  bool UpdateThreadList(lldb_private::ThreadList &old_thread_list,
                        lldb_private::ThreadList &new_thread_list) override;
  bool IsAlive() override;

  size_t DoReadMemory(lldb::addr_t vm_addr, void *buf, size_t size,
                      lldb_private::Error &error) override;
  size_t DoWriteMemory(lldb::addr_t vm_addr, const void *buf, size_t size,
                       lldb_private::Error &error) override;
  lldb_private::Error
  GetMemoryRegionInfo(lldb::addr_t vm_addr,
                      lldb_private::MemoryRegionInfo &info) override;

  // IDebugDelegate overrides.
  void OnExitProcess(uint32_t exit_code) override;
  void OnDebuggerConnected(lldb::addr_t image_base) override;
  ExceptionResult
  OnDebugException(bool first_chance,
                   const lldb_private::ExceptionRecord &record) override;
  void OnCreateThread(const lldb_private::HostThread &thread) override;
  void OnExitThread(lldb::tid_t thread_id, uint32_t exit_code) override;
  void OnLoadDll(const lldb_private::ModuleSpec &module_spec,
                 lldb::addr_t module_addr) override;
  void OnUnloadDll(lldb::addr_t module_addr) override;
  void OnDebugString(const std::string &string) override;
  void OnDebuggerError(const lldb_private::Error &error,
                       uint32_t type) override;

private:
  lldb_private::Error
  WaitForDebuggerConnection(lldb_private::DebuggerThreadSP debugger,
                            lldb_private::HostProcess &process);

  llvm::sys::Mutex m_mutex;

  // Data for the active debugging session.
  std::unique_ptr<lldb_private::ProcessWindowsData> m_session_data;
};
}

#endif // liblldb_Plugins_Process_Windows_Live_ProcessWindowsLive_H_
