//===-- ProcessKDP.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessKDP_h_
#define liblldb_ProcessKDP_h_

// C Includes

// C++ Includes
#include <list>
#include <vector>

// Other libraries and framework includes
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Broadcaster.h"
#include "lldb/Core/StringList.h"
#include "lldb/Core/ThreadSafeValue.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Error.h"
#include "lldb/Utility/StreamString.h"

#include "CommunicationKDP.h"

class ThreadKDP;

class ProcessKDP : public lldb_private::Process {
public:
  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  static lldb::ProcessSP
  CreateInstance(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp,
                 const lldb_private::FileSpec *crash_file_path);

  static void Initialize();

  static void DebuggerInitialize(lldb_private::Debugger &debugger);

  static void Terminate();

  static lldb_private::ConstString GetPluginNameStatic();

  static const char *GetPluginDescriptionStatic();

  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  ProcessKDP(lldb::TargetSP target_sp, lldb::ListenerSP listener);

  ~ProcessKDP() override;

  //------------------------------------------------------------------
  // Check if a given Process
  //------------------------------------------------------------------
  bool CanDebug(lldb::TargetSP target_sp,
                bool plugin_specified_by_name) override;
  lldb_private::CommandObject *GetPluginCommandObject() override;

  //------------------------------------------------------------------
  // Creating a new process, or attaching to an existing one
  //------------------------------------------------------------------
  lldb_private::Error WillLaunch(lldb_private::Module *module) override;

  lldb_private::Error
  DoLaunch(lldb_private::Module *exe_module,
           lldb_private::ProcessLaunchInfo &launch_info) override;

  lldb_private::Error WillAttachToProcessWithID(lldb::pid_t pid) override;

  lldb_private::Error
  WillAttachToProcessWithName(const char *process_name,
                              bool wait_for_launch) override;

  lldb_private::Error DoConnectRemote(lldb_private::Stream *strm,
                                      llvm::StringRef remote_url) override;

  lldb_private::Error DoAttachToProcessWithID(
      lldb::pid_t pid,
      const lldb_private::ProcessAttachInfo &attach_info) override;

  lldb_private::Error DoAttachToProcessWithName(
      const char *process_name,
      const lldb_private::ProcessAttachInfo &attach_info) override;

  void DidAttach(lldb_private::ArchSpec &process_arch) override;

  lldb::addr_t GetImageInfoAddress() override;

  lldb_private::DynamicLoader *GetDynamicLoader() override;

  //------------------------------------------------------------------
  // PluginInterface protocol
  //------------------------------------------------------------------
  lldb_private::ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

  //------------------------------------------------------------------
  // Process Control
  //------------------------------------------------------------------
  lldb_private::Error WillResume() override;

  lldb_private::Error DoResume() override;

  lldb_private::Error DoHalt(bool &caused_stop) override;

  lldb_private::Error DoDetach(bool keep_stopped) override;

  lldb_private::Error DoSignal(int signal) override;

  lldb_private::Error DoDestroy() override;

  void RefreshStateAfterStop() override;

  //------------------------------------------------------------------
  // Process Queries
  //------------------------------------------------------------------
  bool IsAlive() override;

  //------------------------------------------------------------------
  // Process Memory
  //------------------------------------------------------------------
  size_t DoReadMemory(lldb::addr_t addr, void *buf, size_t size,
                      lldb_private::Error &error) override;

  size_t DoWriteMemory(lldb::addr_t addr, const void *buf, size_t size,
                       lldb_private::Error &error) override;

  lldb::addr_t DoAllocateMemory(size_t size, uint32_t permissions,
                                lldb_private::Error &error) override;

  lldb_private::Error DoDeallocateMemory(lldb::addr_t ptr) override;

  //----------------------------------------------------------------------
  // Process Breakpoints
  //----------------------------------------------------------------------
  lldb_private::Error
  EnableBreakpointSite(lldb_private::BreakpointSite *bp_site) override;

  lldb_private::Error
  DisableBreakpointSite(lldb_private::BreakpointSite *bp_site) override;

  //----------------------------------------------------------------------
  // Process Watchpoints
  //----------------------------------------------------------------------
  lldb_private::Error EnableWatchpoint(lldb_private::Watchpoint *wp,
                                       bool notify = true) override;

  lldb_private::Error DisableWatchpoint(lldb_private::Watchpoint *wp,
                                        bool notify = true) override;

  CommunicationKDP &GetCommunication() { return m_comm; }

protected:
  friend class ThreadKDP;
  friend class CommunicationKDP;

  //----------------------------------------------------------------------
  // Accessors
  //----------------------------------------------------------------------
  bool IsRunning(lldb::StateType state) {
    return state == lldb::eStateRunning || IsStepping(state);
  }

  bool IsStepping(lldb::StateType state) {
    return state == lldb::eStateStepping;
  }

  bool CanResume(lldb::StateType state) { return state == lldb::eStateStopped; }

  bool HasExited(lldb::StateType state) { return state == lldb::eStateExited; }

  bool GetHostArchitecture(lldb_private::ArchSpec &arch);

  bool ProcessIDIsValid() const;

  void Clear();

  bool UpdateThreadList(lldb_private::ThreadList &old_thread_list,
                        lldb_private::ThreadList &new_thread_list) override;

  enum {
    eBroadcastBitAsyncContinue = (1 << 0),
    eBroadcastBitAsyncThreadShouldExit = (1 << 1)
  };

  lldb::ThreadSP GetKernelThread();

  //------------------------------------------------------------------
  /// Broadcaster event bits definitions.
  //------------------------------------------------------------------
  CommunicationKDP m_comm;
  lldb_private::Broadcaster m_async_broadcaster;
  lldb_private::HostThread m_async_thread;
  lldb_private::ConstString m_dyld_plugin_name;
  lldb::addr_t m_kernel_load_addr;
  lldb::CommandObjectSP m_command_sp;
  lldb::ThreadWP m_kernel_thread_wp;

  bool StartAsyncThread();

  void StopAsyncThread();

  static void *AsyncThread(void *arg);

private:
  //------------------------------------------------------------------
  // For ProcessKDP only
  //------------------------------------------------------------------

  DISALLOW_COPY_AND_ASSIGN(ProcessKDP);
};

#endif // liblldb_ProcessKDP_h_
