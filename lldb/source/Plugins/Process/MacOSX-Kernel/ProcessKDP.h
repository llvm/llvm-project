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
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/StringList.h"
#include "lldb/Core/ThreadSafeValue.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"

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

  virtual ~ProcessKDP();

  //------------------------------------------------------------------
  // Check if a given Process
  //------------------------------------------------------------------
  virtual bool CanDebug(lldb::TargetSP target_sp,
                        bool plugin_specified_by_name);

  virtual lldb_private::CommandObject *GetPluginCommandObject();

  //------------------------------------------------------------------
  // Creating a new process, or attaching to an existing one
  //------------------------------------------------------------------
  virtual lldb_private::Error WillLaunch(lldb_private::Module *module);

  virtual lldb_private::Error
  DoLaunch(lldb_private::Module *exe_module,
           lldb_private::ProcessLaunchInfo &launch_info);

  virtual lldb_private::Error WillAttachToProcessWithID(lldb::pid_t pid);

  virtual lldb_private::Error
  WillAttachToProcessWithName(const char *process_name, bool wait_for_launch);

  virtual lldb_private::Error DoConnectRemote(lldb_private::Stream *strm,
                                              const char *remote_url);

  virtual lldb_private::Error
  DoAttachToProcessWithID(lldb::pid_t pid,
                          const lldb_private::ProcessAttachInfo &attach_info);

  virtual lldb_private::Error
  DoAttachToProcessWithName(const char *process_name,
                            const lldb_private::ProcessAttachInfo &attach_info);

  virtual void DidAttach(lldb_private::ArchSpec &process_arch);

  lldb::addr_t GetImageInfoAddress();

  lldb_private::DynamicLoader *GetDynamicLoader();

  //------------------------------------------------------------------
  // PluginInterface protocol
  //------------------------------------------------------------------
  virtual lldb_private::ConstString GetPluginName();

  virtual uint32_t GetPluginVersion();

  //------------------------------------------------------------------
  // Process Control
  //------------------------------------------------------------------
  virtual lldb_private::Error WillResume();

  virtual lldb_private::Error DoResume();

  virtual lldb_private::Error DoHalt(bool &caused_stop);

  virtual lldb_private::Error DoDetach(bool keep_stopped);

  virtual lldb_private::Error DoSignal(int signal);

  virtual lldb_private::Error DoDestroy();

  virtual void RefreshStateAfterStop();

  //------------------------------------------------------------------
  // Process Queries
  //------------------------------------------------------------------
  virtual bool IsAlive();

  //------------------------------------------------------------------
  // Process Memory
  //------------------------------------------------------------------
  virtual size_t DoReadMemory(lldb::addr_t addr, void *buf, size_t size,
                              lldb_private::Error &error);

  virtual size_t DoWriteMemory(lldb::addr_t addr, const void *buf, size_t size,
                               lldb_private::Error &error);

  virtual lldb::addr_t DoAllocateMemory(size_t size, uint32_t permissions,
                                        lldb_private::Error &error);

  virtual lldb_private::Error DoDeallocateMemory(lldb::addr_t ptr);

  //----------------------------------------------------------------------
  // Process Breakpoints
  //----------------------------------------------------------------------
  virtual lldb_private::Error
  EnableBreakpointSite(lldb_private::BreakpointSite *bp_site);

  virtual lldb_private::Error
  DisableBreakpointSite(lldb_private::BreakpointSite *bp_site);

  //----------------------------------------------------------------------
  // Process Watchpoints
  //----------------------------------------------------------------------
  virtual lldb_private::Error EnableWatchpoint(lldb_private::Watchpoint *wp,
                                               bool notify = true);

  virtual lldb_private::Error DisableWatchpoint(lldb_private::Watchpoint *wp,
                                                bool notify = true);

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

  virtual bool UpdateThreadList(lldb_private::ThreadList &old_thread_list,
                                lldb_private::ThreadList &new_thread_list);

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
