//===-- ProcessGDBRemote.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessGDBRemote_h_
#define liblldb_ProcessGDBRemote_h_

// C Includes
// C++ Includes
#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Broadcaster.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/LoadedModuleInfoList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/StringList.h"
#include "lldb/Core/StructuredData.h"
#include "lldb/Core/ThreadSafeValue.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/StringExtractor.h"
#include "lldb/lldb-private-forward.h"

#include "GDBRemoteCommunicationClient.h"
#include "GDBRemoteRegisterContext.h"

#include "llvm/ADT/DenseMap.h"

namespace lldb_private {
namespace process_gdb_remote {

class ThreadGDBRemote;

class ProcessGDBRemote : public Process,
                         private GDBRemoteClientBase::ContinueDelegate {
public:
  ProcessGDBRemote(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp);

  ~ProcessGDBRemote() override;

  static lldb::ProcessSP CreateInstance(lldb::TargetSP target_sp,
                                        lldb::ListenerSP listener_sp,
                                        const FileSpec *crash_file_path);

  static void Initialize();

  static void DebuggerInitialize(Debugger &debugger);

  static void Terminate();

  static ConstString GetPluginNameStatic();

  static const char *GetPluginDescriptionStatic();

  //------------------------------------------------------------------
  // Check if a given Process
  //------------------------------------------------------------------
  bool CanDebug(lldb::TargetSP target_sp,
                bool plugin_specified_by_name) override;

  CommandObject *GetPluginCommandObject() override;

  //------------------------------------------------------------------
  // Creating a new process, or attaching to an existing one
  //------------------------------------------------------------------
  Error WillLaunch(Module *module) override;

  Error DoLaunch(Module *exe_module, ProcessLaunchInfo &launch_info) override;

  void DidLaunch() override;

  Error WillAttachToProcessWithID(lldb::pid_t pid) override;

  Error WillAttachToProcessWithName(const char *process_name,
                                    bool wait_for_launch) override;

  Error DoConnectRemote(Stream *strm, const char *remote_url) override;

  Error WillLaunchOrAttach();

  Error DoAttachToProcessWithID(lldb::pid_t pid,
                                const ProcessAttachInfo &attach_info) override;

  Error
  DoAttachToProcessWithName(const char *process_name,
                            const ProcessAttachInfo &attach_info) override;

  void DidAttach(ArchSpec &process_arch) override;

  //------------------------------------------------------------------
  // PluginInterface protocol
  //------------------------------------------------------------------
  ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

  //------------------------------------------------------------------
  // Process Control
  //------------------------------------------------------------------
  Error WillResume() override;

  Error DoResume() override;

  Error DoHalt(bool &caused_stop) override;

  Error DoDetach(bool keep_stopped) override;

  bool DetachRequiresHalt() override { return true; }

  Error DoSignal(int signal) override;

  Error DoDestroy() override;

  void RefreshStateAfterStop() override;

  void SetUnixSignals(const lldb::UnixSignalsSP &signals_sp);

  //------------------------------------------------------------------
  // Process Queries
  //------------------------------------------------------------------
  bool IsAlive() override;

  lldb::addr_t GetImageInfoAddress() override;

  void WillPublicStop() override;

  //------------------------------------------------------------------
  // Process Memory
  //------------------------------------------------------------------
  size_t DoReadMemory(lldb::addr_t addr, void *buf, size_t size,
                      Error &error) override;

  size_t DoWriteMemory(lldb::addr_t addr, const void *buf, size_t size,
                       Error &error) override;

  lldb::addr_t DoAllocateMemory(size_t size, uint32_t permissions,
                                Error &error) override;

  Error GetMemoryRegionInfo(lldb::addr_t load_addr,
                            MemoryRegionInfo &region_info) override;

  Error DoDeallocateMemory(lldb::addr_t ptr) override;

  //------------------------------------------------------------------
  // Process STDIO
  //------------------------------------------------------------------
  size_t PutSTDIN(const char *buf, size_t buf_size, Error &error) override;

  //----------------------------------------------------------------------
  // Process Breakpoints
  //----------------------------------------------------------------------
  Error EnableBreakpointSite(BreakpointSite *bp_site) override;

  Error DisableBreakpointSite(BreakpointSite *bp_site) override;

  //----------------------------------------------------------------------
  // Process Watchpoints
  //----------------------------------------------------------------------
  Error EnableWatchpoint(Watchpoint *wp, bool notify = true) override;

  Error DisableWatchpoint(Watchpoint *wp, bool notify = true) override;

  Error GetWatchpointSupportInfo(uint32_t &num) override;

  Error GetWatchpointSupportInfo(uint32_t &num, bool &after) override;

  bool StartNoticingNewThreads() override;

  bool StopNoticingNewThreads() override;

  GDBRemoteCommunicationClient &GetGDBRemote() { return m_gdb_comm; }

  Error SendEventData(const char *data) override;

  //----------------------------------------------------------------------
  // Override DidExit so we can disconnect from the remote GDB server
  //----------------------------------------------------------------------
  void DidExit() override;

  void SetUserSpecifiedMaxMemoryTransferSize(uint64_t user_specified_max);

  bool GetModuleSpec(const FileSpec &module_file_spec, const ArchSpec &arch,
                     ModuleSpec &module_spec) override;

  void PrefetchModuleSpecs(llvm::ArrayRef<FileSpec> module_file_specs,
                           const llvm::Triple &triple) override;

  bool GetHostOSVersion(uint32_t &major, uint32_t &minor,
                        uint32_t &update) override;

  size_t LoadModules(LoadedModuleInfoList &module_list) override;

  size_t LoadModules() override;

  Error GetFileLoadAddress(const FileSpec &file, bool &is_loaded,
                           lldb::addr_t &load_addr) override;

  void ModulesDidLoad(ModuleList &module_list) override;

  StructuredData::ObjectSP
  GetLoadedDynamicLibrariesInfos(lldb::addr_t image_list_address,
                                 lldb::addr_t image_count) override;

  Error
  ConfigureStructuredData(const ConstString &type_name,
                          const StructuredData::ObjectSP &config_sp) override;

  StructuredData::ObjectSP GetLoadedDynamicLibrariesInfos() override;

  StructuredData::ObjectSP GetLoadedDynamicLibrariesInfos(
      const std::vector<lldb::addr_t> &load_addresses) override;

  StructuredData::ObjectSP
  GetLoadedDynamicLibrariesInfos_sender(StructuredData::ObjectSP args);

  StructuredData::ObjectSP GetSharedCacheInfo() override;

  std::string HarmonizeThreadIdsForProfileData(
      StringExtractorGDBRemote &inputStringExtractor);

protected:
  friend class ThreadGDBRemote;
  friend class GDBRemoteCommunicationClient;
  friend class GDBRemoteRegisterContext;

  //------------------------------------------------------------------
  /// Broadcaster event bits definitions.
  //------------------------------------------------------------------
  enum {
    eBroadcastBitAsyncContinue = (1 << 0),
    eBroadcastBitAsyncThreadShouldExit = (1 << 1),
    eBroadcastBitAsyncThreadDidExit = (1 << 2)
  };

  Flags m_flags; // Process specific flags (see eFlags enums)
  GDBRemoteCommunicationClient m_gdb_comm;
  std::atomic<lldb::pid_t> m_debugserver_pid;
  std::vector<StringExtractorGDBRemote> m_stop_packet_stack; // The stop packet
                                                             // stack replaces
                                                             // the last stop
                                                             // packet variable
  std::recursive_mutex m_last_stop_packet_mutex;
  GDBRemoteDynamicRegisterInfo m_register_info;
  Broadcaster m_async_broadcaster;
  lldb::ListenerSP m_async_listener_sp;
  HostThread m_async_thread;
  std::recursive_mutex m_async_thread_state_mutex;
  typedef std::vector<lldb::tid_t> tid_collection;
  typedef std::vector<std::pair<lldb::tid_t, int>> tid_sig_collection;
  typedef std::map<lldb::addr_t, lldb::addr_t> MMapMap;
  typedef std::map<uint32_t, std::string> ExpeditedRegisterMap;
  tid_collection m_thread_ids; // Thread IDs for all threads. This list gets
                               // updated after stopping
  std::vector<lldb::addr_t> m_thread_pcs;     // PC values for all the threads.
  StructuredData::ObjectSP m_jstopinfo_sp;    // Stop info only for any threads
                                              // that have valid stop infos
  StructuredData::ObjectSP m_jthreadsinfo_sp; // Full stop info, expedited
                                              // registers and memory for all
                                              // threads if "jThreadsInfo"
                                              // packet is supported
  tid_collection m_continue_c_tids;           // 'c' for continue
  tid_sig_collection m_continue_C_tids;       // 'C' for continue with signal
  tid_collection m_continue_s_tids;           // 's' for step
  tid_sig_collection m_continue_S_tids;       // 'S' for step with signal
  uint64_t m_max_memory_size; // The maximum number of bytes to read/write when
                              // reading and writing memory
  uint64_t m_remote_stub_max_memory_size; // The maximum memory size the remote
                                          // gdb stub can handle
  MMapMap m_addr_to_mmap_size;
  lldb::BreakpointSP m_thread_create_bp_sp;
  bool m_waiting_for_attach;
  bool m_destroy_tried_resuming;
  lldb::CommandObjectSP m_command_sp;
  int64_t m_breakpoint_pc_offset;
  lldb::tid_t m_initial_tid; // The initial thread ID, given by stub on attach

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

  bool ProcessIDIsValid() const;

  void Clear();

  Flags &GetFlags() { return m_flags; }

  const Flags &GetFlags() const { return m_flags; }

  bool UpdateThreadList(ThreadList &old_thread_list,
                        ThreadList &new_thread_list) override;

  Error EstablishConnectionIfNeeded(const ProcessInfo &process_info);

  Error LaunchAndConnectToDebugserver(const ProcessInfo &process_info);

  void KillDebugserverProcess();

  void BuildDynamicRegisterInfo(bool force);

  void SetLastStopPacket(const StringExtractorGDBRemote &response);

  bool ParsePythonTargetDefinition(const FileSpec &target_definition_fspec);

  const lldb::DataBufferSP GetAuxvData() override;

  StructuredData::ObjectSP GetExtendedInfoForThread(lldb::tid_t tid);

  void GetMaxMemorySize();

  bool CalculateThreadStopInfo(ThreadGDBRemote *thread);

  size_t UpdateThreadPCsFromStopReplyThreadsValue(std::string &value);

  size_t UpdateThreadIDsFromStopReplyThreadsValue(std::string &value);

  bool HandleNotifyPacket(StringExtractorGDBRemote &packet);

  bool StartAsyncThread();

  void StopAsyncThread();

  static lldb::thread_result_t AsyncThread(void *arg);

  static bool
  MonitorDebugserverProcess(std::weak_ptr<ProcessGDBRemote> process_wp,
                            lldb::pid_t pid, bool exited, int signo,
                            int exit_status);

  lldb::StateType SetThreadStopInfo(StringExtractor &stop_packet);

  bool
  GetThreadStopInfoFromJSON(ThreadGDBRemote *thread,
                            const StructuredData::ObjectSP &thread_infos_sp);

  lldb::ThreadSP SetThreadStopInfo(StructuredData::Dictionary *thread_dict);

  lldb::ThreadSP
  SetThreadStopInfo(lldb::tid_t tid,
                    ExpeditedRegisterMap &expedited_register_map, uint8_t signo,
                    const std::string &thread_name, const std::string &reason,
                    const std::string &description, uint32_t exc_type,
                    const std::vector<lldb::addr_t> &exc_data,
                    lldb::addr_t thread_dispatch_qaddr, bool queue_vars_valid,
                    lldb_private::LazyBool associated_with_libdispatch_queue,
                    lldb::addr_t dispatch_queue_t, std::string &queue_name,
                    lldb::QueueKind queue_kind, uint64_t queue_serial);

  void HandleStopReplySequence();

  void ClearThreadIDList();

  bool UpdateThreadIDList();

  void DidLaunchOrAttach(ArchSpec &process_arch);

  Error ConnectToDebugserver(const char *host_port);

  const char *GetDispatchQueueNameForThread(lldb::addr_t thread_dispatch_qaddr,
                                            std::string &dispatch_queue_name);

  DynamicLoader *GetDynamicLoader() override;

  // Query remote GDBServer for register information
  bool GetGDBServerRegisterInfo(ArchSpec &arch);

  // Query remote GDBServer for a detailed loaded library list
  Error GetLoadedModuleList(LoadedModuleInfoList &);

  lldb::ModuleSP LoadModuleAtAddress(const FileSpec &file,
                                     lldb::addr_t link_map,
                                     lldb::addr_t base_addr,
                                     bool value_is_offset);

private:
  //------------------------------------------------------------------
  // For ProcessGDBRemote only
  //------------------------------------------------------------------
  std::string m_partial_profile_data;
  std::map<uint64_t, uint32_t> m_thread_id_to_used_usec_map;

  static bool NewThreadNotifyBreakpointHit(void *baton,
                                           StoppointCallbackContext *context,
                                           lldb::user_id_t break_id,
                                           lldb::user_id_t break_loc_id);

  //------------------------------------------------------------------
  // ContinueDelegate interface
  //------------------------------------------------------------------
  void HandleAsyncStdout(llvm::StringRef out) override;
  void HandleAsyncMisc(llvm::StringRef data) override;
  void HandleStopReply() override;
  void HandleAsyncStructuredDataPacket(llvm::StringRef data) override;

  using ModuleCacheKey = std::pair<std::string, std::string>;
  // KeyInfo for the cached module spec DenseMap.
  // The invariant is that all real keys will have the file and architecture
  // set.
  // The empty key has an empty file and an empty arch.
  // The tombstone key has an invalid arch and an empty file.
  // The comparison and hash functions take the file name and architecture
  // triple into account.
  struct ModuleCacheInfo {
    static ModuleCacheKey getEmptyKey() { return ModuleCacheKey(); }

    static ModuleCacheKey getTombstoneKey() { return ModuleCacheKey("", "T"); }

    static unsigned getHashValue(const ModuleCacheKey &key) {
      return llvm::hash_combine(key.first, key.second);
    }

    static bool isEqual(const ModuleCacheKey &LHS, const ModuleCacheKey &RHS) {
      return LHS == RHS;
    }
  };

  llvm::DenseMap<ModuleCacheKey, ModuleSpec, ModuleCacheInfo>
      m_cached_module_specs;

  DISALLOW_COPY_AND_ASSIGN(ProcessGDBRemote);
};

} // namespace process_gdb_remote
} // namespace lldb_private

#endif // liblldb_ProcessGDBRemote_h_
