//===-- NativeProcessNetBSD.h --------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_NativeProcessNetBSD_H_
#define liblldb_NativeProcessNetBSD_H_

// C++ Includes

// Other libraries and framework includes

#include "lldb/Core/ArchSpec.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Utility/FileSpec.h"

#include "NativeThreadNetBSD.h"
#include "lldb/Host/common/NativeProcessProtocol.h"

namespace lldb_private {
namespace process_netbsd {
/// @class NativeProcessNetBSD
/// @brief Manages communication with the inferior (debugee) process.
///
/// Upon construction, this class prepares and launches an inferior process for
/// debugging.
///
/// Changes in the inferior process state are broadcasted.
class NativeProcessNetBSD : public NativeProcessProtocol {
  friend Error NativeProcessProtocol::Launch(
      ProcessLaunchInfo &launch_info, NativeDelegate &native_delegate,
      MainLoop &mainloop, NativeProcessProtocolSP &process_sp);

  friend Error NativeProcessProtocol::Attach(
      lldb::pid_t pid, NativeProcessProtocol::NativeDelegate &native_delegate,
      MainLoop &mainloop, NativeProcessProtocolSP &process_sp);

public:
  // ---------------------------------------------------------------------
  // NativeProcessProtocol Interface
  // ---------------------------------------------------------------------
  Error Resume(const ResumeActionList &resume_actions) override;

  Error Halt() override;

  Error Detach() override;

  Error Signal(int signo) override;

  Error Kill() override;

  Error GetMemoryRegionInfo(lldb::addr_t load_addr,
                            MemoryRegionInfo &range_info) override;

  Error ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                   size_t &bytes_read) override;

  Error ReadMemoryWithoutTrap(lldb::addr_t addr, void *buf, size_t size,
                              size_t &bytes_read) override;

  Error WriteMemory(lldb::addr_t addr, const void *buf, size_t size,
                    size_t &bytes_written) override;

  Error AllocateMemory(size_t size, uint32_t permissions,
                       lldb::addr_t &addr) override;

  Error DeallocateMemory(lldb::addr_t addr) override;

  lldb::addr_t GetSharedLibraryInfoAddress() override;

  size_t UpdateThreads() override;

  bool GetArchitecture(ArchSpec &arch) const override;

  Error SetBreakpoint(lldb::addr_t addr, uint32_t size, bool hardware) override;

  Error GetLoadedModuleFileSpec(const char *module_path,
                                FileSpec &file_spec) override;

  Error GetFileLoadAddress(const llvm::StringRef &file_name,
                           lldb::addr_t &load_addr) override;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  GetAuxvData() const override;

  // ---------------------------------------------------------------------
  // Interface used by NativeRegisterContext-derived classes.
  // ---------------------------------------------------------------------
  static Error PtraceWrapper(int req, lldb::pid_t pid, void *addr = nullptr,
                             int data = 0, int *result = nullptr);

protected:
  // ---------------------------------------------------------------------
  // NativeProcessProtocol protected interface
  // ---------------------------------------------------------------------

  Error
  GetSoftwareBreakpointTrapOpcode(size_t trap_opcode_size_hint,
                                  size_t &actual_opcode_size,
                                  const uint8_t *&trap_opcode_bytes) override;

private:
  MainLoop::SignalHandleUP m_sigchld_handle;
  ArchSpec m_arch;
  LazyBool m_supports_mem_region;
  std::vector<std::pair<MemoryRegionInfo, FileSpec>> m_mem_region_cache;

  // ---------------------------------------------------------------------
  // Private Instance Methods
  // ---------------------------------------------------------------------
  NativeProcessNetBSD();

  NativeThreadNetBSDSP AddThread(lldb::tid_t thread_id);

  Error LaunchInferior(MainLoop &mainloop, ProcessLaunchInfo &launch_info);
  void AttachToInferior(MainLoop &mainloop, lldb::pid_t pid, Error &error);

  void MonitorCallback(lldb::pid_t pid, int signal);
  void MonitorExited(lldb::pid_t pid, int signal, int status);
  void MonitorSIGSTOP(lldb::pid_t pid);
  void MonitorSIGTRAP(lldb::pid_t pid);
  void MonitorSignal(lldb::pid_t pid, int signal);

  Error GetSoftwareBreakpointPCOffset(uint32_t &actual_opcode_size);
  Error FixupBreakpointPCAsNeeded(NativeThreadNetBSD &thread);
  Error PopulateMemoryRegionCache();
  void SigchldHandler();

  ::pid_t Attach(lldb::pid_t pid, Error &error);

  Error ReinitializeThreads();
};

} // namespace process_netbsd
} // namespace lldb_private

#endif // #ifndef liblldb_NativeProcessNetBSD_H_
