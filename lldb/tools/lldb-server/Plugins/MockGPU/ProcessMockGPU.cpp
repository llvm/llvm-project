//===-- ProcessMockGPU.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProcessMockGPU.h"
#include "ThreadMockGPU.h"

#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Utility/ProcessInfo.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/UnimplementedError.h"
#include "llvm/Support/Error.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"


using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;

ProcessMockGPU::ProcessMockGPU(lldb::pid_t pid, NativeDelegate &delegate)
    : NativeProcessProtocol(pid, -1, delegate) {
  m_state = eStateStopped;
  UpdateThreads();
}

Status ProcessMockGPU::Resume(const ResumeActionList &resume_actions) {
  SetState(StateType::eStateRunning, true);
  return Status();
}

Status ProcessMockGPU::Halt() {
  SetState(StateType::eStateStopped, true);
  return Status();
}

Status ProcessMockGPU::Detach() {
  SetState(StateType::eStateDetached, true);
  return Status();
}

/// Sends a process a UNIX signal \a signal.
///
/// \return
///     Returns an error object.
Status ProcessMockGPU::Signal(int signo) {
  return Status::FromErrorString("unimplemented");
}

/// Tells a process to interrupt all operations as if by a Ctrl-C.
///
/// The default implementation will send a local host's equivalent of
/// a SIGSTOP to the process via the NativeProcessProtocol::Signal()
/// operation.
///
/// \return
///     Returns an error object.
Status ProcessMockGPU::Interrupt() { return Status(); }

Status ProcessMockGPU::Kill() { return Status(); }

Status ProcessMockGPU::ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                                  size_t &bytes_read) {
  return Status::FromErrorString("unimplemented");
}

Status ProcessMockGPU::WriteMemory(lldb::addr_t addr, const void *buf,
                                   size_t size, size_t &bytes_written) {
  return Status::FromErrorString("unimplemented");
}

lldb::addr_t ProcessMockGPU::GetSharedLibraryInfoAddress() {
  return LLDB_INVALID_ADDRESS;
}

size_t ProcessMockGPU::UpdateThreads() {
  if (m_threads.empty()) {
    lldb::tid_t tid = 3456;
    m_threads.push_back(std::make_unique<ThreadMockGPU>(*this, 3456));
    // ThreadMockGPU &thread = static_cast<ThreadMockGPU &>(*m_threads.back());
    SetCurrentThreadID(tid);
  }
  return m_threads.size();
}

const ArchSpec &ProcessMockGPU::GetArchitecture() const {
  m_arch = ArchSpec("mockgpu");
  return m_arch;
}

// Breakpoint functions
Status ProcessMockGPU::SetBreakpoint(lldb::addr_t addr, uint32_t size,
                                     bool hardware) {
  return Status::FromErrorString("unimplemented");
}

llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
ProcessMockGPU::GetAuxvData() const {
  return nullptr; // TODO: try to return
                  // llvm::make_error<UnimplementedError>();
}

Status ProcessMockGPU::GetLoadedModuleFileSpec(const char *module_path,
                                               FileSpec &file_spec) {
  return Status::FromErrorString("unimplemented");
}

Status ProcessMockGPU::GetFileLoadAddress(const llvm::StringRef &file_name,
                                          lldb::addr_t &load_addr) {
  return Status::FromErrorString("unimplemented");
}

void ProcessMockGPU::SetLaunchInfo(ProcessLaunchInfo &launch_info) {
  static_cast<ProcessInfo &>(m_process_info) =
      static_cast<ProcessInfo &>(launch_info);
}

bool ProcessMockGPU::GetProcessInfo(ProcessInstanceInfo &proc_info) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOGF(log, "ProcessMockGPU::%s() entered", __FUNCTION__);
  m_process_info.SetProcessID(m_pid);
  m_process_info.SetArchitecture(GetArchitecture());
  proc_info = m_process_info;
  return true;
}

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
ProcessManagerMockGPU::Launch(
    ProcessLaunchInfo &launch_info,
    NativeProcessProtocol::NativeDelegate &native_delegate) {
  lldb::pid_t pid = 1234;
  auto proc_up = std::make_unique<ProcessMockGPU>(pid, native_delegate);
  proc_up->SetLaunchInfo(launch_info);
  return proc_up;
}

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
ProcessManagerMockGPU::Attach(
    lldb::pid_t pid, NativeProcessProtocol::NativeDelegate &native_delegate) {
  return llvm::createStringError("Unimplemented function");
}
