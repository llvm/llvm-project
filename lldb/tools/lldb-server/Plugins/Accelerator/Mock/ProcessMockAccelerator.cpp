//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProcessMockAccelerator.h"
#include "ThreadMockAccelerator.h"

#include "lldb/Host/HostInfo.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "llvm/Support/Error.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;

// A fixed, fake pid and tid for the single mock accelerator process/thread.
static constexpr lldb::pid_t kMockPid = 1234;
static constexpr lldb::tid_t kMockTid = 3456;

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
ProcessMockAccelerator::Manager::Launch(ProcessLaunchInfo &launch_info,
                                        NativeDelegate &native_delegate) {
  return std::make_unique<ProcessMockAccelerator>(kMockPid, native_delegate);
}

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
ProcessMockAccelerator::Manager::Attach(lldb::pid_t pid,
                                        NativeDelegate &native_delegate) {
  return llvm::createStringError("attach is not supported by the mock "
                                 "accelerator process");
}

ProcessMockAccelerator::ProcessMockAccelerator(lldb::pid_t pid,
                                               NativeDelegate &delegate)
    : NativeProcessProtocol(pid, /*terminal_fd=*/-1, delegate) {
  m_state = eStateStopped;
  UpdateThreads();
}

Status ProcessMockAccelerator::Resume(const ResumeActionList &resume_actions) {
  // Nothing actually runs; stay stopped.
  return Status();
}

Status ProcessMockAccelerator::Halt() { return Status(); }

Status ProcessMockAccelerator::Detach() {
  SetState(eStateDetached, true);
  return Status();
}

Status ProcessMockAccelerator::Signal(int signo) {
  return Status::FromErrorString("unimplemented");
}

Status ProcessMockAccelerator::Kill() { return Status(); }

Status ProcessMockAccelerator::ReadMemory(lldb::addr_t addr, void *buf,
                                          size_t size, size_t &bytes_read) {
  bytes_read = 0;
  return Status::FromErrorString("unimplemented");
}

Status ProcessMockAccelerator::WriteMemory(lldb::addr_t addr, const void *buf,
                                           size_t size, size_t &bytes_written) {
  bytes_written = 0;
  return Status::FromErrorString("unimplemented");
}

lldb::addr_t ProcessMockAccelerator::GetSharedLibraryInfoAddress() {
  return LLDB_INVALID_ADDRESS;
}

size_t ProcessMockAccelerator::UpdateThreads() {
  if (m_threads.empty()) {
    m_threads.push_back(
        std::make_unique<ThreadMockAccelerator>(*this, kMockTid));
    SetCurrentThreadID(kMockTid);
  }
  return m_threads.size();
}

const ArchSpec &ProcessMockAccelerator::GetArchitecture() const {
  if (!m_arch.IsValid())
    m_arch = HostInfo::GetArchitecture();
  return m_arch;
}

Status ProcessMockAccelerator::SetBreakpoint(lldb::addr_t addr, uint32_t size,
                                             bool hardware) {
  return Status::FromErrorString("unimplemented");
}

llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
ProcessMockAccelerator::GetAuxvData() const {
  return std::error_code(ENOENT, std::generic_category());
}

Status ProcessMockAccelerator::GetLoadedModuleFileSpec(const char *module_path,
                                                       FileSpec &file_spec) {
  return Status::FromErrorString("unimplemented");
}

Status
ProcessMockAccelerator::GetFileLoadAddress(const llvm::StringRef &file_name,
                                           lldb::addr_t &load_addr) {
  return Status::FromErrorString("unimplemented");
}
