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

#include <cstdlib>

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

std::optional<LLDBSettings> ProcessMockAccelerator::GetLLDBSettings() {
  // Tell the client to use the accelerator dynamic loader for this target, so
  // libraries are provided by this plugin rather than by a rendezvous
  // breakpoint.
  LLDBSettings settings;
  settings.dyld_plugin_name = "accelerator-gdb-remote";
  return settings;
}

std::optional<AcceleratorDynamicLoaderResponse>
ProcessMockAccelerator::GetAcceleratorDynamicLoaderLibraryInfos(
    const AcceleratorDynamicLoaderArgs &args) {
  // The test drives which libraries to report (and where they live) via the
  // environment, so it can build real object files and point us at them.
  AcceleratorDynamicLoaderResponse response;

  // Scenario 1: a shared library provided as a whole file on disk, loaded at a
  // fixed base address.
  if (const char *path = ::getenv("LLDB_MOCK_ACCELERATOR_LIB_ONDISK")) {
    AcceleratorDynamicLoaderLibraryInfo info;
    info.pathname = path;
    info.load = true;
    info.load_address = 0x10000000;
    response.library_infos.push_back(std::move(info));
  }

  // Scenario 3: a shared library embedded in a container file (e.g. added to
  // the executable with llvm-objcopy), located by file offset and size.
  if (const char *path = ::getenv("LLDB_MOCK_ACCELERATOR_LIB_CONTAINER")) {
    AcceleratorDynamicLoaderLibraryInfo info;
    info.pathname = path;
    info.load = true;
    info.load_address = 0x20000000;
    if (const char *offset = ::getenv("LLDB_MOCK_ACCELERATOR_LIB_OFFSET"))
      info.file_offset = std::strtoull(offset, nullptr, 0);
    if (const char *size = ::getenv("LLDB_MOCK_ACCELERATOR_LIB_SIZE"))
      info.file_size = std::strtoull(size, nullptr, 0);
    response.library_infos.push_back(std::move(info));
  }

  return response;
}
