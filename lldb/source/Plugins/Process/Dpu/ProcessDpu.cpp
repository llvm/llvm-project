//===-- ProcessDpu.cpp -------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ProcessDpu.h"

// C Includes
#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

// C++ Includes
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>

// Other libraries and framework includes
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/State.h"
#include "lldb/Host/File.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Host/PseudoTerminal.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Host/common/NativeBreakpoint.h"
#include "lldb/Host/common/NativeRegisterContext.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/ProcessLaunchInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StringExtractor.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Threading.h"

#include "DpuRank.h"
#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "ThreadDpu.h"

#include <linux/unistd.h>
#include <sys/timerfd.h> /* TODO only exists on Linux */
#include <sys/types.h>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::dpu;
using namespace lldb_private::process_dpu;
using namespace llvm;

namespace {

const ArchSpec k_dpu_arch("dpu-upmem-dpurte");

// Control interface polling period
const long k_ci_polling_interval_ns = 10000; /* ns */

constexpr lldb::addr_t k_dpu_iram_base = 0x80000000;
constexpr lldb::addr_t k_dpu_mram_base = 0x08000000;
} // end of anonymous namespace

// -----------------------------------------------------------------------------
// Public Static Methods
// -----------------------------------------------------------------------------

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
ProcessDpu::Factory::Launch(ProcessLaunchInfo &launch_info,
                            NativeDelegate &native_delegate,
                            MainLoop &mainloop) const {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));

  // TODO set rankPath and profile from args
  DpuRank *rank = new DpuRank();
  // std::unique_ptr<DpuRank> rank = llvm::make_unique<DpuRank>();
  bool success = rank->Open();
  if (!success)
    return Status("Cannot get a DPU rank").ToError();
  rank->Reset();

  // assert(launch_info.GetArchitecture() == k_dpu_arch);
  Dpu *dpu = rank->GetDpu(0);

  dpu->LoadElf(launch_info.GetExecutableFile());
  dpu->Boot();

  ::pid_t pid = 666 << 5; // TODO unique Rank ID
  LLDB_LOG(log, "Attaching Dpu Rank {0}", pid);

  return std::unique_ptr<ProcessDpu>(
      new ProcessDpu(pid, launch_info.GetPTY().ReleaseMasterFileDescriptor(),
                     native_delegate, k_dpu_arch, mainloop, dpu));
}

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
ProcessDpu::Factory::Attach(
    lldb::pid_t pid, NativeProcessProtocol::NativeDelegate &native_delegate,
    MainLoop &mainloop) const {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
  LLDB_LOG(log, "pid = {0:x}", pid);

  //  return llvm::make_error<StringError>("Cannot find DPU rank",
  //                                       llvm::inconvertibleErrorCode());

  Dpu *dpu = nullptr; // rank->GetDpu(0);

  return std::unique_ptr<ProcessDpu>(
      new ProcessDpu(pid, -1, native_delegate, k_dpu_arch, mainloop, dpu));
}

// -----------------------------------------------------------------------------
// Public Instance Methods
// -----------------------------------------------------------------------------

ProcessDpu::ProcessDpu(::pid_t pid, int terminal_fd, NativeDelegate &delegate,
                       const ArchSpec &arch, MainLoop &mainloop, Dpu *dpu)
    : NativeProcessProtocol(pid, terminal_fd, delegate), m_arch(arch),
      m_dpu(dpu) {

  // Set the timer for polling the CI
  struct itimerspec polling_spec;
  polling_spec.it_value.tv_sec = 0;
  polling_spec.it_value.tv_nsec = k_ci_polling_interval_ns;
  polling_spec.it_interval.tv_sec = 0;
  polling_spec.it_interval.tv_nsec = k_ci_polling_interval_ns;
  int tfd = timerfd_create(CLOCK_MONOTONIC, 0);
  assert(tfd != -1);
  m_timer_fd.reset(new File(tfd, true));

  Status status;
  m_timer_handle = mainloop.RegisterReadObject(
      m_timer_fd, [this](MainLoopBase &) { InterfaceTimerCallback(); }, status);
  timerfd_settime(tfd, 0, &polling_spec, nullptr);
  assert(m_timer_handle && status.Success());

  for (int thread_id = 0; thread_id < m_dpu->GetNrThreads(); thread_id++) {
    m_threads.push_back(
        llvm::make_unique<ThreadDpu>(*this, pid | thread_id, thread_id));
  }
  SetCurrentThreadID(pid);

  m_dpu->StopThreads();

  // Let our process instance know the thread has stopped.
  SetState(StateType::eStateStopped, false);
}

void ProcessDpu::InterfaceTimerCallback() {
  unsigned int exit_status;
  StateType current_state = m_dpu->PollStatus(&exit_status);
  if (current_state != StateType::eStateInvalid) {
    if (current_state == StateType::eStateExited)
      SetExitStatus(WaitStatus(WaitStatus::Exit, (uint8_t)exit_status), true);
    SetState(current_state, true);
  }
}

bool ProcessDpu::SupportHardwareSingleStepping() const { return true; }

Status ProcessDpu::Resume(const ResumeActionList &resume_actions) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
  lldb::tid_t thread_id = GetID();
  LLDB_LOG(log, "pid {0}", thread_id);

  const ResumeAction *action =
      resume_actions.GetActionForThread(thread_id, true);
  if (action == NULL) {
    return Status("No action to perform...");
  }
  assert(action->tid == thread_id || action->tid == LLDB_INVALID_THREAD_ID);
  switch (action->state) {
  case lldb::StateType::eStateRunning:
    if (!m_dpu->ResumeThreads())
      return Status("CNI cannot resume");
    break;
  case lldb::StateType::eStateStepping: {
    ThreadDpu * thread = GetThreadByID(thread_id);
    uint32_t thread_index = thread->GetIndex();
    thread->SetThreadStepping();
    SetState(lldb::StateType::eStateStepping, true);
    LLDB_LOG(log, "stepping thread {0} with signal {1}", thread_index, action->signal);
    if (!m_dpu->StepThread(thread_index))
      return Status("CNI cannot step");
    SetState(lldb::StateType::eStateStopped, true);
  } break;
  default:
    return Status("Unknown resume action!");
  }

  return Status();
}

Status ProcessDpu::Halt() {
  Status error;

  return error;
}

Status ProcessDpu::Detach() {
  Status error;

  if (GetID() == LLDB_INVALID_PROCESS_ID)
    return error;

  // for (const auto &thread : m_threads) {
  //}

  return error;
}

Status ProcessDpu::Signal(int signo) {
  Status error;

  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
  LLDB_LOG(log, "sending signal {0} ({1}) to pid {1}", signo,
           Host::GetSignalAsCString(signo), GetID());

  return error;
}

Status ProcessDpu::Interrupt() {
  m_dpu->StopThreads();
  SetState(StateType::eStateStopped, true);
  return Status();
}

Status ProcessDpu::Kill() {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
  LLDB_LOG(log, "pid {0}", GetID());

  Status error;

  switch (m_state) {
  case StateType::eStateInvalid:
  case StateType::eStateExited:
  case StateType::eStateCrashed:
  case StateType::eStateDetached:
  case StateType::eStateUnloaded:
    // Nothing to do - the process is already dead.
    LLDB_LOG(log, "ignored for PID {0} due to current state: {1}", GetID(),
             m_state);
    return error;

  case StateType::eStateConnected:
  case StateType::eStateAttaching:
  case StateType::eStateLaunching:
  case StateType::eStateStopped:
  case StateType::eStateRunning:
  case StateType::eStateStepping:
  case StateType::eStateSuspended:
    // We can try to kill a process in these states.
    break;
  }

  // TODO: kill it

  return error;
}

Status ProcessDpu::AllocateMemory(size_t size, uint32_t permissions,
                                  lldb::addr_t &addr) {
  return Status("not implemented");
}

Status ProcessDpu::DeallocateMemory(lldb::addr_t addr) {
  return Status("not implemented");
}

lldb::addr_t ProcessDpu::GetSharedLibraryInfoAddress() {
  // punt on this for now
  return LLDB_INVALID_ADDRESS;
}

size_t ProcessDpu::UpdateThreads() { return m_threads.size(); }

Status ProcessDpu::SetBreakpoint(lldb::addr_t addr, uint32_t size,
                                 bool hardware) {
  if (hardware)
    return SetHardwareBreakpoint(addr, size);
  else
    return SetSoftwareBreakpoint(addr, size);
}

Status ProcessDpu::RemoveBreakpoint(lldb::addr_t addr, bool hardware) {
  if (hardware)
    return RemoveHardwareBreakpoint(addr);
  else
    return NativeProcessProtocol::RemoveBreakpoint(addr);
}

Status
ProcessDpu::GetSoftwareBreakpointTrapOpcode(size_t trap_opcode_size_hint,
                                            size_t &actual_opcode_size,
                                            const uint8_t *&trap_opcode_bytes) {
  static const uint8_t g_dpu_breakpoint_opcode[] = {0x00, 0x00, 0x00, 0x20,
                                                    0x63, 0x7e, 0x00, 0x00};

  trap_opcode_bytes = g_dpu_breakpoint_opcode;
  actual_opcode_size = sizeof(g_dpu_breakpoint_opcode);
  return Status();
}

Status ProcessDpu::ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                              size_t &bytes_read) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_MEMORY));
  LLDB_LOG(log, "addr = {0}, buf = {1}, size = {2}", addr, buf, size);

  if (addr >= k_dpu_iram_base) {
    if (!m_dpu->ReadIRAM(addr - k_dpu_iram_base, buf, size))
      return Status("Cannot copy from IRAM");
  } else if (addr >= k_dpu_mram_base) {
    if (!m_dpu->ReadMRAM(addr - k_dpu_mram_base, buf, size))
      return Status("Cannot copy from MRAM");
  } else {
    if (!m_dpu->ReadWRAM(addr, buf, size))
      return Status("Cannot copy from WRAM");
  }
  // TODO proper bytes_read
  bytes_read = size;

  return Status();
}

Status ProcessDpu::ReadMemoryWithoutTrap(lldb::addr_t addr, void *buf,
                                         size_t size, size_t &bytes_read) {
  Status error = ReadMemory(addr, buf, size, bytes_read);
  if (error.Fail())
    return error;
  return m_breakpoint_list.RemoveTrapsFromBuffer(addr, buf, size);
}

Status ProcessDpu::WriteMemory(lldb::addr_t addr, const void *buf, size_t size,
                               size_t &bytes_written) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_MEMORY));
  LLDB_LOG(log, "addr = {0}, buf = {1}, size = {2}", addr, buf, size);

  if (addr >= k_dpu_iram_base) {
    if (!m_dpu->WriteIRAM(addr - k_dpu_iram_base, buf, size))
      return Status("Cannot copy to IRAM");
  } else if (addr >= k_dpu_mram_base) {
    if (!m_dpu->WriteMRAM(addr - k_dpu_mram_base, buf, size))
      return Status("Cannot copy to MRAM");
  } else {
    if (!m_dpu->WriteWRAM(addr, buf, size))
      return Status("Cannot copy to WRAM");
  }
  // TODO proper bytes_written
  bytes_written = size;

  return Status();
}

Status ProcessDpu::GetLoadedModuleFileSpec(const char *module_path,
                                           FileSpec &file_spec) {
  return Status("Not Implemented");
}

Status ProcessDpu::GetFileLoadAddress(const llvm::StringRef &file_name,
                                      lldb::addr_t &load_addr) {
  return Status("Not Implemented");
}

ThreadDpu *ProcessDpu::GetThreadByID(lldb::tid_t tid) {
  return static_cast<ThreadDpu *>(NativeProcessProtocol::GetThreadByID(tid));
}

void ProcessDpu::GetThreadContext(int thread_index, uint32_t *&regs,
                                  uint16_t *&pc, bool *&zf, bool *&cf) {
  regs = m_dpu->ThreadContextRegs(thread_index);
  pc = m_dpu->ThreadContextPC(thread_index);
  zf = m_dpu->ThreadContextZF(thread_index);
  cf = m_dpu->ThreadContextCF(thread_index);
}

lldb::StateType ProcessDpu::GetThreadState(int thread_index,
                                           std::string &description,
                                           lldb::StopReason &stop_reason) {
  return m_dpu->GetThreadState(thread_index, description, stop_reason);
}
