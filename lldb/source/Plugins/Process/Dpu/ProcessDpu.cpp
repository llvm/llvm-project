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
#include "dpu_types.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/Section.h"
#include "lldb/Host/File.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Host/PseudoTerminal.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Host/common/NativeRegisterContext.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/State.h"
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
// Control interface polling period
const long k_ci_polling_interval_ns = 10000; /* ns */
} // end of anonymous namespace

// -----------------------------------------------------------------------------
// Public Static Methods
// -----------------------------------------------------------------------------

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
ProcessDpu::Factory::Launch(ProcessLaunchInfo &launch_info,
                            NativeDelegate &native_delegate,
                            MainLoop &mainloop) const {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));

  int terminal_fd = launch_info.GetPTY().ReleaseMasterFileDescriptor();
  if (terminal_fd == PseudoTerminal::invalid_fd) {
    terminal_fd = open(
        launch_info.GetFileActionForFD(STDOUT_FILENO)->GetPath().str().c_str(),
        O_RDWR);
  }

  FILE *stdout_fd = fdopen(terminal_fd, "w");
  if (stdout_fd == NULL) {
    return Status("Cannot open terminal_fd ").ToError();
  }

  DpuRank *rank = new DpuRank();
  bool success = rank->Open(NULL, stdout_fd);
  if (!success)
    return Status("Cannot get a DPU rank ").ToError();

  success = rank->Reset();
  if (!success)
    return Status("Cannot reset DPU rank ").ToError();

  Dpu *dpu = rank->GetDpu(0);
  success = dpu->LoadElf(launch_info.GetExecutableFile());
  if (!success)
    return Status("Cannot load Elf in DPU rank ").ToError();

  success = dpu->Boot();
  if (!success)
    return Status("Cannot boot DPU rank ").ToError();

  ::pid_t pid = 666 << 5; // TODO unique Rank ID
  LLDB_LOG(log, "Dpu Rank {0}", pid);

  return std::unique_ptr<ProcessDpu>(new ProcessDpu(
      pid, terminal_fd, native_delegate, k_dpu_arch, mainloop, rank, dpu));
}

#define SLICE_DELIM ":"
#define VALUE_DELIM "&"
#define PARSE_ENV(env_var, nr_cis, buffer)                                     \
  do {                                                                         \
    char *_PARSE_ENV_env = std::getenv(env_var);                               \
    if (_PARSE_ENV_env == NULL) {                                              \
      break;                                                                   \
    }                                                                          \
    char *_PARSE_ENV_ptr = strtok(_PARSE_ENV_env, SLICE_DELIM);                \
    if (_PARSE_ENV_ptr == NULL) {                                              \
      break;                                                                   \
    }                                                                          \
    do {                                                                       \
      uint32_t _PARSE_ENV_slice_id = ::strtoll(_PARSE_ENV_ptr, NULL, 10);      \
      _PARSE_ENV_ptr = strtok(NULL, VALUE_DELIM);                              \
      if (_PARSE_ENV_ptr == NULL) {                                            \
        return Status("Could not parse " env_var " correctly ").ToError();     \
      }                                                                        \
      buffer[_PARSE_ENV_slice_id] = ::strtoll(_PARSE_ENV_ptr, NULL, 10);       \
    } while ((_PARSE_ENV_ptr = strtok(NULL, SLICE_DELIM)) != NULL);            \
  } while (0)

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
ProcessDpu::Factory::Attach(
    lldb::pid_t pid, NativeProcessProtocol::NativeDelegate &native_delegate,
    MainLoop &mainloop) const {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
  LLDB_LOG(log, "attaching to pid = {0:x}", pid);

  unsigned int rank_id, slice_id, dpu_id;
  dpu_id = pid % 100;
  slice_id = (pid / 100) % 100;
  rank_id = (pid / (100 * 100)) % 100;

  char rank_path[128];
  sprintf(rank_path, "/dev/dpu_rank%u", rank_id);

  if (access(rank_path, F_OK) != 0) {
    return Status("'%s' does not exist. Run 'dpu-diag' for a full diagnostic  ",
                  rank_path)
        .ToError();
  }

  char profile[256];
  sprintf(profile, "backend=hw,rankPath=%s", rank_path);

  PseudoTerminal pseudo_terminal;
  if (!pseudo_terminal.OpenFirstAvailableMaster(O_RDWR | O_NOCTTY, nullptr,
                                                0)) {
    return Status("Cannot open first available master on pseudo terminal ")
        .ToError();
  }

  int terminal_fd = pseudo_terminal.ReleaseMasterFileDescriptor();
  if (terminal_fd == -1) {
    return Status("Cannot release master file descriptor ").ToError();
  }

  FILE *stdout_fd = fdopen(terminal_fd, "w");
  if (stdout_fd == NULL) {
    return Status("Cannot open terminal_fd ").ToError();
  }

  DpuRank *rank = new DpuRank();
  bool success = rank->Open(profile, stdout_fd);
  if (!success)
    return Status("Cannot get a DPU rank ").ToError();

  Dpu *dpu = rank->GetDpuFromSliceIdAndDpuId(slice_id, dpu_id);
  if (dpu == nullptr)
    return Status("Cannot find the DPU in the rank ").ToError();

  char *nr_tasklets_ptr = std::getenv("UPMEM_LLDB_NR_TASKLETS");

  if (nr_tasklets_ptr != NULL)
    dpu->SetNrThreads(::strtoll(nr_tasklets_ptr, NULL, 10));

  success = rank->SaveContext();
  if (!success)
    return Status("Cannot save the rank context ").ToError();

  uint8_t nr_cis = rank->GetNrCis();
  uint64_t *structures_value = new uint64_t[nr_cis];
  uint64_t *slices_target = new uint64_t[nr_cis];
  dpu_bitfield_t *host_muxs_mram_state = new dpu_bitfield_t[nr_cis];
  memset(structures_value, 0, sizeof(uint64_t) * nr_cis);
  memset(slices_target, 0, sizeof(uint64_t) * nr_cis);
  memset(host_muxs_mram_state, 0, sizeof(dpu_bitfield_t) * nr_cis);
  PARSE_ENV("UPMEM_LLDB_STRUCTURES_VALUE", nr_cis, structures_value);
  PARSE_ENV("UPMEM_LLDB_SLICES_TARGET", nr_cis, slices_target);
  PARSE_ENV("UPMEM_LLDB_HOST_MUXS_MRAM_STATE", nr_cis, host_muxs_mram_state);

  for (uint32_t each_ci = 0; each_ci < nr_cis; each_ci++) {
    rank->SetSliceInfo(each_ci, structures_value[each_ci],
                       slices_target[each_ci], host_muxs_mram_state[each_ci]);
    LLDB_LOG(log, "saving slice context ([{0}]: {1:x}, {2:x}, {3:x})", each_ci,
             structures_value[each_ci], slices_target[each_ci],
             host_muxs_mram_state[each_ci]);
  }
  delete[] structures_value;
  delete[] slices_target;
  delete[] host_muxs_mram_state;

  success = rank->StopDpus();
  if (!success)
    return Status("Cannot stop the rank ").ToError();

  dpu->SetAttachSession();

  return std::unique_ptr<ProcessDpu>(new ProcessDpu(
      pid, terminal_fd, native_delegate, k_dpu_arch, mainloop, rank, dpu));
}

// -----------------------------------------------------------------------------
// Public Instance Methods
// -----------------------------------------------------------------------------

ProcessDpu::ProcessDpu(::pid_t pid, int terminal_fd, NativeDelegate &delegate,
                       const ArchSpec &arch, MainLoop &mainloop, DpuRank *rank,
                       Dpu *dpu)
    : NativeProcessProtocol(pid, terminal_fd, delegate), m_arch(arch),
      m_dpu(dpu), m_rank(rank) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));

  // Set the timer for polling the CI
  LLDB_LOG(log, "Setting timer of polling the CI");
  struct itimerspec polling_spec;
  polling_spec.it_value.tv_sec = 0;
  polling_spec.it_value.tv_nsec = k_ci_polling_interval_ns;
  polling_spec.it_interval.tv_sec = 0;
  polling_spec.it_interval.tv_nsec = k_ci_polling_interval_ns;
  int tfd = timerfd_create(CLOCK_MONOTONIC, 0);
  assert(tfd != -1);
  m_timer_fd.reset(new NativeFile(tfd, File::eOpenOptionRead, true));

  Status status;
  m_timer_handle = mainloop.RegisterReadObject(
      m_timer_fd, [this](MainLoopBase &) { InterfaceTimerCallback(); }, status);
  timerfd_settime(tfd, 0, &polling_spec, nullptr);
  assert(m_timer_handle && status.Success());

  for (int thread_id = 0; thread_id < m_dpu->GetNrThreads(); thread_id++) {
    m_threads.push_back(
        std::make_unique<ThreadDpu>(*this, pid + thread_id, thread_id));
  }

  if (dpu->PrintfEnable()) {
    SetSoftwareBreakpoint(dpu->GetOpenPrintfSequenceAddr() | k_dpu_iram_base,
                          8);
    SetSoftwareBreakpoint(dpu->GetClosePrintfSequenceAddr() | k_dpu_iram_base,
                          8);
  }

  SetCurrentThreadID(pid);
  SetState(StateType::eStateStopped, false);

  dpu_description_t desc = rank->GetDesc();
  unsigned int iram_size =
      desc->hw.memories.iram_size * sizeof(dpuinstruction_t);
  unsigned int mram_size = desc->hw.memories.mram_size;
  unsigned int wram_size = desc->hw.memories.wram_size * sizeof(dpuword_t);

  m_iram_region.GetRange().SetRangeBase(k_dpu_iram_base);
  m_iram_region.GetRange().SetRangeEnd(k_dpu_iram_base + iram_size);
  m_iram_region.SetReadable(MemoryRegionInfo::eYes);
  m_iram_region.SetWritable(MemoryRegionInfo::eYes);
  m_iram_region.SetExecutable(MemoryRegionInfo::eYes);
  m_iram_region.SetMapped(MemoryRegionInfo::eYes);

  m_mram_region.GetRange().SetRangeBase(k_dpu_mram_base);
  m_mram_region.GetRange().SetRangeEnd(k_dpu_mram_base + mram_size);
  m_mram_region.SetReadable(MemoryRegionInfo::eYes);
  m_mram_region.SetWritable(MemoryRegionInfo::eYes);
  m_mram_region.SetExecutable(MemoryRegionInfo::eNo);
  m_mram_region.SetMapped(MemoryRegionInfo::eYes);

  m_wram_region.GetRange().SetRangeBase(k_dpu_wram_base);
  m_wram_region.GetRange().SetRangeEnd(k_dpu_wram_base + wram_size);
  m_wram_region.SetReadable(MemoryRegionInfo::eYes);
  m_wram_region.SetWritable(MemoryRegionInfo::eYes);
  m_wram_region.SetExecutable(MemoryRegionInfo::eNo);
  m_wram_region.SetMapped(MemoryRegionInfo::eYes);
}

void ProcessDpu::InterfaceTimerCallback() {
  unsigned int exit_status;
  StateType current_state = m_dpu->PollStatus(&exit_status);
  if (current_state != StateType::eStateInvalid) {
    if (current_state == StateType::eStateExited) {
      if (m_dpu->AttachSession()) {
        Detach();
      }
      SetExitStatus(WaitStatus(WaitStatus::Exit, (uint8_t)exit_status), true);
    }
    SetState(current_state, true);
  }
}

bool ProcessDpu::SupportHardwareSingleStepping() const { return true; }

bool ProcessDpu::StepThread(uint32_t thread_id) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
  unsigned int exit_status;
  lldb::StateType ret_state_type;
  LLDB_LOG(log, "stepping thread {0}", thread_id);
  ret_state_type = m_dpu->StepThread(thread_id, &exit_status);
  if (ret_state_type == lldb::StateType::eStateExited)
    SetExitStatus(WaitStatus(WaitStatus::Exit, (uint8_t)exit_status), true);
  if (ret_state_type != StateType::eStateStopped) {
    SetState(ret_state_type, true);
    return false;
  }
  return true;
}

Status ProcessDpu::Resume(const ResumeActionList &resume_actions) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
  llvm::SmallVector<uint32_t, 8> resume_list, stepping_list;

  for (const auto &thread : m_threads) {
    lldb::tid_t tid = thread->GetID();
    ThreadDpu *current_thread = GetThreadByID(tid);
    const ResumeAction *action = resume_actions.GetActionForThread(tid, true);
    if (action == NULL) {
      current_thread->SetSteppingMode(false);
      continue;
    }
    switch (action->state) {
    case lldb::StateType::eStateRunning:
      resume_list.push_back(current_thread->GetIndex());
      current_thread->SetSteppingMode(false);
      break;
    case lldb::StateType::eStateStepping:
      stepping_list.push_back(current_thread->GetIndex());
      current_thread->SetSteppingMode(true);
      break;
    default:
      return Status("Unexpected action!");
    }
  }
  LLDB_LOG(log, "{0} to resume {1} to step", resume_list.size(),
           stepping_list.size());

  if (stepping_list.empty()) {
    if (resume_list.empty()) {
      return Status("No Action to perform");
    }
    SetState(lldb::StateType::eStateRunning, true);
    LLDB_LOG(log, "resuming threads");
    if (!m_dpu->ResumeThreads(&resume_list))
      return Status("CNI cannot resume");
  } else {
    SetState(lldb::StateType::eStateStepping, true);
    for (auto thread_id : resume_list) {
      if (!StepThread(thread_id))
        return Status();
    }
    for (auto thread_id : stepping_list) {
      if (!StepThread(thread_id))
        return Status();
    }
    SetState(StateType::eStateStopped, true);
  }

  return Status();
}

Status ProcessDpu::Halt() {
  Status error;

  return error;
}

Status ProcessDpu::Detach() {
  Status error;
  bool success;

  if (m_dpu->PrintfEnable()) {
    RemoveSoftwareBreakpoint(m_dpu->GetOpenPrintfSequenceAddr() |
                             k_dpu_iram_base);
    RemoveSoftwareBreakpoint(m_dpu->GetClosePrintfSequenceAddr() |
                             k_dpu_iram_base);
  }

  success = m_rank->RestoreMuxContext();
  if (!success)
    return Status("Cannot restore the muxs context");

  success = m_rank->ResumeDpus();
  if (!success)
    return Status("Cannot resume DPUs");

  success = m_rank->RestoreContext();
  if (!success)
    return Status("Cannot restore the rank context context");

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
  if (!m_dpu->StopThreads())
    return Status("Cannot interrupt DPU");
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
    Detach();
    // We can try to kill a process in these states.
    break;
  }

  // TODO: kill it

  return error;
}

Status ProcessDpu::GetMemoryRegionInfo(lldb::addr_t load_addr,
                                       MemoryRegionInfo &range_info) {
  if (m_wram_region.GetRange().Contains(load_addr)) {
    range_info = m_wram_region;
  } else if (m_mram_region.GetRange().Contains(load_addr)) {
    range_info = m_mram_region;
  } else if (m_iram_region.GetRange().Contains(load_addr)) {
    range_info = m_iram_region;
  } else {
    range_info.GetRange().SetRangeBase(load_addr);
    range_info.GetRange().SetRangeEnd(LLDB_INVALID_ADDRESS);
    range_info.SetReadable(MemoryRegionInfo::eNo);
    range_info.SetWritable(MemoryRegionInfo::eNo);
    range_info.SetExecutable(MemoryRegionInfo::eNo);
    range_info.SetMapped(MemoryRegionInfo::eNo);
  }

  return Status();
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

Status ProcessDpu::ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                              size_t &bytes_read) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_MEMORY));
  LLDB_LOG(log, "addr = {0:X}, buf = {1}, size = {2}", addr, buf, size);

  bytes_read = 0;
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
  bytes_read = size;

  return Status();
}

Status ProcessDpu::WriteMemory(lldb::addr_t addr, const void *buf, size_t size,
                               size_t &bytes_written) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_MEMORY));
  LLDB_LOG(log, "addr = {0:X}, buf = {1}, size = {2}", addr, buf, size);

  bytes_written = 0;
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
                                  uint16_t *&pc, bool *&zf, bool *&cf,
                                  bool *&registers_has_been_modified) {
  regs = m_dpu->ThreadContextRegs(thread_index);
  pc = m_dpu->ThreadContextPC(thread_index);
  zf = m_dpu->ThreadContextZF(thread_index);
  cf = m_dpu->ThreadContextCF(thread_index);
  registers_has_been_modified = m_dpu->ThreadRegistersHasBeenModified();
}

lldb::StateType ProcessDpu::GetThreadState(int thread_index,
                                           std::string &description,
                                           lldb::StopReason &stop_reason,
                                           bool stepping) {
  return m_dpu->GetThreadState(thread_index, description, stop_reason,
                               stepping);
}

void ProcessDpu::SaveCore(const char *save_core_filename,
                          const char *executable_path, Status &error) {
  uint8_t *iram;
  uint32_t iram_size;
  if (!m_dpu->AllocIRAMBuffer(&iram, &iram_size)) {
    error.SetErrorString("Cannot alloc a IRAM Buffer");
    return;
  }

  // Read the iram without breakpoint
  size_t bytes_read;
  ReadMemoryWithoutTrap(k_dpu_iram_base, (void *)iram, iram_size, bytes_read);
  if (bytes_read != iram_size) {
    error.SetErrorString("Cannot read all the IRAM without trap");
    return;
  }

  if (!m_dpu->GenerateSaveCore(executable_path, save_core_filename, iram,
                               iram_size)) {
    error.SetErrorString("Cannot generate save core");
    return;
  }

  m_dpu->FreeIRAMBuffer(iram);
}

void ProcessDpu::SetDpuPrintInfo(const uint32_t open_print_sequence_addr,
                                 const uint32_t close_print_sequence_addr,
                                 const uint32_t print_buffer_addr,
                                 const uint32_t print_buffer_size,
                                 const uint32_t print_var_addr, Status &error) {
  if (!m_dpu->SetPrintfSequenceAddrs(
          open_print_sequence_addr, close_print_sequence_addr,
          print_buffer_addr, print_buffer_size, print_var_addr))
    error.SetErrorString("Cannot set Dpu print info");

  if (m_dpu->PrintfEnable()) {
    SetSoftwareBreakpoint(m_dpu->GetOpenPrintfSequenceAddr() | k_dpu_iram_base,
                          8);
    SetSoftwareBreakpoint(m_dpu->GetClosePrintfSequenceAddr() | k_dpu_iram_base,
                          8);
  }
}
