//===-- GDBRemoteCommunicationServerLLGS.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <errno.h>

#include "lldb/Host/Config.h"


#include <chrono>
#include <cstring>
#include <thread>

#include "GDBRemoteCommunicationServerLLGS.h"
#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/Debug.h"
#include "lldb/Host/File.h"
#include "lldb/Host/FileAction.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/PosixApi.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Host/common/NativeRegisterContext.h"
#include "lldb/Host/common/NativeThreadProtocol.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Utility/Args.h"
#include "lldb/Utility/DataBuffer.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/GDBRemote.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/State.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/UnimplementedError.h"
#include "lldb/Utility/UriParser.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/ScopedPrinter.h"

#include "ProcessGDBRemote.h"
#include "ProcessGDBRemoteLog.h"
#include "lldb/Utility/StringExtractorGDBRemote.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_gdb_remote;
using namespace llvm;

// GDBRemote Errors

namespace {
enum GDBRemoteServerError {
  // Set to the first unused error number in literal form below
  eErrorFirst = 29,
  eErrorNoProcess = eErrorFirst,
  eErrorResume,
  eErrorExitStatus
};
}

// GDBRemoteCommunicationServerLLGS constructor
GDBRemoteCommunicationServerLLGS::GDBRemoteCommunicationServerLLGS(
    MainLoop &mainloop, const NativeProcessProtocol::Factory &process_factory)
    : GDBRemoteCommunicationServerCommon("gdb-remote.server",
                                         "gdb-remote.server.rx_packet"),
      m_mainloop(mainloop), m_process_factory(process_factory),
      m_stdio_communication("process.stdio") {
  RegisterPacketHandlers();
}

void GDBRemoteCommunicationServerLLGS::RegisterPacketHandlers() {
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_C,
                                &GDBRemoteCommunicationServerLLGS::Handle_C);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_c,
                                &GDBRemoteCommunicationServerLLGS::Handle_c);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_D,
                                &GDBRemoteCommunicationServerLLGS::Handle_D);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_H,
                                &GDBRemoteCommunicationServerLLGS::Handle_H);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_I,
                                &GDBRemoteCommunicationServerLLGS::Handle_I);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_interrupt,
      &GDBRemoteCommunicationServerLLGS::Handle_interrupt);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_m,
      &GDBRemoteCommunicationServerLLGS::Handle_memory_read);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_M,
                                &GDBRemoteCommunicationServerLLGS::Handle_M);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_p,
                                &GDBRemoteCommunicationServerLLGS::Handle_p);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_P,
                                &GDBRemoteCommunicationServerLLGS::Handle_P);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_qC,
                                &GDBRemoteCommunicationServerLLGS::Handle_qC);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qfThreadInfo,
      &GDBRemoteCommunicationServerLLGS::Handle_qfThreadInfo);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qFileLoadAddress,
      &GDBRemoteCommunicationServerLLGS::Handle_qFileLoadAddress);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qGetWorkingDir,
      &GDBRemoteCommunicationServerLLGS::Handle_qGetWorkingDir);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qMemoryRegionInfo,
      &GDBRemoteCommunicationServerLLGS::Handle_qMemoryRegionInfo);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qMemoryRegionInfoSupported,
      &GDBRemoteCommunicationServerLLGS::Handle_qMemoryRegionInfoSupported);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qProcessInfo,
      &GDBRemoteCommunicationServerLLGS::Handle_qProcessInfo);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qRegisterInfo,
      &GDBRemoteCommunicationServerLLGS::Handle_qRegisterInfo);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_QRestoreRegisterState,
      &GDBRemoteCommunicationServerLLGS::Handle_QRestoreRegisterState);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_QSaveRegisterState,
      &GDBRemoteCommunicationServerLLGS::Handle_QSaveRegisterState);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_QSetDisableASLR,
      &GDBRemoteCommunicationServerLLGS::Handle_QSetDisableASLR);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_QSetWorkingDir,
      &GDBRemoteCommunicationServerLLGS::Handle_QSetWorkingDir);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qsThreadInfo,
      &GDBRemoteCommunicationServerLLGS::Handle_qsThreadInfo);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qThreadStopInfo,
      &GDBRemoteCommunicationServerLLGS::Handle_qThreadStopInfo);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_jThreadsInfo,
      &GDBRemoteCommunicationServerLLGS::Handle_jThreadsInfo);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qWatchpointSupportInfo,
      &GDBRemoteCommunicationServerLLGS::Handle_qWatchpointSupportInfo);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qXfer,
      &GDBRemoteCommunicationServerLLGS::Handle_qXfer);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_s,
                                &GDBRemoteCommunicationServerLLGS::Handle_s);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_stop_reason,
      &GDBRemoteCommunicationServerLLGS::Handle_stop_reason); // ?
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_vAttach,
      &GDBRemoteCommunicationServerLLGS::Handle_vAttach);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_vCont,
      &GDBRemoteCommunicationServerLLGS::Handle_vCont);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_vCont_actions,
      &GDBRemoteCommunicationServerLLGS::Handle_vCont_actions);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_x,
      &GDBRemoteCommunicationServerLLGS::Handle_memory_read);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_Z,
                                &GDBRemoteCommunicationServerLLGS::Handle_Z);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_z,
                                &GDBRemoteCommunicationServerLLGS::Handle_z);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_QPassSignals,
      &GDBRemoteCommunicationServerLLGS::Handle_QPassSignals);

  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_jTraceStart,
      &GDBRemoteCommunicationServerLLGS::Handle_jTraceStart);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_jTraceBufferRead,
      &GDBRemoteCommunicationServerLLGS::Handle_jTraceRead);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_jTraceMetaRead,
      &GDBRemoteCommunicationServerLLGS::Handle_jTraceRead);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_jTraceStop,
      &GDBRemoteCommunicationServerLLGS::Handle_jTraceStop);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_jTraceConfigRead,
      &GDBRemoteCommunicationServerLLGS::Handle_jTraceConfigRead);

  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_g,
                                &GDBRemoteCommunicationServerLLGS::Handle_g);

  RegisterPacketHandler(StringExtractorGDBRemote::eServerPacketType_k,
                        [this](StringExtractorGDBRemote packet, Status &error,
                               bool &interrupt, bool &quit) {
                          quit = true;
                          return this->Handle_k(packet);
                        });
}

void GDBRemoteCommunicationServerLLGS::SetLaunchInfo(const ProcessLaunchInfo &info) {
  m_process_launch_info = info;
}

Status GDBRemoteCommunicationServerLLGS::LaunchProcess() {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  if (!m_process_launch_info.GetArguments().GetArgumentCount())
    return Status("%s: no process command line specified to launch",
                  __FUNCTION__);

  const bool should_forward_stdio =
      m_process_launch_info.GetFileActionForFD(STDIN_FILENO) == nullptr ||
      m_process_launch_info.GetFileActionForFD(STDOUT_FILENO) == nullptr ||
      m_process_launch_info.GetFileActionForFD(STDERR_FILENO) == nullptr;
  m_process_launch_info.SetLaunchInSeparateProcessGroup(true);
  m_process_launch_info.GetFlags().Set(eLaunchFlagDebug);

  if (should_forward_stdio) {
    // Temporarily relax the following for Windows until we can take advantage
    // of the recently added pty support. This doesn't really affect the use of
    // lldb-server on Windows.
#if !defined(_WIN32)
    if (llvm::Error Err = m_process_launch_info.SetUpPtyRedirection())
      return Status(std::move(Err));
#endif
  }

  {
    std::lock_guard<std::recursive_mutex> guard(m_debugged_process_mutex);
    assert(!m_debugged_process_up && "lldb-server creating debugged "
                                     "process but one already exists");
    auto process_or =
        m_process_factory.Launch(m_process_launch_info, *this, m_mainloop);
    if (!process_or)
      return Status(process_or.takeError());
    m_debugged_process_up = std::move(*process_or);
  }

  // Handle mirroring of inferior stdout/stderr over the gdb-remote protocol as
  // needed. llgs local-process debugging may specify PTY paths, which will
  // make these file actions non-null process launch -i/e/o will also make
  // these file actions non-null nullptr means that the traffic is expected to
  // flow over gdb-remote protocol
  if (should_forward_stdio) {
    // nullptr means it's not redirected to file or pty (in case of LLGS local)
    // at least one of stdio will be transferred pty<->gdb-remote we need to
    // give the pty master handle to this object to read and/or write
    LLDB_LOG(log,
             "pid = {0}: setting up stdout/stderr redirection via $O "
             "gdb-remote commands",
             m_debugged_process_up->GetID());

    // Setup stdout/stderr mapping from inferior to $O
    auto terminal_fd = m_debugged_process_up->GetTerminalFileDescriptor();
    if (terminal_fd >= 0) {
      LLDB_LOGF(log,
                "ProcessGDBRemoteCommunicationServerLLGS::%s setting "
                "inferior STDIO fd to %d",
                __FUNCTION__, terminal_fd);
      Status status = SetSTDIOFileDescriptor(terminal_fd);
      if (status.Fail())
        return status;
    } else {
      LLDB_LOGF(log,
                "ProcessGDBRemoteCommunicationServerLLGS::%s ignoring "
                "inferior STDIO since terminal fd reported as %d",
                __FUNCTION__, terminal_fd);
    }
  } else {
    LLDB_LOG(log,
             "pid = {0} skipping stdout/stderr redirection via $O: inferior "
             "will communicate over client-provided file descriptors",
             m_debugged_process_up->GetID());
  }

  printf("Launched '%s' as process %" PRIu64 "...\n",
         m_process_launch_info.GetArguments().GetArgumentAtIndex(0),
         m_debugged_process_up->GetID());

  return Status();
}

Status GDBRemoteCommunicationServerLLGS::AttachToProcess(lldb::pid_t pid) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
  LLDB_LOGF(log, "GDBRemoteCommunicationServerLLGS::%s pid %" PRIu64,
            __FUNCTION__, pid);

  // Before we try to attach, make sure we aren't already monitoring something
  // else.
  if (m_debugged_process_up &&
      m_debugged_process_up->GetID() != LLDB_INVALID_PROCESS_ID)
    return Status("cannot attach to process %" PRIu64
                  " when another process with pid %" PRIu64
                  " is being debugged.",
                  pid, m_debugged_process_up->GetID());

  // Try to attach.
  auto process_or = m_process_factory.Attach(pid, *this, m_mainloop);
  if (!process_or) {
    Status status(process_or.takeError());
    llvm::errs() << llvm::formatv("failed to attach to process {0}: {1}", pid,
                                  status);
    return status;
  }
  m_debugged_process_up = std::move(*process_or);

  // Setup stdout/stderr mapping from inferior.
  auto terminal_fd = m_debugged_process_up->GetTerminalFileDescriptor();
  if (terminal_fd >= 0) {
    LLDB_LOGF(log,
              "ProcessGDBRemoteCommunicationServerLLGS::%s setting "
              "inferior STDIO fd to %d",
              __FUNCTION__, terminal_fd);
    Status status = SetSTDIOFileDescriptor(terminal_fd);
    if (status.Fail())
      return status;
  } else {
    LLDB_LOGF(log,
              "ProcessGDBRemoteCommunicationServerLLGS::%s ignoring "
              "inferior STDIO since terminal fd reported as %d",
              __FUNCTION__, terminal_fd);
  }

  printf("Attached to process %" PRIu64 "...\n", pid);
  return Status();
}

void GDBRemoteCommunicationServerLLGS::InitializeDelegate(
    NativeProcessProtocol *process) {
  assert(process && "process cannot be NULL");
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
  if (log) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s called with "
              "NativeProcessProtocol pid %" PRIu64 ", current state: %s",
              __FUNCTION__, process->GetID(),
              StateAsCString(process->GetState()));
  }
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::SendWResponse(
    NativeProcessProtocol *process) {
  assert(process && "process cannot be NULL");
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  // send W notification
  auto wait_status = process->GetExitStatus();
  if (!wait_status) {
    LLDB_LOG(log, "pid = {0}, failed to retrieve process exit status",
             process->GetID());

    StreamGDBRemote response;
    response.PutChar('E');
    response.PutHex8(GDBRemoteServerError::eErrorExitStatus);
    return SendPacketNoLock(response.GetString());
  }

  LLDB_LOG(log, "pid = {0}, returning exit type {1}", process->GetID(),
           *wait_status);

  StreamGDBRemote response;
  response.Format("{0:g}", *wait_status);
  return SendPacketNoLock(response.GetString());
}

static void AppendHexValue(StreamString &response, const uint8_t *buf,
                           uint32_t buf_size, bool swap) {
  int64_t i;
  if (swap) {
    for (i = buf_size - 1; i >= 0; i--)
      response.PutHex8(buf[i]);
  } else {
    for (i = 0; i < buf_size; i++)
      response.PutHex8(buf[i]);
  }
}

static llvm::StringRef GetEncodingNameOrEmpty(const RegisterInfo &reg_info) {
  switch (reg_info.encoding) {
  case eEncodingUint:
    return "uint";
  case eEncodingSint:
    return "sint";
  case eEncodingIEEE754:
    return "ieee754";
  case eEncodingVector:
    return "vector";
  default:
    return "";
  }
}

static llvm::StringRef GetFormatNameOrEmpty(const RegisterInfo &reg_info) {
  switch (reg_info.format) {
  case eFormatBinary:
    return "binary";
  case eFormatDecimal:
    return "decimal";
  case eFormatHex:
    return "hex";
  case eFormatFloat:
    return "float";
  case eFormatVectorOfSInt8:
    return "vector-sint8";
  case eFormatVectorOfUInt8:
    return "vector-uint8";
  case eFormatVectorOfSInt16:
    return "vector-sint16";
  case eFormatVectorOfUInt16:
    return "vector-uint16";
  case eFormatVectorOfSInt32:
    return "vector-sint32";
  case eFormatVectorOfUInt32:
    return "vector-uint32";
  case eFormatVectorOfFloat32:
    return "vector-float32";
  case eFormatVectorOfUInt64:
    return "vector-uint64";
  case eFormatVectorOfUInt128:
    return "vector-uint128";
  default:
    return "";
  };
}

static llvm::StringRef GetKindGenericOrEmpty(const RegisterInfo &reg_info) {
  switch (reg_info.kinds[RegisterKind::eRegisterKindGeneric]) {
  case LLDB_REGNUM_GENERIC_PC:
    return "pc";
  case LLDB_REGNUM_GENERIC_SP:
    return "sp";
  case LLDB_REGNUM_GENERIC_FP:
    return "fp";
  case LLDB_REGNUM_GENERIC_RA:
    return "ra";
  case LLDB_REGNUM_GENERIC_FLAGS:
    return "flags";
  case LLDB_REGNUM_GENERIC_ARG1:
    return "arg1";
  case LLDB_REGNUM_GENERIC_ARG2:
    return "arg2";
  case LLDB_REGNUM_GENERIC_ARG3:
    return "arg3";
  case LLDB_REGNUM_GENERIC_ARG4:
    return "arg4";
  case LLDB_REGNUM_GENERIC_ARG5:
    return "arg5";
  case LLDB_REGNUM_GENERIC_ARG6:
    return "arg6";
  case LLDB_REGNUM_GENERIC_ARG7:
    return "arg7";
  case LLDB_REGNUM_GENERIC_ARG8:
    return "arg8";
  default:
    return "";
  }
}

static void CollectRegNums(const uint32_t *reg_num, StreamString &response,
                           bool usehex) {
  for (int i = 0; *reg_num != LLDB_INVALID_REGNUM; ++reg_num, ++i) {
    if (i > 0)
      response.PutChar(',');
    if (usehex)
      response.Printf("%" PRIx32, *reg_num);
    else
      response.Printf("%" PRIu32, *reg_num);
  }
}

static void WriteRegisterValueInHexFixedWidth(
    StreamString &response, NativeRegisterContext &reg_ctx,
    const RegisterInfo &reg_info, const RegisterValue *reg_value_p,
    lldb::ByteOrder byte_order) {
  RegisterValue reg_value;
  if (!reg_value_p) {
    Status error = reg_ctx.ReadRegister(&reg_info, reg_value);
    if (error.Success())
      reg_value_p = &reg_value;
    // else log.
  }

  if (reg_value_p) {
    AppendHexValue(response, (const uint8_t *)reg_value_p->GetBytes(),
                   reg_value_p->GetByteSize(),
                   byte_order == lldb::eByteOrderLittle);
  } else {
    // Zero-out any unreadable values.
    if (reg_info.byte_size > 0) {
      std::basic_string<uint8_t> zeros(reg_info.byte_size, '\0');
      AppendHexValue(response, zeros.data(), zeros.size(), false);
    }
  }
}

static llvm::Expected<json::Object>
GetRegistersAsJSON(NativeThreadProtocol &thread) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

  NativeRegisterContext& reg_ctx = thread.GetRegisterContext();

  json::Object register_object;

#ifdef LLDB_JTHREADSINFO_FULL_REGISTER_SET
  // Expedite all registers in the first register set (i.e. should be GPRs)
  // that are not contained in other registers.
  const RegisterSet *reg_set_p = reg_ctx_sp->GetRegisterSet(0);
  if (!reg_set_p)
    return llvm::make_error<llvm::StringError>("failed to get registers",
                                               llvm::inconvertibleErrorCode());
  for (const uint32_t *reg_num_p = reg_set_p->registers;
       *reg_num_p != LLDB_INVALID_REGNUM; ++reg_num_p) {
    uint32_t reg_num = *reg_num_p;
#else
  // Expedite only a couple of registers until we figure out why sending
  // registers is expensive.
  static const uint32_t k_expedited_registers[] = {
      LLDB_REGNUM_GENERIC_PC, LLDB_REGNUM_GENERIC_SP, LLDB_REGNUM_GENERIC_FP,
      LLDB_REGNUM_GENERIC_RA, LLDB_INVALID_REGNUM};

  for (const uint32_t *generic_reg_p = k_expedited_registers;
       *generic_reg_p != LLDB_INVALID_REGNUM; ++generic_reg_p) {
    uint32_t reg_num = reg_ctx.ConvertRegisterKindToRegisterNumber(
        eRegisterKindGeneric, *generic_reg_p);
    if (reg_num == LLDB_INVALID_REGNUM)
      continue; // Target does not support the given register.
#endif

    const RegisterInfo *const reg_info_p =
        reg_ctx.GetRegisterInfoAtIndex(reg_num);
    if (reg_info_p == nullptr) {
      LLDB_LOGF(log,
                "%s failed to get register info for register index %" PRIu32,
                __FUNCTION__, reg_num);
      continue;
    }

    if (reg_info_p->value_regs != nullptr)
      continue; // Only expedite registers that are not contained in other
                // registers.

    RegisterValue reg_value;
    Status error = reg_ctx.ReadRegister(reg_info_p, reg_value);
    if (error.Fail()) {
      LLDB_LOGF(log, "%s failed to read register '%s' index %" PRIu32 ": %s",
                __FUNCTION__,
                reg_info_p->name ? reg_info_p->name : "<unnamed-register>",
                reg_num, error.AsCString());
      continue;
    }

    StreamString stream;
    WriteRegisterValueInHexFixedWidth(stream, reg_ctx, *reg_info_p,
                                      &reg_value, lldb::eByteOrderBig);

    register_object.try_emplace(llvm::to_string(reg_num),
                                stream.GetString().str());
  }

  return register_object;
}

static const char *GetStopReasonString(StopReason stop_reason) {
  switch (stop_reason) {
  case eStopReasonTrace:
    return "trace";
  case eStopReasonBreakpoint:
    return "breakpoint";
  case eStopReasonWatchpoint:
    return "watchpoint";
  case eStopReasonSignal:
    return "signal";
  case eStopReasonException:
    return "exception";
  case eStopReasonExec:
    return "exec";
  case eStopReasonInstrumentation:
  case eStopReasonInvalid:
  case eStopReasonPlanComplete:
  case eStopReasonThreadExiting:
  case eStopReasonNone:
    break; // ignored
  }
  return nullptr;
}

static llvm::Expected<json::Array>
GetJSONThreadsInfo(NativeProcessProtocol &process, bool abridged) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_THREAD));

  json::Array threads_array;

  // Ensure we can get info on the given thread.
  uint32_t thread_idx = 0;
  for (NativeThreadProtocol *thread;
       (thread = process.GetThreadAtIndex(thread_idx)) != nullptr;
       ++thread_idx) {

    lldb::tid_t tid = thread->GetID();

    // Grab the reason this thread stopped.
    struct ThreadStopInfo tid_stop_info;
    std::string description;
    if (!thread->GetStopReason(tid_stop_info, description))
      return llvm::make_error<llvm::StringError>(
          "failed to get stop reason", llvm::inconvertibleErrorCode());

    const int signum = tid_stop_info.details.signal.signo;
    if (log) {
      LLDB_LOGF(log,
                "GDBRemoteCommunicationServerLLGS::%s pid %" PRIu64
                " tid %" PRIu64
                " got signal signo = %d, reason = %d, exc_type = %" PRIu64,
                __FUNCTION__, process.GetID(), tid, signum,
                tid_stop_info.reason, tid_stop_info.details.exception.type);
    }

    json::Object thread_obj;

    if (!abridged) {
      if (llvm::Expected<json::Object> registers =
              GetRegistersAsJSON(*thread)) {
        thread_obj.try_emplace("registers", std::move(*registers));
      } else {
        return registers.takeError();
      }
    }

    thread_obj.try_emplace("tid", static_cast<int64_t>(tid));

    if (signum != 0)
      thread_obj.try_emplace("signal", signum);

    const std::string thread_name = thread->GetName();
    if (!thread_name.empty())
      thread_obj.try_emplace("name", thread_name);

    const char *stop_reason = GetStopReasonString(tid_stop_info.reason);
    if (stop_reason)
      thread_obj.try_emplace("reason", stop_reason);

    if (!description.empty())
      thread_obj.try_emplace("description", description);

    if ((tid_stop_info.reason == eStopReasonException) &&
        tid_stop_info.details.exception.type) {
      thread_obj.try_emplace(
          "metype", static_cast<int64_t>(tid_stop_info.details.exception.type));

      json::Array medata_array;
      for (uint32_t i = 0; i < tid_stop_info.details.exception.data_count;
           ++i) {
        medata_array.push_back(
            static_cast<int64_t>(tid_stop_info.details.exception.data[i]));
      }
      thread_obj.try_emplace("medata", std::move(medata_array));
    }
    threads_array.push_back(std::move(thread_obj));
  }
  return threads_array;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::SendStopReplyPacketForThread(
    lldb::tid_t tid) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_THREAD));

  // Ensure we have a debugged process.
  if (!m_debugged_process_up ||
      (m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID))
    return SendErrorResponse(50);

  LLDB_LOG(log, "preparing packet for pid {0} tid {1}",
           m_debugged_process_up->GetID(), tid);

  // Ensure we can get info on the given thread.
  NativeThreadProtocol *thread = m_debugged_process_up->GetThreadByID(tid);
  if (!thread)
    return SendErrorResponse(51);

  // Grab the reason this thread stopped.
  struct ThreadStopInfo tid_stop_info;
  std::string description;
  if (!thread->GetStopReason(tid_stop_info, description))
    return SendErrorResponse(52);

  // FIXME implement register handling for exec'd inferiors.
  // if (tid_stop_info.reason == eStopReasonExec) {
  //     const bool force = true;
  //     InitializeRegisters(force);
  // }

  StreamString response;
  // Output the T packet with the thread
  response.PutChar('T');
  int signum = tid_stop_info.details.signal.signo;
  LLDB_LOG(
      log,
      "pid {0}, tid {1}, got signal signo = {2}, reason = {3}, exc_type = {4}",
      m_debugged_process_up->GetID(), tid, signum, int(tid_stop_info.reason),
      tid_stop_info.details.exception.type);

  // Print the signal number.
  response.PutHex8(signum & 0xff);

  // Include the tid.
  response.Printf("thread:%" PRIx64 ";", tid);

  // Include the thread name if there is one.
  const std::string thread_name = thread->GetName();
  if (!thread_name.empty()) {
    size_t thread_name_len = thread_name.length();

    if (::strcspn(thread_name.c_str(), "$#+-;:") == thread_name_len) {
      response.PutCString("name:");
      response.PutCString(thread_name);
    } else {
      // The thread name contains special chars, send as hex bytes.
      response.PutCString("hexname:");
      response.PutStringAsRawHex8(thread_name);
    }
    response.PutChar(';');
  }

  // If a 'QListThreadsInStopReply' was sent to enable this feature, we will
  // send all thread IDs back in the "threads" key whose value is a list of hex
  // thread IDs separated by commas:
  //  "threads:10a,10b,10c;"
  // This will save the debugger from having to send a pair of qfThreadInfo and
  // qsThreadInfo packets, but it also might take a lot of room in the stop
  // reply packet, so it must be enabled only on systems where there are no
  // limits on packet lengths.
  if (m_list_threads_in_stop_reply) {
    response.PutCString("threads:");

    uint32_t thread_index = 0;
    NativeThreadProtocol *listed_thread;
    for (listed_thread = m_debugged_process_up->GetThreadAtIndex(thread_index);
         listed_thread; ++thread_index,
        listed_thread = m_debugged_process_up->GetThreadAtIndex(thread_index)) {
      if (thread_index > 0)
        response.PutChar(',');
      response.Printf("%" PRIx64, listed_thread->GetID());
    }
    response.PutChar(';');

    // Include JSON info that describes the stop reason for any threads that
    // actually have stop reasons. We use the new "jstopinfo" key whose values
    // is hex ascii JSON that contains the thread IDs thread stop info only for
    // threads that have stop reasons. Only send this if we have more than one
    // thread otherwise this packet has all the info it needs.
    if (thread_index > 1) {
      const bool threads_with_valid_stop_info_only = true;
      llvm::Expected<json::Array> threads_info = GetJSONThreadsInfo(
          *m_debugged_process_up, threads_with_valid_stop_info_only);
      if (threads_info) {
        response.PutCString("jstopinfo:");
        StreamString unescaped_response;
        unescaped_response.AsRawOstream() << std::move(*threads_info);
        response.PutStringAsRawHex8(unescaped_response.GetData());
        response.PutChar(';');
      } else {
        LLDB_LOG_ERROR(log, threads_info.takeError(),
                       "failed to prepare a jstopinfo field for pid {1}: {0}",
                       m_debugged_process_up->GetID());
      }
    }

    uint32_t i = 0;
    response.PutCString("thread-pcs");
    char delimiter = ':';
    for (NativeThreadProtocol *thread;
         (thread = m_debugged_process_up->GetThreadAtIndex(i)) != nullptr;
         ++i) {
      NativeRegisterContext& reg_ctx = thread->GetRegisterContext();

      uint32_t reg_to_read = reg_ctx.ConvertRegisterKindToRegisterNumber(
          eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
      const RegisterInfo *const reg_info_p =
          reg_ctx.GetRegisterInfoAtIndex(reg_to_read);

      RegisterValue reg_value;
      Status error = reg_ctx.ReadRegister(reg_info_p, reg_value);
      if (error.Fail()) {
        LLDB_LOGF(log, "%s failed to read register '%s' index %" PRIu32 ": %s",
                  __FUNCTION__,
                  reg_info_p->name ? reg_info_p->name : "<unnamed-register>",
                  reg_to_read, error.AsCString());
        continue;
      }

      response.PutChar(delimiter);
      delimiter = ',';
      WriteRegisterValueInHexFixedWidth(response, reg_ctx, *reg_info_p,
                                        &reg_value, endian::InlHostByteOrder());
    }

    response.PutChar(';');
  }

  //
  // Expedite registers.
  //

  // Grab the register context.
  NativeRegisterContext& reg_ctx = thread->GetRegisterContext();
  // Expedite all registers in the first register set (i.e. should be GPRs)
  // that are not contained in other registers.
  const RegisterSet *reg_set_p;
  if (reg_ctx.GetRegisterSetCount() > 0 &&
      ((reg_set_p = reg_ctx.GetRegisterSet(0)) != nullptr)) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s expediting registers "
              "from set '%s' (registers set count: %zu)",
              __FUNCTION__, reg_set_p->name ? reg_set_p->name : "<unnamed-set>",
              reg_set_p->num_registers);

    for (const uint32_t *reg_num_p = reg_set_p->registers;
         *reg_num_p != LLDB_INVALID_REGNUM; ++reg_num_p) {
      const RegisterInfo *const reg_info_p =
          reg_ctx.GetRegisterInfoAtIndex(*reg_num_p);
      if (reg_info_p == nullptr) {
        LLDB_LOGF(log,
                  "GDBRemoteCommunicationServerLLGS::%s failed to get "
                  "register info for register set '%s', register index "
                  "%" PRIu32,
                  __FUNCTION__,
                  reg_set_p->name ? reg_set_p->name : "<unnamed-set>",
                  *reg_num_p);
      } else if (reg_info_p->value_regs == nullptr) {
        // Only expediate registers that are not contained in other registers.
        RegisterValue reg_value;
        Status error = reg_ctx.ReadRegister(reg_info_p, reg_value);
        if (error.Success()) {
          response.Printf("%.02x:", *reg_num_p);
          WriteRegisterValueInHexFixedWidth(response, reg_ctx, *reg_info_p,
                                            &reg_value, lldb::eByteOrderBig);
          response.PutChar(';');
        } else {
          LLDB_LOGF(log,
                    "GDBRemoteCommunicationServerLLGS::%s failed to read "
                    "register '%s' index %" PRIu32 ": %s",
                    __FUNCTION__,
                    reg_info_p->name ? reg_info_p->name : "<unnamed-register>",
                    *reg_num_p, error.AsCString());
        }
      }
    }
  }

  const char *reason_str = GetStopReasonString(tid_stop_info.reason);
  if (reason_str != nullptr) {
    response.Printf("reason:%s;", reason_str);
  }

  if (!description.empty()) {
    // Description may contains special chars, send as hex bytes.
    response.PutCString("description:");
    response.PutStringAsRawHex8(description);
    response.PutChar(';');
  } else if ((tid_stop_info.reason == eStopReasonException) &&
             tid_stop_info.details.exception.type) {
    response.PutCString("metype:");
    response.PutHex64(tid_stop_info.details.exception.type);
    response.PutCString(";mecount:");
    response.PutHex32(tid_stop_info.details.exception.data_count);
    response.PutChar(';');

    for (uint32_t i = 0; i < tid_stop_info.details.exception.data_count; ++i) {
      response.PutCString("medata:");
      response.PutHex64(tid_stop_info.details.exception.data[i]);
      response.PutChar(';');
    }
  }

  return SendPacketNoLock(response.GetString());
}

void GDBRemoteCommunicationServerLLGS::HandleInferiorState_Exited(
    NativeProcessProtocol *process) {
  assert(process && "process cannot be NULL");

  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
  LLDB_LOGF(log, "GDBRemoteCommunicationServerLLGS::%s called", __FUNCTION__);

  PacketResult result = SendStopReasonForState(StateType::eStateExited);
  if (result != PacketResult::Success) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed to send stop "
              "notification for PID %" PRIu64 ", state: eStateExited",
              __FUNCTION__, process->GetID());
  }

  // Close the pipe to the inferior terminal i/o if we launched it and set one
  // up.
  MaybeCloseInferiorTerminalConnection();

  // We are ready to exit the debug monitor.
  m_exit_now = true;
  m_mainloop.RequestTermination();
}

void GDBRemoteCommunicationServerLLGS::HandleInferiorState_Stopped(
    NativeProcessProtocol *process) {
  assert(process && "process cannot be NULL");

  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
  LLDB_LOGF(log, "GDBRemoteCommunicationServerLLGS::%s called", __FUNCTION__);

  // Send the stop reason unless this is the stop after the launch or attach.
  switch (m_inferior_prev_state) {
  case eStateLaunching:
  case eStateAttaching:
    // Don't send anything per debugserver behavior.
    break;
  default:
    // In all other cases, send the stop reason.
    PacketResult result = SendStopReasonForState(StateType::eStateStopped);
    if (result != PacketResult::Success) {
      LLDB_LOGF(log,
                "GDBRemoteCommunicationServerLLGS::%s failed to send stop "
                "notification for PID %" PRIu64 ", state: eStateExited",
                __FUNCTION__, process->GetID());
    }
    break;
  }
}

void GDBRemoteCommunicationServerLLGS::ProcessStateChanged(
    NativeProcessProtocol *process, lldb::StateType state) {
  assert(process && "process cannot be NULL");
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
  if (log) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s called with "
              "NativeProcessProtocol pid %" PRIu64 ", state: %s",
              __FUNCTION__, process->GetID(), StateAsCString(state));
  }

  switch (state) {
  case StateType::eStateRunning:
    StartSTDIOForwarding();
    break;

  case StateType::eStateStopped:
    // Make sure we get all of the pending stdout/stderr from the inferior and
    // send it to the lldb host before we send the state change notification
    SendProcessOutput();
    // Then stop the forwarding, so that any late output (see llvm.org/pr25652)
    // does not interfere with our protocol.
    StopSTDIOForwarding();
    HandleInferiorState_Stopped(process);
    break;

  case StateType::eStateExited:
    // Same as above
    SendProcessOutput();
    StopSTDIOForwarding();
    HandleInferiorState_Exited(process);
    break;

  default:
    if (log) {
      LLDB_LOGF(log,
                "GDBRemoteCommunicationServerLLGS::%s didn't handle state "
                "change for pid %" PRIu64 ", new state: %s",
                __FUNCTION__, process->GetID(), StateAsCString(state));
    }
    break;
  }

  // Remember the previous state reported to us.
  m_inferior_prev_state = state;
}

void GDBRemoteCommunicationServerLLGS::DidExec(NativeProcessProtocol *process) {
  ClearProcessSpecificData();
}

void GDBRemoteCommunicationServerLLGS::DataAvailableCallback() {
  Log *log(GetLogIfAnyCategoriesSet(GDBR_LOG_COMM));

  if (!m_handshake_completed) {
    if (!HandshakeWithClient()) {
      LLDB_LOGF(log,
                "GDBRemoteCommunicationServerLLGS::%s handshake with "
                "client failed, exiting",
                __FUNCTION__);
      m_mainloop.RequestTermination();
      return;
    }
    m_handshake_completed = true;
  }

  bool interrupt = false;
  bool done = false;
  Status error;
  while (true) {
    const PacketResult result = GetPacketAndSendResponse(
        std::chrono::microseconds(0), error, interrupt, done);
    if (result == PacketResult::ErrorReplyTimeout)
      break; // No more packets in the queue

    if ((result != PacketResult::Success)) {
      LLDB_LOGF(log,
                "GDBRemoteCommunicationServerLLGS::%s processing a packet "
                "failed: %s",
                __FUNCTION__, error.AsCString());
      m_mainloop.RequestTermination();
      break;
    }
  }
}

Status GDBRemoteCommunicationServerLLGS::InitializeConnection(
    std::unique_ptr<Connection> connection) {
  IOObjectSP read_object_sp = connection->GetReadObject();
  GDBRemoteCommunicationServer::SetConnection(std::move(connection));

  Status error;
  m_network_handle_up = m_mainloop.RegisterReadObject(
      read_object_sp, [this](MainLoopBase &) { DataAvailableCallback(); },
      error);
  return error;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::SendONotification(const char *buffer,
                                                    uint32_t len) {
  if ((buffer == nullptr) || (len == 0)) {
    // Nothing to send.
    return PacketResult::Success;
  }

  StreamString response;
  response.PutChar('O');
  response.PutBytesAsRawHex8(buffer, len);

  return SendPacketNoLock(response.GetString());
}

Status GDBRemoteCommunicationServerLLGS::SetSTDIOFileDescriptor(int fd) {
  Status error;

  // Set up the reading/handling of process I/O
  std::unique_ptr<ConnectionFileDescriptor> conn_up(
      new ConnectionFileDescriptor(fd, true));
  if (!conn_up) {
    error.SetErrorString("failed to create ConnectionFileDescriptor");
    return error;
  }

  m_stdio_communication.SetCloseOnEOF(false);
  m_stdio_communication.SetConnection(std::move(conn_up));
  if (!m_stdio_communication.IsConnected()) {
    error.SetErrorString(
        "failed to set connection for inferior I/O communication");
    return error;
  }

  return Status();
}

void GDBRemoteCommunicationServerLLGS::StartSTDIOForwarding() {
  // Don't forward if not connected (e.g. when attaching).
  if (!m_stdio_communication.IsConnected())
    return;

  Status error;
  lldbassert(!m_stdio_handle_up);
  m_stdio_handle_up = m_mainloop.RegisterReadObject(
      m_stdio_communication.GetConnection()->GetReadObject(),
      [this](MainLoopBase &) { SendProcessOutput(); }, error);

  if (!m_stdio_handle_up) {
    // Not much we can do about the failure. Log it and continue without
    // forwarding.
    if (Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS))
      LLDB_LOGF(log,
                "GDBRemoteCommunicationServerLLGS::%s Failed to set up stdio "
                "forwarding: %s",
                __FUNCTION__, error.AsCString());
  }
}

void GDBRemoteCommunicationServerLLGS::StopSTDIOForwarding() {
  m_stdio_handle_up.reset();
}

void GDBRemoteCommunicationServerLLGS::SendProcessOutput() {
  char buffer[1024];
  ConnectionStatus status;
  Status error;
  while (true) {
    size_t bytes_read = m_stdio_communication.Read(
        buffer, sizeof buffer, std::chrono::microseconds(0), status, &error);
    switch (status) {
    case eConnectionStatusSuccess:
      SendONotification(buffer, bytes_read);
      break;
    case eConnectionStatusLostConnection:
    case eConnectionStatusEndOfFile:
    case eConnectionStatusError:
    case eConnectionStatusNoConnection:
      if (Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS))
        LLDB_LOGF(log,
                  "GDBRemoteCommunicationServerLLGS::%s Stopping stdio "
                  "forwarding as communication returned status %d (error: "
                  "%s)",
                  __FUNCTION__, status, error.AsCString());
      m_stdio_handle_up.reset();
      return;

    case eConnectionStatusInterrupted:
    case eConnectionStatusTimedOut:
      return;
    }
  }
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_jTraceStart(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
  // Fail if we don't have a current process.
  if (!m_debugged_process_up ||
      (m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID))
    return SendErrorResponse(68);

  if (!packet.ConsumeFront("jTraceStart:"))
    return SendIllFormedResponse(packet, "jTraceStart: Ill formed packet ");

  TraceOptions options;
  uint64_t type = std::numeric_limits<uint64_t>::max();
  uint64_t buffersize = std::numeric_limits<uint64_t>::max();
  lldb::tid_t tid = LLDB_INVALID_THREAD_ID;
  uint64_t metabuffersize = std::numeric_limits<uint64_t>::max();

  auto json_object = StructuredData::ParseJSON(packet.Peek());

  if (!json_object ||
      json_object->GetType() != lldb::eStructuredDataTypeDictionary)
    return SendIllFormedResponse(packet, "jTraceStart: Ill formed packet ");

  auto json_dict = json_object->GetAsDictionary();

  json_dict->GetValueForKeyAsInteger("metabuffersize", metabuffersize);
  options.setMetaDataBufferSize(metabuffersize);

  json_dict->GetValueForKeyAsInteger("buffersize", buffersize);
  options.setTraceBufferSize(buffersize);

  json_dict->GetValueForKeyAsInteger("type", type);
  options.setType(static_cast<lldb::TraceType>(type));

  json_dict->GetValueForKeyAsInteger("threadid", tid);
  options.setThreadID(tid);

  StructuredData::ObjectSP custom_params_sp =
      json_dict->GetValueForKey("params");
  if (custom_params_sp &&
      custom_params_sp->GetType() != lldb::eStructuredDataTypeDictionary)
    return SendIllFormedResponse(packet, "jTraceStart: Ill formed packet ");

  options.setTraceParams(
      std::static_pointer_cast<StructuredData::Dictionary>(custom_params_sp));

  if (buffersize == std::numeric_limits<uint64_t>::max() ||
      type != lldb::TraceType::eTraceTypeProcessorTrace) {
    LLDB_LOG(log, "Ill formed packet buffersize = {0} type = {1}", buffersize,
             type);
    return SendIllFormedResponse(packet, "JTrace:start: Ill formed packet ");
  }

  Status error;
  lldb::user_id_t uid = LLDB_INVALID_UID;
  uid = m_debugged_process_up->StartTrace(options, error);
  LLDB_LOG(log, "uid is {0} , error is {1}", uid, error.GetError());
  if (error.Fail())
    return SendErrorResponse(error);

  StreamGDBRemote response;
  response.Printf("%" PRIx64, uid);
  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_jTraceStop(
    StringExtractorGDBRemote &packet) {
  // Fail if we don't have a current process.
  if (!m_debugged_process_up ||
      (m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID))
    return SendErrorResponse(68);

  if (!packet.ConsumeFront("jTraceStop:"))
    return SendIllFormedResponse(packet, "jTraceStop: Ill formed packet ");

  lldb::user_id_t uid = LLDB_INVALID_UID;
  lldb::tid_t tid = LLDB_INVALID_THREAD_ID;

  auto json_object = StructuredData::ParseJSON(packet.Peek());

  if (!json_object ||
      json_object->GetType() != lldb::eStructuredDataTypeDictionary)
    return SendIllFormedResponse(packet, "jTraceStop: Ill formed packet ");

  auto json_dict = json_object->GetAsDictionary();

  if (!json_dict->GetValueForKeyAsInteger("traceid", uid))
    return SendIllFormedResponse(packet, "jTraceStop: Ill formed packet ");

  json_dict->GetValueForKeyAsInteger("threadid", tid);

  Status error = m_debugged_process_up->StopTrace(uid, tid);

  if (error.Fail())
    return SendErrorResponse(error);

  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_jTraceConfigRead(
    StringExtractorGDBRemote &packet) {

  // Fail if we don't have a current process.
  if (!m_debugged_process_up ||
      (m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID))
    return SendErrorResponse(68);

  if (!packet.ConsumeFront("jTraceConfigRead:"))
    return SendIllFormedResponse(packet,
                                 "jTraceConfigRead: Ill formed packet ");

  lldb::user_id_t uid = LLDB_INVALID_UID;
  lldb::tid_t threadid = LLDB_INVALID_THREAD_ID;

  auto json_object = StructuredData::ParseJSON(packet.Peek());

  if (!json_object ||
      json_object->GetType() != lldb::eStructuredDataTypeDictionary)
    return SendIllFormedResponse(packet,
                                 "jTraceConfigRead: Ill formed packet ");

  auto json_dict = json_object->GetAsDictionary();

  if (!json_dict->GetValueForKeyAsInteger("traceid", uid))
    return SendIllFormedResponse(packet,
                                 "jTraceConfigRead: Ill formed packet ");

  json_dict->GetValueForKeyAsInteger("threadid", threadid);

  TraceOptions options;
  StreamGDBRemote response;

  options.setThreadID(threadid);
  Status error = m_debugged_process_up->GetTraceConfig(uid, options);

  if (error.Fail())
    return SendErrorResponse(error);

  StreamGDBRemote escaped_response;
  StructuredData::Dictionary json_packet;

  json_packet.AddIntegerItem("type", options.getType());
  json_packet.AddIntegerItem("buffersize", options.getTraceBufferSize());
  json_packet.AddIntegerItem("metabuffersize", options.getMetaDataBufferSize());

  StructuredData::DictionarySP custom_params = options.getTraceParams();
  if (custom_params)
    json_packet.AddItem("params", custom_params);

  StreamString json_string;
  json_packet.Dump(json_string, false);
  escaped_response.PutEscapedBytes(json_string.GetData(),
                                   json_string.GetSize());
  return SendPacketNoLock(escaped_response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_jTraceRead(
    StringExtractorGDBRemote &packet) {

  // Fail if we don't have a current process.
  if (!m_debugged_process_up ||
      (m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID))
    return SendErrorResponse(68);

  enum PacketType { MetaData, BufferData };
  PacketType tracetype = MetaData;

  if (packet.ConsumeFront("jTraceBufferRead:"))
    tracetype = BufferData;
  else if (packet.ConsumeFront("jTraceMetaRead:"))
    tracetype = MetaData;
  else {
    return SendIllFormedResponse(packet, "jTrace: Ill formed packet ");
  }

  lldb::user_id_t uid = LLDB_INVALID_UID;

  uint64_t byte_count = std::numeric_limits<uint64_t>::max();
  lldb::tid_t tid = LLDB_INVALID_THREAD_ID;
  uint64_t offset = std::numeric_limits<uint64_t>::max();

  auto json_object = StructuredData::ParseJSON(packet.Peek());

  if (!json_object ||
      json_object->GetType() != lldb::eStructuredDataTypeDictionary)
    return SendIllFormedResponse(packet, "jTrace: Ill formed packet ");

  auto json_dict = json_object->GetAsDictionary();

  if (!json_dict->GetValueForKeyAsInteger("traceid", uid) ||
      !json_dict->GetValueForKeyAsInteger("offset", offset) ||
      !json_dict->GetValueForKeyAsInteger("buffersize", byte_count))
    return SendIllFormedResponse(packet, "jTrace: Ill formed packet ");

  json_dict->GetValueForKeyAsInteger("threadid", tid);

  // Allocate the response buffer.
  std::unique_ptr<uint8_t[]> buffer (new (std::nothrow) uint8_t[byte_count]);
  if (!buffer)
    return SendErrorResponse(0x78);

  StreamGDBRemote response;
  Status error;
  llvm::MutableArrayRef<uint8_t> buf(buffer.get(), byte_count);

  if (tracetype == BufferData)
    error = m_debugged_process_up->GetData(uid, tid, buf, offset);
  else if (tracetype == MetaData)
    error = m_debugged_process_up->GetMetaData(uid, tid, buf, offset);

  if (error.Fail())
    return SendErrorResponse(error);

  for (auto i : buf)
    response.PutHex8(i);

  StreamGDBRemote escaped_response;
  escaped_response.PutEscapedBytes(response.GetData(), response.GetSize());
  return SendPacketNoLock(escaped_response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qProcessInfo(
    StringExtractorGDBRemote &packet) {
  // Fail if we don't have a current process.
  if (!m_debugged_process_up ||
      (m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID))
    return SendErrorResponse(68);

  lldb::pid_t pid = m_debugged_process_up->GetID();

  if (pid == LLDB_INVALID_PROCESS_ID)
    return SendErrorResponse(1);

  ProcessInstanceInfo proc_info;
  if (!Host::GetProcessInfo(pid, proc_info))
    return SendErrorResponse(1);

  StreamString response;
  CreateProcessInfoResponse_DebugServerStyle(proc_info, response);
  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qC(StringExtractorGDBRemote &packet) {
  // Fail if we don't have a current process.
  if (!m_debugged_process_up ||
      (m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID))
    return SendErrorResponse(68);

  // Make sure we set the current thread so g and p packets return the data the
  // gdb will expect.
  lldb::tid_t tid = m_debugged_process_up->GetCurrentThreadID();
  SetCurrentThreadID(tid);

  NativeThreadProtocol *thread = m_debugged_process_up->GetCurrentThread();
  if (!thread)
    return SendErrorResponse(69);

  StreamString response;
  response.Printf("QC%" PRIx64, thread->GetID());

  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_k(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  StopSTDIOForwarding();

  if (!m_debugged_process_up) {
    LLDB_LOG(log, "No debugged process found.");
    return PacketResult::Success;
  }

  Status error = m_debugged_process_up->Kill();
  if (error.Fail())
    LLDB_LOG(log, "Failed to kill debugged process {0}: {1}",
             m_debugged_process_up->GetID(), error);

  // No OK response for kill packet.
  // return SendOKResponse ();
  return PacketResult::Success;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_QSetDisableASLR(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("QSetDisableASLR:"));
  if (packet.GetU32(0))
    m_process_launch_info.GetFlags().Set(eLaunchFlagDisableASLR);
  else
    m_process_launch_info.GetFlags().Clear(eLaunchFlagDisableASLR);
  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_QSetWorkingDir(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("QSetWorkingDir:"));
  std::string path;
  packet.GetHexByteString(path);
  m_process_launch_info.SetWorkingDirectory(FileSpec(path));
  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qGetWorkingDir(
    StringExtractorGDBRemote &packet) {
  FileSpec working_dir{m_process_launch_info.GetWorkingDirectory()};
  if (working_dir) {
    StreamString response;
    response.PutStringAsRawHex8(working_dir.GetCString());
    return SendPacketNoLock(response.GetString());
  }

  return SendErrorResponse(14);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_C(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_THREAD));
  LLDB_LOGF(log, "GDBRemoteCommunicationServerLLGS::%s called", __FUNCTION__);

  // Ensure we have a native process.
  if (!m_debugged_process_up) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s no debugged process "
              "shared pointer",
              __FUNCTION__);
    return SendErrorResponse(0x36);
  }

  // Pull out the signal number.
  packet.SetFilePos(::strlen("C"));
  if (packet.GetBytesLeft() < 1) {
    // Shouldn't be using a C without a signal.
    return SendIllFormedResponse(packet, "C packet specified without signal.");
  }
  const uint32_t signo =
      packet.GetHexMaxU32(false, std::numeric_limits<uint32_t>::max());
  if (signo == std::numeric_limits<uint32_t>::max())
    return SendIllFormedResponse(packet, "failed to parse signal number");

  // Handle optional continue address.
  if (packet.GetBytesLeft() > 0) {
    // FIXME add continue at address support for $C{signo}[;{continue-address}].
    if (*packet.Peek() == ';')
      return SendUnimplementedResponse(packet.GetStringRef().data());
    else
      return SendIllFormedResponse(
          packet, "unexpected content after $C{signal-number}");
  }

  ResumeActionList resume_actions(StateType::eStateRunning,
                                  LLDB_INVALID_SIGNAL_NUMBER);
  Status error;

  // We have two branches: what to do if a continue thread is specified (in
  // which case we target sending the signal to that thread), or when we don't
  // have a continue thread set (in which case we send a signal to the
  // process).

  // TODO discuss with Greg Clayton, make sure this makes sense.

  lldb::tid_t signal_tid = GetContinueThreadID();
  if (signal_tid != LLDB_INVALID_THREAD_ID) {
    // The resume action for the continue thread (or all threads if a continue
    // thread is not set).
    ResumeAction action = {GetContinueThreadID(), StateType::eStateRunning,
                           static_cast<int>(signo)};

    // Add the action for the continue thread (or all threads when the continue
    // thread isn't present).
    resume_actions.Append(action);
  } else {
    // Send the signal to the process since we weren't targeting a specific
    // continue thread with the signal.
    error = m_debugged_process_up->Signal(signo);
    if (error.Fail()) {
      LLDB_LOG(log, "failed to send signal for process {0}: {1}",
               m_debugged_process_up->GetID(), error);

      return SendErrorResponse(0x52);
    }
  }

  // Resume the threads.
  error = m_debugged_process_up->Resume(resume_actions);
  if (error.Fail()) {
    LLDB_LOG(log, "failed to resume threads for process {0}: {1}",
             m_debugged_process_up->GetID(), error);

    return SendErrorResponse(0x38);
  }

  // Don't send an "OK" packet; response is the stopped/exited message.
  return PacketResult::Success;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_c(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_THREAD));
  LLDB_LOGF(log, "GDBRemoteCommunicationServerLLGS::%s called", __FUNCTION__);

  packet.SetFilePos(packet.GetFilePos() + ::strlen("c"));

  // For now just support all continue.
  const bool has_continue_address = (packet.GetBytesLeft() > 0);
  if (has_continue_address) {
    LLDB_LOG(log, "not implemented for c[address] variant [{0} remains]",
             packet.Peek());
    return SendUnimplementedResponse(packet.GetStringRef().data());
  }

  // Ensure we have a native process.
  if (!m_debugged_process_up) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s no debugged process "
              "shared pointer",
              __FUNCTION__);
    return SendErrorResponse(0x36);
  }

  // Build the ResumeActionList
  ResumeActionList actions(StateType::eStateRunning,
                           LLDB_INVALID_SIGNAL_NUMBER);

  Status error = m_debugged_process_up->Resume(actions);
  if (error.Fail()) {
    LLDB_LOG(log, "c failed for process {0}: {1}",
             m_debugged_process_up->GetID(), error);
    return SendErrorResponse(GDBRemoteServerError::eErrorResume);
  }

  LLDB_LOG(log, "continued process {0}", m_debugged_process_up->GetID());
  // No response required from continue.
  return PacketResult::Success;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_vCont_actions(
    StringExtractorGDBRemote &packet) {
  StreamString response;
  response.Printf("vCont;c;C;s;S");

  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_vCont(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
  LLDB_LOGF(log, "GDBRemoteCommunicationServerLLGS::%s handling vCont packet",
            __FUNCTION__);

  packet.SetFilePos(::strlen("vCont"));

  if (packet.GetBytesLeft() == 0) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s missing action from "
              "vCont package",
              __FUNCTION__);
    return SendIllFormedResponse(packet, "Missing action from vCont package");
  }

  // Check if this is all continue (no options or ";c").
  if (::strcmp(packet.Peek(), ";c") == 0) {
    // Move past the ';', then do a simple 'c'.
    packet.SetFilePos(packet.GetFilePos() + 1);
    return Handle_c(packet);
  } else if (::strcmp(packet.Peek(), ";s") == 0) {
    // Move past the ';', then do a simple 's'.
    packet.SetFilePos(packet.GetFilePos() + 1);
    return Handle_s(packet);
  }

  // Ensure we have a native process.
  if (!m_debugged_process_up) {
    LLDB_LOG(log, "no debugged process");
    return SendErrorResponse(0x36);
  }

  ResumeActionList thread_actions;

  while (packet.GetBytesLeft() && *packet.Peek() == ';') {
    // Skip the semi-colon.
    packet.GetChar();

    // Build up the thread action.
    ResumeAction thread_action;
    thread_action.tid = LLDB_INVALID_THREAD_ID;
    thread_action.state = eStateInvalid;
    thread_action.signal = LLDB_INVALID_SIGNAL_NUMBER;

    const char action = packet.GetChar();
    switch (action) {
    case 'C':
      thread_action.signal = packet.GetHexMaxU32(false, 0);
      if (thread_action.signal == 0)
        return SendIllFormedResponse(
            packet, "Could not parse signal in vCont packet C action");
      LLVM_FALLTHROUGH;

    case 'c':
      // Continue
      thread_action.state = eStateRunning;
      break;

    case 'S':
      thread_action.signal = packet.GetHexMaxU32(false, 0);
      if (thread_action.signal == 0)
        return SendIllFormedResponse(
            packet, "Could not parse signal in vCont packet S action");
      LLVM_FALLTHROUGH;

    case 's':
      // Step
      thread_action.state = eStateStepping;
      break;

    default:
      return SendIllFormedResponse(packet, "Unsupported vCont action");
      break;
    }

    // Parse out optional :{thread-id} value.
    if (packet.GetBytesLeft() && (*packet.Peek() == ':')) {
      // Consume the separator.
      packet.GetChar();

      thread_action.tid = packet.GetHexMaxU32(false, LLDB_INVALID_THREAD_ID);
      if (thread_action.tid == LLDB_INVALID_THREAD_ID)
        return SendIllFormedResponse(
            packet, "Could not parse thread number in vCont packet");
    }

    thread_actions.Append(thread_action);
  }

  Status error = m_debugged_process_up->Resume(thread_actions);
  if (error.Fail()) {
    LLDB_LOG(log, "vCont failed for process {0}: {1}",
             m_debugged_process_up->GetID(), error);
    return SendErrorResponse(GDBRemoteServerError::eErrorResume);
  }

  LLDB_LOG(log, "continued process {0}", m_debugged_process_up->GetID());
  // No response required from vCont.
  return PacketResult::Success;
}

void GDBRemoteCommunicationServerLLGS::SetCurrentThreadID(lldb::tid_t tid) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));
  LLDB_LOG(log, "setting current thread id to {0}", tid);

  m_current_tid = tid;
  if (m_debugged_process_up)
    m_debugged_process_up->SetCurrentThreadID(m_current_tid);
}

void GDBRemoteCommunicationServerLLGS::SetContinueThreadID(lldb::tid_t tid) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));
  LLDB_LOG(log, "setting continue thread id to {0}", tid);

  m_continue_tid = tid;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_stop_reason(
    StringExtractorGDBRemote &packet) {
  // Handle the $? gdbremote command.

  // If no process, indicate error
  if (!m_debugged_process_up)
    return SendErrorResponse(02);

  return SendStopReasonForState(m_debugged_process_up->GetState());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::SendStopReasonForState(
    lldb::StateType process_state) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  switch (process_state) {
  case eStateAttaching:
  case eStateLaunching:
  case eStateRunning:
  case eStateStepping:
  case eStateDetached:
    // NOTE: gdb protocol doc looks like it should return $OK
    // when everything is running (i.e. no stopped result).
    return PacketResult::Success; // Ignore

  case eStateSuspended:
  case eStateStopped:
  case eStateCrashed: {
    lldb::tid_t tid = m_debugged_process_up->GetCurrentThreadID();
    // Make sure we set the current thread so g and p packets return the data
    // the gdb will expect.
    SetCurrentThreadID(tid);
    return SendStopReplyPacketForThread(tid);
  }

  case eStateInvalid:
  case eStateUnloaded:
  case eStateExited:
    return SendWResponse(m_debugged_process_up.get());

  default:
    LLDB_LOG(log, "pid {0}, current state reporting not handled: {1}",
             m_debugged_process_up->GetID(), process_state);
    break;
  }

  return SendErrorResponse(0);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qRegisterInfo(
    StringExtractorGDBRemote &packet) {
  // Fail if we don't have a current process.
  if (!m_debugged_process_up ||
      (m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID))
    return SendErrorResponse(68);

  // Ensure we have a thread.
  NativeThreadProtocol *thread = m_debugged_process_up->GetThreadAtIndex(0);
  if (!thread)
    return SendErrorResponse(69);

  // Get the register context for the first thread.
  NativeRegisterContext &reg_context = thread->GetRegisterContext();

  // Parse out the register number from the request.
  packet.SetFilePos(strlen("qRegisterInfo"));
  const uint32_t reg_index =
      packet.GetHexMaxU32(false, std::numeric_limits<uint32_t>::max());
  if (reg_index == std::numeric_limits<uint32_t>::max())
    return SendErrorResponse(69);

  // Return the end of registers response if we've iterated one past the end of
  // the register set.
  if (reg_index >= reg_context.GetUserRegisterCount())
    return SendErrorResponse(69);

  const RegisterInfo *reg_info = reg_context.GetRegisterInfoAtIndex(reg_index);
  if (!reg_info)
    return SendErrorResponse(69);

  // Build the reginfos response.
  StreamGDBRemote response;

  response.PutCString("name:");
  response.PutCString(reg_info->name);
  response.PutChar(';');

  if (reg_info->alt_name && reg_info->alt_name[0]) {
    response.PutCString("alt-name:");
    response.PutCString(reg_info->alt_name);
    response.PutChar(';');
  }

  response.Printf("bitsize:%" PRIu32 ";offset:%" PRIu32 ";",
                  reg_info->byte_size * 8, reg_info->byte_offset);

  llvm::StringRef encoding = GetEncodingNameOrEmpty(*reg_info);
  if (!encoding.empty())
    response << "encoding:" << encoding << ';';

  llvm::StringRef format = GetFormatNameOrEmpty(*reg_info);
  if (!format.empty())
    response << "format:" << format << ';';

  const char *const register_set_name =
      reg_context.GetRegisterSetNameForRegisterAtIndex(reg_index);
  if (register_set_name)
    response << "set:" << register_set_name << ';';

  if (reg_info->kinds[RegisterKind::eRegisterKindEHFrame] !=
      LLDB_INVALID_REGNUM)
    response.Printf("ehframe:%" PRIu32 ";",
                    reg_info->kinds[RegisterKind::eRegisterKindEHFrame]);

  if (reg_info->kinds[RegisterKind::eRegisterKindDWARF] != LLDB_INVALID_REGNUM)
    response.Printf("dwarf:%" PRIu32 ";",
                    reg_info->kinds[RegisterKind::eRegisterKindDWARF]);

  llvm::StringRef kind_generic = GetKindGenericOrEmpty(*reg_info);
  if (!kind_generic.empty())
    response << "generic:" << kind_generic << ';';

  if (reg_info->value_regs && reg_info->value_regs[0] != LLDB_INVALID_REGNUM) {
    response.PutCString("container-regs:");
    CollectRegNums(reg_info->value_regs, response, true);
    response.PutChar(';');
  }

  if (reg_info->invalidate_regs && reg_info->invalidate_regs[0]) {
    response.PutCString("invalidate-regs:");
    CollectRegNums(reg_info->invalidate_regs, response, true);
    response.PutChar(';');
  }

  if (reg_info->dynamic_size_dwarf_expr_bytes) {
    const size_t dwarf_opcode_len = reg_info->dynamic_size_dwarf_len;
    response.PutCString("dynamic_size_dwarf_expr_bytes:");
    for (uint32_t i = 0; i < dwarf_opcode_len; ++i)
      response.PutHex8(reg_info->dynamic_size_dwarf_expr_bytes[i]);
    response.PutChar(';');
  }
  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qfThreadInfo(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

  // Fail if we don't have a current process.
  if (!m_debugged_process_up ||
      (m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID)) {
    LLDB_LOG(log, "no process ({0}), returning OK",
             m_debugged_process_up ? "invalid process id"
                                   : "null m_debugged_process_up");
    return SendOKResponse();
  }

  StreamGDBRemote response;
  response.PutChar('m');

  LLDB_LOG(log, "starting thread iteration");
  NativeThreadProtocol *thread;
  uint32_t thread_index;
  for (thread_index = 0,
      thread = m_debugged_process_up->GetThreadAtIndex(thread_index);
       thread; ++thread_index,
      thread = m_debugged_process_up->GetThreadAtIndex(thread_index)) {
    LLDB_LOG(log, "iterated thread {0}(tid={2})", thread_index,
             thread->GetID());
    if (thread_index > 0)
      response.PutChar(',');
    response.Printf("%" PRIx64, thread->GetID());
  }

  LLDB_LOG(log, "finished thread iteration");
  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qsThreadInfo(
    StringExtractorGDBRemote &packet) {
  // FIXME for now we return the full thread list in the initial packet and
  // always do nothing here.
  return SendPacketNoLock("l");
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_g(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

  // Move past packet name.
  packet.SetFilePos(strlen("g"));

  // Get the thread to use.
  NativeThreadProtocol *thread = GetThreadFromSuffix(packet);
  if (!thread) {
    LLDB_LOG(log, "failed, no thread available");
    return SendErrorResponse(0x15);
  }

  // Get the thread's register context.
  NativeRegisterContext &reg_ctx = thread->GetRegisterContext();

  std::vector<uint8_t> regs_buffer;
  for (uint32_t reg_num = 0; reg_num < reg_ctx.GetUserRegisterCount();
       ++reg_num) {
    const RegisterInfo *reg_info = reg_ctx.GetRegisterInfoAtIndex(reg_num);

    if (reg_info == nullptr) {
      LLDB_LOG(log, "failed to get register info for register index {0}",
               reg_num);
      return SendErrorResponse(0x15);
    }

    if (reg_info->value_regs != nullptr)
      continue; // skip registers that are contained in other registers

    RegisterValue reg_value;
    Status error = reg_ctx.ReadRegister(reg_info, reg_value);
    if (error.Fail()) {
      LLDB_LOG(log, "failed to read register at index {0}", reg_num);
      return SendErrorResponse(0x15);
    }

    if (reg_info->byte_offset + reg_info->byte_size >= regs_buffer.size())
      // Resize the buffer to guarantee it can store the register offsetted
      // data.
      regs_buffer.resize(reg_info->byte_offset + reg_info->byte_size);

    // Copy the register offsetted data to the buffer.
    memcpy(regs_buffer.data() + reg_info->byte_offset, reg_value.GetBytes(),
           reg_info->byte_size);
  }

  // Write the response.
  StreamGDBRemote response;
  response.PutBytesAsRawHex8(regs_buffer.data(), regs_buffer.size());

  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_p(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

  // Parse out the register number from the request.
  packet.SetFilePos(strlen("p"));
  const uint32_t reg_index =
      packet.GetHexMaxU32(false, std::numeric_limits<uint32_t>::max());
  if (reg_index == std::numeric_limits<uint32_t>::max()) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed, could not "
              "parse register number from request \"%s\"",
              __FUNCTION__, packet.GetStringRef().data());
    return SendErrorResponse(0x15);
  }

  // Get the thread to use.
  NativeThreadProtocol *thread = GetThreadFromSuffix(packet);
  if (!thread) {
    LLDB_LOG(log, "failed, no thread available");
    return SendErrorResponse(0x15);
  }

  // Get the thread's register context.
  NativeRegisterContext &reg_context = thread->GetRegisterContext();

  // Return the end of registers response if we've iterated one past the end of
  // the register set.
  if (reg_index >= reg_context.GetUserRegisterCount()) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed, requested "
              "register %" PRIu32 " beyond register count %" PRIu32,
              __FUNCTION__, reg_index, reg_context.GetUserRegisterCount());
    return SendErrorResponse(0x15);
  }

  const RegisterInfo *reg_info = reg_context.GetRegisterInfoAtIndex(reg_index);
  if (!reg_info) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed, requested "
              "register %" PRIu32 " returned NULL",
              __FUNCTION__, reg_index);
    return SendErrorResponse(0x15);
  }

  // Build the reginfos response.
  StreamGDBRemote response;

  // Retrieve the value
  RegisterValue reg_value;
  Status error = reg_context.ReadRegister(reg_info, reg_value);
  if (error.Fail()) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed, read of "
              "requested register %" PRIu32 " (%s) failed: %s",
              __FUNCTION__, reg_index, reg_info->name, error.AsCString());
    return SendErrorResponse(0x15);
  }

  const uint8_t *const data =
      static_cast<const uint8_t *>(reg_value.GetBytes());
  if (!data) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed to get data "
              "bytes from requested register %" PRIu32,
              __FUNCTION__, reg_index);
    return SendErrorResponse(0x15);
  }

  // FIXME flip as needed to get data in big/little endian format for this host.
  for (uint32_t i = 0; i < reg_value.GetByteSize(); ++i)
    response.PutHex8(data[i]);

  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_P(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

  // Ensure there is more content.
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(packet, "Empty P packet");

  // Parse out the register number from the request.
  packet.SetFilePos(strlen("P"));
  const uint32_t reg_index =
      packet.GetHexMaxU32(false, std::numeric_limits<uint32_t>::max());
  if (reg_index == std::numeric_limits<uint32_t>::max()) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed, could not "
              "parse register number from request \"%s\"",
              __FUNCTION__, packet.GetStringRef().data());
    return SendErrorResponse(0x29);
  }

  // Note debugserver would send an E30 here.
  if ((packet.GetBytesLeft() < 1) || (packet.GetChar() != '='))
    return SendIllFormedResponse(
        packet, "P packet missing '=' char after register number");

  // Parse out the value.
  uint8_t reg_bytes[RegisterValue::kMaxRegisterByteSize];
  size_t reg_size = packet.GetHexBytesAvail(reg_bytes);

  // Get the thread to use.
  NativeThreadProtocol *thread = GetThreadFromSuffix(packet);
  if (!thread) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed, no thread "
              "available (thread index 0)",
              __FUNCTION__);
    return SendErrorResponse(0x28);
  }

  // Get the thread's register context.
  NativeRegisterContext &reg_context = thread->GetRegisterContext();
  const RegisterInfo *reg_info = reg_context.GetRegisterInfoAtIndex(reg_index);
  if (!reg_info) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed, requested "
              "register %" PRIu32 " returned NULL",
              __FUNCTION__, reg_index);
    return SendErrorResponse(0x48);
  }

  // Return the end of registers response if we've iterated one past the end of
  // the register set.
  if (reg_index >= reg_context.GetUserRegisterCount()) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed, requested "
              "register %" PRIu32 " beyond register count %" PRIu32,
              __FUNCTION__, reg_index, reg_context.GetUserRegisterCount());
    return SendErrorResponse(0x47);
  }

  // The dwarf expression are evaluate on host site which may cause register
  // size to change Hence the reg_size may not be same as reg_info->bytes_size
  if ((reg_size != reg_info->byte_size) &&
      !(reg_info->dynamic_size_dwarf_expr_bytes)) {
    return SendIllFormedResponse(packet, "P packet register size is incorrect");
  }

  // Build the reginfos response.
  StreamGDBRemote response;

  RegisterValue reg_value(
      reg_bytes, reg_size,
      m_debugged_process_up->GetArchitecture().GetByteOrder());
  Status error = reg_context.WriteRegister(reg_info, reg_value);
  if (error.Fail()) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed, write of "
              "requested register %" PRIu32 " (%s) failed: %s",
              __FUNCTION__, reg_index, reg_info->name, error.AsCString());
    return SendErrorResponse(0x32);
  }

  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_H(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

  // Fail if we don't have a current process.
  if (!m_debugged_process_up ||
      (m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID)) {
    LLDB_LOGF(
        log,
        "GDBRemoteCommunicationServerLLGS::%s failed, no process available",
        __FUNCTION__);
    return SendErrorResponse(0x15);
  }

  // Parse out which variant of $H is requested.
  packet.SetFilePos(strlen("H"));
  if (packet.GetBytesLeft() < 1) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed, H command "
              "missing {g,c} variant",
              __FUNCTION__);
    return SendIllFormedResponse(packet, "H command missing {g,c} variant");
  }

  const char h_variant = packet.GetChar();
  switch (h_variant) {
  case 'g':
    break;

  case 'c':
    break;

  default:
    LLDB_LOGF(
        log,
        "GDBRemoteCommunicationServerLLGS::%s failed, invalid $H variant %c",
        __FUNCTION__, h_variant);
    return SendIllFormedResponse(packet,
                                 "H variant unsupported, should be c or g");
  }

  // Parse out the thread number.
  // FIXME return a parse success/fail value.  All values are valid here.
  const lldb::tid_t tid =
      packet.GetHexMaxU64(false, std::numeric_limits<lldb::tid_t>::max());

  // Ensure we have the given thread when not specifying -1 (all threads) or 0
  // (any thread).
  if (tid != LLDB_INVALID_THREAD_ID && tid != 0) {
    NativeThreadProtocol *thread = m_debugged_process_up->GetThreadByID(tid);
    if (!thread) {
      LLDB_LOGF(log,
                "GDBRemoteCommunicationServerLLGS::%s failed, tid %" PRIu64
                " not found",
                __FUNCTION__, tid);
      return SendErrorResponse(0x15);
    }
  }

  // Now switch the given thread type.
  switch (h_variant) {
  case 'g':
    SetCurrentThreadID(tid);
    break;

  case 'c':
    SetContinueThreadID(tid);
    break;

  default:
    assert(false && "unsupported $H variant - shouldn't get here");
    return SendIllFormedResponse(packet,
                                 "H variant unsupported, should be c or g");
  }

  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_I(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

  // Fail if we don't have a current process.
  if (!m_debugged_process_up ||
      (m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID)) {
    LLDB_LOGF(
        log,
        "GDBRemoteCommunicationServerLLGS::%s failed, no process available",
        __FUNCTION__);
    return SendErrorResponse(0x15);
  }

  packet.SetFilePos(::strlen("I"));
  uint8_t tmp[4096];
  for (;;) {
    size_t read = packet.GetHexBytesAvail(tmp);
    if (read == 0) {
      break;
    }
    // write directly to stdin *this might block if stdin buffer is full*
    // TODO: enqueue this block in circular buffer and send window size to
    // remote host
    ConnectionStatus status;
    Status error;
    m_stdio_communication.Write(tmp, read, status, &error);
    if (error.Fail()) {
      return SendErrorResponse(0x15);
    }
  }

  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_interrupt(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_THREAD));

  // Fail if we don't have a current process.
  if (!m_debugged_process_up ||
      (m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID)) {
    LLDB_LOG(log, "failed, no process available");
    return SendErrorResponse(0x15);
  }

  // Interrupt the process.
  Status error = m_debugged_process_up->Interrupt();
  if (error.Fail()) {
    LLDB_LOG(log, "failed for process {0}: {1}", m_debugged_process_up->GetID(),
             error);
    return SendErrorResponse(GDBRemoteServerError::eErrorResume);
  }

  LLDB_LOG(log, "stopped process {0}", m_debugged_process_up->GetID());

  // No response required from stop all.
  return PacketResult::Success;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_memory_read(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  if (!m_debugged_process_up ||
      (m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID)) {
    LLDB_LOGF(
        log,
        "GDBRemoteCommunicationServerLLGS::%s failed, no process available",
        __FUNCTION__);
    return SendErrorResponse(0x15);
  }

  // Parse out the memory address.
  packet.SetFilePos(strlen("m"));
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(packet, "Too short m packet");

  // Read the address.  Punting on validation.
  // FIXME replace with Hex U64 read with no default value that fails on failed
  // read.
  const lldb::addr_t read_addr = packet.GetHexMaxU64(false, 0);

  // Validate comma.
  if ((packet.GetBytesLeft() < 1) || (packet.GetChar() != ','))
    return SendIllFormedResponse(packet, "Comma sep missing in m packet");

  // Get # bytes to read.
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(packet, "Length missing in m packet");

  const uint64_t byte_count = packet.GetHexMaxU64(false, 0);
  if (byte_count == 0) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s nothing to read: "
              "zero-length packet",
              __FUNCTION__);
    return SendOKResponse();
  }

  // Allocate the response buffer.
  std::string buf(byte_count, '\0');
  if (buf.empty())
    return SendErrorResponse(0x78);

  // Retrieve the process memory.
  size_t bytes_read = 0;
  Status error = m_debugged_process_up->ReadMemoryWithoutTrap(
      read_addr, &buf[0], byte_count, bytes_read);
  if (error.Fail()) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s pid %" PRIu64
              " mem 0x%" PRIx64 ": failed to read. Error: %s",
              __FUNCTION__, m_debugged_process_up->GetID(), read_addr,
              error.AsCString());
    return SendErrorResponse(0x08);
  }

  if (bytes_read == 0) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s pid %" PRIu64
              " mem 0x%" PRIx64 ": read 0 of %" PRIu64 " requested bytes",
              __FUNCTION__, m_debugged_process_up->GetID(), read_addr,
              byte_count);
    return SendErrorResponse(0x08);
  }

  StreamGDBRemote response;
  packet.SetFilePos(0);
  char kind = packet.GetChar('?');
  if (kind == 'x')
    response.PutEscapedBytes(buf.data(), byte_count);
  else {
    assert(kind == 'm');
    for (size_t i = 0; i < bytes_read; ++i)
      response.PutHex8(buf[i]);
  }

  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_M(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  if (!m_debugged_process_up ||
      (m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID)) {
    LLDB_LOGF(
        log,
        "GDBRemoteCommunicationServerLLGS::%s failed, no process available",
        __FUNCTION__);
    return SendErrorResponse(0x15);
  }

  // Parse out the memory address.
  packet.SetFilePos(strlen("M"));
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(packet, "Too short M packet");

  // Read the address.  Punting on validation.
  // FIXME replace with Hex U64 read with no default value that fails on failed
  // read.
  const lldb::addr_t write_addr = packet.GetHexMaxU64(false, 0);

  // Validate comma.
  if ((packet.GetBytesLeft() < 1) || (packet.GetChar() != ','))
    return SendIllFormedResponse(packet, "Comma sep missing in M packet");

  // Get # bytes to read.
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(packet, "Length missing in M packet");

  const uint64_t byte_count = packet.GetHexMaxU64(false, 0);
  if (byte_count == 0) {
    LLDB_LOG(log, "nothing to write: zero-length packet");
    return PacketResult::Success;
  }

  // Validate colon.
  if ((packet.GetBytesLeft() < 1) || (packet.GetChar() != ':'))
    return SendIllFormedResponse(
        packet, "Comma sep missing in M packet after byte length");

  // Allocate the conversion buffer.
  std::vector<uint8_t> buf(byte_count, 0);
  if (buf.empty())
    return SendErrorResponse(0x78);

  // Convert the hex memory write contents to bytes.
  StreamGDBRemote response;
  const uint64_t convert_count = packet.GetHexBytes(buf, 0);
  if (convert_count != byte_count) {
    LLDB_LOG(log,
             "pid {0} mem {1:x}: asked to write {2} bytes, but only found {3} "
             "to convert.",
             m_debugged_process_up->GetID(), write_addr, byte_count,
             convert_count);
    return SendIllFormedResponse(packet, "M content byte length specified did "
                                         "not match hex-encoded content "
                                         "length");
  }

  // Write the process memory.
  size_t bytes_written = 0;
  Status error = m_debugged_process_up->WriteMemory(write_addr, &buf[0],
                                                    byte_count, bytes_written);
  if (error.Fail()) {
    LLDB_LOG(log, "pid {0} mem {1:x}: failed to write. Error: {2}",
             m_debugged_process_up->GetID(), write_addr, error);
    return SendErrorResponse(0x09);
  }

  if (bytes_written == 0) {
    LLDB_LOG(log, "pid {0} mem {1:x}: wrote 0 of {2} requested bytes",
             m_debugged_process_up->GetID(), write_addr, byte_count);
    return SendErrorResponse(0x09);
  }

  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qMemoryRegionInfoSupported(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  // Currently only the NativeProcessProtocol knows if it can handle a
  // qMemoryRegionInfoSupported request, but we're not guaranteed to be
  // attached to a process.  For now we'll assume the client only asks this
  // when a process is being debugged.

  // Ensure we have a process running; otherwise, we can't figure this out
  // since we won't have a NativeProcessProtocol.
  if (!m_debugged_process_up ||
      (m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID)) {
    LLDB_LOGF(
        log,
        "GDBRemoteCommunicationServerLLGS::%s failed, no process available",
        __FUNCTION__);
    return SendErrorResponse(0x15);
  }

  // Test if we can get any region back when asking for the region around NULL.
  MemoryRegionInfo region_info;
  const Status error =
      m_debugged_process_up->GetMemoryRegionInfo(0, region_info);
  if (error.Fail()) {
    // We don't support memory region info collection for this
    // NativeProcessProtocol.
    return SendUnimplementedResponse("");
  }

  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qMemoryRegionInfo(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  // Ensure we have a process.
  if (!m_debugged_process_up ||
      (m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID)) {
    LLDB_LOGF(
        log,
        "GDBRemoteCommunicationServerLLGS::%s failed, no process available",
        __FUNCTION__);
    return SendErrorResponse(0x15);
  }

  // Parse out the memory address.
  packet.SetFilePos(strlen("qMemoryRegionInfo:"));
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(packet, "Too short qMemoryRegionInfo: packet");

  // Read the address.  Punting on validation.
  const lldb::addr_t read_addr = packet.GetHexMaxU64(false, 0);

  StreamGDBRemote response;

  // Get the memory region info for the target address.
  MemoryRegionInfo region_info;
  const Status error =
      m_debugged_process_up->GetMemoryRegionInfo(read_addr, region_info);
  if (error.Fail()) {
    // Return the error message.

    response.PutCString("error:");
    response.PutStringAsRawHex8(error.AsCString());
    response.PutChar(';');
  } else {
    // Range start and size.
    response.Printf("start:%" PRIx64 ";size:%" PRIx64 ";",
                    region_info.GetRange().GetRangeBase(),
                    region_info.GetRange().GetByteSize());

    // Permissions.
    if (region_info.GetReadable() || region_info.GetWritable() ||
        region_info.GetExecutable()) {
      // Write permissions info.
      response.PutCString("permissions:");

      if (region_info.GetReadable())
        response.PutChar('r');
      if (region_info.GetWritable())
        response.PutChar('w');
      if (region_info.GetExecutable())
        response.PutChar('x');

      response.PutChar(';');
    }

    // Name
    ConstString name = region_info.GetName();
    if (name) {
      response.PutCString("name:");
      response.PutStringAsRawHex8(name.GetStringRef());
      response.PutChar(';');
    }
  }

  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_Z(StringExtractorGDBRemote &packet) {
  // Ensure we have a process.
  if (!m_debugged_process_up ||
      (m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID)) {
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
    LLDB_LOG(log, "failed, no process available");
    return SendErrorResponse(0x15);
  }

  // Parse out software or hardware breakpoint or watchpoint requested.
  packet.SetFilePos(strlen("Z"));
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(
        packet, "Too short Z packet, missing software/hardware specifier");

  bool want_breakpoint = true;
  bool want_hardware = false;
  uint32_t watch_flags = 0;

  const GDBStoppointType stoppoint_type =
      GDBStoppointType(packet.GetS32(eStoppointInvalid));
  switch (stoppoint_type) {
  case eBreakpointSoftware:
    want_hardware = false;
    want_breakpoint = true;
    break;
  case eBreakpointHardware:
    want_hardware = true;
    want_breakpoint = true;
    break;
  case eWatchpointWrite:
    watch_flags = 1;
    want_hardware = true;
    want_breakpoint = false;
    break;
  case eWatchpointRead:
    watch_flags = 2;
    want_hardware = true;
    want_breakpoint = false;
    break;
  case eWatchpointReadWrite:
    watch_flags = 3;
    want_hardware = true;
    want_breakpoint = false;
    break;
  case eStoppointInvalid:
    return SendIllFormedResponse(
        packet, "Z packet had invalid software/hardware specifier");
  }

  if ((packet.GetBytesLeft() < 1) || packet.GetChar() != ',')
    return SendIllFormedResponse(
        packet, "Malformed Z packet, expecting comma after stoppoint type");

  // Parse out the stoppoint address.
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(packet, "Too short Z packet, missing address");
  const lldb::addr_t addr = packet.GetHexMaxU64(false, 0);

  if ((packet.GetBytesLeft() < 1) || packet.GetChar() != ',')
    return SendIllFormedResponse(
        packet, "Malformed Z packet, expecting comma after address");

  // Parse out the stoppoint size (i.e. size hint for opcode size).
  const uint32_t size =
      packet.GetHexMaxU32(false, std::numeric_limits<uint32_t>::max());
  if (size == std::numeric_limits<uint32_t>::max())
    return SendIllFormedResponse(
        packet, "Malformed Z packet, failed to parse size argument");

  if (want_breakpoint) {
    // Try to set the breakpoint.
    const Status error =
        m_debugged_process_up->SetBreakpoint(addr, size, want_hardware);
    if (error.Success())
      return SendOKResponse();
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_BREAKPOINTS));
    LLDB_LOG(log, "pid {0} failed to set breakpoint: {1}",
             m_debugged_process_up->GetID(), error);
    return SendErrorResponse(0x09);
  } else {
    // Try to set the watchpoint.
    const Status error = m_debugged_process_up->SetWatchpoint(
        addr, size, watch_flags, want_hardware);
    if (error.Success())
      return SendOKResponse();
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_WATCHPOINTS));
    LLDB_LOG(log, "pid {0} failed to set watchpoint: {1}",
             m_debugged_process_up->GetID(), error);
    return SendErrorResponse(0x09);
  }
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_z(StringExtractorGDBRemote &packet) {
  // Ensure we have a process.
  if (!m_debugged_process_up ||
      (m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID)) {
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
    LLDB_LOG(log, "failed, no process available");
    return SendErrorResponse(0x15);
  }

  // Parse out software or hardware breakpoint or watchpoint requested.
  packet.SetFilePos(strlen("z"));
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(
        packet, "Too short z packet, missing software/hardware specifier");

  bool want_breakpoint = true;
  bool want_hardware = false;

  const GDBStoppointType stoppoint_type =
      GDBStoppointType(packet.GetS32(eStoppointInvalid));
  switch (stoppoint_type) {
  case eBreakpointHardware:
    want_breakpoint = true;
    want_hardware = true;
    break;
  case eBreakpointSoftware:
    want_breakpoint = true;
    break;
  case eWatchpointWrite:
    want_breakpoint = false;
    break;
  case eWatchpointRead:
    want_breakpoint = false;
    break;
  case eWatchpointReadWrite:
    want_breakpoint = false;
    break;
  default:
    return SendIllFormedResponse(
        packet, "z packet had invalid software/hardware specifier");
  }

  if ((packet.GetBytesLeft() < 1) || packet.GetChar() != ',')
    return SendIllFormedResponse(
        packet, "Malformed z packet, expecting comma after stoppoint type");

  // Parse out the stoppoint address.
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(packet, "Too short z packet, missing address");
  const lldb::addr_t addr = packet.GetHexMaxU64(false, 0);

  if ((packet.GetBytesLeft() < 1) || packet.GetChar() != ',')
    return SendIllFormedResponse(
        packet, "Malformed z packet, expecting comma after address");

  /*
  // Parse out the stoppoint size (i.e. size hint for opcode size).
  const uint32_t size = packet.GetHexMaxU32 (false,
  std::numeric_limits<uint32_t>::max ());
  if (size == std::numeric_limits<uint32_t>::max ())
      return SendIllFormedResponse(packet, "Malformed z packet, failed to parse
  size argument");
  */

  if (want_breakpoint) {
    // Try to clear the breakpoint.
    const Status error =
        m_debugged_process_up->RemoveBreakpoint(addr, want_hardware);
    if (error.Success())
      return SendOKResponse();
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_BREAKPOINTS));
    LLDB_LOG(log, "pid {0} failed to remove breakpoint: {1}",
             m_debugged_process_up->GetID(), error);
    return SendErrorResponse(0x09);
  } else {
    // Try to clear the watchpoint.
    const Status error = m_debugged_process_up->RemoveWatchpoint(addr);
    if (error.Success())
      return SendOKResponse();
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_WATCHPOINTS));
    LLDB_LOG(log, "pid {0} failed to remove watchpoint: {1}",
             m_debugged_process_up->GetID(), error);
    return SendErrorResponse(0x09);
  }
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_s(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_THREAD));

  // Ensure we have a process.
  if (!m_debugged_process_up ||
      (m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID)) {
    LLDB_LOGF(
        log,
        "GDBRemoteCommunicationServerLLGS::%s failed, no process available",
        __FUNCTION__);
    return SendErrorResponse(0x32);
  }

  // We first try to use a continue thread id.  If any one or any all set, use
  // the current thread. Bail out if we don't have a thread id.
  lldb::tid_t tid = GetContinueThreadID();
  if (tid == 0 || tid == LLDB_INVALID_THREAD_ID)
    tid = GetCurrentThreadID();
  if (tid == LLDB_INVALID_THREAD_ID)
    return SendErrorResponse(0x33);

  // Double check that we have such a thread.
  // TODO investigate: on MacOSX we might need to do an UpdateThreads () here.
  NativeThreadProtocol *thread = m_debugged_process_up->GetThreadByID(tid);
  if (!thread)
    return SendErrorResponse(0x33);

  // Create the step action for the given thread.
  ResumeAction action = {tid, eStateStepping, LLDB_INVALID_SIGNAL_NUMBER};

  // Setup the actions list.
  ResumeActionList actions;
  actions.Append(action);

  // All other threads stop while we're single stepping a thread.
  actions.SetDefaultThreadActionIfNeeded(eStateStopped, 0);
  Status error = m_debugged_process_up->Resume(actions);
  if (error.Fail()) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s pid %" PRIu64
              " tid %" PRIu64 " Resume() failed with error: %s",
              __FUNCTION__, m_debugged_process_up->GetID(), tid,
              error.AsCString());
    return SendErrorResponse(0x49);
  }

  // No response here - the stop or exit will come from the resulting action.
  return PacketResult::Success;
}

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
GDBRemoteCommunicationServerLLGS::BuildTargetXml() {
  // Ensure we have a thread.
  NativeThreadProtocol *thread = m_debugged_process_up->GetThreadAtIndex(0);
  if (!thread)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No thread available");

  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_THREAD));
  // Get the register context for the first thread.
  NativeRegisterContext &reg_context = thread->GetRegisterContext();

  StreamString response;

  response.Printf("<?xml version=\"1.0\"?>");
  response.Printf("<target version=\"1.0\">");

  response.Printf("<architecture>%s</architecture>",
                  m_debugged_process_up->GetArchitecture()
                      .GetTriple()
                      .getArchName()
                      .str()
                      .c_str());

  response.Printf("<feature>");

  const int registers_count = reg_context.GetUserRegisterCount();
  for (int reg_index = 0; reg_index < registers_count; reg_index++) {
    const RegisterInfo *reg_info =
        reg_context.GetRegisterInfoAtIndex(reg_index);

    if (!reg_info) {
      LLDB_LOGF(log,
                "%s failed to get register info for register index %" PRIu32,
                "target.xml", reg_index);
      continue;
    }

    response.Printf("<reg name=\"%s\" bitsize=\"%" PRIu32 "\" offset=\"%" PRIu32
                    "\" regnum=\"%d\" ",
                    reg_info->name, reg_info->byte_size * 8,
                    reg_info->byte_offset, reg_index);

    if (reg_info->alt_name && reg_info->alt_name[0])
      response.Printf("altname=\"%s\" ", reg_info->alt_name);

    llvm::StringRef encoding = GetEncodingNameOrEmpty(*reg_info);
    if (!encoding.empty())
      response << "encoding=\"" << encoding << "\" ";

    llvm::StringRef format = GetFormatNameOrEmpty(*reg_info);
    if (!format.empty())
      response << "format=\"" << format << "\" ";

    const char *const register_set_name =
        reg_context.GetRegisterSetNameForRegisterAtIndex(reg_index);
    if (register_set_name)
      response << "group=\"" << register_set_name << "\" ";

    if (reg_info->kinds[RegisterKind::eRegisterKindEHFrame] !=
        LLDB_INVALID_REGNUM)
      response.Printf("ehframe_regnum=\"%" PRIu32 "\" ",
                      reg_info->kinds[RegisterKind::eRegisterKindEHFrame]);

    if (reg_info->kinds[RegisterKind::eRegisterKindDWARF] !=
        LLDB_INVALID_REGNUM)
      response.Printf("dwarf_regnum=\"%" PRIu32 "\" ",
                      reg_info->kinds[RegisterKind::eRegisterKindDWARF]);

    llvm::StringRef kind_generic = GetKindGenericOrEmpty(*reg_info);
    if (!kind_generic.empty())
      response << "generic=\"" << kind_generic << "\" ";

    if (reg_info->value_regs &&
        reg_info->value_regs[0] != LLDB_INVALID_REGNUM) {
      response.PutCString("value_regnums=\"");
      CollectRegNums(reg_info->value_regs, response, false);
      response.Printf("\" ");
    }

    if (reg_info->invalidate_regs && reg_info->invalidate_regs[0]) {
      response.PutCString("invalidate_regnums=\"");
      CollectRegNums(reg_info->invalidate_regs, response, false);
      response.Printf("\" ");
    }

    if (reg_info->dynamic_size_dwarf_expr_bytes) {
      const size_t dwarf_opcode_len = reg_info->dynamic_size_dwarf_len;
      response.PutCString("dynamic_size_dwarf_expr_bytes=\"");
      for (uint32_t i = 0; i < dwarf_opcode_len; ++i)
        response.PutHex8(reg_info->dynamic_size_dwarf_expr_bytes[i]);
      response.Printf("\" ");
    }

    response.Printf("/>");
  }

  response.Printf("</feature>");
  response.Printf("</target>");
  return MemoryBuffer::getMemBufferCopy(response.GetString(), "target.xml");
}

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
GDBRemoteCommunicationServerLLGS::ReadXferObject(llvm::StringRef object,
                                                 llvm::StringRef annex) {
  // Make sure we have a valid process.
  if (!m_debugged_process_up ||
      (m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID)) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No process available");
  }

  if (object == "auxv") {
    // Grab the auxv data.
    auto buffer_or_error = m_debugged_process_up->GetAuxvData();
    if (!buffer_or_error)
      return llvm::errorCodeToError(buffer_or_error.getError());
    return std::move(*buffer_or_error);
  }

  if (object == "libraries-svr4") {
    auto library_list = m_debugged_process_up->GetLoadedSVR4Libraries();
    if (!library_list)
      return library_list.takeError();

    StreamString response;
    response.Printf("<library-list-svr4 version=\"1.0\">");
    for (auto const &library : *library_list) {
      response.Printf("<library name=\"%s\" ",
                      XMLEncodeAttributeValue(library.name.c_str()).c_str());
      response.Printf("lm=\"0x%" PRIx64 "\" ", library.link_map);
      response.Printf("l_addr=\"0x%" PRIx64 "\" ", library.base_addr);
      response.Printf("l_ld=\"0x%" PRIx64 "\" />", library.ld_addr);
    }
    response.Printf("</library-list-svr4>");
    return MemoryBuffer::getMemBufferCopy(response.GetString(), __FUNCTION__);
  }

  if (object == "features" && annex == "target.xml")
    return BuildTargetXml();

  return llvm::make_error<UnimplementedError>();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qXfer(
    StringExtractorGDBRemote &packet) {
  SmallVector<StringRef, 5> fields;
  // The packet format is "qXfer:<object>:<action>:<annex>:offset,length"
  StringRef(packet.GetStringRef()).split(fields, ':', 4);
  if (fields.size() != 5)
    return SendIllFormedResponse(packet, "malformed qXfer packet");
  StringRef &xfer_object = fields[1];
  StringRef &xfer_action = fields[2];
  StringRef &xfer_annex = fields[3];
  StringExtractor offset_data(fields[4]);
  if (xfer_action != "read")
    return SendUnimplementedResponse("qXfer action not supported");
  // Parse offset.
  const uint64_t xfer_offset =
      offset_data.GetHexMaxU64(false, std::numeric_limits<uint64_t>::max());
  if (xfer_offset == std::numeric_limits<uint64_t>::max())
    return SendIllFormedResponse(packet, "qXfer packet missing offset");
  // Parse out comma.
  if (offset_data.GetChar() != ',')
    return SendIllFormedResponse(packet,
                                 "qXfer packet missing comma after offset");
  // Parse out the length.
  const uint64_t xfer_length =
      offset_data.GetHexMaxU64(false, std::numeric_limits<uint64_t>::max());
  if (xfer_length == std::numeric_limits<uint64_t>::max())
    return SendIllFormedResponse(packet, "qXfer packet missing length");

  // Get a previously constructed buffer if it exists or create it now.
  std::string buffer_key = (xfer_object + xfer_action + xfer_annex).str();
  auto buffer_it = m_xfer_buffer_map.find(buffer_key);
  if (buffer_it == m_xfer_buffer_map.end()) {
    auto buffer_up = ReadXferObject(xfer_object, xfer_annex);
    if (!buffer_up)
      return SendErrorResponse(buffer_up.takeError());
    buffer_it = m_xfer_buffer_map
                    .insert(std::make_pair(buffer_key, std::move(*buffer_up)))
                    .first;
  }

  // Send back the response
  StreamGDBRemote response;
  bool done_with_buffer = false;
  llvm::StringRef buffer = buffer_it->second->getBuffer();
  if (xfer_offset >= buffer.size()) {
    // We have nothing left to send.  Mark the buffer as complete.
    response.PutChar('l');
    done_with_buffer = true;
  } else {
    // Figure out how many bytes are available starting at the given offset.
    buffer = buffer.drop_front(xfer_offset);
    // Mark the response type according to whether we're reading the remainder
    // of the data.
    if (xfer_length >= buffer.size()) {
      // There will be nothing left to read after this
      response.PutChar('l');
      done_with_buffer = true;
    } else {
      // There will still be bytes to read after this request.
      response.PutChar('m');
      buffer = buffer.take_front(xfer_length);
    }
    // Now write the data in encoded binary form.
    response.PutEscapedBytes(buffer.data(), buffer.size());
  }

  if (done_with_buffer)
    m_xfer_buffer_map.erase(buffer_it);

  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_QSaveRegisterState(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

  // Move past packet name.
  packet.SetFilePos(strlen("QSaveRegisterState"));

  // Get the thread to use.
  NativeThreadProtocol *thread = GetThreadFromSuffix(packet);
  if (!thread) {
    if (m_thread_suffix_supported)
      return SendIllFormedResponse(
          packet, "No thread specified in QSaveRegisterState packet");
    else
      return SendIllFormedResponse(packet,
                                   "No thread was is set with the Hg packet");
  }

  // Grab the register context for the thread.
  NativeRegisterContext& reg_context = thread->GetRegisterContext();

  // Save registers to a buffer.
  DataBufferSP register_data_sp;
  Status error = reg_context.ReadAllRegisterValues(register_data_sp);
  if (error.Fail()) {
    LLDB_LOG(log, "pid {0} failed to save all register values: {1}",
             m_debugged_process_up->GetID(), error);
    return SendErrorResponse(0x75);
  }

  // Allocate a new save id.
  const uint32_t save_id = GetNextSavedRegistersID();
  assert((m_saved_registers_map.find(save_id) == m_saved_registers_map.end()) &&
         "GetNextRegisterSaveID() returned an existing register save id");

  // Save the register data buffer under the save id.
  {
    std::lock_guard<std::mutex> guard(m_saved_registers_mutex);
    m_saved_registers_map[save_id] = register_data_sp;
  }

  // Write the response.
  StreamGDBRemote response;
  response.Printf("%" PRIu32, save_id);
  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_QRestoreRegisterState(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

  // Parse out save id.
  packet.SetFilePos(strlen("QRestoreRegisterState:"));
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(
        packet, "QRestoreRegisterState packet missing register save id");

  const uint32_t save_id = packet.GetU32(0);
  if (save_id == 0) {
    LLDB_LOG(log, "QRestoreRegisterState packet has malformed save id, "
                  "expecting decimal uint32_t");
    return SendErrorResponse(0x76);
  }

  // Get the thread to use.
  NativeThreadProtocol *thread = GetThreadFromSuffix(packet);
  if (!thread) {
    if (m_thread_suffix_supported)
      return SendIllFormedResponse(
          packet, "No thread specified in QRestoreRegisterState packet");
    else
      return SendIllFormedResponse(packet,
                                   "No thread was is set with the Hg packet");
  }

  // Grab the register context for the thread.
  NativeRegisterContext &reg_context = thread->GetRegisterContext();

  // Retrieve register state buffer, then remove from the list.
  DataBufferSP register_data_sp;
  {
    std::lock_guard<std::mutex> guard(m_saved_registers_mutex);

    // Find the register set buffer for the given save id.
    auto it = m_saved_registers_map.find(save_id);
    if (it == m_saved_registers_map.end()) {
      LLDB_LOG(log,
               "pid {0} does not have a register set save buffer for id {1}",
               m_debugged_process_up->GetID(), save_id);
      return SendErrorResponse(0x77);
    }
    register_data_sp = it->second;

    // Remove it from the map.
    m_saved_registers_map.erase(it);
  }

  Status error = reg_context.WriteAllRegisterValues(register_data_sp);
  if (error.Fail()) {
    LLDB_LOG(log, "pid {0} failed to restore all register values: {1}",
             m_debugged_process_up->GetID(), error);
    return SendErrorResponse(0x77);
  }

  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_vAttach(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  // Consume the ';' after vAttach.
  packet.SetFilePos(strlen("vAttach"));
  if (!packet.GetBytesLeft() || packet.GetChar() != ';')
    return SendIllFormedResponse(packet, "vAttach missing expected ';'");

  // Grab the PID to which we will attach (assume hex encoding).
  lldb::pid_t pid = packet.GetU32(LLDB_INVALID_PROCESS_ID, 16);
  if (pid == LLDB_INVALID_PROCESS_ID)
    return SendIllFormedResponse(packet,
                                 "vAttach failed to parse the process id");

  // Attempt to attach.
  LLDB_LOGF(log,
            "GDBRemoteCommunicationServerLLGS::%s attempting to attach to "
            "pid %" PRIu64,
            __FUNCTION__, pid);

  Status error = AttachToProcess(pid);

  if (error.Fail()) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed to attach to "
              "pid %" PRIu64 ": %s\n",
              __FUNCTION__, pid, error.AsCString());
    return SendErrorResponse(error);
  }

  // Notify we attached by sending a stop packet.
  return SendStopReasonForState(m_debugged_process_up->GetState());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_D(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  StopSTDIOForwarding();

  // Fail if we don't have a current process.
  if (!m_debugged_process_up ||
      (m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID)) {
    LLDB_LOGF(
        log,
        "GDBRemoteCommunicationServerLLGS::%s failed, no process available",
        __FUNCTION__);
    return SendErrorResponse(0x15);
  }

  lldb::pid_t pid = LLDB_INVALID_PROCESS_ID;

  // Consume the ';' after D.
  packet.SetFilePos(1);
  if (packet.GetBytesLeft()) {
    if (packet.GetChar() != ';')
      return SendIllFormedResponse(packet, "D missing expected ';'");

    // Grab the PID from which we will detach (assume hex encoding).
    pid = packet.GetU32(LLDB_INVALID_PROCESS_ID, 16);
    if (pid == LLDB_INVALID_PROCESS_ID)
      return SendIllFormedResponse(packet, "D failed to parse the process id");
  }

  if (pid != LLDB_INVALID_PROCESS_ID && m_debugged_process_up->GetID() != pid) {
    return SendIllFormedResponse(packet, "Invalid pid");
  }

  const Status error = m_debugged_process_up->Detach();
  if (error.Fail()) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed to detach from "
              "pid %" PRIu64 ": %s\n",
              __FUNCTION__, m_debugged_process_up->GetID(), error.AsCString());
    return SendErrorResponse(0x01);
  }

  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qThreadStopInfo(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

  packet.SetFilePos(strlen("qThreadStopInfo"));
  const lldb::tid_t tid = packet.GetHexMaxU32(false, LLDB_INVALID_THREAD_ID);
  if (tid == LLDB_INVALID_THREAD_ID) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed, could not "
              "parse thread id from request \"%s\"",
              __FUNCTION__, packet.GetStringRef().data());
    return SendErrorResponse(0x15);
  }
  return SendStopReplyPacketForThread(tid);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_jThreadsInfo(
    StringExtractorGDBRemote &) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_THREAD));

  // Ensure we have a debugged process.
  if (!m_debugged_process_up ||
      (m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID))
    return SendErrorResponse(50);
  LLDB_LOG(log, "preparing packet for pid {0}", m_debugged_process_up->GetID());

  StreamString response;
  const bool threads_with_valid_stop_info_only = false;
  llvm::Expected<json::Value> threads_info = GetJSONThreadsInfo(
      *m_debugged_process_up, threads_with_valid_stop_info_only);
  if (!threads_info) {
    LLDB_LOG_ERROR(log, threads_info.takeError(),
                   "failed to prepare a packet for pid {1}: {0}",
                   m_debugged_process_up->GetID());
    return SendErrorResponse(52);
  }

  response.AsRawOstream() << *threads_info;
  StreamGDBRemote escaped_response;
  escaped_response.PutEscapedBytes(response.GetData(), response.GetSize());
  return SendPacketNoLock(escaped_response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qWatchpointSupportInfo(
    StringExtractorGDBRemote &packet) {
  // Fail if we don't have a current process.
  if (!m_debugged_process_up ||
      m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID)
    return SendErrorResponse(68);

  packet.SetFilePos(strlen("qWatchpointSupportInfo"));
  if (packet.GetBytesLeft() == 0)
    return SendOKResponse();
  if (packet.GetChar() != ':')
    return SendErrorResponse(67);

  auto hw_debug_cap = m_debugged_process_up->GetHardwareDebugSupportInfo();

  StreamGDBRemote response;
  if (hw_debug_cap == llvm::None)
    response.Printf("num:0;");
  else
    response.Printf("num:%d;", hw_debug_cap->second);

  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qFileLoadAddress(
    StringExtractorGDBRemote &packet) {
  // Fail if we don't have a current process.
  if (!m_debugged_process_up ||
      m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID)
    return SendErrorResponse(67);

  packet.SetFilePos(strlen("qFileLoadAddress:"));
  if (packet.GetBytesLeft() == 0)
    return SendErrorResponse(68);

  std::string file_name;
  packet.GetHexByteString(file_name);

  lldb::addr_t file_load_address = LLDB_INVALID_ADDRESS;
  Status error =
      m_debugged_process_up->GetFileLoadAddress(file_name, file_load_address);
  if (error.Fail())
    return SendErrorResponse(69);

  if (file_load_address == LLDB_INVALID_ADDRESS)
    return SendErrorResponse(1); // File not loaded

  StreamGDBRemote response;
  response.PutHex64(file_load_address);
  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_QPassSignals(
    StringExtractorGDBRemote &packet) {
  std::vector<int> signals;
  packet.SetFilePos(strlen("QPassSignals:"));

  // Read sequence of hex signal numbers divided by a semicolon and optionally
  // spaces.
  while (packet.GetBytesLeft() > 0) {
    int signal = packet.GetS32(-1, 16);
    if (signal < 0)
      return SendIllFormedResponse(packet, "Failed to parse signal number.");
    signals.push_back(signal);

    packet.SkipSpaces();
    char separator = packet.GetChar();
    if (separator == '\0')
      break; // End of string
    if (separator != ';')
      return SendIllFormedResponse(packet, "Invalid separator,"
                                            " expected semicolon.");
  }

  // Fail if we don't have a current process.
  if (!m_debugged_process_up)
    return SendErrorResponse(68);

  Status error = m_debugged_process_up->IgnoreSignals(signals);
  if (error.Fail())
    return SendErrorResponse(69);

  return SendOKResponse();
}

void GDBRemoteCommunicationServerLLGS::MaybeCloseInferiorTerminalConnection() {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  // Tell the stdio connection to shut down.
  if (m_stdio_communication.IsConnected()) {
    auto connection = m_stdio_communication.GetConnection();
    if (connection) {
      Status error;
      connection->Disconnect(&error);

      if (error.Success()) {
        LLDB_LOGF(log,
                  "GDBRemoteCommunicationServerLLGS::%s disconnect process "
                  "terminal stdio - SUCCESS",
                  __FUNCTION__);
      } else {
        LLDB_LOGF(log,
                  "GDBRemoteCommunicationServerLLGS::%s disconnect process "
                  "terminal stdio - FAIL: %s",
                  __FUNCTION__, error.AsCString());
      }
    }
  }
}

NativeThreadProtocol *GDBRemoteCommunicationServerLLGS::GetThreadFromSuffix(
    StringExtractorGDBRemote &packet) {
  // We have no thread if we don't have a process.
  if (!m_debugged_process_up ||
      m_debugged_process_up->GetID() == LLDB_INVALID_PROCESS_ID)
    return nullptr;

  // If the client hasn't asked for thread suffix support, there will not be a
  // thread suffix. Use the current thread in that case.
  if (!m_thread_suffix_supported) {
    const lldb::tid_t current_tid = GetCurrentThreadID();
    if (current_tid == LLDB_INVALID_THREAD_ID)
      return nullptr;
    else if (current_tid == 0) {
      // Pick a thread.
      return m_debugged_process_up->GetThreadAtIndex(0);
    } else
      return m_debugged_process_up->GetThreadByID(current_tid);
  }

  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

  // Parse out the ';'.
  if (packet.GetBytesLeft() < 1 || packet.GetChar() != ';') {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s gdb-remote parse "
              "error: expected ';' prior to start of thread suffix: packet "
              "contents = '%s'",
              __FUNCTION__, packet.GetStringRef().data());
    return nullptr;
  }

  if (!packet.GetBytesLeft())
    return nullptr;

  // Parse out thread: portion.
  if (strncmp(packet.Peek(), "thread:", strlen("thread:")) != 0) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s gdb-remote parse "
              "error: expected 'thread:' but not found, packet contents = "
              "'%s'",
              __FUNCTION__, packet.GetStringRef().data());
    return nullptr;
  }
  packet.SetFilePos(packet.GetFilePos() + strlen("thread:"));
  const lldb::tid_t tid = packet.GetHexMaxU64(false, 0);
  if (tid != 0)
    return m_debugged_process_up->GetThreadByID(tid);

  return nullptr;
}

lldb::tid_t GDBRemoteCommunicationServerLLGS::GetCurrentThreadID() const {
  if (m_current_tid == 0 || m_current_tid == LLDB_INVALID_THREAD_ID) {
    // Use whatever the debug process says is the current thread id since the
    // protocol either didn't specify or specified we want any/all threads
    // marked as the current thread.
    if (!m_debugged_process_up)
      return LLDB_INVALID_THREAD_ID;
    return m_debugged_process_up->GetCurrentThreadID();
  }
  // Use the specific current thread id set by the gdb remote protocol.
  return m_current_tid;
}

uint32_t GDBRemoteCommunicationServerLLGS::GetNextSavedRegistersID() {
  std::lock_guard<std::mutex> guard(m_saved_registers_mutex);
  return m_next_saved_registers_id++;
}

void GDBRemoteCommunicationServerLLGS::ClearProcessSpecificData() {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  LLDB_LOG(log, "clearing {0} xfer buffers", m_xfer_buffer_map.size());
  m_xfer_buffer_map.clear();
}

FileSpec
GDBRemoteCommunicationServerLLGS::FindModuleFile(const std::string &module_path,
                                                 const ArchSpec &arch) {
  if (m_debugged_process_up) {
    FileSpec file_spec;
    if (m_debugged_process_up
            ->GetLoadedModuleFileSpec(module_path.c_str(), file_spec)
            .Success()) {
      if (FileSystem::Instance().Exists(file_spec))
        return file_spec;
    }
  }

  return GDBRemoteCommunicationServerCommon::FindModuleFile(module_path, arch);
}

std::string GDBRemoteCommunicationServerLLGS::XMLEncodeAttributeValue(
    llvm::StringRef value) {
  std::string result;
  for (const char &c : value) {
    switch (c) {
    case '\'':
      result += "&apos;";
      break;
    case '"':
      result += "&quot;";
      break;
    case '<':
      result += "&lt;";
      break;
    case '>':
      result += "&gt;";
      break;
    default:
      result += c;
      break;
    }
  }
  return result;
}
