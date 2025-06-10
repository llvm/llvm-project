//===-- LLDBServerPluginAMDGPU.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLDBServerPluginAMDGPU.h"
#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServerLLGS.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "ProcessAMDGPU.h"
#include "ThreadAMDGPU.h"
#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Host/posix/ConnectionFileDescriptorPosix.h"
#include "llvm/Support/Error.h"

#include <cinttypes>
#include <sys/ptrace.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <thread>
#include <unistd.h>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;

static amd_dbgapi_status_t amd_dbgapi_client_process_get_info_callback(
    amd_dbgapi_client_process_id_t client_process_id,
    amd_dbgapi_client_process_info_t query, size_t value_size, void *value) {
  LLDBServerPluginAMDGPU *debugger =
      reinterpret_cast<LLDBServerPluginAMDGPU *>(client_process_id);
  lldb::pid_t pid = debugger->GetNativeProcess()->GetID();
  LLDB_LOGF(GetLog(GDBRLog::Plugin),
            "amd_dbgapi_client_process_get_info_callback callback, with query "
            "%d, pid %lu",
            query, (unsigned long)pid);
  switch (query) {
  case AMD_DBGAPI_CLIENT_PROCESS_INFO_OS_PID: {
    if (value_size != sizeof(amd_dbgapi_os_process_id_t))
      return AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY;
    *static_cast<amd_dbgapi_os_process_id_t *>(value) = pid;
    return AMD_DBGAPI_STATUS_SUCCESS;
  }
  case AMD_DBGAPI_CLIENT_PROCESS_INFO_CORE_STATE: {
    return AMD_DBGAPI_STATUS_SUCCESS;
  }
  }
  return AMD_DBGAPI_STATUS_SUCCESS;
}

static amd_dbgapi_status_t amd_dbgapi_insert_breakpoint_callback(
    amd_dbgapi_client_process_id_t client_process_id,
    amd_dbgapi_global_address_t address,
    amd_dbgapi_breakpoint_id_t breakpoint_id) {
  LLDB_LOGF(GetLog(GDBRLog::Plugin),
            "insert_breakpoint callback at address: 0x%" PRIx64, address);
  LLDBServerPluginAMDGPU *debugger =
      reinterpret_cast<LLDBServerPluginAMDGPU *>(client_process_id);
  debugger->GetNativeProcess()->Halt();
  LLDB_LOGF(GetLog(GDBRLog::Plugin), "insert_breakpoint callback success");
  LLDBServerPluginAMDGPU::GPUInternalBreakpoinInfo bp_info;
  bp_info.addr = address;
  bp_info.breakpoind_id = breakpoint_id;
  debugger->m_gpu_internal_bp.emplace(std::move(bp_info));
  debugger->m_wait_for_gpu_internal_bp_stop = true;
  return AMD_DBGAPI_STATUS_SUCCESS;
}

/* remove_breakpoint callback.  */

static amd_dbgapi_status_t amd_dbgapi_remove_breakpoint_callback(
    amd_dbgapi_client_process_id_t client_process_id,
    amd_dbgapi_breakpoint_id_t breakpoint_id) {
  LLDB_LOGF(GetLog(GDBRLog::Plugin), "remove_breakpoint callback for %" PRIu64,
            breakpoint_id.handle);
  return AMD_DBGAPI_STATUS_SUCCESS;
}

/* xfer_global_memory callback.  */

static amd_dbgapi_status_t amd_dbgapi_xfer_global_memory_callback(
    amd_dbgapi_client_process_id_t client_process_id,
    amd_dbgapi_global_address_t global_address, amd_dbgapi_size_t *value_size,
    void *read_buffer, const void *write_buffer) {
  LLDB_LOGF(GetLog(GDBRLog::Plugin), "xfer_global_memory callback");

  return AMD_DBGAPI_STATUS_SUCCESS;
}

static void amd_dbgapi_log_message_callback(amd_dbgapi_log_level_t level,
                                            const char *message) {
  LLDB_LOGF(GetLog(GDBRLog::Plugin), "ROCdbgapi [%d]: %s", level, message);
}

static amd_dbgapi_callbacks_t s_dbgapi_callbacks = {
    malloc,
    free,
    amd_dbgapi_client_process_get_info_callback,
    amd_dbgapi_insert_breakpoint_callback,
    amd_dbgapi_remove_breakpoint_callback,
    amd_dbgapi_xfer_global_memory_callback,
    amd_dbgapi_log_message_callback,
};

LLDBServerPluginAMDGPU::LLDBServerPluginAMDGPU(
    LLDBServerPlugin::GDBServer &native_process, MainLoop &main_loop)
    : LLDBServerPlugin(native_process, main_loop) {
  m_process_manager_up.reset(new ProcessManagerAMDGPU(main_loop));
  m_gdb_server.reset(new GDBRemoteCommunicationServerLLGS(
      m_main_loop, *m_process_manager_up, "amd-gpu.server"));
}

LLDBServerPluginAMDGPU::~LLDBServerPluginAMDGPU() { CloseFDs(); }

llvm::StringRef LLDBServerPluginAMDGPU::GetPluginName() { return "amd-gpu"; }

void LLDBServerPluginAMDGPU::CloseFDs() {
  if (m_fds[0] != -1) {
    close(m_fds[0]);
    m_fds[0] = -1;
  }
  if (m_fds[1] != -1) {
    close(m_fds[1]);
    m_fds[1] = -1;
  }
}

int LLDBServerPluginAMDGPU::GetEventFileDescriptorAtIndex(size_t idx) {
  if (idx != 0)
    return -1;
  if (m_fds[0] == -1) {
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, m_fds) == -1) {
      m_fds[0] = -1;
      m_fds[1] = -1;
    }
  }
  return m_fds[0];
}

bool LLDBServerPluginAMDGPU::initRocm() {
  // Initialize AMD Debug API with callbacks
  amd_dbgapi_status_t status = amd_dbgapi_initialize(&s_dbgapi_callbacks);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "Failed to initialize AMD debug API");
    exit(-1);
  }

  // Attach to the process with AMD Debug API
  status = amd_dbgapi_process_attach(
      reinterpret_cast<amd_dbgapi_client_process_id_t>(
          this), // Use pid_ as client_process_id
      &m_gpu_pid);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(GetLog(GDBRLog::Plugin),
              "Failed to attach to process with AMD debug API: %d", status);
    amd_dbgapi_finalize();
    return false;
  }

  // Get the process notifier
  status =
      amd_dbgapi_process_get_info(m_gpu_pid, AMD_DBGAPI_PROCESS_INFO_NOTIFIER,
                                  sizeof(m_notifier_fd), &m_notifier_fd);

  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "Failed to get process notifier: %d",
              status);
    amd_dbgapi_process_detach(m_gpu_pid);
    amd_dbgapi_finalize();
    return false;
  }

  // event_handler_.addNotifierFd(m_notifier_fd);
  LLDB_LOGF(GetLog(GDBRLog::Plugin), "Process notifier fd: %d", m_notifier_fd);

  amd_dbgapi_event_id_t eventId;
  amd_dbgapi_event_kind_t eventKind;
  // Process all pending events
  if (amd_dbgapi_process_next_pending_event(m_gpu_pid, &eventId, &eventKind) ==
      AMD_DBGAPI_STATUS_SUCCESS) {
    GetGPUProcess()->handleDebugEvent(eventId, eventKind);

    // Mark event as processed
    amd_dbgapi_event_processed(eventId);
  }

  amd_dbgapi_architecture_id_t architecture_id;
  // TODO: do not hardcode the device id
  status = amd_dbgapi_get_architecture(0x04C, &architecture_id);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    // Handle error
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "amd_dbgapi_get_architecture failed");
    return false;
  }
  m_architecture_id = architecture_id;

  return true;
}

bool LLDBServerPluginAMDGPU::processGPUEvent() {
  LLDB_LOGF(GetLog(GDBRLog::Plugin), "processGPUEvent");
  char buf[256];
  ssize_t bytesRead = 0;
  bool result = false;
  do {
    do {
      bytesRead = read(m_notifier_fd, buf, sizeof(buf));
    } while (bytesRead <= 0);

    auto *process = GetGPUProcess();
    process->m_wave_ids.clear();
    amd_dbgapi_status_t status = amd_dbgapi_process_set_progress(
        m_gpu_pid, AMD_DBGAPI_PROGRESS_NO_FORWARD);
    assert(status == AMD_DBGAPI_STATUS_SUCCESS);
    process_event_queue(AMD_DBGAPI_EVENT_KIND_NONE);
    if (process->m_gpu_state == ProcessAMDGPU::State::GPUStopped) {
      for (auto wave_id : process->m_wave_ids) {
        process->AddThread(wave_id);
      }
      process->Halt();
    }
    status =
        amd_dbgapi_process_set_progress(m_gpu_pid, AMD_DBGAPI_PROGRESS_NORMAL);
    assert(status == AMD_DBGAPI_STATUS_SUCCESS);
    break;
  } while (true);
  return result;
}

bool LLDBServerPluginAMDGPU::HandleEventFileDescriptorEvent(int fd) {
  return processGPUEvent();
}

void LLDBServerPluginAMDGPU::AcceptAndMainLoopThread(
    std::unique_ptr<TCPSocket> listen_socket_up) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOGF(log, "%s spawned", __PRETTY_FUNCTION__);
  Socket *socket = nullptr;
  Status error = listen_socket_up->Accept(std::chrono::seconds(30), socket);
  // Scope for lock guard.
  {
    // Protect access to m_is_listening and m_is_connected.
    std::lock_guard<std::mutex> guard(m_connect_mutex);
    m_is_listening = false;
    if (error.Fail()) {
      LLDB_LOGF(log, "%s error returned from Accept(): %s", __PRETTY_FUNCTION__,
                error.AsCString());
      return;
    }
    m_is_connected = true;
  }

  LLDB_LOGF(log, "%s initializing connection", __PRETTY_FUNCTION__);
  std::unique_ptr<Connection> connection_up(
      new ConnectionFileDescriptor(socket));
  m_gdb_server->InitializeConnection(std::move(connection_up));
  LLDB_LOGF(log, "%s running main loop", __PRETTY_FUNCTION__);
  m_main_loop_status = m_main_loop.Run();
  LLDB_LOGF(log, "%s main loop exited!", __PRETTY_FUNCTION__);
  if (m_main_loop_status.Fail()) {
    LLDB_LOGF(log, "%s main loop exited with an error: %s", __PRETTY_FUNCTION__,
              m_main_loop_status.AsCString());
  }
  // Protect access to m_is_connected.
  std::lock_guard<std::mutex> guard(m_connect_mutex);
  m_is_connected = false;
}

std::optional<GPUPluginConnectionInfo>
LLDBServerPluginAMDGPU::CreateConnection() {
  std::lock_guard<std::mutex> guard(m_connect_mutex);
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOGF(log, "%s called", __PRETTY_FUNCTION__);
  if (m_is_connected) {
    LLDB_LOGF(log, "%s already connected", __PRETTY_FUNCTION__);
    return std::nullopt;
  }
  if (m_is_listening) {
    LLDB_LOGF(log, "%s already listening", __PRETTY_FUNCTION__);
    return std::nullopt;
  }
  m_is_listening = true;
  LLDB_LOGF(log, "%s trying to listen on port 0", __PRETTY_FUNCTION__);
  llvm::Expected<std::unique_ptr<TCPSocket>> sock =
      Socket::TcpListen("localhost:0", 5);
  if (sock) {
    GPUPluginConnectionInfo connection_info;
    const uint16_t listen_port = (*sock)->GetLocalPortNumber();
    connection_info.connect_url =
        llvm::formatv("connect://localhost:{}", listen_port);
    LLDB_LOGF(log, "%s listening to %u", __PRETTY_FUNCTION__, listen_port);
    // std::thread t(&LLDBServerPluginAMDGPU::AcceptAndMainLoopThread, this,
    //               std::move(*sock));
    // t.detach();

    // Store the socket in the member variable to keep it alive
    m_listen_socket = std::move(*sock);
    auto extra_args =
        llvm::formatv("gpu-url:connect://localhost:{};", listen_port);
    m_is_connected = false;
    llvm::Expected<std::vector<MainLoopBase::ReadHandleUP>> res =
        m_listen_socket->Accept(
            m_main_loop, [this](std::unique_ptr<Socket> socket) {
              Log *log = GetLog(GDBRLog::Plugin);
              LLDB_LOGF(log,
                        "LLDBServerPluginAMDGPU::AcceptAndMainLoopThread() "
                        "initializing connection");
              std::unique_ptr<Connection> connection_up(
                  new ConnectionFileDescriptor(socket.release()));
              this->m_gdb_server->InitializeConnection(
                  std::move(connection_up));
              m_is_connected = true;
            });
    if (res) {
      m_read_handles = std::move(*res);
    } else {
      LLDB_LOGF(
          log,
          "LLDBServerPluginAMDGPU::GetConnectionURL() failed to accept: %s",
          llvm::toString(res.takeError()).c_str());
    }

    return connection_info;
  } else {
    std::string error = llvm::toString(sock.takeError());
    LLDB_LOGF(log, "%s failed to listen to localhost:0: %s",
              __PRETTY_FUNCTION__, error.c_str());
  }
  m_is_listening = false;
  return std::nullopt;
}

std::optional<GPUActions> LLDBServerPluginAMDGPU::NativeProcessIsStopping() {
  Log *log = GetLog(GDBRLog::Plugin);
  if (!m_is_connected) {
    initRocm();
    ProcessManagerAMDGPU *manager =
        (ProcessManagerAMDGPU *)m_process_manager_up.get();
    manager->m_debugger = this;

    GPUActions actions;
    actions.plugin_name = GetPluginName();

    Status error;
    LLDBServerPluginAMDGPU *amdGPUPlugin = this;
    m_gpu_event_io_obj_sp = std::make_shared<GPUIOObject>(m_notifier_fd);
    m_gpu_event_read_up = m_main_loop.RegisterReadObject(
        m_gpu_event_io_obj_sp,
        [amdGPUPlugin](MainLoopBase &) {
          amdGPUPlugin->HandleEventFileDescriptorEvent(
              amdGPUPlugin->m_notifier_fd);
        },
        error);
    if (error.Fail()) {
      LLDB_LOGF(log, "LLDBServerPluginAMDGPU::NativeProcessIsStopping() failed "
                     "to RegisterReadObject");
      // TODO: how to report this error?
    } else {
      LLDB_LOGF(
          log,
          "LLDBServerPluginAMDGPU::LLDBServerPluginAMDGPU() faking launch...");
      ProcessLaunchInfo info;
      info.GetFlags().Set(eLaunchFlagStopAtEntry | eLaunchFlagDebug |
                          eLaunchFlagDisableASLR);
      Args args;
      args.AppendArgument("/pretend/path/to/amdgpu");
      args.AppendArgument("--option1");
      args.AppendArgument("--option2");
      args.AppendArgument("--option3");
      info.SetArguments(args, true);
      info.GetEnvironment() = Host::GetEnvironment();
      info.SetProcessID(m_gpu_pid.handle);
      m_gdb_server->SetLaunchInfo(info);
      Status error = m_gdb_server->LaunchProcess();
      if (error.Fail()) {
        LLDB_LOGF(log,
                  "LLDBServerPluginAMDGPU::LLDBServerPluginAMDGPU() failed to "
                  "launch: %s",
                  error.AsCString());
      } else {
        LLDB_LOGF(log, "LLDBServerPluginAMDGPU::LLDBServerPluginAMDGPU() "
                       "launched successfully");
      }
      actions.connect_info = CreateConnection();
    }
    return actions;
  } else {
    if (m_wait_for_gpu_internal_bp_stop && m_gpu_internal_bp.has_value()) {
      LLDB_LOGF(GetLog(GDBRLog::Plugin), "Please set gpu breakpoint at 0x%p",
                (void *)m_gpu_internal_bp->addr);
      GPUActions actions;
      actions.plugin_name = GetPluginName();

      GPUBreakpointByAddress bp_addr;
      bp_addr.load_address = m_gpu_internal_bp->addr;

      GPUBreakpointInfo bp;
      bp.identifier = "GPU loader breakpoint";
      bp.addr_info.emplace(bp_addr);

      std::vector<GPUBreakpointInfo> breakpoints;
      breakpoints.emplace_back(std::move(bp));

      actions.breakpoints = std::move(breakpoints);
      m_wait_for_gpu_internal_bp_stop = false;
      return actions;
    }
  }
  return std::nullopt;
}

bool LLDBServerPluginAMDGPU::HandleGPUInternalBreakpointHit(
    const GPUInternalBreakpoinInfo &bp, bool &has_new_libraries) {
  LLDB_LOGF(GetLog(GDBRLog::Plugin),
            "Hit GPU loader breakpoint at address: 0x%" PRIx64, bp.addr);
  has_new_libraries = false;
  amd_dbgapi_breakpoint_id_t breakpoint_id{bp.breakpoind_id};
  amd_dbgapi_breakpoint_action_t action;

  auto status = amd_dbgapi_report_breakpoint_hit(
      breakpoint_id, reinterpret_cast<amd_dbgapi_client_thread_id_t>(this),
      &action);

  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(GetLog(GDBRLog::Plugin),
              "amd_dbgapi_report_breakpoint_hit failed: %d", status);
    return false;
  }

  if (action == AMD_DBGAPI_BREAKPOINT_ACTION_RESUME) {
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "AMD_DBGAPI_BREAKPOINT_ACTION_RESUME");
    return true;
  } else if (action == AMD_DBGAPI_BREAKPOINT_ACTION_HALT) {
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "AMD_DBGAPI_BREAKPOINT_ACTION_HALT");

    amd_dbgapi_event_id_t resume_event_id =
        process_event_queue(AMD_DBGAPI_EVENT_KIND_BREAKPOINT_RESUME);
    amd_dbgapi_event_processed(resume_event_id);
    if (!GetGPUProcess()->GetGPUModules().empty()) {
      has_new_libraries = true;
    }
    return true;
  } else {
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "Unknown action: %d", action);
    return false;
  }
  return true;
}

amd_dbgapi_event_id_t LLDBServerPluginAMDGPU::process_event_queue(
    amd_dbgapi_event_kind_t until_event_kind) {
  while (true) {
    amd_dbgapi_event_id_t event_id;
    amd_dbgapi_event_kind_t event_kind;
    amd_dbgapi_status_t status = amd_dbgapi_process_next_pending_event(
        m_gpu_pid, &event_id, &event_kind);

    if (status != AMD_DBGAPI_STATUS_SUCCESS) {
      LLDB_LOGF(GetLog(GDBRLog::Plugin),
                "amd_dbgapi_process_next_pending_event failed: %d", status);
      return AMD_DBGAPI_EVENT_NONE;
    }

    if (event_kind != AMD_DBGAPI_EVENT_KIND_NONE)
      LLDB_LOGF(GetLog(GDBRLog::Plugin),
                "event_kind != AMD_DBGAPI_EVENT_KIND_NONE: %d", event_kind);

    if (event_id.handle == AMD_DBGAPI_EVENT_NONE.handle ||
        event_kind == until_event_kind)
      return event_id;

    GetGPUProcess()->handleDebugEvent(event_id, event_kind);
    amd_dbgapi_event_processed(event_id);
  }
  return AMD_DBGAPI_EVENT_NONE;
}

bool LLDBServerPluginAMDGPU::SetGPUBreakpoint(uint64_t addr,
                                              const uint8_t *bp_instruction,
                                              size_t size) {
  struct BreakpointInfo {
    uint64_t addr;
    std::vector<uint8_t> original_bytes;
    std::vector<uint8_t> breakpoint_instruction;
    std::optional<amd_dbgapi_breakpoint_id_t> gpu_breakpoint_id;
  };

  BreakpointInfo bp;
  bp.addr = addr;
  bp.breakpoint_instruction.assign(bp_instruction, bp_instruction + size);
  bp.original_bytes.resize(size);
  bp.gpu_breakpoint_id =
      std::nullopt; // No GPU breakpoint ID for ptrace version

  auto pid = GetNativeProcess()->GetID();
  // Read original bytes word by word
  std::vector<long> original_words;
  for (size_t i = 0; i < size; i += sizeof(long)) {
    long word = ptrace(PTRACE_PEEKDATA, pid, addr + i, nullptr);
    assert(word != -1 && errno == 0);

    original_words.push_back(word);
    // Copy bytes from the word into our original_bytes
    size_t bytes_to_copy = std::min(sizeof(long), size - i);
    memcpy(&bp.original_bytes[i], &word, bytes_to_copy);
  }

  // Write breakpoint instruction word by word
  for (size_t i = 0; i < size; i += sizeof(long)) {
    long word = original_words[i / sizeof(long)];
    size_t bytes_to_copy = std::min(sizeof(long), size - i);
    memcpy(&word, &bp_instruction[i], bytes_to_copy);

    auto ret = ptrace(PTRACE_POKEDATA, pid, addr + i, word);
    assert(ret != -1 && errno == 0);
  }
  return true;
}

bool LLDBServerPluginAMDGPU::CreateGPUBreakpoint(uint64_t addr) {
  // First get the architecture ID for this process
  amd_dbgapi_architecture_id_t arch_id;
  amd_dbgapi_status_t status = amd_dbgapi_get_architecture(0x02C, &arch_id);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    // Handle error
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "amd_dbgapi_get_architecture failed");
    return false;
  }

  // Get breakpoint instruction
  const uint8_t *bp_instruction;
  status = amd_dbgapi_architecture_get_info(
      arch_id, AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION,
      sizeof(bp_instruction), &bp_instruction);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(GetLog(GDBRLog::Plugin),
              "AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION failed");
    return false;
  }

  // Get breakpoint instruction size
  size_t bp_size;
  status = amd_dbgapi_architecture_get_info(
      arch_id, AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION_SIZE,
      sizeof(bp_size), &bp_size);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(
        GetLog(GDBRLog::Plugin),
        "AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION_SIZE failed");
    return false;
  }

  // Now call SetGPUBreakpoint with the retrieved instruction and size
  return SetGPUBreakpoint(addr, bp_instruction, bp_size);
}

llvm::Expected<GPUPluginBreakpointHitResponse>
LLDBServerPluginAMDGPU::BreakpointWasHit(GPUPluginBreakpointHitArgs &args) {
  Log *log = GetLog(GDBRLog::Plugin);
  std::string json_string;
  std::string &bp_identifier = args.breakpoint.identifier;
  llvm::raw_string_ostream os(json_string);
  os << toJSON(args);
  LLDB_LOGF(log, "LLDBServerPluginAMDGPU::BreakpointWasHit(\"%s\"):\nJSON:\n%s",
            bp_identifier.c_str(), json_string.c_str());

  GPUPluginBreakpointHitResponse response;
  response.actions.plugin_name = GetPluginName();
  if (bp_identifier == "GPU loader breakpoint") {
    bool has_new_libraries = false;
    bool success = HandleGPUInternalBreakpointHit(m_gpu_internal_bp.value(),
                                                  has_new_libraries);
    assert(success);
    if (has_new_libraries) {
      response.actions.wait_for_gpu_process_to_resume = true;
      auto process = m_gdb_server->GetCurrentProcess();
      ThreadAMDGPU *thread = (ThreadAMDGPU *)process->GetCurrentThread();
      thread->SetStopReason(lldb::eStopReasonDynammicLoader);
      process->Halt();
    }
  }
  return response;
}

GPUActions LLDBServerPluginAMDGPU::GetInitializeActions() {
  GPUActions init_actions;
  init_actions.plugin_name = GetPluginName();

  GPUBreakpointInfo bp1;
  bp1.identifier = "gpu_initialize";
  bp1.name_info = {"a.out", "gpu_initialize"};
  bp1.symbol_names.push_back("gpu_shlib_load");
  init_actions.breakpoints.emplace_back(std::move(bp1));
  return init_actions;
}
