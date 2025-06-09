//===-- LLDBServerPluginMockGPU.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLDBServerPluginMockGPU.h"
#include "ProcessMockGPU.h"
#include "lldb/Host/common/TCPSocket.h"
#include "llvm/Support/Error.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServerLLGS.h"
#include "lldb/Host/posix/ConnectionFileDescriptorPosix.h"

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <thread>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;

LLDBServerPluginMockGPU::LLDBServerPluginMockGPU(
  LLDBServerPlugin::GDBServer &native_process)
    : LLDBServerPlugin(native_process) {
  m_process_manager_up.reset(new ProcessMockGPU::Manager(m_main_loop));
  m_gdb_server.reset(new GDBRemoteCommunicationServerLLGS(
      m_main_loop, *m_process_manager_up, "mock-gpu.server"));

  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOGF(log, "LLDBServerPluginMockGPU::LLDBServerPluginMockGPU() faking launch...");
  ProcessLaunchInfo info;
  info.GetFlags().Set(eLaunchFlagStopAtEntry | eLaunchFlagDebug |
                      eLaunchFlagDisableASLR);
  Args args;
  args.AppendArgument("/pretend/path/to/mockgpu");
  args.AppendArgument("--option1");
  args.AppendArgument("--option2");
  args.AppendArgument("--option3");
  info.SetArguments(args, true);
  info.GetEnvironment() = Host::GetEnvironment();
  m_gdb_server->SetLaunchInfo(info);
  Status error = m_gdb_server->LaunchProcess();
  if (error.Fail()) {
    LLDB_LOGF(log, "LLDBServerPluginMockGPU::LLDBServerPluginMockGPU() failed to launch: %s", error.AsCString());
  } else {
    LLDB_LOGF(log, "LLDBServerPluginMockGPU::LLDBServerPluginMockGPU() launched successfully");
  }
}

LLDBServerPluginMockGPU::~LLDBServerPluginMockGPU() {
  CloseFDs();
}

llvm::StringRef LLDBServerPluginMockGPU::GetPluginName() {
  return "mock-gpu";
}

void LLDBServerPluginMockGPU::CloseFDs() {
  if (m_fds[0] != -1) {
    close(m_fds[0]);
    m_fds[0] = -1;
  }
  if (m_fds[1] != -1) {
    close(m_fds[1]);
    m_fds[1] = -1;
  }
}

int LLDBServerPluginMockGPU::GetEventFileDescriptorAtIndex(size_t idx) {
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


bool LLDBServerPluginMockGPU::HandleEventFileDescriptorEvent(int fd) { 
  if (fd == m_fds[0]) {
    char buf[1];
    // Read 1 bytes from the fd
    read(m_fds[0], buf, sizeof(buf));
    return true;
  }
  return false;
}

void LLDBServerPluginMockGPU::AcceptAndMainLoopThread(
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

std::optional<GPUPluginConnectionInfo> LLDBServerPluginMockGPU::CreateConnection() {
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
    connection_info.connect_url = llvm::formatv("connect://localhost:{}", 
                                                listen_port);
    LLDB_LOGF(log, "%s listening to %u", __PRETTY_FUNCTION__, listen_port);
    std::thread t(&LLDBServerPluginMockGPU::AcceptAndMainLoopThread, this, 
                  std::move(*sock));
    t.detach();
    return connection_info;
  } else {
    std::string error = llvm::toString(sock.takeError());
    LLDB_LOGF(log, "%s failed to listen to localhost:0: %s", 
              __PRETTY_FUNCTION__, error.c_str());
  }
  m_is_listening = false;
  return std::nullopt;
}

std::optional<GPUActions> LLDBServerPluginMockGPU::NativeProcessIsStopping() {
  NativeProcessProtocol *native_process = m_native_process.GetCurrentProcess();
  // Show that we can return a valid GPUActions object from a stop event.
  if (native_process->GetStopID() == 3) {
    GPUActions actions;
    actions.plugin_name = GetPluginName();
    GPUBreakpointInfo bp;
    bp.identifier = "3rd stop breakpoint";
    bp.name_info = {"a.out", "gpu_third_stop"};
    actions.breakpoints.emplace_back(std::move(bp));
    return actions;
  }
  return std::nullopt;
}

GPUPluginBreakpointHitResponse 
LLDBServerPluginMockGPU::BreakpointWasHit(GPUPluginBreakpointHitArgs &args) {
  Log *log = GetLog(GDBRLog::Plugin);
  std::string json_string;
  std::string &bp_identifier = args.breakpoint.identifier;
  llvm::raw_string_ostream os(json_string);
  os << toJSON(args);
  LLDB_LOGF(log, "LLDBServerPluginMockGPU::BreakpointWasHit(\"%s\"):\nJSON:\n%s", 
            bp_identifier.c_str(), json_string.c_str());

  GPUPluginBreakpointHitResponse response;
  response.actions.plugin_name = GetPluginName();
  if (bp_identifier == "gpu_initialize") {
    response.disable_bp = true;
    LLDB_LOGF(log, "LLDBServerPluginMockGPU::BreakpointWasHit(\"%s\") disabling breakpoint", 
              bp_identifier.c_str());
    response.actions.connect_info = CreateConnection();

    // We asked for the symbol "gpu_shlib_load" to be delivered as a symbol
    // value when the "gpu_initialize" breakpoint was set. So we will use this
    // to set a breakpoint by address to test setting breakpoints by address.
    std::optional<uint64_t> gpu_shlib_load_addr = 
        args.GetSymbolValue("gpu_shlib_load");
    if (gpu_shlib_load_addr) {
      GPUBreakpointInfo bp;
      bp.identifier = "gpu_shlib_load";
      bp.addr_info = {*gpu_shlib_load_addr};
      bp.symbol_names.push_back("g_shlib_list");
      bp.symbol_names.push_back("invalid_symbol");
      response.actions.breakpoints.emplace_back(std::move(bp));
    }
  } else if (bp_identifier == "gpu_shlib_load") {
    // Tell the native process to tell the GPU process to load libraries.
    response.actions.load_libraries = true;
  }
  return response;
}

GPUActions LLDBServerPluginMockGPU::GetInitializeActions() {
  GPUActions init_actions;
  init_actions.plugin_name = GetPluginName();
  
  GPUBreakpointInfo bp1;
  bp1.identifier = "gpu_initialize";
  bp1.name_info = {"a.out", "gpu_initialize"};  
  bp1.symbol_names.push_back("gpu_shlib_load");
  init_actions.breakpoints.emplace_back(std::move(bp1));
  return init_actions;
}
