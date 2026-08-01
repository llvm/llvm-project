//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLDBServerMockAcceleratorPlugin.h"
#include "ProcessMockAccelerator.h"

#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServerLLGS.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Host/Socket.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Host/posix/ConnectionFileDescriptorPosix.h"
#include "lldb/Utility/Args.h"
#include "lldb/Utility/Connection.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/Support/FormatVariadic.h"

#include <cstdlib>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;

// Read a mock-accelerator setting from an environment variable so tests can
// configure the connection the mock advertises; falls back to default_value.
static std::string GetMockEnvSetting(const char *env_var,
                                     std::string default_value) {
  if (const char *value = ::getenv(env_var))
    return value;
  return default_value;
}

LLDBServerMockAcceleratorPlugin::LLDBServerMockAcceleratorPlugin(
    GDBServer &native_gdb_server, MainLoop &native_main_loop)
    : LLDBServerAcceleratorPlugin(native_gdb_server, native_main_loop) {}

LLDBServerMockAcceleratorPlugin::~LLDBServerMockAcceleratorPlugin() {
  // Stop the mock main loop and wait for its thread, then tear down the objects
  // that reference the loop before it (a member) is destroyed.
  m_mock_main_loop.AddPendingCallback(
      [](MainLoopBase &loop) { loop.RequestTermination(); });
  if (m_mock_main_loop_thread.IsJoinable())
    m_mock_main_loop_thread.Join(/*result=*/nullptr);
  m_read_handles.clear();
  m_listen_socket.reset();
  m_accelerator_gdb_server.reset();
  m_process_manager_up.reset();
}

llvm::StringRef LLDBServerMockAcceleratorPlugin::GetPluginName() {
  return "mock";
}

std::optional<AcceleratorActions>
LLDBServerMockAcceleratorPlugin::GetInitializeActions() {
  AcceleratorActions actions = GetNewAcceleratorAction();

  // Set a breakpoint by function name (no shared library scope) on the
  // dedicated "mock_gpu_accelerator_initialize" hook and ask for the load
  // address of "mock_gpu_accelerator_compute" to be delivered when it is hit.
  // Using a dedicated, uniquely named function (rather than "main") keeps this
  // mock from affecting other inferiors that lldb-server launches when the
  // plugin is compiled in.
  AcceleratorBreakpointInfo bp;
  bp.identifier = kBreakpointIDInitialize;
  bp.by_name = AcceleratorBreakpointByName{std::nullopt,
                                           "mock_gpu_accelerator_initialize"};
  bp.symbol_names.push_back("mock_gpu_accelerator_compute");
  actions.breakpoints.push_back(std::move(bp));

  return actions;
}

llvm::Expected<AcceleratorBreakpointHitResponse>
LLDBServerMockAcceleratorPlugin::BreakpointWasHit(
    AcceleratorBreakpointHitArgs &args) {
  AcceleratorBreakpointHitResponse response;

  switch (args.breakpoint.identifier) {
  case kBreakpointIDInitialize: {
    // The initialize breakpoint was hit. Arm the remaining test breakpoints:
    // two more breakpoint types, plus the connection hook now that the
    // accelerator has initialized.
    response.disable_bp = true;
    response.auto_resume_native = false;

    AcceleratorActions actions = GetNewAcceleratorAction();

    // Tests build to "a.out", so scope this by-name breakpoint to it.
    AcceleratorBreakpointInfo by_name_shlib;
    by_name_shlib.identifier = kBreakpointIDByNameShlib;
    by_name_shlib.by_name =
        AcceleratorBreakpointByName{"a.out", "mock_gpu_accelerator_finish"};
    actions.breakpoints.push_back(std::move(by_name_shlib));

    if (std::optional<uint64_t> compute_addr =
            args.GetSymbolValue("mock_gpu_accelerator_compute")) {
      AcceleratorBreakpointInfo by_address;
      by_address.identifier = kBreakpointIDByAddress;
      by_address.by_address = AcceleratorBreakpointByAddress{*compute_addr};
      actions.breakpoints.push_back(std::move(by_address));
    }

    // Now that the accelerator has initialized, set the breakpoint on the
    // dedicated connection hook. Arming it only after the initialize hit
    // (rather than up front) mirrors how a real GPU plugin connects once the
    // runtime is ready.
    AcceleratorBreakpointInfo connect_bp;
    connect_bp.identifier = kBreakpointIDConnect;
    connect_bp.by_name = AcceleratorBreakpointByName{
        std::nullopt, "mock_gpu_accelerator_connect"};
    actions.breakpoints.push_back(std::move(connect_bp));

    response.actions = std::move(actions);
    break;
  }
  case kBreakpointIDByAddress:
  case kBreakpointIDByNameShlib:
    // Disable and stop the native process so the hit is observable.
    response.disable_bp = true;
    response.auto_resume_native = false;
    break;
  case kBreakpointIDConnect: {
    // The program reached its connection hook. Ask the client to create a
    // second target and connect to our in-process mock accelerator GDB
    // server.
    response.disable_bp = true;
    response.auto_resume_native = false;
    AcceleratorActions actions = GetNewAcceleratorAction();
    actions.session_name = "Mock Accelerator Session";
    actions.connect_info = CreateConnection();
    response.actions = std::move(actions);
    break;
  }
  }

  return response;
}

std::optional<AcceleratorConnectionInfo>
LLDBServerMockAcceleratorPlugin::CreateConnection() {
  Log *log = GetLog(GDBRLog::Plugin);

  // An in-process gdb-remote server backed by a synthetic
  // ProcessMockAccelerator; no real process is launched.
  m_process_manager_up =
      std::make_unique<ProcessMockAccelerator::Manager>(m_mock_main_loop);
  m_accelerator_gdb_server = std::make_unique<GDBRemoteCommunicationServerLLGS>(
      m_mock_main_loop, *m_process_manager_up);

  // LLGS creates its current process from a launch; the manager ignores the
  // path but requires a non-empty argument list.
  ProcessLaunchInfo launch_info;
  Args args;
  args.AppendArgument("/pretend/path/to/mockgpu");
  launch_info.SetArguments(args, /*first_arg_is_executable=*/true);
  m_accelerator_gdb_server->SetLaunchInfo(launch_info);
  if (Status error = m_accelerator_gdb_server->LaunchProcess(); error.Fail())
    LLDB_LOG(log, "failed to create mock accelerator process: {0}",
             error.AsCString());

  // Register the accept handler before starting the loop thread so the loop's
  // handles are only touched on that thread.
  llvm::Expected<std::unique_ptr<TCPSocket>> sock =
      Socket::TcpListen("localhost:0");
  if (!sock) {
    LLDB_LOG_ERROR(log, sock.takeError(),
                   "mock accelerator failed to listen: {0}");
    return std::nullopt;
  }
  m_listen_socket = std::move(*sock);
  llvm::Expected<std::vector<MainLoopBase::ReadHandleUP>> handles =
      m_listen_socket->Accept(
          m_mock_main_loop, [this](std::unique_ptr<Socket> socket) {
            std::unique_ptr<Connection> connection_up =
                std::make_unique<ConnectionFileDescriptor>(std::move(socket));
            m_accelerator_gdb_server->InitializeConnection(
                std::move(connection_up));
          });
  if (!handles) {
    LLDB_LOG_ERROR(log, handles.takeError(),
                   "mock accelerator failed to accept: {0}");
    return std::nullopt;
  }
  m_read_handles = std::move(*handles);

  llvm::Expected<HostThread> loop_thread = ThreadLauncher::LaunchThread(
      "mock-accel.loop", [this]() -> lldb::thread_result_t {
        m_mock_main_loop.Run();
        return {};
      });
  if (!loop_thread) {
    LLDB_LOG_ERROR(log, loop_thread.takeError(),
                   "mock accelerator failed to start its main loop: {0}");
    m_read_handles.clear();
    m_listen_socket.reset();
    return std::nullopt;
  }
  m_mock_main_loop_thread = *loop_thread;

  AcceleratorConnectionInfo info;
  info.connect_url = llvm::formatv("connect://localhost:{0}",
                                   m_listen_socket->GetLocalPortNumber());
  // Default to the host; tests override via env vars to exercise
  // invalid-platform and incompatible-triple failures.
  info.platform_name =
      GetMockEnvSetting("LLDB_MOCK_ACCELERATOR_PLATFORM", "host");
  info.triple =
      GetMockEnvSetting("LLDB_MOCK_ACCELERATOR_TRIPLE",
                        HostInfo::GetArchitecture().GetTriple().str());
  info.synchronous = true;
  return info;
}
