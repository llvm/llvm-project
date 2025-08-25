//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Protocol/MCP/Transport.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/Socket.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <thread>

using namespace llvm;
using namespace lldb;
using namespace lldb_private;

namespace lldb_protocol::mcp {

static Expected<sys::ProcessInfo> StartServer() {
  static once_flag f;
  static FileSpec candidate;
  llvm::call_once(f, [] {
    HostInfo::Initialize();
    candidate = HostInfo::GetSupportExeDir();
    candidate.AppendPathComponent("lldb-mcp");
  });

  if (!FileSystem::Instance().Exists(candidate))
    return createStringError("lldb-mcp executable not found");
  std::vector<StringRef> args = {candidate.GetPath(), "--server"};
  sys::ProcessInfo proc =
      sys::ExecuteNoWait(candidate.GetPath(), args, std::nullopt, {}, 0,
                         nullptr, nullptr, nullptr, /*DetachProcess=*/true);
  if (proc.Pid == sys::ProcessInfo::InvalidPid)
    return createStringError("Failed to start server: " + candidate.GetPath());
  StringRef socket_path = CommunicationSocketPath();
  while (!sys::fs::exists(socket_path))
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  return proc;
}

Transport::Transport(lldb::IOObjectSP input, lldb::IOObjectSP output,
                     std::string client_name, LogCallback log_callback)
    : JSONRPCTransport(input, output), m_client_name(client_name),
      m_log_callback(log_callback) {}

void Transport::Log(llvm::StringRef message) {
  if (m_log_callback)
    m_log_callback(llvm::formatv("{0}: {1}", m_client_name, message).str());
}

llvm::StringRef CommunicationSocketPath() {
  static std::once_flag f;
  static SmallString<256> socket_path;
  llvm::call_once(f, [] {
    assert(sys::path::home_directory(socket_path) &&
           "failed to get home directory");
    sys::path::append(socket_path, ".lldb-mcp-sock");
  });
  return socket_path.str();
}

Expected<IOObjectSP> Connect() {
  StringRef socket_path = CommunicationSocketPath();
  if (!sys::fs::exists(socket_path))
    if (llvm::Error err = StartServer().takeError())
      return err;

  Socket::SocketProtocol protocol = Socket::ProtocolUnixDomain;
  Status error;
  std::unique_ptr<Socket> socket = Socket::Create(protocol, error);
  if (error.Fail())
    return error.takeError();
  std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (std::chrono::steady_clock::now() < deadline) {
    Status error = socket->Connect(socket_path);
    if (error.Success()) {
      return socket;
    }
    if (error.Fail() && error.GetError() != ECONNREFUSED &&
        error.GetError() != ENOENT)
      return error.takeError();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  return createStringError("failed to connect to lldb-mcp multiplexer");
}

Expected<MCPTransportUP> Transport::Connect(llvm::raw_ostream *logger) {
  Expected<IOObjectSP> maybe_sock = lldb_protocol::mcp::Connect();
  if (!maybe_sock)
    return maybe_sock.takeError();

  return std::make_unique<Transport>(*maybe_sock, *maybe_sock, "client",
                                     [logger](StringRef msg) {
                                       if (logger)
                                         *logger << msg << "\n";
                                     });
}

} // namespace lldb_protocol::mcp
