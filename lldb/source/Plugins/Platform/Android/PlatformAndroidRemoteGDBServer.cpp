//===-- PlatformAndroidRemoteGDBServer.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/UriParser.h"

#include "PlatformAndroidRemoteGDBServer.h"

#include <sstream>

using namespace lldb;
using namespace lldb_private;
using namespace platform_android;

static const lldb::pid_t g_remote_platform_pid =
    0; // Alias for the process id of lldb-platform

static Status ForwardPortWithAdb(
    const uint16_t local_port, const uint16_t remote_port,
    llvm::StringRef remote_socket_name,
    const llvm::Optional<AdbClient::UnixSocketNamespace> &socket_namespace,
    std::string &device_id) {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));

  AdbClient adb;
  auto error = AdbClient::CreateByDeviceID(device_id, adb);
  if (error.Fail())
    return error;

  device_id = adb.GetDeviceID();
  if (log)
    log->Printf("Connected to Android device \"%s\"", device_id.c_str());

  if (remote_port != 0) {
    if (log)
      log->Printf("Forwarding remote TCP port %d to local TCP port %d",
                  remote_port, local_port);
    return adb.SetPortForwarding(local_port, remote_port);
  }

  if (log)
    log->Printf("Forwarding remote socket \"%s\" to local TCP port %d",
                remote_socket_name.str().c_str(), local_port);

  if (!socket_namespace)
    return Status("Invalid socket namespace");

  return adb.SetPortForwarding(local_port, remote_socket_name,
                               *socket_namespace);
}

static Status DeleteForwardPortWithAdb(uint16_t local_port,
                                       const std::string &device_id) {
  AdbClient adb(device_id);
  return adb.DeletePortForwarding(local_port);
}

static Status FindUnusedPort(uint16_t &port) {
  Status error;
  std::unique_ptr<TCPSocket> tcp_socket(new TCPSocket(true, false));
  if (error.Fail())
    return error;

  error = tcp_socket->Listen("127.0.0.1:0", 1);
  if (error.Success())
    port = tcp_socket->GetLocalPortNumber();

  return error;
}

PlatformAndroidRemoteGDBServer::PlatformAndroidRemoteGDBServer() {}

PlatformAndroidRemoteGDBServer::~PlatformAndroidRemoteGDBServer() {
  for (const auto &it : m_port_forwards)
    DeleteForwardPortWithAdb(it.second, m_device_id);
}

bool PlatformAndroidRemoteGDBServer::LaunchGDBServer(lldb::pid_t &pid,
                                                     std::string &connect_url) {
  uint16_t remote_port = 0;
  std::string socket_name;
  if (!m_gdb_client.LaunchGDBServer("127.0.0.1", pid, remote_port, socket_name))
    return false;

  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));

  auto error =
      MakeConnectURL(pid, remote_port, socket_name.c_str(), connect_url);
  if (error.Success() && log)
    log->Printf("gdbserver connect URL: %s", connect_url.c_str());

  return error.Success();
}

bool PlatformAndroidRemoteGDBServer::KillSpawnedProcess(lldb::pid_t pid) {
  DeleteForwardPort(pid);
  return m_gdb_client.KillSpawnedProcess(pid);
}

Status PlatformAndroidRemoteGDBServer::ConnectRemote(Args &args) {
  m_device_id.clear();

  if (args.GetArgumentCount() != 1)
    return Status(
        "\"platform connect\" takes a single argument: <connect-url>");

  int remote_port;
  llvm::StringRef scheme, host, path;
  const char *url = args.GetArgumentAtIndex(0);
  if (!url)
    return Status("URL is null.");
  if (!UriParser::Parse(url, scheme, host, remote_port, path))
    return Status("Invalid URL: %s", url);
  if (host != "localhost")
    m_device_id = host;

  m_socket_namespace.reset();
  if (scheme == ConnectionFileDescriptor::UNIX_CONNECT_SCHEME)
    m_socket_namespace = AdbClient::UnixSocketNamespaceFileSystem;
  else if (scheme == ConnectionFileDescriptor::UNIX_ABSTRACT_CONNECT_SCHEME)
    m_socket_namespace = AdbClient::UnixSocketNamespaceAbstract;

  std::string connect_url;
  auto error =
      MakeConnectURL(g_remote_platform_pid, (remote_port < 0) ? 0 : remote_port,
                     path, connect_url);

  if (error.Fail())
    return error;

  args.ReplaceArgumentAtIndex(0, connect_url);

  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));
  if (log)
    log->Printf("Rewritten platform connect URL: %s", connect_url.c_str());

  error = PlatformRemoteGDBServer::ConnectRemote(args);
  if (error.Fail())
    DeleteForwardPort(g_remote_platform_pid);

  return error;
}

Status PlatformAndroidRemoteGDBServer::DisconnectRemote() {
  DeleteForwardPort(g_remote_platform_pid);
  return PlatformRemoteGDBServer::DisconnectRemote();
}

void PlatformAndroidRemoteGDBServer::DeleteForwardPort(lldb::pid_t pid) {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));

  auto it = m_port_forwards.find(pid);
  if (it == m_port_forwards.end())
    return;

  const auto port = it->second;
  const auto error = DeleteForwardPortWithAdb(port, m_device_id);
  if (error.Fail()) {
    if (log)
      log->Printf("Failed to delete port forwarding (pid=%" PRIu64
                  ", port=%d, device=%s): %s",
                  pid, port, m_device_id.c_str(), error.AsCString());
  }
  m_port_forwards.erase(it);
}

Status PlatformAndroidRemoteGDBServer::MakeConnectURL(
    const lldb::pid_t pid, const uint16_t remote_port,
    llvm::StringRef remote_socket_name, std::string &connect_url) {
  static const int kAttempsNum = 5;

  Status error;
  // There is a race possibility that somebody will occupy a port while we're
  // in between FindUnusedPort and ForwardPortWithAdb - adding the loop to
  // mitigate such problem.
  for (auto i = 0; i < kAttempsNum; ++i) {
    uint16_t local_port = 0;
    error = FindUnusedPort(local_port);
    if (error.Fail())
      return error;

    error = ForwardPortWithAdb(local_port, remote_port, remote_socket_name,
                               m_socket_namespace, m_device_id);
    if (error.Success()) {
      m_port_forwards[pid] = local_port;
      std::ostringstream url_str;
      url_str << "connect://localhost:" << local_port;
      connect_url = url_str.str();
      break;
    }
  }

  return error;
}

lldb::ProcessSP PlatformAndroidRemoteGDBServer::ConnectProcess(
    llvm::StringRef connect_url, llvm::StringRef plugin_name,
    lldb_private::Debugger &debugger, lldb_private::Target *target,
    lldb_private::Status &error) {
  // We don't have the pid of the remote gdbserver when it isn't started by us
  // but we still want to store the list of port forwards we set up in our port
  // forward map. Generate a fake pid for these cases what won't collide with
  // any other valid pid on android.
  static lldb::pid_t s_remote_gdbserver_fake_pid = 0xffffffffffffffffULL;

  int remote_port;
  llvm::StringRef scheme, host, path;
  if (!UriParser::Parse(connect_url, scheme, host, remote_port, path)) {
    error.SetErrorStringWithFormat("Invalid URL: %s",
                                   connect_url.str().c_str());
    return nullptr;
  }

  std::string new_connect_url;
  error = MakeConnectURL(s_remote_gdbserver_fake_pid--,
                         (remote_port < 0) ? 0 : remote_port, path,
                         new_connect_url);
  if (error.Fail())
    return nullptr;

  return PlatformRemoteGDBServer::ConnectProcess(new_connect_url, plugin_name,
                                                 debugger, target, error);
}
