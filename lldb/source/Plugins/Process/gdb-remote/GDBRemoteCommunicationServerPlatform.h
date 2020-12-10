//===-- GDBRemoteCommunicationServerPlatform.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_GDB_REMOTE_GDBREMOTECOMMUNICATIONSERVERPLATFORM_H
#define LLDB_SOURCE_PLUGINS_PROCESS_GDB_REMOTE_GDBREMOTECOMMUNICATIONSERVERPLATFORM_H

#include <map>
#include <mutex>
#include <set>

#include "GDBRemoteCommunicationServerCommon.h"
#include "lldb/Host/Socket.h"

namespace lldb_private {
namespace process_gdb_remote {

class GDBRemoteCommunicationServerPlatform
    : public GDBRemoteCommunicationServerCommon {
public:
  typedef std::map<uint16_t, lldb::pid_t> PortMap;

  GDBRemoteCommunicationServerPlatform(
      const Socket::SocketProtocol socket_protocol, const char *socket_scheme);

  ~GDBRemoteCommunicationServerPlatform() override;

  Status LaunchProcess() override;

  // Set both ports to zero to let the platform automatically bind to
  // a port chosen by the OS.
  void SetPortMap(PortMap &&port_map);

  // If we are using a port map where we can only use certain ports,
  // get the next available port.
  //
  // If we are using a port map and we are out of ports, return UINT16_MAX
  //
  // If we aren't using a port map, return 0 to indicate we should bind to
  // port 0 and then figure out which port we used.
  uint16_t GetNextAvailablePort();

  bool AssociatePortWithProcess(uint16_t port, lldb::pid_t pid);

  bool FreePort(uint16_t port);

  bool FreePortForProcess(lldb::pid_t pid);

  void SetPortOffset(uint16_t port_offset);

  void SetInferiorArguments(const lldb_private::Args &args);

  Status LaunchGDBServer(const lldb_private::Args &args, std::string hostname,
                         lldb::pid_t &pid, uint16_t &port,
                         std::string &socket_name);

  void SetPendingGdbServer(lldb::pid_t pid, uint16_t port,
                           const std::string &socket_name);

protected:
  const Socket::SocketProtocol m_socket_protocol;
  const std::string m_socket_scheme;
  std::recursive_mutex m_spawned_pids_mutex;
  std::set<lldb::pid_t> m_spawned_pids;

  PortMap m_port_map;
  uint16_t m_port_offset;
  struct {
    lldb::pid_t pid;
    uint16_t port;
    std::string socket_name;
  } m_pending_gdb_server;

  PacketResult Handle_qLaunchGDBServer(StringExtractorGDBRemote &packet);

  PacketResult Handle_qQueryGDBServer(StringExtractorGDBRemote &packet);

  PacketResult Handle_qKillSpawnedProcess(StringExtractorGDBRemote &packet);

  PacketResult Handle_qPathComplete(StringExtractorGDBRemote &packet);

  PacketResult Handle_qProcessInfo(StringExtractorGDBRemote &packet);

  PacketResult Handle_qGetWorkingDir(StringExtractorGDBRemote &packet);

  PacketResult Handle_QSetWorkingDir(StringExtractorGDBRemote &packet);

  PacketResult Handle_qC(StringExtractorGDBRemote &packet);

  PacketResult Handle_jSignalsInfo(StringExtractorGDBRemote &packet);

private:
  bool KillSpawnedProcess(lldb::pid_t pid);

  bool DebugserverProcessReaped(lldb::pid_t pid);

  static const FileSpec &GetDomainSocketDir();

  static FileSpec GetDomainSocketPath(const char *prefix);

  GDBRemoteCommunicationServerPlatform(
      const GDBRemoteCommunicationServerPlatform &) = delete;
  const GDBRemoteCommunicationServerPlatform &
  operator=(const GDBRemoteCommunicationServerPlatform &) = delete;
};

} // namespace process_gdb_remote
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_GDB_REMOTE_GDBREMOTECOMMUNICATIONSERVERPLATFORM_H
