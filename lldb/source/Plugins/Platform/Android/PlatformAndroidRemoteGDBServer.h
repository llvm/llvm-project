//===-- PlatformAndroidRemoteGDBServer.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformAndroidRemoteGDBServer_h_
#define liblldb_PlatformAndroidRemoteGDBServer_h_

#include <map>
#include <utility>

#include "Plugins/Platform/gdb-server/PlatformRemoteGDBServer.h"

#include "llvm/ADT/Optional.h"

#include "AdbClient.h"

namespace lldb_private {
namespace platform_android {

class PlatformAndroidRemoteGDBServer
    : public platform_gdb_server::PlatformRemoteGDBServer {
public:
  PlatformAndroidRemoteGDBServer();

  ~PlatformAndroidRemoteGDBServer() override;

  Status ConnectRemote(Args &args) override;

  Status DisconnectRemote() override;

  lldb::ProcessSP ConnectProcess(llvm::StringRef connect_url,
                                 llvm::StringRef plugin_name,
                                 lldb_private::Debugger &debugger,
                                 lldb_private::Target *target,
                                 lldb_private::Status &error) override;

protected:
  std::string m_device_id;
  std::map<lldb::pid_t, uint16_t> m_port_forwards;
  llvm::Optional<AdbClient::UnixSocketNamespace> m_socket_namespace;

  bool LaunchGDBServer(lldb::pid_t &pid, std::string &connect_url) override;

  bool KillSpawnedProcess(lldb::pid_t pid) override;

  void DeleteForwardPort(lldb::pid_t pid);

  Status MakeConnectURL(const lldb::pid_t pid, const uint16_t remote_port,
                        llvm::StringRef remote_socket_name,
                        std::string &connect_url);

private:
  DISALLOW_COPY_AND_ASSIGN(PlatformAndroidRemoteGDBServer);
};

} // namespace platform_android
} // namespace lldb_private

#endif // liblldb_PlatformAndroidRemoteGDBServer_h_
