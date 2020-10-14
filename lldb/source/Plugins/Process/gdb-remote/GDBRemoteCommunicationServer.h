//===-- GDBRemoteCommunicationServer.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_GDB_REMOTE_GDBREMOTECOMMUNICATIONSERVER_H
#define LLDB_SOURCE_PLUGINS_PROCESS_GDB_REMOTE_GDBREMOTECOMMUNICATIONSERVER_H

#include <functional>
#include <map>

#include "GDBRemoteCommunication.h"
#include "lldb/lldb-private-forward.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

class StringExtractorGDBRemote;

namespace lldb_private {
namespace process_gdb_remote {

class ProcessGDBRemote;

class GDBRemoteCommunicationServer : public GDBRemoteCommunication {
public:
  using PortMap = std::map<uint16_t, lldb::pid_t>;
  using PacketHandler =
      std::function<PacketResult(StringExtractorGDBRemote &packet,
                                 Status &error, bool &interrupt, bool &quit)>;

  GDBRemoteCommunicationServer(const char *comm_name,
                               const char *listener_name);

  ~GDBRemoteCommunicationServer() override;

  void
  RegisterPacketHandler(StringExtractorGDBRemote::ServerPacketType packet_type,
                        PacketHandler handler);

  PacketResult GetPacketAndSendResponse(Timeout<std::micro> timeout,
                                        Status &error, bool &interrupt,
                                        bool &quit);

  // After connecting, do a little handshake with the client to make sure
  // we are at least communicating
  bool HandshakeWithClient();

protected:
  std::map<StringExtractorGDBRemote::ServerPacketType, PacketHandler>
      m_packet_handlers;
  bool m_exit_now; // use in asynchronous handling to indicate process should
                   // exit.

  bool m_send_error_strings = false; // If the client enables this then
                                     // we will send error strings as well.

  PacketResult Handle_QErrorStringEnable(StringExtractorGDBRemote &packet);

  PacketResult SendErrorResponse(const Status &error);

  PacketResult SendErrorResponse(llvm::Error error);

  PacketResult SendUnimplementedResponse(const char *packet);

  PacketResult SendErrorResponse(uint8_t error);

  PacketResult SendIllFormedResponse(const StringExtractorGDBRemote &packet,
                                     const char *error_message);

  PacketResult SendOKResponse();

private:
  GDBRemoteCommunicationServer(const GDBRemoteCommunicationServer &) = delete;
  const GDBRemoteCommunicationServer &
  operator=(const GDBRemoteCommunicationServer &) = delete;
};

} // namespace process_gdb_remote
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_GDB_REMOTE_GDBREMOTECOMMUNICATIONSERVER_H
