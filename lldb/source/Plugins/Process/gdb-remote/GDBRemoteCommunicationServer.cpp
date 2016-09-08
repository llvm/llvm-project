//===-- GDBRemoteCommunicationServer.cpp ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <errno.h>

#include "lldb/Host/Config.h"

#include "GDBRemoteCommunicationServer.h"

// C Includes
// C++ Includes
#include <cstring>

// Project includes
#include "ProcessGDBRemoteLog.h"
#include "Utility/StringExtractorGDBRemote.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_gdb_remote;

GDBRemoteCommunicationServer::GDBRemoteCommunicationServer(
    const char *comm_name, const char *listener_name)
    : GDBRemoteCommunication(comm_name, listener_name), m_exit_now(false) {}

GDBRemoteCommunicationServer::~GDBRemoteCommunicationServer() {}

void GDBRemoteCommunicationServer::RegisterPacketHandler(
    StringExtractorGDBRemote::ServerPacketType packet_type,
    PacketHandler handler) {
  m_packet_handlers[packet_type] = std::move(handler);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::GetPacketAndSendResponse(uint32_t timeout_usec,
                                                       Error &error,
                                                       bool &interrupt,
                                                       bool &quit) {
  StringExtractorGDBRemote packet;

  PacketResult packet_result =
      WaitForPacketWithTimeoutMicroSecondsNoLock(packet, timeout_usec, false);
  if (packet_result == PacketResult::Success) {
    const StringExtractorGDBRemote::ServerPacketType packet_type =
        packet.GetServerPacketType();
    switch (packet_type) {
    case StringExtractorGDBRemote::eServerPacketType_nack:
    case StringExtractorGDBRemote::eServerPacketType_ack:
      break;

    case StringExtractorGDBRemote::eServerPacketType_invalid:
      error.SetErrorString("invalid packet");
      quit = true;
      break;

    case StringExtractorGDBRemote::eServerPacketType_unimplemented:
      packet_result = SendUnimplementedResponse(packet.GetStringRef().c_str());
      break;

    default:
      auto handler_it = m_packet_handlers.find(packet_type);
      if (handler_it == m_packet_handlers.end())
        packet_result =
            SendUnimplementedResponse(packet.GetStringRef().c_str());
      else
        packet_result = handler_it->second(packet, error, interrupt, quit);
      break;
    }
  } else {
    if (!IsConnected()) {
      error.SetErrorString("lost connection");
      quit = true;
    } else {
      error.SetErrorString("timeout");
    }
  }

  // Check if anything occurred that would force us to want to exit.
  if (m_exit_now)
    quit = true;

  return packet_result;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::SendUnimplementedResponse(const char *) {
  // TODO: Log the packet we aren't handling...
  return SendPacketNoLock("", 0);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::SendErrorResponse(uint8_t err) {
  char packet[16];
  int packet_len = ::snprintf(packet, sizeof(packet), "E%2.2x", err);
  assert(packet_len < (int)sizeof(packet));
  return SendPacketNoLock(packet, packet_len);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::SendIllFormedResponse(
    const StringExtractorGDBRemote &failed_packet, const char *message) {
  Log *log(ProcessGDBRemoteLog::GetLogIfAllCategoriesSet(GDBR_LOG_PACKETS));
  if (log)
    log->Printf("GDBRemoteCommunicationServer::%s: ILLFORMED: '%s' (%s)",
                __FUNCTION__, failed_packet.GetStringRef().c_str(),
                message ? message : "");
  return SendErrorResponse(0x03);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServer::SendOKResponse() {
  return SendPacketNoLock("OK", 2);
}

bool GDBRemoteCommunicationServer::HandshakeWithClient() {
  return GetAck() == PacketResult::Success;
}
