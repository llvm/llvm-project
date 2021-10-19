//===-- GDBRemoteCommunicationReplayServer.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cerrno>

#include "lldb/Host/Config.h"
#include "llvm/ADT/ScopeExit.h"

#include "GDBRemoteCommunicationReplayServer.h"
#include "ProcessGDBRemoteLog.h"

// C Includes
// C++ Includes
#include <cstring>

// Project includes
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Event.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/StringExtractorGDBRemote.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_gdb_remote;

/// Check if the given expected packet matches the actual packet.
static bool unexpected(llvm::StringRef expected, llvm::StringRef actual) {
  // The 'expected' string contains the raw data, including the leading $ and
  // trailing checksum. The 'actual' string contains only the packet's content.
  if (expected.contains(actual))
    return false;
  // Contains a PID which might be different.
  if (expected.contains("vAttach"))
    return false;
  // Contains a ascii-hex-path.
  if (expected.contains("QSetSTD"))
    return false;
  // Contains environment values.
  if (expected.contains("QEnvironment"))
    return false;

  return true;
}

/// Check if we should reply to the given packet.
static bool skip(llvm::StringRef data) {
  assert(!data.empty() && "Empty packet?");

  // We've already acknowledge the '+' packet so we're done here.
  if (data == "+")
    return true;

  /// Don't 't reply to ^C. We need this because of stop reply packets, which
  /// are only returned when the target halts. Reproducers synchronize these
  /// 'asynchronous' replies, by recording them as a regular replies to the
  /// previous packet (e.g. vCont). As a result, we should ignore real
  /// asynchronous requests.
  if (data.data()[0] == 0x03)
    return true;

  return false;
}

GDBRemoteCommunicationReplayServer::GDBRemoteCommunicationReplayServer()
    : GDBRemoteCommunication("gdb-replay", "gdb-replay.rx_packet"),
      m_async_broadcaster(nullptr, "lldb.gdb-replay.async-broadcaster"),
      m_async_listener_sp(
          Listener::MakeListener("lldb.gdb-replay.async-listener")),
      m_async_thread_state_mutex() {
  m_async_broadcaster.SetEventName(eBroadcastBitAsyncContinue,
                                   "async thread continue");
  m_async_broadcaster.SetEventName(eBroadcastBitAsyncThreadShouldExit,
                                   "async thread should exit");

  const uint32_t async_event_mask =
      eBroadcastBitAsyncContinue | eBroadcastBitAsyncThreadShouldExit;
  m_async_listener_sp->StartListeningForEvents(&m_async_broadcaster,
                                               async_event_mask);
}

GDBRemoteCommunicationReplayServer::~GDBRemoteCommunicationReplayServer() {
  StopAsyncThread();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationReplayServer::GetPacketAndSendResponse(
    Timeout<std::micro> timeout, Status &error, bool &interrupt, bool &quit) {
  std::lock_guard<std::recursive_mutex> guard(m_async_thread_state_mutex);

  StringExtractorGDBRemote packet;
  PacketResult packet_result = WaitForPacketNoLock(packet, timeout, false);

  if (packet_result != PacketResult::Success) {
    if (!IsConnected()) {
      error.SetErrorString("lost connection");
      quit = true;
    } else {
      error.SetErrorString("timeout");
    }
    return packet_result;
  }

  m_async_broadcaster.BroadcastEvent(eBroadcastBitAsyncContinue);

  // Check if we should reply to this packet.
  if (skip(packet.GetStringRef()))
    return PacketResult::Success;

  // This completes the handshake. Since m_send_acks was true, we can unset it
  // already.
  if (packet.GetStringRef() == "QStartNoAckMode")
    m_send_acks = false;

  // A QEnvironment packet is sent for every environment variable. If the
  // number of environment variables is different during replay, the replies
  // become out of sync.
  if (packet.GetStringRef().find("QEnvironment") == 0)
    return SendRawPacketNoLock("$OK#9a");

  Log *log(ProcessGDBRemoteLog::GetLogIfAllCategoriesSet(GDBR_LOG_PROCESS));
  while (!m_packet_history.empty()) {
    // Pop last packet from the history.
    GDBRemotePacket entry = m_packet_history.back();
    m_packet_history.pop_back();

    // Decode run-length encoding.
    const std::string expanded_data =
        GDBRemoteCommunication::ExpandRLE(entry.packet.data);

    // We've handled the handshake implicitly before. Skip the packet and move
    // on.
    if (entry.packet.data == "+")
      continue;

    if (entry.type == GDBRemotePacket::ePacketTypeSend) {
      if (unexpected(expanded_data, packet.GetStringRef())) {
        LLDB_LOG(log,
                 "GDBRemoteCommunicationReplayServer expected packet: '{0}'",
                 expanded_data);
        LLDB_LOG(log, "GDBRemoteCommunicationReplayServer actual packet: '{0}'",
                 packet.GetStringRef());
#ifndef NDEBUG
        // This behaves like a regular assert, but prints the expected and
        // received packet before aborting.
        printf("Reproducer expected packet: '%s'\n", expanded_data.c_str());
        printf("Reproducer received packet: '%s'\n",
               packet.GetStringRef().data());
        llvm::report_fatal_error("Encountered unexpected packet during replay");
#endif
        return PacketResult::ErrorSendFailed;
      }

      // Ignore QEnvironment packets as they're handled earlier.
      if (expanded_data.find("QEnvironment") == 1) {
        assert(m_packet_history.back().type ==
               GDBRemotePacket::ePacketTypeRecv);
        m_packet_history.pop_back();
      }

      continue;
    }

    if (entry.type == GDBRemotePacket::ePacketTypeInvalid) {
      LLDB_LOG(
          log,
          "GDBRemoteCommunicationReplayServer skipped invalid packet: '{0}'",
          packet.GetStringRef());
      continue;
    }

    LLDB_LOG(log,
             "GDBRemoteCommunicationReplayServer replied to '{0}' with '{1}'",
             packet.GetStringRef(), entry.packet.data);
    return SendRawPacketNoLock(entry.packet.data);
  }

  quit = true;

  return packet_result;
}

llvm::Error
GDBRemoteCommunicationReplayServer::LoadReplayHistory(const FileSpec &path) {
  auto error_or_file = MemoryBuffer::getFile(path.GetPath());
  if (auto err = error_or_file.getError())
    return errorCodeToError(err);

  yaml::Input yin((*error_or_file)->getBuffer());
  yin >> m_packet_history;

  if (auto err = yin.error())
    return errorCodeToError(err);

  // We want to manipulate the vector like a stack so we need to reverse the
  // order of the packets to have the oldest on at the back.
  std::reverse(m_packet_history.begin(), m_packet_history.end());

  return Error::success();
}

bool GDBRemoteCommunicationReplayServer::StartAsyncThread() {
  std::lock_guard<std::recursive_mutex> guard(m_async_thread_state_mutex);
  if (!m_async_thread.IsJoinable()) {
    // Create a thread that watches our internal state and controls which
    // events make it to clients (into the DCProcess event queue).
    llvm::Expected<HostThread> async_thread = ThreadLauncher::LaunchThread(
        "<lldb.gdb-replay.async>",
        GDBRemoteCommunicationReplayServer::AsyncThread, this);
    if (!async_thread) {
      LLDB_LOG_ERROR(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST),
                     async_thread.takeError(),
                     "failed to launch host thread: {}");
      return false;
    }
    m_async_thread = *async_thread;
  }

  // Wait for handshake.
  m_async_broadcaster.BroadcastEvent(eBroadcastBitAsyncContinue);

  return m_async_thread.IsJoinable();
}

void GDBRemoteCommunicationReplayServer::StopAsyncThread() {
  std::lock_guard<std::recursive_mutex> guard(m_async_thread_state_mutex);

  if (!m_async_thread.IsJoinable())
    return;

  // Request thread to stop.
  m_async_broadcaster.BroadcastEvent(eBroadcastBitAsyncThreadShouldExit);

  // Disconnect client.
  Disconnect();

  // Stop the thread.
  m_async_thread.Join(nullptr);
  m_async_thread.Reset();
}

void GDBRemoteCommunicationReplayServer::ReceivePacket(
    GDBRemoteCommunicationReplayServer &server, bool &done) {
  Status error;
  bool interrupt;
  auto packet_result = server.GetPacketAndSendResponse(std::chrono::seconds(1),
                                                       error, interrupt, done);
  if (packet_result != GDBRemoteCommunication::PacketResult::Success &&
      packet_result !=
          GDBRemoteCommunication::PacketResult::ErrorReplyTimeout) {
    done = true;
  } else {
    server.m_async_broadcaster.BroadcastEvent(eBroadcastBitAsyncContinue);
  }
}

thread_result_t GDBRemoteCommunicationReplayServer::AsyncThread(void *arg) {
  GDBRemoteCommunicationReplayServer *server =
      (GDBRemoteCommunicationReplayServer *)arg;
  auto D = make_scope_exit([&]() { server->Disconnect(); });
  EventSP event_sp;
  bool done = false;
  while (!done) {
    if (server->m_async_listener_sp->GetEvent(event_sp, llvm::None)) {
      const uint32_t event_type = event_sp->GetType();
      if (event_sp->BroadcasterIs(&server->m_async_broadcaster)) {
        switch (event_type) {
        case eBroadcastBitAsyncContinue:
          ReceivePacket(*server, done);
          if (done)
            return {};
          break;
        case eBroadcastBitAsyncThreadShouldExit:
        default:
          return {};
        }
      }
    }
  }

  return {};
}

Status GDBRemoteCommunicationReplayServer::Connect(
    process_gdb_remote::GDBRemoteCommunicationClient &client) {
  repro::Loader *loader = repro::Reproducer::Instance().GetLoader();
  if (!loader)
    return Status("No loader provided.");

  static std::unique_ptr<repro::MultiLoader<repro::GDBRemoteProvider>>
      multi_loader = repro::MultiLoader<repro::GDBRemoteProvider>::Create(
          repro::Reproducer::Instance().GetLoader());
  if (!multi_loader)
    return Status("No gdb remote provider found.");

  llvm::Optional<std::string> history_file = multi_loader->GetNextFile();
  if (!history_file)
    return Status("No gdb remote packet log found.");

  if (auto error = LoadReplayHistory(FileSpec(*history_file)))
    return Status("Unable to load replay history");

  if (auto error = GDBRemoteCommunication::ConnectLocally(client, *this))
    return Status("Unable to connect to replay server");

  return {};
}
