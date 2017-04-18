//===-- GDBRemoteCommunication.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_GDBRemoteCommunication_h_
#define liblldb_GDBRemoteCommunication_h_

// C Includes
// C++ Includes
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/Communication.h"
#include "lldb/Core/Listener.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/Predicate.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/lldb-public.h"

#include "Utility/StringExtractorGDBRemote.h"

namespace lldb_private {
namespace process_gdb_remote {

typedef enum {
  eStoppointInvalid = -1,
  eBreakpointSoftware = 0,
  eBreakpointHardware,
  eWatchpointWrite,
  eWatchpointRead,
  eWatchpointReadWrite
} GDBStoppointType;

enum class CompressionType {
  None = 0,    // no compression
  ZlibDeflate, // zlib's deflate compression scheme, requires zlib or Apple's
               // libcompression
  LZFSE,       // an Apple compression scheme, requires Apple's libcompression
  LZ4, // lz compression - called "lz4 raw" in libcompression terms, compat with
       // https://code.google.com/p/lz4/
  LZMA, // Lempel–Ziv–Markov chain algorithm
};

class ProcessGDBRemote;

class GDBRemoteCommunication : public Communication {
public:
  enum {
    eBroadcastBitRunPacketSent = kLoUserBroadcastBit,
    eBroadcastBitGdbReadThreadGotNotify =
        kLoUserBroadcastBit << 1 // Sent when we received a notify packet.
  };

  enum class PacketType { Invalid = 0, Standard, Notify };

  enum class PacketResult {
    Success = 0,        // Success
    ErrorSendFailed,    // Error sending the packet
    ErrorSendAck,       // Didn't get an ack back after sending a packet
    ErrorReplyFailed,   // Error getting the reply
    ErrorReplyTimeout,  // Timed out waiting for reply
    ErrorReplyInvalid,  // Got a reply but it wasn't valid for the packet that
                        // was sent
    ErrorReplyAck,      // Sending reply ack failed
    ErrorDisconnected,  // We were disconnected
    ErrorNoSequenceLock // We couldn't get the sequence lock for a multi-packet
                        // request
  };

  // Class to change the timeout for a given scope and restore it to the
  // original value when the
  // created ScopedTimeout object got out of scope
  class ScopedTimeout {
  public:
    ScopedTimeout(GDBRemoteCommunication &gdb_comm,
                  std::chrono::seconds timeout);
    ~ScopedTimeout();

  private:
    GDBRemoteCommunication &m_gdb_comm;
    std::chrono::seconds m_saved_timeout;
    // Don't ever reduce the timeout for a packet, only increase it. If the
    // requested timeout if less than the current timeout, we don't set it
    // and won't need to restore it.
    bool m_timeout_modified;
  };

  GDBRemoteCommunication(const char *comm_name, const char *listener_name);

  ~GDBRemoteCommunication() override;

  PacketResult GetAck();

  size_t SendAck();

  size_t SendNack();

  char CalculcateChecksum(llvm::StringRef payload);

  PacketType CheckForPacket(const uint8_t *src, size_t src_len,
                            StringExtractorGDBRemote &packet);

  bool GetSendAcks() { return m_send_acks; }

  //------------------------------------------------------------------
  // Set the global packet timeout.
  //
  // For clients, this is the timeout that gets used when sending
  // packets and waiting for responses. For servers, this is used when waiting
  // for ACKs.
  //------------------------------------------------------------------
  std::chrono::seconds SetPacketTimeout(std::chrono::seconds packet_timeout) {
    const auto old_packet_timeout = m_packet_timeout;
    m_packet_timeout = packet_timeout;
    return old_packet_timeout;
  }

  std::chrono::seconds GetPacketTimeout() const { return m_packet_timeout; }

  //------------------------------------------------------------------
  // Start a debugserver instance on the current host using the
  // supplied connection URL.
  //------------------------------------------------------------------
  Error StartDebugserverProcess(
      const char *url,
      Platform *platform, // If non nullptr, then check with the platform for
                          // the GDB server binary if it can't be located
      ProcessLaunchInfo &launch_info, uint16_t *port, const Args *inferior_args,
      int pass_comm_fd); // Communication file descriptor to pass during
                         // fork/exec to avoid having to connect/accept

  void DumpHistory(Stream &strm);

protected:
  class History {
  public:
    enum PacketType {
      ePacketTypeInvalid = 0,
      ePacketTypeSend,
      ePacketTypeRecv
    };

    struct Entry {
      Entry()
          : packet(), type(ePacketTypeInvalid), bytes_transmitted(0),
            packet_idx(0), tid(LLDB_INVALID_THREAD_ID) {}

      void Clear() {
        packet.clear();
        type = ePacketTypeInvalid;
        bytes_transmitted = 0;
        packet_idx = 0;
        tid = LLDB_INVALID_THREAD_ID;
      }
      std::string packet;
      PacketType type;
      uint32_t bytes_transmitted;
      uint32_t packet_idx;
      lldb::tid_t tid;
    };

    History(uint32_t size);

    ~History();

    // For single char packets for ack, nack and /x03
    void AddPacket(char packet_char, PacketType type,
                   uint32_t bytes_transmitted);

    void AddPacket(const std::string &src, uint32_t src_len, PacketType type,
                   uint32_t bytes_transmitted);

    void Dump(Stream &strm) const;

    void Dump(Log *log) const;

    bool DidDumpToLog() const { return m_dumped_to_log; }

  protected:
    uint32_t GetFirstSavedPacketIndex() const {
      if (m_total_packet_count < m_packets.size())
        return 0;
      else
        return m_curr_idx + 1;
    }

    uint32_t GetNumPacketsInHistory() const {
      if (m_total_packet_count < m_packets.size())
        return m_total_packet_count;
      else
        return (uint32_t)m_packets.size();
    }

    uint32_t GetNextIndex() {
      ++m_total_packet_count;
      const uint32_t idx = m_curr_idx;
      m_curr_idx = NormalizeIndex(idx + 1);
      return idx;
    }

    uint32_t NormalizeIndex(uint32_t i) const { return i % m_packets.size(); }

    std::vector<Entry> m_packets;
    uint32_t m_curr_idx;
    uint32_t m_total_packet_count;
    mutable bool m_dumped_to_log;
  };

  std::chrono::seconds m_packet_timeout;
  uint32_t m_echo_number;
  LazyBool m_supports_qEcho;
  History m_history;
  bool m_send_acks;
  bool m_is_platform; // Set to true if this class represents a platform,
                      // false if this class represents a debug session for
                      // a single process

  CompressionType m_compression_type;

  PacketResult SendPacketNoLock(llvm::StringRef payload);

  PacketResult ReadPacket(StringExtractorGDBRemote &response,
                          Timeout<std::micro> timeout, bool sync_on_timeout);

  // Pop a packet from the queue in a thread safe manner
  PacketResult PopPacketFromQueue(StringExtractorGDBRemote &response,
                                  Timeout<std::micro> timeout);

  PacketResult WaitForPacketNoLock(StringExtractorGDBRemote &response,
                                   Timeout<std::micro> timeout,
                                   bool sync_on_timeout);

  bool CompressionIsEnabled() {
    return m_compression_type != CompressionType::None;
  }

  // If compression is enabled, decompress the packet in m_bytes and update
  // m_bytes with the uncompressed version.
  // Returns 'true' packet was decompressed and m_bytes is the now-decompressed
  // text.
  // Returns 'false' if unable to decompress or if the checksum was invalid.
  //
  // NB: Once the packet has been decompressed, checksum cannot be computed
  // based
  // on m_bytes.  The checksum was for the compressed packet.
  bool DecompressPacket();

  Error StartListenThread(const char *hostname = "127.0.0.1",
                          uint16_t port = 0);

  bool JoinListenThread();

  static lldb::thread_result_t ListenThread(lldb::thread_arg_t arg);

  // GDB-Remote read thread
  //  . this thread constantly tries to read from the communication
  //    class and stores all packets received in a queue.  The usual
  //    threads read requests simply pop packets off the queue in the
  //    usual order.
  //    This setup allows us to intercept and handle async packets, such
  //    as the notify packet.

  // This method is defined as part of communication.h
  // when the read thread gets any bytes it will pass them on to this function
  void AppendBytesToCache(const uint8_t *bytes, size_t len, bool broadcast,
                          lldb::ConnectionStatus status) override;

private:
  std::queue<StringExtractorGDBRemote> m_packet_queue; // The packet queue
  std::mutex m_packet_queue_mutex; // Mutex for accessing queue
  std::condition_variable
      m_condition_queue_not_empty; // Condition variable to wait for packets

  HostThread m_listen_thread;
  std::string m_listen_url;

  DISALLOW_COPY_AND_ASSIGN(GDBRemoteCommunication);
};

} // namespace process_gdb_remote
} // namespace lldb_private

#endif // liblldb_GDBRemoteCommunication_h_
