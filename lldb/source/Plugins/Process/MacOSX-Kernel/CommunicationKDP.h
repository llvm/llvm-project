//===-- CommunicationKDP.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommunicationKDP_h_
#define liblldb_CommunicationKDP_h_

// C Includes
// C++ Includes
#include <list>
#include <string>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/Communication.h"
#include "lldb/Core/Listener.h"
#include "lldb/Core/StreamBuffer.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Host/Predicate.h"
#include "lldb/Host/TimeValue.h"
#include "lldb/lldb-private.h"

class CommunicationKDP : public lldb_private::Communication {
public:
  enum { eBroadcastBitRunPacketSent = kLoUserBroadcastBit };

  const static uint32_t kMaxPacketSize = 1200;
  const static uint32_t kMaxDataSize = 1024;
  typedef lldb_private::StreamBuffer<1024> PacketStreamType;
  typedef enum {
    KDP_CONNECT = 0u,
    KDP_DISCONNECT,
    KDP_HOSTINFO,
    KDP_VERSION,
    KDP_MAXBYTES,
    KDP_READMEM,
    KDP_WRITEMEM,
    KDP_READREGS,
    KDP_WRITEREGS,
    KDP_LOAD,
    KDP_IMAGEPATH,
    KDP_SUSPEND,
    KDP_RESUMECPUS,
    KDP_EXCEPTION,
    KDP_TERMINATION,
    KDP_BREAKPOINT_SET,
    KDP_BREAKPOINT_REMOVE,
    KDP_REGIONS,
    KDP_REATTACH,
    KDP_HOSTREBOOT,
    KDP_READMEM64,
    KDP_WRITEMEM64,
    KDP_BREAKPOINT_SET64,
    KDP_BREAKPOINT_REMOVE64,
    KDP_KERNELVERSION,
    KDP_READPHYSMEM64,
    KDP_WRITEPHYSMEM64,
    KDP_READIOPORT,
    KDP_WRITEIOPORT,
    KDP_READMSR64,
    KDP_WRITEMSR64,
    KDP_DUMPINFO
  } CommandType;

  enum { KDP_FEATURE_BP = (1u << 0) };

  typedef enum {
    KDP_PROTERR_SUCCESS = 0,
    KDP_PROTERR_ALREADY_CONNECTED,
    KDP_PROTERR_BAD_NBYTES,
    KDP_PROTERR_BADFLAVOR
  } KDPError;

  typedef enum {
    ePacketTypeRequest = 0x00u,
    ePacketTypeReply = 0x80u,
    ePacketTypeMask = 0x80u,
    eCommandTypeMask = 0x7fu
  } PacketType;
  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  CommunicationKDP(const char *comm_name);

  virtual ~CommunicationKDP();

  bool SendRequestPacket(const PacketStreamType &request_packet);

  // Wait for a packet within 'nsec' seconds
  size_t
  WaitForPacketWithTimeoutMicroSeconds(lldb_private::DataExtractor &response,
                                       uint32_t usec);

  bool GetSequenceMutex(lldb_private::Mutex::Locker &locker);

  bool CheckForPacket(const uint8_t *src, size_t src_len,
                      lldb_private::DataExtractor &packet);
  bool IsRunning() const { return m_is_running.GetValue(); }

  //------------------------------------------------------------------
  // Set the global packet timeout.
  //
  // For clients, this is the timeout that gets used when sending
  // packets and waiting for responses. For servers, this might not
  // get used, and if it doesn't this should be moved to the
  // CommunicationKDPClient.
  //------------------------------------------------------------------
  uint32_t SetPacketTimeout(uint32_t packet_timeout) {
    const uint32_t old_packet_timeout = m_packet_timeout;
    m_packet_timeout = packet_timeout;
    return old_packet_timeout;
  }

  uint32_t GetPacketTimeoutInMicroSeconds() const {
    return m_packet_timeout * lldb_private::TimeValue::MicroSecPerSec;
  }

  //------------------------------------------------------------------
  // Public Request Packets
  //------------------------------------------------------------------
  bool SendRequestConnect(uint16_t reply_port, uint16_t exc_port,
                          const char *greeting);

  bool SendRequestReattach(uint16_t reply_port);

  bool SendRequestDisconnect();

  uint32_t SendRequestReadMemory(lldb::addr_t addr, void *dst,
                                 uint32_t dst_size, lldb_private::Error &error);

  uint32_t SendRequestWriteMemory(lldb::addr_t addr, const void *src,
                                  uint32_t src_len, lldb_private::Error &error);

  bool SendRawRequest(uint8_t command_byte, const void *src, uint32_t src_len,
                      lldb_private::DataExtractor &reply,
                      lldb_private::Error &error);

  uint32_t SendRequestReadRegisters(uint32_t cpu, uint32_t flavor, void *dst,
                                    uint32_t dst_size,
                                    lldb_private::Error &error);

  uint32_t SendRequestWriteRegisters(uint32_t cpu, uint32_t flavor,
                                     const void *src, uint32_t src_size,
                                     lldb_private::Error &error);

  const char *GetKernelVersion();

  // Disable KDP_IMAGEPATH for now, it seems to hang the KDP connection...
  // const char *
  // GetImagePath ();

  uint32_t GetVersion();

  uint32_t GetFeatureFlags();

  bool LocalBreakpointsAreSupported() {
    return (GetFeatureFlags() & KDP_FEATURE_BP) != 0;
  }

  uint32_t GetCPUMask();

  uint32_t GetCPUType();

  uint32_t GetCPUSubtype();

  lldb_private::UUID GetUUID();

  bool RemoteIsEFI();

  bool RemoteIsDarwinKernel();

  lldb::addr_t GetLoadAddress();

  bool SendRequestResume();

  bool SendRequestSuspend();

  bool SendRequestBreakpoint(bool set, lldb::addr_t addr);

protected:
  bool SendRequestPacketNoLock(const PacketStreamType &request_packet);

  size_t WaitForPacketWithTimeoutMicroSecondsNoLock(
      lldb_private::DataExtractor &response, uint32_t timeout_usec);

  bool WaitForNotRunningPrivate(const lldb_private::TimeValue *timeout_ptr);

  void MakeRequestPacketHeader(CommandType request_type,
                               PacketStreamType &request_packet,
                               uint16_t request_length);

  //------------------------------------------------------------------
  // Protected Request Packets (use public accessors which will cache
  // results.
  //------------------------------------------------------------------
  bool SendRequestVersion();

  bool SendRequestHostInfo();

  bool SendRequestKernelVersion();

  // Disable KDP_IMAGEPATH for now, it seems to hang the KDP connection...
  // bool
  // SendRequestImagePath ();

  void DumpPacket(lldb_private::Stream &s, const void *data, uint32_t data_len);

  void DumpPacket(lldb_private::Stream &s,
                  const lldb_private::DataExtractor &extractor);

  bool VersionIsValid() const { return m_kdp_version_version != 0; }

  bool HostInfoIsValid() const { return m_kdp_hostinfo_cpu_type != 0; }

  bool ExtractIsReply(uint8_t first_packet_byte) const {
    // TODO: handle big endian...
    return (first_packet_byte & ePacketTypeMask) != 0;
  }

  CommandType ExtractCommand(uint8_t first_packet_byte) const {
    // TODO: handle big endian...
    return (CommandType)(first_packet_byte & eCommandTypeMask);
  }

  static const char *GetCommandAsCString(uint8_t command);

  void ClearKDPSettings();

  bool SendRequestAndGetReply(const CommandType command,
                              const PacketStreamType &request_packet,
                              lldb_private::DataExtractor &reply_packet);
  //------------------------------------------------------------------
  // Classes that inherit from CommunicationKDP can see and modify these
  //------------------------------------------------------------------
  uint32_t m_addr_byte_size;
  lldb::ByteOrder m_byte_order;
  uint32_t m_packet_timeout;
  lldb_private::Mutex m_sequence_mutex; // Restrict access to sending/receiving
                                        // packets to a single thread at a time
  lldb_private::Predicate<bool> m_is_running;
  uint32_t m_session_key;
  uint8_t m_request_sequence_id;
  uint8_t m_exception_sequence_id;
  uint32_t m_kdp_version_version;
  uint32_t m_kdp_version_feature;
  uint32_t m_kdp_hostinfo_cpu_mask;
  uint32_t m_kdp_hostinfo_cpu_type;
  uint32_t m_kdp_hostinfo_cpu_subtype;
  std::string m_kernel_version;
  // std::string m_image_path; // Disable KDP_IMAGEPATH for now, it seems to
  // hang the KDP connection...
  lldb::addr_t m_last_read_memory_addr; // Last memory read address for logging
private:
  //------------------------------------------------------------------
  // For CommunicationKDP only
  //------------------------------------------------------------------
  DISALLOW_COPY_AND_ASSIGN(CommunicationKDP);
};

#endif // liblldb_CommunicationKDP_h_
