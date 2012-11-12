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
#include <list>
#include <string>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-public.h"
#include "lldb/Core/Communication.h"
#include "lldb/Core/Listener.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Host/Predicate.h"
#include "lldb/Host/TimeValue.h"

#include "Utility/StringExtractorGDBRemote.h"

class ProcessGDBRemote;

class GDBRemoteCommunication : public lldb_private::Communication
{
public:
    enum
    {
        eBroadcastBitRunPacketSent = kLoUserBroadcastBit
    };
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    GDBRemoteCommunication(const char *comm_name, 
                           const char *listener_name,
                           bool is_platform);

    virtual
    ~GDBRemoteCommunication();

    char
    GetAck ();

    size_t
    SendAck ();

    size_t
    SendNack ();

    char
    CalculcateChecksum (const char *payload,
                        size_t payload_length);

    bool
    GetSequenceMutex (lldb_private::Mutex::Locker& locker, const char *failure_message = NULL);

    bool
    CheckForPacket (const uint8_t *src, 
                    size_t src_len, 
                    StringExtractorGDBRemote &packet);
    bool
    IsRunning() const
    {
        return m_public_is_running.GetValue();
    }

    bool
    GetSendAcks ()
    {
        return m_send_acks;
    }

    //------------------------------------------------------------------
    // Client and server must implement these pure virtual functions
    //------------------------------------------------------------------
    virtual bool
    GetThreadSuffixSupported () = 0;

    //------------------------------------------------------------------
    // Set the global packet timeout.
    //
    // For clients, this is the timeout that gets used when sending
    // packets and waiting for responses. For servers, this might not
    // get used, and if it doesn't this should be moved to the
    // GDBRemoteCommunicationClient.
    //------------------------------------------------------------------
    uint32_t 
    SetPacketTimeout (uint32_t packet_timeout)
    {
        const uint32_t old_packet_timeout = m_packet_timeout;
        m_packet_timeout = packet_timeout;
        return old_packet_timeout;
    }

    uint32_t
    GetPacketTimeoutInMicroSeconds () const
    {
        return m_packet_timeout * lldb_private::TimeValue::MicroSecPerSec;
    }
    //------------------------------------------------------------------
    // Start a debugserver instance on the current host using the
    // supplied connection URL.
    //------------------------------------------------------------------
    lldb_private::Error
    StartDebugserverProcess (const char *connect_url,
                             const char *unix_socket_name,
                             lldb_private::ProcessLaunchInfo &launch_info); 

    void
    DumpHistory(lldb_private::Stream &strm);
    
protected:

    class History
    {
    public:
        enum PacketType
        {
            ePacketTypeInvalid = 0,
            ePacketTypeSend,
            ePacketTypeRecv
        };

        struct Entry
        {
            Entry() :
                packet(),
                type (ePacketTypeInvalid),
                bytes_transmitted (0),
                packet_idx (0),
                tid (LLDB_INVALID_THREAD_ID)
            {
            }
            
            void
            Clear ()
            {
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

        History (uint32_t size);
        
        ~History ();

        // For single char packets for ack, nack and /x03
        void
        AddPacket (char packet_char,
                   PacketType type,
                   uint32_t bytes_transmitted);
        void
        AddPacket (const std::string &src,
                   uint32_t src_len,
                   PacketType type,
                   uint32_t bytes_transmitted);
        
        void
        Dump (lldb_private::Stream &strm) const;

        void
        Dump (lldb_private::Log *log) const;

        bool
        DidDumpToLog () const
        {
            return m_dumped_to_log;
        }
    
protected:
        uint32_t
        GetFirstSavedPacketIndex () const
        {
            if (m_total_packet_count < m_packets.size())
                return 0;
            else
                return m_curr_idx + 1;
        }

        uint32_t
        GetNumPacketsInHistory () const
        {
            if (m_total_packet_count < m_packets.size())
                return m_total_packet_count;
            else
                return (uint32_t)m_packets.size();
        }

        uint32_t
        GetNextIndex()
        {
            ++m_total_packet_count;
            const uint32_t idx = m_curr_idx;
            m_curr_idx = NormalizeIndex(idx + 1);
            return idx;
        }

        uint32_t
        NormalizeIndex (uint32_t i) const
        {
            return i % m_packets.size();
        }

        
        std::vector<Entry> m_packets;
        uint32_t m_curr_idx;
        uint32_t m_total_packet_count;
        mutable bool m_dumped_to_log;
    };

    size_t
    SendPacket (const char *payload,
                size_t payload_length);

    size_t
    SendPacketNoLock (const char *payload, 
                      size_t payload_length);

    size_t
    WaitForPacketWithTimeoutMicroSecondsNoLock (StringExtractorGDBRemote &response, 
                                                uint32_t timeout_usec);

    bool
    WaitForNotRunningPrivate (const lldb_private::TimeValue *timeout_ptr);

    //------------------------------------------------------------------
    // Classes that inherit from GDBRemoteCommunication can see and modify these
    //------------------------------------------------------------------
    uint32_t m_packet_timeout;
#ifdef LLDB_CONFIGURATION_DEBUG
    lldb_private::TrackingMutex m_sequence_mutex;
#else
    lldb_private::Mutex m_sequence_mutex;    // Restrict access to sending/receiving packets to a single thread at a time
#endif
    lldb_private::Predicate<bool> m_public_is_running;
    lldb_private::Predicate<bool> m_private_is_running;
    History m_history;
    bool m_send_acks;
    bool m_is_platform; // Set to true if this class represents a platform,
                        // false if this class represents a debug session for
                        // a single process
    



private:
    //------------------------------------------------------------------
    // For GDBRemoteCommunication only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (GDBRemoteCommunication);
};

#endif  // liblldb_GDBRemoteCommunication_h_
