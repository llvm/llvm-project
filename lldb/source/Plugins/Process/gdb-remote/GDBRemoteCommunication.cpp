//===-- GDBRemoteCommunication.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "GDBRemoteCommunication.h"

// C Includes
#include <limits.h>
#include <string.h>

// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/TimeValue.h"
#include "lldb/Target/Process.h"

// Project includes
#include "ProcessGDBRemoteLog.h"

#define DEBUGSERVER_BASENAME    "debugserver"

using namespace lldb;
using namespace lldb_private;

GDBRemoteCommunication::History::History (uint32_t size) :
    m_packets(),
    m_curr_idx (0),
    m_total_packet_count (0),
    m_dumped_to_log (false)
{
    m_packets.resize(size);
}

GDBRemoteCommunication::History::~History ()
{
}

void
GDBRemoteCommunication::History::AddPacket (char packet_char,
                                            PacketType type,
                                            uint32_t bytes_transmitted)
{
    const size_t size = m_packets.size();
    if (size > 0)
    {
        const uint32_t idx = GetNextIndex();
        m_packets[idx].packet.assign (1, packet_char);
        m_packets[idx].type = type;
        m_packets[idx].bytes_transmitted = bytes_transmitted;
        m_packets[idx].packet_idx = m_total_packet_count;
        m_packets[idx].tid = Host::GetCurrentThreadID();
    }
}

void
GDBRemoteCommunication::History::AddPacket (const std::string &src,
                                            uint32_t src_len,
                                            PacketType type,
                                            uint32_t bytes_transmitted)
{
    const size_t size = m_packets.size();
    if (size > 0)
    {
        const uint32_t idx = GetNextIndex();
        m_packets[idx].packet.assign (src, 0, src_len);
        m_packets[idx].type = type;
        m_packets[idx].bytes_transmitted = bytes_transmitted;
        m_packets[idx].packet_idx = m_total_packet_count;
        m_packets[idx].tid = Host::GetCurrentThreadID();
    }
}

void
GDBRemoteCommunication::History::Dump (lldb_private::Stream &strm) const
{
    const uint32_t size = GetNumPacketsInHistory ();
    const uint32_t first_idx = GetFirstSavedPacketIndex ();
    const uint32_t stop_idx = m_curr_idx + size;
    for (uint32_t i = first_idx;  i < stop_idx; ++i)
    {
        const uint32_t idx = NormalizeIndex (i);
        const Entry &entry = m_packets[idx];
        if (entry.type == ePacketTypeInvalid || entry.packet.empty())
            break;
        strm.Printf ("history[%u] tid=0x%4.4" PRIx64 " <%4u> %s packet: %s\n",
                     entry.packet_idx,
                     entry.tid,
                     entry.bytes_transmitted,
                     (entry.type == ePacketTypeSend) ? "send" : "read",
                     entry.packet.c_str());
    }
}

void
GDBRemoteCommunication::History::Dump (lldb_private::Log *log) const
{
    if (log && !m_dumped_to_log)
    {
        m_dumped_to_log = true;
        const uint32_t size = GetNumPacketsInHistory ();
        const uint32_t first_idx = GetFirstSavedPacketIndex ();
        const uint32_t stop_idx = m_curr_idx + size;
        for (uint32_t i = first_idx;  i < stop_idx; ++i)
        {
            const uint32_t idx = NormalizeIndex (i);
            const Entry &entry = m_packets[idx];
            if (entry.type == ePacketTypeInvalid || entry.packet.empty())
                break;
            log->Printf ("history[%u] tid=0x%4.4" PRIx64 " <%4u> %s packet: %s",
                         entry.packet_idx,
                         entry.tid,
                         entry.bytes_transmitted,
                         (entry.type == ePacketTypeSend) ? "send" : "read",
                         entry.packet.c_str());
        }
    }
}

//----------------------------------------------------------------------
// GDBRemoteCommunication constructor
//----------------------------------------------------------------------
GDBRemoteCommunication::GDBRemoteCommunication(const char *comm_name, 
                                               const char *listener_name, 
                                               bool is_platform) :
    Communication(comm_name),
    m_packet_timeout (1),
    m_sequence_mutex (Mutex::eMutexTypeRecursive),
    m_public_is_running (false),
    m_private_is_running (false),
    m_history (512),
    m_send_acks (true),
    m_is_platform (is_platform)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
GDBRemoteCommunication::~GDBRemoteCommunication()
{
    if (IsConnected())
    {
        Disconnect();
    }
}

char
GDBRemoteCommunication::CalculcateChecksum (const char *payload, size_t payload_length)
{
    int checksum = 0;

    // We only need to compute the checksum if we are sending acks
    if (GetSendAcks ())
    {
        for (size_t i = 0; i < payload_length; ++i)
            checksum += payload[i];
    }
    return checksum & 255;
}

size_t
GDBRemoteCommunication::SendAck ()
{
    Log *log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PACKETS));
    ConnectionStatus status = eConnectionStatusSuccess;
    char ch = '+';
    const size_t bytes_written = Write (&ch, 1, status, NULL);
    if (log)
        log->Printf ("<%4zu> send packet: %c", bytes_written, ch);
    m_history.AddPacket (ch, History::ePacketTypeSend, bytes_written);
    return bytes_written;
}

size_t
GDBRemoteCommunication::SendNack ()
{
    Log *log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PACKETS));
    ConnectionStatus status = eConnectionStatusSuccess;
    char ch = '-';
    const size_t bytes_written = Write (&ch, 1, status, NULL);
    if (log)
        log->Printf ("<%4zu> send packet: %c", bytes_written, ch);
    m_history.AddPacket (ch, History::ePacketTypeSend, bytes_written);
    return bytes_written;
}

size_t
GDBRemoteCommunication::SendPacket (const char *payload, size_t payload_length)
{
    Mutex::Locker locker(m_sequence_mutex);
    return SendPacketNoLock (payload, payload_length);
}

size_t
GDBRemoteCommunication::SendPacketNoLock (const char *payload, size_t payload_length)
{
    if (IsConnected())
    {
        StreamString packet(0, 4, eByteOrderBig);

        packet.PutChar('$');
        packet.Write (payload, payload_length);
        packet.PutChar('#');
        packet.PutHex8(CalculcateChecksum (payload, payload_length));

        Log *log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PACKETS));
        ConnectionStatus status = eConnectionStatusSuccess;
        size_t bytes_written = Write (packet.GetData(), packet.GetSize(), status, NULL);
        if (log)
        {
            // If logging was just enabled and we have history, then dump out what
            // we have to the log so we get the historical context. The Dump() call that
            // logs all of the packet will set a boolean so that we don't dump this more
            // than once
            if (!m_history.DidDumpToLog ())
                m_history.Dump (log);

            log->Printf ("<%4zu> send packet: %.*s", bytes_written, (int)packet.GetSize(), packet.GetData());
        }

        m_history.AddPacket (packet.GetString(), packet.GetSize(), History::ePacketTypeSend, bytes_written);


        if (bytes_written == packet.GetSize())
        {
            if (GetSendAcks ())
            {
                if (GetAck () != '+')
                {
                    if (log)
                        log->Printf("get ack failed...");
                    return 0;
                }
            }
        }
        else
        {
            if (log)
                log->Printf ("error: failed to send packet: %.*s", (int)packet.GetSize(), packet.GetData());
        }
        return bytes_written;
    }
    return 0;
}

char
GDBRemoteCommunication::GetAck ()
{
    StringExtractorGDBRemote packet;
    if (WaitForPacketWithTimeoutMicroSecondsNoLock (packet, GetPacketTimeoutInMicroSeconds ()) == 1)
        return packet.GetChar();
    return 0;
}

bool
GDBRemoteCommunication::GetSequenceMutex (Mutex::Locker& locker, const char *failure_message)
{
    if (IsRunning())
        return locker.TryLock (m_sequence_mutex, failure_message);

    locker.Lock (m_sequence_mutex);
    return true;
}


bool
GDBRemoteCommunication::WaitForNotRunningPrivate (const TimeValue *timeout_ptr)
{
    return m_private_is_running.WaitForValueEqualTo (false, timeout_ptr, NULL);
}

size_t
GDBRemoteCommunication::WaitForPacketWithTimeoutMicroSecondsNoLock (StringExtractorGDBRemote &packet, uint32_t timeout_usec)
{
    uint8_t buffer[8192];
    Error error;

    Log *log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PACKETS | GDBR_LOG_VERBOSE));

    // Check for a packet from our cache first without trying any reading...
    if (CheckForPacket (NULL, 0, packet))
        return packet.GetStringRef().size();

    bool timed_out = false;
    while (IsConnected() && !timed_out)
    {
        lldb::ConnectionStatus status = eConnectionStatusNoConnection;
        size_t bytes_read = Read (buffer, sizeof(buffer), timeout_usec, status, &error);
        
        if (log)
            log->Printf ("%s: Read (buffer, (sizeof(buffer), timeout_usec = 0x%x, status = %s, error = %s) => bytes_read = %" PRIu64,
                         __PRETTY_FUNCTION__,
                         timeout_usec, 
                         Communication::ConnectionStatusAsCString (status),
                         error.AsCString(), 
                         (uint64_t)bytes_read);

        if (bytes_read > 0)
        {
            if (CheckForPacket (buffer, bytes_read, packet))
                return packet.GetStringRef().size();
        }
        else
        {
            switch (status)
            {
            case eConnectionStatusTimedOut:
                timed_out = true;
                break;
            case eConnectionStatusSuccess:
                //printf ("status = success but error = %s\n", error.AsCString("<invalid>"));
                break;
                
            case eConnectionStatusEndOfFile:
            case eConnectionStatusNoConnection:
            case eConnectionStatusLostConnection:
            case eConnectionStatusError:
                Disconnect();
                break;
            }
        }
    }
    packet.Clear ();    
    return 0;
}

bool
GDBRemoteCommunication::CheckForPacket (const uint8_t *src, size_t src_len, StringExtractorGDBRemote &packet)
{
    // Put the packet data into the buffer in a thread safe fashion
    Mutex::Locker locker(m_bytes_mutex);
    
    Log *log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PACKETS));

    if (src && src_len > 0)
    {
        if (log && log->GetVerbose())
        {
            StreamString s;
            log->Printf ("GDBRemoteCommunication::%s adding %u bytes: %.*s",
                         __FUNCTION__, 
                         (uint32_t)src_len, 
                         (uint32_t)src_len, 
                         src);
        }
        m_bytes.append ((const char *)src, src_len);
    }

    // Parse up the packets into gdb remote packets
    if (!m_bytes.empty())
    {
        // end_idx must be one past the last valid packet byte. Start
        // it off with an invalid value that is the same as the current
        // index.
        size_t content_start = 0;
        size_t content_length = 0;
        size_t total_length = 0;
        size_t checksum_idx = std::string::npos;

        switch (m_bytes[0])
        {
            case '+':       // Look for ack
            case '-':       // Look for cancel
            case '\x03':    // ^C to halt target
                content_length = total_length = 1;  // The command is one byte long...
                break;

            case '$':
                // Look for a standard gdb packet?
                {
                    size_t hash_pos = m_bytes.find('#');
                    if (hash_pos != std::string::npos)
                    {
                        if (hash_pos + 2 < m_bytes.size())
                        {
                            checksum_idx = hash_pos + 1;
                            // Skip the dollar sign
                            content_start = 1; 
                            // Don't include the # in the content or the $ in the content length
                            content_length = hash_pos - 1;  
                            
                            total_length = hash_pos + 3; // Skip the # and the two hex checksum bytes
                        }
                        else
                        {
                            // Checksum bytes aren't all here yet
                            content_length = std::string::npos;
                        }
                    }
                }
                break;

            default:
                {
                    // We have an unexpected byte and we need to flush all bad 
                    // data that is in m_bytes, so we need to find the first
                    // byte that is a '+' (ACK), '-' (NACK), \x03 (CTRL+C interrupt),
                    // or '$' character (start of packet header) or of course,
                    // the end of the data in m_bytes...
                    const size_t bytes_len = m_bytes.size();
                    bool done = false;
                    uint32_t idx;
                    for (idx = 1; !done && idx < bytes_len; ++idx)
                    {
                        switch (m_bytes[idx])
                        {
                        case '+':
                        case '-':
                        case '\x03':
                        case '$':
                            done = true;
                            break;
                                
                        default:
                            break;
                        }
                    }
                    if (log)
                        log->Printf ("GDBRemoteCommunication::%s tossing %u junk bytes: '%.*s'",
                                     __FUNCTION__, idx, idx, m_bytes.c_str());
                    m_bytes.erase(0, idx);
                }
                break;
        }

        if (content_length == std::string::npos)
        {
            packet.Clear();
            return false;
        }
        else if (total_length > 0)
        {

            // We have a valid packet...
            assert (content_length <= m_bytes.size());
            assert (total_length <= m_bytes.size());
            assert (content_length <= total_length);
            
            bool success = true;
            std::string &packet_str = packet.GetStringRef();
            
            
            if (log)
            {
                // If logging was just enabled and we have history, then dump out what
                // we have to the log so we get the historical context. The Dump() call that
                // logs all of the packet will set a boolean so that we don't dump this more
                // than once
                if (!m_history.DidDumpToLog ())
                    m_history.Dump (log);
                
                log->Printf ("<%4zu> read packet: %.*s", total_length, (int)(total_length), m_bytes.c_str());
            }

            m_history.AddPacket (m_bytes.c_str(), total_length, History::ePacketTypeRecv, total_length);

            packet_str.assign (m_bytes, content_start, content_length);
            
            if (m_bytes[0] == '$')
            {
                assert (checksum_idx < m_bytes.size());
                if (::isxdigit (m_bytes[checksum_idx+0]) || 
                    ::isxdigit (m_bytes[checksum_idx+1]))
                {
                    if (GetSendAcks ())
                    {
                        const char *packet_checksum_cstr = &m_bytes[checksum_idx];
                        char packet_checksum = strtol (packet_checksum_cstr, NULL, 16);
                        char actual_checksum = CalculcateChecksum (packet_str.c_str(), packet_str.size());
                        success = packet_checksum == actual_checksum;
                        if (!success)
                        {
                            if (log)
                                log->Printf ("error: checksum mismatch: %.*s expected 0x%2.2x, got 0x%2.2x", 
                                             (int)(total_length), 
                                             m_bytes.c_str(),
                                             (uint8_t)packet_checksum,
                                             (uint8_t)actual_checksum);
                        }
                        // Send the ack or nack if needed
                        if (!success)
                            SendNack();
                        else
                            SendAck();
                    }
                }
                else
                {
                    success = false;
                    if (log)
                        log->Printf ("error: invalid checksum in packet: '%s'\n", m_bytes.c_str());
                }
            }
            
            m_bytes.erase(0, total_length);
            packet.SetFilePos(0);
            return success;
        }
    }
    packet.Clear();
    return false;
}

Error
GDBRemoteCommunication::StartDebugserverProcess (const char *debugserver_url,
                                                 const char *unix_socket_name,  // For handshaking
                                                 lldb_private::ProcessLaunchInfo &launch_info)
{
    Error error;
    // If we locate debugserver, keep that located version around
    static FileSpec g_debugserver_file_spec;
    
    // This function will fill in the launch information for the debugserver
    // instance that gets launched.
    launch_info.Clear();
    
    char debugserver_path[PATH_MAX];
    FileSpec &debugserver_file_spec = launch_info.GetExecutableFile();
    
    // Always check to see if we have an environment override for the path
    // to the debugserver to use and use it if we do.
    const char *env_debugserver_path = getenv("LLDB_DEBUGSERVER_PATH");
    if (env_debugserver_path)
        debugserver_file_spec.SetFile (env_debugserver_path, false);
    else
        debugserver_file_spec = g_debugserver_file_spec;
    bool debugserver_exists = debugserver_file_spec.Exists();
    if (!debugserver_exists)
    {
        // The debugserver binary is in the LLDB.framework/Resources
        // directory. 
        if (Host::GetLLDBPath (ePathTypeSupportExecutableDir, debugserver_file_spec))
        {
            debugserver_file_spec.GetFilename().SetCString(DEBUGSERVER_BASENAME);
            debugserver_exists = debugserver_file_spec.Exists();
            if (debugserver_exists)
            {
                g_debugserver_file_spec = debugserver_file_spec;
            }
            else
            {
                g_debugserver_file_spec.Clear();
                debugserver_file_spec.Clear();
            }
        }
    }
    
    if (debugserver_exists)
    {
        debugserver_file_spec.GetPath (debugserver_path, sizeof(debugserver_path));

        Args &debugserver_args = launch_info.GetArguments();
        debugserver_args.Clear();
        char arg_cstr[PATH_MAX];
        
        // Start args with "debugserver /file/path -r --"
        debugserver_args.AppendArgument(debugserver_path);
        debugserver_args.AppendArgument(debugserver_url);
        // use native registers, not the GDB registers
        debugserver_args.AppendArgument("--native-regs");   
        // make debugserver run in its own session so signals generated by 
        // special terminal key sequences (^C) don't affect debugserver
        debugserver_args.AppendArgument("--setsid");
        
        if (unix_socket_name && unix_socket_name[0])
        {
            debugserver_args.AppendArgument("--unix-socket");
            debugserver_args.AppendArgument(unix_socket_name);
        }

        const char *env_debugserver_log_file = getenv("LLDB_DEBUGSERVER_LOG_FILE");
        if (env_debugserver_log_file)
        {
            ::snprintf (arg_cstr, sizeof(arg_cstr), "--log-file=%s", env_debugserver_log_file);
            debugserver_args.AppendArgument(arg_cstr);
        }
        
        const char *env_debugserver_log_flags = getenv("LLDB_DEBUGSERVER_LOG_FLAGS");
        if (env_debugserver_log_flags)
        {
            ::snprintf (arg_cstr, sizeof(arg_cstr), "--log-flags=%s", env_debugserver_log_flags);
            debugserver_args.AppendArgument(arg_cstr);
        }
        //            debugserver_args.AppendArgument("--log-file=/tmp/debugserver.txt");
        //            debugserver_args.AppendArgument("--log-flags=0x802e0e");
        
        // We currently send down all arguments, attach pids, or attach 
        // process names in dedicated GDB server packets, so we don't need
        // to pass them as arguments. This is currently because of all the
        // things we need to setup prior to launching: the environment,
        // current working dir, file actions, etc.
#if 0
        // Now append the program arguments
        if (inferior_argv)
        {
            // Terminate the debugserver args so we can now append the inferior args
            debugserver_args.AppendArgument("--");
            
            for (int i = 0; inferior_argv[i] != NULL; ++i)
                debugserver_args.AppendArgument (inferior_argv[i]);
        }
        else if (attach_pid != LLDB_INVALID_PROCESS_ID)
        {
            ::snprintf (arg_cstr, sizeof(arg_cstr), "--attach=%u", attach_pid);
            debugserver_args.AppendArgument (arg_cstr);
        }
        else if (attach_name && attach_name[0])
        {
            if (wait_for_launch)
                debugserver_args.AppendArgument ("--waitfor");
            else
                debugserver_args.AppendArgument ("--attach");
            debugserver_args.AppendArgument (attach_name);
        }
#endif
        
        // Close STDIN, STDOUT and STDERR. We might need to redirect them
        // to "/dev/null" if we run into any problems.
//        launch_info.AppendCloseFileAction (STDIN_FILENO);
//        launch_info.AppendCloseFileAction (STDOUT_FILENO);
//        launch_info.AppendCloseFileAction (STDERR_FILENO);
        
        error = Host::LaunchProcess(launch_info);
    }
    else
    {
        error.SetErrorStringWithFormat ("unable to locate " DEBUGSERVER_BASENAME );
    }
    return error;
}

void
GDBRemoteCommunication::DumpHistory(Stream &strm)
{
    m_history.Dump (strm);
}
