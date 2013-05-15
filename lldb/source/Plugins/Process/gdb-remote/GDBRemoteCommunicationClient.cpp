//===-- GDBRemoteCommunicationClient.cpp ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "GDBRemoteCommunicationClient.h"

// C Includes
// C++ Includes
#include <sstream>

// Other libraries and framework includes
#include "llvm/ADT/Triple.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/ConnectionFileDescriptor.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/State.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/Endian.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/TimeValue.h"

// Project includes
#include "Utility/StringExtractorGDBRemote.h"
#include "ProcessGDBRemote.h"
#include "ProcessGDBRemoteLog.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// GDBRemoteCommunicationClient constructor
//----------------------------------------------------------------------
GDBRemoteCommunicationClient::GDBRemoteCommunicationClient(bool is_platform) :
    GDBRemoteCommunication("gdb-remote.client", "gdb-remote.client.rx_packet", is_platform),
    m_supports_not_sending_acks (eLazyBoolCalculate),
    m_supports_thread_suffix (eLazyBoolCalculate),
    m_supports_threads_in_stop_reply (eLazyBoolCalculate),
    m_supports_vCont_all (eLazyBoolCalculate),
    m_supports_vCont_any (eLazyBoolCalculate),
    m_supports_vCont_c (eLazyBoolCalculate),
    m_supports_vCont_C (eLazyBoolCalculate),
    m_supports_vCont_s (eLazyBoolCalculate),
    m_supports_vCont_S (eLazyBoolCalculate),
    m_qHostInfo_is_valid (eLazyBoolCalculate),
    m_qProcessInfo_is_valid (eLazyBoolCalculate),
    m_supports_alloc_dealloc_memory (eLazyBoolCalculate),
    m_supports_memory_region_info  (eLazyBoolCalculate),
    m_supports_watchpoint_support_info  (eLazyBoolCalculate),
    m_supports_detach_stay_stopped (eLazyBoolCalculate),
    m_watchpoints_trigger_after_instruction(eLazyBoolCalculate),
    m_attach_or_wait_reply(eLazyBoolCalculate),
    m_prepare_for_reg_writing_reply (eLazyBoolCalculate),
    m_supports_qProcessInfoPID (true),
    m_supports_qfProcessInfo (true),
    m_supports_qUserName (true),
    m_supports_qGroupName (true),
    m_supports_qThreadStopInfo (true),
    m_supports_z0 (true),
    m_supports_z1 (true),
    m_supports_z2 (true),
    m_supports_z3 (true),
    m_supports_z4 (true),
    m_curr_tid (LLDB_INVALID_THREAD_ID),
    m_curr_tid_run (LLDB_INVALID_THREAD_ID),
    m_num_supported_hardware_watchpoints (0),
    m_async_mutex (Mutex::eMutexTypeRecursive),
    m_async_packet_predicate (false),
    m_async_packet (),
    m_async_response (),
    m_async_signal (-1),
    m_thread_id_to_used_usec_map (),
    m_host_arch(),
    m_process_arch(),
    m_os_version_major (UINT32_MAX),
    m_os_version_minor (UINT32_MAX),
    m_os_version_update (UINT32_MAX)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
GDBRemoteCommunicationClient::~GDBRemoteCommunicationClient()
{
    if (IsConnected())
        Disconnect();
}

bool
GDBRemoteCommunicationClient::HandshakeWithServer (Error *error_ptr)
{
    // Start the read thread after we send the handshake ack since if we
    // fail to send the handshake ack, there is no reason to continue...
    if (SendAck())
        return true;
    
    if (error_ptr)
        error_ptr->SetErrorString("failed to send the handshake ack");
    return false;
}

void
GDBRemoteCommunicationClient::QueryNoAckModeSupported ()
{
    if (m_supports_not_sending_acks == eLazyBoolCalculate)
    {
        m_send_acks = true;
        m_supports_not_sending_acks = eLazyBoolNo;

        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse("QStartNoAckMode", response, false))
        {
            if (response.IsOKResponse())
            {
                m_send_acks = false;
                m_supports_not_sending_acks = eLazyBoolYes;
            }
        }
    }
}

void
GDBRemoteCommunicationClient::GetListThreadsInStopReplySupported ()
{
    if (m_supports_threads_in_stop_reply == eLazyBoolCalculate)
    {
        m_supports_threads_in_stop_reply = eLazyBoolNo;
        
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse("QListThreadsInStopReply", response, false))
        {
            if (response.IsOKResponse())
                m_supports_threads_in_stop_reply = eLazyBoolYes;
        }
    }
}

bool
GDBRemoteCommunicationClient::GetVAttachOrWaitSupported ()
{
    if (m_attach_or_wait_reply == eLazyBoolCalculate)
    {
        m_attach_or_wait_reply = eLazyBoolNo;
        
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse("qVAttachOrWaitSupported", response, false))
        {
            if (response.IsOKResponse())
                m_attach_or_wait_reply = eLazyBoolYes;
        }
    }
    if (m_attach_or_wait_reply == eLazyBoolYes)
        return true;
    else
        return false;
}

bool
GDBRemoteCommunicationClient::GetSyncThreadStateSupported ()
{
    if (m_prepare_for_reg_writing_reply == eLazyBoolCalculate)
    {
        m_prepare_for_reg_writing_reply = eLazyBoolNo;
        
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse("qSyncThreadStateSupported", response, false))
        {
            if (response.IsOKResponse())
                m_prepare_for_reg_writing_reply = eLazyBoolYes;
        }
    }
    if (m_prepare_for_reg_writing_reply == eLazyBoolYes)
        return true;
    else
        return false;
}


void
GDBRemoteCommunicationClient::ResetDiscoverableSettings()
{
    m_supports_not_sending_acks = eLazyBoolCalculate;
    m_supports_thread_suffix = eLazyBoolCalculate;
    m_supports_threads_in_stop_reply = eLazyBoolCalculate;
    m_supports_vCont_c = eLazyBoolCalculate;
    m_supports_vCont_C = eLazyBoolCalculate;
    m_supports_vCont_s = eLazyBoolCalculate;
    m_supports_vCont_S = eLazyBoolCalculate;
    m_qHostInfo_is_valid = eLazyBoolCalculate;
    m_qProcessInfo_is_valid = eLazyBoolCalculate;
    m_supports_alloc_dealloc_memory = eLazyBoolCalculate;
    m_supports_memory_region_info = eLazyBoolCalculate;
    m_prepare_for_reg_writing_reply = eLazyBoolCalculate;
    m_attach_or_wait_reply = eLazyBoolCalculate;

    m_supports_qProcessInfoPID = true;
    m_supports_qfProcessInfo = true;
    m_supports_qUserName = true;
    m_supports_qGroupName = true;
    m_supports_qThreadStopInfo = true;
    m_supports_z0 = true;
    m_supports_z1 = true;
    m_supports_z2 = true;
    m_supports_z3 = true;
    m_supports_z4 = true;
    m_host_arch.Clear();
    m_process_arch.Clear();
}


bool
GDBRemoteCommunicationClient::GetThreadSuffixSupported ()
{
    if (m_supports_thread_suffix == eLazyBoolCalculate)
    {
        StringExtractorGDBRemote response;
        m_supports_thread_suffix = eLazyBoolNo;
        if (SendPacketAndWaitForResponse("QThreadSuffixSupported", response, false))
        {
            if (response.IsOKResponse())
                m_supports_thread_suffix = eLazyBoolYes;
        }
    }
    return m_supports_thread_suffix;
}
bool
GDBRemoteCommunicationClient::GetVContSupported (char flavor)
{
    if (m_supports_vCont_c == eLazyBoolCalculate)
    {
        StringExtractorGDBRemote response;
        m_supports_vCont_any = eLazyBoolNo;
        m_supports_vCont_all = eLazyBoolNo;
        m_supports_vCont_c = eLazyBoolNo;
        m_supports_vCont_C = eLazyBoolNo;
        m_supports_vCont_s = eLazyBoolNo;
        m_supports_vCont_S = eLazyBoolNo;
        if (SendPacketAndWaitForResponse("vCont?", response, false))
        {
            const char *response_cstr = response.GetStringRef().c_str();
            if (::strstr (response_cstr, ";c"))
                m_supports_vCont_c = eLazyBoolYes;

            if (::strstr (response_cstr, ";C"))
                m_supports_vCont_C = eLazyBoolYes;

            if (::strstr (response_cstr, ";s"))
                m_supports_vCont_s = eLazyBoolYes;

            if (::strstr (response_cstr, ";S"))
                m_supports_vCont_S = eLazyBoolYes;

            if (m_supports_vCont_c == eLazyBoolYes &&
                m_supports_vCont_C == eLazyBoolYes &&
                m_supports_vCont_s == eLazyBoolYes &&
                m_supports_vCont_S == eLazyBoolYes)
            {
                m_supports_vCont_all = eLazyBoolYes;
            }
            
            if (m_supports_vCont_c == eLazyBoolYes ||
                m_supports_vCont_C == eLazyBoolYes ||
                m_supports_vCont_s == eLazyBoolYes ||
                m_supports_vCont_S == eLazyBoolYes)
            {
                m_supports_vCont_any = eLazyBoolYes;
            }
        }
    }
    
    switch (flavor)
    {
    case 'a': return m_supports_vCont_any;
    case 'A': return m_supports_vCont_all;
    case 'c': return m_supports_vCont_c;
    case 'C': return m_supports_vCont_C;
    case 's': return m_supports_vCont_s;
    case 'S': return m_supports_vCont_S;
    default: break;
    }
    return false;
}


size_t
GDBRemoteCommunicationClient::SendPacketAndWaitForResponse
(
    const char *payload,
    StringExtractorGDBRemote &response,
    bool send_async
)
{
    return SendPacketAndWaitForResponse (payload, 
                                         ::strlen (payload),
                                         response,
                                         send_async);
}

size_t
GDBRemoteCommunicationClient::SendPacketAndWaitForResponse
(
    const char *payload,
    size_t payload_length,
    StringExtractorGDBRemote &response,
    bool send_async
)
{
    Mutex::Locker locker;
    Log *log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PROCESS));
    size_t response_len = 0;
    if (GetSequenceMutex (locker))
    {
        if (SendPacketNoLock (payload, payload_length))
           response_len = WaitForPacketWithTimeoutMicroSecondsNoLock (response, GetPacketTimeoutInMicroSeconds ());
        else 
        {
            if (log)
                log->Printf("error: failed to send '%*s'", (int) payload_length, payload);   
        }
    }
    else
    {
        if (send_async)
        {
            if (IsRunning())
            {
                Mutex::Locker async_locker (m_async_mutex);
                m_async_packet.assign(payload, payload_length);
                m_async_packet_predicate.SetValue (true, eBroadcastNever);
                
                if (log) 
                    log->Printf ("async: async packet = %s", m_async_packet.c_str());

                bool timed_out = false;
                if (SendInterrupt(locker, 2, timed_out))
                {
                    if (m_interrupt_sent)
                    {
                        m_interrupt_sent = false;
                        TimeValue timeout_time;
                        timeout_time = TimeValue::Now();
                        timeout_time.OffsetWithSeconds (m_packet_timeout);

                        if (log) 
                            log->Printf ("async: sent interrupt");

                        if (m_async_packet_predicate.WaitForValueEqualTo (false, &timeout_time, &timed_out))
                        {
                            if (log) 
                                log->Printf ("async: got response");

                            // Swap the response buffer to avoid malloc and string copy
                            response.GetStringRef().swap (m_async_response.GetStringRef());
                            response_len = response.GetStringRef().size();
                        }
                        else
                        {
                            if (log) 
                                log->Printf ("async: timed out waiting for response");
                        }
                        
                        // Make sure we wait until the continue packet has been sent again...
                        if (m_private_is_running.WaitForValueEqualTo (true, &timeout_time, &timed_out))
                        {
                            if (log)
                            {
                                if (timed_out) 
                                    log->Printf ("async: timed out waiting for process to resume, but process was resumed");
                                else
                                    log->Printf ("async: async packet sent");
                            }
                        }
                        else
                        {
                            if (log) 
                                log->Printf ("async: timed out waiting for process to resume");
                        }
                    }
                    else
                    {
                        // We had a racy condition where we went to send the interrupt
                        // yet we were able to get the lock, so the process must have
                        // just stopped?
                        if (log) 
                            log->Printf ("async: got lock without sending interrupt");
                        // Send the packet normally since we got the lock
                        if (SendPacketNoLock (payload, payload_length))
                            response_len = WaitForPacketWithTimeoutMicroSecondsNoLock (response, GetPacketTimeoutInMicroSeconds ());
                        else 
                        {
                            if (log)
                                log->Printf("error: failed to send '%*s'", (int) payload_length, payload);   
                        }
                    }
                }
                else
                {
                    if (log) 
                        log->Printf ("async: failed to interrupt");
                }
            }
            else
            {
                if (log) 
                    log->Printf ("async: not running, async is ignored");
            }
        }
        else
        {
            if (log) 
                log->Printf("error: failed to get packet sequence mutex, not sending packet '%*s'", (int) payload_length, payload);
        }
    }
    if (response_len == 0)
    {
        if (log) 
            log->Printf("error: failed to get response for '%*s'", (int) payload_length, payload);
    }        
    return response_len;
}

static const char *end_delimiter = "--end--;";
static const int end_delimiter_len = 8;

std::string
GDBRemoteCommunicationClient::HarmonizeThreadIdsForProfileData
(   ProcessGDBRemote *process,
    StringExtractorGDBRemote& profileDataExtractor
)
{
    std::map<uint64_t, uint32_t> new_thread_id_to_used_usec_map;
    std::stringstream final_output;
    std::string name, value;

    // Going to assuming thread_used_usec comes first, else bail out.
    while (profileDataExtractor.GetNameColonValue(name, value))
    {
        if (name.compare("thread_used_id") == 0)
        {
            StringExtractor threadIDHexExtractor(value.c_str());
            uint64_t thread_id = threadIDHexExtractor.GetHexMaxU64(false, 0);
            
            bool has_used_usec = false;
            uint32_t curr_used_usec = 0;
            std::string usec_name, usec_value;
            uint32_t input_file_pos = profileDataExtractor.GetFilePos();
            if (profileDataExtractor.GetNameColonValue(usec_name, usec_value))
            {
                if (usec_name.compare("thread_used_usec") == 0)
                {
                    has_used_usec = true;
                    curr_used_usec = strtoull(usec_value.c_str(), NULL, 0);
                }
                else
                {
                    // We didn't find what we want, it is probably
                    // an older version. Bail out.
                    profileDataExtractor.SetFilePos(input_file_pos);
                }
            }

            if (has_used_usec)
            {
                uint32_t prev_used_usec = 0;
                std::map<uint64_t, uint32_t>::iterator iterator = m_thread_id_to_used_usec_map.find(thread_id);
                if (iterator != m_thread_id_to_used_usec_map.end())
                {
                    prev_used_usec = m_thread_id_to_used_usec_map[thread_id];
                }
                
                uint32_t real_used_usec = curr_used_usec - prev_used_usec;
                // A good first time record is one that runs for at least 0.25 sec
                bool good_first_time = (prev_used_usec == 0) && (real_used_usec > 250000);
                bool good_subsequent_time = (prev_used_usec > 0) &&
                    ((real_used_usec > 0) || (process->HasAssignedIndexIDToThread(thread_id)));
                
                if (good_first_time || good_subsequent_time)
                {
                    // We try to avoid doing too many index id reservation,
                    // resulting in fast increase of index ids.
                    
                    final_output << name << ":";
                    int32_t index_id = process->AssignIndexIDToThread(thread_id);
                    final_output << index_id << ";";
                    
                    final_output << usec_name << ":" << usec_value << ";";
                }
                else
                {
                    // Skip past 'thread_used_name'.
                    std::string local_name, local_value;
                    profileDataExtractor.GetNameColonValue(local_name, local_value);
                }
                
                // Store current time as previous time so that they can be compared later.
                new_thread_id_to_used_usec_map[thread_id] = curr_used_usec;
            }
            else
            {
                // Bail out and use old string.
                final_output << name << ":" << value << ";";
            }
        }
        else
        {
            final_output << name << ":" << value << ";";
        }
    }
    final_output << end_delimiter;
    m_thread_id_to_used_usec_map = new_thread_id_to_used_usec_map;
    
    return final_output.str();
}

StateType
GDBRemoteCommunicationClient::SendContinuePacketAndWaitForResponse
(
    ProcessGDBRemote *process,
    const char *payload,
    size_t packet_length,
    StringExtractorGDBRemote &response
)
{
    m_curr_tid = LLDB_INVALID_THREAD_ID;
    Log *log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PROCESS));
    if (log)
        log->Printf ("GDBRemoteCommunicationClient::%s ()", __FUNCTION__);

    Mutex::Locker locker(m_sequence_mutex);
    StateType state = eStateRunning;

    BroadcastEvent(eBroadcastBitRunPacketSent, NULL);
    m_public_is_running.SetValue (true, eBroadcastNever);
    // Set the starting continue packet into "continue_packet". This packet
    // may change if we are interrupted and we continue after an async packet...
    std::string continue_packet(payload, packet_length);
    
    bool got_async_packet = false;
    
    while (state == eStateRunning)
    {
        if (!got_async_packet)
        {
            if (log)
                log->Printf ("GDBRemoteCommunicationClient::%s () sending continue packet: %s", __FUNCTION__, continue_packet.c_str());
            if (SendPacketNoLock(continue_packet.c_str(), continue_packet.size()) == 0)
                state = eStateInvalid;
        
            m_private_is_running.SetValue (true, eBroadcastAlways);
        }
        
        got_async_packet = false;

        if (log)
            log->Printf ("GDBRemoteCommunicationClient::%s () WaitForPacket(%s)", __FUNCTION__, continue_packet.c_str());

        if (WaitForPacketWithTimeoutMicroSecondsNoLock(response, UINT32_MAX))
        {
            if (response.Empty())
                state = eStateInvalid;
            else
            {
                const char stop_type = response.GetChar();
                if (log)
                    log->Printf ("GDBRemoteCommunicationClient::%s () got packet: %s", __FUNCTION__, response.GetStringRef().c_str());
                switch (stop_type)
                {
                case 'T':
                case 'S':
                    {
                        if (process->GetStopID() == 0)
                        {
                            if (process->GetID() == LLDB_INVALID_PROCESS_ID)
                            {
                                lldb::pid_t pid = GetCurrentProcessID ();
                                if (pid != LLDB_INVALID_PROCESS_ID)
                                    process->SetID (pid);
                            }
                            process->BuildDynamicRegisterInfo (true);
                        }

                        // Privately notify any internal threads that we have stopped
                        // in case we wanted to interrupt our process, yet we might
                        // send a packet and continue without returning control to the
                        // user.
                        m_private_is_running.SetValue (false, eBroadcastAlways);

                        const uint8_t signo = response.GetHexU8 (UINT8_MAX);

                        bool continue_after_async = m_async_signal != -1 || m_async_packet_predicate.GetValue();
                        if (continue_after_async || m_interrupt_sent)
                        {
                            // We sent an interrupt packet to stop the inferior process
                            // for an async signal or to send an async packet while running
                            // but we might have been single stepping and received the
                            // stop packet for the step instead of for the interrupt packet.
                            // Typically when an interrupt is sent a SIGINT or SIGSTOP
                            // is used, so if we get anything else, we need to try and
                            // get another stop reply packet that may have been sent
                            // due to sending the interrupt when the target is stopped
                            // which will just re-send a copy of the last stop reply
                            // packet. If we don't do this, then the reply for our
                            // async packet will be the repeat stop reply packet and cause
                            // a lot of trouble for us!
                            if (signo != SIGINT && signo != SIGSTOP)
                            {
                                continue_after_async = false;

                                // We didn't get a a SIGINT or SIGSTOP, so try for a
                                // very brief time (1 ms) to get another stop reply
                                // packet to make sure it doesn't get in the way
                                StringExtractorGDBRemote extra_stop_reply_packet;
                                uint32_t timeout_usec = 1000;
                                if (WaitForPacketWithTimeoutMicroSecondsNoLock (extra_stop_reply_packet, timeout_usec))
                                {
                                    switch (extra_stop_reply_packet.GetChar())
                                    {
                                    case 'T':
                                    case 'S':
                                        // We did get an extra stop reply, which means
                                        // our interrupt didn't stop the target so we
                                        // shouldn't continue after the async signal
                                        // or packet is sent...
                                        continue_after_async = false;
                                        break;
                                    }
                                }
                            }
                        }

                        if (m_async_signal != -1)
                        {
                            if (log)
                                log->Printf ("async: send signo = %s", Host::GetSignalAsCString (m_async_signal));

                            // Save off the async signal we are supposed to send
                            const int async_signal = m_async_signal;
                            // Clear the async signal member so we don't end up
                            // sending the signal multiple times...
                            m_async_signal = -1;
                            // Check which signal we stopped with
                            if (signo == async_signal)
                            {
                                if (log) 
                                    log->Printf ("async: stopped with signal %s, we are done running", Host::GetSignalAsCString (signo));

                                // We already stopped with a signal that we wanted
                                // to stop with, so we are done
                            }
                            else
                            {
                                // We stopped with a different signal that the one
                                // we wanted to stop with, so now we must resume
                                // with the signal we want
                                char signal_packet[32];
                                int signal_packet_len = 0;
                                signal_packet_len = ::snprintf (signal_packet,
                                                                sizeof (signal_packet),
                                                                "C%2.2x",
                                                                async_signal);

                                if (log) 
                                    log->Printf ("async: stopped with signal %s, resume with %s", 
                                                       Host::GetSignalAsCString (signo),
                                                       Host::GetSignalAsCString (async_signal));

                                // Set the continue packet to resume even if the
                                // interrupt didn't cause our stop (ignore continue_after_async)
                                continue_packet.assign(signal_packet, signal_packet_len);
                                continue;
                            }
                        }
                        else if (m_async_packet_predicate.GetValue())
                        {
                            Log * packet_log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PACKETS));

                            // We are supposed to send an asynchronous packet while
                            // we are running. 
                            m_async_response.Clear();
                            if (m_async_packet.empty())
                            {
                                if (packet_log) 
                                    packet_log->Printf ("async: error: empty async packet");                            

                            }
                            else
                            {
                                if (packet_log) 
                                    packet_log->Printf ("async: sending packet");
                                
                                SendPacketAndWaitForResponse (&m_async_packet[0], 
                                                              m_async_packet.size(),
                                                              m_async_response,
                                                              false);
                            }
                            // Let the other thread that was trying to send the async
                            // packet know that the packet has been sent and response is
                            // ready...
                            m_async_packet_predicate.SetValue(false, eBroadcastAlways);

                            if (packet_log) 
                                packet_log->Printf ("async: sent packet, continue_after_async = %i", continue_after_async);

                            // Set the continue packet to resume if our interrupt
                            // for the async packet did cause the stop
                            if (continue_after_async)
                            {
                                // Reverting this for now as it is causing deadlocks
                                // in programs (<rdar://problem/11529853>). In the future
                                // we should check our thread list and "do the right thing"
                                // for new threads that show up while we stop and run async
                                // packets. Setting the packet to 'c' to continue all threads
                                // is the right thing to do 99.99% of the time because if a
                                // thread was single stepping, and we sent an interrupt, we
                                // will notice above that we didn't stop due to an interrupt
                                // but stopped due to stepping and we would _not_ continue.
                                continue_packet.assign (1, 'c');
                                continue;
                            }
                        }
                        // Stop with signal and thread info
                        state = eStateStopped;
                    }
                    break;

                case 'W':
                case 'X':
                    // process exited
                    state = eStateExited;
                    break;

                case 'O':
                    // STDOUT
                    {
                        got_async_packet = true;
                        std::string inferior_stdout;
                        inferior_stdout.reserve(response.GetBytesLeft () / 2);
                        char ch;
                        while ((ch = response.GetHexU8()) != '\0')
                            inferior_stdout.append(1, ch);
                        process->AppendSTDOUT (inferior_stdout.c_str(), inferior_stdout.size());
                    }
                    break;

                case 'A':
                    // Async miscellaneous reply. Right now, only profile data is coming through this channel.
                    {
                        got_async_packet = true;
                        std::string input = response.GetStringRef().substr(1); // '1' to move beyond 'A'
                        if (m_partial_profile_data.length() > 0)
                        {
                            m_partial_profile_data.append(input);
                            input = m_partial_profile_data;
                            m_partial_profile_data.clear();
                        }
                        
                        size_t found, pos = 0, len = input.length();
                        while ((found = input.find(end_delimiter, pos)) != std::string::npos)
                        {
                            StringExtractorGDBRemote profileDataExtractor(input.substr(pos, found).c_str());
                            const std::string& profile_data = HarmonizeThreadIdsForProfileData(process, profileDataExtractor);
                            process->BroadcastAsyncProfileData (profile_data.c_str(), profile_data.length());
                            
                            pos = found + end_delimiter_len;
                        }
                        
                        if (pos < len)
                        {
                            // Last incomplete chunk.
                            m_partial_profile_data = input.substr(pos);
                        }
                    }
                    break;

                case 'E':
                    // ERROR
                    state = eStateInvalid;
                    break;

                default:
                    if (log)
                        log->Printf ("GDBRemoteCommunicationClient::%s () unrecognized async packet", __FUNCTION__);
                    state = eStateInvalid;
                    break;
                }
            }
        }
        else
        {
            if (log)
                log->Printf ("GDBRemoteCommunicationClient::%s () WaitForPacket(...) => false", __FUNCTION__);
            state = eStateInvalid;
        }
    }
    if (log)
        log->Printf ("GDBRemoteCommunicationClient::%s () => %s", __FUNCTION__, StateAsCString(state));
    response.SetFilePos(0);
    m_private_is_running.SetValue (false, eBroadcastAlways);
    m_public_is_running.SetValue (false, eBroadcastAlways);
    return state;
}

bool
GDBRemoteCommunicationClient::SendAsyncSignal (int signo)
{
    Mutex::Locker async_locker (m_async_mutex);
    m_async_signal = signo;
    bool timed_out = false;
    Mutex::Locker locker;
    if (SendInterrupt (locker, 1, timed_out))
        return true;
    m_async_signal = -1;
    return false;
}

// This function takes a mutex locker as a parameter in case the GetSequenceMutex
// actually succeeds. If it doesn't succeed in acquiring the sequence mutex 
// (the expected result), then it will send the halt packet. If it does succeed
// then the caller that requested the interrupt will want to keep the sequence
// locked down so that no one else can send packets while the caller has control.
// This function usually gets called when we are running and need to stop the 
// target. It can also be used when we are running and and we need to do something
// else (like read/write memory), so we need to interrupt the running process
// (gdb remote protocol requires this), and do what we need to do, then resume.

bool
GDBRemoteCommunicationClient::SendInterrupt
(
    Mutex::Locker& locker, 
    uint32_t seconds_to_wait_for_stop,             
    bool &timed_out
)
{
    timed_out = false;
    Log *log (ProcessGDBRemoteLog::GetLogIfAnyCategoryIsSet (GDBR_LOG_PROCESS | GDBR_LOG_PACKETS));

    if (IsRunning())
    {
        // Only send an interrupt if our debugserver is running...
        if (GetSequenceMutex (locker))
        {
            if (log)
                log->Printf ("SendInterrupt () - got sequence mutex without having to interrupt");
        }
        else
        {
            // Someone has the mutex locked waiting for a response or for the
            // inferior to stop, so send the interrupt on the down low...
            char ctrl_c = '\x03';
            ConnectionStatus status = eConnectionStatusSuccess;
            size_t bytes_written = Write (&ctrl_c, 1, status, NULL);
            if (log)
                log->PutCString("send packet: \\x03");
            if (bytes_written > 0)
            {
                m_interrupt_sent = true;
                if (seconds_to_wait_for_stop)
                {
                    TimeValue timeout;
                    if (seconds_to_wait_for_stop)
                    {
                        timeout = TimeValue::Now();
                        timeout.OffsetWithSeconds (seconds_to_wait_for_stop);
                    }
                    if (m_private_is_running.WaitForValueEqualTo (false, &timeout, &timed_out))
                    {
                        if (log)
                            log->PutCString ("SendInterrupt () - sent interrupt, private state stopped");
                        return true;
                    }
                    else
                    {
                        if (log)
                            log->Printf ("SendInterrupt () - sent interrupt, timed out wating for async thread resume");
                    }
                }
                else
                {
                    if (log)
                        log->Printf ("SendInterrupt () - sent interrupt, not waiting for stop...");
                    return true;
                }
            }
            else
            {
                if (log)
                    log->Printf ("SendInterrupt () - failed to write interrupt");
            }
            return false;
        }
    }
    else
    {
        if (log)
            log->Printf ("SendInterrupt () - not running");
    }
    return true;
}

lldb::pid_t
GDBRemoteCommunicationClient::GetCurrentProcessID ()
{
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse("qC", strlen("qC"), response, false))
    {
        if (response.GetChar() == 'Q')
            if (response.GetChar() == 'C')
                return response.GetHexMaxU32 (false, LLDB_INVALID_PROCESS_ID);
    }
    return LLDB_INVALID_PROCESS_ID;
}

bool
GDBRemoteCommunicationClient::GetLaunchSuccess (std::string &error_str)
{
    error_str.clear();
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse("qLaunchSuccess", strlen("qLaunchSuccess"), response, false))
    {
        if (response.IsOKResponse())
            return true;
        if (response.GetChar() == 'E')
        {
            // A string the describes what failed when launching...
            error_str = response.GetStringRef().substr(1);
        }
        else
        {
            error_str.assign ("unknown error occurred launching process");
        }
    }
    else
    {
        error_str.assign ("timed out waiting for app to launch");
    }
    return false;
}

int
GDBRemoteCommunicationClient::SendArgumentsPacket (char const *argv[])
{
    if (argv && argv[0])
    {
        StreamString packet;
        packet.PutChar('A');
        const char *arg;
        for (uint32_t i = 0; (arg = argv[i]) != NULL; ++i)
        {
            const int arg_len = strlen(arg);
            if (i > 0)
                packet.PutChar(',');
            packet.Printf("%i,%i,", arg_len * 2, i);
            packet.PutBytesAsRawHex8 (arg, arg_len);
        }

        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false))
        {
            if (response.IsOKResponse())
                return 0;
            uint8_t error = response.GetError();
            if (error)
                return error;
        }
    }
    return -1;
}

int
GDBRemoteCommunicationClient::SendEnvironmentPacket (char const *name_equal_value)
{
    if (name_equal_value && name_equal_value[0])
    {
        StreamString packet;
        packet.Printf("QEnvironment:%s", name_equal_value);
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false))
        {
            if (response.IsOKResponse())
                return 0;
            uint8_t error = response.GetError();
            if (error)
                return error;
        }
    }
    return -1;
}

int
GDBRemoteCommunicationClient::SendLaunchArchPacket (char const *arch)
{
    if (arch && arch[0])
    {
        StreamString packet;
        packet.Printf("QLaunchArch:%s", arch);
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false))
        {
            if (response.IsOKResponse())
                return 0;
            uint8_t error = response.GetError();
            if (error)
                return error;
        }
    }
    return -1;
}

bool
GDBRemoteCommunicationClient::GetOSVersion (uint32_t &major, 
                                            uint32_t &minor, 
                                            uint32_t &update)
{
    if (GetHostInfo ())
    {
        if (m_os_version_major != UINT32_MAX)
        {
            major = m_os_version_major;
            minor = m_os_version_minor;
            update = m_os_version_update;
            return true;
        }
    }
    return false;
}

bool
GDBRemoteCommunicationClient::GetOSBuildString (std::string &s)
{
    if (GetHostInfo ())
    {
        if (!m_os_build.empty())
        {
            s = m_os_build;
            return true;
        }
    }
    s.clear();
    return false;
}


bool
GDBRemoteCommunicationClient::GetOSKernelDescription (std::string &s)
{
    if (GetHostInfo ())
    {
        if (!m_os_kernel.empty())
        {
            s = m_os_kernel;
            return true;
        }
    }
    s.clear();
    return false;
}

bool
GDBRemoteCommunicationClient::GetHostname (std::string &s)
{
    if (GetHostInfo ())
    {
        if (!m_hostname.empty())
        {
            s = m_hostname;
            return true;
        }
    }
    s.clear();
    return false;
}

ArchSpec
GDBRemoteCommunicationClient::GetSystemArchitecture ()
{
    if (GetHostInfo ())
        return m_host_arch;
    return ArchSpec();
}

const lldb_private::ArchSpec &
GDBRemoteCommunicationClient::GetProcessArchitecture ()
{
    if (m_qProcessInfo_is_valid == eLazyBoolCalculate)
        GetCurrentProcessInfo ();
    return m_process_arch;
}


bool
GDBRemoteCommunicationClient::GetHostInfo (bool force)
{
    if (force || m_qHostInfo_is_valid == eLazyBoolCalculate)
    {
        m_qHostInfo_is_valid = eLazyBoolNo;
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse ("qHostInfo", response, false))
        {
            if (response.IsNormalResponse())
            {
                std::string name;
                std::string value;
                uint32_t cpu = LLDB_INVALID_CPUTYPE;
                uint32_t sub = 0;
                std::string arch_name;
                std::string os_name;
                std::string vendor_name;
                std::string triple;
                uint32_t pointer_byte_size = 0;
                StringExtractor extractor;
                ByteOrder byte_order = eByteOrderInvalid;
                uint32_t num_keys_decoded = 0;
                while (response.GetNameColonValue(name, value))
                {
                    if (name.compare("cputype") == 0)
                    {
                        // exception type in big endian hex
                        cpu = Args::StringToUInt32 (value.c_str(), LLDB_INVALID_CPUTYPE, 0);
                        if (cpu != LLDB_INVALID_CPUTYPE)
                            ++num_keys_decoded;
                    }
                    else if (name.compare("cpusubtype") == 0)
                    {
                        // exception count in big endian hex
                        sub = Args::StringToUInt32 (value.c_str(), 0, 0);
                        if (sub != 0)
                            ++num_keys_decoded;
                    }
                    else if (name.compare("arch") == 0)
                    {
                        arch_name.swap (value);
                        ++num_keys_decoded;
                    }
                    else if (name.compare("triple") == 0)
                    {
                        // The triple comes as ASCII hex bytes since it contains '-' chars
                        extractor.GetStringRef().swap(value);
                        extractor.SetFilePos(0);
                        extractor.GetHexByteString (triple);
                        ++num_keys_decoded;
                    }
                    else if (name.compare("os_build") == 0)
                    {
                        extractor.GetStringRef().swap(value);
                        extractor.SetFilePos(0);
                        extractor.GetHexByteString (m_os_build);
                        ++num_keys_decoded;
                    }
                    else if (name.compare("hostname") == 0)
                    {
                        extractor.GetStringRef().swap(value);
                        extractor.SetFilePos(0);
                        extractor.GetHexByteString (m_hostname);
                        ++num_keys_decoded;
                    }
                    else if (name.compare("os_kernel") == 0)
                    {
                        extractor.GetStringRef().swap(value);
                        extractor.SetFilePos(0);
                        extractor.GetHexByteString (m_os_kernel);
                        ++num_keys_decoded;
                    }
                    else if (name.compare("ostype") == 0)
                    {
                        os_name.swap (value);
                        ++num_keys_decoded;
                    }
                    else if (name.compare("vendor") == 0)
                    {
                        vendor_name.swap(value);
                        ++num_keys_decoded;
                    }
                    else if (name.compare("endian") == 0)
                    {
                        ++num_keys_decoded;
                        if (value.compare("little") == 0)
                            byte_order = eByteOrderLittle;
                        else if (value.compare("big") == 0)
                            byte_order = eByteOrderBig;
                        else if (value.compare("pdp") == 0)
                            byte_order = eByteOrderPDP;
                        else
                            --num_keys_decoded;
                    }
                    else if (name.compare("ptrsize") == 0)
                    {
                        pointer_byte_size = Args::StringToUInt32 (value.c_str(), 0, 0);
                        if (pointer_byte_size != 0)
                            ++num_keys_decoded;
                    }
                    else if (name.compare("os_version") == 0)
                    {
                        Args::StringToVersion (value.c_str(), 
                                               m_os_version_major,
                                               m_os_version_minor,
                                               m_os_version_update);
                        if (m_os_version_major != UINT32_MAX)
                            ++num_keys_decoded;
                    }
                    else if (name.compare("watchpoint_exceptions_received") == 0)
                    {
                        ++num_keys_decoded;
                        if (strcmp(value.c_str(),"before") == 0)
                            m_watchpoints_trigger_after_instruction = eLazyBoolNo;
                        else if (strcmp(value.c_str(),"after") == 0)
                            m_watchpoints_trigger_after_instruction = eLazyBoolYes;
                        else
                            --num_keys_decoded;
                    }

                }
                
                if (num_keys_decoded > 0)
                    m_qHostInfo_is_valid = eLazyBoolYes;

                if (triple.empty())
                {
                    if (arch_name.empty())
                    {
                        if (cpu != LLDB_INVALID_CPUTYPE)
                        {
                            m_host_arch.SetArchitecture (eArchTypeMachO, cpu, sub);
                            if (pointer_byte_size)
                            {
                                assert (pointer_byte_size == m_host_arch.GetAddressByteSize());
                            }
                            if (byte_order != eByteOrderInvalid)
                            {
                                assert (byte_order == m_host_arch.GetByteOrder());
                            }

                            if (!os_name.empty() && vendor_name.compare("apple") == 0 && os_name.find("darwin") == 0)
                            {
                                switch (m_host_arch.GetMachine())
                                {
                                case llvm::Triple::arm:
                                case llvm::Triple::thumb:
                                    os_name = "ios";
                                    break;
                                default:
                                    os_name = "macosx";
                                    break;
                                }
                            }
                            if (!vendor_name.empty())
                                m_host_arch.GetTriple().setVendorName (llvm::StringRef (vendor_name));
                            if (!os_name.empty())
                                m_host_arch.GetTriple().setOSName (llvm::StringRef (os_name));
                                
                        }
                    }
                    else
                    {
                        std::string triple;
                        triple += arch_name;
                        if (!vendor_name.empty() || !os_name.empty())
                        {
                            triple += '-';
                            if (vendor_name.empty())
                                triple += "unknown";
                            else
                                triple += vendor_name;
                            triple += '-';
                            if (os_name.empty())
                                triple += "unknown";
                            else
                                triple += os_name;
                        }
                        m_host_arch.SetTriple (triple.c_str());
                        
                        llvm::Triple &host_triple = m_host_arch.GetTriple();
                        if (host_triple.getVendor() == llvm::Triple::Apple && host_triple.getOS() == llvm::Triple::Darwin)
                        {
                            switch (m_host_arch.GetMachine())
                            {
                                case llvm::Triple::arm:
                                case llvm::Triple::thumb:
                                    host_triple.setOS(llvm::Triple::IOS);
                                    break;
                                default:
                                    host_triple.setOS(llvm::Triple::MacOSX);
                                    break;
                            }
                        }
                        if (pointer_byte_size)
                        {
                            assert (pointer_byte_size == m_host_arch.GetAddressByteSize());
                        }
                        if (byte_order != eByteOrderInvalid)
                        {
                            assert (byte_order == m_host_arch.GetByteOrder());
                        }
                        
                    }
                }
                else
                {
                    m_host_arch.SetTriple (triple.c_str());
                    if (pointer_byte_size)
                    {
                        assert (pointer_byte_size == m_host_arch.GetAddressByteSize());
                    }
                    if (byte_order != eByteOrderInvalid)
                    {
                        assert (byte_order == m_host_arch.GetByteOrder());
                    }
                }       
            }
        }
    }
    return m_qHostInfo_is_valid == eLazyBoolYes;
}

int
GDBRemoteCommunicationClient::SendAttach 
(
    lldb::pid_t pid, 
    StringExtractorGDBRemote& response
)
{
    if (pid != LLDB_INVALID_PROCESS_ID)
    {
        char packet[64];
        const int packet_len = ::snprintf (packet, sizeof(packet), "vAttach;%" PRIx64, pid);
        assert (packet_len < sizeof(packet));
        if (SendPacketAndWaitForResponse (packet, packet_len, response, false))
        {
            if (response.IsErrorResponse())
                return response.GetError();
            return 0;
        }
    }
    return -1;
}

const lldb_private::ArchSpec &
GDBRemoteCommunicationClient::GetHostArchitecture ()
{
    if (m_qHostInfo_is_valid == eLazyBoolCalculate)
        GetHostInfo ();
    return m_host_arch;
}

addr_t
GDBRemoteCommunicationClient::AllocateMemory (size_t size, uint32_t permissions)
{
    if (m_supports_alloc_dealloc_memory != eLazyBoolNo)
    {
        m_supports_alloc_dealloc_memory = eLazyBoolYes;
        char packet[64];
        const int packet_len = ::snprintf (packet, sizeof(packet), "_M%" PRIx64 ",%s%s%s",
                                           (uint64_t)size,
                                           permissions & lldb::ePermissionsReadable ? "r" : "",
                                           permissions & lldb::ePermissionsWritable ? "w" : "",
                                           permissions & lldb::ePermissionsExecutable ? "x" : "");
        assert (packet_len < sizeof(packet));
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet, packet_len, response, false))
        {
            if (!response.IsErrorResponse())
                return response.GetHexMaxU64(false, LLDB_INVALID_ADDRESS);
        }
        else
        {
            m_supports_alloc_dealloc_memory = eLazyBoolNo;
        }
    }
    return LLDB_INVALID_ADDRESS;
}

bool
GDBRemoteCommunicationClient::DeallocateMemory (addr_t addr)
{
    if (m_supports_alloc_dealloc_memory != eLazyBoolNo)
    {
        m_supports_alloc_dealloc_memory = eLazyBoolYes;
        char packet[64];
        const int packet_len = ::snprintf(packet, sizeof(packet), "_m%" PRIx64, (uint64_t)addr);
        assert (packet_len < sizeof(packet));
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet, packet_len, response, false))
        {
            if (response.IsOKResponse())
                return true;
        }
        else
        {
            m_supports_alloc_dealloc_memory = eLazyBoolNo;
        }
    }
    return false;
}

Error
GDBRemoteCommunicationClient::Detach (bool keep_stopped)
{
    Error error;
    
    if (keep_stopped)
    {
        if (m_supports_detach_stay_stopped == eLazyBoolCalculate)
        {
            char packet[64];
            const int packet_len = ::snprintf(packet, sizeof(packet), "qSupportsDetachAndStayStopped:");
            assert (packet_len < sizeof(packet));
            StringExtractorGDBRemote response;
            if (SendPacketAndWaitForResponse (packet, packet_len, response, false))
            {
                m_supports_detach_stay_stopped = eLazyBoolYes;        
            }
            else
            {
                m_supports_detach_stay_stopped = eLazyBoolNo;
            }
        }

        if (m_supports_detach_stay_stopped == eLazyBoolNo)
        {
            error.SetErrorString("Stays stopped not supported by this target.");
            return error;
        }
        else
        {
            size_t num_sent = SendPacket ("D1", 2);
            if (num_sent == 0)
                error.SetErrorString ("Sending extended disconnect packet failed.");
        }
    }
    else
    {
        size_t num_sent = SendPacket ("D", 1);
        if (num_sent == 0)
            error.SetErrorString ("Sending disconnect packet failed.");
    }
    return error;
}

Error
GDBRemoteCommunicationClient::GetMemoryRegionInfo (lldb::addr_t addr, 
                                                  lldb_private::MemoryRegionInfo &region_info)
{
    Error error;
    region_info.Clear();

    if (m_supports_memory_region_info != eLazyBoolNo)
    {
        m_supports_memory_region_info = eLazyBoolYes;
        char packet[64];
        const int packet_len = ::snprintf(packet, sizeof(packet), "qMemoryRegionInfo:%" PRIx64, (uint64_t)addr);
        assert (packet_len < sizeof(packet));
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet, packet_len, response, false))
        {
            std::string name;
            std::string value;
            addr_t addr_value;
            bool success = true;
            bool saw_permissions = false;
            while (success && response.GetNameColonValue(name, value))
            {
                if (name.compare ("start") == 0)
                {
                    addr_value = Args::StringToUInt64(value.c_str(), LLDB_INVALID_ADDRESS, 16, &success);
                    if (success)
                        region_info.GetRange().SetRangeBase(addr_value);
                }
                else if (name.compare ("size") == 0)
                {
                    addr_value = Args::StringToUInt64(value.c_str(), 0, 16, &success);
                    if (success)
                        region_info.GetRange().SetByteSize (addr_value);
                }
                else if (name.compare ("permissions") == 0 && region_info.GetRange().IsValid())
                {
                    saw_permissions = true;
                    if (region_info.GetRange().Contains (addr))
                    {
                        if (value.find('r') != std::string::npos)
                            region_info.SetReadable (MemoryRegionInfo::eYes);
                        else
                            region_info.SetReadable (MemoryRegionInfo::eNo);

                        if (value.find('w') != std::string::npos)
                            region_info.SetWritable (MemoryRegionInfo::eYes);
                        else
                            region_info.SetWritable (MemoryRegionInfo::eNo);

                        if (value.find('x') != std::string::npos)
                            region_info.SetExecutable (MemoryRegionInfo::eYes);
                        else
                            region_info.SetExecutable (MemoryRegionInfo::eNo);
                    }
                    else
                    {
                        // The reported region does not contain this address -- we're looking at an unmapped page
                        region_info.SetReadable (MemoryRegionInfo::eNo);
                        region_info.SetWritable (MemoryRegionInfo::eNo);
                        region_info.SetExecutable (MemoryRegionInfo::eNo);
                    }
                }
                else if (name.compare ("error") == 0)
                {
                    StringExtractorGDBRemote name_extractor;
                    // Swap "value" over into "name_extractor"
                    name_extractor.GetStringRef().swap(value);
                    // Now convert the HEX bytes into a string value
                    name_extractor.GetHexByteString (value);
                    error.SetErrorString(value.c_str());
                }
            }

            // We got a valid address range back but no permissions -- which means this is an unmapped page
            if (region_info.GetRange().IsValid() && saw_permissions == false)
            {
                region_info.SetReadable (MemoryRegionInfo::eNo);
                region_info.SetWritable (MemoryRegionInfo::eNo);
                region_info.SetExecutable (MemoryRegionInfo::eNo);
            }
        }
        else
        {
            m_supports_memory_region_info = eLazyBoolNo;
        }
    }

    if (m_supports_memory_region_info == eLazyBoolNo)
    {
        error.SetErrorString("qMemoryRegionInfo is not supported");
    }
    if (error.Fail())
        region_info.Clear();
    return error;

}

Error
GDBRemoteCommunicationClient::GetWatchpointSupportInfo (uint32_t &num)
{
    Error error;

    if (m_supports_watchpoint_support_info == eLazyBoolYes)
    {
        num = m_num_supported_hardware_watchpoints;
        return error;
    }

    // Set num to 0 first.
    num = 0;
    if (m_supports_watchpoint_support_info != eLazyBoolNo)
    {
        char packet[64];
        const int packet_len = ::snprintf(packet, sizeof(packet), "qWatchpointSupportInfo:");
        assert (packet_len < sizeof(packet));
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet, packet_len, response, false))
        {
            m_supports_watchpoint_support_info = eLazyBoolYes;        
            std::string name;
            std::string value;
            while (response.GetNameColonValue(name, value))
            {
                if (name.compare ("num") == 0)
                {
                    num = Args::StringToUInt32(value.c_str(), 0, 0);
                    m_num_supported_hardware_watchpoints = num;
                }
            }
        }
        else
        {
            m_supports_watchpoint_support_info = eLazyBoolNo;
        }
    }

    if (m_supports_watchpoint_support_info == eLazyBoolNo)
    {
        error.SetErrorString("qWatchpointSupportInfo is not supported");
    }
    return error;

}

lldb_private::Error
GDBRemoteCommunicationClient::GetWatchpointSupportInfo (uint32_t &num, bool& after)
{
    Error error(GetWatchpointSupportInfo(num));
    if (error.Success())
        error = GetWatchpointsTriggerAfterInstruction(after);
    return error;
}

lldb_private::Error
GDBRemoteCommunicationClient::GetWatchpointsTriggerAfterInstruction (bool &after)
{
    Error error;
    
    // we assume watchpoints will happen after running the relevant opcode
    // and we only want to override this behavior if we have explicitly
    // received a qHostInfo telling us otherwise
    if (m_qHostInfo_is_valid != eLazyBoolYes)
        after = true;
    else
        after = (m_watchpoints_trigger_after_instruction != eLazyBoolNo);
    return error;
}

int
GDBRemoteCommunicationClient::SetSTDIN (char const *path)
{
    if (path && path[0])
    {
        StreamString packet;
        packet.PutCString("QSetSTDIN:");
        packet.PutBytesAsRawHex8(path, strlen(path));

        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false))
        {
            if (response.IsOKResponse())
                return 0;
            uint8_t error = response.GetError();
            if (error)
                return error;
        }
    }
    return -1;
}

int
GDBRemoteCommunicationClient::SetSTDOUT (char const *path)
{
    if (path && path[0])
    {
        StreamString packet;
        packet.PutCString("QSetSTDOUT:");
        packet.PutBytesAsRawHex8(path, strlen(path));
        
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false))
        {
            if (response.IsOKResponse())
                return 0;
            uint8_t error = response.GetError();
            if (error)
                return error;
        }
    }
    return -1;
}

int
GDBRemoteCommunicationClient::SetSTDERR (char const *path)
{
    if (path && path[0])
    {
        StreamString packet;
        packet.PutCString("QSetSTDERR:");
        packet.PutBytesAsRawHex8(path, strlen(path));
        
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false))
        {
            if (response.IsOKResponse())
                return 0;
            uint8_t error = response.GetError();
            if (error)
                return error;
        }
    }
    return -1;
}

int
GDBRemoteCommunicationClient::SetWorkingDir (char const *path)
{
    if (path && path[0])
    {
        StreamString packet;
        packet.PutCString("QSetWorkingDir:");
        packet.PutBytesAsRawHex8(path, strlen(path));
        
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false))
        {
            if (response.IsOKResponse())
                return 0;
            uint8_t error = response.GetError();
            if (error)
                return error;
        }
    }
    return -1;
}

int
GDBRemoteCommunicationClient::SetDisableASLR (bool enable)
{
    char packet[32];
    const int packet_len = ::snprintf (packet, sizeof (packet), "QSetDisableASLR:%i", enable ? 1 : 0);
    assert (packet_len < sizeof(packet));
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse (packet, packet_len, response, false))
    {
        if (response.IsOKResponse())
            return 0;
        uint8_t error = response.GetError();
        if (error)
            return error;
    }
    return -1;
}

bool
GDBRemoteCommunicationClient::DecodeProcessInfoResponse (StringExtractorGDBRemote &response, ProcessInstanceInfo &process_info)
{
    if (response.IsNormalResponse())
    {
        std::string name;
        std::string value;
        StringExtractor extractor;
        
        while (response.GetNameColonValue(name, value))
        {
            if (name.compare("pid") == 0)
            {
                process_info.SetProcessID (Args::StringToUInt32 (value.c_str(), LLDB_INVALID_PROCESS_ID, 0));
            }
            else if (name.compare("ppid") == 0)
            {
                process_info.SetParentProcessID (Args::StringToUInt32 (value.c_str(), LLDB_INVALID_PROCESS_ID, 0));
            }
            else if (name.compare("uid") == 0)
            {
                process_info.SetUserID (Args::StringToUInt32 (value.c_str(), UINT32_MAX, 0));
            }
            else if (name.compare("euid") == 0)
            {
                process_info.SetEffectiveUserID (Args::StringToUInt32 (value.c_str(), UINT32_MAX, 0));
            }
            else if (name.compare("gid") == 0)
            {
                process_info.SetGroupID (Args::StringToUInt32 (value.c_str(), UINT32_MAX, 0));
            }
            else if (name.compare("egid") == 0)
            {
                process_info.SetEffectiveGroupID (Args::StringToUInt32 (value.c_str(), UINT32_MAX, 0));
            }
            else if (name.compare("triple") == 0)
            {
                // The triple comes as ASCII hex bytes since it contains '-' chars
                extractor.GetStringRef().swap(value);
                extractor.SetFilePos(0);
                extractor.GetHexByteString (value);
                process_info.GetArchitecture ().SetTriple (value.c_str());
            }
            else if (name.compare("name") == 0)
            {
                StringExtractor extractor;
                // The process name from ASCII hex bytes since we can't 
                // control the characters in a process name
                extractor.GetStringRef().swap(value);
                extractor.SetFilePos(0);
                extractor.GetHexByteString (value);
                process_info.GetExecutableFile().SetFile (value.c_str(), false);
            }
        }
        
        if (process_info.GetProcessID() != LLDB_INVALID_PROCESS_ID)
            return true;
    }
    return false;
}

bool
GDBRemoteCommunicationClient::GetProcessInfo (lldb::pid_t pid, ProcessInstanceInfo &process_info)
{
    process_info.Clear();
    
    if (m_supports_qProcessInfoPID)
    {
        char packet[32];
        const int packet_len = ::snprintf (packet, sizeof (packet), "qProcessInfoPID:%" PRIu64, pid);
        assert (packet_len < sizeof(packet));
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet, packet_len, response, false))
        {
            return DecodeProcessInfoResponse (response, process_info);
        }
        else
        {
            m_supports_qProcessInfoPID = false;
            return false;
        }
    }
    return false;
}

bool
GDBRemoteCommunicationClient::GetCurrentProcessInfo ()
{
    if (m_qProcessInfo_is_valid == eLazyBoolYes)
        return true;
    if (m_qProcessInfo_is_valid == eLazyBoolNo)
        return false;

    GetHostInfo ();

    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse ("qProcessInfo", response, false))
    {
        if (response.IsNormalResponse())
        {
            std::string name;
            std::string value;
            uint32_t cpu = LLDB_INVALID_CPUTYPE;
            uint32_t sub = 0;
            std::string arch_name;
            std::string os_name;
            std::string vendor_name;
            std::string triple;
            uint32_t pointer_byte_size = 0;
            StringExtractor extractor;
            ByteOrder byte_order = eByteOrderInvalid;
            uint32_t num_keys_decoded = 0;
            while (response.GetNameColonValue(name, value))
            {
                if (name.compare("cputype") == 0)
                {
                    cpu = Args::StringToUInt32 (value.c_str(), LLDB_INVALID_CPUTYPE, 16);
                    if (cpu != LLDB_INVALID_CPUTYPE)
                        ++num_keys_decoded;
                }
                else if (name.compare("cpusubtype") == 0)
                {
                    sub = Args::StringToUInt32 (value.c_str(), 0, 16);
                    if (sub != 0)
                        ++num_keys_decoded;
                }
                else if (name.compare("ostype") == 0)
                {
                    os_name.swap (value);
                    ++num_keys_decoded;
                }
                else if (name.compare("vendor") == 0)
                {
                    vendor_name.swap(value);
                    ++num_keys_decoded;
                }
                else if (name.compare("endian") == 0)
                {
                    ++num_keys_decoded;
                    if (value.compare("little") == 0)
                        byte_order = eByteOrderLittle;
                    else if (value.compare("big") == 0)
                        byte_order = eByteOrderBig;
                    else if (value.compare("pdp") == 0)
                        byte_order = eByteOrderPDP;
                    else
                        --num_keys_decoded;
                }
                else if (name.compare("ptrsize") == 0)
                {
                    pointer_byte_size = Args::StringToUInt32 (value.c_str(), 0, 16);
                    if (pointer_byte_size != 0)
                        ++num_keys_decoded;
                }
            }
            if (num_keys_decoded > 0)
                m_qProcessInfo_is_valid = eLazyBoolYes;
            if (cpu != LLDB_INVALID_CPUTYPE && !os_name.empty() && !vendor_name.empty())
            {
                m_process_arch.SetArchitecture (eArchTypeMachO, cpu, sub);
                if (pointer_byte_size)
                {
                    assert (pointer_byte_size == m_process_arch.GetAddressByteSize());
                }
                m_host_arch.GetTriple().setVendorName (llvm::StringRef (vendor_name));
                m_host_arch.GetTriple().setOSName (llvm::StringRef (os_name));
                return true;
            }
        }
    }
    else
    {
        m_qProcessInfo_is_valid = eLazyBoolNo;
    }

    return false;
}


uint32_t
GDBRemoteCommunicationClient::FindProcesses (const ProcessInstanceInfoMatch &match_info,
                                             ProcessInstanceInfoList &process_infos)
{
    process_infos.Clear();
    
    if (m_supports_qfProcessInfo)
    {
        StreamString packet;
        packet.PutCString ("qfProcessInfo");
        if (!match_info.MatchAllProcesses())
        {
            packet.PutChar (':');
            const char *name = match_info.GetProcessInfo().GetName();
            bool has_name_match = false;
            if (name && name[0])
            {
                has_name_match = true;
                NameMatchType name_match_type = match_info.GetNameMatchType();
                switch (name_match_type)
                {
                case eNameMatchIgnore:  
                    has_name_match = false;
                    break;

                case eNameMatchEquals:  
                    packet.PutCString ("name_match:equals;"); 
                    break;

                case eNameMatchContains:
                    packet.PutCString ("name_match:contains;"); 
                    break;
                
                case eNameMatchStartsWith:
                    packet.PutCString ("name_match:starts_with;"); 
                    break;
                
                case eNameMatchEndsWith:
                    packet.PutCString ("name_match:ends_with;"); 
                    break;

                case eNameMatchRegularExpression:
                    packet.PutCString ("name_match:regex;"); 
                    break;
                }
                if (has_name_match)
                {
                    packet.PutCString ("name:");
                    packet.PutBytesAsRawHex8(name, ::strlen(name));
                    packet.PutChar (';');
                }
            }
            
            if (match_info.GetProcessInfo().ProcessIDIsValid())
                packet.Printf("pid:%" PRIu64 ";",match_info.GetProcessInfo().GetProcessID());
            if (match_info.GetProcessInfo().ParentProcessIDIsValid())
                packet.Printf("parent_pid:%" PRIu64 ";",match_info.GetProcessInfo().GetParentProcessID());
            if (match_info.GetProcessInfo().UserIDIsValid())
                packet.Printf("uid:%u;",match_info.GetProcessInfo().GetUserID());
            if (match_info.GetProcessInfo().GroupIDIsValid())
                packet.Printf("gid:%u;",match_info.GetProcessInfo().GetGroupID());
            if (match_info.GetProcessInfo().EffectiveUserIDIsValid())
                packet.Printf("euid:%u;",match_info.GetProcessInfo().GetEffectiveUserID());
            if (match_info.GetProcessInfo().EffectiveGroupIDIsValid())
                packet.Printf("egid:%u;",match_info.GetProcessInfo().GetEffectiveGroupID());
            if (match_info.GetProcessInfo().EffectiveGroupIDIsValid())
                packet.Printf("all_users:%u;",match_info.GetMatchAllUsers() ? 1 : 0);
            if (match_info.GetProcessInfo().GetArchitecture().IsValid())
            {
                const ArchSpec &match_arch = match_info.GetProcessInfo().GetArchitecture();
                const llvm::Triple &triple = match_arch.GetTriple();
                packet.PutCString("triple:");
                packet.PutCStringAsRawHex8(triple.getTriple().c_str());
                packet.PutChar (';');
            }
        }
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false))
        {
            do
            {
                ProcessInstanceInfo process_info;
                if (!DecodeProcessInfoResponse (response, process_info))
                    break;
                process_infos.Append(process_info);
                response.GetStringRef().clear();
                response.SetFilePos(0);
            } while (SendPacketAndWaitForResponse ("qsProcessInfo", strlen ("qsProcessInfo"), response, false));
        }
        else
        {
            m_supports_qfProcessInfo = false;
            return 0;
        }
    }
    return process_infos.GetSize();
    
}

bool
GDBRemoteCommunicationClient::GetUserName (uint32_t uid, std::string &name)
{
    if (m_supports_qUserName)
    {
        char packet[32];
        const int packet_len = ::snprintf (packet, sizeof (packet), "qUserName:%i", uid);
        assert (packet_len < sizeof(packet));
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet, packet_len, response, false))
        {
            if (response.IsNormalResponse())
            {
                // Make sure we parsed the right number of characters. The response is
                // the hex encoded user name and should make up the entire packet.
                // If there are any non-hex ASCII bytes, the length won't match below..
                if (response.GetHexByteString (name) * 2 == response.GetStringRef().size())
                    return true;
            }
        }
        else
        {
            m_supports_qUserName = false;
            return false;
        }        
    }
    return false;

}

bool
GDBRemoteCommunicationClient::GetGroupName (uint32_t gid, std::string &name)
{
    if (m_supports_qGroupName)
    {
        char packet[32];
        const int packet_len = ::snprintf (packet, sizeof (packet), "qGroupName:%i", gid);
        assert (packet_len < sizeof(packet));
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet, packet_len, response, false))
        {
            if (response.IsNormalResponse())
            {
                // Make sure we parsed the right number of characters. The response is
                // the hex encoded group name and should make up the entire packet.
                // If there are any non-hex ASCII bytes, the length won't match below..
                if (response.GetHexByteString (name) * 2 == response.GetStringRef().size())
                    return true;
            }
        }
        else
        {
            m_supports_qGroupName = false;
            return false;
        }
    }
    return false;
}

void
GDBRemoteCommunicationClient::TestPacketSpeed (const uint32_t num_packets)
{
    uint32_t i;
    TimeValue start_time, end_time;
    uint64_t total_time_nsec;
    float packets_per_second;
    if (SendSpeedTestPacket (0, 0))
    {
        for (uint32_t send_size = 0; send_size <= 1024; send_size *= 2)
        {
            for (uint32_t recv_size = 0; recv_size <= 1024; recv_size *= 2)
            {
                start_time = TimeValue::Now();
                for (i=0; i<num_packets; ++i)
                {
                    SendSpeedTestPacket (send_size, recv_size);
                }
                end_time = TimeValue::Now();
                total_time_nsec = end_time.GetAsNanoSecondsSinceJan1_1970() - start_time.GetAsNanoSecondsSinceJan1_1970();
                packets_per_second = (((float)num_packets)/(float)total_time_nsec) * (float)TimeValue::NanoSecPerSec;
                printf ("%u qSpeedTest(send=%-5u, recv=%-5u) in %" PRIu64 ".%9.9" PRIu64 " sec for %f packets/sec.\n",
                        num_packets, 
                        send_size,
                        recv_size,
                        total_time_nsec / TimeValue::NanoSecPerSec,
                        total_time_nsec % TimeValue::NanoSecPerSec, 
                        packets_per_second);
                if (recv_size == 0)
                    recv_size = 32;
            }
            if (send_size == 0)
                send_size = 32;
        }
    }
    else
    {
        start_time = TimeValue::Now();
        for (i=0; i<num_packets; ++i)
        {
            GetCurrentProcessID ();
        }
        end_time = TimeValue::Now();
        total_time_nsec = end_time.GetAsNanoSecondsSinceJan1_1970() - start_time.GetAsNanoSecondsSinceJan1_1970();
        packets_per_second = (((float)num_packets)/(float)total_time_nsec) * (float)TimeValue::NanoSecPerSec;
        printf ("%u 'qC' packets packets in 0x%" PRIu64 "%9.9" PRIu64 " sec for %f packets/sec.\n",
                num_packets, 
                total_time_nsec / TimeValue::NanoSecPerSec, 
                total_time_nsec % TimeValue::NanoSecPerSec, 
                packets_per_second);
    }
}

bool
GDBRemoteCommunicationClient::SendSpeedTestPacket (uint32_t send_size, uint32_t recv_size)
{
    StreamString packet;
    packet.Printf ("qSpeedTest:response_size:%i;data:", recv_size);
    uint32_t bytes_left = send_size;
    while (bytes_left > 0)
    {
        if (bytes_left >= 26)
        {
            packet.PutCString("abcdefghijklmnopqrstuvwxyz");
            bytes_left -= 26;
        }
        else
        {
            packet.Printf ("%*.*s;", bytes_left, bytes_left, "abcdefghijklmnopqrstuvwxyz");
            bytes_left = 0;
        }
    }

    StringExtractorGDBRemote response;
    return SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false) > 0;
    return false;
}

uint16_t
GDBRemoteCommunicationClient::LaunchGDBserverAndGetPort ()
{
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse("qLaunchGDBServer", strlen("qLaunchGDBServer"), response, false))
    {
        std::string name;
        std::string value;
        uint16_t port = 0;
        //lldb::pid_t pid = LLDB_INVALID_PROCESS_ID;
        while (response.GetNameColonValue(name, value))
        {
            if (name.size() == 4 && name.compare("port") == 0)
                port = Args::StringToUInt32(value.c_str(), 0, 0);
//            if (name.size() == 3 && name.compare("pid") == 0)
//                pid = Args::StringToUInt32(value.c_str(), LLDB_INVALID_PROCESS_ID, 0);
        }
        return port;
    }
    return 0;
}

bool
GDBRemoteCommunicationClient::SetCurrentThread (uint64_t tid)
{
    if (m_curr_tid == tid)
        return true;

    char packet[32];
    int packet_len;
    if (tid == UINT64_MAX)
        packet_len = ::snprintf (packet, sizeof(packet), "Hg-1");
    else
        packet_len = ::snprintf (packet, sizeof(packet), "Hg%" PRIx64, tid);
    assert (packet_len + 1 < sizeof(packet));
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse(packet, packet_len, response, false))
    {
        if (response.IsOKResponse())
        {
            m_curr_tid = tid;
            return true;
        }
    }
    return false;
}

bool
GDBRemoteCommunicationClient::SetCurrentThreadForRun (uint64_t tid)
{
    if (m_curr_tid_run == tid)
        return true;

    char packet[32];
    int packet_len;
    if (tid == UINT64_MAX)
        packet_len = ::snprintf (packet, sizeof(packet), "Hc-1");
    else
        packet_len = ::snprintf (packet, sizeof(packet), "Hc%" PRIx64, tid);

    assert (packet_len + 1 < sizeof(packet));
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse(packet, packet_len, response, false))
    {
        if (response.IsOKResponse())
        {
            m_curr_tid_run = tid;
            return true;
        }
    }
    return false;
}

bool
GDBRemoteCommunicationClient::GetStopReply (StringExtractorGDBRemote &response)
{
    if (SendPacketAndWaitForResponse("?", 1, response, false))
        return response.IsNormalResponse();
    return false;
}

bool
GDBRemoteCommunicationClient::GetThreadStopInfo (lldb::tid_t tid, StringExtractorGDBRemote &response)
{
    if (m_supports_qThreadStopInfo)
    {
        char packet[256];
        int packet_len = ::snprintf(packet, sizeof(packet), "qThreadStopInfo%" PRIx64, tid);
        assert (packet_len < sizeof(packet));
        if (SendPacketAndWaitForResponse(packet, packet_len, response, false))
        {
            if (response.IsNormalResponse())
                return true;
            else
                return false;
        }
        else
        {
            m_supports_qThreadStopInfo = false;
        }
    }
//    if (SetCurrentThread (tid))
//        return GetStopReply (response);
    return false;
}


uint8_t
GDBRemoteCommunicationClient::SendGDBStoppointTypePacket (GDBStoppointType type, bool insert,  addr_t addr, uint32_t length)
{
    switch (type)
    {
    case eBreakpointSoftware:   if (!m_supports_z0) return UINT8_MAX; break;
    case eBreakpointHardware:   if (!m_supports_z1) return UINT8_MAX; break;
    case eWatchpointWrite:      if (!m_supports_z2) return UINT8_MAX; break;
    case eWatchpointRead:       if (!m_supports_z3) return UINT8_MAX; break;
    case eWatchpointReadWrite:  if (!m_supports_z4) return UINT8_MAX; break;
    }

    char packet[64];
    const int packet_len = ::snprintf (packet, 
                                       sizeof(packet), 
                                       "%c%i,%" PRIx64 ",%x",
                                       insert ? 'Z' : 'z', 
                                       type, 
                                       addr, 
                                       length);

    assert (packet_len + 1 < sizeof(packet));
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse(packet, packet_len, response, true))
    {
        if (response.IsOKResponse())
            return 0;
        else if (response.IsErrorResponse())
            return response.GetError();
    }
    else
    {
        switch (type)
        {
            case eBreakpointSoftware:   m_supports_z0 = false; break;
            case eBreakpointHardware:   m_supports_z1 = false; break;
            case eWatchpointWrite:      m_supports_z2 = false; break;
            case eWatchpointRead:       m_supports_z3 = false; break;
            case eWatchpointReadWrite:  m_supports_z4 = false; break;
        }
    }

    return UINT8_MAX;
}

size_t
GDBRemoteCommunicationClient::GetCurrentThreadIDs (std::vector<lldb::tid_t> &thread_ids, 
                                                   bool &sequence_mutex_unavailable)
{
    Mutex::Locker locker;
    thread_ids.clear();
    
    if (GetSequenceMutex (locker, "ProcessGDBRemote::UpdateThreadList() failed due to not getting the sequence mutex"))
    {
        sequence_mutex_unavailable = false;
        StringExtractorGDBRemote response;
        
        for (SendPacketNoLock ("qfThreadInfo", strlen("qfThreadInfo")) && WaitForPacketWithTimeoutMicroSecondsNoLock (response, GetPacketTimeoutInMicroSeconds ());
             response.IsNormalResponse();
             SendPacketNoLock ("qsThreadInfo", strlen("qsThreadInfo")) && WaitForPacketWithTimeoutMicroSecondsNoLock (response, GetPacketTimeoutInMicroSeconds ()))
        {
            char ch = response.GetChar();
            if (ch == 'l')
                break;
            if (ch == 'm')
            {
                do
                {
                    tid_t tid = response.GetHexMaxU64(false, LLDB_INVALID_THREAD_ID);
                    
                    if (tid != LLDB_INVALID_THREAD_ID)
                    {
                        thread_ids.push_back (tid);
                    }
                    ch = response.GetChar();    // Skip the command separator
                } while (ch == ',');            // Make sure we got a comma separator
            }
        }
    }
    else
    {
#if defined (LLDB_CONFIGURATION_DEBUG)
        // assert(!"ProcessGDBRemote::UpdateThreadList() failed due to not getting the sequence mutex");
#else
        Log *log (ProcessGDBRemoteLog::GetLogIfAnyCategoryIsSet (GDBR_LOG_PROCESS | GDBR_LOG_PACKETS));
        if (log)
            log->Printf("error: failed to get packet sequence mutex, not sending packet 'qfThreadInfo'");
#endif
        sequence_mutex_unavailable = true;
    }
    return thread_ids.size();
}

lldb::addr_t
GDBRemoteCommunicationClient::GetShlibInfoAddr()
{
    if (!IsRunning())
    {
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse("qShlibInfoAddr", ::strlen ("qShlibInfoAddr"), response, false))
        {
            if (response.IsNormalResponse())
                return response.GetHexMaxU64(false, LLDB_INVALID_ADDRESS);
        }
    }
    return LLDB_INVALID_ADDRESS;
}

