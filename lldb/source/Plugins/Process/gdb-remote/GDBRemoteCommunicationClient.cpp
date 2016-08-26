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
#include <math.h>
#include <sys/stat.h>

// C++ Includes
#include <sstream>
#include <numeric>

// Other libraries and framework includes
#include "lldb/Core/Log.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/State.h"
#include "lldb/Core/StreamGDBRemote.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/StringConvert.h"
#include "lldb/Host/TimeValue.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/UnixSignals.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Target/Target.h"

// Project includes
#include "Utility/StringExtractorGDBRemote.h"
#include "ProcessGDBRemote.h"
#include "ProcessGDBRemoteLog.h"
#include "lldb/Host/Config.h"

#if defined (HAVE_LIBCOMPRESSION)
#include <compression.h>
#endif

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_gdb_remote;

//----------------------------------------------------------------------
// GDBRemoteCommunicationClient constructor
//----------------------------------------------------------------------
GDBRemoteCommunicationClient::GDBRemoteCommunicationClient()
    : GDBRemoteClientBase("gdb-remote.client", "gdb-remote.client.rx_packet"),
      m_supports_not_sending_acks(eLazyBoolCalculate),
      m_supports_thread_suffix(eLazyBoolCalculate),
      m_supports_threads_in_stop_reply(eLazyBoolCalculate),
      m_supports_vCont_all(eLazyBoolCalculate),
      m_supports_vCont_any(eLazyBoolCalculate),
      m_supports_vCont_c(eLazyBoolCalculate),
      m_supports_vCont_C(eLazyBoolCalculate),
      m_supports_vCont_s(eLazyBoolCalculate),
      m_supports_vCont_S(eLazyBoolCalculate),
      m_qHostInfo_is_valid(eLazyBoolCalculate),
      m_curr_pid_is_valid(eLazyBoolCalculate),
      m_qProcessInfo_is_valid(eLazyBoolCalculate),
      m_qGDBServerVersion_is_valid(eLazyBoolCalculate),
      m_supports_alloc_dealloc_memory(eLazyBoolCalculate),
      m_supports_memory_region_info(eLazyBoolCalculate),
      m_supports_watchpoint_support_info(eLazyBoolCalculate),
      m_supports_detach_stay_stopped(eLazyBoolCalculate),
      m_watchpoints_trigger_after_instruction(eLazyBoolCalculate),
      m_attach_or_wait_reply(eLazyBoolCalculate),
      m_prepare_for_reg_writing_reply(eLazyBoolCalculate),
      m_supports_p(eLazyBoolCalculate),
      m_supports_x(eLazyBoolCalculate),
      m_avoid_g_packets(eLazyBoolCalculate),
      m_supports_QSaveRegisterState(eLazyBoolCalculate),
      m_supports_qXfer_auxv_read(eLazyBoolCalculate),
      m_supports_qXfer_libraries_read(eLazyBoolCalculate),
      m_supports_qXfer_libraries_svr4_read(eLazyBoolCalculate),
      m_supports_qXfer_features_read(eLazyBoolCalculate),
      m_supports_augmented_libraries_svr4_read(eLazyBoolCalculate),
      m_supports_jThreadExtendedInfo(eLazyBoolCalculate),
      m_supports_jLoadedDynamicLibrariesInfos(eLazyBoolCalculate),
      m_supports_jGetSharedCacheInfo(eLazyBoolCalculate),
      m_supports_qProcessInfoPID(true),
      m_supports_qfProcessInfo(true),
      m_supports_qUserName(true),
      m_supports_qGroupName(true),
      m_supports_qThreadStopInfo(true),
      m_supports_z0(true),
      m_supports_z1(true),
      m_supports_z2(true),
      m_supports_z3(true),
      m_supports_z4(true),
      m_supports_QEnvironment(true),
      m_supports_QEnvironmentHexEncoded(true),
      m_supports_qSymbol(true),
      m_qSymbol_requests_done(false),
      m_supports_qModuleInfo(true),
      m_supports_jThreadsInfo(true),
      m_curr_pid(LLDB_INVALID_PROCESS_ID),
      m_curr_tid(LLDB_INVALID_THREAD_ID),
      m_curr_tid_run(LLDB_INVALID_THREAD_ID),
      m_num_supported_hardware_watchpoints(0),
      m_host_arch(),
      m_process_arch(),
      m_os_version_major(UINT32_MAX),
      m_os_version_minor(UINT32_MAX),
      m_os_version_update(UINT32_MAX),
      m_os_build(),
      m_os_kernel(),
      m_hostname(),
      m_gdb_server_name(),
      m_gdb_server_version(UINT32_MAX),
      m_default_packet_timeout(0),
      m_max_packet_size(0),
      m_qSupported_response(),
      m_supported_async_json_packets_is_valid(false),
      m_supported_async_json_packets_sp()
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
    ResetDiscoverableSettings(false);

    // Start the read thread after we send the handshake ack since if we
    // fail to send the handshake ack, there is no reason to continue...
    if (SendAck())
    {
        // Wait for any responses that might have been queued up in the remote
        // GDB server and flush them all
        StringExtractorGDBRemote response;
        PacketResult packet_result = PacketResult::Success;
        const uint32_t timeout_usec = 10 * 1000; // Wait for 10 ms for a response
        while (packet_result == PacketResult::Success)
            packet_result = ReadPacket (response, timeout_usec, false);

        // The return value from QueryNoAckModeSupported() is true if the packet
        // was sent and _any_ response (including UNIMPLEMENTED) was received),
        // or false if no response was received. This quickly tells us if we have
        // a live connection to a remote GDB server...
        if (QueryNoAckModeSupported())
        {
            return true;
        }
        else
        {
            if (error_ptr)
                error_ptr->SetErrorString("failed to get reply to handshake packet");
        }
    }
    else
    {
        if (error_ptr)
            error_ptr->SetErrorString("failed to send the handshake ack");
    }
    return false;
}

bool
GDBRemoteCommunicationClient::GetEchoSupported ()
{
    if (m_supports_qEcho == eLazyBoolCalculate)
    {
        GetRemoteQSupported();
    }
    return m_supports_qEcho == eLazyBoolYes;
}


bool
GDBRemoteCommunicationClient::GetAugmentedLibrariesSVR4ReadSupported ()
{
    if (m_supports_augmented_libraries_svr4_read == eLazyBoolCalculate)
    {
        GetRemoteQSupported();
    }
    return m_supports_augmented_libraries_svr4_read == eLazyBoolYes;
}

bool
GDBRemoteCommunicationClient::GetQXferLibrariesSVR4ReadSupported ()
{
    if (m_supports_qXfer_libraries_svr4_read == eLazyBoolCalculate)
    {
        GetRemoteQSupported();
    }
    return m_supports_qXfer_libraries_svr4_read == eLazyBoolYes;
}

bool
GDBRemoteCommunicationClient::GetQXferLibrariesReadSupported ()
{
    if (m_supports_qXfer_libraries_read == eLazyBoolCalculate)
    {
        GetRemoteQSupported();
    }
    return m_supports_qXfer_libraries_read == eLazyBoolYes;
}

bool
GDBRemoteCommunicationClient::GetQXferAuxvReadSupported ()
{
    if (m_supports_qXfer_auxv_read == eLazyBoolCalculate)
    {
        GetRemoteQSupported();
    }
    return m_supports_qXfer_auxv_read == eLazyBoolYes;
}

bool
GDBRemoteCommunicationClient::GetQXferFeaturesReadSupported ()
{
    if (m_supports_qXfer_features_read == eLazyBoolCalculate)
    {
        GetRemoteQSupported();
    }
    return m_supports_qXfer_features_read == eLazyBoolYes;
}

uint64_t
GDBRemoteCommunicationClient::GetRemoteMaxPacketSize()
{
    if (m_max_packet_size == 0)
    {
        GetRemoteQSupported();
    }
    return m_max_packet_size;
}

bool
GDBRemoteCommunicationClient::QueryNoAckModeSupported ()
{
    if (m_supports_not_sending_acks == eLazyBoolCalculate)
    {
        m_send_acks = true;
        m_supports_not_sending_acks = eLazyBoolNo;

        // This is the first real packet that we'll send in a debug session and it may take a little
        // longer than normal to receive a reply.  Wait at least 6 seconds for a reply to this packet.

        const uint32_t minimum_timeout = 6;
        uint32_t old_timeout = GetPacketTimeoutInMicroSeconds() / lldb_private::TimeValue::MicroSecPerSec;
        GDBRemoteCommunication::ScopedTimeout timeout (*this, std::max (old_timeout, minimum_timeout));

        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse("QStartNoAckMode", response, false) == PacketResult::Success)
        {
            if (response.IsOKResponse())
            {
                m_send_acks = false;
                m_supports_not_sending_acks = eLazyBoolYes;
            }
            return true;
        }
    }
    return false;
}

void
GDBRemoteCommunicationClient::GetListThreadsInStopReplySupported ()
{
    if (m_supports_threads_in_stop_reply == eLazyBoolCalculate)
    {
        m_supports_threads_in_stop_reply = eLazyBoolNo;
        
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse("QListThreadsInStopReply", response, false) == PacketResult::Success)
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
        if (SendPacketAndWaitForResponse("qVAttachOrWaitSupported", response, false) == PacketResult::Success)
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
        if (SendPacketAndWaitForResponse("qSyncThreadStateSupported", response, false) == PacketResult::Success)
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
GDBRemoteCommunicationClient::ResetDiscoverableSettings (bool did_exec)
{
    if (did_exec == false)
    {
        // Hard reset everything, this is when we first connect to a GDB server
        m_supports_not_sending_acks = eLazyBoolCalculate;
        m_supports_thread_suffix = eLazyBoolCalculate;
        m_supports_threads_in_stop_reply = eLazyBoolCalculate;
        m_supports_vCont_c = eLazyBoolCalculate;
        m_supports_vCont_C = eLazyBoolCalculate;
        m_supports_vCont_s = eLazyBoolCalculate;
        m_supports_vCont_S = eLazyBoolCalculate;
        m_supports_p = eLazyBoolCalculate;
        m_supports_x = eLazyBoolCalculate;
        m_supports_QSaveRegisterState = eLazyBoolCalculate;
        m_qHostInfo_is_valid = eLazyBoolCalculate;
        m_curr_pid_is_valid = eLazyBoolCalculate;
        m_qGDBServerVersion_is_valid = eLazyBoolCalculate;
        m_supports_alloc_dealloc_memory = eLazyBoolCalculate;
        m_supports_memory_region_info = eLazyBoolCalculate;
        m_prepare_for_reg_writing_reply = eLazyBoolCalculate;
        m_attach_or_wait_reply = eLazyBoolCalculate;
        m_avoid_g_packets = eLazyBoolCalculate;
        m_supports_qXfer_auxv_read = eLazyBoolCalculate;
        m_supports_qXfer_libraries_read = eLazyBoolCalculate;
        m_supports_qXfer_libraries_svr4_read = eLazyBoolCalculate;
        m_supports_qXfer_features_read = eLazyBoolCalculate;
        m_supports_augmented_libraries_svr4_read = eLazyBoolCalculate;
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
        m_supports_QEnvironment = true;
        m_supports_QEnvironmentHexEncoded = true;
        m_supports_qSymbol = true;
        m_qSymbol_requests_done = false;
        m_supports_qModuleInfo = true;
        m_host_arch.Clear();
        m_os_version_major = UINT32_MAX;
        m_os_version_minor = UINT32_MAX;
        m_os_version_update = UINT32_MAX;
        m_os_build.clear();
        m_os_kernel.clear();
        m_hostname.clear();
        m_gdb_server_name.clear();
        m_gdb_server_version = UINT32_MAX;
        m_default_packet_timeout = 0;
        m_max_packet_size = 0;
        m_qSupported_response.clear();
        m_supported_async_json_packets_is_valid = false;
        m_supported_async_json_packets_sp.reset();
    }

    // These flags should be reset when we first connect to a GDB server
    // and when our inferior process execs
    m_qProcessInfo_is_valid = eLazyBoolCalculate;
    m_process_arch.Clear();
}

void
GDBRemoteCommunicationClient::GetRemoteQSupported ()
{
    // Clear out any capabilities we expect to see in the qSupported response
    m_supports_qXfer_auxv_read = eLazyBoolNo;
    m_supports_qXfer_libraries_read = eLazyBoolNo;
    m_supports_qXfer_libraries_svr4_read = eLazyBoolNo;
    m_supports_augmented_libraries_svr4_read = eLazyBoolNo;
    m_supports_qXfer_features_read = eLazyBoolNo;
    m_max_packet_size = UINT64_MAX;  // It's supposed to always be there, but if not, we assume no limit

    // build the qSupported packet
    std::vector<std::string> features = {"xmlRegisters=i386,arm,mips"};
    StreamString packet;
    packet.PutCString( "qSupported" );
    for ( uint32_t i = 0; i < features.size( ); ++i )
    {
        packet.PutCString( i==0 ? ":" : ";");
        packet.PutCString( features[i].c_str( ) );
    }

    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse(packet.GetData(),
                                     response,
                                     /*send_async=*/false) == PacketResult::Success)
    {
        const char *response_cstr = response.GetStringRef().c_str();

        // Hang on to the qSupported packet, so that platforms can do custom
        // configuration of the transport before attaching/launching the
        // process.
        m_qSupported_response = response_cstr;

        if (::strstr (response_cstr, "qXfer:auxv:read+"))
            m_supports_qXfer_auxv_read = eLazyBoolYes;
        if (::strstr (response_cstr, "qXfer:libraries-svr4:read+"))
            m_supports_qXfer_libraries_svr4_read = eLazyBoolYes;
        if (::strstr (response_cstr, "augmented-libraries-svr4-read"))
        {
            m_supports_qXfer_libraries_svr4_read = eLazyBoolYes;  // implied
            m_supports_augmented_libraries_svr4_read = eLazyBoolYes;
        }
        if (::strstr (response_cstr, "qXfer:libraries:read+"))
            m_supports_qXfer_libraries_read = eLazyBoolYes;
        if (::strstr (response_cstr, "qXfer:features:read+"))
            m_supports_qXfer_features_read = eLazyBoolYes;


        // Look for a list of compressions in the features list e.g.
        // qXfer:features:read+;PacketSize=20000;qEcho+;SupportedCompressions=zlib-deflate,lzma
        const char *features_list = ::strstr (response_cstr, "qXfer:features:");
        if (features_list)
        {
            const char *compressions = ::strstr (features_list, "SupportedCompressions=");
            if (compressions)
            {
                std::vector<std::string> supported_compressions;
                compressions += sizeof ("SupportedCompressions=") - 1;
                const char *end_of_compressions = strchr (compressions, ';');
                if (end_of_compressions == NULL)
                {
                    end_of_compressions = strchr (compressions, '\0');
                }
                const char *current_compression = compressions;
                while (current_compression < end_of_compressions)
                {
                    const char *next_compression_name = strchr (current_compression, ',');
                    const char *end_of_this_word = next_compression_name;
                    if (next_compression_name == NULL || end_of_compressions < next_compression_name)
                    {
                        end_of_this_word = end_of_compressions;
                    }

                    if (end_of_this_word)
                    {
                        if (end_of_this_word == current_compression)
                        {
                            current_compression++;
                        }
                        else
                        {
                            std::string this_compression (current_compression, end_of_this_word - current_compression);
                            supported_compressions.push_back (this_compression);
                            current_compression = end_of_this_word + 1;
                        }
                    }
                    else
                    {
                        supported_compressions.push_back (current_compression);
                        current_compression = end_of_compressions;
                    }
                }

                if (supported_compressions.size() > 0)
                {
                    MaybeEnableCompression (supported_compressions);
                }
            }
        }

        if (::strstr (response_cstr, "qEcho"))
            m_supports_qEcho = eLazyBoolYes;
        else
            m_supports_qEcho = eLazyBoolNo;

        const char *packet_size_str = ::strstr (response_cstr, "PacketSize=");
        if (packet_size_str)
        {
            StringExtractorGDBRemote packet_response(packet_size_str + strlen("PacketSize="));
            m_max_packet_size = packet_response.GetHexMaxU64(/*little_endian=*/false, UINT64_MAX);
            if (m_max_packet_size == 0)
            {
                m_max_packet_size = UINT64_MAX;  // Must have been a garbled response
                Log *log (ProcessGDBRemoteLog::GetLogIfAllCategoriesSet (GDBR_LOG_PROCESS));
                if (log)
                    log->Printf ("Garbled PacketSize spec in qSupported response");
            }
        }
    }
}

void
GDBRemoteCommunicationClient::ComputeThreadSuffixSupport()
{
    if (m_supports_thread_suffix != eLazyBoolCalculate)
        return;

    StringExtractorGDBRemote response;
    m_supports_thread_suffix = eLazyBoolNo;
    if (SendPacketAndWaitForResponse("QThreadSuffixSupported", response, false) == PacketResult::Success)
    {
        if (response.IsOKResponse())
            m_supports_thread_suffix = eLazyBoolYes;
    }
}

bool
GDBRemoteCommunicationClient::GetThreadSuffixSupported()
{
    return m_supports_thread_suffix == eLazyBoolYes;
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
        if (SendPacketAndWaitForResponse("vCont?", response, false) == PacketResult::Success)
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

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationClient::SendThreadSpecificPacketAndWaitForResponse(lldb::tid_t tid, StreamString &&payload,
                                                                         StringExtractorGDBRemote &response,
                                                                         const Lock &lock)
{
    if (GetThreadSuffixSupported())
        payload.Printf(";thread:%4.4" PRIx64 ";", tid);
    else
    {
        if (!SetCurrentThread(tid, lock))
            return PacketResult::ErrorSendFailed;
    }

    return SendPacketAndWaitForResponse(payload.GetString(), response, lock);
}

// Check if the target supports 'p' packet. It sends out a 'p'
// packet and checks the response. A normal packet will tell us
// that support is available.
//
// Takes a valid thread ID because p needs to apply to a thread.
bool
GDBRemoteCommunicationClient::GetpPacketSupported (lldb::tid_t tid)
{
    if (m_supports_p == eLazyBoolCalculate)
    {
        Lock lock(*this, false);
        if (!lock)
        {
            Log *log(ProcessGDBRemoteLog::GetLogIfAnyCategoryIsSet(GDBR_LOG_PROCESS | GDBR_LOG_PACKETS));
            if (log)
                log->Printf("GDBRemoteCommunicationClient::%s failed to get sequence mutex", __FUNCTION__);
            return false;
        }

        m_supports_p = eLazyBoolNo;
        StreamString payload;
        payload.PutCString("p0");
        StringExtractorGDBRemote response;
        if (SendThreadSpecificPacketAndWaitForResponse(tid, std::move(payload), response, lock) ==
                PacketResult::Success &&
            response.IsNormalResponse())
        {
            m_supports_p = eLazyBoolYes;
        }
    }
    return m_supports_p;
}

StructuredData::ObjectSP
GDBRemoteCommunicationClient::GetThreadsInfo()
{
    // Get information on all threads at one using the "jThreadsInfo" packet
    StructuredData::ObjectSP object_sp;

    if (m_supports_jThreadsInfo)
    {
        StringExtractorGDBRemote response;
        response.SetResponseValidatorToJSON();
        if (SendPacketAndWaitForResponse("jThreadsInfo", response, false) == PacketResult::Success)
        {
            if (response.IsUnsupportedResponse())
            {
                m_supports_jThreadsInfo = false;
            }
            else if (!response.Empty())
            {
                object_sp = StructuredData::ParseJSON (response.GetStringRef());
            }
        }
    }
    return object_sp;
}


bool
GDBRemoteCommunicationClient::GetThreadExtendedInfoSupported ()
{
    if (m_supports_jThreadExtendedInfo == eLazyBoolCalculate)
    {
        StringExtractorGDBRemote response;
        m_supports_jThreadExtendedInfo = eLazyBoolNo;
        if (SendPacketAndWaitForResponse("jThreadExtendedInfo:", response, false) == PacketResult::Success)
        {
            if (response.IsOKResponse())
            {
                m_supports_jThreadExtendedInfo = eLazyBoolYes;
            }
        }
    }
    return m_supports_jThreadExtendedInfo;
}

bool
GDBRemoteCommunicationClient::GetLoadedDynamicLibrariesInfosSupported ()
{
    if (m_supports_jLoadedDynamicLibrariesInfos == eLazyBoolCalculate)
    {
        StringExtractorGDBRemote response;
        m_supports_jLoadedDynamicLibrariesInfos = eLazyBoolNo;
        if (SendPacketAndWaitForResponse("jGetLoadedDynamicLibrariesInfos:", response, false) == PacketResult::Success)
        {
            if (response.IsOKResponse())
            {
                m_supports_jLoadedDynamicLibrariesInfos = eLazyBoolYes;
            }
        }
    }
    return m_supports_jLoadedDynamicLibrariesInfos;
}

bool
GDBRemoteCommunicationClient::GetSharedCacheInfoSupported ()
{
    if (m_supports_jGetSharedCacheInfo == eLazyBoolCalculate)
    {
        StringExtractorGDBRemote response;
        m_supports_jGetSharedCacheInfo = eLazyBoolNo;
        if (SendPacketAndWaitForResponse("jGetSharedCacheInfo:", response, false) == PacketResult::Success)
        {
            if (response.IsOKResponse())
            {
                m_supports_jGetSharedCacheInfo = eLazyBoolYes;
            }
        }
    }
    return m_supports_jGetSharedCacheInfo;
}

bool
GDBRemoteCommunicationClient::GetxPacketSupported ()
{
    if (m_supports_x == eLazyBoolCalculate)
    {
        StringExtractorGDBRemote response;
        m_supports_x = eLazyBoolNo;
        char packet[256];
        snprintf (packet, sizeof (packet), "x0,0");
        if (SendPacketAndWaitForResponse(packet, response, false) == PacketResult::Success)
        {
            if (response.IsOKResponse())
                m_supports_x = eLazyBoolYes;
        }
    }
    return m_supports_x;
}

GDBRemoteCommunicationClient::PacketResult
GDBRemoteCommunicationClient::SendPacketsAndConcatenateResponses
(
    const char *payload_prefix,
    std::string &response_string
)
{
    Lock lock(*this, false);
    if (!lock)
    {
        Log *log (ProcessGDBRemoteLog::GetLogIfAnyCategoryIsSet (GDBR_LOG_PROCESS | GDBR_LOG_PACKETS));
        if (log)
            log->Printf("error: failed to get packet sequence mutex, not sending packets with prefix '%s'",
                        payload_prefix);
        return PacketResult::ErrorNoSequenceLock;
    }

    response_string = "";
    std::string payload_prefix_str(payload_prefix);
    unsigned int response_size = 0x1000;
    if (response_size > GetRemoteMaxPacketSize()) {  // May send qSupported packet
        response_size = GetRemoteMaxPacketSize();
    }

    for (unsigned int offset = 0; true; offset += response_size)
    {
        StringExtractorGDBRemote this_response;
        // Construct payload
        char sizeDescriptor[128];
        snprintf(sizeDescriptor, sizeof(sizeDescriptor), "%x,%x", offset, response_size);
        PacketResult result = SendPacketAndWaitForResponse(payload_prefix_str + sizeDescriptor, this_response, lock);
        if (result != PacketResult::Success)
            return result;

        const std::string &this_string = this_response.GetStringRef();

        // Check for m or l as first character; l seems to mean this is the last chunk
        char first_char = *this_string.c_str();
        if (first_char != 'm' && first_char != 'l')
        {
            return PacketResult::ErrorReplyInvalid;
        }
        // Concatenate the result so far (skipping 'm' or 'l')
        response_string.append(this_string, 1, std::string::npos);
        if (first_char == 'l')
            // We're done
            return PacketResult::Success;
    }
}

lldb::pid_t
GDBRemoteCommunicationClient::GetCurrentProcessID (bool allow_lazy)
{
    if (allow_lazy && m_curr_pid_is_valid == eLazyBoolYes)
        return m_curr_pid;
    
    // First try to retrieve the pid via the qProcessInfo request.
    GetCurrentProcessInfo (allow_lazy);
    if (m_curr_pid_is_valid == eLazyBoolYes)
    {
        // We really got it.
        return m_curr_pid;
    }
    else
    {
        // If we don't get a response for qProcessInfo, check if $qC gives us a result.
        // $qC only returns a real process id on older debugserver and lldb-platform stubs.
        // The gdb remote protocol documents $qC as returning the thread id, which newer
        // debugserver and lldb-gdbserver stubs return correctly.
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse("qC", strlen("qC"), response, false) == PacketResult::Success)
        {
            if (response.GetChar() == 'Q')
            {
                if (response.GetChar() == 'C')
                {
                    m_curr_pid = response.GetHexMaxU32 (false, LLDB_INVALID_PROCESS_ID);
                    if (m_curr_pid != LLDB_INVALID_PROCESS_ID)
                    {
                        m_curr_pid_is_valid = eLazyBoolYes;
                        return m_curr_pid;
                    }
                }
            }
        }

        // If we don't get a response for $qC, check if $qfThreadID gives us a result.
        if (m_curr_pid == LLDB_INVALID_PROCESS_ID)
        {
            std::vector<lldb::tid_t> thread_ids;
            bool sequence_mutex_unavailable;
            size_t size;
            size = GetCurrentThreadIDs (thread_ids, sequence_mutex_unavailable);
            if (size && sequence_mutex_unavailable == false)
            {
                m_curr_pid = thread_ids.front();
                m_curr_pid_is_valid = eLazyBoolYes;
                return m_curr_pid;
            }
        }
    }
    
    return LLDB_INVALID_PROCESS_ID;
}

bool
GDBRemoteCommunicationClient::GetLaunchSuccess (std::string &error_str)
{
    error_str.clear();
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse("qLaunchSuccess", strlen("qLaunchSuccess"), response, false) == PacketResult::Success)
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
GDBRemoteCommunicationClient::SendArgumentsPacket (const ProcessLaunchInfo &launch_info)
{
    // Since we don't get the send argv0 separate from the executable path, we need to
    // make sure to use the actual executable path found in the launch_info...
    std::vector<const char *> argv;
    FileSpec exe_file = launch_info.GetExecutableFile();
    std::string exe_path;
    const char *arg = NULL;
    const Args &launch_args = launch_info.GetArguments();
    if (exe_file)
        exe_path = exe_file.GetPath(false);
    else
    {
        arg = launch_args.GetArgumentAtIndex(0);
        if (arg)
            exe_path = arg;
    }
    if (!exe_path.empty())
    {
        argv.push_back(exe_path.c_str());
        for (uint32_t i=1; (arg = launch_args.GetArgumentAtIndex(i)) != NULL; ++i)
        {
            if (arg)
                argv.push_back(arg);
        }
    }
    if (!argv.empty())
    {
        StreamString packet;
        packet.PutChar('A');
        for (size_t i = 0, n = argv.size(); i < n; ++i)
        {
            arg = argv[i];
            const int arg_len = strlen(arg);
            if (i > 0)
                packet.PutChar(',');
            packet.Printf("%i,%i,", arg_len * 2, (int)i);
            packet.PutBytesAsRawHex8 (arg, arg_len);
        }

        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false) == PacketResult::Success)
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
        bool send_hex_encoding = false;
        for (const char *p = name_equal_value; *p != '\0' && send_hex_encoding == false; ++p)
        {
            if (isprint(*p))
            {
                switch (*p)
                {
                    case '$':
                    case '#':
                    case '*':
                    case '}':
                        send_hex_encoding = true;
                        break;
                    default:
                        break;
                }
            }
            else
            {
                // We have non printable characters, lets hex encode this...
                send_hex_encoding = true;
            }
        }
        
        StringExtractorGDBRemote response;
        if (send_hex_encoding)
        {
            if (m_supports_QEnvironmentHexEncoded)
            {
                packet.PutCString("QEnvironmentHexEncoded:");
                packet.PutBytesAsRawHex8 (name_equal_value, strlen(name_equal_value));
                if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false) == PacketResult::Success)
                {
                    if (response.IsOKResponse())
                        return 0;
                    uint8_t error = response.GetError();
                    if (error)
                        return error;
                    if (response.IsUnsupportedResponse())
                        m_supports_QEnvironmentHexEncoded = false;
                }
            }
            
        }
        else if (m_supports_QEnvironment)
        {
            packet.Printf("QEnvironment:%s", name_equal_value);
            if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false) == PacketResult::Success)
            {
                if (response.IsOKResponse())
                    return 0;
                uint8_t error = response.GetError();
                if (error)
                    return error;
                if (response.IsUnsupportedResponse())
                    m_supports_QEnvironment = false;
            }
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
        if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false) == PacketResult::Success)
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
GDBRemoteCommunicationClient::SendLaunchEventDataPacket (char const *data, bool *was_supported)
{
    if (data && *data != '\0')
    {
        StreamString packet;
        packet.Printf("QSetProcessEvent:%s", data);
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false) == PacketResult::Success)
        {
            if (response.IsOKResponse())
            {
                if (was_supported)
                    *was_supported = true;
                return 0;
            }
            else if (response.IsUnsupportedResponse())
            {
                if (was_supported)
                    *was_supported = false;
                return -1;
            }
            else
            {
                uint8_t error = response.GetError();
                if (was_supported)
                    *was_supported = true;
                if (error)
                    return error;
            }
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
GDBRemoteCommunicationClient::GetGDBServerVersion()
{
    if (m_qGDBServerVersion_is_valid == eLazyBoolCalculate)
    {
        m_gdb_server_name.clear();
        m_gdb_server_version = 0;
        m_qGDBServerVersion_is_valid = eLazyBoolNo;

        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse ("qGDBServerVersion", response, false) == PacketResult::Success)
        {
            if (response.IsNormalResponse())
            {
                std::string name;
                std::string value;
                bool success = false;
                while (response.GetNameColonValue(name, value))
                {
                    if (name.compare("name") == 0)
                    {
                        success = true;
                        m_gdb_server_name.swap(value);
                    }
                    else if (name.compare("version") == 0)
                    {
                        size_t dot_pos = value.find('.');
                        if (dot_pos != std::string::npos)
                            value[dot_pos] = '\0';
                        const uint32_t version = StringConvert::ToUInt32(value.c_str(), UINT32_MAX, 0);
                        if (version != UINT32_MAX)
                        {
                            success = true;
                            m_gdb_server_version = version;
                        }
                    }
                }
                if (success)
                    m_qGDBServerVersion_is_valid = eLazyBoolYes;
            }
        }
    }
    return m_qGDBServerVersion_is_valid == eLazyBoolYes;
}

void
GDBRemoteCommunicationClient::MaybeEnableCompression (std::vector<std::string> supported_compressions)
{
    CompressionType avail_type = CompressionType::None;
    std::string avail_name;

#if defined (HAVE_LIBCOMPRESSION)
    // libcompression is weak linked so test if compression_decode_buffer() is available
    if (compression_decode_buffer != NULL && avail_type == CompressionType::None)
    {
        for (auto compression : supported_compressions)
        {
            if (compression == "lzfse")
            {
                avail_type = CompressionType::LZFSE;
                avail_name = compression;
                break;
            }
        }
    }
#endif

#if defined (HAVE_LIBCOMPRESSION)
    // libcompression is weak linked so test if compression_decode_buffer() is available
    if (compression_decode_buffer != NULL && avail_type == CompressionType::None)
    {
        for (auto compression : supported_compressions)
        {
            if (compression == "zlib-deflate")
            {
                avail_type = CompressionType::ZlibDeflate;
                avail_name = compression;
                break;
            }
        }
    }
#endif

#if defined (HAVE_LIBZ)
    if (avail_type == CompressionType::None)
    {
        for (auto compression : supported_compressions)
        {
            if (compression == "zlib-deflate")
            {
                avail_type = CompressionType::ZlibDeflate;
                avail_name = compression;
                break;
            }
        }
    }
#endif

#if defined (HAVE_LIBCOMPRESSION)
    // libcompression is weak linked so test if compression_decode_buffer() is available
    if (compression_decode_buffer != NULL && avail_type == CompressionType::None)
    {
        for (auto compression : supported_compressions)
        {
            if (compression == "lz4")
            {
                avail_type = CompressionType::LZ4;
                avail_name = compression;
                break;
            }
        }
    }
#endif

#if defined (HAVE_LIBCOMPRESSION)
    // libcompression is weak linked so test if compression_decode_buffer() is available
    if (compression_decode_buffer != NULL && avail_type == CompressionType::None)
    {
        for (auto compression : supported_compressions)
        {
            if (compression == "lzma")
            {
                avail_type = CompressionType::LZMA;
                avail_name = compression;
                break;
            }
        }
    }
#endif

    if (avail_type != CompressionType::None)
    {
        StringExtractorGDBRemote response;
        std::string packet = "QEnableCompression:type:" + avail_name + ";";
        if (SendPacketAndWaitForResponse (packet.c_str(), response, false) !=  PacketResult::Success)
            return;
    
        if (response.IsOKResponse())
        {
            m_compression_type = avail_type;
        }
    }
}

const char *
GDBRemoteCommunicationClient::GetGDBServerProgramName()
{
    if (GetGDBServerVersion())
    {
        if (!m_gdb_server_name.empty())
            return m_gdb_server_name.c_str();
    }
    return NULL;
}

uint32_t
GDBRemoteCommunicationClient::GetGDBServerProgramVersion()
{
    if (GetGDBServerVersion())
        return m_gdb_server_version;
    return 0;
}

bool
GDBRemoteCommunicationClient::GetDefaultThreadId (lldb::tid_t &tid)
{
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse("qC",response,false) !=  PacketResult::Success)
        return false;

    if (!response.IsNormalResponse())
        return false;

    if (response.GetChar() == 'Q' && response.GetChar() == 'C')
        tid = response.GetHexMaxU32(true, -1);

    return true;
}

bool
GDBRemoteCommunicationClient::GetHostInfo (bool force)
{
    Log *log (ProcessGDBRemoteLog::GetLogIfAnyCategoryIsSet (GDBR_LOG_PROCESS));

    if (force || m_qHostInfo_is_valid == eLazyBoolCalculate)
    {
        m_qHostInfo_is_valid = eLazyBoolNo;
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse ("qHostInfo", response, false) == PacketResult::Success)
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
                std::string distribution_id;
                uint32_t pointer_byte_size = 0;
                StringExtractor extractor;
                ByteOrder byte_order = eByteOrderInvalid;
                uint32_t num_keys_decoded = 0;
                while (response.GetNameColonValue(name, value))
                {
                    if (name.compare("cputype") == 0)
                    {
                        // exception type in big endian hex
                        cpu = StringConvert::ToUInt32 (value.c_str(), LLDB_INVALID_CPUTYPE, 0);
                        if (cpu != LLDB_INVALID_CPUTYPE)
                            ++num_keys_decoded;
                    }
                    else if (name.compare("cpusubtype") == 0)
                    {
                        // exception count in big endian hex
                        sub = StringConvert::ToUInt32 (value.c_str(), 0, 0);
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
                        extractor.GetStringRef ().swap (value);
                        extractor.SetFilePos(0);
                        extractor.GetHexByteString (triple);
                        ++num_keys_decoded;
                    }
                    else if (name.compare ("distribution_id") == 0)
                    {
                        extractor.GetStringRef ().swap (value);
                        extractor.SetFilePos (0);
                        extractor.GetHexByteString (distribution_id);
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
                        pointer_byte_size = StringConvert::ToUInt32 (value.c_str(), 0, 0);
                        if (pointer_byte_size != 0)
                            ++num_keys_decoded;
                    }
                    else if ((name.compare("os_version") == 0) ||
                             (name.compare("version") == 0)) // Older debugserver binaries used the "version" key instead of "os_version"...
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
                    else if (name.compare("default_packet_timeout") == 0)
                    {
                        m_default_packet_timeout = StringConvert::ToUInt32(value.c_str(), 0);
                        if (m_default_packet_timeout > 0)
                        {
                            SetPacketTimeout(m_default_packet_timeout);
                            ++num_keys_decoded;
                        }
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
                                case llvm::Triple::aarch64:
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

                    if (log)
                        log->Printf ("GDBRemoteCommunicationClient::%s parsed host architecture as %s, triple as %s from triple text %s", __FUNCTION__, m_host_arch.GetArchitectureName () ? m_host_arch.GetArchitectureName () : "<null-arch-name>", m_host_arch.GetTriple ().getTriple ().c_str(), triple.c_str ());
                }
                if (!distribution_id.empty ())
                    m_host_arch.SetDistributionId (distribution_id.c_str ());
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
        assert (packet_len < (int)sizeof(packet));
        if (SendPacketAndWaitForResponse (packet, packet_len, response, false) == PacketResult::Success)
        {
            if (response.IsErrorResponse())
                return response.GetError();
            return 0;
        }
    }
    return -1;
}

int
GDBRemoteCommunicationClient::SendStdinNotification (const char* data, size_t data_len)
{
    StreamString packet;
    packet.PutCString("I");
    packet.PutBytesAsRawHex8(data, data_len);
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false) == PacketResult::Success)
    {
        return 0;
    }
    return response.GetError();

}

const lldb_private::ArchSpec &
GDBRemoteCommunicationClient::GetHostArchitecture ()
{
    if (m_qHostInfo_is_valid == eLazyBoolCalculate)
        GetHostInfo ();
    return m_host_arch;
}

uint32_t
GDBRemoteCommunicationClient::GetHostDefaultPacketTimeout ()
{
    if (m_qHostInfo_is_valid == eLazyBoolCalculate)
        GetHostInfo ();
    return m_default_packet_timeout;
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
        assert (packet_len < (int)sizeof(packet));
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet, packet_len, response, false) == PacketResult::Success)
        {
            if (response.IsUnsupportedResponse())
                m_supports_alloc_dealloc_memory = eLazyBoolNo;
            else if (!response.IsErrorResponse())
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
        assert (packet_len < (int)sizeof(packet));
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet, packet_len, response, false) == PacketResult::Success)
        {
            if (response.IsUnsupportedResponse())
                m_supports_alloc_dealloc_memory = eLazyBoolNo;
            else if (response.IsOKResponse())
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
            assert (packet_len < (int)sizeof(packet));
            StringExtractorGDBRemote response;
            if (SendPacketAndWaitForResponse (packet, packet_len, response, false) == PacketResult::Success
                  && response.IsOKResponse())
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
            StringExtractorGDBRemote response;
            PacketResult packet_result = SendPacketAndWaitForResponse ("D1", 2, response, false);
            if (packet_result != PacketResult::Success)
                error.SetErrorString ("Sending extended disconnect packet failed.");
        }
    }
    else
    {
        StringExtractorGDBRemote response;
        PacketResult packet_result = SendPacketAndWaitForResponse ("D", 1, response, false);
        if (packet_result != PacketResult::Success)
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
        assert (packet_len < (int)sizeof(packet));
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet, packet_len, response, false) == PacketResult::Success)
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
                    addr_value = StringConvert::ToUInt64(value.c_str(), LLDB_INVALID_ADDRESS, 16, &success);
                    if (success)
                        region_info.GetRange().SetRangeBase(addr_value);
                }
                else if (name.compare ("size") == 0)
                {
                    addr_value = StringConvert::ToUInt64(value.c_str(), 0, 16, &success);
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

                        region_info.SetMapped(MemoryRegionInfo::eYes);
                    }
                    else
                    {
                        // The reported region does not contain this address -- we're looking at an unmapped page
                        region_info.SetReadable (MemoryRegionInfo::eNo);
                        region_info.SetWritable (MemoryRegionInfo::eNo);
                        region_info.SetExecutable (MemoryRegionInfo::eNo);
                        region_info.SetMapped(MemoryRegionInfo::eNo);
                    }
                }
                else if (name.compare ("name") == 0)
                {
                    StringExtractorGDBRemote name_extractor;
                    name_extractor.GetStringRef().swap(value);
                    name_extractor.GetHexByteString(value);
                    region_info.SetName(value.c_str());
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
                region_info.SetMapped(MemoryRegionInfo::eNo);
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
        assert (packet_len < (int)sizeof(packet));
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet, packet_len, response, false) == PacketResult::Success)
        {
            m_supports_watchpoint_support_info = eLazyBoolYes;        
            std::string name;
            std::string value;
            while (response.GetNameColonValue(name, value))
            {
                if (name.compare ("num") == 0)
                {
                    num = StringConvert::ToUInt32(value.c_str(), 0, 0);
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
GDBRemoteCommunicationClient::GetWatchpointSupportInfo (uint32_t &num, bool& after, const ArchSpec &arch)
{
    Error error(GetWatchpointSupportInfo(num));
    if (error.Success())
        error = GetWatchpointsTriggerAfterInstruction(after, arch);
    return error;
}

lldb_private::Error
GDBRemoteCommunicationClient::GetWatchpointsTriggerAfterInstruction (bool &after, const ArchSpec &arch)
{
    Error error;
    llvm::Triple::ArchType atype = arch.GetMachine();
    
    // we assume watchpoints will happen after running the relevant opcode
    // and we only want to override this behavior if we have explicitly
    // received a qHostInfo telling us otherwise
    if (m_qHostInfo_is_valid != eLazyBoolYes)
    {
        // On targets like MIPS, watchpoint exceptions are always generated 
        // before the instruction is executed. The connected target may not 
        // support qHostInfo or qWatchpointSupportInfo packets.
        if (atype == llvm::Triple::mips || atype == llvm::Triple::mipsel
            || atype == llvm::Triple::mips64 || atype == llvm::Triple::mips64el)
            after = false;
        else
            after = true;
    }
    else
    {
        // For MIPS, set m_watchpoints_trigger_after_instruction to eLazyBoolNo 
        // if it is not calculated before.
        if (m_watchpoints_trigger_after_instruction == eLazyBoolCalculate &&
            (atype == llvm::Triple::mips || atype == llvm::Triple::mipsel
            || atype == llvm::Triple::mips64 || atype == llvm::Triple::mips64el))
            m_watchpoints_trigger_after_instruction = eLazyBoolNo;

        after = (m_watchpoints_trigger_after_instruction != eLazyBoolNo);
    }
    return error;
}

int
GDBRemoteCommunicationClient::SetSTDIN(const FileSpec &file_spec)
{
    if (file_spec)
    {
        std::string path{file_spec.GetPath(false)};
        StreamString packet;
        packet.PutCString("QSetSTDIN:");
        packet.PutCStringAsRawHex8(path.c_str());

        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false) == PacketResult::Success)
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
GDBRemoteCommunicationClient::SetSTDOUT(const FileSpec &file_spec)
{
    if (file_spec)
    {
        std::string path{file_spec.GetPath(false)};
        StreamString packet;
        packet.PutCString("QSetSTDOUT:");
        packet.PutCStringAsRawHex8(path.c_str());

        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false) == PacketResult::Success)
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
GDBRemoteCommunicationClient::SetSTDERR(const FileSpec &file_spec)
{
    if (file_spec)
    {
        std::string path{file_spec.GetPath(false)};
        StreamString packet;
        packet.PutCString("QSetSTDERR:");
        packet.PutCStringAsRawHex8(path.c_str());

        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false) == PacketResult::Success)
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
GDBRemoteCommunicationClient::GetWorkingDir(FileSpec &working_dir)
{
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse ("qGetWorkingDir", response, false) == PacketResult::Success)
    {
        if (response.IsUnsupportedResponse())
            return false;
        if (response.IsErrorResponse())
            return false;
        std::string cwd;
        response.GetHexByteString(cwd);
        working_dir.SetFile(cwd, false, GetHostArchitecture());
        return !cwd.empty();
    }
    return false;
}

int
GDBRemoteCommunicationClient::SetWorkingDir(const FileSpec &working_dir)
{
    if (working_dir)
    {
        std::string path{working_dir.GetPath(false)};
        StreamString packet;
        packet.PutCString("QSetWorkingDir:");
        packet.PutCStringAsRawHex8(path.c_str());

        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false) == PacketResult::Success)
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
    assert (packet_len < (int)sizeof(packet));
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse (packet, packet_len, response, false) == PacketResult::Success)
    {
        if (response.IsOKResponse())
            return 0;
        uint8_t error = response.GetError();
        if (error)
            return error;
    }
    return -1;
}

int
GDBRemoteCommunicationClient::SetDetachOnError (bool enable)
{
    char packet[32];
    const int packet_len = ::snprintf (packet, sizeof (packet), "QSetDetachOnError:%i", enable ? 1 : 0);
    assert (packet_len < (int)sizeof(packet));
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse (packet, packet_len, response, false) == PacketResult::Success)
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

        uint32_t cpu = LLDB_INVALID_CPUTYPE;
        uint32_t sub = 0;
        std::string vendor;
        std::string os_type;
        
        while (response.GetNameColonValue(name, value))
        {
            if (name.compare("pid") == 0)
            {
                process_info.SetProcessID (StringConvert::ToUInt32 (value.c_str(), LLDB_INVALID_PROCESS_ID, 0));
            }
            else if (name.compare("ppid") == 0)
            {
                process_info.SetParentProcessID (StringConvert::ToUInt32 (value.c_str(), LLDB_INVALID_PROCESS_ID, 0));
            }
            else if (name.compare("uid") == 0)
            {
                process_info.SetUserID (StringConvert::ToUInt32 (value.c_str(), UINT32_MAX, 0));
            }
            else if (name.compare("euid") == 0)
            {
                process_info.SetEffectiveUserID (StringConvert::ToUInt32 (value.c_str(), UINT32_MAX, 0));
            }
            else if (name.compare("gid") == 0)
            {
                process_info.SetGroupID (StringConvert::ToUInt32 (value.c_str(), UINT32_MAX, 0));
            }
            else if (name.compare("egid") == 0)
            {
                process_info.SetEffectiveGroupID (StringConvert::ToUInt32 (value.c_str(), UINT32_MAX, 0));
            }
            else if (name.compare("triple") == 0)
            {
                StringExtractor extractor;
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
            else if (name.compare("cputype") == 0)
            {
                cpu = StringConvert::ToUInt32 (value.c_str(), LLDB_INVALID_CPUTYPE, 16);
            }
            else if (name.compare("cpusubtype") == 0)
            {
                sub = StringConvert::ToUInt32 (value.c_str(), 0, 16);
            }
            else if (name.compare("vendor") == 0)
            {
                vendor = value;
            }
            else if (name.compare("ostype") == 0)
            {
                os_type = value;
            }
        }

        if (cpu != LLDB_INVALID_CPUTYPE && !vendor.empty() && !os_type.empty())
        {
            if (vendor == "apple")
            {
                process_info.GetArchitecture().SetArchitecture (eArchTypeMachO, cpu, sub);
                process_info.GetArchitecture().GetTriple().setVendorName (llvm::StringRef (vendor));
                process_info.GetArchitecture().GetTriple().setOSName (llvm::StringRef (os_type));
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
        assert (packet_len < (int)sizeof(packet));
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet, packet_len, response, false) == PacketResult::Success)
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
GDBRemoteCommunicationClient::GetCurrentProcessInfo (bool allow_lazy)
{
    Log *log (ProcessGDBRemoteLog::GetLogIfAnyCategoryIsSet (GDBR_LOG_PROCESS | GDBR_LOG_PACKETS));

    if (allow_lazy)
    {
        if (m_qProcessInfo_is_valid == eLazyBoolYes)
            return true;
        if (m_qProcessInfo_is_valid == eLazyBoolNo)
            return false;
    }

    GetHostInfo ();

    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse ("qProcessInfo", response, false) == PacketResult::Success)
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
            lldb::pid_t pid = LLDB_INVALID_PROCESS_ID;
            while (response.GetNameColonValue(name, value))
            {
                if (name.compare("cputype") == 0)
                {
                    cpu = StringConvert::ToUInt32 (value.c_str(), LLDB_INVALID_CPUTYPE, 16);
                    if (cpu != LLDB_INVALID_CPUTYPE)
                        ++num_keys_decoded;
                }
                else if (name.compare("cpusubtype") == 0)
                {
                    sub = StringConvert::ToUInt32 (value.c_str(), 0, 16);
                    if (sub != 0)
                        ++num_keys_decoded;
                }
                else if (name.compare("triple") == 0)
                {
                    StringExtractor extractor;
                    extractor.GetStringRef().swap(value);
                    extractor.SetFilePos(0);
                    extractor.GetHexByteString (triple);
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
                    pointer_byte_size = StringConvert::ToUInt32 (value.c_str(), 0, 16);
                    if (pointer_byte_size != 0)
                        ++num_keys_decoded;
                }
                else if (name.compare("pid") == 0)
                {
                    pid = StringConvert::ToUInt64(value.c_str(), 0, 16);
                    if (pid != LLDB_INVALID_PROCESS_ID)
                        ++num_keys_decoded;
                }
            }
            if (num_keys_decoded > 0)
                m_qProcessInfo_is_valid = eLazyBoolYes;
            if (pid != LLDB_INVALID_PROCESS_ID)
            {
                m_curr_pid_is_valid = eLazyBoolYes;
                m_curr_pid = pid;
            }

            // Set the ArchSpec from the triple if we have it.
            if (!triple.empty ())
            {
                m_process_arch.SetTriple (triple.c_str ());
                if (pointer_byte_size)
                {
                    assert (pointer_byte_size == m_process_arch.GetAddressByteSize());
                }
            }
            else if (cpu != LLDB_INVALID_CPUTYPE && !os_name.empty() && !vendor_name.empty())
            {
                llvm::Triple triple(llvm::Twine("-") + vendor_name + "-" + os_name);

                assert(triple.getObjectFormat() != llvm::Triple::UnknownObjectFormat);
                switch (triple.getObjectFormat()) {
                    case llvm::Triple::MachO:
                        m_process_arch.SetArchitecture (eArchTypeMachO, cpu, sub);
                        break;
                    case llvm::Triple::ELF:
                        m_process_arch.SetArchitecture (eArchTypeELF, cpu, sub);
                        break;
                    case llvm::Triple::COFF:
                        m_process_arch.SetArchitecture (eArchTypeCOFF, cpu, sub);
                        break;
                    case llvm::Triple::UnknownObjectFormat:
                        if (log)
                            log->Printf("error: failed to determine target architecture");
                        return false;
                }

                if (pointer_byte_size)
                {
                    assert (pointer_byte_size == m_process_arch.GetAddressByteSize());
                }
                if (byte_order != eByteOrderInvalid)
                {
                    assert (byte_order == m_process_arch.GetByteOrder());
                }
                m_process_arch.GetTriple().setVendorName (llvm::StringRef (vendor_name));
                m_process_arch.GetTriple().setOSName(llvm::StringRef (os_name));
                m_host_arch.GetTriple().setVendorName (llvm::StringRef (vendor_name));
                m_host_arch.GetTriple().setOSName (llvm::StringRef (os_name));
            }
            return true;
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
                packet.PutCString(triple.getTriple().c_str());
                packet.PutChar (';');
            }
        }
        StringExtractorGDBRemote response;
        // Increase timeout as the first qfProcessInfo packet takes a long time
        // on Android. The value of 1min was arrived at empirically.
        GDBRemoteCommunication::ScopedTimeout timeout (*this, 60);
        if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false) == PacketResult::Success)
        {
            do
            {
                ProcessInstanceInfo process_info;
                if (!DecodeProcessInfoResponse (response, process_info))
                    break;
                process_infos.Append(process_info);
                response.GetStringRef().clear();
                response.SetFilePos(0);
            } while (SendPacketAndWaitForResponse ("qsProcessInfo", strlen ("qsProcessInfo"), response, false) == PacketResult::Success);
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
        assert (packet_len < (int)sizeof(packet));
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet, packet_len, response, false) == PacketResult::Success)
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
        assert (packet_len < (int)sizeof(packet));
        StringExtractorGDBRemote response;
        if (SendPacketAndWaitForResponse (packet, packet_len, response, false) == PacketResult::Success)
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

bool
GDBRemoteCommunicationClient::SetNonStopMode (const bool enable)
{
    // Form non-stop packet request
    char packet[32];
    const int packet_len = ::snprintf(packet, sizeof(packet), "QNonStop:%1d", (int)enable);
    assert(packet_len < (int)sizeof(packet));

    StringExtractorGDBRemote response;
    // Send to target
    if (SendPacketAndWaitForResponse(packet, packet_len, response, false) == PacketResult::Success)
        if (response.IsOKResponse())
            return true;

    // Failed or not supported
    return false;

}

static void
MakeSpeedTestPacket(StreamString &packet, uint32_t send_size, uint32_t recv_size)
{
    packet.Clear();
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
}

template<typename T>
T calculate_standard_deviation(const std::vector<T> &v)
{
    T sum = std::accumulate(std::begin(v), std::end(v), T(0));
    T mean =  sum / (T)v.size();
    T accum = T(0);
    std::for_each (std::begin(v), std::end(v), [&](const T d) {
        T delta = d - mean;
        accum += delta * delta;
    });

    T stdev = sqrt(accum / (v.size()-1));
    return stdev;
}

void
GDBRemoteCommunicationClient::TestPacketSpeed (const uint32_t num_packets, uint32_t max_send, uint32_t max_recv, bool json, Stream &strm)
{
    uint32_t i;
    TimeValue start_time, end_time;
    uint64_t total_time_nsec;
    if (SendSpeedTestPacket (0, 0))
    {
        StreamString packet;
        if (json)
            strm.Printf("{ \"packet_speeds\" : {\n    \"num_packets\" : %u,\n    \"results\" : [", num_packets);
        else
            strm.Printf("Testing sending %u packets of various sizes:\n", num_packets);
        strm.Flush();

        uint32_t result_idx = 0;
        uint32_t send_size;
        std::vector<float> packet_times;

        for (send_size = 0; send_size <= max_send; send_size ? send_size *= 2 : send_size = 4)
        {
            for (uint32_t recv_size = 0; recv_size <= max_recv; recv_size ? recv_size *= 2 : recv_size = 4)
            {
                MakeSpeedTestPacket (packet, send_size, recv_size);

                packet_times.clear();
                // Test how long it takes to send 'num_packets' packets
                start_time = TimeValue::Now();
                for (i=0; i<num_packets; ++i)
                {
                    TimeValue packet_start_time = TimeValue::Now();
                    StringExtractorGDBRemote response;
                    SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false);
                    TimeValue packet_end_time = TimeValue::Now();
                    uint64_t packet_time_nsec = packet_end_time.GetAsNanoSecondsSinceJan1_1970() - packet_start_time.GetAsNanoSecondsSinceJan1_1970();
                    packet_times.push_back((float)packet_time_nsec);
                }
                end_time = TimeValue::Now();
                total_time_nsec = end_time.GetAsNanoSecondsSinceJan1_1970() - start_time.GetAsNanoSecondsSinceJan1_1970();

                float packets_per_second = (((float)num_packets)/(float)total_time_nsec) * (float)TimeValue::NanoSecPerSec;
                float total_ms = (float)total_time_nsec/(float)TimeValue::NanoSecPerMilliSec;
                float average_ms_per_packet = total_ms / num_packets;
                const float standard_deviation = calculate_standard_deviation<float>(packet_times);
                if (json)
                {
                    strm.Printf ("%s\n     {\"send_size\" : %6" PRIu32 ", \"recv_size\" : %6" PRIu32 ", \"total_time_nsec\" : %12" PRIu64 ", \"standard_deviation_nsec\" : %9" PRIu64 " }", result_idx > 0 ? "," : "", send_size, recv_size, total_time_nsec, (uint64_t)standard_deviation);
                    ++result_idx;
                }
                else
                {
                    strm.Printf ("qSpeedTest(send=%-7u, recv=%-7u) in %" PRIu64 ".%9.9" PRIu64 " sec for %9.2f packets/sec (%10.6f ms per packet) with standard deviation of %10.6f ms\n",
                                 send_size,
                                 recv_size,
                                 total_time_nsec / TimeValue::NanoSecPerSec,
                                 total_time_nsec % TimeValue::NanoSecPerSec,
                                 packets_per_second,
                                 average_ms_per_packet,
                                 standard_deviation/(float)TimeValue::NanoSecPerMilliSec);
                }
                strm.Flush();
            }
        }

        const uint64_t k_recv_amount = 4*1024*1024; // Receive amount in bytes

        const float k_recv_amount_mb = (float)k_recv_amount/(1024.0f*1024.0f);
        if (json)
            strm.Printf("\n    ]\n  },\n  \"download_speed\" : {\n    \"byte_size\" : %" PRIu64 ",\n    \"results\" : [", k_recv_amount);
        else
            strm.Printf("Testing receiving %2.1fMB of data using varying receive packet sizes:\n", k_recv_amount_mb);
        strm.Flush();
        send_size = 0;
        result_idx = 0;
        for (uint32_t recv_size = 32; recv_size <= max_recv; recv_size *= 2)
        {
            MakeSpeedTestPacket (packet, send_size, recv_size);

            // If we have a receive size, test how long it takes to receive 4MB of data
            if (recv_size > 0)
            {
                start_time = TimeValue::Now();
                uint32_t bytes_read = 0;
                uint32_t packet_count = 0;
                while (bytes_read < k_recv_amount)
                {
                    StringExtractorGDBRemote response;
                    SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false);
                    bytes_read += recv_size;
                    ++packet_count;
                }
                end_time = TimeValue::Now();
                total_time_nsec = end_time.GetAsNanoSecondsSinceJan1_1970() - start_time.GetAsNanoSecondsSinceJan1_1970();
                float mb_second = ((((float)k_recv_amount)/(float)total_time_nsec) * (float)TimeValue::NanoSecPerSec) / (1024.0*1024.0);
                float packets_per_second = (((float)packet_count)/(float)total_time_nsec) * (float)TimeValue::NanoSecPerSec;
                float total_ms = (float)total_time_nsec/(float)TimeValue::NanoSecPerMilliSec;
                float average_ms_per_packet = total_ms / packet_count;

                if (json)
                {
                    strm.Printf ("%s\n     {\"send_size\" : %6" PRIu32 ", \"recv_size\" : %6" PRIu32 ", \"total_time_nsec\" : %12" PRIu64 " }", result_idx > 0 ? "," : "", send_size, recv_size, total_time_nsec);
                    ++result_idx;
                }
                else
                {
                    strm.Printf ("qSpeedTest(send=%-7u, recv=%-7u) %6u packets needed to receive %2.1fMB in %" PRIu64 ".%9.9" PRIu64 " sec for %f MB/sec for %9.2f packets/sec (%10.6f ms per packet)\n",
                                 send_size,
                                 recv_size,
                                 packet_count,
                                 k_recv_amount_mb,
                                 total_time_nsec / TimeValue::NanoSecPerSec,
                                 total_time_nsec % TimeValue::NanoSecPerSec,
                                 mb_second,
                                 packets_per_second,
                                 average_ms_per_packet);
                }
                strm.Flush();
            }
        }
        if (json)
            strm.Printf("\n    ]\n  }\n}\n");
        else
            strm.EOL();
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
    return SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false)  == PacketResult::Success;
}

bool
GDBRemoteCommunicationClient::LaunchGDBServer (const char *remote_accept_hostname,
                                               lldb::pid_t &pid,
                                               uint16_t &port,
                                               std::string &socket_name)
{
    pid = LLDB_INVALID_PROCESS_ID;
    port = 0;
    socket_name.clear();

    StringExtractorGDBRemote response;
    StreamString stream;
    stream.PutCString("qLaunchGDBServer;");
    std::string hostname;
    if (remote_accept_hostname  && remote_accept_hostname[0])
        hostname = remote_accept_hostname;
    else
    {
        if (HostInfo::GetHostname(hostname))
        {
            // Make the GDB server we launch only accept connections from this host
            stream.Printf("host:%s;", hostname.c_str());
        }
        else
        {
            // Make the GDB server we launch accept connections from any host since we can't figure out the hostname
            stream.Printf("host:*;");
        }
    }
    const char *packet = stream.GetData();
    int packet_len = stream.GetSize();

    // give the process a few seconds to startup
    GDBRemoteCommunication::ScopedTimeout timeout (*this, 10);

    if (SendPacketAndWaitForResponse(packet, packet_len, response, false) == PacketResult::Success)
    {
        std::string name;
        std::string value;
        StringExtractor extractor;
        while (response.GetNameColonValue(name, value))
        {
            if (name.compare("port") == 0)
                port = StringConvert::ToUInt32(value.c_str(), 0, 0);
            else if (name.compare("pid") == 0)
                pid = StringConvert::ToUInt64(value.c_str(), LLDB_INVALID_PROCESS_ID, 0);
            else if (name.compare("socket_name") == 0)
            {
                extractor.GetStringRef().swap(value);
                extractor.SetFilePos(0);
                extractor.GetHexByteString(value);

                socket_name = value;
            }
        }
        return true;
    }
    return false;
}

size_t
GDBRemoteCommunicationClient::QueryGDBServer (std::vector<std::pair<uint16_t, std::string>>& connection_urls)
{
    connection_urls.clear();

    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse("qQueryGDBServer", response, false) != PacketResult::Success)
        return 0;

    StructuredData::ObjectSP data = StructuredData::ParseJSON(response.GetStringRef());
    if (!data)
        return 0;

    StructuredData::Array* array = data->GetAsArray();
    if (!array)
        return 0;

    for (size_t i = 0, count = array->GetSize(); i < count; ++i)
    {
        StructuredData::Dictionary* element = nullptr;
        if (!array->GetItemAtIndexAsDictionary(i, element))
            continue;

        uint16_t port = 0;
        if (StructuredData::ObjectSP port_osp = element->GetValueForKey(llvm::StringRef("port")))
            port = port_osp->GetIntegerValue(0);

        std::string socket_name;
        if (StructuredData::ObjectSP socket_name_osp = element->GetValueForKey(llvm::StringRef("socket_name")))
            socket_name = socket_name_osp->GetStringValue();

        if (port != 0 || !socket_name.empty())
            connection_urls.emplace_back(port, socket_name);
    }
    return connection_urls.size();
}

bool
GDBRemoteCommunicationClient::KillSpawnedProcess (lldb::pid_t pid)
{
    StreamString stream;
    stream.Printf ("qKillSpawnedProcess:%" PRId64 , pid);
    const char *packet = stream.GetData();
    int packet_len = stream.GetSize();

    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse(packet, packet_len, response, false) == PacketResult::Success)
    {
        if (response.IsOKResponse())
            return true;
    }
    return false;
}

bool
GDBRemoteCommunicationClient::SetCurrentThread(uint64_t tid, const Lock &lock)
{
    if (m_curr_tid == tid)
        return true;

    char packet[32];
    int packet_len;
    if (tid == UINT64_MAX)
        packet_len = ::snprintf (packet, sizeof(packet), "Hg-1");
    else
        packet_len = ::snprintf (packet, sizeof(packet), "Hg%" PRIx64, tid);
    assert (packet_len + 1 < (int)sizeof(packet));
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse(llvm::StringRef(packet, packet_len), response, lock) == PacketResult::Success)
    {
        if (response.IsOKResponse())
        {
            m_curr_tid = tid;
            return true;
        }

        /*
         * Connected bare-iron target (like YAMON gdb-stub) may not have support for Hg packet.
         * The reply from '?' packet could be as simple as 'S05'. There is no packet which can
         * give us pid and/or tid. Assume pid=tid=1 in such cases.
        */
        if (response.IsUnsupportedResponse() && IsConnected())
        {
            m_curr_tid = 1;
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

    assert (packet_len + 1 < (int)sizeof(packet));
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse(packet, packet_len, response, false) == PacketResult::Success)
    {
        if (response.IsOKResponse())
        {
            m_curr_tid_run = tid;
            return true;
        }

        /*
         * Connected bare-iron target (like YAMON gdb-stub) may not have support for Hc packet.
         * The reply from '?' packet could be as simple as 'S05'. There is no packet which can
         * give us pid and/or tid. Assume pid=tid=1 in such cases.
        */
        if (response.IsUnsupportedResponse() && IsConnected())
        {
            m_curr_tid_run = 1;
            return true;
        }
    }
    return false;
}

bool
GDBRemoteCommunicationClient::GetStopReply (StringExtractorGDBRemote &response)
{
    if (SendPacketAndWaitForResponse("?", 1, response, false) == PacketResult::Success)
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
        assert (packet_len < (int)sizeof(packet));
        if (SendPacketAndWaitForResponse(packet, packet_len, response, false) == PacketResult::Success)
        {
            if (response.IsUnsupportedResponse())
                m_supports_qThreadStopInfo = false;
            else if (response.IsNormalResponse())
                return true;
            else
                return false;
        }
        else
        {
            m_supports_qThreadStopInfo = false;
        }
    }
    return false;
}


uint8_t
GDBRemoteCommunicationClient::SendGDBStoppointTypePacket (GDBStoppointType type, bool insert,  addr_t addr, uint32_t length)
{
    Log *log (GetLogIfAnyCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));
    if (log)
        log->Printf ("GDBRemoteCommunicationClient::%s() %s at addr = 0x%" PRIx64,
                     __FUNCTION__, insert ? "add" : "remove", addr);

    // Check if the stub is known not to support this breakpoint type
    if (!SupportsGDBStoppointPacket(type))
        return UINT8_MAX;
    // Construct the breakpoint packet
    char packet[64];
    const int packet_len = ::snprintf (packet, 
                                       sizeof(packet), 
                                       "%c%i,%" PRIx64 ",%x",
                                       insert ? 'Z' : 'z', 
                                       type, 
                                       addr, 
                                       length);
    // Check we haven't overwritten the end of the packet buffer
    assert (packet_len + 1 < (int)sizeof(packet));
    StringExtractorGDBRemote response;
    // Make sure the response is either "OK", "EXX" where XX are two hex digits, or "" (unsupported)
    response.SetResponseValidatorToOKErrorNotSupported();
    // Try to send the breakpoint packet, and check that it was correctly sent
    if (SendPacketAndWaitForResponse(packet, packet_len, response, true) == PacketResult::Success)
    {
        // Receive and OK packet when the breakpoint successfully placed
        if (response.IsOKResponse())
            return 0;

        // Error while setting breakpoint, send back specific error
        if (response.IsErrorResponse())
            return response.GetError();

        // Empty packet informs us that breakpoint is not supported
        if (response.IsUnsupportedResponse())
        {
            // Disable this breakpoint type since it is unsupported
            switch (type)
            {
            case eBreakpointSoftware:   m_supports_z0 = false; break;
            case eBreakpointHardware:   m_supports_z1 = false; break;
            case eWatchpointWrite:      m_supports_z2 = false; break;
            case eWatchpointRead:       m_supports_z3 = false; break;
            case eWatchpointReadWrite:  m_supports_z4 = false; break;
            case eStoppointInvalid:     return UINT8_MAX;
            }
        }
    }
    // Signal generic failure
    return UINT8_MAX;
}

size_t
GDBRemoteCommunicationClient::GetCurrentThreadIDs (std::vector<lldb::tid_t> &thread_ids, 
                                                   bool &sequence_mutex_unavailable)
{
    thread_ids.clear();

    Lock lock(*this, false);
    if (lock)
    {
        sequence_mutex_unavailable = false;
        StringExtractorGDBRemote response;
        
        PacketResult packet_result;
        for (packet_result = SendPacketAndWaitForResponse("qfThreadInfo", response, lock);
             packet_result == PacketResult::Success && response.IsNormalResponse();
             packet_result = SendPacketAndWaitForResponse("qsThreadInfo", response, lock))
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

        /*
         * Connected bare-iron target (like YAMON gdb-stub) may not have support for
         * qProcessInfo, qC and qfThreadInfo packets. The reply from '?' packet could
         * be as simple as 'S05'. There is no packet which can give us pid and/or tid.
         * Assume pid=tid=1 in such cases.
        */
        if (response.IsUnsupportedResponse() && thread_ids.size() == 0 && IsConnected())
        {
            thread_ids.push_back (1);
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
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse("qShlibInfoAddr", response, false) != PacketResult::Success || !response.IsNormalResponse())
        return LLDB_INVALID_ADDRESS;
    return response.GetHexMaxU64(false, LLDB_INVALID_ADDRESS);
}

lldb_private::Error
GDBRemoteCommunicationClient::RunShellCommand(const char *command,           // Shouldn't be NULL
                                              const FileSpec &working_dir,   // Pass empty FileSpec to use the current working directory
                                              int *status_ptr,               // Pass NULL if you don't want the process exit status
                                              int *signo_ptr,                // Pass NULL if you don't want the signal that caused the process to exit
                                              std::string *command_output,   // Pass NULL if you don't want the command output
                                              uint32_t timeout_sec)          // Timeout in seconds to wait for shell program to finish
{
    lldb_private::StreamString stream;
    stream.PutCString("qPlatform_shell:");
    stream.PutBytesAsRawHex8(command, strlen(command));
    stream.PutChar(',');
    stream.PutHex32(timeout_sec);
    if (working_dir)
    {
        std::string path{working_dir.GetPath(false)};
        stream.PutChar(',');
        stream.PutCStringAsRawHex8(path.c_str());
    }
    const char *packet = stream.GetData();
    int packet_len = stream.GetSize();
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse(packet, packet_len, response, false) == PacketResult::Success)
    {
        if (response.GetChar() != 'F')
            return Error("malformed reply");
        if (response.GetChar() != ',')
            return Error("malformed reply");
        uint32_t exitcode = response.GetHexMaxU32(false, UINT32_MAX);
        if (exitcode == UINT32_MAX)
            return Error("unable to run remote process");
        else if (status_ptr)
            *status_ptr = exitcode;
        if (response.GetChar() != ',')
            return Error("malformed reply");
        uint32_t signo = response.GetHexMaxU32(false, UINT32_MAX);
        if (signo_ptr)
            *signo_ptr = signo;
        if (response.GetChar() != ',')
            return Error("malformed reply");
        std::string output;
        response.GetEscapedBinaryData(output);
        if (command_output)
            command_output->assign(output);
        return Error();
    }
    return Error("unable to send packet");
}

Error
GDBRemoteCommunicationClient::MakeDirectory(const FileSpec &file_spec,
                                            uint32_t file_permissions)
{
    std::string path{file_spec.GetPath(false)};
    lldb_private::StreamString stream;
    stream.PutCString("qPlatform_mkdir:");
    stream.PutHex32(file_permissions);
    stream.PutChar(',');
    stream.PutCStringAsRawHex8(path.c_str());
    const char *packet = stream.GetData();
    int packet_len = stream.GetSize();
    StringExtractorGDBRemote response;

    if (SendPacketAndWaitForResponse(packet, packet_len, response, false) != PacketResult::Success)
        return Error("failed to send '%s' packet", packet);

    if (response.GetChar() != 'F')
        return Error("invalid response to '%s' packet", packet);

    return Error(response.GetU32(UINT32_MAX), eErrorTypePOSIX);
}

Error
GDBRemoteCommunicationClient::SetFilePermissions(const FileSpec &file_spec,
                                                 uint32_t file_permissions)
{
    std::string path{file_spec.GetPath(false)};
    lldb_private::StreamString stream;
    stream.PutCString("qPlatform_chmod:");
    stream.PutHex32(file_permissions);
    stream.PutChar(',');
    stream.PutCStringAsRawHex8(path.c_str());
    const char *packet = stream.GetData();
    int packet_len = stream.GetSize();
    StringExtractorGDBRemote response;

    if (SendPacketAndWaitForResponse(packet, packet_len, response, false) != PacketResult::Success)
        return Error("failed to send '%s' packet", packet);

    if (response.GetChar() != 'F')
        return Error("invalid response to '%s' packet", packet);

    return Error(response.GetU32(UINT32_MAX), eErrorTypePOSIX);
}

static uint64_t
ParseHostIOPacketResponse (StringExtractorGDBRemote &response,
                           uint64_t fail_result,
                           Error &error)
{
    response.SetFilePos(0);
    if (response.GetChar() != 'F')
        return fail_result;
    int32_t result = response.GetS32 (-2);
    if (result == -2)
        return fail_result;
    if (response.GetChar() == ',')
    {
        int result_errno = response.GetS32 (-2);
        if (result_errno != -2)
            error.SetError(result_errno, eErrorTypePOSIX);
        else
            error.SetError(-1, eErrorTypeGeneric);
    }
    else
        error.Clear();
    return  result;
}
lldb::user_id_t
GDBRemoteCommunicationClient::OpenFile (const lldb_private::FileSpec& file_spec,
                                        uint32_t flags,
                                        mode_t mode,
                                        Error &error)
{
    std::string path(file_spec.GetPath(false));
    lldb_private::StreamString stream;
    stream.PutCString("vFile:open:");
    if (path.empty())
        return UINT64_MAX;
    stream.PutCStringAsRawHex8(path.c_str());
    stream.PutChar(',');
    stream.PutHex32(flags);
    stream.PutChar(',');
    stream.PutHex32(mode);
    const char* packet = stream.GetData();
    int packet_len = stream.GetSize();
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse(packet, packet_len, response, false) == PacketResult::Success)
    {
        return ParseHostIOPacketResponse (response, UINT64_MAX, error);
    }
    return UINT64_MAX;
}

bool
GDBRemoteCommunicationClient::CloseFile (lldb::user_id_t fd,
                                         Error &error)
{
    lldb_private::StreamString stream;
    stream.Printf("vFile:close:%i", (int)fd);
    const char* packet = stream.GetData();
    int packet_len = stream.GetSize();
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse(packet, packet_len, response, false) == PacketResult::Success)
    {
        return ParseHostIOPacketResponse (response, -1, error) == 0;
    }
    return false;
}

// Extension of host I/O packets to get the file size.
lldb::user_id_t
GDBRemoteCommunicationClient::GetFileSize (const lldb_private::FileSpec& file_spec)
{
    std::string path(file_spec.GetPath(false));
    lldb_private::StreamString stream;
    stream.PutCString("vFile:size:");
    stream.PutCStringAsRawHex8(path.c_str());
    const char* packet = stream.GetData();
    int packet_len = stream.GetSize();
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse(packet, packet_len, response, false) == PacketResult::Success)
    {
        if (response.GetChar() != 'F')
            return UINT64_MAX;
        uint32_t retcode = response.GetHexMaxU64(false, UINT64_MAX);
        return retcode;
    }
    return UINT64_MAX;
}

Error
GDBRemoteCommunicationClient::GetFilePermissions(const FileSpec &file_spec,
                                                 uint32_t &file_permissions)
{
    std::string path{file_spec.GetPath(false)};
    Error error;
    lldb_private::StreamString stream;
    stream.PutCString("vFile:mode:");
    stream.PutCStringAsRawHex8(path.c_str());
    const char* packet = stream.GetData();
    int packet_len = stream.GetSize();
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse(packet, packet_len, response, false) == PacketResult::Success)
    {
        if (response.GetChar() != 'F')
        {
            error.SetErrorStringWithFormat ("invalid response to '%s' packet", packet);
        }
        else
        {
            const uint32_t mode = response.GetS32(-1);
            if (static_cast<int32_t>(mode) == -1)
            {
                if (response.GetChar() == ',')
                {
                    int response_errno = response.GetS32(-1);
                    if (response_errno > 0)
                        error.SetError(response_errno, lldb::eErrorTypePOSIX);
                    else
                        error.SetErrorToGenericError();
                }
                else
                    error.SetErrorToGenericError();
            }
            else
            {
                file_permissions = mode & (S_IRWXU|S_IRWXG|S_IRWXO);
            }
        }
    }
    else
    {
        error.SetErrorStringWithFormat ("failed to send '%s' packet", packet);
    }
    return error;
}

uint64_t
GDBRemoteCommunicationClient::ReadFile (lldb::user_id_t fd,
                                        uint64_t offset,
                                        void *dst,
                                        uint64_t dst_len,
                                        Error &error)
{
    lldb_private::StreamString stream;
    stream.Printf("vFile:pread:%i,%" PRId64 ",%" PRId64, (int)fd, dst_len, offset);
    const char* packet = stream.GetData();
    int packet_len = stream.GetSize();
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse(packet, packet_len, response, false) == PacketResult::Success)
    {
        if (response.GetChar() != 'F')
            return 0;
        uint32_t retcode = response.GetHexMaxU32(false, UINT32_MAX);
        if (retcode == UINT32_MAX)
            return retcode;
        const char next = (response.Peek() ? *response.Peek() : 0);
        if (next == ',')
            return 0;
        if (next == ';')
        {
            response.GetChar(); // skip the semicolon
            std::string buffer;
            if (response.GetEscapedBinaryData(buffer))
            {
                const uint64_t data_to_write = std::min<uint64_t>(dst_len, buffer.size());
                if (data_to_write > 0)
                    memcpy(dst, &buffer[0], data_to_write);
                return data_to_write;
            }
        }
    }
    return 0;
}

uint64_t
GDBRemoteCommunicationClient::WriteFile (lldb::user_id_t fd,
                                         uint64_t offset,
                                         const void* src,
                                         uint64_t src_len,
                                         Error &error)
{
    lldb_private::StreamGDBRemote stream;
    stream.Printf("vFile:pwrite:%i,%" PRId64 ",", (int)fd, offset);
    stream.PutEscapedBytes(src, src_len);
    const char* packet = stream.GetData();
    int packet_len = stream.GetSize();
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse(packet, packet_len, response, false) == PacketResult::Success)
    {
        if (response.GetChar() != 'F')
        {
            error.SetErrorStringWithFormat("write file failed");
            return 0;
        }
        uint64_t bytes_written = response.GetU64(UINT64_MAX);
        if (bytes_written == UINT64_MAX)
        {
            error.SetErrorToGenericError();
            if (response.GetChar() == ',')
            {
                int response_errno = response.GetS32(-1);
                if (response_errno > 0)
                    error.SetError(response_errno, lldb::eErrorTypePOSIX);
            }
            return 0;
        }
        return bytes_written;
    }
    else
    {
        error.SetErrorString ("failed to send vFile:pwrite packet");
    }
    return 0;
}

Error
GDBRemoteCommunicationClient::CreateSymlink(const FileSpec &src, const FileSpec &dst)
{
    std::string src_path{src.GetPath(false)},
                dst_path{dst.GetPath(false)};
    Error error;
    lldb_private::StreamGDBRemote stream;
    stream.PutCString("vFile:symlink:");
    // the unix symlink() command reverses its parameters where the dst if first,
    // so we follow suit here
    stream.PutCStringAsRawHex8(dst_path.c_str());
    stream.PutChar(',');
    stream.PutCStringAsRawHex8(src_path.c_str());
    const char* packet = stream.GetData();
    int packet_len = stream.GetSize();
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse(packet, packet_len, response, false) == PacketResult::Success)
    {
        if (response.GetChar() == 'F')
        {
            uint32_t result = response.GetU32(UINT32_MAX);
            if (result != 0)
            {
                error.SetErrorToGenericError();
                if (response.GetChar() == ',')
                {
                    int response_errno = response.GetS32(-1);
                    if (response_errno > 0)
                        error.SetError(response_errno, lldb::eErrorTypePOSIX);
                }
            }
        }
        else
        {
            // Should have returned with 'F<result>[,<errno>]'
            error.SetErrorStringWithFormat("symlink failed");
        }
    }
    else
    {
        error.SetErrorString ("failed to send vFile:symlink packet");
    }
    return error;
}

Error
GDBRemoteCommunicationClient::Unlink(const FileSpec &file_spec)
{
    std::string path{file_spec.GetPath(false)};
    Error error;
    lldb_private::StreamGDBRemote stream;
    stream.PutCString("vFile:unlink:");
    // the unix symlink() command reverses its parameters where the dst if first,
    // so we follow suit here
    stream.PutCStringAsRawHex8(path.c_str());
    const char* packet = stream.GetData();
    int packet_len = stream.GetSize();
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse(packet, packet_len, response, false) == PacketResult::Success)
    {
        if (response.GetChar() == 'F')
        {
            uint32_t result = response.GetU32(UINT32_MAX);
            if (result != 0)
            {
                error.SetErrorToGenericError();
                if (response.GetChar() == ',')
                {
                    int response_errno = response.GetS32(-1);
                    if (response_errno > 0)
                        error.SetError(response_errno, lldb::eErrorTypePOSIX);
                }
            }
        }
        else
        {
            // Should have returned with 'F<result>[,<errno>]'
            error.SetErrorStringWithFormat("unlink failed");
        }
    }
    else
    {
        error.SetErrorString ("failed to send vFile:unlink packet");
    }
    return error;
}

// Extension of host I/O packets to get whether a file exists.
bool
GDBRemoteCommunicationClient::GetFileExists (const lldb_private::FileSpec& file_spec)
{
    std::string path(file_spec.GetPath(false));
    lldb_private::StreamString stream;
    stream.PutCString("vFile:exists:");
    stream.PutCStringAsRawHex8(path.c_str());
    const char* packet = stream.GetData();
    int packet_len = stream.GetSize();
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse(packet, packet_len, response, false) == PacketResult::Success)
    {
        if (response.GetChar() != 'F')
            return false;
        if (response.GetChar() != ',')
            return false;
        bool retcode = (response.GetChar() != '0');
        return retcode;
    }
    return false;
}

bool
GDBRemoteCommunicationClient::CalculateMD5 (const lldb_private::FileSpec& file_spec,
                                            uint64_t &high,
                                            uint64_t &low)
{
    std::string path(file_spec.GetPath(false));
    lldb_private::StreamString stream;
    stream.PutCString("vFile:MD5:");
    stream.PutCStringAsRawHex8(path.c_str());
    const char* packet = stream.GetData();
    int packet_len = stream.GetSize();
    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse(packet, packet_len, response, false) == PacketResult::Success)
    {
        if (response.GetChar() != 'F')
            return false;
        if (response.GetChar() != ',')
            return false;
        if (response.Peek() && *response.Peek() == 'x')
            return false;
        low = response.GetHexMaxU64(false, UINT64_MAX);
        high = response.GetHexMaxU64(false, UINT64_MAX);
        return true;
    }
    return false;
}

bool
GDBRemoteCommunicationClient::AvoidGPackets (ProcessGDBRemote *process)
{
    // Some targets have issues with g/G packets and we need to avoid using them
    if (m_avoid_g_packets == eLazyBoolCalculate)
    {
        if (process)
        {
            m_avoid_g_packets = eLazyBoolNo;
            const ArchSpec &arch = process->GetTarget().GetArchitecture();
            if (arch.IsValid()
                && arch.GetTriple().getVendor() == llvm::Triple::Apple
                && arch.GetTriple().getOS() == llvm::Triple::IOS
                && arch.GetTriple().getArch() == llvm::Triple::aarch64)
            {
                m_avoid_g_packets = eLazyBoolYes;
                uint32_t gdb_server_version = GetGDBServerProgramVersion();
                if (gdb_server_version != 0)
                {
                    const char *gdb_server_name = GetGDBServerProgramName();
                    if (gdb_server_name && strcmp(gdb_server_name, "debugserver") == 0)
                    {
                        if (gdb_server_version >= 310)
                            m_avoid_g_packets = eLazyBoolNo;
                    }
                }
            }
        }
    }
    return m_avoid_g_packets == eLazyBoolYes;
}

DataBufferSP
GDBRemoteCommunicationClient::ReadRegister(lldb::tid_t tid, uint32_t reg, const Lock &lock)
{
    StreamString payload;
    payload.Printf("p%x", reg);
    StringExtractorGDBRemote response;
    if (SendThreadSpecificPacketAndWaitForResponse(tid, std::move(payload), response, lock) != PacketResult::Success ||
        !response.IsNormalResponse())
        return nullptr;

    DataBufferSP buffer_sp(new DataBufferHeap(response.GetStringRef().size() / 2, 0));
    response.GetHexBytes(buffer_sp->GetBytes(), buffer_sp->GetByteSize(), '\xcc');
    return buffer_sp;
}

DataBufferSP
GDBRemoteCommunicationClient::ReadAllRegisters(lldb::tid_t tid, const Lock &lock)
{
    StreamString payload;
    payload.PutChar('g');
    StringExtractorGDBRemote response;
    if (SendThreadSpecificPacketAndWaitForResponse(tid, std::move(payload), response, lock) != PacketResult::Success ||
        !response.IsNormalResponse())
        return nullptr;

    DataBufferSP buffer_sp(new DataBufferHeap(response.GetStringRef().size() / 2, 0));
    response.GetHexBytes(buffer_sp->GetBytes(), buffer_sp->GetByteSize(), '\xcc');
    return buffer_sp;
}

bool
GDBRemoteCommunicationClient::WriteRegister(lldb::tid_t tid, uint32_t reg_num, llvm::ArrayRef<uint8_t> data,
                                            const Lock &lock)
{
    StreamString payload;
    payload.Printf("P%x=", reg_num);
    payload.PutBytesAsRawHex8(data.data(), data.size(), endian::InlHostByteOrder(), endian::InlHostByteOrder());
    StringExtractorGDBRemote response;
    return SendThreadSpecificPacketAndWaitForResponse(tid, std::move(payload), response, lock) ==
               PacketResult::Success &&
           response.IsOKResponse();
}

bool
GDBRemoteCommunicationClient::WriteAllRegisters(lldb::tid_t tid, llvm::ArrayRef<uint8_t> data, const Lock &lock)
{
    StreamString payload;
    payload.PutChar('G');
    payload.PutBytesAsRawHex8(data.data(), data.size(), endian::InlHostByteOrder(), endian::InlHostByteOrder());
    StringExtractorGDBRemote response;
    return SendThreadSpecificPacketAndWaitForResponse(tid, std::move(payload), response, lock) ==
               PacketResult::Success &&
           response.IsOKResponse();
}

bool
GDBRemoteCommunicationClient::SaveRegisterState (lldb::tid_t tid, uint32_t &save_id)
{
    save_id = 0; // Set to invalid save ID
    if (m_supports_QSaveRegisterState == eLazyBoolNo)
        return false;

    Lock lock(*this, false);
    if (!lock)
    {
        Log *log(ProcessGDBRemoteLog::GetLogIfAnyCategoryIsSet(GDBR_LOG_PROCESS | GDBR_LOG_PACKETS));
        if (log)
            log->Printf("GDBRemoteCommunicationClient::%s failed to get sequence mutex", __FUNCTION__);
        return false;
    }

    m_supports_QSaveRegisterState = eLazyBoolYes;
    StreamString payload;
    payload.PutCString("QSaveRegisterState");
    StringExtractorGDBRemote response;
    if (SendThreadSpecificPacketAndWaitForResponse(tid, std::move(payload), response, lock) != PacketResult::Success)
        return false;

    if (response.IsUnsupportedResponse())
        m_supports_QSaveRegisterState = eLazyBoolNo;

    const uint32_t response_save_id = response.GetU32(0);
    if (response_save_id == 0)
        return false;

    save_id = response_save_id;
    return true;
}

bool
GDBRemoteCommunicationClient::RestoreRegisterState (lldb::tid_t tid, uint32_t save_id)
{
    // We use the "m_supports_QSaveRegisterState" variable here because the
    // QSaveRegisterState and QRestoreRegisterState packets must both be supported in
    // order to be useful
    if (m_supports_QSaveRegisterState == eLazyBoolNo)
        return false;

    Lock lock(*this, false);
    if (!lock)
    {
        Log *log(ProcessGDBRemoteLog::GetLogIfAnyCategoryIsSet(GDBR_LOG_PROCESS | GDBR_LOG_PACKETS));
        if (log)
            log->Printf("GDBRemoteCommunicationClient::%s failed to get sequence mutex", __FUNCTION__);
        return false;
    }

    StreamString payload;
    payload.Printf("QRestoreRegisterState:%u", save_id);
    StringExtractorGDBRemote response;
    if (SendThreadSpecificPacketAndWaitForResponse(tid, std::move(payload), response, lock) != PacketResult::Success)
        return false;

    if (response.IsOKResponse())
        return true;

    if (response.IsUnsupportedResponse())
        m_supports_QSaveRegisterState = eLazyBoolNo;
    return false;
}

bool
GDBRemoteCommunicationClient::SyncThreadState(lldb::tid_t tid)
{
    if (!GetSyncThreadStateSupported())
        return false;

    StreamString packet;
    StringExtractorGDBRemote response;
    packet.Printf("QSyncThreadState:%4.4" PRIx64 ";", tid);
    return SendPacketAndWaitForResponse(packet.GetString(), response, false) ==
               GDBRemoteCommunication::PacketResult::Success &&
           response.IsOKResponse();
}

bool
GDBRemoteCommunicationClient::GetModuleInfo(const FileSpec &module_file_spec, const lldb_private::ArchSpec &arch_spec,
                                            ModuleSpec &module_spec)
{
    if (!m_supports_qModuleInfo)
        return false;

    std::string module_path = module_file_spec.GetPath (false);
    if (module_path.empty ())
        return false;

    StreamString packet;
    packet.PutCString("qModuleInfo:");
    packet.PutCStringAsRawHex8(module_path.c_str());
    packet.PutCString(";");
    const auto& triple = arch_spec.GetTriple().getTriple();
    packet.PutCStringAsRawHex8(triple.c_str());

    StringExtractorGDBRemote response;
    if (SendPacketAndWaitForResponse (packet.GetData(), packet.GetSize(), response, false) != PacketResult::Success)
        return false;

    if (response.IsErrorResponse ())
        return false;

    if (response.IsUnsupportedResponse ())
    {
        m_supports_qModuleInfo = false;
        return false;
    }

    std::string name;
    std::string value;
    bool success;
    StringExtractor extractor;

    module_spec.Clear ();
    module_spec.GetFileSpec () = module_file_spec;

    while (response.GetNameColonValue (name, value))
    {
        if (name == "uuid" || name == "md5")
        {
            extractor.GetStringRef ().swap (value);
            extractor.SetFilePos (0);
            extractor.GetHexByteString (value);
            module_spec.GetUUID().SetFromCString (value.c_str(), value.size() / 2);
        }
        else if (name == "triple")
        {
            extractor.GetStringRef ().swap (value);
            extractor.SetFilePos (0);
            extractor.GetHexByteString (value);
            module_spec.GetArchitecture().SetTriple (value.c_str ());
        }
        else if (name == "file_offset")
        {
            const auto ival = StringConvert::ToUInt64 (value.c_str (), 0, 16, &success);
            if (success)
                module_spec.SetObjectOffset (ival);
        }
        else if (name == "file_size")
        {
            const auto ival = StringConvert::ToUInt64 (value.c_str (), 0, 16, &success);
            if (success)
                module_spec.SetObjectSize (ival);
        }
        else if (name == "file_path")
        {
            extractor.GetStringRef ().swap (value);
            extractor.SetFilePos (0);
            extractor.GetHexByteString (value);
            module_spec.GetFileSpec() = FileSpec(value.c_str(), false, arch_spec);
        }
    }

    return true;
}

// query the target remote for extended information using the qXfer packet
//
// example: object='features', annex='target.xml', out=<xml output>
// return:  'true'  on success
//          'false' on failure (err set)
bool
GDBRemoteCommunicationClient::ReadExtFeature (const lldb_private::ConstString object,
                                              const lldb_private::ConstString annex,
                                              std::string & out,
                                              lldb_private::Error & err) {

    std::stringstream output;
    StringExtractorGDBRemote chunk;

    uint64_t size = GetRemoteMaxPacketSize();
    if (size == 0)
        size = 0x1000;
    size = size - 1; // Leave space for the 'm' or 'l' character in the response
    int offset = 0;
    bool active = true;

    // loop until all data has been read
    while ( active ) {

        // send query extended feature packet
        std::stringstream packet;
        packet << "qXfer:" 
               << object.AsCString("") << ":read:" 
               << annex.AsCString("")  << ":" 
               << std::hex << offset  << "," 
               << std::hex << size;

        GDBRemoteCommunication::PacketResult res =
            SendPacketAndWaitForResponse( packet.str().c_str(),
                                          chunk,
                                          false );

        if ( res != GDBRemoteCommunication::PacketResult::Success ) {
            err.SetErrorString( "Error sending $qXfer packet" );
            return false;
        }

        const std::string & str = chunk.GetStringRef( );
        if ( str.length() == 0 ) {
            // should have some data in chunk
            err.SetErrorString( "Empty response from $qXfer packet" );
            return false;
        }

        // check packet code
        switch ( str[0] ) {
            // last chunk
        case ( 'l' ):
            active = false;
            LLVM_FALLTHROUGH;

            // more chunks
        case ( 'm' ) :
            if ( str.length() > 1 )
                output << &str[1];
            offset += size;
            break;

            // unknown chunk
        default:
            err.SetErrorString( "Invalid continuation code from $qXfer packet" );
            return false;
        }
    }

    out = output.str( );
    err.Success( );
    return true;
}

// Notify the target that gdb is prepared to serve symbol lookup requests.
//  packet: "qSymbol::"
//  reply:
//  OK                  The target does not need to look up any (more) symbols.
//  qSymbol:<sym_name>  The target requests the value of symbol sym_name (hex encoded).
//                      LLDB may provide the value by sending another qSymbol packet
//                      in the form of"qSymbol:<sym_value>:<sym_name>".
//
//  Three examples:
//
//  lldb sends:    qSymbol::
//  lldb receives: OK
//     Remote gdb stub does not need to know the addresses of any symbols, lldb does not
//     need to ask again in this session.
//
//  lldb sends:    qSymbol::
//  lldb receives: qSymbol:64697370617463685f71756575655f6f666673657473
//  lldb sends:    qSymbol::64697370617463685f71756575655f6f666673657473
//  lldb receives: OK
//     Remote gdb stub asks for address of 'dispatch_queue_offsets'.  lldb does not know
//     the address at this time.  lldb needs to send qSymbol:: again when it has more
//     solibs loaded.
//
//  lldb sends:    qSymbol::
//  lldb receives: qSymbol:64697370617463685f71756575655f6f666673657473
//  lldb sends:    qSymbol:2bc97554:64697370617463685f71756575655f6f666673657473
//  lldb receives: OK
//     Remote gdb stub asks for address of 'dispatch_queue_offsets'.  lldb says that it
//     is at address 0x2bc97554.  Remote gdb stub sends 'OK' indicating that it does not
//     need any more symbols.  lldb does not need to ask again in this session.

void
GDBRemoteCommunicationClient::ServeSymbolLookups(lldb_private::Process *process)
{
    // Set to true once we've resolved a symbol to an address for the remote stub.
    // If we get an 'OK' response after this, the remote stub doesn't need any more
    // symbols and we can stop asking.
    bool symbol_response_provided = false;

    // Is this the initial qSymbol:: packet?
    bool first_qsymbol_query = true;

    if (m_supports_qSymbol && m_qSymbol_requests_done == false)
    {
        Lock lock(*this, false);
        if (lock)
        {
            StreamString packet;
            packet.PutCString ("qSymbol::");
            StringExtractorGDBRemote response;
            while (SendPacketAndWaitForResponse(packet.GetString(), response, lock) == PacketResult::Success)
            {
                if (response.IsOKResponse())
                {
                    if (symbol_response_provided || first_qsymbol_query)
                    {
                        m_qSymbol_requests_done = true;
                    }

                    // We are done serving symbols requests
                    return;
                }
                first_qsymbol_query = false;

                if (response.IsUnsupportedResponse())
                {
                    // qSymbol is not supported by the current GDB server we are connected to
                    m_supports_qSymbol = false;
                    return;
                }
                else
                {
                    llvm::StringRef response_str(response.GetStringRef());
                    if (response_str.startswith("qSymbol:"))
                    {
                        response.SetFilePos(strlen("qSymbol:"));
                        std::string symbol_name;
                        if (response.GetHexByteString(symbol_name))
                        {
                            if (symbol_name.empty())
                                return;

                            addr_t symbol_load_addr = LLDB_INVALID_ADDRESS;
                            lldb_private::SymbolContextList sc_list;
                            if (process->GetTarget().GetImages().FindSymbolsWithNameAndType(ConstString(symbol_name), eSymbolTypeAny, sc_list))
                            {
                                const size_t num_scs = sc_list.GetSize();
                                for (size_t sc_idx=0; sc_idx<num_scs && symbol_load_addr == LLDB_INVALID_ADDRESS; ++sc_idx)
                                {
                                    SymbolContext sc;
                                    if (sc_list.GetContextAtIndex(sc_idx, sc))
                                    {
                                        if (sc.symbol)
                                        {
                                            switch (sc.symbol->GetType())
                                            {
                                            case eSymbolTypeInvalid:
                                            case eSymbolTypeAbsolute:
                                            case eSymbolTypeUndefined:
                                            case eSymbolTypeSourceFile:
                                            case eSymbolTypeHeaderFile:
                                            case eSymbolTypeObjectFile:
                                            case eSymbolTypeCommonBlock:
                                            case eSymbolTypeBlock:
                                            case eSymbolTypeLocal:
                                            case eSymbolTypeParam:
                                            case eSymbolTypeVariable:
                                            case eSymbolTypeVariableType:
                                            case eSymbolTypeLineEntry:
                                            case eSymbolTypeLineHeader:
                                            case eSymbolTypeScopeBegin:
                                            case eSymbolTypeScopeEnd:
                                            case eSymbolTypeAdditional:
                                            case eSymbolTypeCompiler:
                                            case eSymbolTypeInstrumentation:
                                            case eSymbolTypeTrampoline:
                                            case eSymbolTypeASTFile:
                                            case eSymbolTypeMetadata:
                                                break;

                                            case eSymbolTypeIVarOffset:
                                            case eSymbolTypeCode:
                                            case eSymbolTypeResolver:
                                            case eSymbolTypeData:
                                            case eSymbolTypeRuntime:
                                            case eSymbolTypeException:
                                            case eSymbolTypeObjCClass:
                                            case eSymbolTypeObjCMetaClass:
                                            case eSymbolTypeObjCIVar:
                                            case eSymbolTypeReExported:
                                                symbol_load_addr = sc.symbol->GetLoadAddress(&process->GetTarget());
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                            // This is the normal path where our symbol lookup was successful and we want
                            // to send a packet with the new symbol value and see if another lookup needs to be
                            // done.

                            // Change "packet" to contain the requested symbol value and name
                            packet.Clear();
                            packet.PutCString("qSymbol:");
                            if (symbol_load_addr != LLDB_INVALID_ADDRESS)
                            {
                                packet.Printf("%" PRIx64, symbol_load_addr);
                                symbol_response_provided = true;
                            }
                            else
                            {
                                symbol_response_provided = false;
                            }
                            packet.PutCString(":");
                            packet.PutBytesAsRawHex8(symbol_name.data(), symbol_name.size());
                            continue; // go back to the while loop and send "packet" and wait for another response
                        }
                    }
                }
            }
            // If we make it here, the symbol request packet response wasn't valid or
            // our symbol lookup failed so we must abort
            return;

        }
        else if (Log *log = ProcessGDBRemoteLog::GetLogIfAnyCategoryIsSet(GDBR_LOG_PROCESS | GDBR_LOG_PACKETS))
        {
            log->Printf("GDBRemoteCommunicationClient::%s: Didn't get sequence mutex.", __FUNCTION__);
        }
    }
}

StructuredData::Array*
GDBRemoteCommunicationClient::GetSupportedStructuredDataPlugins()
{
    if (!m_supported_async_json_packets_is_valid)
    {
        // Query the server for the array of supported asynchronous JSON
        // packets.
        m_supported_async_json_packets_is_valid = true;

        Log *log(ProcessGDBRemoteLog::GetLogIfAllCategoriesSet(
            GDBR_LOG_PROCESS));

        // Poll it now.
        StringExtractorGDBRemote response;
        const bool send_async = false;
        if (SendPacketAndWaitForResponse("qStructuredDataPlugins", response,
                                         send_async) == PacketResult::Success)
        {
            m_supported_async_json_packets_sp = StructuredData::ParseJSON(
                response.GetStringRef());
            if (m_supported_async_json_packets_sp &&
                !m_supported_async_json_packets_sp->GetAsArray())
            {
                // We were returned something other than a JSON array.  This
                // is invalid.  Clear it out.
                if (log)
                    log->Printf("GDBRemoteCommunicationClient::%s(): "
                                "QSupportedAsyncJSONPackets returned invalid "
                                "result: %s", __FUNCTION__,
                                response.GetStringRef().c_str());
                m_supported_async_json_packets_sp.reset();
            }
        }
        else
        {
            if (log)
                log->Printf("GDBRemoteCommunicationClient::%s(): "
                            "QSupportedAsyncJSONPackets unsupported",
                            __FUNCTION__);
        }

        if (log && m_supported_async_json_packets_sp)
        {
            StreamString  stream;
            m_supported_async_json_packets_sp->Dump(stream);
            log->Printf("GDBRemoteCommunicationClient::%s(): supported async "
                        "JSON packets: %s", __FUNCTION__,
                        stream.GetString().c_str());
        }
    }

    return m_supported_async_json_packets_sp
        ? m_supported_async_json_packets_sp->GetAsArray()
        : nullptr;
}

Error
GDBRemoteCommunicationClient::ConfigureRemoteStructuredData(
    const ConstString &type_name,
    const StructuredData::ObjectSP &config_sp)
{
    Error error;

    if (type_name.GetLength() == 0)
    {
        error.SetErrorString("invalid type_name argument");
        return error;
    }

    // Build command: Configure{type_name}: serialized config
    // data.
    StreamGDBRemote stream;
    stream.PutCString("QConfigure");
    stream.PutCString(type_name.AsCString());
    stream.PutChar(':');
    if (config_sp)
    {
        // Gather the plain-text version of the configuration data.
        StreamString unescaped_stream;
        config_sp->Dump(unescaped_stream);
        unescaped_stream.Flush();

        // Add it to the stream in escaped fashion.
        stream.PutEscapedBytes(unescaped_stream.GetData(),
                               unescaped_stream.GetSize());
    }
    stream.Flush();

    // Send the packet.
    const bool send_async = false;
    StringExtractorGDBRemote response;
    auto result = SendPacketAndWaitForResponse(stream.GetString().c_str(),
                                               response, send_async);
    if (result == PacketResult::Success)
    {
        // We failed if the config result comes back other than OK.
        if (strcmp(response.GetStringRef().c_str(), "OK") == 0)
        {
            // Okay!
            error.Clear();
        }
        else
        {
            error.SetErrorStringWithFormat("configuring StructuredData feature "
                                           "%s failed with error %s",
                                           type_name.AsCString(),
                                           response.GetStringRef().c_str());
        }
    }
    else
    {
        // Can we get more data here on the failure?
        error.SetErrorStringWithFormat("configuring StructuredData feature %s "
                                       "failed when sending packet: "
                                       "PacketResult=%d", type_name.AsCString(),
                                       result);
    }
    return error;
}

void
GDBRemoteCommunicationClient::OnRunPacketSent(bool first)
{
    GDBRemoteClientBase::OnRunPacketSent(first);
    m_curr_tid = LLDB_INVALID_THREAD_ID;
}
