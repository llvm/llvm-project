//===-- GDBRemoteCommunicationServer.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_GDBRemoteCommunicationServer_h_
#define liblldb_GDBRemoteCommunicationServer_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Process.h"

#include "GDBRemoteCommunication.h"

class ProcessGDBRemote;
class StringExtractorGDBRemote;

class GDBRemoteCommunicationServer : public GDBRemoteCommunication
{
public:
    enum
    {
        eBroadcastBitRunPacketSent = kLoUserBroadcastBit
    };
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    GDBRemoteCommunicationServer(bool is_platform);

    virtual
    ~GDBRemoteCommunicationServer();

    bool
    GetPacketAndSendResponse (uint32_t timeout_usec,
                              lldb_private::Error &error,
                              bool &interrupt, 
                              bool &quit);

    virtual bool
    GetThreadSuffixSupported ()
    {
        return true;
    }

    // After connecting, do a little handshake with the client to make sure
    // we are at least communicating
    bool
    HandshakeWithClient (lldb_private::Error *error_ptr);

    // Set both ports to zero to let the platform automatically bind to 
    // a port chosen by the OS.
    void
    SetPortRange (uint16_t lo_port_num, uint16_t hi_port_num)
    {
        m_lo_port_num = lo_port_num;
        m_hi_port_num = hi_port_num;
    }

protected:
    //typedef std::map<uint16_t, lldb::pid_t> PortToPIDMap;

    lldb::thread_t m_async_thread;
    lldb_private::ProcessLaunchInfo m_process_launch_info;
    lldb_private::Error m_process_launch_error;
    lldb_private::ProcessInstanceInfoList m_proc_infos;
    uint32_t m_proc_infos_index;
    uint16_t m_lo_port_num;
    uint16_t m_hi_port_num;
    //PortToPIDMap m_port_to_pid_map;

    size_t
    SendUnimplementedResponse (const char *packet);

    size_t
    SendErrorResponse (uint8_t error);

    size_t
    SendOKResponse ();

    bool
    Handle_A (StringExtractorGDBRemote &packet);

    bool
    Handle_qLaunchSuccess (StringExtractorGDBRemote &packet);

    bool
    Handle_qHostInfo (StringExtractorGDBRemote &packet);
    
    bool
    Handle_qLaunchGDBServer (StringExtractorGDBRemote &packet);

    bool
    Handle_qProcessInfoPID (StringExtractorGDBRemote &packet);
    
    bool
    Handle_qfProcessInfo (StringExtractorGDBRemote &packet);
    
    bool 
    Handle_qsProcessInfo (StringExtractorGDBRemote &packet);

    bool
    Handle_qC (StringExtractorGDBRemote &packet);

    bool 
    Handle_qUserName (StringExtractorGDBRemote &packet);

    bool 
    Handle_qGroupName (StringExtractorGDBRemote &packet);

    bool
    Handle_qSpeedTest (StringExtractorGDBRemote &packet);

    bool
    Handle_QEnvironment  (StringExtractorGDBRemote &packet);
    
    bool
    Handle_QSetDisableASLR (StringExtractorGDBRemote &packet);

    bool
    Handle_QSetWorkingDir (StringExtractorGDBRemote &packet);

    bool
    Handle_QStartNoAckMode (StringExtractorGDBRemote &packet);

    bool
    Handle_QSetSTDIN (StringExtractorGDBRemote &packet);

    bool
    Handle_QSetSTDOUT (StringExtractorGDBRemote &packet);

    bool
    Handle_QSetSTDERR (StringExtractorGDBRemote &packet);
    
private:
    //------------------------------------------------------------------
    // For GDBRemoteCommunicationServer only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (GDBRemoteCommunicationServer);
};

#endif  // liblldb_GDBRemoteCommunicationServer_h_
