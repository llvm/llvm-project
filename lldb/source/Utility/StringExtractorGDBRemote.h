//===-- StringExtractorGDBRemote.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef utility_StringExtractorGDBRemote_h_
#define utility_StringExtractorGDBRemote_h_

// C Includes
// C++ Includes
#include <string>
// Other libraries and framework includes
// Project includes
#include "Utility/StringExtractor.h"

class StringExtractorGDBRemote : public StringExtractor
{
public:

    StringExtractorGDBRemote() :
        StringExtractor ()
    {
    }

    StringExtractorGDBRemote(const char *cstr) :
        StringExtractor (cstr)
    {
    }
    StringExtractorGDBRemote(const StringExtractorGDBRemote& rhs) :
        StringExtractor (rhs)
    {
    }

    virtual ~StringExtractorGDBRemote()
    {
    }

    enum ServerPacketType
    {
        eServerPacketType_nack = 0,
        eServerPacketType_ack,
        eServerPacketType_invalid,
        eServerPacketType_unimplemented,
        eServerPacketType_interrupt, // CTRL+c packet or "\x03"
        eServerPacketType_A, // Program arguments packet
        eServerPacketType_qfProcessInfo,
        eServerPacketType_qsProcessInfo,
        eServerPacketType_qC,
        eServerPacketType_qGroupName,
        eServerPacketType_qHostInfo,
        eServerPacketType_qLaunchGDBServer,
        eServerPacketType_qLaunchSuccess,
        eServerPacketType_qProcessInfoPID,
        eServerPacketType_qSpeedTest,
        eServerPacketType_qUserName,
        eServerPacketType_QEnvironment,
        eServerPacketType_QSetDisableASLR,
        eServerPacketType_QSetSTDIN,
        eServerPacketType_QSetSTDOUT,
        eServerPacketType_QSetSTDERR,
        eServerPacketType_QSetWorkingDir,
        eServerPacketType_QStartNoAckMode
    };
    
    ServerPacketType
    GetServerPacketType () const;

    enum ResponseType
    {
        eUnsupported = 0,
        eAck,
        eNack,
        eError,
        eOK,
        eResponse
    };

    ResponseType
    GetResponseType () const;

    bool
    IsOKResponse() const;

    bool
    IsUnsupportedResponse() const;

    bool
    IsNormalResponse () const;

    bool
    IsErrorResponse() const;

    // Returns zero if the packet isn't a EXX packet where XX are two hex
    // digits. Otherwise the error encoded in XX is returned.
    uint8_t
    GetError();
};

#endif  // utility_StringExtractorGDBRemote_h_
