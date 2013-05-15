//===-- StringExtractorGDBRemote.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <string.h>

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "Utility/StringExtractorGDBRemote.h"



StringExtractorGDBRemote::ResponseType
StringExtractorGDBRemote::GetResponseType () const
{
    if (m_packet.empty())
        return eUnsupported;

    switch (m_packet[0])
    {
    case 'E':
        if (m_packet.size() == 3 &&
            isxdigit(m_packet[1]) &&
            isxdigit(m_packet[2]))
            return eError;
        break;

    case 'O':
        if (m_packet.size() == 2 && m_packet[1] == 'K')
            return eOK;
        break;

    case '+':
        if (m_packet.size() == 1)
            return eAck;
        break;

    case '-':
        if (m_packet.size() == 1)
            return eNack;
        break;
    }
    return eResponse;
}

StringExtractorGDBRemote::ServerPacketType
StringExtractorGDBRemote::GetServerPacketType () const
{
#define PACKET_MATCHES(s) ((packet_size == (sizeof(s)-1)) && (strcmp((packet_cstr),(s)) == 0))
#define PACKET_STARTS_WITH(s) ((packet_size >= (sizeof(s)-1)) && ::strncmp(packet_cstr, s, (sizeof(s)-1))==0)
    
    // Empty is not a supported packet...
    if (m_packet.empty())
        return eServerPacketType_invalid;

    const size_t packet_size = m_packet.size();
    const char *packet_cstr = m_packet.c_str();
    switch (m_packet[0])
    {
    case '\x03':
        if (packet_size == 1) return eServerPacketType_interrupt;
        break;

    case '-':
        if (packet_size == 1) return eServerPacketType_nack;
        break;

    case '+':
        if (packet_size == 1) return eServerPacketType_ack;
        break;

    case 'A':
        return eServerPacketType_A;
            
    case 'Q':
        switch (packet_cstr[1])
        {
        case 'E':
            if (PACKET_STARTS_WITH ("QEnvironment:"))           return eServerPacketType_QEnvironment; 
            break;

        case 'S':
            if (PACKET_MATCHES ("QStartNoAckMode"))             return eServerPacketType_QStartNoAckMode;
            else if (PACKET_STARTS_WITH ("QSetDisableASLR:"))   return eServerPacketType_QSetDisableASLR;
            else if (PACKET_STARTS_WITH ("QSetSTDIN:"))         return eServerPacketType_QSetSTDIN;
            else if (PACKET_STARTS_WITH ("QSetSTDOUT:"))        return eServerPacketType_QSetSTDOUT;
            else if (PACKET_STARTS_WITH ("QSetSTDERR:"))        return eServerPacketType_QSetSTDERR;
            else if (PACKET_STARTS_WITH ("QSetWorkingDir:"))    return eServerPacketType_QSetWorkingDir;
            break;
        }
        break;
            
    case 'q':
        switch (packet_cstr[1])
        {
        case 's':
            if (PACKET_MATCHES ("qsProcessInfo"))               return eServerPacketType_qsProcessInfo;
            break;

        case 'f':
            if (PACKET_STARTS_WITH ("qfProcessInfo"))           return eServerPacketType_qfProcessInfo;
            break;

        case 'C':
            if (packet_size == 2)                               return eServerPacketType_qC;
            break;

        case 'G':
            if (PACKET_STARTS_WITH ("qGroupName:"))             return eServerPacketType_qGroupName;
            break;

        case 'H':
            if (PACKET_MATCHES ("qHostInfo"))                   return eServerPacketType_qHostInfo;
            break;

        case 'L':
            if (PACKET_MATCHES ("qLaunchGDBServer"))            return eServerPacketType_qLaunchGDBServer;
            if (PACKET_MATCHES ("qLaunchSuccess"))              return eServerPacketType_qLaunchSuccess;
            break;
            
        case 'P':
            if (PACKET_STARTS_WITH ("qProcessInfoPID:"))        return eServerPacketType_qProcessInfoPID;
            break;

        case 'S':
            if (PACKET_STARTS_WITH ("qSpeedTest:"))             return eServerPacketType_qSpeedTest;
            break;

        case 'U':
            if (PACKET_STARTS_WITH ("qUserName:"))              return eServerPacketType_qUserName;
            break;
        }
        break;
    }
    return eServerPacketType_unimplemented;
}

bool
StringExtractorGDBRemote::IsOKResponse() const
{
    return GetResponseType () == eOK;
}


bool
StringExtractorGDBRemote::IsUnsupportedResponse() const
{
    return GetResponseType () == eUnsupported;
}

bool
StringExtractorGDBRemote::IsNormalResponse() const
{
    return GetResponseType () == eResponse;
}

bool
StringExtractorGDBRemote::IsErrorResponse() const
{
    return GetResponseType () == eError &&
           m_packet.size() == 3 &&
           isxdigit(m_packet[1]) &&
           isxdigit(m_packet[2]);
}

uint8_t
StringExtractorGDBRemote::GetError ()
{
    if (GetResponseType() == eError)
    {
        SetFilePos(1);
        return GetHexU8(255);
    }
    return 0;
}
