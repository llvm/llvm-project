//===-- StringExtractor.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Utility/StringExtractor.h"

// C Includes
#include <stdlib.h>

// C++ Includes
// Other libraries and framework includes
// Project includes

static const uint8_t
g_hex_ascii_to_hex_integer[256] = {
    
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7,
    0x8, 0x9, 255, 255, 255, 255, 255, 255,
    255, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
};

static inline int
xdigit_to_sint (char ch)
{
    if (ch >= 'a' && ch <= 'f')
        return 10 + ch - 'a';
    if (ch >= 'A' && ch <= 'F')
        return 10 + ch - 'A';
    return ch - '0';
}

static inline unsigned int
xdigit_to_uint (uint8_t ch)
{
    if (ch >= 'a' && ch <= 'f')
        return 10u + ch - 'a';
    if (ch >= 'A' && ch <= 'F')
        return 10u + ch - 'A';
    return ch - '0';
}

//----------------------------------------------------------------------
// StringExtractor constructor
//----------------------------------------------------------------------
StringExtractor::StringExtractor() :
    m_packet(),
    m_index (0)
{
}


StringExtractor::StringExtractor(const char *packet_cstr) :
    m_packet(),
    m_index (0)
{
    if (packet_cstr)
        m_packet.assign (packet_cstr);
}


//----------------------------------------------------------------------
// StringExtractor copy constructor
//----------------------------------------------------------------------
StringExtractor::StringExtractor(const StringExtractor& rhs) :
    m_packet (rhs.m_packet),
    m_index (rhs.m_index)
{

}

//----------------------------------------------------------------------
// StringExtractor assignment operator
//----------------------------------------------------------------------
const StringExtractor&
StringExtractor::operator=(const StringExtractor& rhs)
{
    if (this != &rhs)
    {
        m_packet = rhs.m_packet;
        m_index = rhs.m_index;

    }
    return *this;
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
StringExtractor::~StringExtractor()
{
}


char
StringExtractor::GetChar (char fail_value)
{
    if (m_index < m_packet.size())
    {
        char ch = m_packet[m_index];
        ++m_index;
        return ch;
    }
    m_index = UINT64_MAX;
    return fail_value;
}

//----------------------------------------------------------------------
// Extract an unsigned character from two hex ASCII chars in the packet
// string
//----------------------------------------------------------------------
uint8_t
StringExtractor::GetHexU8 (uint8_t fail_value, bool set_eof_on_fail)
{
    uint32_t i = m_index;
    if ((i + 2) <= m_packet.size())
    {
        const uint8_t hi_nibble = g_hex_ascii_to_hex_integer[static_cast<uint8_t>(m_packet[i])];
        const uint8_t lo_nibble = g_hex_ascii_to_hex_integer[static_cast<uint8_t>(m_packet[i+1])];
        if (hi_nibble < 16 && lo_nibble < 16)
        {
            m_index += 2;
            return (hi_nibble << 4) + lo_nibble;
        }
    }
    if (set_eof_on_fail || m_index >= m_packet.size())
        m_index = UINT64_MAX;
    return fail_value;
}

uint32_t
StringExtractor::GetU32 (uint32_t fail_value, int base)
{
    if (m_index < m_packet.size())
    {
        char *end = NULL;
        const char *start = m_packet.c_str();
        const char *uint_cstr = start + m_index;
        uint32_t result = ::strtoul (uint_cstr, &end, base);

        if (end && end != uint_cstr)
        {
            m_index = end - start;
            return result;
        }
    }
    return fail_value;
}


uint32_t
StringExtractor::GetHexMaxU32 (bool little_endian, uint32_t fail_value)
{
    uint32_t result = 0;
    uint32_t nibble_count = 0;

    if (little_endian)
    {
        uint32_t shift_amount = 0;
        while (m_index < m_packet.size() && ::isxdigit (m_packet[m_index]))
        {
            // Make sure we don't exceed the size of a uint32_t...
            if (nibble_count >= (sizeof(uint32_t) * 2))
            {
                m_index = UINT64_MAX;
                return fail_value;
            }

            uint8_t nibble_lo;
            uint8_t nibble_hi = xdigit_to_sint (m_packet[m_index]);
            ++m_index;
            if (m_index < m_packet.size() && ::isxdigit (m_packet[m_index]))
            {
                nibble_lo = xdigit_to_sint (m_packet[m_index]);
                ++m_index;
                result |= ((uint32_t)nibble_hi << (shift_amount + 4));
                result |= ((uint32_t)nibble_lo << shift_amount);
                nibble_count += 2;
                shift_amount += 8;
            }
            else
            {
                result |= ((uint32_t)nibble_hi << shift_amount);
                nibble_count += 1;
                shift_amount += 4;
            }

        }
    }
    else
    {
        while (m_index < m_packet.size() && ::isxdigit (m_packet[m_index]))
        {
            // Make sure we don't exceed the size of a uint32_t...
            if (nibble_count >= (sizeof(uint32_t) * 2))
            {
                m_index = UINT64_MAX;
                return fail_value;
            }

            uint8_t nibble = xdigit_to_sint (m_packet[m_index]);
            // Big Endian
            result <<= 4;
            result |= nibble;

            ++m_index;
            ++nibble_count;
        }
    }
    return result;
}

uint64_t
StringExtractor::GetHexMaxU64 (bool little_endian, uint64_t fail_value)
{
    uint64_t result = 0;
    uint32_t nibble_count = 0;

    if (little_endian)
    {
        uint32_t shift_amount = 0;
        while (m_index < m_packet.size() && ::isxdigit (m_packet[m_index]))
        {
            // Make sure we don't exceed the size of a uint64_t...
            if (nibble_count >= (sizeof(uint64_t) * 2))
            {
                m_index = UINT64_MAX;
                return fail_value;
            }

            uint8_t nibble_lo;
            uint8_t nibble_hi = xdigit_to_sint (m_packet[m_index]);
            ++m_index;
            if (m_index < m_packet.size() && ::isxdigit (m_packet[m_index]))
            {
                nibble_lo = xdigit_to_sint (m_packet[m_index]);
                ++m_index;
                result |= ((uint64_t)nibble_hi << (shift_amount + 4));
                result |= ((uint64_t)nibble_lo << shift_amount);
                nibble_count += 2;
                shift_amount += 8;
            }
            else
            {
                result |= ((uint64_t)nibble_hi << shift_amount);
                nibble_count += 1;
                shift_amount += 4;
            }

        }
    }
    else
    {
        while (m_index < m_packet.size() && ::isxdigit (m_packet[m_index]))
        {
            // Make sure we don't exceed the size of a uint64_t...
            if (nibble_count >= (sizeof(uint64_t) * 2))
            {
                m_index = UINT64_MAX;
                return fail_value;
            }

            uint8_t nibble = xdigit_to_sint (m_packet[m_index]);
            // Big Endian
            result <<= 4;
            result |= nibble;

            ++m_index;
            ++nibble_count;
        }
    }
    return result;
}

size_t
StringExtractor::GetHexBytes (void *dst_void, size_t dst_len, uint8_t fail_fill_value)
{
    uint8_t *dst = (uint8_t*)dst_void;
    size_t bytes_extracted = 0;
    while (bytes_extracted < dst_len && GetBytesLeft ())
    {
        dst[bytes_extracted] = GetHexU8 (fail_fill_value);
        if (IsGood())
            ++bytes_extracted;
        else
            break;
    }

    for (size_t i = bytes_extracted; i < dst_len; ++i)
        dst[i] = fail_fill_value;

    return bytes_extracted;
}


// Consume ASCII hex nibble character pairs until we have decoded byte_size
// bytes of data.

uint64_t
StringExtractor::GetHexWithFixedSize (uint32_t byte_size, bool little_endian, uint64_t fail_value)
{
    if (byte_size <= 8 && GetBytesLeft() >= byte_size * 2)
    {
        uint64_t result = 0;
        uint32_t i;
        if (little_endian)
        {
            // Little Endian
            uint32_t shift_amount;
            for (i = 0, shift_amount = 0;
                 i < byte_size && IsGood();
                 ++i, shift_amount += 8)
            {
                result |= ((uint64_t)GetHexU8() << shift_amount);
            }
        }
        else
        {
            // Big Endian
            for (i = 0; i < byte_size && IsGood(); ++i)
            {
                result <<= 8;
                result |= GetHexU8();
            }
        }
    }
    m_index = UINT64_MAX;
    return fail_value;
}

size_t
StringExtractor::GetHexByteString (std::string &str)
{
    str.clear();
    char ch;
    while ((ch = GetHexU8()) != '\0')
        str.append(1, ch);
    return str.size();
}

bool
StringExtractor::GetNameColonValue (std::string &name, std::string &value)
{
    // Read something in the form of NNNN:VVVV; where NNNN is any character
    // that is not a colon, followed by a ':' character, then a value (one or
    // more ';' chars), followed by a ';'
    if (m_index < m_packet.size())
    {
        const size_t colon_idx = m_packet.find (':', m_index);
        if (colon_idx != std::string::npos)
        {
            const size_t semicolon_idx = m_packet.find (';', colon_idx);
            if (semicolon_idx != std::string::npos)
            {
                name.assign (m_packet, m_index, colon_idx - m_index);
                value.assign (m_packet, colon_idx + 1, semicolon_idx - (colon_idx + 1));
                m_index = semicolon_idx + 1;
                return true;
            }
        }
    }
    m_index = UINT64_MAX;
    return false;
}
