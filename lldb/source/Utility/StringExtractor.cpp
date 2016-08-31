//===-- StringExtractor.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/StringExtractor.h"

// C Includes
#include <stdlib.h>

// C++ Includes
#include <tuple>
// Other libraries and framework includes
// Project includes
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Endian.h"

static inline int
xdigit_to_sint (char ch)
{
    if (ch >= 'a' && ch <= 'f')
        return 10 + ch - 'a';
    if (ch >= 'A' && ch <= 'F')
        return 10 + ch - 'A';
    if (ch >= '0' && ch <= '9')
        return ch - '0';
    return -1;
}

//----------------------------------------------------------------------
// StringExtractor constructor
//----------------------------------------------------------------------
StringExtractor::StringExtractor() :
    m_packet(),
    m_index (0)
{
}

StringExtractor::StringExtractor(llvm::StringRef packet_str) : m_packet(), m_index(0)
{
    m_packet.assign(packet_str.begin(), packet_str.end());
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

static llvm::Optional<uint8_t>
translateHexChar(char ch1, char ch2)
{
    const int hi_nibble = xdigit_to_sint(ch1);
    const int lo_nibble = xdigit_to_sint(ch2);
    if (hi_nibble == -1 || lo_nibble == -1)
        return llvm::None;
    return (uint8_t)((hi_nibble << 4) + lo_nibble);
}

//----------------------------------------------------------------------
// If a pair of valid hex digits exist at the head of the
// StringExtractor they are decoded into an unsigned byte and returned
// by this function
//
// If there is not a pair of valid hex digits at the head of the
// StringExtractor, it is left unchanged and -1 is returned
//----------------------------------------------------------------------
int
StringExtractor::DecodeHexU8()
{
    SkipSpaces();
    if (GetBytesLeft() < 2)
        return -1;
    auto result = translateHexChar(m_packet[m_index], m_packet[m_index + 1]);
    if (!result.hasValue())
        return -1;
    m_index += 2;
    return *result;
}

//----------------------------------------------------------------------
// Extract an unsigned character from two hex ASCII chars in the packet
// string, or return fail_value on failure
//----------------------------------------------------------------------
uint8_t
StringExtractor::GetHexU8 (uint8_t fail_value, bool set_eof_on_fail)
{
    // On success, fail_value will be overwritten with the next
    // character in the stream
    GetHexU8Ex(fail_value, set_eof_on_fail);
    return fail_value;
}

bool
StringExtractor::GetHexU8Ex (uint8_t& ch, bool set_eof_on_fail)
{
    int byte = DecodeHexU8();
    if (byte == -1)
    {
        if (set_eof_on_fail || m_index >= m_packet.size())
            m_index = UINT64_MAX;
        // ch should not be changed in case of failure
        return false;
    }
    ch = (uint8_t)byte;
    return true;
}

uint32_t
StringExtractor::GetU32 (uint32_t fail_value, int base)
{
    if (m_index < m_packet.size())
    {
        char *end = nullptr;
        const char *start = m_packet.c_str();
        const char *cstr = start + m_index;
        uint32_t result = static_cast<uint32_t>(::strtoul (cstr, &end, base));

        if (end && end != cstr)
        {
            m_index = end - start;
            return result;
        }
    }
    return fail_value;
}

int32_t
StringExtractor::GetS32 (int32_t fail_value, int base)
{
    if (m_index < m_packet.size())
    {
        char *end = nullptr;
        const char *start = m_packet.c_str();
        const char *cstr = start + m_index;
        int32_t result = static_cast<int32_t>(::strtol (cstr, &end, base));
        
        if (end && end != cstr)
        {
            m_index = end - start;
            return result;
        }
    }
    return fail_value;
}


uint64_t
StringExtractor::GetU64 (uint64_t fail_value, int base)
{
    if (m_index < m_packet.size())
    {
        char *end = nullptr;
        const char *start = m_packet.c_str();
        const char *cstr = start + m_index;
        uint64_t result = ::strtoull (cstr, &end, base);
        
        if (end && end != cstr)
        {
            m_index = end - start;
            return result;
        }
    }
    return fail_value;
}

int64_t
StringExtractor::GetS64 (int64_t fail_value, int base)
{
    if (m_index < m_packet.size())
    {
        char *end = nullptr;
        const char *start = m_packet.c_str();
        const char *cstr = start + m_index;
        int64_t result = ::strtoll (cstr, &end, base);
        
        if (end && end != cstr)
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
    SkipSpaces();

    // Allocate enough space for 2 uint32's.  In big endian, if the user writes
    // "AB" then this should be treated as 0xAB, not 0xAB000000.  In order to
    // do this, we decode into the second half of the array, and then shift the
    // starting point of the big endian translation left by however many bytes
    // of a uint32 were missing from the input.  We're essentially padding left
    // with 0's.
    uint8_t bytes[2 * sizeof(uint32_t) - 1] = {0};
    llvm::MutableArrayRef<uint8_t> byte_array(bytes);
    llvm::MutableArrayRef<uint8_t> decode_loc = byte_array.take_back(sizeof(uint32_t));
    uint32_t bytes_decoded = GetHexBytesAvail(decode_loc);
    if (bytes_decoded == sizeof(uint32_t) && ::isxdigit(PeekChar()))
        return fail();

    using namespace llvm::support;
    if (little_endian)
        return endian::read<uint32_t, endianness::little>(decode_loc.data());
    else
    {
        decode_loc = byte_array.drop_front(bytes_decoded - 1).take_front(sizeof(uint32_t));
        return endian::read<uint32_t, endianness::big>(decode_loc.data());
    }
}

uint64_t
StringExtractor::GetHexMaxU64 (bool little_endian, uint64_t fail_value)
{
    SkipSpaces();

    // Allocate enough space for 2 uint64's.  In big endian, if the user writes
    // "AB" then this should be treated as 0x000000AB, not 0xAB000000.  In order
    // to do this, we decode into the second half of the array, and then shift
    // the starting point of the big endian translation left by however many bytes
    // of a uint32 were missing from the input.  We're essentially padding left
    // with 0's.
    uint8_t bytes[2 * sizeof(uint64_t) - 1] = {0};
    llvm::MutableArrayRef<uint8_t> byte_array(bytes);
    llvm::MutableArrayRef<uint8_t> decode_loc = byte_array.take_back(sizeof(uint64_t));
    uint32_t bytes_decoded = GetHexBytesAvail(decode_loc);
    if (bytes_decoded == sizeof(uint64_t) && ::isxdigit(PeekChar()))
        return fail();

    using namespace llvm::support;
    if (little_endian)
        return endian::read<uint64_t, endianness::little>(decode_loc.data());
    else
    {
        decode_loc = byte_array.drop_front(bytes_decoded - 1).take_front(sizeof(uint64_t));
        return endian::read<uint64_t, endianness::big>(decode_loc.data());
    }
}

size_t
StringExtractor::GetHexBytes (llvm::MutableArrayRef<uint8_t> dest, uint8_t fail_fill_value)
{
    size_t bytes_extracted = 0;
    while (!dest.empty() && GetBytesLeft() > 0)
    {
        dest[0] = GetHexU8 (fail_fill_value);
        if (!IsGood())
            break;
        ++bytes_extracted;
        dest = dest.drop_front();
    }

    if (!dest.empty())
        ::memset(dest.data(), fail_fill_value, dest.size());

    return bytes_extracted;
}

//----------------------------------------------------------------------
// Decodes all valid hex encoded bytes at the head of the
// StringExtractor, limited by dst_len.
//
// Returns the number of bytes successfully decoded
//----------------------------------------------------------------------
size_t
StringExtractor::GetHexBytesAvail (llvm::MutableArrayRef<uint8_t> dest)
{
    size_t bytes_extracted = 0;
    while (!dest.empty())
    {
        int decode = DecodeHexU8();
        if (decode == -1)
            break;
        dest[0] = (uint8_t)decode;
        dest = dest.drop_front();
        ++bytes_extracted;
    }
    return bytes_extracted;
}

size_t
StringExtractor::GetHexByteString (std::string &str)
{
    str.clear();
    str.reserve(GetBytesLeft() / 2);
    char ch;
    while ((ch = GetHexU8()) != '\0')
        str.append(1, ch);
    return str.size();
}

size_t
StringExtractor::GetHexByteStringFixedLength (std::string &str, uint32_t nibble_length)
{
    str.clear();
    llvm::StringRef nibs = Peek().take_front(nibble_length);
    while (nibs.size() >= 2)
    {
        auto ch = translateHexChar(nibs[0], nibs[1]);
        if (!ch.hasValue())
            break;
        str.push_back(*ch);
        nibs = nibs.drop_front(2);
    }
    m_index += str.size() * 2;
    return str.size();
}

size_t
StringExtractor::GetHexByteStringTerminatedBy (std::string &str,
                                               char terminator)
{
    str.clear();
    char ch;
    while ((ch = GetHexU8(0,false)) != '\0')
        str.append(1, ch);
    if (GetBytesLeft() > 0 && PeekChar() == terminator)
        return str.size();

    str.clear();
    return str.size();
}

bool
StringExtractor::GetNameColonValue(llvm::StringRef &name, llvm::StringRef &value)
{
    // Read something in the form of NNNN:VVVV; where NNNN is any character
    // that is not a colon, followed by a ':' character, then a value (one or
    // more ';' chars), followed by a ';'
    if (m_index >= m_packet.size())
        return fail();

    llvm::StringRef view(m_packet);
    if (view.empty())
        return fail();

    llvm::StringRef a, b, c, d;
    view = view.substr(m_index);
    std::tie(a, b) = view.split(':');
    if (a.empty() || b.empty())
        return fail();
    std::tie(c, d) = b.split(';');
    if (b == c && d.empty())
        return fail();

    name = a;
    value = c;
    if (d.empty())
        m_index = m_packet.size();
    else
    {
        size_t bytes_consumed = d.data() - view.data();
        m_index += bytes_consumed;
    }
    return true;
}

void
StringExtractor::SkipSpaces ()
{
    const size_t n = m_packet.size();
    while (m_index < n && isspace(m_packet[m_index]))
        ++m_index;
}

