//===-- Stream.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Stream.h"
#include "lldb/Host/PosixApi.h"
#include "lldb/Utility/Endian.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <inttypes.h>

using namespace lldb;
using namespace lldb_private;

Stream::Stream(uint32_t flags, uint32_t addr_size, ByteOrder byte_order)
    : m_flags(flags), m_addr_size(addr_size), m_byte_order(byte_order),
      m_indent_level(0) {}

Stream::Stream()
    : m_flags(0), m_addr_size(4), m_byte_order(endian::InlHostByteOrder()),
      m_indent_level(0) {}

//------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------
Stream::~Stream() {}

ByteOrder Stream::SetByteOrder(ByteOrder byte_order) {
  ByteOrder old_byte_order = m_byte_order;
  m_byte_order = byte_order;
  return old_byte_order;
}

//------------------------------------------------------------------
// Put an offset "uval" out to the stream using the printf format
// in "format".
//------------------------------------------------------------------
void Stream::Offset(uint32_t uval, const char *format) { Printf(format, uval); }

//------------------------------------------------------------------
// Put an SLEB128 "uval" out to the stream using the printf format
// in "format".
//------------------------------------------------------------------
size_t Stream::PutSLEB128(int64_t sval) {
  size_t bytes_written = 0;
  if (m_flags.Test(eBinary)) {
    bool more = true;
    while (more) {
      uint8_t byte = sval & 0x7fu;
      sval >>= 7;
      /* sign bit of byte is 2nd high order bit (0x40) */
      if ((sval == 0 && !(byte & 0x40)) || (sval == -1 && (byte & 0x40)))
        more = false;
      else
        // more bytes to come
        byte |= 0x80u;
      bytes_written += Write(&byte, 1);
    }
  } else {
    bytes_written = Printf("0x%" PRIi64, sval);
  }

  return bytes_written;
}

//------------------------------------------------------------------
// Put an ULEB128 "uval" out to the stream using the printf format
// in "format".
//------------------------------------------------------------------
size_t Stream::PutULEB128(uint64_t uval) {
  size_t bytes_written = 0;
  if (m_flags.Test(eBinary)) {
    do {

      uint8_t byte = uval & 0x7fu;
      uval >>= 7;
      if (uval != 0) {
        // more bytes to come
        byte |= 0x80u;
      }
      bytes_written += Write(&byte, 1);
    } while (uval != 0);
  } else {
    bytes_written = Printf("0x%" PRIx64, uval);
  }
  return bytes_written;
}

//------------------------------------------------------------------
// Print a raw NULL terminated C string to the stream.
//------------------------------------------------------------------
size_t Stream::PutCString(llvm::StringRef str) {
  size_t bytes_written = 0;
  bytes_written = Write(str.data(), str.size());

  // when in binary mode, emit the NULL terminator
  if (m_flags.Test(eBinary))
    bytes_written += PutChar('\0');
  return bytes_written;
}

//------------------------------------------------------------------
// Print a double quoted NULL terminated C string to the stream
// using the printf format in "format".
//------------------------------------------------------------------
void Stream::QuotedCString(const char *cstr, const char *format) {
  Printf(format, cstr);
}

//------------------------------------------------------------------
// Put an address "addr" out to the stream with optional prefix
// and suffix strings.
//------------------------------------------------------------------
void Stream::Address(uint64_t addr, uint32_t addr_size, const char *prefix,
                     const char *suffix) {
  if (prefix == NULL)
    prefix = "";
  if (suffix == NULL)
    suffix = "";
  //    int addr_width = m_addr_size << 1;
  //    Printf ("%s0x%0*" PRIx64 "%s", prefix, addr_width, addr, suffix);
  Printf("%s0x%0*" PRIx64 "%s", prefix, addr_size * 2, (uint64_t)addr, suffix);
}

//------------------------------------------------------------------
// Put an address range out to the stream with optional prefix
// and suffix strings.
//------------------------------------------------------------------
void Stream::AddressRange(uint64_t lo_addr, uint64_t hi_addr,
                          uint32_t addr_size, const char *prefix,
                          const char *suffix) {
  if (prefix && prefix[0])
    PutCString(prefix);
  Address(lo_addr, addr_size, "[");
  Address(hi_addr, addr_size, "-", ")");
  if (suffix && suffix[0])
    PutCString(suffix);
}

size_t Stream::PutChar(char ch) { return Write(&ch, 1); }

//------------------------------------------------------------------
// Print some formatted output to the stream.
//------------------------------------------------------------------
size_t Stream::Printf(const char *format, ...) {
  va_list args;
  va_start(args, format);
  size_t result = PrintfVarArg(format, args);
  va_end(args);
  return result;
}

//------------------------------------------------------------------
// Print some formatted output to the stream.
//------------------------------------------------------------------
size_t Stream::PrintfVarArg(const char *format, va_list args) {
  char str[1024];
  va_list args_copy;

  va_copy(args_copy, args);

  size_t bytes_written = 0;
  // Try and format our string into a fixed buffer first and see if it fits
  size_t length = ::vsnprintf(str, sizeof(str), format, args);
  if (length < sizeof(str)) {
    // Include the NULL termination byte for binary output
    if (m_flags.Test(eBinary))
      length += 1;
    // The formatted string fit into our stack based buffer, so we can just
    // append that to our packet
    bytes_written = Write(str, length);
  } else {
    // Our stack buffer wasn't big enough to contain the entire formatted
    // string, so lets let vasprintf create the string for us!
    char *str_ptr = NULL;
    length = ::vasprintf(&str_ptr, format, args_copy);
    if (str_ptr) {
      // Include the NULL termination byte for binary output
      if (m_flags.Test(eBinary))
        length += 1;
      bytes_written = Write(str_ptr, length);
      ::free(str_ptr);
    }
  }
  va_end(args_copy);
  return bytes_written;
}

//------------------------------------------------------------------
// Print and End of Line character to the stream
//------------------------------------------------------------------
size_t Stream::EOL() { return PutChar('\n'); }

//------------------------------------------------------------------
// Indent the current line using the current indentation level and
// print an optional string following the indentation spaces.
//------------------------------------------------------------------
size_t Stream::Indent(const char *s) {
  return Printf("%*.*s%s", m_indent_level, m_indent_level, "", s ? s : "");
}

size_t Stream::Indent(llvm::StringRef str) {
  return Printf("%*.*s%s", m_indent_level, m_indent_level, "",
                str.str().c_str());
}

//------------------------------------------------------------------
// Stream a character "ch" out to this stream.
//------------------------------------------------------------------
Stream &Stream::operator<<(char ch) {
  PutChar(ch);
  return *this;
}

//------------------------------------------------------------------
// Stream the NULL terminated C string out to this stream.
//------------------------------------------------------------------
Stream &Stream::operator<<(const char *s) {
  Printf("%s", s);
  return *this;
}

Stream &Stream::operator<<(llvm::StringRef str) {
  Write(str.data(), str.size());
  return *this;
}

//------------------------------------------------------------------
// Stream the pointer value out to this stream.
//------------------------------------------------------------------
Stream &Stream::operator<<(const void *p) {
  Printf("0x%.*tx", (int)sizeof(const void *) * 2, (ptrdiff_t)p);
  return *this;
}

//------------------------------------------------------------------
// Stream a uint8_t "uval" out to this stream.
//------------------------------------------------------------------
Stream &Stream::operator<<(uint8_t uval) {
  PutHex8(uval);
  return *this;
}

//------------------------------------------------------------------
// Stream a uint16_t "uval" out to this stream.
//------------------------------------------------------------------
Stream &Stream::operator<<(uint16_t uval) {
  PutHex16(uval, m_byte_order);
  return *this;
}

//------------------------------------------------------------------
// Stream a uint32_t "uval" out to this stream.
//------------------------------------------------------------------
Stream &Stream::operator<<(uint32_t uval) {
  PutHex32(uval, m_byte_order);
  return *this;
}

//------------------------------------------------------------------
// Stream a uint64_t "uval" out to this stream.
//------------------------------------------------------------------
Stream &Stream::operator<<(uint64_t uval) {
  PutHex64(uval, m_byte_order);
  return *this;
}

//------------------------------------------------------------------
// Stream a int8_t "sval" out to this stream.
//------------------------------------------------------------------
Stream &Stream::operator<<(int8_t sval) {
  Printf("%i", (int)sval);
  return *this;
}

//------------------------------------------------------------------
// Stream a int16_t "sval" out to this stream.
//------------------------------------------------------------------
Stream &Stream::operator<<(int16_t sval) {
  Printf("%i", (int)sval);
  return *this;
}

//------------------------------------------------------------------
// Stream a int32_t "sval" out to this stream.
//------------------------------------------------------------------
Stream &Stream::operator<<(int32_t sval) {
  Printf("%i", (int)sval);
  return *this;
}

//------------------------------------------------------------------
// Stream a int64_t "sval" out to this stream.
//------------------------------------------------------------------
Stream &Stream::operator<<(int64_t sval) {
  Printf("%" PRIi64, sval);
  return *this;
}

//------------------------------------------------------------------
// Get the current indentation level
//------------------------------------------------------------------
int Stream::GetIndentLevel() const { return m_indent_level; }

//------------------------------------------------------------------
// Set the current indentation level
//------------------------------------------------------------------
void Stream::SetIndentLevel(int indent_level) { m_indent_level = indent_level; }

//------------------------------------------------------------------
// Increment the current indentation level
//------------------------------------------------------------------
void Stream::IndentMore(int amount) { m_indent_level += amount; }

//------------------------------------------------------------------
// Decrement the current indentation level
//------------------------------------------------------------------
void Stream::IndentLess(int amount) {
  if (m_indent_level >= amount)
    m_indent_level -= amount;
  else
    m_indent_level = 0;
}

//------------------------------------------------------------------
// Get the address size in bytes
//------------------------------------------------------------------
uint32_t Stream::GetAddressByteSize() const { return m_addr_size; }

//------------------------------------------------------------------
// Set the address size in bytes
//------------------------------------------------------------------
void Stream::SetAddressByteSize(uint32_t addr_size) { m_addr_size = addr_size; }

//------------------------------------------------------------------
// The flags get accessor
//------------------------------------------------------------------
Flags &Stream::GetFlags() { return m_flags; }

//------------------------------------------------------------------
// The flags const get accessor
//------------------------------------------------------------------
const Flags &Stream::GetFlags() const { return m_flags; }

//------------------------------------------------------------------
// The byte order get accessor
//------------------------------------------------------------------

lldb::ByteOrder Stream::GetByteOrder() const { return m_byte_order; }

size_t Stream::PrintfAsRawHex8(const char *format, ...) {
  va_list args;
  va_list args_copy;
  va_start(args, format);
  va_copy(args_copy, args); // Copy this so we

  char str[1024];
  size_t bytes_written = 0;
  // Try and format our string into a fixed buffer first and see if it fits
  size_t length = ::vsnprintf(str, sizeof(str), format, args);
  if (length < sizeof(str)) {
    // The formatted string fit into our stack based buffer, so we can just
    // append that to our packet
    for (size_t i = 0; i < length; ++i)
      bytes_written += _PutHex8(str[i], false);
  } else {
    // Our stack buffer wasn't big enough to contain the entire formatted
    // string, so lets let vasprintf create the string for us!
    char *str_ptr = NULL;
    length = ::vasprintf(&str_ptr, format, args_copy);
    if (str_ptr) {
      for (size_t i = 0; i < length; ++i)
        bytes_written += _PutHex8(str_ptr[i], false);
      ::free(str_ptr);
    }
  }
  va_end(args);
  va_end(args_copy);

  return bytes_written;
}

size_t Stream::PutNHex8(size_t n, uint8_t uvalue) {
  size_t bytes_written = 0;
  for (size_t i = 0; i < n; ++i)
    bytes_written += _PutHex8(uvalue, false);
  return bytes_written;
}

size_t Stream::_PutHex8(uint8_t uvalue, bool add_prefix) {
  size_t bytes_written = 0;
  if (m_flags.Test(eBinary)) {
    bytes_written = Write(&uvalue, 1);
  } else {
    if (add_prefix)
      PutCString("0x");

    static char g_hex_to_ascii_hex_char[16] = {'0', '1', '2', '3', '4', '5',
                                               '6', '7', '8', '9', 'a', 'b',
                                               'c', 'd', 'e', 'f'};
    char nibble_chars[2];
    nibble_chars[0] = g_hex_to_ascii_hex_char[(uvalue >> 4) & 0xf];
    nibble_chars[1] = g_hex_to_ascii_hex_char[(uvalue >> 0) & 0xf];
    bytes_written = Write(nibble_chars, sizeof(nibble_chars));
  }
  return bytes_written;
}

size_t Stream::PutHex8(uint8_t uvalue) { return _PutHex8(uvalue, false); }

size_t Stream::PutHex16(uint16_t uvalue, ByteOrder byte_order) {
  if (byte_order == eByteOrderInvalid)
    byte_order = m_byte_order;

  size_t bytes_written = 0;
  if (byte_order == eByteOrderLittle) {
    for (size_t byte = 0; byte < sizeof(uvalue); ++byte)
      bytes_written += _PutHex8((uint8_t)(uvalue >> (byte * 8)), false);
  } else {
    for (size_t byte = sizeof(uvalue) - 1; byte < sizeof(uvalue); --byte)
      bytes_written += _PutHex8((uint8_t)(uvalue >> (byte * 8)), false);
  }
  return bytes_written;
}

size_t Stream::PutHex32(uint32_t uvalue, ByteOrder byte_order) {
  if (byte_order == eByteOrderInvalid)
    byte_order = m_byte_order;

  size_t bytes_written = 0;
  if (byte_order == eByteOrderLittle) {
    for (size_t byte = 0; byte < sizeof(uvalue); ++byte)
      bytes_written += _PutHex8((uint8_t)(uvalue >> (byte * 8)), false);
  } else {
    for (size_t byte = sizeof(uvalue) - 1; byte < sizeof(uvalue); --byte)
      bytes_written += _PutHex8((uint8_t)(uvalue >> (byte * 8)), false);
  }
  return bytes_written;
}

size_t Stream::PutHex64(uint64_t uvalue, ByteOrder byte_order) {
  if (byte_order == eByteOrderInvalid)
    byte_order = m_byte_order;

  size_t bytes_written = 0;
  if (byte_order == eByteOrderLittle) {
    for (size_t byte = 0; byte < sizeof(uvalue); ++byte)
      bytes_written += _PutHex8((uint8_t)(uvalue >> (byte * 8)), false);
  } else {
    for (size_t byte = sizeof(uvalue) - 1; byte < sizeof(uvalue); --byte)
      bytes_written += _PutHex8((uint8_t)(uvalue >> (byte * 8)), false);
  }
  return bytes_written;
}

size_t Stream::PutMaxHex64(uint64_t uvalue, size_t byte_size,
                           lldb::ByteOrder byte_order) {
  switch (byte_size) {
  case 1:
    return PutHex8((uint8_t)uvalue);
  case 2:
    return PutHex16((uint16_t)uvalue);
  case 4:
    return PutHex32((uint32_t)uvalue);
  case 8:
    return PutHex64(uvalue);
  }
  return 0;
}

size_t Stream::PutPointer(void *ptr) {
  return PutRawBytes(&ptr, sizeof(ptr), endian::InlHostByteOrder(),
                     endian::InlHostByteOrder());
}

size_t Stream::PutFloat(float f, ByteOrder byte_order) {
  if (byte_order == eByteOrderInvalid)
    byte_order = m_byte_order;

  return PutRawBytes(&f, sizeof(f), endian::InlHostByteOrder(), byte_order);
}

size_t Stream::PutDouble(double d, ByteOrder byte_order) {
  if (byte_order == eByteOrderInvalid)
    byte_order = m_byte_order;

  return PutRawBytes(&d, sizeof(d), endian::InlHostByteOrder(), byte_order);
}

size_t Stream::PutLongDouble(long double ld, ByteOrder byte_order) {
  if (byte_order == eByteOrderInvalid)
    byte_order = m_byte_order;

  return PutRawBytes(&ld, sizeof(ld), endian::InlHostByteOrder(), byte_order);
}

size_t Stream::PutRawBytes(const void *s, size_t src_len,
                           ByteOrder src_byte_order, ByteOrder dst_byte_order) {
  if (src_byte_order == eByteOrderInvalid)
    src_byte_order = m_byte_order;

  if (dst_byte_order == eByteOrderInvalid)
    dst_byte_order = m_byte_order;

  size_t bytes_written = 0;
  const uint8_t *src = (const uint8_t *)s;
  bool binary_was_set = m_flags.Test(eBinary);
  if (!binary_was_set)
    m_flags.Set(eBinary);
  if (src_byte_order == dst_byte_order) {
    for (size_t i = 0; i < src_len; ++i)
      bytes_written += _PutHex8(src[i], false);
  } else {
    for (size_t i = src_len - 1; i < src_len; --i)
      bytes_written += _PutHex8(src[i], false);
  }
  if (!binary_was_set)
    m_flags.Clear(eBinary);

  return bytes_written;
}

size_t Stream::PutBytesAsRawHex8(const void *s, size_t src_len,
                                 ByteOrder src_byte_order,
                                 ByteOrder dst_byte_order) {
  if (src_byte_order == eByteOrderInvalid)
    src_byte_order = m_byte_order;

  if (dst_byte_order == eByteOrderInvalid)
    dst_byte_order = m_byte_order;

  size_t bytes_written = 0;
  const uint8_t *src = (const uint8_t *)s;
  bool binary_is_set = m_flags.Test(eBinary);
  m_flags.Clear(eBinary);
  if (src_byte_order == dst_byte_order) {
    for (size_t i = 0; i < src_len; ++i)
      bytes_written += _PutHex8(src[i], false);
  } else {
    for (size_t i = src_len - 1; i < src_len; --i)
      bytes_written += _PutHex8(src[i], false);
  }
  if (binary_is_set)
    m_flags.Set(eBinary);

  return bytes_written;
}

size_t Stream::PutCStringAsRawHex8(const char *s) {
  size_t bytes_written = 0;
  bool binary_is_set = m_flags.Test(eBinary);
  m_flags.Clear(eBinary);
  do {
    bytes_written += _PutHex8(*s, false);
    ++s;
  } while (*s);
  if (binary_is_set)
    m_flags.Set(eBinary);
  return bytes_written;
}

void Stream::UnitTest(Stream *s) {
  s->PutHex8(0x12);

  s->PutChar(' ');
  s->PutHex16(0x3456, endian::InlHostByteOrder());
  s->PutChar(' ');
  s->PutHex16(0x3456, eByteOrderBig);
  s->PutChar(' ');
  s->PutHex16(0x3456, eByteOrderLittle);

  s->PutChar(' ');
  s->PutHex32(0x789abcde, endian::InlHostByteOrder());
  s->PutChar(' ');
  s->PutHex32(0x789abcde, eByteOrderBig);
  s->PutChar(' ');
  s->PutHex32(0x789abcde, eByteOrderLittle);

  s->PutChar(' ');
  s->PutHex64(0x1122334455667788ull, endian::InlHostByteOrder());
  s->PutChar(' ');
  s->PutHex64(0x1122334455667788ull, eByteOrderBig);
  s->PutChar(' ');
  s->PutHex64(0x1122334455667788ull, eByteOrderLittle);

  const char *hola = "Hello World!!!";
  s->PutChar(' ');
  s->PutCString(hola);

  s->PutChar(' ');
  s->Write(hola, 5);

  s->PutChar(' ');
  s->PutCStringAsRawHex8(hola);

  s->PutChar(' ');
  s->PutCStringAsRawHex8("01234");

  s->PutChar(' ');
  s->Printf("pid=%i", 12733);

  s->PutChar(' ');
  s->PrintfAsRawHex8("pid=%i", 12733);
  s->PutChar('\n');
}
