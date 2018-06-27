//===-- UUID.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/UUID.h"

// Other libraries and framework includes
// Project includes
#include "lldb/Utility/Stream.h"
#include "llvm/ADT/StringRef.h"

// C Includes
#include <ctype.h>
#include <stdio.h>
#include <string.h>

using namespace lldb_private;

UUID::UUID(llvm::ArrayRef<uint8_t> bytes) {
  if (bytes.size() != 20 && bytes.size() != 16)
    bytes = {};

  m_num_uuid_bytes = bytes.size();
  std::memcpy(m_uuid, bytes.data(), bytes.size());
}

std::string UUID::GetAsString(const char *separator) const {
  std::string result;
  char buf[256];
  if (!separator)
    separator = "-";
  const uint8_t *u = GetBytes().data();
  if (sizeof(buf) >
      (size_t)snprintf(buf, sizeof(buf), "%2.2X%2.2X%2.2X%2.2X%s%2.2X%2.2X%s%2."
                                         "2X%2.2X%s%2.2X%2.2X%s%2.2X%2.2X%2.2X%"
                                         "2.2X%2.2X%2.2X",
                       u[0], u[1], u[2], u[3], separator, u[4], u[5], separator,
                       u[6], u[7], separator, u[8], u[9], separator, u[10],
                       u[11], u[12], u[13], u[14], u[15])) {
    result.append(buf);
    if (m_num_uuid_bytes == 20) {
      if (sizeof(buf) > (size_t)snprintf(buf, sizeof(buf),
                                         "%s%2.2X%2.2X%2.2X%2.2X", separator,
                                         u[16], u[17], u[18], u[19]))
        result.append(buf);
    }
  }
  return result;
}

void UUID::Dump(Stream *s) const {
  s->PutCString(GetAsString().c_str());
}

static inline int xdigit_to_int(char ch) {
  ch = tolower(ch);
  if (ch >= 'a' && ch <= 'f')
    return 10 + ch - 'a';
  return ch - '0';
}

llvm::StringRef UUID::DecodeUUIDBytesFromString(llvm::StringRef p,
                                                ValueType &uuid_bytes,
                                                uint32_t &bytes_decoded,
                                                uint32_t num_uuid_bytes) {
  ::memset(uuid_bytes, 0, sizeof(uuid_bytes));
  size_t uuid_byte_idx = 0;
  while (!p.empty()) {
    if (isxdigit(p[0]) && isxdigit(p[1])) {
      int hi_nibble = xdigit_to_int(p[0]);
      int lo_nibble = xdigit_to_int(p[1]);
      // Translate the two hex nibble characters into a byte
      uuid_bytes[uuid_byte_idx] = (hi_nibble << 4) + lo_nibble;

      // Skip both hex digits
      p = p.drop_front(2);

      // Increment the byte that we are decoding within the UUID value and
      // break out if we are done
      if (++uuid_byte_idx == num_uuid_bytes)
        break;
    } else if (p.front() == '-') {
      // Skip dashes
      p = p.drop_front();
    } else {
      // UUID values can only consist of hex characters and '-' chars
      break;
    }
  }

  // Clear trailing bytes to 0.
  for (uint32_t i = uuid_byte_idx; i < sizeof(ValueType); i++)
    uuid_bytes[i] = 0;
  bytes_decoded = uuid_byte_idx;
  return p;
}

size_t UUID::SetFromStringRef(llvm::StringRef str, uint32_t num_uuid_bytes) {
  llvm::StringRef p = str;

  // Skip leading whitespace characters
  p = p.ltrim();

  ValueType bytes;
  uint32_t bytes_decoded = 0;
  llvm::StringRef rest =
      UUID::DecodeUUIDBytesFromString(p, bytes, bytes_decoded, num_uuid_bytes);

  // If we successfully decoded a UUID, return the amount of characters that
  // were consumed
  if (bytes_decoded == num_uuid_bytes) {
    *this = fromData(bytes, bytes_decoded);
    return str.size() - rest.size();
  }

  // Else return zero to indicate we were not able to parse a UUID value
  return 0;
}

bool lldb_private::operator==(const lldb_private::UUID &lhs,
                              const lldb_private::UUID &rhs) {
  return lhs.GetBytes() == rhs.GetBytes();
}

bool lldb_private::operator!=(const lldb_private::UUID &lhs,
                              const lldb_private::UUID &rhs) {
  return !(lhs == rhs);
}

bool lldb_private::operator<(const lldb_private::UUID &lhs,
                             const lldb_private::UUID &rhs) {
  if (lhs.GetBytes().size() != rhs.GetBytes().size())
    return lhs.GetBytes().size() < rhs.GetBytes().size();

  return std::memcmp(lhs.GetBytes().data(), rhs.GetBytes().data(),
                     lhs.GetBytes().size());
}

bool lldb_private::operator<=(const lldb_private::UUID &lhs,
                              const lldb_private::UUID &rhs) {
  return !(lhs > rhs);
}

bool lldb_private::operator>(const lldb_private::UUID &lhs,
                             const lldb_private::UUID &rhs) {
  return rhs < lhs;
}

bool lldb_private::operator>=(const lldb_private::UUID &lhs,
                              const lldb_private::UUID &rhs) {
  return !(lhs < rhs);
}
