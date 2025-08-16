//===-- Utility.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utility.h"
#include "CoreSpec.h"

void add_uint64_swap(std::vector<uint8_t> &buf, uint64_t val) {
  for (int byte = 7; byte >= 0; byte--)
    buf.push_back((val >> (byte * 8)) & 0xff);
}
void add_uint64(std::vector<uint8_t> &buf, uint64_t val) {
  for (int byte = 0; byte < 8; byte++)
    buf.push_back((val >> (byte * 8)) & 0xff);
}

void add_uint32_swap(std::vector<uint8_t> &buf, uint32_t val) {
  for (int byte = 3; byte >= 0; byte--)
    buf.push_back((val >> (byte * 8)) & 0xff);
}

void add_uint32(std::vector<uint8_t> &buf, uint32_t val) {
  for (int byte = 0; byte < 4; byte++)
    buf.push_back((val >> (byte * 8)) & 0xff);
}

void add_uint64(const CoreSpec &spec, std::vector<uint8_t> &buf, uint64_t val) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  if (spec.endian == Endian::big)
    add_uint64(buf, val);
  else
    add_uint64_swap(buf, val);
#else
  if (spec.endian == Endian::Little)
    add_uint64(buf, val);
  else
    add_uint64_swap(buf, val);
#endif
}

void add_uint32(const CoreSpec &spec, std::vector<uint8_t> &buf, uint32_t val) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  if (spec.endian == Endian::big)
    add_uint32(buf, val);
  else
    add_uint32_swap(buf, val);
#else
  if (spec.endian == Endian::Little)
    add_uint32(buf, val);
  else
    add_uint32_swap(buf, val);
#endif
}
