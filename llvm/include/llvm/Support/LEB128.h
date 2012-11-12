//===- llvm/Support/LEB128.h - [SU]LEB128 utility functions -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares some utility functions for encoding SLEB128 and
// ULEB128 values.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_LEB128_H
#define LLVM_SYSTEM_LEB128_H

#include "llvm/Support/raw_ostream.h"

namespace llvm {

/// Utility function to encode a SLEB128 value to an output stream.
static inline void encodeSLEB128(int64_t Value, raw_ostream &OS) {
  bool More;
  do {
    uint8_t Byte = Value & 0x7f;
    // NOTE: this assumes that this signed shift is an arithmetic right shift.
    Value >>= 7;
    More = !((((Value == 0 ) && ((Byte & 0x40) == 0)) ||
              ((Value == -1) && ((Byte & 0x40) != 0))));
    if (More)
      Byte |= 0x80; // Mark this byte that that more bytes will follow.
    OS << char(Byte);
  } while (More);
}

/// Utility function to encode a ULEB128 value to an output stream.
static inline void encodeULEB128(uint64_t Value, raw_ostream &OS,
                                 unsigned Padding = 0) {
  do {
    uint8_t Byte = Value & 0x7f;
    Value >>= 7;
    if (Value != 0 || Padding != 0)
      Byte |= 0x80; // Mark this byte that that more bytes will follow.
    OS << char(Byte);
  } while (Value != 0);

  // Pad with 0x80 and emit a null byte at the end.
  if (Padding != 0) {
    for (; Padding != 1; --Padding)
      OS << '\x80';
    OS << '\x00';
  }
}

/// Utility function to encode a ULEB128 value to a buffer. Returns
/// the length in bytes of the encoded value.
static inline unsigned encodeULEB128(uint64_t Value, uint8_t *p,
                                     unsigned Padding = 0) {
  uint8_t *orig_p = p;
  do {
    uint8_t Byte = Value & 0x7f;
    Value >>= 7;
    if (Value != 0 || Padding != 0)
      Byte |= 0x80; // Mark this byte that that more bytes will follow.
    *p++ = Byte;
  } while (Value != 0);

  // Pad with 0x80 and emit a null byte at the end.
  if (Padding != 0) {
    for (; Padding != 1; --Padding)
      *p++ = '\x80';
    *p++ = '\x00';
  }
  return (unsigned)(p - orig_p);
}


/// Utility function to decode a ULEB128 value.
static inline uint64_t decodeULEB128(const uint8_t *p, unsigned *n = 0) {
  const uint8_t *orig_p = p;
  uint64_t Value = 0;
  unsigned Shift = 0;
  do {
    Value += (*p & 0x7f) << Shift;
    Shift += 7;
  } while (*p++ >= 128);
  if (n)
    *n = (unsigned)(p - orig_p);
  return Value;
}

}  // namespace llvm

#endif  // LLVM_SYSTEM_LEB128_H
