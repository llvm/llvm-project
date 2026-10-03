//===-- llvm/Support/CRC.h - Cyclic Redundancy Check-------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains implementations of CRC functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CRC_H
#define LLVM_SUPPORT_CRC_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
template <typename T> class ArrayRef;

// Compute the CRC-32 of Data.
LLVM_ABI uint32_t crc32(ArrayRef<uint8_t> Data);

// Compute the running CRC-32 of Data, with CRC being the previous value of the
// checksum.
LLVM_ABI uint32_t crc32(uint32_t CRC, ArrayRef<uint8_t> Data);

// Calculate bit-reflected CRC for given initial CRC, Data and Polynomial.
// It processes the lower DataBytes of Data from LSB onwards.
// DataBytes must be 1/2/4/8.
// Poly must already be in the bit-reflected form.
constexpr inline uint32_t calculateReflectedCRC32(uint32_t Crc, uint64_t Data,
                                                  unsigned DataBytes,
                                                  uint32_t Poly) {
  uint32_t Result = Crc;
  // Process each byte
  for (unsigned I = 0; I != DataBytes; ++I) {
    uint8_t Byte = static_cast<uint8_t>((Data >> (I * 8)) & 0xFF);
    Result ^= Byte;
    for (int J = 0; J != 8; ++J) {
      Result = (Result >> 1) ^ ((Result & 1) ? Poly : 0);
    }
  }

  return Result;
}

// Class for computing the JamCRC.
//
// We will use the "Rocksoft^tm Model CRC Algorithm" to describe the properties
// of this CRC:
//   Width  : 32
//   Poly   : 04C11DB7
//   Init   : FFFFFFFF
//   RefIn  : True
//   RefOut : True
//   XorOut : 00000000
//   Check  : 340BC6D9 (result of CRC for "123456789")
//
// In other words, this is the same as CRC-32, except that XorOut is 0 instead
// of FFFFFFFF.
//
// N.B.  We permit flexibility of the "Init" value.  Some consumers of this need
//       it to be zero.
class JamCRC {
public:
  JamCRC(uint32_t Init = 0xFFFFFFFFU) : CRC(Init) {}

  // Update the CRC calculation with Data.
  LLVM_ABI void update(ArrayRef<uint8_t> Data);

  uint32_t getCRC() const { return CRC; }

private:
  uint32_t CRC;
};

} // end namespace llvm

#endif
