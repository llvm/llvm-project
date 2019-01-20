//===-- llvm/Support/JamCRC.h - Cyclic Redundancy Check ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains an implementation of JamCRC.
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
// N.B.  We permit flexibility of the "Init" value.  Some consumers of this need
//       it to be zero.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_JAMCRC_H
#define LLVM_SUPPORT_JAMCRC_H

#include "llvm/Support/DataTypes.h"

namespace llvm {
template <typename T> class ArrayRef;

class JamCRC {
public:
  JamCRC(uint32_t Init = 0xFFFFFFFFU) : CRC(Init) {}

  // Update the CRC calculation with Data.
  void update(ArrayRef<char> Data);

  uint32_t getCRC() const { return CRC; }

private:
  uint32_t CRC;
};
} // End of namespace llvm

#endif
