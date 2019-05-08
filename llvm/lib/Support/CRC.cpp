//===--- CRC.cpp - Cyclic Redundancy Check implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements llvm::crc32 function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CRC.h"
#include "llvm/Config/config.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Threading.h"
#include <array>

using namespace llvm;

#if LLVM_ENABLE_ZLIB == 0 || !HAVE_ZLIB_H
using CRC32Table = std::array<uint32_t, 256>;

static void initCRC32Table(CRC32Table *Tbl) {
  auto Shuffle = [](uint32_t V) {
    return (V & 1) ? (V >> 1) ^ 0xEDB88320U : V >> 1;
  };

  for (size_t I = 0; I < Tbl->size(); ++I) {
    uint32_t V = Shuffle(I);
    V = Shuffle(V);
    V = Shuffle(V);
    V = Shuffle(V);
    V = Shuffle(V);
    V = Shuffle(V);
    V = Shuffle(V);
    (*Tbl)[I] = Shuffle(V);
  }
}

uint32_t llvm::crc32(uint32_t CRC, StringRef S) {
  static llvm::once_flag InitFlag;
  static CRC32Table Tbl;
  llvm::call_once(InitFlag, initCRC32Table, &Tbl);

  const uint8_t *P = reinterpret_cast<const uint8_t *>(S.data());
  size_t Len = S.size();
  CRC ^= 0xFFFFFFFFU;
  for (; Len >= 8; Len -= 8) {
    CRC = Tbl[(CRC ^ *P++) & 0xFF] ^ (CRC >> 8);
    CRC = Tbl[(CRC ^ *P++) & 0xFF] ^ (CRC >> 8);
    CRC = Tbl[(CRC ^ *P++) & 0xFF] ^ (CRC >> 8);
    CRC = Tbl[(CRC ^ *P++) & 0xFF] ^ (CRC >> 8);
    CRC = Tbl[(CRC ^ *P++) & 0xFF] ^ (CRC >> 8);
    CRC = Tbl[(CRC ^ *P++) & 0xFF] ^ (CRC >> 8);
    CRC = Tbl[(CRC ^ *P++) & 0xFF] ^ (CRC >> 8);
    CRC = Tbl[(CRC ^ *P++) & 0xFF] ^ (CRC >> 8);
  }
  while (Len--)
    CRC = Tbl[(CRC ^ *P++) & 0xFF] ^ (CRC >> 8);
  return CRC ^ 0xFFFFFFFFU;
}
#else
#include <zlib.h>
uint32_t llvm::crc32(uint32_t CRC, StringRef S) {
  return ::crc32(CRC, (const Bytef *)S.data(), S.size());
}
#endif
