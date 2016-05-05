//===-- TypeStream.cpp ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Utilities for parsing CodeView type streams.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/DebugInfo/CodeView/TypeStream.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::support;

bool llvm::codeview::decodeNumericLeaf(ArrayRef<uint8_t> &Data, APSInt &Num) {
  // Used to avoid overload ambiguity on APInt construtor.
  bool FalseVal = false;
  if (Data.size() < 2)
    return false;
  uint16_t Short = *reinterpret_cast<const ulittle16_t *>(Data.data());
  Data = Data.drop_front(2);
  if (Short < LF_NUMERIC) {
    Num = APSInt(APInt(/*numBits=*/16, Short, /*isSigned=*/false),
                 /*isUnsigned=*/true);
    return true;
  }
  switch (Short) {
  case LF_CHAR:
    Num = APSInt(APInt(/*numBits=*/8,
                       *reinterpret_cast<const int8_t *>(Data.data()),
                       /*isSigned=*/true),
                 /*isUnsigned=*/false);
    Data = Data.drop_front(1);
    return true;
  case LF_SHORT:
    Num = APSInt(APInt(/*numBits=*/16,
                       *reinterpret_cast<const little16_t *>(Data.data()),
                       /*isSigned=*/true),
                 /*isUnsigned=*/false);
    Data = Data.drop_front(2);
    return true;
  case LF_USHORT:
    Num = APSInt(APInt(/*numBits=*/16,
                       *reinterpret_cast<const ulittle16_t *>(Data.data()),
                       /*isSigned=*/false),
                 /*isUnsigned=*/true);
    Data = Data.drop_front(2);
    return true;
  case LF_LONG:
    Num = APSInt(APInt(/*numBits=*/32,
                       *reinterpret_cast<const little32_t *>(Data.data()),
                       /*isSigned=*/true),
                 /*isUnsigned=*/false);
    Data = Data.drop_front(4);
    return true;
  case LF_ULONG:
    Num = APSInt(APInt(/*numBits=*/32,
                       *reinterpret_cast<const ulittle32_t *>(Data.data()),
                       /*isSigned=*/FalseVal),
                 /*isUnsigned=*/true);
    Data = Data.drop_front(4);
    return true;
  case LF_QUADWORD:
    Num = APSInt(APInt(/*numBits=*/64,
                       *reinterpret_cast<const little64_t *>(Data.data()),
                       /*isSigned=*/true),
                 /*isUnsigned=*/false);
    Data = Data.drop_front(8);
    return true;
  case LF_UQUADWORD:
    Num = APSInt(APInt(/*numBits=*/64,
                       *reinterpret_cast<const ulittle64_t *>(Data.data()),
                       /*isSigned=*/false),
                 /*isUnsigned=*/true);
    Data = Data.drop_front(8);
    return true;
  }
  return false;
}

/// Decode a numeric leaf value that is known to be a uint32_t.
bool llvm::codeview::decodeUIntLeaf(ArrayRef<uint8_t> &Data, uint64_t &Num) {
  APSInt N;
  if (!decodeNumericLeaf(Data, N))
    return false;
  if (N.isSigned() || !N.isIntN(64))
    return false;
  Num = N.getLimitedValue();
  return true;
}
