//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the ContiguousBlobAccumulator methods declared in
/// ContiguousBlobAccumulator.h.
///
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/ContiguousBlobAccumulator.h"
#include "llvm/ObjectYAML/YAML.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/LEB128.h"

using namespace llvm;
using namespace llvm::yaml;

bool ContiguousBlobAccumulator::checkLimit(uint64_t Size) {
  if (!ReachedLimitErr && getOffset() + Size <= MaxSize)
    return true;
  if (!ReachedLimitErr)
    ReachedLimitErr = createStringError(errc::invalid_argument,
                                        "reached the output size limit");
  return false;
}

uint64_t ContiguousBlobAccumulator::padToAlignment(unsigned Align) {
  uint64_t CurrentOffset = getOffset();
  if (ReachedLimitErr)
    return CurrentOffset;

  uint64_t AlignedOffset = alignTo(CurrentOffset, Align == 0 ? 1 : Align);
  uint64_t PaddingSize = AlignedOffset - CurrentOffset;
  if (!checkLimit(PaddingSize))
    return CurrentOffset;

  writeZeros(PaddingSize);
  return AlignedOffset;
}

void ContiguousBlobAccumulator::writeAsBinary(const BinaryRef &Bin,
                                              uint64_t N) {
  if (!checkLimit(Bin.binary_size()))
    return;
  Bin.writeAsBinary(OS, N);
}

unsigned ContiguousBlobAccumulator::writeULEB128(uint64_t Val) {
  if (!checkLimit(sizeof(uint64_t)))
    return 0;
  return encodeULEB128(Val, OS);
}

unsigned ContiguousBlobAccumulator::writeSLEB128(int64_t Val) {
  if (!checkLimit(10))
    return 0;
  return encodeSLEB128(Val, OS);
}

void ContiguousBlobAccumulator::updateDataAt(uint64_t Pos, void *Data,
                                             size_t Size) {
  assert(Pos >= InitialOffset && Pos + Size <= getOffset());
  memcpy(&Buf[Pos - InitialOffset], Data, Size);
}
