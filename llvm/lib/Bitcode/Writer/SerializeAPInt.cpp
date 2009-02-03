//===-- SerializeAPInt.cpp - Serialization for APInts ----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements serialization of APInts.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APInt.h"
#include "llvm/Bitcode/Serialize.h"
#include <cassert>

using namespace llvm;

void APInt::Emit(Serializer& S) const {
  S.EmitInt(BitWidth);

  if (isSingleWord())
    S.EmitInt(VAL);
  else {
    uint32_t NumWords = getNumWords();
    S.EmitInt(NumWords);
    for (unsigned i = 0; i < NumWords; ++i)
      S.EmitInt(pVal[i]);
  }
}
