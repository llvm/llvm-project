//===-- DeserializeAPInt.cpp - Deserialization for APInts ------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements deserialization of APInts.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APInt.h"
#include "llvm/Bitcode/Deserialize.h"
#include <cassert>

using namespace llvm;

void APInt::Read(Deserializer& D) {
  BitWidth = D.ReadInt();
  
  if (isSingleWord())
    VAL = D.ReadInt();
  else {
    uint32_t NumWords = D.ReadInt();
    assert (NumWords > 1);
    pVal = new uint64_t[NumWords];
    assert (pVal && "Allocation in deserialization of APInt failed.");
    for (unsigned i = 0; i < NumWords; ++i)
      pVal[i] = D.ReadInt();    
  }
}
