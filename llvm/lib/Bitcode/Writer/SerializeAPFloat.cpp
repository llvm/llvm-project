//===-- SerializeAPInt.cpp - Serialization for APFloat ---------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements serialization of APFloat.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APFloat.h"
#include "llvm/Bitcode/Serialize.h"

using namespace llvm;

void APFloat::Emit(Serializer& S) const {
  S.Emit(bitcastToAPInt());
}
