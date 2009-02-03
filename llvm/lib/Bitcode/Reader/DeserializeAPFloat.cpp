//===-- SerializeAPInt.cpp - Serialization for APFloat ---------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements deserialization of APFloat.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APFloat.h"
#include "llvm/Bitcode/Deserialize.h"

using namespace llvm;

APFloat APFloat::ReadVal(Deserializer& D) {
  APInt x;
  D.Read(x);
  return APFloat(x);
}

