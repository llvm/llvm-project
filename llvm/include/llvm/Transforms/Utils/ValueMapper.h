//===- ValueMapper.h - Interface shared by lib/Transforms/Utils -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MapValue interface which is used by various parts of
// the Transforms/Utils library to implement cloning and linking facilities.
//
//===----------------------------------------------------------------------===//

#ifndef VALUEMAPPER_H
#define VALUEMAPPER_H

#include "llvm/ADT/DenseMap.h"

namespace llvm {
  class Value;
  class Instruction;
  typedef DenseMap<const Value *, Value *> ValueMapTy;

  Value *MapValue(const Value *V, ValueMapTy &VM);
  void RemapInstruction(Instruction *I, ValueMapTy &VM);
} // End llvm namespace

#endif
