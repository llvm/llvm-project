//===-- AGXMachineFunctionInfo.cpp - AGX machine function info ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AGXMachineFunctionInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

unsigned
AGXMachineFunctionInfo::allocateThreadgroupVariable(const DataLayout &DL,
                                                    const GlobalVariable &GV) {
  auto [It, Inserted] = ThreadgroupOffsets.try_emplace(&GV, 0);
  if (!Inserted)
    return It->second;
  Align A = DL.getValueOrABITypeAlignment(GV.getAlign(), GV.getValueType());
  unsigned Offset = alignTo(ThreadgroupSize, A);
  It->second = Offset;
  ThreadgroupSize = Offset + DL.getTypeAllocSize(GV.getValueType());
  return Offset;
}
