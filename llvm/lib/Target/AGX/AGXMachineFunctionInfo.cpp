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
#include "llvm/IR/Module.h"
#include "Utils/AGXBaseInfo.h"
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

bool AGX::isGlobalDataVariable(const GlobalVariable &GV) {
  unsigned AS = GV.getAddressSpace();
  return AS == AGX::GENERIC_ADDRESS || AS == AGX::DEVICE_ADDRESS ||
         AS == AGX::CONSTANT_ADDRESS;
}

AGX::GlobalDataLayout AGX::computeGlobalDataLayout(const Module &M) {
  GlobalDataLayout Layout;
  const DataLayout &DL = M.getDataLayout();
  for (const GlobalVariable &GV : M.globals()) {
    if (!isGlobalDataVariable(GV) || GV.isDeclaration() ||
        GV.getName().starts_with("llvm."))
      continue;
    Align A = DL.getValueOrABITypeAlignment(GV.getAlign(), GV.getValueType());
    uint64_t Offset = alignTo(Layout.Size, A);
    Layout.Offsets.push_back({&GV, Offset});
    Layout.Size = Offset + DL.getTypeAllocSize(GV.getValueType());
    Layout.MaxAlign = std::max(Layout.MaxAlign, A);
  }
  return Layout;
}

std::optional<uint64_t>
AGX::GlobalDataLayout::getOffset(const GlobalVariable *GV) const {
  for (const auto &[V, Offset] : Offsets)
    if (V == GV)
      return Offset;
  return std::nullopt;
}
