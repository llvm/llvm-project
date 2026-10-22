//===-- AGXMachineFunctionInfo.cpp - AGX machine function info ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AGXMachineFunctionInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

// Whether \p C is used, directly or through constant expressions, by an
// instruction of \p F
static bool isUsedIn(const Constant &C, const Function &F) {
  for (const User *U : C.users()) {
    if (const auto *I = dyn_cast<Instruction>(U)) {
      if (I->getFunction() == &F)
        return true;
    } else if (const auto *CU = dyn_cast<Constant>(U)) {
      if (!isa<GlobalValue>(CU) && isUsedIn(*CU, F))
        return true;
    }
  }
  return false;
}

unsigned
AGXMachineFunctionInfo::allocateThreadgroupVariable(const Function &F,
                                                    const GlobalVariable &GV) {
  const DataLayout &DL = F.getDataLayout();
  auto Allocate = [&](const GlobalVariable &V) {
    if (ThreadgroupOffsets.count(&V))
      return;
    Align A = DL.getValueOrABITypeAlignment(V.getAlign(), V.getValueType());
    unsigned Offset = alignTo(ThreadgroupSize, A);
    ThreadgroupOffsets[&V] = Offset;
    ThreadgroupSize = Offset + DL.getTypeAllocSize(V.getValueType());
  };
  if (ThreadgroupOffsets.empty()) {
    SmallVector<const GlobalVariable *, 4> Dynamic;
    for (const GlobalVariable &V : F.getParent()->globals()) {
      if (V.getAddressSpace() != AGX::THREADGROUP_ADDRESS || !isUsedIn(V, F))
        continue;
      if (DL.getTypeAllocSize(V.getValueType()) == 0)
        Dynamic.push_back(&V);
      else
        Allocate(V);
    }
    for (const GlobalVariable *V : Dynamic)
      Allocate(*V);
  }
  Allocate(GV);
  return ThreadgroupOffsets[&GV];
}

bool AGX::isGlobalDataVariable(const GlobalVariable &GV) {
  unsigned AS = GV.getAddressSpace();
  return AS == AGX::GENERIC_ADDRESS || AS == AGX::DEVICE_ADDRESS ||
         AS == AGX::CONSTANT_ADDRESS;
}

bool AGX::usesGlobalData(const Function &F) {
  SmallVector<const User *, 16> Work;
  SmallPtrSet<const User *, 16> Seen;
  for (const GlobalVariable &GV : F.getParent()->globals())
    if (isGlobalDataVariable(GV) && !GV.isDeclaration())
      for (const User *U : GV.users())
        Work.push_back(U);
  // Through constant expressions and initializers to the instructions
  while (!Work.empty()) {
    const User *U = Work.pop_back_val();
    if (!Seen.insert(U).second)
      continue;
    if (const auto *I = dyn_cast<Instruction>(U)) {
      if (I->getFunction() == &F)
        return true;
    } else if (isa<ConstantExpr>(U) || isa<ConstantAggregate>(U)) {
      for (const User *UU : U->users())
        Work.push_back(UU);
    }
  }
  return false;
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
