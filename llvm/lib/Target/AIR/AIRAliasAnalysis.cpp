//===-- AIRAliasAnalysis.cpp - AIR address space alias analysis -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The address space alias analysis of GPU memory for the optimizations that
// run under the AIR target: device, threadgroup, constant and thread memory do
// not overlap, and constant memory is read only. A generic pointer may point
// into any of them.
//
// Two numberings meet here. A module in AIR (the "air" triple) has AIR's:
// thread 0, device 1, constant 2, threadgroup 3 and the generic hint 4
// (llvm/IR/AIR.h). A module without a triple is a front end's IR on its way to
// AIR, as ZLUDA optimizes PTX before writing AIR, in the numbering of the GPU
// targets (NVPTX, AMDGPU, AGX): generic 0, global 1, shared 3, constant 4,
// private 5. Any other triple gets no answer from this analysis.
//
//===----------------------------------------------------------------------===//

#include "AIRAliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

AnalysisKey AIRAA::Key;

static cl::opt<bool> EnableAIRAA("air-aa", cl::init(true), cl::Hidden,
                                 cl::desc("Address space alias analysis for "
                                          "the AIR target's optimizations"));

namespace {
struct Spaces {
  unsigned Generic, Constant;
};

/// The numbering of the module of V, if this analysis knows it.
std::optional<Spaces> spacesOf(const Value *V) {
  const Module *M = nullptr;
  if (auto *I = dyn_cast<Instruction>(V))
    M = I->getModule();
  else if (auto *A = dyn_cast<Argument>(V))
    M = A->getParent()->getParent();
  else if (auto *G = dyn_cast<GlobalValue>(V))
    M = G->getParent();
  if (!M)
    return std::nullopt;
  const Triple &T = M->getTargetTriple();
  if (T.getArchName() == "air")
    return Spaces{4, 2};
  if (T.str().empty())
    return Spaces{0, 4};
  return std::nullopt;
}

/// The address space of a pointer, looking through a few generic casts to
/// the object.
unsigned addressSpace(const Value *V, unsigned Generic) {
  auto AS = [&](const Value *V) {
    if (auto *T = dyn_cast<PointerType>(V->getType()))
      return T->getAddressSpace();
    return Generic;
  };
  for (unsigned I = 0; I < 6 && AS(V) == Generic; ++I) {
    const Value *U = getUnderlyingObject(V, 1);
    if (U == V)
      break;
    V = U;
  }
  return AS(V);
}
} // namespace

AliasResult AIRAAResult::alias(const MemoryLocation &LocA,
                               const MemoryLocation &LocB, AAQueryInfo &,
                               const Instruction *) {
  if (!EnableAIRAA)
    return AliasResult::MayAlias;
  std::optional<Spaces> S = spacesOf(LocA.Ptr);
  if (!S)
    S = spacesOf(LocB.Ptr);
  if (!S)
    return AliasResult::MayAlias;
  unsigned A = addressSpace(LocA.Ptr, S->Generic),
           B = addressSpace(LocB.Ptr, S->Generic);
  if (A == S->Generic || B == S->Generic || A == B)
    return AliasResult::MayAlias;
  return AliasResult::NoAlias;
}

ModRefInfo AIRAAResult::getModRefInfoMask(const MemoryLocation &Loc,
                                          AAQueryInfo &, bool) {
  std::optional<Spaces> S =
      EnableAIRAA ? spacesOf(Loc.Ptr) : std::nullopt;
  if (S && addressSpace(Loc.Ptr, S->Generic) == S->Constant)
    return ModRefInfo::NoModRef;
  return ModRefInfo::ModRef;
}
