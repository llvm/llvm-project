//===- AssignGUIDAnalysis.cpp - assign a GUID to GV -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Having a consistent GUID for GVs throughout frontend optimization, thinlto,
// and backend optimization is essential to PGO.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AssignGUIDAnalysis.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

using namespace llvm;

AnalysisKey AssignGUIDAnalysis::Key;

AssignGUIDAnalysis::Result::Result(Module &M) : M(M) {
  for (auto &GV : M.globals())
    GV.setGUIDIfNotPresent();
  for (auto &F : M.functions())
    F.setGUIDIfNotPresent();
}

AssignGUIDAnalysis::Result AssignGUIDAnalysis::run(Module &M, ModuleAnalysisManager &MAM) {
  return Result(M);
}

void AssignGUIDAnalysis::Result::generateGuidTable() {
  auto *N = M.getOrInsertNamedMetadata("guid_table");
  assert(N->getNumOperands() == 0 && "Expected guid_table to be empty");
  for (auto &GV : M.globals())
    if (auto *MD = GV.getMetadata("guid"))
      N->addOperand(
          MDNode::get(GV.getContext(), {ValueAsMetadata::get(&GV), MD}));
  for (auto &F : M.functions())
    if (auto *MD = F.getMetadata("guid"))
      N->addOperand(
          MDNode::get(F.getContext(), {ValueAsMetadata::get(&F), MD}));
}