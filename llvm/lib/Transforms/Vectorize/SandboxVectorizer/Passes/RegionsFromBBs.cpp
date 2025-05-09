//===- RegionsFromBBs.cpp - A helper to test RegionPasses -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/RegionsFromBBs.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/SandboxIR/Region.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/SandboxVectorizerPassBuilder.h"

namespace llvm::sandboxir {

RegionsFromBBs::RegionsFromBBs(StringRef Pipeline)
    : FunctionPass("regions-from-bbs"),
      RPM("rpm", Pipeline, SandboxVectorizerPassBuilder::createRegionPass) {}

bool RegionsFromBBs::runOnFunction(Function &F, const Analyses &A) {
  SmallVector<std::unique_ptr<Region>, 16> Regions;
  // Create a region for each BB.
  for (BasicBlock &BB : F) {
    Regions.push_back(std::make_unique<Region>(F.getContext(), A.getTTI()));
    auto &RgnPtr = Regions.back();
    for (Instruction &I : BB)
      RgnPtr->add(&I);
  }
  // For each region run the region pass pipeline.
  for (auto &RgnPtr : Regions)
    RPM.runOnRegion(*RgnPtr, A);
  return false;
}

} // namespace llvm::sandboxir
