//===- RegionsFromMetadata.cpp - A helper to test RegionPasses -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/RegionsFromMetadata.h"

#include "llvm/SandboxIR/Region.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/SandboxVectorizerPassBuilder.h"

namespace llvm::sandboxir {

RegionsFromMetadata::RegionsFromMetadata(StringRef Pipeline)
    : FunctionPass("regions-from-metadata"),
      RPM("rpm", Pipeline, SandboxVectorizerPassBuilder::createRegionPass) {}

bool RegionsFromMetadata::runOnFunction(Function &F, const Analyses &A) {
  SmallVector<std::unique_ptr<sandboxir::Region>> Regions =
      sandboxir::Region::createRegionsFromMD(F);
  for (auto &R : Regions) {
    RPM.runOnRegion(*R, A);
  }
  return false;
}

} // namespace llvm::sandboxir
