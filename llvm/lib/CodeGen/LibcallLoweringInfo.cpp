//===- LibcallLoweringInfo.cpp - Legacy wrapper for libcall lowering ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LibcallLoweringInfo.h"
#include "llvm/Analysis/RuntimeLibcallInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

const LibcallLoweringInfo &
llvm::getLibcallLowering(const ModuleLibcallLoweringInfo &ModuleInfo,
                         const TargetSubtargetInfo &Subtarget) {
  return ModuleInfo.getLibcallLowering(
      &Subtarget, [&](LibcallLoweringInfo &Info) {
        Subtarget.initLibcallLoweringInfo(Info);
      });
}

INITIALIZE_PASS_BEGIN(LibcallLoweringInfoWrapper, "libcall-lowering-info",
                      "Library Function Lowering Analysis", false, true)
INITIALIZE_PASS_DEPENDENCY(RuntimeLibraryInfoWrapper)
INITIALIZE_PASS_END(LibcallLoweringInfoWrapper, "libcall-lowering-info",
                    "Library Function Lowering Analysis", false, true)

char LibcallLoweringInfoWrapper::ID = 0;

LibcallLoweringInfoWrapper::LibcallLoweringInfoWrapper() : ImmutablePass(ID) {}

const LibcallLoweringInfo &LibcallLoweringInfoWrapper::getLibcallLowering(
    const Module &M, const TargetSubtargetInfo &Subtarget) {
  return getResult(M).getLibcallLowering(
      &Subtarget, [&](LibcallLoweringInfo &Info) {
        Subtarget.initLibcallLoweringInfo(Info);
      });
}

const ModuleLibcallLoweringInfo &
LibcallLoweringInfoWrapper::getResult(const Module &M) {
  if (!Result)
    Result.init(&RuntimeLibcallsWrapper->getRTLCI(M));
  return Result;
}

void LibcallLoweringInfoWrapper::initializePass() {
  RuntimeLibcallsWrapper = &getAnalysis<RuntimeLibraryInfoWrapper>();
}

void LibcallLoweringInfoWrapper::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<RuntimeLibraryInfoWrapper>();
  AU.setPreservesAll();
}

void LibcallLoweringInfoWrapper::releaseMemory() { Result.clear(); }

ModulePass *llvm::createLibcallLoweringInfoWrapper() {
  return new LibcallLoweringInfoWrapper();
}
