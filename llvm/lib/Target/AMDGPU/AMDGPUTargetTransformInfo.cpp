//===-- AMDGPUTargetTransformInfo.cpp - AMDGPU specific TTI pass ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// \file
// This file implements a TargetTransformInfo analysis pass specific to the
// AMDGPU target machine. It uses the target's detailed information to provide
// more precise answers to certain TTI queries, while letting the target
// independent and default TTI implementations handle the rest.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUTargetTransformInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/CostTable.h"
#include "llvm/Target/TargetLowering.h"
using namespace llvm;

#define DEBUG_TYPE "AMDGPUtti"

void AMDGPUTTIImpl::getUnrollingPreferences(Loop *L,
                                            TTI::UnrollingPreferences &UP) {
  UP.Threshold = 300; // Twice the default.
  UP.MaxCount = UINT_MAX;
  UP.Partial = true;

  // TODO: Do we want runtime unrolling?

  for (const BasicBlock *BB : L->getBlocks()) {
    const DataLayout &DL = BB->getModule()->getDataLayout();
    for (const Instruction &I : *BB) {
      const GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(&I);
      if (!GEP || GEP->getAddressSpace() != AMDGPUAS::PRIVATE_ADDRESS)
        continue;

      const Value *Ptr = GEP->getPointerOperand();
      const AllocaInst *Alloca =
          dyn_cast<AllocaInst>(GetUnderlyingObject(Ptr, DL));
      if (Alloca) {
        // We want to do whatever we can to limit the number of alloca
        // instructions that make it through to the code generator.  allocas
        // require us to use indirect addressing, which is slow and prone to
        // compiler bugs.  If this loop does an address calculation on an
        // alloca ptr, then we want to use a higher than normal loop unroll
        // threshold. This will give SROA a better chance to eliminate these
        // allocas.
        //
        // Don't use the maximum allowed value here as it will make some
        // programs way too big.
        UP.Threshold = 800;
      }
    }
  }
}

unsigned AMDGPUTTIImpl::getNumberOfRegisters(bool Vec) {
  if (Vec)
    return 0;

  // Number of VGPRs on SI.
  if (ST->getGeneration() >= AMDGPUSubtarget::SOUTHERN_ISLANDS)
    return 256;

  return 4 * 128; // XXX - 4 channels. Should these count as vector instead?
}

unsigned AMDGPUTTIImpl::getRegisterBitWidth(bool) { return 32; }

unsigned AMDGPUTTIImpl::getMaxInterleaveFactor(unsigned VF) {
  // Semi-arbitrary large amount.
  return 64;
}
