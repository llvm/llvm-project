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

#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/CostTable.h"
#include "llvm/Target/TargetLowering.h"
using namespace llvm;

#define DEBUG_TYPE "AMDGPUtti"

// Declare the pass initialization routine locally as target-specific passes
// don't have a target-wide initialization entry point, and so we rely on the
// pass constructor initialization.
namespace llvm {
void initializeAMDGPUTTIPass(PassRegistry &);
}

namespace {

class AMDGPUTTI final : public ImmutablePass, public TargetTransformInfo {
  const AMDGPUTargetMachine *TM;
  const AMDGPUSubtarget *ST;
  const AMDGPUTargetLowering *TLI;

  /// Estimate the overhead of scalarizing an instruction. Insert and Extract
  /// are set if the result needs to be inserted and/or extracted from vectors.
  unsigned getScalarizationOverhead(Type *Ty, bool Insert, bool Extract) const;

public:
  AMDGPUTTI() : ImmutablePass(ID), TM(nullptr), ST(nullptr), TLI(nullptr) {
    llvm_unreachable("This pass cannot be directly constructed");
  }

  AMDGPUTTI(const AMDGPUTargetMachine *TM)
      : ImmutablePass(ID), TM(TM), ST(TM->getSubtargetImpl()),
        TLI(TM->getSubtargetImpl()->getTargetLowering()) {
    initializeAMDGPUTTIPass(*PassRegistry::getPassRegistry());
  }

  void initializePass() override { pushTTIStack(this); }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    TargetTransformInfo::getAnalysisUsage(AU);
  }

  /// Pass identification.
  static char ID;

  /// Provide necessary pointer adjustments for the two base classes.
  void *getAdjustedAnalysisPointer(const void *ID) override {
    if (ID == &TargetTransformInfo::ID)
      return (TargetTransformInfo *)this;
    return this;
  }

  bool hasBranchDivergence() const override;

  void getUnrollingPreferences(const Function *F, Loop *L,
                               UnrollingPreferences &UP) const override;

  PopcntSupportKind getPopcntSupport(unsigned IntTyWidthInBit) const override;

  unsigned getNumberOfRegisters(bool Vector) const override;
  unsigned getRegisterBitWidth(bool Vector) const override;
  unsigned getMaxInterleaveFactor() const override;
};

} // end anonymous namespace

INITIALIZE_AG_PASS(AMDGPUTTI, TargetTransformInfo, "AMDGPUtti",
                   "AMDGPU Target Transform Info", true, true, false)
char AMDGPUTTI::ID = 0;

ImmutablePass *
llvm::createAMDGPUTargetTransformInfoPass(const AMDGPUTargetMachine *TM) {
  return new AMDGPUTTI(TM);
}

bool AMDGPUTTI::hasBranchDivergence() const { return true; }

void AMDGPUTTI::getUnrollingPreferences(const Function *, Loop *L,
                                        UnrollingPreferences &UP) const {
  UP.Threshold = 300; // Twice the default.
  UP.MaxCount = UINT_MAX;
  UP.Partial = true;

  // TODO: Do we want runtime unrolling?

  for (const BasicBlock *BB : L->getBlocks()) {
    for (const Instruction &I : *BB) {
      const GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(&I);
      if (!GEP || GEP->getAddressSpace() != AMDGPUAS::PRIVATE_ADDRESS)
        continue;

      const Value *Ptr = GEP->getPointerOperand();
      const AllocaInst *Alloca = dyn_cast<AllocaInst>(GetUnderlyingObject(Ptr));
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

AMDGPUTTI::PopcntSupportKind
AMDGPUTTI::getPopcntSupport(unsigned TyWidth) const {
  assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
  return ST->hasBCNT(TyWidth) ? PSK_FastHardware : PSK_Software;
}

unsigned AMDGPUTTI::getNumberOfRegisters(bool Vec) const {
  if (Vec)
    return 0;

  // Number of VGPRs on SI.
  if (ST->getGeneration() >= AMDGPUSubtarget::SOUTHERN_ISLANDS)
    return 256;

  return 4 * 128; // XXX - 4 channels. Should these count as vector instead?
}

unsigned AMDGPUTTI::getRegisterBitWidth(bool) const {
  return 32;
}

unsigned AMDGPUTTI::getMaxInterleaveFactor() const {
  // Semi-arbitrary large amount.
  return 64;
}
