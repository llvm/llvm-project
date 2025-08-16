//===- SIConvertWaveSize.cpp ----------------------------------------------===//
//
//   Automatically converts wave32 kernels to wave64
//
// Part of the LLVM Project, under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
/// \file
// Small short living kernels may become waveslot limited.
// To work around the problem an optimization is proposed to convert such
// kernels from wave32 to wave64 automatically.These kernels shall conform to a
// strict set of limitations and satisfy profitability conditions.
//
// 1. A kernel shall have no function calls as we cannot analyze call stack
// requirements (nor will it fall into a category of short living kernels
// anyway).
// 2. A kernel itself shall not be called from a device enqueue call.
// 3. A kernel shall not attempt to access EXEC or VCC in any user visible
// way.
// 4. A kernel must not use readlane/readfirstlane or any cross-lane/DPP
// operations in general.
// 5. A kernel shall not read wavefront size or use ballot through
// intrinsics (a use of pre-defined frontend wave size macro was deemed
// permissible for now).
// 6. There shall be no atomic operations of any sort as these may be used
// for cross-thread communication.
// 7. There shall be no LDS access as the allocation is usually tied to the
// workgroup size and we generally cannot extend it. It is also changing
// occupancy which is tied to the wave size.
// 8. There shall be no inline asm calls.
// 9 .There shall be no dynamic VGPRs.
// 10 .Starting from GFX11 some instructions (such as WMMA on GFX11+ and
// transpose loads on GFX12+) work differently (have different operands) in
// wave32 and wave64. The kernel shall not have intrinsics to invoke such
// instructions.

#include "AMDGPUConvertWaveSize.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-convert-wave-size"

namespace {
class AMDGPUConvertWaveSize {
  const GCNTargetMachine *TM;
  const LoopInfo *LI;
  ScalarEvolution *SE;
  TargetTransformInfo *TTI;

  InstructionCost TotalCost = 0;

  static const unsigned MaxLatency = 2000;

  SmallVector<Function *> Callees;

public:
  AMDGPUConvertWaveSize(const GCNTargetMachine *TM, const LoopInfo *LI,
                        ScalarEvolution *SE, TargetTransformInfo *TTI)
      : TM(TM), LI(LI), SE(SE), TTI(TTI) {}

  bool run(Function &F);
};

class AMDGPUConvertWaveSizeLegacy : public FunctionPass {
  const GCNTargetMachine *TM;

public:
  static char ID;
  AMDGPUConvertWaveSizeLegacy(const GCNTargetMachine *TM) : FunctionPass(ID), TM(TM) {}
  bool runOnFunction(Function &F) override {
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    AMDGPUConvertWaveSize Impl(TM, &LI, &SE, &TTI);
    return Impl.run(F);
  }
  StringRef getPassName() const override { return "AMDGPU convert wave size"; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.setPreservesAll();
    FunctionPass::getAnalysisUsage(AU);
  }
};
} // end anonymous namespace

void printFunctionAttributes(const Function &F) {
  LLVM_DEBUG(dbgs() << "Function: " << F.getName() << "\n");
  for (const auto &Attr : F.getAttributes()) {
    LLVM_DEBUG(dbgs() << "  Attribute: " << Attr.getAsString() << "\n");
  }
}

bool AMDGPUConvertWaveSize::run(Function &F) {

  // Check if the function is a kernel.
  if (F.getCallingConv() != CallingConv::AMDGPU_KERNEL)
    return false;

  const GCNSubtarget &ST = TM->getSubtarget<GCNSubtarget>(F);
  if (!ST.isWave32()) {
    LLVM_DEBUG(dbgs() << "AMDGPUConvertWaveSize: Kernel is not wave32.\n");
    return false;
  }

  for (const auto &Arg : F.args()) {
    if (Arg.getType()->isPointerTy() &&
        Arg.getType()->getPointerAddressSpace() == AMDGPUAS::LOCAL_ADDRESS) {
      LLVM_DEBUG(dbgs() << "AMDGPUConvertWaveSize: Kernel argument " << Arg
                        << " points to LDS object\n");
      return false;
    }
  }

  // Check for static LDS uses
  const Module *M = F.getParent();
  for (const GlobalVariable &GV : M->globals()) {
    if (GV.getAddressSpace() != AMDGPUAS::LOCAL_ADDRESS)
      continue;

    for (auto User : GV.users()) {
      if (auto UseI = dyn_cast<Instruction>(User)) {
        if (UseI->getFunction() == &F) {
          LLVM_DEBUG(dbgs() << "AMDGPUConvertWaveSize: Global variable " << GV
                            << " points to LDS object and is used\n");
          return false;
        }
      }
    }
  }

  // Check if the kernel can be called via device enqueue.
  if (F.hasAddressTaken()) {
    LLVM_DEBUG(dbgs() << "AMDGPUConvertWaveSize: Kernel address is taken.\n");
    return false;
  }

  // Check if a trip count is a compile time constant for all loops in the
  // kernel
  for (Loop *L : *LI) {
    const SCEV *TripCountSCEV = SE->getBackedgeTakenCount(L);
    if (!isa<SCEVConstant>(TripCountSCEV)) {
      LLVM_DEBUG(
          dbgs() << "AMDGPUConvertWaveSize: Trip count is not a compile time "
                    "constant.\n");
      return false;
    }
  }

  for (const auto &BB : F) {
    InstructionCost BlockCost = 0;
    for (const auto &I : BB) {

      // Atomic operations are not allowed.
      if (I.isAtomic()) {
        LLVM_DEBUG(
            dbgs() << "AMDGPUConvertWaveSize: Atomic operation detected.\n");
        return false;
      }

      if (const CallBase *CB = dyn_cast<CallBase>(&I)) {
        // FIXME: Any calls are not allowed. Only non-converged intrinsic calls
        // and amdgsn_s_barrier are exempt. InlineAsm is checked separately
        // for debug purposes. This will be changed in the final version.
        if (CB->isInlineAsm()) {
          // Inline assembly is not allowed.
          LLVM_DEBUG(dbgs()
                     << "AMDGPUConvertWaveSize: Inline assembly detected.\n");
          return false;
        }

        if (Function *Callee = CB->getCalledFunction()) {
          // assuming readlane/readfirstlane or any cross-lane/DPP
          // operations have "let isConvergent = 1" in IntrinsicsAMDGPU.td
          if (Callee->isIntrinsic()) {
              if (Callee->hasFnAttribute(Attribute::Convergent)) {
                if (Callee->getIntrinsicID() != Intrinsic::amdgcn_s_barrier) {
                  // TODO: what else should go in a "white list" ?
                  LLVM_DEBUG(dbgs()
                             << "AMDGPUConvertWaveSize: Convergent intrinsic "
                             << Callee->getName() << " detected.\n");
                  return false;
                }
              }

              if (Callee->getIntrinsicID() == Intrinsic::read_register ||
                  Callee->getIntrinsicID() == Intrinsic::write_register) {

                LLVM_DEBUG(dbgs()
                           << "AMDGPUConvertWaveSize: read/write_register "
                              "intrinsic detected.\n");
                return false;
              }

            // Save callee as a candidate for attribute change
            Callees.push_back(Callee);
          }
        } else {
          // General calls are not allowed.
          LLVM_DEBUG(dbgs() << "AMDGPUConvertWaveSize: function call detected.\n");
          return false;
        }
      }
      // No  LDS access is allowed

      // We already ensured we have no LDS pointers passed as arguments.
      // Now take care of those cast from flat or global

      // Bail out early, before we come across the LDS addres use.
      if (const auto AC = dyn_cast<AddrSpaceCastInst>(&I)) {
        if (AC->getDestTy()->getPointerAddressSpace() ==
            AMDGPUAS::LOCAL_ADDRESS) {
          LLVM_DEBUG(
              dbgs()
              << "AMDGPUConvertWaveSize: addrspacecast to LDS detected.\n");
          return false;
        }
      }

      if (const auto I2P = dyn_cast<IntToPtrInst>(&I)) {
        if (I2P->getDestTy()->isPointerTy() &&
            I2P->getDestTy()->getPointerAddressSpace() ==
                AMDGPUAS::LOCAL_ADDRESS) {
          LLVM_DEBUG(dbgs() << "AMDGPUConvertWaveSize: convertion int to LDS "
                               "pointer detected.\n");
          return false;
        }
      }

      // TODO: Dynamic VGPRS and GFX11+ special operations ???

      BlockCost +=
          TTI->getInstructionCost(&I, TargetTransformInfo::TCK_Latency);
    }
    if (auto L = LI->getLoopFor(&BB)) {
      const SCEV *TripCount = SE->getBackedgeTakenCount(L);
      if (auto *C = dyn_cast<SCEVConstant>(TripCount)) {
        uint64_t TC = C->getValue()->getZExtValue() + 1;
        size_t Depth = LI->getLoopDepth(&BB);
        BlockCost *= TC * Depth;
      } else
        llvm_unreachable("AMDGPUConvertWaveSize: only loops with compile time "
                         "constant trip count could reach here!\n");
    }
    TotalCost += BlockCost;
    if (TotalCost.isValid()) {
      if (TotalCost.getValue().value() >= MaxLatency) {
        LLVM_DEBUG(
            dbgs() << "AMDGPUConvertWaveSize: Total latency of the kernel ["
                   << TotalCost.getValue().value()
                   << "] exceeds the limit of 2000 cycles - not profitable!\n");
        return false;
      }
    } else
      llvm_unreachable(
          "AMDGPUConvertWaveSize: Cost model error - invalid state!\n");
  }

  // Additional checks can be added here...

  // If all checks pass, convert wave size from wave32 to wave64.
  F.addFnAttr("target-features", "+wavefrontsize64");
  LLVM_DEBUG(dbgs() << "AMDGPUConvertWaveSize: Converted wave size for "
                    << F.getName() << " from wave32 to wave64.\n");
  // Now take care of the intrinsic calls
  for (auto C : Callees) {
    C->addFnAttr("target-features", "+wavefrontsize64");
    LLVM_DEBUG(dbgs() << "AMDGPUConvertWaveSize: Converted wave size for "
                      << C->getName() << " from wave32 to wave64.\n");
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

INITIALIZE_PASS_BEGIN(AMDGPUConvertWaveSizeLegacy, DEBUG_TYPE, "AMDGPU convert wave size",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(AMDGPUConvertWaveSizeLegacy, DEBUG_TYPE, "AMDGPU convert wave size",
                    false, false)

char AMDGPUConvertWaveSizeLegacy::ID = 0;

char &llvm::AMDGPUConvertWaveSizeLegacyID = AMDGPUConvertWaveSizeLegacy::ID;

FunctionPass *llvm::createAMDGPUConvertWaveSizeLegacyPass(const GCNTargetMachine *TM) {
  return new AMDGPUConvertWaveSizeLegacy(TM);
}

PreservedAnalyses AMDGPUConvertWaveSizePass::run(
    Function &F, FunctionAnalysisManager &FAM) {
  auto &LI = FAM.getResult<LoopAnalysis>(F);
  auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
  auto &TTI = FAM.getResult<TargetIRAnalysis>(F);

  AMDGPUConvertWaveSize Impl(TM, &LI, &SE, &TTI);
  bool Changed = Impl.run(F);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
