//===- SIConvertWaveSize.cpp - Automatically converts wave32 kernels to wave64
//---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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

#include "SIConvertWaveSize.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-convert-wave-size"

namespace {
class SIConvertWaveSize {
  const TargetMachine *TM;
  const LoopInfo *LI;
  ScalarEvolution *SE;
  TargetTransformInfo *TTI;

  InstructionCost TotalCost = 0;

  static const unsigned MaxLatency = 2000;

  SmallVector<Function *> Callees;

public:
  SIConvertWaveSize(const TargetMachine *TM, const LoopInfo *LI,
                    ScalarEvolution *SE, TargetTransformInfo *TTI)
      : TM(TM), LI(LI), SE(SE), TTI(TTI) {}

  bool run(Function &F);

  bool changeWaveSizeAttr(Function *F);
};

class SIConvertWaveSizeLegacy : public FunctionPass {
  const TargetMachine *TM;

public:
  static char ID;
  SIConvertWaveSizeLegacy(const TargetMachine *TM) : FunctionPass(ID), TM(TM) {}
  bool runOnFunction(Function &F) override {
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    SIConvertWaveSize Impl(TM, &LI, &SE, &TTI);
    return Impl.run(F);
  }
  StringRef getPassName() const override { return "SI convert wave size"; }
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

bool SIConvertWaveSize::run(Function &F) {
  LLVM_DEBUG(dbgs() << "Running SIConvertWaveSize on function: " << F.getName() << "\n");
  LLVM_DEBUG(printFunctionAttributes(F));

  const GCNSubtarget &ST = TM->getSubtarget<GCNSubtarget>(F);
  if (ST.getGeneration() < AMDGPUSubtarget::GFX11)
    return false;

  // Check if the function is a kernel.
  if (F.getCallingConv() != CallingConv::AMDGPU_KERNEL)
    return false;

  // Check if the kernel is wave32
  if (F.hasFnAttribute("target-features")) {
    if (!F.getFnAttribute("target-features")
            .getValueAsString().contains("wavefrontsize32")) {
      LLVM_DEBUG(dbgs() << "SIConvertWaveSize: Kernel is not wave32.\n");
      return false;
    }
  }

  // Check if the function is a device enqueue call.
  if (F.hasFnAttribute("amdgpu-device-enqueue")) {
    LLVM_DEBUG(dbgs() << "SIConvertWaveSize: Device enqueue call detected.\n");
    return false;
  }

  // Check if a trip count is a compile time constant for all loops in the
  // kernel
  for (Loop *L : *LI) {
    const SCEV *TripCountSCEV = SE->getBackedgeTakenCount(L);
    if (!isa<SCEVConstant>(TripCountSCEV)) {
      LLVM_DEBUG(
          dbgs() << "SIConvertWaveSize: Trip count is not a compile time "
                    "constant.\n");
      return false;
    }
  }

  for (const auto &BB : F) {
    InstructionCost BlockCost = 0;
    for (const auto &I : BB) {
      if (const CallBase *CB = dyn_cast<CallBase>(&I)) {
        // FIXME: Any calls are not allowed. Only non-converged intrinsic clls
        // and amdgsn_s_barrier are exempt. InlineAsm and Atomics are checkedd
        // separately for debug purposes. This will be changed in the final
        // version.
        if (CB->isInlineAsm()) {
          // Inline assembly is not allowed.
          LLVM_DEBUG(dbgs()
                     << "SIConvertWaveSize: Inline assembly detected.\n");
          return false;
        }
        if (CB->isAtomic()) {
          // Atomic operations are not allowed.
          LLVM_DEBUG(dbgs()
                     << "SIConvertWaveSize: Atomic operation detected.\n");
          return false;
        }
        if (Function *Callee = CB->getCalledFunction()) {
          // assuming readlane/readfirstlane or any cross-lane/DPP
          // operations have "let isConvergent = 1" in IntrinsicsAMDGPU.td
          if (Callee->isIntrinsic()) {
              if (Callee->hasFnAttribute(Attribute::Convergent)) {
                if (Callee->getIntrinsicID() != Intrinsic::amdgcn_s_barrier) {
                  // TODO: what else should go in a "white list" ?
                  // Intrinsic::amdgcn_s_barrier_wavefront ?
                  // Intrinsic::amdgcn_s_barrier_signal ?
                  LLVM_DEBUG(dbgs()
                             << "SIConvertWaveSize: Convergent intrinsic "
                             << Callee->getName() << " detected.\n");
                  return false;
                }
              }

            if (Callee->getIntrinsicID() == Intrinsic::read_register) {
              if (const auto *MDVal =
                      dyn_cast<MetadataAsValue>(CB->getArgOperand(0))) {
                Metadata *MD = MDVal->getMetadata();
                if (auto *MDNodeVal = dyn_cast<MDNode>(MD)) {
                  if (MDNodeVal->getNumOperands() >= 1) {
                    if (auto *MDStr =
                            dyn_cast<MDString>(MDNodeVal->getOperand(0))) {
                      if (MDStr->getString().starts_with("exec") ||
                          MDStr->getString().starts_with("vcc")) {
                        LLVM_DEBUG(dbgs() << "SIConvertWaveSize: read_register("
                                          << MDStr->getString()
                                          << ") intrinsic detected.\n");
                        return false;
                      }
                    }
                  }
                }
              }
            }

            // Save callee as a candidate for attribute change
            Callees.push_back(Callee);
          }
        } else {
          // General calls are not allowed.
          LLVM_DEBUG(dbgs() << "SIConvertWaveSize: function call detected.\n");
          return false;
        }
      }
      // No  LDS access is allowed
      if (auto LI = dyn_cast<LoadInst>(&I)) {
        if (LI->getPointerAddressSpace() == AMDGPUAS::LOCAL_ADDRESS) {
          LLVM_DEBUG(dbgs() << "SIConvertWaveSize: LDS access detected.\n");
          return false;
        }
      }
      if (auto SI = dyn_cast<StoreInst>(&I)) {
        if (SI->getPointerAddressSpace() == AMDGPUAS::LOCAL_ADDRESS) {
          LLVM_DEBUG(dbgs() << "SIConvertWaveSize: LDS access detected.\n");
          return false;
        }
      }
      // TODO: All atomics are not allowed?
      // if (auto AI = dyn_cast<AtomicRMWInst>(&I)) {
      //   if (AI->getPointerAddressSpace() == AMDGPUAS::LOCAL_ADDRESS) {
      //     LLVM_DEBUG(dbgs() << "SIConvertWaveSize: LDS access
      //     detected.\n"); return false;
      //   }
      // }

      // TODO: Dynamic VGPRS and GFX11+ special operations ???
      BlockCost +=
          TTI->getInstructionCost(&I, TargetTransformInfo::TCK_RecipThroughput);
    }
    if (auto L = LI->getLoopFor(&BB)) {
      const SCEV *TripCount = SE->getBackedgeTakenCount(L);
      if (auto *C = dyn_cast<SCEVConstant>(TripCount)) {
        uint64_t TC = C->getValue()->getZExtValue() + 1;
        size_t Depth = LI->getLoopDepth(&BB);
        BlockCost *= TC * Depth;
      } else
        llvm_unreachable("SIConvertWaveSize: only loops with compile time "
                         "constant trip count could reach here!\n");
    }
    TotalCost += BlockCost;
    if (TotalCost.isValid()) {
      if (TotalCost.getValue().value() >= MaxLatency) {
        LLVM_DEBUG(
            dbgs() << "SIConvertWaveSize: Total latency of the kernel ["
                   << TotalCost.getValue().value()
                   << "] exceeds the limit of 2000 cycles - not profitable!\n");
        return false;
      }
    } else
      llvm_unreachable(
          "SIConvertWaveSize: Cost model error - invalid state!\n");
  }

  // Additional checks can be added here...

  // If all checks pass, convert wave size from wave32 to wave64.
  // Conversion logic goes here...
  bool Changed = changeWaveSizeAttr(&F);
  if (Changed)
    // Now take care of the intrinsic calls
    for (auto C : Callees) {
      // TODO: if we could not change Attr for one of the callee
      // we need to rollback all the changes!
      changeWaveSizeAttr(C);
    }

  return Changed;
  }

bool SIConvertWaveSize::changeWaveSizeAttr(Function *F) {
  auto Attr = F->getFnAttribute("target-features");
  if (Attr.isValid()) {
    StringRef AttrStr = Attr.getValueAsString();
    size_t Pos = AttrStr.find("+wavefrontsize32");
    if (Pos != StringRef::npos) {
      // Remove the "+wavefrontsize32" attribute.
      std::string NewBegin = AttrStr.substr(0, Pos).str().append("+wavefrontsize64");
      std::string End = AttrStr.substr(Pos + strlen("+wavefrontsize32")).str();
      std::string NewAttrStr = NewBegin + End;
      // Add the "+wavefrontsize64" attribute.
      F->removeFnAttr("target-features");
      F->addFnAttr("target-features", NewAttrStr);
      LLVM_DEBUG(dbgs() << "SIConvertWaveSize: Converted wave size for "
                        << F->getName()
                        << " from wave32 "
                           "to wave64.\n");
      return true;
    }
  }
  return false;
}

INITIALIZE_PASS_BEGIN(SIConvertWaveSizeLegacy, DEBUG_TYPE, "SI convert wave size",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(SIConvertWaveSizeLegacy, DEBUG_TYPE, "SI convert wave size",
                    false, false)

char SIConvertWaveSizeLegacy::ID = 0;

char &llvm::SIConvertWaveSizeLegacyID = SIConvertWaveSizeLegacy::ID;

FunctionPass *llvm::createSIConvertWaveSizeLegacyPass(const TargetMachine *TM) {
  return new SIConvertWaveSizeLegacy(TM);
}

PreservedAnalyses SIConvertWaveSizePass::run(
    Function &F, FunctionAnalysisManager &FAM) {
      auto &LI = FAM.getResult<LoopAnalysis>(F);
  auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
  auto &TTI = FAM.getResult<TargetIRAnalysis>(F);

  SIConvertWaveSize Impl(TM, &LI, &SE, &TTI);
  bool Changed = Impl.run(F);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
