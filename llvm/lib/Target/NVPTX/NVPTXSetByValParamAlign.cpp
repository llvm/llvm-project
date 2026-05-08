//===-- NVPTXSetByValParamAlign.cpp - Set byval param alignment -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Set explicit alignment on byval parameter attributes in the NVPTX backend.
// Without this, the alignment is left unspecified and IR-level analyses (e.g.,
// computeKnownBits via Value::getPointerAlignment) conservatively assume
// Align(1), since the actual alignment is a target-specific codegen detail not
// visible at the IR level.
//
// The alignment is chosen as follows:
//   - Externally-visible functions: ABI type alignment (capped at 128).
//   - Internal/private functions: max(16, ABI align) to enable 128-bit
//     vectorized param loads. The compiler can _increase_ alignment beyond ABI
//     in this case because it has control over all of the call sites and byval
//     parameters are copies allocated by the caller in .param space.
//
// After updating the attribute, the pass propagates the improved alignment to
// all loads from the byval pointer that use a known constant offset.
//
// TODO: Consider removing the load propagation in favor of infer-alignment,
// which should be able to pick up the improved alignment from the attribute.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "NVPTXUtilities.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include <queue>

#define DEBUG_TYPE "nvptx-set-byval-param-align"

using namespace llvm;

namespace {
class NVPTXSetByValParamAlignLegacyPass : public FunctionPass {
  bool runOnFunction(Function &F) override;

public:
  static char ID;
  NVPTXSetByValParamAlignLegacyPass() : FunctionPass(ID) {}
  StringRef getPassName() const override {
    return "Set alignment of byval parameters (NVPTX)";
  }
};
} // namespace

char NVPTXSetByValParamAlignLegacyPass::ID = 0;

INITIALIZE_PASS(NVPTXSetByValParamAlignLegacyPass,
                "nvptx-set-byval-param-align",
                "Set alignment of byval parameters (NVPTX)", false, false)

static Align setByValParamAlign(Argument *Arg) {
  Function *F = Arg->getParent();
  Type *ByValType = Arg->getParamByValType();
  const DataLayout &DL = F->getDataLayout();

  const Align OptimizedAlign = getFunctionParamOptimizedAlign(F, ByValType, DL);
  const Align CurrentAlign = Arg->getParamAlign().valueOrOne();

  if (CurrentAlign >= OptimizedAlign)
    return CurrentAlign;

  LLVM_DEBUG(dbgs() << "Try to use alignment " << OptimizedAlign.value()
                    << " instead of " << CurrentAlign.value() << " for " << *Arg
                    << '\n');

  Arg->removeAttr(Attribute::Alignment);
  Arg->addAttr(Attribute::getWithAlignment(F->getContext(), OptimizedAlign));

  return OptimizedAlign;
}

// Adjust alignment of arguments passed byval in .param address space. We can
// increase alignment of such arguments in a way that ensures that we can
// effectively vectorize their loads. We should also traverse all loads from
// byval pointer and adjust their alignment, if those were using known offset.
// Such alignment changes must be conformed with parameter store and load in
// NVPTXTargetLowering::LowerCall.
static void propagateAlignmentToLoads(Value *Val, Align NewAlign,
                                      const DataLayout &DL) {
  struct Load {
    LoadInst *Inst;
    uint64_t Offset;
  };

  struct LoadContext {
    Value *InitialVal;
    uint64_t Offset;
  };

  SmallVector<Load> Loads;
  std::queue<LoadContext> Worklist;
  Worklist.push({Val, 0});

  while (!Worklist.empty()) {
    LoadContext Ctx = Worklist.front();
    Worklist.pop();

    for (User *CurUser : Ctx.InitialVal->users()) {
      if (auto *I = dyn_cast<LoadInst>(CurUser))
        Loads.push_back({I, Ctx.Offset});
      else if (isa<BitCastInst>(CurUser) || isa<AddrSpaceCastInst>(CurUser))
        Worklist.push({cast<Instruction>(CurUser), Ctx.Offset});
      else if (auto *I = dyn_cast<GetElementPtrInst>(CurUser)) {
        APInt OffsetAccumulated =
            APInt::getZero(DL.getIndexTypeSizeInBits(I->getType()));

        if (!I->accumulateConstantOffset(DL, OffsetAccumulated))
          continue;

        uint64_t OffsetLimit = -1;
        uint64_t Offset = OffsetAccumulated.getLimitedValue(OffsetLimit);
        assert(Offset != OffsetLimit && "Expect Offset less than UINT64_MAX");

        Worklist.push({I, Ctx.Offset + Offset});
      }
    }
  }

  for (Load &CurLoad : Loads) {
    Align NewLoadAlign = commonAlignment(NewAlign, CurLoad.Offset);
    Align CurLoadAlign = CurLoad.Inst->getAlign();
    CurLoad.Inst->setAlignment(std::max(NewLoadAlign, CurLoadAlign));
  }
}

static bool setByValParamAlignment(Function &F) {
  const DataLayout &DL = F.getDataLayout();
  bool Changed = false;
  for (Argument &Arg : F.args()) {
    if (!Arg.hasByValAttr())
      continue;
    const Align NewArgAlign = setByValParamAlign(&Arg);
    propagateAlignmentToLoads(&Arg, NewArgAlign, DL);
    Changed = true;
  }
  return Changed;
}

bool NVPTXSetByValParamAlignLegacyPass::runOnFunction(Function &F) {
  return setByValParamAlignment(F);
}

FunctionPass *llvm::createNVPTXSetByValParamAlignPass() {
  return new NVPTXSetByValParamAlignLegacyPass();
}

PreservedAnalyses
NVPTXSetByValParamAlignPass::run(Function &F, FunctionAnalysisManager &AM) {
  return setByValParamAlignment(F) ? PreservedAnalyses::none()
                                   : PreservedAnalyses::all();
}
