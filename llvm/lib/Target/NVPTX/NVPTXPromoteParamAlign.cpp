//===-- NVPTXPromoteParamAlign.cpp - Promote .param alignment ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Increase the .param-space alignment of NVPTX arguments and return values so
// their loads and stores can be vectorized. On every defined function:
//
//   1. Give each byval param an explicit ABI `align` for its pointee type
//      (capped at the PTX max). byval already implies this, but the alignment
//      is otherwise invisible to IR alignment analyses.
//   2. For a local function whose every use is a type-compatible direct call,
//      we control all call sites and raise aggregate/byval param and return
//      alignment to at least 16 (for 128-bit vectorization). This is recorded
//      as `stackalign` and mirrored onto the calls.
//   3. Propagate the result onto byval loads at a known offset, since `align`
//      and `stackalign` aren't both picked up by IR alignment analyses.
//
// (2) runs before (1) so byval `align` still matches between caller and callee
// while stackalign is mirrored onto the calls.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "NVPTXUtilities.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include <optional>
#include <queue>

#define DEBUG_TYPE "nvptx-promote-param-align"

using namespace llvm;

namespace {
class NVPTXPromoteParamAlignLegacyPass : public ModulePass {
  bool runOnModule(Module &M) override;

public:
  static char ID;
  NVPTXPromoteParamAlignLegacyPass() : ModulePass(ID) {}
  StringRef getPassName() const override {
    return "Promote alignment of parameters and return values (NVPTX)";
  }
};
} // namespace

char NVPTXPromoteParamAlignLegacyPass::ID = 0;

INITIALIZE_PASS(NVPTXPromoteParamAlignLegacyPass, "nvptx-promote-param-align",
                "Promote alignment of parameters and return values (NVPTX)",
                false, false)

// Return true if the attributes that determine an NVPTX .param slot's layout
// match.
static bool layoutAttrsMatch(AttributeSet CalleeAttrs, AttributeSet CallAttrs) {
  if (CalleeAttrs.getByValType() != CallAttrs.getByValType() ||
      CalleeAttrs.getStackAlignment() != CallAttrs.getStackAlignment())
    return false;

  // `align` only affects the layout for byval parameters.
  return !CalleeAttrs.getByValType() ||
         CalleeAttrs.getAlignment() == CallAttrs.getAlignment();
}

static bool callSiteMatchesCalleeABI(const CallBase &CB, const Function &F) {
  const AttributeList CalleeAttrs = F.getAttributes();
  const AttributeList CallAttrs = CB.getAttributes();

  if (!layoutAttrsMatch(CalleeAttrs.getRetAttrs(), CallAttrs.getRetAttrs()))
    return false;

  return all_of(seq(F.arg_size()), [&](size_t I) {
    return layoutAttrsMatch(CalleeAttrs.getParamAttrs(I),
                            CallAttrs.getParamAttrs(I));
  });
}

// Promotable if the function is local and every use is an ABI-compatible direct
// call, so we control every call site and can raise alignment on both sides.
static bool canPromoteParamAlign(Function &F) {
  if (F.isDeclaration() || !F.hasLocalLinkage())
    return false;

  if (F.hasAddressTaken(/*Users=*/nullptr, /*IgnoreCallbackUses=*/false,
                        /*IgnoreAssumeLikeCalls=*/true,
                        /*IgnoreLLVMUsed=*/true))
    return false;

  return all_of(F.users(), [&](const User *U) {
    const auto *CB = dyn_cast<CallBase>(U);
    if (!CB || CB->getCalledOperand() != &F)
      return true;
    return CB->getFunctionType() == F.getFunctionType() &&
           callSiteMatchesCalleeABI(*CB, F);
  });
}

// Raise the alignment of every load reachable from a byval pointer at a known
// constant offset. Must stay in sync with the param load/store in LowerCall.
static bool propagateAlignmentToLoads(Value *Val, Align NewAlign,
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

  bool Changed = false;
  for (Load &CurLoad : Loads) {
    Align NewLoadAlign = commonAlignment(NewAlign, CurLoad.Offset);
    if (NewLoadAlign > CurLoad.Inst->getAlign()) {
      CurLoad.Inst->setAlignment(NewLoadAlign);
      Changed = true;
    }
  }
  return Changed;
}

// Bump an alignment up to at least 16 (for 128-bit vectorization), or nullopt
// if it's already large enough.
static MaybeAlign getPromotedParamAlign(Align CurrentAlign) {
  const Align PromotedAlign = std::max(CurrentAlign, Align(16));
  if (PromotedAlign > CurrentAlign)
    return PromotedAlign;
  return std::nullopt;
}

static bool promoteParamAlign(Function &F) {
  if (!canPromoteParamAlign(F))
    return false;

  LLVMContext &Ctx = F.getContext();
  const DataLayout &DL = F.getDataLayout();

  // Promoted (arg index, new alignment) pairs, to mirror onto call sites.
  SmallVector<std::pair<unsigned, Align>, 8> PromotedParams;
  MaybeAlign PromotedRet;

  // Promote aggregate and byval parameters.
  for (Argument &Arg : F.args()) {
    const bool IsByVal = Arg.hasByValAttr();
    Type *ArgTy = IsByVal ? Arg.getParamByValType() : Arg.getType();
    if (ArgTy->isEmptyTy() || (!IsByVal && !shouldPassAsArray(ArgTy)))
      continue;

    // An explicit stackalign already wins at emission time, nothing to promote.
    if (Arg.getParamStackAlign())
      continue;
    const unsigned ArgNo = Arg.getArgNo();

    // `align` only applies to byval (pointer) args, not by-value aggregates.
    Align CurrentAlign = getPTXParamTypeAlign(ArgTy, DL);
    if (IsByVal)
      CurrentAlign = std::max(CurrentAlign, Arg.getParamAlign().valueOrOne());
    const MaybeAlign PromotedAlign = getPromotedParamAlign(CurrentAlign);
    if (!PromotedAlign)
      continue;

    LLVM_DEBUG(dbgs() << "Promoting alignment of " << Arg << " to "
                      << PromotedAlign->value() << '\n');
    Arg.addAttr(Attribute::getWithStackAlignment(Ctx, *PromotedAlign));
    PromotedParams.emplace_back(ArgNo, *PromotedAlign);
  }

  // Promote an aggregate return value.
  Type *RetTy = F.getReturnType();
  if (shouldPassAsArray(RetTy) && !RetTy->isEmptyTy() &&
      !F.getAttributes().getRetStackAlignment()) {
    const MaybeAlign PromotedAlign =
        getPromotedParamAlign(getPTXParamTypeAlign(RetTy, DL));
    if (PromotedAlign) {
      F.addRetAttr(Attribute::getWithStackAlignment(Ctx, *PromotedAlign));
      PromotedRet = *PromotedAlign;
    }
  }

  if (PromotedParams.empty() && !PromotedRet)
    return false;

  // Mirror the promotion onto every direct call site so both sides agree on the
  // .param layout. canPromoteParamAlign already verified they're
  // ABI-compatible.
  for (User *U : F.users()) {
    auto *CB = dyn_cast<CallBase>(U);
    if (!CB || CB->getCalledOperand() != &F)
      continue;

    for (const auto &[ArgNo, PromotedAlign] : PromotedParams)
      CB->addParamAttr(ArgNo,
                       Attribute::getWithStackAlignment(Ctx, PromotedAlign));
    if (PromotedRet)
      CB->addRetAttr(Attribute::getWithStackAlignment(Ctx, *PromotedRet));

    assert(callSiteMatchesCalleeABI(*CB, F) &&
           "mirroring must preserve call-site/callee ABI compatibility");
  }

  return true;
}

// Spell out each byval parameter's ABI alignment as an explicit `align` (step 1
// above). Runs after promoteParamAlign, which needs byval `align` to still
// match between callers and callees.
static bool setByValParamABIAlign(Function &F) {
  if (F.isDeclaration())
    return false;

  LLVMContext &Ctx = F.getContext();
  const DataLayout &DL = F.getDataLayout();
  bool Changed = false;
  for (Argument &Arg : F.args()) {
    if (!Arg.hasByValAttr())
      continue;
    Type *ETy = Arg.getParamByValType();
    if (ETy->isEmptyTy())
      continue;
    const Align ABIAlign = getPTXParamTypeAlign(ETy, DL);
    if (Arg.getParamAlign().valueOrOne() >= ABIAlign)
      continue;
    Arg.removeAttr(Attribute::Alignment);
    Arg.addAttr(Attribute::getWithAlignment(Ctx, ABIAlign));
    Changed = true;
  }
  return Changed;
}

// Propagate each byval parameter's .param alignment onto its constant-offset
// loads (step 3 above). Runs after promoteParamAlign so the promoted
// `stackalign` is included, and on every function so kernels and external
// functions benefit too.
static bool propagateByValParamLoadAlign(Function &F) {
  if (F.isDeclaration())
    return false;

  const DataLayout &DL = F.getDataLayout();
  bool Changed = false;
  for (Argument &Arg : F.args()) {
    if (!Arg.hasByValAttr())
      continue;
    Type *ETy = Arg.getParamByValType();
    if (ETy->isEmptyTy())
      continue;
    const unsigned ParamIdx = Arg.getArgNo() + AttributeList::FirstArgIndex;
    const Align ParamAlign = getDeviceByValParamAlign(&F, ETy, ParamIdx, DL);
    Changed |= propagateAlignmentToLoads(&Arg, ParamAlign, DL);
  }
  return Changed;
}

static bool promoteParamAlignModule(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    // Order matters (see the file header): promote, normalize `align`, then
    // propagate to loads.
    Changed |= promoteParamAlign(F);
    Changed |= setByValParamABIAlign(F);
    Changed |= propagateByValParamLoadAlign(F);
  }
  return Changed;
}

bool NVPTXPromoteParamAlignLegacyPass::runOnModule(Module &M) {
  return promoteParamAlignModule(M);
}

ModulePass *llvm::createNVPTXPromoteParamAlignPass() {
  return new NVPTXPromoteParamAlignLegacyPass();
}

PreservedAnalyses NVPTXPromoteParamAlignPass::run(Module &M,
                                                  ModuleAnalysisManager &AM) {
  return promoteParamAlignModule(M) ? PreservedAnalyses::none()
                                    : PreservedAnalyses::all();
}
