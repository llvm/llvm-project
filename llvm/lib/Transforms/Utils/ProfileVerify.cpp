//===- ProfileVerify.cpp - Verify profile info for testing ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/ProfileVerify.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
static cl::opt<int64_t>
    DefaultFunctionEntryCount("profcheck-default-function-entry-count",
                              cl::init(1000));
static cl::opt<bool>
    AnnotateSelect("profcheck-annotate-select", cl::init(true),
                   cl::desc("Also inject (if missing) and verify MD_prof for "
                            "`select` instructions"));
static cl::opt<bool>
    WeightsForTest("profcheck-weights-for-test", cl::init(false),
                   cl::desc("Generate weights with small values for tests."));

static cl::opt<uint32_t> SelectTrueWeight(
    "profcheck-default-select-true-weight", cl::init(2U),
    cl::desc("When annotating `select` instructions, this value will be used "
             "for the first ('true') case."));
static cl::opt<uint32_t> SelectFalseWeight(
    "profcheck-default-select-false-weight", cl::init(3U),
    cl::desc("When annotating `select` instructions, this value will be used "
             "for the second ('false') case."));
namespace {
class ProfileInjector {
  Function &F;
  FunctionAnalysisManager &FAM;

public:
  static const Instruction *
  getTerminatorBenefitingFromMDProf(const BasicBlock &BB) {
    if (succ_size(&BB) < 2)
      return nullptr;
    auto *Term = BB.getTerminator();
    return (isa<BranchInst>(Term) || isa<SwitchInst>(Term) ||
            isa<IndirectBrInst>(Term) || isa<CallBrInst>(Term))
               ? Term
               : nullptr;
  }

  static Instruction *getTerminatorBenefitingFromMDProf(BasicBlock &BB) {
    return const_cast<Instruction *>(
        getTerminatorBenefitingFromMDProf(const_cast<const BasicBlock &>(BB)));
  }

  ProfileInjector(Function &F, FunctionAnalysisManager &FAM) : F(F), FAM(FAM) {}
  bool inject();
};

bool isAsmOnly(const Function &F) {
  if (!F.hasFnAttribute(Attribute::AttrKind::Naked))
    return false;
  for (const auto &BB : F)
    for (const auto &I : drop_end(BB.instructionsWithoutDebug())) {
      const auto *CB = dyn_cast<CallBase>(&I);
      if (!CB || !CB->isInlineAsm())
        return false;
    }
  return true;
}

void emitProfileError(StringRef Msg, Function &F) {
  F.getContext().emitError("Profile verification failed for function '" +
                           F.getName() + "': " + Msg);
}

} // namespace

// FIXME: currently this injects only for terminators. Select isn't yet
// supported.
bool ProfileInjector::inject() {
  // skip purely asm functions
  if (isAsmOnly(F))
    return false;
  // Get whatever branch probability info can be derived from the given IR -
  // whether it has or not metadata. The main intention for this pass is to
  // ensure that other passes don't drop or "forget" to update MD_prof. We do
  // this as a mode in which lit tests would run. We want to avoid changing the
  // behavior of those tests. A pass may use BPI (or BFI, which is computed from
  // BPI). If no metadata is present, BPI is guesstimated by
  // BranchProbabilityAnalysis. The injector (this pass) only persists whatever
  // information the analysis provides, in other words, the pass being tested
  // will get the same BPI it does if the injector wasn't running.
  auto &BPI = FAM.getResult<BranchProbabilityAnalysis>(F);

  // Inject a function count if there's none. It's reasonable for a pass to
  // want to clear the MD_prof of a function with zero entry count. If the
  // original profile (iFDO or AFDO) is empty for a function, it's simpler to
  // require assigning it the 0-entry count explicitly than to mark every branch
  // as cold (we do want some explicit information in the spirit of what this
  // verifier wants to achieve - make dropping / corrupting MD_prof
  // unit-testable)
  if (!F.getEntryCount(/*AllowSynthetic=*/true))
    F.setEntryCount(DefaultFunctionEntryCount);
  // If there is an entry count that's 0, then don't bother injecting. We won't
  // verify these either.
  if (F.getEntryCount(/*AllowSynthetic=*/true)->getCount() == 0)
    return false;
  bool Changed = false;
  // Cycle through the weights list. If we didn't, tests with more than (say)
  // one conditional branch would have the same !prof metadata on all of them,
  // and numerically that may make for a poor unit test.
  uint32_t WeightsForTestOffset = 0;
  for (auto &BB : F) {
    if (AnnotateSelect) {
      for (auto &I : BB) {
        if (auto *SI = dyn_cast<SelectInst>(&I)) {
          if (SI->getCondition()->getType()->isVectorTy())
            continue;
          if (I.getMetadata(LLVMContext::MD_prof))
            continue;
          setBranchWeights(I, {SelectTrueWeight, SelectFalseWeight},
                           /*IsExpected=*/false);
        }
      }
    }
    auto *Term = getTerminatorBenefitingFromMDProf(BB);
    if (!Term || Term->getMetadata(LLVMContext::MD_prof))
      continue;
    SmallVector<BranchProbability> Probs;

    SmallVector<uint32_t> Weights;
    Weights.reserve(Term->getNumSuccessors());
    if (WeightsForTest) {
      static const std::array Primes{3,  5,  7,  11, 13, 17, 19, 23, 29, 31,
                                     37, 41, 43, 47, 53, 59, 61, 67, 71};
      for (uint32_t I = 0, E = Term->getNumSuccessors(); I < E; ++I)
        Weights.emplace_back(
            Primes[(WeightsForTestOffset + I) % Primes.size()]);
      ++WeightsForTestOffset;
    } else {
      Probs.reserve(Term->getNumSuccessors());
      for (auto I = 0U, E = Term->getNumSuccessors(); I < E; ++I)
        Probs.emplace_back(BPI.getEdgeProbability(&BB, Term->getSuccessor(I)));

      assert(llvm::find_if(Probs,
                           [](const BranchProbability &P) {
                             return P.isUnknown();
                           }) == Probs.end() &&
             "All branch probabilities should be valid");
      const auto *FirstZeroDenominator =
          find_if(Probs, [](const BranchProbability &P) {
            return P.getDenominator() == 0;
          });
      (void)FirstZeroDenominator;
      assert(FirstZeroDenominator == Probs.end());
      const auto *FirstNonZeroNumerator = find_if(
          Probs, [](const BranchProbability &P) { return !P.isZero(); });
      assert(FirstNonZeroNumerator != Probs.end());
      DynamicAPInt LCM(Probs[0].getDenominator());
      DynamicAPInt GCD(FirstNonZeroNumerator->getNumerator());
      for (const auto &Prob : drop_begin(Probs)) {
        if (!Prob.getNumerator())
          continue;
        LCM = llvm::lcm(LCM, DynamicAPInt(Prob.getDenominator()));
        GCD = llvm::gcd(GCD, DynamicAPInt(Prob.getNumerator()));
      }
      for (const auto &Prob : Probs) {
        DynamicAPInt W =
            (Prob.getNumerator() * LCM / GCD) / Prob.getDenominator();
        Weights.emplace_back(static_cast<uint32_t>((int64_t)W));
      }
    }
    setBranchWeights(*Term, Weights, /*IsExpected=*/false);
    Changed = true;
  }
  return Changed;
}

PreservedAnalyses ProfileInjectorPass::run(Function &F,
                                           FunctionAnalysisManager &FAM) {
  ProfileInjector PI(F, FAM);
  if (!PI.inject())
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

PreservedAnalyses ProfileVerifierPass::run(Module &M,
                                           ModuleAnalysisManager &MAM) {
  auto PopulateIgnoreList = [&](StringRef GVName) {
    if (const auto *CT = M.getGlobalVariable(GVName))
      if (const auto *CA =
              dyn_cast_if_present<ConstantArray>(CT->getInitializer()))
        for (const auto &Elt : CA->operands())
          if (const auto *CS = dyn_cast<ConstantStruct>(Elt))
            if (CS->getNumOperands() >= 2 && CS->getOperand(1))
              if (const auto *F = dyn_cast<Function>(
                      CS->getOperand(1)->stripPointerCasts()))
                IgnoreList.insert(F);
  };
  PopulateIgnoreList("llvm.global_ctors");
  PopulateIgnoreList("llvm.global_dtors");

  // expose the function-level run as public through a wrapper, so we can use
  // pass manager mechanisms dealing with declarations and with composing the
  // returned PreservedAnalyses values.
  struct Wrapper : PassInfoMixin<Wrapper> {
    ProfileVerifierPass &PVP;
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
      return PVP.run(F, FAM);
    }
    explicit Wrapper(ProfileVerifierPass &PVP) : PVP(PVP) {}
  };

  return createModuleToFunctionPassAdaptor(Wrapper(*this)).run(M, MAM);
}

PreservedAnalyses ProfileVerifierPass::run(Function &F,
                                           FunctionAnalysisManager &FAM) {
  // skip purely asm functions
  if (isAsmOnly(F))
    return PreservedAnalyses::all();
  if (IgnoreList.contains(&F))
    return PreservedAnalyses::all();

  const auto EntryCount = F.getEntryCount(/*AllowSynthetic=*/true);
  if (!EntryCount) {
    auto *MD = F.getMetadata(LLVMContext::MD_prof);
    if (!MD || !isExplicitlyUnknownProfileMetadata(*MD)) {
      emitProfileError("function entry count missing (set to 0 if cold)", F);
      return PreservedAnalyses::all();
    }
  } else if (EntryCount->getCount() == 0) {
    return PreservedAnalyses::all();
  }
  for (const auto &BB : F) {
    if (AnnotateSelect) {
      for (const auto &I : BB)
        if (auto *SI = dyn_cast<SelectInst>(&I)) {
          if (SI->getCondition()->getType()->isVectorTy())
            continue;
          if (I.getMetadata(LLVMContext::MD_prof))
            continue;
          emitProfileError("select annotation missing", F);
        }
    }
    if (const auto *Term =
            ProfileInjector::getTerminatorBenefitingFromMDProf(BB))
      if (!Term->getMetadata(LLVMContext::MD_prof))
        emitProfileError("branch annotation missing", F);
  }
  return PreservedAnalyses::all();
}
