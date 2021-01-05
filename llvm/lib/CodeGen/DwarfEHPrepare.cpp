//===- DwarfEHPrepare - Prepare exception handling for code generation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass mulches exception handling code into a form adapted to code
// generation. Required if using dwarf exception handling.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/RuntimeLibcalls.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cstddef>

using namespace llvm;

#define DEBUG_TYPE "dwarfehprepare"

STATISTIC(NumResumesLowered, "Number of resume calls lowered");

namespace {

class DwarfEHPrepare {
  CodeGenOpt::Level OptLevel;

  // RewindFunction - _Unwind_Resume or the target equivalent.
  FunctionCallee &RewindFunction;

  Function &F;
  const TargetLowering &TLI;
  DomTreeUpdater *DTU;
  const TargetTransformInfo *TTI;

  /// Return the exception object from the value passed into
  /// the 'resume' instruction (typically an aggregate). Clean up any dead
  /// instructions, including the 'resume' instruction.
  Value *GetExceptionObject(ResumeInst *RI);

  /// Replace resumes that are not reachable from a cleanup landing pad with
  /// unreachable and then simplify those blocks.
  size_t
  pruneUnreachableResumes(SmallVectorImpl<ResumeInst *> &Resumes,
                          SmallVectorImpl<LandingPadInst *> &CleanupLPads);

  /// Convert the ResumeInsts that are still present
  /// into calls to the appropriate _Unwind_Resume function.
  bool InsertUnwindResumeCalls();

public:
  DwarfEHPrepare(CodeGenOpt::Level OptLevel_, FunctionCallee &RewindFunction_,
                 Function &F_, const TargetLowering &TLI_, DomTreeUpdater *DTU_,
                 const TargetTransformInfo *TTI_)
      : OptLevel(OptLevel_), RewindFunction(RewindFunction_), F(F_), TLI(TLI_),
        DTU(DTU_), TTI(TTI_) {}

  bool run();
};

} // namespace

Value *DwarfEHPrepare::GetExceptionObject(ResumeInst *RI) {
  Value *V = RI->getOperand(0);
  Value *ExnObj = nullptr;
  InsertValueInst *SelIVI = dyn_cast<InsertValueInst>(V);
  LoadInst *SelLoad = nullptr;
  InsertValueInst *ExcIVI = nullptr;
  bool EraseIVIs = false;

  if (SelIVI) {
    if (SelIVI->getNumIndices() == 1 && *SelIVI->idx_begin() == 1) {
      ExcIVI = dyn_cast<InsertValueInst>(SelIVI->getOperand(0));
      if (ExcIVI && isa<UndefValue>(ExcIVI->getOperand(0)) &&
          ExcIVI->getNumIndices() == 1 && *ExcIVI->idx_begin() == 0) {
        ExnObj = ExcIVI->getOperand(1);
        SelLoad = dyn_cast<LoadInst>(SelIVI->getOperand(1));
        EraseIVIs = true;
      }
    }
  }

  if (!ExnObj)
    ExnObj = ExtractValueInst::Create(RI->getOperand(0), 0, "exn.obj", RI);

  RI->eraseFromParent();

  if (EraseIVIs) {
    if (SelIVI->use_empty())
      SelIVI->eraseFromParent();
    if (ExcIVI->use_empty())
      ExcIVI->eraseFromParent();
    if (SelLoad && SelLoad->use_empty())
      SelLoad->eraseFromParent();
  }

  return ExnObj;
}

size_t DwarfEHPrepare::pruneUnreachableResumes(
    SmallVectorImpl<ResumeInst *> &Resumes,
    SmallVectorImpl<LandingPadInst *> &CleanupLPads) {
  assert(DTU && "Should have DomTreeUpdater here.");

  BitVector ResumeReachable(Resumes.size());
  size_t ResumeIndex = 0;
  for (auto *RI : Resumes) {
    for (auto *LP : CleanupLPads) {
      if (isPotentiallyReachable(LP, RI, nullptr, &DTU->getDomTree())) {
        ResumeReachable.set(ResumeIndex);
        break;
      }
    }
    ++ResumeIndex;
  }

  // If everything is reachable, there is no change.
  if (ResumeReachable.all())
    return Resumes.size();

  LLVMContext &Ctx = F.getContext();

  // Otherwise, insert unreachable instructions and call simplifycfg.
  size_t ResumesLeft = 0;
  for (size_t I = 0, E = Resumes.size(); I < E; ++I) {
    ResumeInst *RI = Resumes[I];
    if (ResumeReachable[I]) {
      Resumes[ResumesLeft++] = RI;
    } else {
      BasicBlock *BB = RI->getParent();
      new UnreachableInst(Ctx, RI);
      RI->eraseFromParent();
      simplifyCFG(BB, *TTI, RequireAndPreserveDomTree ? DTU : nullptr);
    }
  }
  Resumes.resize(ResumesLeft);
  return ResumesLeft;
}

bool DwarfEHPrepare::InsertUnwindResumeCalls() {
  SmallVector<ResumeInst *, 16> Resumes;
  SmallVector<LandingPadInst *, 16> CleanupLPads;
  for (BasicBlock &BB : F) {
    if (auto *RI = dyn_cast<ResumeInst>(BB.getTerminator()))
      Resumes.push_back(RI);
    if (auto *LP = BB.getLandingPadInst())
      if (LP->isCleanup())
        CleanupLPads.push_back(LP);
  }

  if (Resumes.empty())
    return false;

  // Check the personality, don't do anything if it's scope-based.
  EHPersonality Pers = classifyEHPersonality(F.getPersonalityFn());
  if (isScopedEHPersonality(Pers))
    return false;

  LLVMContext &Ctx = F.getContext();

  size_t ResumesLeft = Resumes.size();
  if (OptLevel != CodeGenOpt::None)
    ResumesLeft = pruneUnreachableResumes(Resumes, CleanupLPads);

  if (ResumesLeft == 0)
    return true; // We pruned them all.

  // Find the rewind function if we didn't already.
  if (!RewindFunction) {
    FunctionType *FTy =
        FunctionType::get(Type::getVoidTy(Ctx), Type::getInt8PtrTy(Ctx), false);
    const char *RewindName = TLI.getLibcallName(RTLIB::UNWIND_RESUME);
    RewindFunction = F.getParent()->getOrInsertFunction(RewindName, FTy);
  }

  // Create the basic block where the _Unwind_Resume call will live.
  if (ResumesLeft == 1) {
    // Instead of creating a new BB and PHI node, just append the call to
    // _Unwind_Resume to the end of the single resume block.
    ResumeInst *RI = Resumes.front();
    BasicBlock *UnwindBB = RI->getParent();
    Value *ExnObj = GetExceptionObject(RI);

    // Call the _Unwind_Resume function.
    CallInst *CI = CallInst::Create(RewindFunction, ExnObj, "", UnwindBB);
    CI->setCallingConv(TLI.getLibcallCallingConv(RTLIB::UNWIND_RESUME));

    // We never expect _Unwind_Resume to return.
    CI->setDoesNotReturn();
    new UnreachableInst(Ctx, UnwindBB);
    return true;
  }

  std::vector<DominatorTree::UpdateType> Updates;
  Updates.reserve(Resumes.size());

  BasicBlock *UnwindBB = BasicBlock::Create(Ctx, "unwind_resume", &F);
  PHINode *PN = PHINode::Create(Type::getInt8PtrTy(Ctx), ResumesLeft, "exn.obj",
                                UnwindBB);

  // Extract the exception object from the ResumeInst and add it to the PHI node
  // that feeds the _Unwind_Resume call.
  for (ResumeInst *RI : Resumes) {
    BasicBlock *Parent = RI->getParent();
    BranchInst::Create(UnwindBB, Parent);
    Updates.push_back({DominatorTree::Insert, Parent, UnwindBB});

    Value *ExnObj = GetExceptionObject(RI);
    PN->addIncoming(ExnObj, Parent);

    ++NumResumesLowered;
  }

  // Call the function.
  CallInst *CI = CallInst::Create(RewindFunction, PN, "", UnwindBB);
  CI->setCallingConv(TLI.getLibcallCallingConv(RTLIB::UNWIND_RESUME));

  // We never expect _Unwind_Resume to return.
  CI->setDoesNotReturn();
  new UnreachableInst(Ctx, UnwindBB);

  if (DTU && RequireAndPreserveDomTree)
    DTU->applyUpdates(Updates);

  return true;
}

bool DwarfEHPrepare::run() {
  assert(((OptLevel == CodeGenOpt::None || !RequireAndPreserveDomTree) ||
          (DTU &&
           DTU->getDomTree().verify(DominatorTree::VerificationLevel::Full))) &&
         "Original domtree is invalid?");

  bool Changed = InsertUnwindResumeCalls();

  assert(((OptLevel == CodeGenOpt::None || !RequireAndPreserveDomTree) ||
          (DTU &&
           DTU->getDomTree().verify(DominatorTree::VerificationLevel::Full))) &&
         "Original domtree is invalid?");

  return Changed;
}

static bool prepareDwarfEH(CodeGenOpt::Level OptLevel,
                           FunctionCallee &RewindFunction, Function &F,
                           const TargetLowering &TLI, DominatorTree *DT,
                           const TargetTransformInfo *TTI) {
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Eager);

  return DwarfEHPrepare(OptLevel, RewindFunction, F, TLI, DT ? &DTU : nullptr,
                        TTI)
      .run();
}

namespace {

class DwarfEHPrepareLegacyPass : public FunctionPass {
  // RewindFunction - _Unwind_Resume or the target equivalent.
  FunctionCallee RewindFunction = nullptr;

  CodeGenOpt::Level OptLevel;

public:
  static char ID; // Pass identification, replacement for typeid.

  DwarfEHPrepareLegacyPass(CodeGenOpt::Level OptLevel = CodeGenOpt::Default)
      : FunctionPass(ID), OptLevel(OptLevel) {}

  bool runOnFunction(Function &F) override {
    const TargetMachine &TM =
        getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
    const TargetLowering &TLI = *TM.getSubtargetImpl(F)->getTargetLowering();
    DominatorTree *DT = nullptr;
    const TargetTransformInfo *TTI = nullptr;
    if (OptLevel != CodeGenOpt::None) {
      DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
      TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    }
    return prepareDwarfEH(OptLevel, RewindFunction, F, TLI, DT, TTI);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    if (OptLevel != CodeGenOpt::None) {
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<TargetTransformInfoWrapperPass>();
      if (RequireAndPreserveDomTree)
        AU.addPreserved<DominatorTreeWrapperPass>();
    }
  }

  StringRef getPassName() const override {
    return "Exception handling preparation";
  }
};

} // end anonymous namespace

char DwarfEHPrepareLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(DwarfEHPrepareLegacyPass, DEBUG_TYPE,
                      "Prepare DWARF exceptions", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(DwarfEHPrepareLegacyPass, DEBUG_TYPE,
                    "Prepare DWARF exceptions", false, false)

FunctionPass *llvm::createDwarfEHPass(CodeGenOpt::Level OptLevel) {
  return new DwarfEHPrepareLegacyPass(OptLevel);
}
