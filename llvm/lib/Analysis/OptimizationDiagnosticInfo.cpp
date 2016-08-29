//===- OptimizationDiagnosticInfo.cpp - Optimization Diagnostic -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Optimization diagnostic interfaces.  It's packaged as an analysis pass so
// that by using this service passes become dependent on BFI as well.  BFI is
// used to compute the "hotness" of the diagnostic message.
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/OptimizationDiagnosticInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LazyBlockFrequencyInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LLVMContext.h"

using namespace llvm;

OptimizationRemarkEmitter::OptimizationRemarkEmitter(Function *F)
    : F(F), BFI(nullptr) {
  if (!F->getContext().getDiagnosticHotnessRequested())
    return;

  // First create a dominator tree.
  DominatorTree DT;
  DT.recalculate(*F);

  // Generate LoopInfo from it.
  LoopInfo LI;
  LI.analyze(DT);

  // Then compute BranchProbabilityInfo.
  BranchProbabilityInfo BPI;
  BPI.calculate(*F, LI);

  // Finally compute BFI.
  OwnedBFI = llvm::make_unique<BlockFrequencyInfo>(*F, BPI, LI);
  BFI = OwnedBFI.get();
}

Optional<uint64_t> OptimizationRemarkEmitter::computeHotness(const Value *V) {
  if (!BFI)
    return None;

  return BFI->getBlockProfileCount(cast<BasicBlock>(V));
}

void OptimizationRemarkEmitter::emitOptimizationRemark(const char *PassName,
                                                       const DebugLoc &DLoc,
                                                       const Value *V,
                                                       const Twine &Msg) {
  LLVMContext &Ctx = F->getContext();
  Ctx.diagnose(DiagnosticInfoOptimizationRemark(PassName, *F, DLoc, Msg,
                                                computeHotness(V)));
}

void OptimizationRemarkEmitter::emitOptimizationRemark(const char *PassName,
                                                       Loop *L,
                                                       const Twine &Msg) {
  emitOptimizationRemark(PassName, L->getStartLoc(), L->getHeader(), Msg);
}

void OptimizationRemarkEmitter::emitOptimizationRemarkMissed(
    const char *PassName, const DebugLoc &DLoc, const Value *V,
    const Twine &Msg, bool IsVerbose) {
  LLVMContext &Ctx = F->getContext();
  if (!IsVerbose || shouldEmitVerbose())
    Ctx.diagnose(DiagnosticInfoOptimizationRemarkMissed(PassName, *F, DLoc, Msg,
                                                        computeHotness(V)));
}

void OptimizationRemarkEmitter::emitOptimizationRemarkMissed(
    const char *PassName, Loop *L, const Twine &Msg, bool IsVerbose) {
  emitOptimizationRemarkMissed(PassName, L->getStartLoc(), L->getHeader(), Msg,
                               IsVerbose);
}

void OptimizationRemarkEmitter::emitOptimizationRemarkAnalysis(
    const char *PassName, const DebugLoc &DLoc, const Value *V,
    const Twine &Msg, bool IsVerbose) {
  LLVMContext &Ctx = F->getContext();
  if (!IsVerbose || shouldEmitVerbose())
    Ctx.diagnose(DiagnosticInfoOptimizationRemarkAnalysis(
        PassName, *F, DLoc, Msg, computeHotness(V)));
}

void OptimizationRemarkEmitter::emitOptimizationRemarkAnalysis(
    const char *PassName, Loop *L, const Twine &Msg, bool IsVerbose) {
  emitOptimizationRemarkAnalysis(PassName, L->getStartLoc(), L->getHeader(),
                                 Msg, IsVerbose);
}

void OptimizationRemarkEmitter::emitOptimizationRemarkAnalysisFPCommute(
    const char *PassName, const DebugLoc &DLoc, const Value *V,
    const Twine &Msg) {
  LLVMContext &Ctx = F->getContext();
  Ctx.diagnose(DiagnosticInfoOptimizationRemarkAnalysisFPCommute(
      PassName, *F, DLoc, Msg, computeHotness(V)));
}

void OptimizationRemarkEmitter::emitOptimizationRemarkAnalysisAliasing(
    const char *PassName, const DebugLoc &DLoc, const Value *V,
    const Twine &Msg) {
  LLVMContext &Ctx = F->getContext();
  Ctx.diagnose(DiagnosticInfoOptimizationRemarkAnalysisAliasing(
      PassName, *F, DLoc, Msg, computeHotness(V)));
}

void OptimizationRemarkEmitter::emitOptimizationRemarkAnalysisAliasing(
    const char *PassName, Loop *L, const Twine &Msg) {
  emitOptimizationRemarkAnalysisAliasing(PassName, L->getStartLoc(),
                                         L->getHeader(), Msg);
}

OptimizationRemarkEmitterWrapperPass::OptimizationRemarkEmitterWrapperPass()
    : FunctionPass(ID) {
  initializeOptimizationRemarkEmitterWrapperPassPass(
      *PassRegistry::getPassRegistry());
}

bool OptimizationRemarkEmitterWrapperPass::runOnFunction(Function &Fn) {
  BlockFrequencyInfo *BFI;

  if (Fn.getContext().getDiagnosticHotnessRequested())
    BFI = &getAnalysis<LazyBlockFrequencyInfoPass>().getBFI();
  else
    BFI = nullptr;

  ORE = llvm::make_unique<OptimizationRemarkEmitter>(&Fn, BFI);
  return false;
}

void OptimizationRemarkEmitterWrapperPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  LazyBlockFrequencyInfoPass::getLazyBFIAnalysisUsage(AU);
  AU.setPreservesAll();
}

char OptimizationRemarkEmitterAnalysis::PassID;

OptimizationRemarkEmitter
OptimizationRemarkEmitterAnalysis::run(Function &F,
                                       FunctionAnalysisManager &AM) {
  BlockFrequencyInfo *BFI;

  if (F.getContext().getDiagnosticHotnessRequested())
    BFI = &AM.getResult<BlockFrequencyAnalysis>(F);
  else
    BFI = nullptr;

  return OptimizationRemarkEmitter(&F, BFI);
}

char OptimizationRemarkEmitterWrapperPass::ID = 0;
static const char ore_name[] = "Optimization Remark Emitter";
#define ORE_NAME "opt-remark-emitter"

INITIALIZE_PASS_BEGIN(OptimizationRemarkEmitterWrapperPass, ORE_NAME, ore_name,
                      false, true)
INITIALIZE_PASS_DEPENDENCY(LazyBFIPass)
INITIALIZE_PASS_END(OptimizationRemarkEmitterWrapperPass, ORE_NAME, ore_name,
                    false, true)
