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
#include "llvm/Analysis/LazyBlockFrequencyInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/LLVMContext.h"

using namespace llvm;

OptimizationRemarkEmitter::OptimizationRemarkEmitter() : FunctionPass(ID) {
  initializeOptimizationRemarkEmitterPass(*PassRegistry::getPassRegistry());
}

Optional<uint64_t> OptimizationRemarkEmitter::computeHotness(Value *V) {
  if (!BFI)
    return None;

  return BFI->getBlockProfileCount(cast<BasicBlock>(V));
}

void OptimizationRemarkEmitter::emitOptimizationRemarkMissed(
    const char *PassName, const DebugLoc &DLoc, Value *V, const Twine &Msg) {
  LLVMContext &Ctx = F->getContext();
  Ctx.diagnose(DiagnosticInfoOptimizationRemarkMissed(PassName, *F, DLoc, Msg,
                                                      computeHotness(V)));
}

void OptimizationRemarkEmitter::emitOptimizationRemarkMissed(
    const char *PassName, Loop *L, const Twine &Msg) {
  emitOptimizationRemarkMissed(PassName, L->getStartLoc(), L->getHeader(), Msg);
}

bool OptimizationRemarkEmitter::runOnFunction(Function &Fn) {
  F = &Fn;

  if (Fn.getContext().getDiagnosticHotnessRequested())
    BFI = &getAnalysis<LazyBlockFrequencyInfoPass>().getBFI();
  else
    BFI = nullptr;

  return false;
}

void OptimizationRemarkEmitter::getAnalysisUsage(AnalysisUsage &AU) const {
  LazyBlockFrequencyInfoPass::getLazyBFIAnalysisUsage(AU);
  AU.setPreservesAll();
}

char OptimizationRemarkEmitter::ID = 0;
static const char ore_name[] = "Optimization Remark Emitter";
#define ORE_NAME "opt-remark-emitter"

INITIALIZE_PASS_BEGIN(OptimizationRemarkEmitter, ORE_NAME, ore_name, false,
                      true)
INITIALIZE_PASS_DEPENDENCY(LazyBFIPass)
INITIALIZE_PASS_END(OptimizationRemarkEmitter, ORE_NAME, ore_name, false, true)
