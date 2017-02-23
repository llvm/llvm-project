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
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LLVMContext.h"

using namespace llvm;

OptimizationRemarkEmitter::OptimizationRemarkEmitter(const Function *F)
    : F(F), BFI(nullptr) {
  if (!F->getContext().getDiagnosticHotnessRequested())
    return;

  // First create a dominator tree.
  DominatorTree DT;
  DT.recalculate(*const_cast<Function *>(F));

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

bool OptimizationRemarkEmitter::invalidate(
    Function &F, const PreservedAnalyses &PA,
    FunctionAnalysisManager::Invalidator &Inv) {
  // This analysis has no state and so can be trivially preserved but it needs
  // a fresh view of BFI if it was constructed with one.
  if (BFI && Inv.invalidate<BlockFrequencyAnalysis>(F, PA))
    return true;

  // Otherwise this analysis result remains valid.
  return false;
}

Optional<uint64_t> OptimizationRemarkEmitter::computeHotness(const Value *V) {
  if (!BFI)
    return None;

  return BFI->getBlockProfileCount(cast<BasicBlock>(V));
}

namespace llvm {
namespace yaml {

void MappingTraits<DiagnosticInfoOptimizationBase *>::mapping(
    IO &io, DiagnosticInfoOptimizationBase *&OptDiag) {
  assert(io.outputting() && "input not yet implemented");

  if (io.mapTag("!Passed",
                (OptDiag->getKind() == DK_OptimizationRemark ||
                 OptDiag->getKind() == DK_MachineOptimizationRemark)))
    ;
  else if (io.mapTag(
               "!Missed",
               (OptDiag->getKind() == DK_OptimizationRemarkMissed ||
                OptDiag->getKind() == DK_MachineOptimizationRemarkMissed)))
    ;
  else if (io.mapTag(
               "!Analysis",
               (OptDiag->getKind() == DK_OptimizationRemarkAnalysis ||
                OptDiag->getKind() == DK_MachineOptimizationRemarkAnalysis)))
    ;
  else if (io.mapTag("!AnalysisFPCommute",
                     OptDiag->getKind() ==
                         DK_OptimizationRemarkAnalysisFPCommute))
    ;
  else if (io.mapTag("!AnalysisAliasing",
                     OptDiag->getKind() ==
                         DK_OptimizationRemarkAnalysisAliasing))
    ;
  else if (io.mapTag("!Failure", OptDiag->getKind() == DK_OptimizationFailure))
    ;
  else
    llvm_unreachable("Unknown remark type");

  // These are read-only for now.
  DiagnosticLocation DL = OptDiag->getLocation();
  StringRef FN =
      GlobalValue::getRealLinkageName(OptDiag->getFunction().getName());

  StringRef PassName(OptDiag->PassName);
  io.mapRequired("Pass", PassName);
  io.mapRequired("Name", OptDiag->RemarkName);
  if (!io.outputting() || DL.isValid())
    io.mapOptional("DebugLoc", DL);
  io.mapRequired("Function", FN);
  io.mapOptional("Hotness", OptDiag->Hotness);
  io.mapOptional("Args", OptDiag->Args);
}

template <> struct MappingTraits<DiagnosticLocation> {
  static void mapping(IO &io, DiagnosticLocation &DL) {
    assert(io.outputting() && "input not yet implemented");

    StringRef File = DL.getFilename();
    unsigned Line = DL.getLine();
    unsigned Col = DL.getColumn();

    io.mapRequired("File", File);
    io.mapRequired("Line", Line);
    io.mapRequired("Column", Col);
  }

  static const bool flow = true;
};

// Implement this as a mapping for now to get proper quotation for the value.
template <> struct MappingTraits<DiagnosticInfoOptimizationBase::Argument> {
  static void mapping(IO &io, DiagnosticInfoOptimizationBase::Argument &A) {
    assert(io.outputting() && "input not yet implemented");
    io.mapRequired(A.Key.data(), A.Val);
    if (A.Loc.isValid())
      io.mapOptional("DebugLoc", A.Loc);
  }
};

} // end namespace yaml
} // end namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(DiagnosticInfoOptimizationBase::Argument)

void OptimizationRemarkEmitter::computeHotness(
    DiagnosticInfoIROptimization &OptDiag) {
  const Value *V = OptDiag.getCodeRegion();
  if (V)
    OptDiag.setHotness(computeHotness(V));
}

void OptimizationRemarkEmitter::emit(
    DiagnosticInfoOptimizationBase &OptDiagBase) {
  auto &OptDiag = cast<DiagnosticInfoIROptimization>(OptDiagBase);
  computeHotness(OptDiag);

  yaml::Output *Out = F->getContext().getDiagnosticsOutputFile();
  if (Out) {
    auto *P = const_cast<DiagnosticInfoOptimizationBase *>(&OptDiagBase);
    *Out << P;
  }
  // FIXME: now that IsVerbose is part of DI, filtering for this will be moved
  // from here to clang.
  if (!OptDiag.isVerbose() || shouldEmitVerbose())
    F->getContext().diagnose(OptDiag);
}

void OptimizationRemarkEmitter::emitOptimizationRemark(const char *PassName,
                                                       const DebugLoc &DLoc,
                                                       const Value *V,
                                                       const Twine &Msg) {
  LLVMContext &Ctx = F->getContext();
  Ctx.diagnose(OptimizationRemark(PassName, *F, DLoc, Msg, computeHotness(V)));
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
    Ctx.diagnose(
        OptimizationRemarkMissed(PassName, *F, DLoc, Msg, computeHotness(V)));
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
    Ctx.diagnose(
        OptimizationRemarkAnalysis(PassName, *F, DLoc, Msg, computeHotness(V)));
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
  Ctx.diagnose(OptimizationRemarkAnalysisFPCommute(PassName, *F, DLoc, Msg,
                                                   computeHotness(V)));
}

void OptimizationRemarkEmitter::emitOptimizationRemarkAnalysisAliasing(
    const char *PassName, const DebugLoc &DLoc, const Value *V,
    const Twine &Msg) {
  LLVMContext &Ctx = F->getContext();
  Ctx.diagnose(OptimizationRemarkAnalysisAliasing(PassName, *F, DLoc, Msg,
                                                  computeHotness(V)));
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

AnalysisKey OptimizationRemarkEmitterAnalysis::Key;

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
