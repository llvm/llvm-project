//===- PGOFlowVerify.cpp - PGO flow verification -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// After-pass hook and standalone pass. Flag behavior is on the cl::opt
// definitions below.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/PGOFlowVerify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ProfileSummary.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "verify-pgo-flow"

static cl::opt<bool> VerifyPGOFlow(
    "verify-pgo-flow", cl::init(false), cl::Hidden,
    cl::desc("Run InstrProf flow checks after IR-changing passes"));

static cl::opt<bool> VerifyPGOFlowPrintDiagnostics(
    "verify-pgo-flow-print-diagnostics", cl::init(true), cl::Hidden,
    cl::desc("Print verify-pgo-flow banners and findings to stderr"));

static cl::opt<bool> VerifyPGOFlowFatal(
    "verify-pgo-flow-fatal", cl::init(false), cl::Hidden,
    cl::desc("Abort after a flow-check finding (no-op until checks land)"));

static void printVerifyBanner(StringRef PassName, bool Skipped) {
  if (!VerifyPGOFlowPrintDiagnostics)
    return;
  errs() << "*** PGO Flow Verification After " << PassName
         << (Skipped ? " (Skipped)" : "") << " ***\n";
}

bool PGOFlowVerifier::isHookEnabled() { return VerifyPGOFlow; }

void PGOFlowVerifier::registerCallbacks(PassInstrumentationCallbacks &PIC) {
  if (!VerifyPGOFlow) {
    LLVM_DEBUG(dbgs() << "PGOFlowVerifier: hook not registered "
                         "(-verify-pgo-flow is off)\n");
    return;
  }

  LLVM_DEBUG(dbgs() << "PGOFlowVerifier: registering after-pass hook\n");
  PIC.registerAfterPassCallback([this](StringRef PassName, IRUnitRef IR,
                                       const PreservedAnalyses &PA) {
    // Same ignore list as print-changed. Adaptors would re-walk the
    // function after every nested loop pass.
    static const std::vector<StringRef> Ignored = {"PassManager",
                                                   "PassAdaptor",
                                                   "AnalysisManagerProxy",
                                                   "DevirtSCCRepeatedPass",
                                                   "ModuleInlinerWrapperPass",
                                                   "VerifierPass",
                                                   "PrintModulePass",
                                                   "PrintMIRPass",
                                                   "PrintMIRPreparePass",
                                                   "RequireAnalysisPass",
                                                   "InvalidateAnalysisPass"};
    if (isSpecialPass(PassName, Ignored)) {
      LLVM_DEBUG(dbgs() << "PGOFlowVerifier: after " << PassName
                        << " (skip, ignored pass)\n");
      return;
    }
    bool Changed = !PA.areAllPreserved();
    LLVM_DEBUG(
        dbgs() << "PGOFlowVerifier: after " << PassName
               << (Changed ? " (walk)\n" : " (skip, PA all preserved)\n"));
    printVerifyBanner(PassName, /*Skipped=*/!Changed);
    if (!Changed)
      return;
    runAfterPass(PassName, IR);
  });
}

void PGOFlowVerifier::runAfterPass(StringRef PassID, IRUnitRef IR) {
  if (const auto *M = dyn_cast<Module>(IR)) {
    LLVM_DEBUG(dbgs() << "PGOFlowVerifier: module IR after " << PassID << "\n");
    runAfterPass(M);
  } else if (const auto *F = dyn_cast<Function>(IR)) {
    LLVM_DEBUG(dbgs() << "PGOFlowVerifier: function IR '" << F->getName()
                      << "' after " << PassID << "\n");
    runAfterPass(F);
  } else if (const auto *C = dyn_cast<LazyCallGraph::SCC>(IR)) {
    LLVM_DEBUG(dbgs() << "PGOFlowVerifier: SCC IR after " << PassID << "\n");
    runAfterPass(C);
  } else if (const auto *L = dyn_cast<Loop>(IR)) {
    LLVM_DEBUG(dbgs() << "PGOFlowVerifier: loop IR after " << PassID << "\n");
    runAfterPass(L);
  } else {
    LLVM_DEBUG(dbgs() << "PGOFlowVerifier: unhandled IR unit after " << PassID
                      << "\n");
  }
}

void PGOFlowVerifier::runAfterPass(const Module *M) {
  if (!M)
    return;
  for (const Function &F : *M)
    runAfterPass(&F);
}

void PGOFlowVerifier::runAfterPass(const Function *F) {
  if (!F || F->isDeclaration())
    return;
  if (!hasInstrProfUseSummary(F->getParent())) {
    LLVM_DEBUG(dbgs() << "PGOFlowVerifier: skip '" << F->getName()
                      << "' (no InstrProf use-phase summary)\n");
    return;
  }
}

bool PGOFlowVerifier::hasInstrProfUseSummary(const Module *M) const {
  if (!M)
    return false;
  Metadata *SummaryMD = M->getProfileSummary(/*IsCS=*/false);
  if (!SummaryMD)
    return false;
  std::unique_ptr<ProfileSummary> PS(ProfileSummary::getFromMD(SummaryMD));
  return PS && (PS->getKind() == ProfileSummary::PSK_Instr ||
                PS->getKind() == ProfileSummary::PSK_CSInstr);
}

void PGOFlowVerifier::runAfterPass(const LazyCallGraph::SCC *C) {
  if (!C)
    return;
  for (const LazyCallGraph::Node &N : *C)
    runAfterPass(&N.getFunction());
}

void PGOFlowVerifier::runAfterPass(const Loop *L) {
  if (!L)
    return;
  runAfterPass(L->getHeader()->getParent());
}

PreservedAnalyses PGOFlowVerifierPass::run(Module &M,
                                           ModuleAnalysisManager &MAM) {
  (void)MAM;
  LLVM_DEBUG(dbgs() << "PGOFlowVerifier: pipeline pass (module)\n");
  printVerifyBanner("verify-pgo-flow", /*Skipped=*/false);
  Verifier.runAfterPass("verify-pgo-flow", M);
  return PreservedAnalyses::all();
}

PreservedAnalyses PGOFlowVerifierPass::run(Function &F,
                                           FunctionAnalysisManager &FAM) {
  (void)FAM;
  LLVM_DEBUG(dbgs() << "PGOFlowVerifier: pipeline pass (function "
                    << F.getName() << ")\n");
  printVerifyBanner("verify-pgo-flow", /*Skipped=*/false);
  Verifier.runAfterPass("verify-pgo-flow", F);
  return PreservedAnalyses::all();
}
