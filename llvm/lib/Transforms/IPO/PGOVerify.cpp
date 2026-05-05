//===- PGOVerify.cpp - PGO Verification ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// IPGOVerifier currently provides registration-only diagnostics for
// pass-instrumentation tracing under `-verify-ipgo`.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/PGOVerify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "verify-ipgo"

static cl::opt<bool> VerifyIPGOPrintDiagnostics(
    "verify-ipgo-print-diagnostics", cl::init(true), cl::Hidden,
    cl::desc("Print verify-ipgo diagnostics to stderr"));

static cl::opt<bool>
    VerifyIPGO("verify-ipgo", cl::init(false), cl::Hidden,
               cl::desc("Enable Instrumented PGO verification"));

/// Register post-pass diagnostic callbacks for `-verify-ipgo`.
///
/// \param PIC Pass instrumentation callback registry.
void IPGOVerifier::registerCallbacks(PassInstrumentationCallbacks &PIC) {
  if (!VerifyIPGO)
    return;

  PIC.registerAfterPassCallback(
      [this](StringRef PassName, Any IR, const PreservedAnalyses &PA) {
        bool IsChanged = !PA.areAllPreserved();

        StringRef Skipped = IsChanged ? "" : " (Skipped)";
        if (VerifyIPGOPrintDiagnostics)
          errs() << "*** IPGO Verification After " << PassName << Skipped
                 << " ***\n";
        LLVM_DEBUG(dbgs() << "\n*** IPGO Verification After " << PassName
                          << Skipped << " ***\n");
        if (!IsChanged) {
          // Pass made no IR changes; skip verification.
          return;
        }

        runAfterPass(PassName, IR);
      });
}

/// Dispatch post-pass handling for supported IR unit kinds.
///
/// \param PassID Name of the pass that completed.
/// \param IR IR unit received from pass instrumentation callbacks.
void IPGOVerifier::runAfterPass(StringRef PassID, Any IR) {
  (void)PassID;

  if (const auto *M = any_cast<const Module *>(&IR))
    runAfterPass(*M);
  else if (const auto *F = any_cast<const Function *>(&IR)) {
    // The verifier does not mutate IR, but the handler API is function-based,
    // so adapt the callback payload here.
    auto *NonConstF = const_cast<Function *>(*F);
    runAfterPass(NonConstF);
  } else if (const auto *C = any_cast<const LazyCallGraph::SCC *>(&IR))
    runAfterPass(*C);
  else if (const auto *L = any_cast<const Loop *>(&IR))
    runAfterPass(*L);
  else {
    return;
  }
}

/// Delegate module callback handling to the function handler.
///
/// \param M Module callback payload.
void IPGOVerifier::runAfterPass(const Module *M) {
  for (const Function &F : *M) {
    if (F.isDeclaration())
      continue;

    runAfterPass(const_cast<Function *>(&F));
  }
}

/// Per-function post-pass handler.
///
/// \param F Function callback payload.
void IPGOVerifier::runAfterPass(const Function *F) {
  if (!F || F->isDeclaration())
    return;
}

/// Delegate SCC callback handling to the function handler.
///
/// \param C SCC callback payload.
void IPGOVerifier::runAfterPass(const LazyCallGraph::SCC *C) {
  for (const LazyCallGraph::Node &N : *C)
    runAfterPass(&N.getFunction());
}

/// Delegate loop callback handling to the containing function handler.
///
/// \param L Loop callback payload.
void IPGOVerifier::runAfterPass(const Loop *L) {
  runAfterPass(L->getHeader()->getParent());
}
