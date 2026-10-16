//===- Verifier.cpp - Diagnostic Verifier for llvm-mc ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Verifier.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
/// SourceMgr::PrintMessage unconditionally re-invokes SrcMgr's installed
/// DiagHandler if one is set, rather than only doing so for diagnostics that
/// didn't originate from the handler itself. Since MCVerifier is installed as
/// that handler, and its own printing (near-miss / not-produced messages)
/// goes through SourceMgr::PrintMessage, calling into it from inside the
/// handler would otherwise re-enter the handler and recurse indefinitely.
/// This temporarily clears the handler for the duration of the guard so that
/// PrintMessage prints directly instead.
class ScopedHandlerDisable {
  SourceMgr &SrcMgr;
  SourceMgr::DiagHandlerTy SavedHandler;
  void *SavedContext;

public:
  explicit ScopedHandlerDisable(SourceMgr &SrcMgr)
      : SrcMgr(SrcMgr), SavedHandler(SrcMgr.getDiagHandler()),
        SavedContext(SrcMgr.getDiagContext()) {
    SrcMgr.setDiagHandler(nullptr, nullptr);
  }
  ~ScopedHandlerDisable() { SrcMgr.setDiagHandler(SavedHandler, SavedContext); }
};
} // namespace

MCVerifier::MCVerifier(SourceMgr &SrcMgr, ArrayRef<std::string> Prefixes,
                       ArrayRef<std::string> CommentPrefixes)
    : SrcMgr(SrcMgr), Verifier(Prefixes, CommentPrefixes) {
  // Scan every buffer already known to SrcMgr up front, so that an expected
  // diagnostic that's never actually produced (e.g. because the run never
  // reports any diagnostic at all) is still recorded and reported as missing
  // by verify(), rather than never being noticed since process() was never
  // called for that buffer. Guarded like every other call into Verifier in
  // this file: a malformed directive (e.g. an invalid '-re' regex) makes
  // computeExpectedDiags print through SrcMgr, which would otherwise re-enter
  // installHandlers's handler if the constructor ever ran after it (it
  // currently never does, at the sole call site in llvm-mc.cpp).
  ScopedHandlerDisable Guard(SrcMgr);
  for (unsigned I = 0, E = SrcMgr.getNumBuffers(); I != E; ++I)
    (void)Verifier.computeExpectedDiags(errs(), SrcMgr,
                                        SrcMgr.getMemoryBuffer(I + 1));
}

MCVerifier::~MCVerifier() {
  // SrcMgr may outlive this MCVerifier; don't leave its handler pointing at
  // freed memory.
  if (SrcMgr.getDiagHandler() == &MCVerifier::handleSourceMgrDiag)
    SrcMgr.setDiagHandler(nullptr, nullptr);
}

void MCVerifier::installHandlers(MCContext &Ctx) {
  SrcMgr.setDiagHandler(&MCVerifier::handleSourceMgrDiag, this);
  Ctx.setDiagnosticHandler(
      [this](const SMDiagnostic &Diag, bool /*InlineAsm*/,
             const SourceMgr & /*SrcMgr*/,
             std::vector<const MDNode *> & /*LocInfos*/) { process(Diag); });
}

bool MCVerifier::verify() {
  ScopedHandlerDisable Guard(SrcMgr);
  return Verifier.verify(errs(), SrcMgr);
}

void MCVerifier::handleSourceMgrDiag(const SMDiagnostic &Diag, void *Context) {
  static_cast<MCVerifier *>(Context)->process(Diag);
}

void MCVerifier::process(const SMDiagnostic &Diag) {
  ScopedHandlerDisable Guard(SrcMgr);

  bool HasLoc = Diag.getLoc().isValid();
  const MemoryBuffer *Buf = nullptr;
  if (HasLoc) {
    if (unsigned ID = SrcMgr.FindBufferContainingLoc(Diag.getLoc()))
      Buf = SrcMgr.getMemoryBuffer(ID);
  }

  auto Result = Verifier.process(
      errs(), SrcMgr, Diag.getKind(), HasLoc, Buf,
      HasLoc ? static_cast<unsigned>(Diag.getLineNo()) : 0, Diag.getMessage());
  if (Result == SourceMgrDiagnosticVerifier::MatchResult::Unexpected)
    SrcMgr.PrintMessage(errs(), Diag);
}
