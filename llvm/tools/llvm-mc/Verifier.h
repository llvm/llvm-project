//===- Verifier.h - Diagnostic Verifier for llvm-mc -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements '-verify=<prefixes>', a mode that checks emitted
// diagnostics against '<prefix>-error'/'-warning'/'-note'/'-remark' comments
// in the input, similar to clang's '-verify' flag.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MC_VERIFIER_H
#define LLVM_TOOLS_LLVM_MC_VERIFIER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/SourceMgrDiagnosticVerifier.h"
#include <string>

namespace llvm {

class MCContext;
class SourceMgr;
class SMDiagnostic;

/// Checks the diagnostics produced while assembling/disassembling against
/// '<prefix>-<kind>' comments in the input.
class MCVerifier {
public:
  /// \p Prefixes are the accepted directive prefixes (from '-verify=...').
  /// \p CommentPrefixes are the comment-start strings recognized by the
  /// current target, so a directive outside of a comment is ignored.
  MCVerifier(SourceMgr &SrcMgr, ArrayRef<std::string> Prefixes,
             ArrayRef<std::string> CommentPrefixes);

  /// Clears SrcMgr's diagnostic handler if it's still this object's, so a
  /// SourceMgr that outlives this MCVerifier is never left with a dangling
  /// handler pointing at freed memory.
  ~MCVerifier();

  /// Installs this as the diagnostic handler for both the SourceMgr passed to
  /// the constructor and \p Ctx. Both are needed: most parser/disassembler
  /// diagnostics go through SourceMgr, but some (e.g. the DWARF CFI checker's,
  /// via MCContext::reportError/reportWarning) are reported directly through
  /// MCContext and never reach SourceMgr's handler at all.
  void installHandlers(MCContext &Ctx);

  /// Reports any expected diagnostic that was never produced, and returns
  /// whether verification succeeded overall.
  bool verify();

private:
  static void handleSourceMgrDiag(const SMDiagnostic &Diag, void *Context);
  void process(const SMDiagnostic &Diag);

  SourceMgr &SrcMgr;
  SourceMgrDiagnosticVerifier Verifier;
};

} // namespace llvm

#endif // LLVM_TOOLS_LLVM_MC_VERIFIER_H
