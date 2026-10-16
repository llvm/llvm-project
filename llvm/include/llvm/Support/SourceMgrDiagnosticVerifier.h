//===- SourceMgrDiagnosticVerifier.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a utility for verifying that diagnostics reported through
// a SourceMgr match 'expected-<kind>' comments in the source, for
// implementing '-verify'-style diagnostic tests on top of a plain
// llvm::SourceMgr.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_SOURCEMGRDIAGNOSTICVERIFIER_H
#define LLVM_SUPPORT_SOURCEMGRDIAGNOSTICVERIFIER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include <optional>

namespace llvm {

class MemoryBuffer;
class raw_ostream;

/// Scans SourceMgr source buffers for 'expected-<kind>' comments (e.g.
/// 'expected-error {{message}}') and verifies that the diagnostics reported
/// to it via \c process match them, exactly once each.
class LLVM_ABI SourceMgrDiagnosticVerifier {
public:
  /// A single diagnostic expected via an 'expected-<kind>' comment.
  struct ExpectedDiag {
    ExpectedDiag(SourceMgr::DiagKind kind, unsigned lineNo, SMLoc fileLoc,
                 StringRef substring)
        : kind(kind), lineNo(lineNo), fileLoc(fileLoc), substring(substring) {
    }

    /// Returns true if this diagnostic matches the given message.
    bool match(StringRef str) const;

    /// Computes the regex matcher for a '-re' diagnostic's substring.
    /// Returns false and prints a message through \p mgr on error.
    bool computeRegex(raw_ostream &os, SourceMgr &mgr);

    /// Prints \p msg at this diagnostic's location and returns false, for
    /// use as `return emitError(...);` in functions that report failure via
    /// a bool return.
    bool emitError(raw_ostream &os, SourceMgr &mgr, const Twine &msg) const;

    /// The severity of the diagnostic expected.
    SourceMgr::DiagKind kind;
    /// The line number the expected diagnostic should be on.
    unsigned lineNo;
    /// The location of the expected diagnostic within the input file.
    SMLoc fileLoc;
    /// A flag indicating if the expected diagnostic has been matched yet.
    bool matched = false;
    /// The substring that is expected to be within the diagnostic.
    StringRef substring;
    /// An optional regex matcher, if the expected diagnostic substring was a
    /// regex string.
    std::optional<Regex> substringRegex;
  };

  /// The result of matching a single actual diagnostic against the expected
  /// diagnostics recorded for its location.
  enum class MatchResult {
    /// The diagnostic matched an expected diagnostic (kind and text).
    Matched,
    /// The diagnostic's text matched an expected diagnostic, but its kind
    /// didn't; a message about this has already been printed.
    NearMiss,
    /// The diagnostic did not match any expected diagnostic. The caller is
    /// responsible for reporting it, if desired.
    Unexpected,
    /// The diagnostic did not match, but \p reportUnexpected was false, so
    /// nothing was printed and nothing needs to be done.
    Ignored,
  };

  SourceMgrDiagnosticVerifier() = default;

  /// Computes and caches the list of expected diagnostics for \p buf, if not
  /// already cached. Returns the (mutable) cached list.
  MutableArrayRef<ExpectedDiag> computeExpectedDiags(raw_ostream &os,
                                                      SourceMgr &mgr,
                                                      const MemoryBuffer *buf);

  /// Returns the cached expected diagnostics for the buffer named \p bufName,
  /// or std::nullopt if \p computeExpectedDiags hasn't been called for it.
  std::optional<MutableArrayRef<ExpectedDiag>>
  getExpectedDiags(StringRef bufName);

  /// Returns the expected diagnostics with an '@unknown' location.
  MutableArrayRef<ExpectedDiag> getExpectedUnknownLocDiags() {
    return expectedUnknownLocDiags;
  }

  /// Matches a single actual diagnostic against the expected diagnostics
  /// recorded for \p buf / \p lineNo, computing them first via \p
  /// computeExpectedDiags if they haven't been already. If \p hasLoc is
  /// false, the diagnostic has no location and is matched against the
  /// '@unknown' list instead (\p buf / \p lineNo are ignored). If \p hasLoc
  /// is true but \p buf is null (e.g. the diagnostic's file isn't a known
  /// buffer), the diagnostic is matched against an empty list, i.e. it can
  /// never match and is always unexpected. On a near miss, prints a message
  /// through \p mgr. \p reportUnexpected controls whether near misses /
  /// unexpected diagnostics are reported at all.
  MatchResult process(raw_ostream &os, SourceMgr &mgr, SourceMgr::DiagKind kind,
                       bool hasLoc, const MemoryBuffer *buf, unsigned lineNo,
                       StringRef message, bool reportUnexpected = true);

  /// Reports (through \p mgr) any expected diagnostic that was never matched
  /// by a call to \p process. Returns whether verification succeeded overall,
  /// i.e. no diagnostic mismatches were recorded either here or by \p
  /// process.
  bool verify(raw_ostream &os, SourceMgr &mgr);

private:
  /// Regex used to recognize 'expected-<kind>' comments.
  Regex expected{"expected-(error|note|remark|warning)(-re)? "
                 "*(@([+-][0-9]+|above|below|unknown))? *{{(.*)}}$"};

  /// The expected diagnostics for each buffer that has been scanned so far,
  /// keyed by buffer identifier (i.e. file name).
  StringMap<SmallVector<ExpectedDiag, 2>> expectedDiagsPerFile;

  /// The expected diagnostics with an '@unknown' location.
  SmallVector<ExpectedDiag, 2> expectedUnknownLocDiags;

  /// Whether any diagnostic mismatch has been recorded so far.
  bool ok = true;
};

} // namespace llvm

#endif // LLVM_SUPPORT_SOURCEMGRDIAGNOSTICVERIFIER_H
