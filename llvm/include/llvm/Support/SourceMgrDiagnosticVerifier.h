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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include <optional>
#include <string>

namespace llvm {

class MemoryBuffer;
class raw_ostream;

/// Scans SourceMgr source buffers for 'expected-<kind>' comments (e.g.
/// 'expected-error {{message}}') and verifies that the diagnostics reported
/// to it via \c process match them, and that every expected diagnostic is
/// matched at least once.
class LLVM_ABI SourceMgrDiagnosticVerifier {
public:
  /// The result of matching a single actual diagnostic against the expected
  /// diagnostics recorded for its location.
  enum class MatchResult {
    /// The diagnostic matched an expected diagnostic (kind and text).
    Matched,
    /// The diagnostic's text matched an expected diagnostic, but its kind
    /// didn't; a message about this has already been printed.
    NearMiss,
    /// The diagnostic did not match any expected diagnostic.
    Unexpected,
    /// The diagnostic did not match, but \p reportUnexpected was false, so
    /// nothing was printed and nothing needs to be done.
    Ignored,
  };

  /// The location of an actual diagnostic.
  struct Location {
    /// The buffer containing the diagnostic, or null if it is in a file that
    /// is not known to the SourceMgr (such a diagnostic never matches).
    const MemoryBuffer *buf;
    /// The 1-based line number of the diagnostic.
    unsigned lineNo;
  };

  /// Scans all buffers currently registered with \p mgr for expected
  /// diagnostics. Buffers added later are scanned the first time a
  /// diagnostic is reported in them. Verifier errors (near misses, unmatched
  /// expected diagnostics, malformed directives) are printed to \p os,
  /// bypassing \p mgr's diagnostic handler.
  ///
  /// \p prefixes are the directive prefixes to recognize (e.g. "expected"
  /// for 'expected-error'); they must be non-empty. If \p commentPrefixes is
  /// non-empty, directives are only recognized after the first occurrence of
  /// one of these strings on a line, so that text outside of comments is
  /// ignored.
  SourceMgrDiagnosticVerifier(const SourceMgr &mgr, raw_ostream &os,
                              ArrayRef<StringRef> prefixes = {"expected"},
                              ArrayRef<StringRef> commentPrefixes = {});

  /// Matches a diagnostic against the expected diagnostics for \p loc, or
  /// against the '@unknown' expected diagnostics if \p loc is std::nullopt.
  /// Prints a message on a near miss. Reporting an unexpected diagnostic is
  /// left to the caller. If \p reportUnexpected is false, a diagnostic that
  /// does not match is ignored instead of failing verification.
  MatchResult process(SourceMgr::DiagKind kind, std::optional<Location> loc,
                      StringRef message, bool reportUnexpected = true);

  /// Like the overload above, but takes the location from \p diag and also
  /// prints \p diag if it is unexpected.
  MatchResult process(const SMDiagnostic &diag, bool reportUnexpected = true);

  /// Reports any expected diagnostic that was never matched by a call to
  /// \c process. Returns whether verification succeeded overall, i.e. no
  /// diagnostic mismatches were recorded either here or by \c process.
  bool verify();

private:
  /// A single diagnostic expected via an 'expected-<kind>' comment.
  struct ExpectedDiag {
    ExpectedDiag(SourceMgr::DiagKind kind, unsigned lineNo, SMLoc fileLoc,
                 StringRef substring)
        : kind(kind), lineNo(lineNo), fileLoc(fileLoc), substring(substring) {}

    /// Returns true if this diagnostic matches the given message.
    LLVM_ABI bool match(StringRef str) const;

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

  /// Prints \p diag to \c os, bypassing the SourceMgr's diagnostic handler so
  /// that a verifier installed as that handler does not re-enter itself.
  void printDiag(const SMDiagnostic &diag) const;

  /// Records a verification failure, prints \p msg at \p diag's location and
  /// returns false.
  bool emitError(const ExpectedDiag &diag, const Twine &msg);

  /// Computes the regex matcher for a '-re' diagnostic's substring. Returns
  /// false and prints a message on error.
  bool computeRegex(ExpectedDiag &diag);

  /// Scans \p buf for expected diagnostics and returns the list for its
  /// buffer identifier.
  MutableArrayRef<ExpectedDiag> computeExpectedDiags(const MemoryBuffer *buf);

  const SourceMgr &mgr;
  raw_ostream &os;

  /// Regex used to recognize '<prefix>-<kind>' comments.
  Regex expected;

  /// The strings that start a comment, see the constructor.
  SmallVector<std::string, 0> commentPrefixes;

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
