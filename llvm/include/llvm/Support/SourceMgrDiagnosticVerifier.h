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
#include <vector>

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
    ExpectedDiag(SourceMgr::DiagKind Kind, unsigned LineNo, SMLoc FileLoc,
                 StringRef Substring)
        : Kind(Kind), LineNo(LineNo), FileLoc(FileLoc), Substring(Substring) {}

    /// Returns true if this diagnostic matches the given message.
    bool match(StringRef Str) const;

    /// Computes the regex matcher for a '-re' diagnostic's substring.
    /// Returns false and prints a message through \p Mgr on error.
    bool computeRegex(raw_ostream &OS, SourceMgr &Mgr);

    /// Prints \p Msg at this diagnostic's location and returns false, for
    /// use as `return emitError(...);` in functions that report failure via
    /// a bool return.
    bool emitError(raw_ostream &OS, SourceMgr &Mgr, const Twine &Msg) const;

    /// The severity of the diagnostic expected.
    SourceMgr::DiagKind Kind;
    /// The line number the expected diagnostic should be on.
    unsigned LineNo;
    /// The column the expected diagnostic should be at, or 0 if the column
    /// isn't checked (only the line is).
    unsigned ColNo = 0;
    /// The location of the expected diagnostic within the input file.
    SMLoc FileLoc;
    /// A flag indicating if the expected diagnostic has been matched yet.
    bool Matched = false;
    /// The substring that is expected to be within the diagnostic.
    StringRef Substring;
    /// An optional regex matcher, if the expected diagnostic substring was a
    /// regex string.
    std::optional<Regex> SubstringRegex;
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
    /// The diagnostic did not match, but \p ReportUnexpected was false, so
    /// nothing was printed and nothing needs to be done.
    Ignored,
  };

  /// \param Prefixes The comment prefixes that introduce an expected
  /// diagnostic, e.g. \c {"expected"} to recognize 'expected-error'. Matched
  /// literally, not as a regex. Must not contain an empty string.
  /// \param CommentPrefixes If non-empty, only text at or after the earliest
  /// occurrence of one of these strings on a line is scanned for expected
  /// diagnostics, so a magic string that happens to appear outside of a
  /// comment (e.g. in an instruction operand) is ignored. If empty, the
  /// whole line is eligible.
  explicit SourceMgrDiagnosticVerifier(
      ArrayRef<std::string> Prefixes = {"expected"},
      ArrayRef<std::string> CommentPrefixes = {});

  /// Computes and caches the list of expected diagnostics for \p Buf, if not
  /// already cached. Returns the (mutable) cached list.
  MutableArrayRef<ExpectedDiag> computeExpectedDiags(raw_ostream &OS,
                                                     SourceMgr &Mgr,
                                                     const MemoryBuffer *Buf);

  /// Returns the cached expected diagnostics for the buffer named \p BufName,
  /// or std::nullopt if \p computeExpectedDiags hasn't been called for it.
  std::optional<MutableArrayRef<ExpectedDiag>>
  getExpectedDiags(StringRef BufName);

  /// Returns the expected diagnostics with an '@unknown' location.
  MutableArrayRef<ExpectedDiag> getExpectedUnknownLocDiags() {
    return ExpectedUnknownLocDiags;
  }

  /// Matches a single actual diagnostic against the expected diagnostics
  /// recorded for \p Buf / \p LineNo / \p ColNo, computing them first via \p
  /// computeExpectedDiags if they haven't been already. If \p HasLoc is
  /// false, the diagnostic has no location and is matched against the
  /// '@unknown' list instead (\p Buf / \p LineNo / \p ColNo are ignored). If
  /// \p HasLoc is true but \p Buf is null (e.g. the diagnostic's file isn't a
  /// known buffer), the diagnostic is matched against an empty list, i.e. it
  /// can never match and is always unexpected. \p ColNo is only checked
  /// against expected diagnostics that requested a column (via ':<col>');
  /// others match on line alone regardless of \p ColNo. On a near miss,
  /// prints a message through \p Mgr. \p ReportUnexpected controls whether
  /// near misses / unexpected diagnostics are reported at all.
  MatchResult process(raw_ostream &OS, SourceMgr &Mgr, SourceMgr::DiagKind Kind,
                      bool HasLoc, const MemoryBuffer *Buf, unsigned LineNo,
                      unsigned ColNo, StringRef Message,
                      bool ReportUnexpected = true);

  /// Reports (through \p Mgr) any expected diagnostic that was never matched
  /// by a call to \p process. Returns whether verification succeeded overall,
  /// i.e. no diagnostic mismatches were recorded either here or by \p
  /// process.
  bool verify(raw_ostream &OS, SourceMgr &Mgr);

private:
  /// Regex used to recognize '<prefix>-<kind>' comments, built from the
  /// \p Prefixes passed to the constructor.
  Regex Expected;

  /// If non-empty, only text at or after the earliest occurrence of one of
  /// these strings on a line is eligible to match \p Expected.
  std::vector<std::string> CommentPrefixes;

  /// The expected diagnostics for each buffer that has been scanned so far,
  /// keyed by buffer identifier (i.e. file name).
  StringMap<SmallVector<ExpectedDiag, 2>> ExpectedDiagsPerFile;

  /// The expected diagnostics with an '@unknown' location.
  SmallVector<ExpectedDiag, 2> ExpectedUnknownLocDiags;

  /// Whether any diagnostic mismatch has been recorded so far.
  bool OK = true;
};

} // namespace llvm

#endif // LLVM_SUPPORT_SOURCEMGRDIAGNOSTICVERIFIER_H
