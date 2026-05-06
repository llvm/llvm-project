//==-- llvm/FileCheck/FileCheck.h --------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file has some utilities to use FileCheck as an API
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FILECHECK_FILECHECK_H
#define LLVM_FILECHECK_FILECHECK_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SMLoc.h"
#include <bitset>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace llvm {
class MemoryBuffer;
class SourceMgr;
template <typename T> class SmallVectorImpl;

/// Contains info about various FileCheck options.
struct FileCheckRequest {
  std::vector<StringRef> CheckPrefixes;
  std::vector<StringRef> CommentPrefixes;
  bool NoCanonicalizeWhiteSpace = false;
  std::vector<StringRef> ImplicitCheckNot;
  std::vector<StringRef> GlobalDefines;
  bool AllowEmptyInput = false;
  bool AllowUnusedPrefixes = false;
  bool MatchFullLines = false;
  bool IgnoreCase = false;
  bool IsDefaultCheckPrefix = false;
  bool EnableVarScope = false;
  bool AllowDeprecatedDagOverlap = false;
  bool Verbose = false;
  bool VerboseVerbose = false;
};

namespace Check {

enum FileCheckKind {
  CheckNone = 0,
  CheckMisspelled,
  CheckPlain,
  CheckNext,
  CheckSame,
  CheckNot,
  CheckDAG,
  CheckLabel,
  CheckEmpty,
  CheckComment,

  /// Indicates the pattern only matches the end of file. This is used for
  /// trailing CHECK-NOTs.
  CheckEOF,

  /// Marks when parsing found a -NOT check combined with another CHECK suffix.
  CheckBadNot,

  /// Marks when parsing found a -COUNT directive with invalid count value.
  CheckBadCount
};

enum FileCheckKindModifier {
  /// Modifies directive to perform literal match.
  ModifierLiteral = 0,

  // The number of modifier.
  Size
};

class FileCheckType {
  FileCheckKind Kind;
  int Count; ///< optional Count for some checks
  /// Modifers for the check directive.
  std::bitset<FileCheckKindModifier::Size> Modifiers;

public:
  FileCheckType(FileCheckKind Kind = CheckNone) : Kind(Kind), Count(1) {}
  FileCheckType(const FileCheckType &) = default;
  FileCheckType &operator=(const FileCheckType &) = default;

  operator FileCheckKind() const { return Kind; }

  int getCount() const { return Count; }
  LLVM_ABI FileCheckType &setCount(int C);

  bool isLiteralMatch() const {
    return Modifiers[FileCheckKindModifier::ModifierLiteral];
  }
  FileCheckType &setLiteralMatch(bool Literal = true) {
    Modifiers.set(FileCheckKindModifier::ModifierLiteral, Literal);
    return *this;
  }

  // \returns a description of \p Prefix.
  LLVM_ABI std::string getDescription(StringRef Prefix) const;

  // \returns a description of \p Modifiers.
  LLVM_ABI std::string getModifiersDescription() const;
};
} // namespace Check

class MatchResultDiag;

/// Abstract base class for recording a FileCheck diagnostic for a pattern
/// (e.g., \c CHECK-NEXT directive or \c --implicit-check-not).
///
/// \c FileCheckDiag has two direct derived classes:
/// - \c MatchResultDiag records a match result for a pattern.  There might be
///   more than one for a single pattern.  For example, for \c CHECK-DAG there
///   might be several discarded matches before either a good match or a failure
///   to match.
/// - \c MatchNoteDiag provides an additional note about the most recent
///   \c MatchResultDiag emitted by a FileCheck invocation.  For example, there
///   might be a fuzzy match after a failure to match.
class FileCheckDiag {
public:
  enum FileCheckDiagKind { FCDK_MatchResultDiag, FCDK_MatchNoteDiag };

  /// What type of match result does this diagnostic describe?
  ///
  /// A directive's supplied pattern is said to be either expected or excluded
  /// depending on whether the pattern must have or must not have a match in
  /// order for the directive to succeed.  For example, a CHECK directive's
  /// pattern is expected, and a CHECK-NOT directive's pattern is excluded.
  enum MatchType {
    /// Indicates a good match for an expected pattern.
    MatchFoundAndExpected,
    /// Indicates a match for an excluded pattern.
    MatchFoundButExcluded,
    /// Indicates a match for an expected pattern, but the match is on the
    /// wrong line.
    MatchFoundButWrongLine,
    /// Indicates a discarded match for an expected pattern.
    MatchFoundButDiscarded,
    /// Indicates an error while processing a match after the match was found
    /// for an expected or excluded pattern.  The error is specified by \c Note,
    /// to which it should be appropriate to prepend "error: " later.  The full
    /// match itself should be recorded in a preceding diagnostic of a different
    /// \c MatchFound match type.
    MatchFoundErrorNote,
    /// Indicates no match for an excluded pattern.
    MatchNoneAndExcluded,
    /// Indicates no match for an expected pattern, but this might follow good
    /// matches when multiple matches are expected for the pattern, or it might
    /// follow discarded matches for the pattern.
    MatchNoneButExpected,
    /// Indicates no match due to an expected or excluded pattern that has
    /// proven to be invalid at match time.  The exact problems are usually
    /// reported in subsequent diagnostics of the same match type but with
    /// \c Note set.
    MatchNoneForInvalidPattern,
    /// Indicates a fuzzy match that serves as a suggestion for the next
    /// intended match for an expected pattern with too few or no good matches.
    MatchFuzzy,
  };

private:
  const FileCheckDiagKind Kind;
  MatchType MatchTy;
  SMRange InputRange;

public:
  FileCheckDiag(FileCheckDiagKind Kind, MatchType MatchTy, SMRange InputRange)
      : Kind(Kind), MatchTy(MatchTy), InputRange(InputRange) {}
  /// Destructor is purely virtual to ensure this remains an abstract class.
  virtual ~FileCheckDiag() = 0;
  /// Of what derived class is this an instance?
  FileCheckDiagKind getKind() const { return Kind; }
  /// If this is a \c MatchResultDiag, return itself.  If this is a
  /// \c MatchNoteDiag, return its associated \c MatchResultDiag.
  virtual const MatchResultDiag &getMatchResultDiag() const = 0;
  /// Adjust the match type.
  void adjustMatchType(MatchType MatchTy) { this->MatchTy = MatchTy; }
  /// Get the match type.
  MatchType getMatchType() const { return MatchTy; }
  /// The search range if MatchTy starts with MatchNone, or the match range
  /// otherwise.
  SMRange getInputRange() const { return InputRange; }
};

/// Class for recording a FileCheck diagnostic that reports a match result for a
/// pattern.
class MatchResultDiag : public FileCheckDiag {
private:
  Check::FileCheckType CheckTy;
  SMLoc CheckLoc;

public:
  MatchResultDiag(const Check::FileCheckType &CheckTy, SMLoc CheckLoc,
                  MatchType MatchTy, SMRange InputRange)
      : FileCheckDiag(FCDK_MatchResultDiag, MatchTy, InputRange),
        CheckTy(CheckTy), CheckLoc(CheckLoc) {}
  /// Is \p FCD an instance of \c MatchResultDiag?
  static bool classof(const FileCheckDiag *FCD) {
    return FCD->getKind() == FCDK_MatchResultDiag;
  }
  /// Get itself.
  const MatchResultDiag &getMatchResultDiag() const override { return *this; }
  /// What is the type of pattern for this match result?
  Check::FileCheckType getCheckTy() const { return CheckTy; }
  /// Where is the pattern for this match result?
  SMLoc getCheckLoc() const { return CheckLoc; }
};

/// Class for recording a FileCheck diagnostic that provides an additional note
/// (possibly an additional error) about the most recent \c MatchResultDiag.
class MatchNoteDiag : public FileCheckDiag {
private:
  MatchResultDiag *MRD;
  std::optional<std::string> CustomNote;

public:
  MatchNoteDiag(MatchType MatchTy, SMRange InputRange,
                std::optional<StringRef> CustomNote = std::nullopt)
      : FileCheckDiag(FCDK_MatchNoteDiag, MatchTy, InputRange), MRD(nullptr),
        CustomNote(CustomNote) {}
  /// Is \p FCD an instance of \c MatchNoteDiag?
  static bool classof(const FileCheckDiag *FCD) {
    return FCD->getKind() == FCDK_MatchNoteDiag;
  }
  /// Get the note's associated \c MatchResultDiag.
  const MatchResultDiag &getMatchResultDiag() const override { return *MRD; }
  /// Set the note's associated \c MatchResultDiag.
  void setMatchResultDiag(MatchResultDiag *MRDNew) {
    assert(!MRD && "expected setMatchResultDiag to be called only once");
    MRD = MRDNew;
  }
  /// A note to replace the one normally indicated by the match type, or the
  /// empty string if none.
  const std::optional<std::string> &getCustomNote() const { return CustomNote; }
};

/// A \c FileCheckDiag series emitted by the FileCheck library.
class FileCheckDiagList {
private:
  MatchResultDiag *CurMatchResultDiag = nullptr;
  using vector_type = std::vector<std::unique_ptr<FileCheckDiag>>;
  vector_type DiagList;

public:
  /// Emplace a new \c FileCheckDiag of type \c DiagTy.  If it's a
  /// \c MatchNoteDiag, associate it with its \c MatchResultDiag.
  ///
  /// \c FileCheckTest.cpp calls \c Pattern::printVariableDefs directly, so it
  /// can add a \c MatchNoteDiag without a previous \c MatchResultDiag.
  /// Otherwise, there should always be a previous \c MatchResultDiag.
  template <typename DiagTy, typename... ArgTys>
  void emplace(ArgTys &&...Args) {
    DiagList.emplace_back(
        std::make_unique<DiagTy>(std::forward<ArgTys>(Args)...));
    FileCheckDiag *Diag = DiagList.back().get();
    if (MatchResultDiag *MRD = dyn_cast<MatchResultDiag>(Diag)) {
      CurMatchResultDiag = MRD;
      return;
    }
    MatchNoteDiag *Note = cast<MatchNoteDiag>(Diag);
    if (!CurMatchResultDiag)
      return;
    Note->setMatchResultDiag(CurMatchResultDiag);
  }
  /// Adjust the most recent \c MatchResultDiag, which must exist, and every
  /// \c MatchResultNote after it to have the match type \c MatchTy.
  void adjustPrevDiags(FileCheckDiag::MatchType MatchTy) {
    assert(CurMatchResultDiag && "expected previous MatchResultDiag");
    for (auto I = DiagList.rbegin(), E = DiagList.rend();
         I != E && &**I != CurMatchResultDiag; ++I)
      (*I)->adjustMatchType(MatchTy);
    CurMatchResultDiag->adjustMatchType(MatchTy);
  }
  class const_iterator {
    friend FileCheckDiagList;

  public:
    using difference_type = std::ptrdiff_t;
    using value_type = FileCheckDiag;
    using pointer = const FileCheckDiag *;
    using reference = const FileCheckDiag &;
    using iterator_category = std::forward_iterator_tag;

  private:
    vector_type::const_iterator Itr;
    const_iterator(vector_type::const_iterator Itr) : Itr(Itr) {}

  public:
    reference operator*() const { return **Itr; }
    pointer operator->() const { return &operator*(); }
    const_iterator &operator++() {
      ++Itr;
      return *this;
    }
    const_iterator operator++(int) {
      const_iterator Old = *this;
      ++Itr;
      return Old;
    }
    bool operator==(const const_iterator &Other) const {
      return Itr == Other.Itr;
    }
    bool operator!=(const const_iterator &Other) const {
      return Itr != Other.Itr;
    }
  };

  using size_type = vector_type::size_type;
  const_iterator begin() const { return const_iterator(DiagList.begin()); }
  const_iterator end() const { return const_iterator(DiagList.end()); }
  const FileCheckDiag &operator[](size_type I) const { return *DiagList[I]; }
  size_type size() const { return DiagList.size(); }
};

class FileCheckPatternContext;
struct FileCheckString;

/// FileCheck class takes the request and exposes various methods that
/// use information from the request.
class FileCheck {
  FileCheckRequest Req;
  std::unique_ptr<FileCheckPatternContext> PatternContext;
  std::vector<FileCheckString> CheckStrings;

public:
  LLVM_ABI explicit FileCheck(FileCheckRequest Req);
  LLVM_ABI ~FileCheck();

  /// Reads the check file from \p Buffer and records the expected strings it
  /// contains. Errors are reported against \p SM.
  ///
  /// If \p ImpPatBufferIDRange, then the range (inclusive start, exclusive end)
  /// of IDs for source buffers added to \p SM for implicit patterns are
  /// recorded in it.  The range is empty if there are none.
  LLVM_ABI bool
  readCheckFile(SourceMgr &SM, StringRef Buffer,
                std::pair<unsigned, unsigned> *ImpPatBufferIDRange = nullptr);

  LLVM_ABI bool ValidateCheckPrefixes();

  /// Canonicalizes whitespaces in the file. Line endings are replaced with
  /// UNIX-style '\n'.
  LLVM_ABI StringRef CanonicalizeFile(MemoryBuffer &MB,
                                      SmallVectorImpl<char> &OutputBuffer);

  /// Checks the input to FileCheck provided in the \p Buffer against the
  /// expected strings read from the check file and record diagnostics emitted
  /// in \p Diags. Errors are recorded against \p SM.
  ///
  /// \returns false if the input fails to satisfy the checks.
  LLVM_ABI bool checkInput(SourceMgr &SM, StringRef Buffer,
                           FileCheckDiagList *Diags = nullptr);
};

} // namespace llvm

#endif
