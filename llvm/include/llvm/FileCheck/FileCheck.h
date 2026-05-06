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
///
/// Throughout this class hierarchy, a pattern is said to be either expected or
/// excluded depending on whether the pattern must have or must not have a match
/// in order for it to succeed.  For example, a \c CHECK directive's pattern is
/// expected, and a \c CHECK-NOT directive's pattern is excluded.
class FileCheckDiag {
public:
  enum FileCheckDiagKind {
    // MatchResultDiag
    MatchResultDiag_First,
    MatchFoundDiag = MatchResultDiag_First,
    MatchNoneDiag,
    MatchResultDiag_Last = MatchNoneDiag,
    // MatchNoteDiag
    MatchNoteDiag_First,
    MatchFuzzyDiag = MatchNoteDiag_First,
    MatchCustomNoteDiag,
    MatchNoteDiag_Last = MatchCustomNoteDiag
  };

private:
  const FileCheckDiagKind Kind;

public:
  FileCheckDiag(FileCheckDiagKind Kind) : Kind(Kind) {}
  /// Destructor is purely virtual to ensure this remains an abstract class.
  virtual ~FileCheckDiag() = 0;
  /// Of what derived class is this an instance?
  FileCheckDiagKind getKind() const { return Kind; }
  /// If this is a \c MatchResultDiag, return itself.  If this is a
  /// \c MatchNoteDiag, return its associated \c MatchResultDiag.
  virtual const MatchResultDiag &getMatchResultDiag() const = 0;
  /// Does this diagnostic reveal a new error?
  ///
  /// For \c MatchResultDiag, \c !isError() is not always the same as a
  /// successful pattern match result.  For \c MatchNoteDiag, \c !isError()
  /// does not indicate the lack of an error but rather the lack of an
  /// additional error beyond its associated \c MatchResultDiag.  See
  /// documentation on derived types for details.
  virtual bool isError() const = 0;
  /// Return the input range for which this diagnostic indicates text that was
  /// matched in some way (e.g., successful pattern match, discarded pattern
  /// match, or variable capture), or return \c std::nullopt if the diagnostic
  /// has no such input range.
  virtual std::optional<SMRange> getMatchRange() const = 0;
};

/// Abstract base class for recording a FileCheck diagnostic that reports a
/// match result for a pattern.
class MatchResultDiag : public FileCheckDiag {
private:
  Check::FileCheckType CheckTy;
  SMLoc CheckLoc;
  SMRange SearchRange;

public:
  MatchResultDiag(FileCheckDiagKind Kind, const Check::FileCheckType &CheckTy,
                  SMLoc CheckLoc, SMRange SearchRange)
      : FileCheckDiag(Kind), CheckTy(CheckTy), CheckLoc(CheckLoc),
        SearchRange(SearchRange) {}
  /// Destructor is purely virtual to ensure this remains an abstract class.
  virtual ~MatchResultDiag() = 0;
  /// Is \p FCD an instance of \c MatchResultDiag?
  static bool classof(const FileCheckDiag *FCD) {
    FileCheckDiagKind Kind = FCD->getKind();
    return MatchResultDiag_First <= Kind && Kind <= MatchResultDiag_Last;
  }
  /// Get itself.
  const MatchResultDiag &getMatchResultDiag() const override { return *this; }
  /// What is the type of pattern for this match result?
  Check::FileCheckType getCheckTy() const { return CheckTy; }
  /// Where is the pattern for this match result?
  SMLoc getCheckLoc() const { return CheckLoc; }
  /// What is the search range for the match result?
  SMRange getSearchRange() const { return SearchRange; }
};

/// \c MatchResultDiag for a pattern that matched the input.
class MatchFoundDiag : public MatchResultDiag {
public:
  enum StatusTy {
    /// Indicates a good match for an expected pattern.
    Success,
    /// Indicates a match for an excluded pattern (error).
    Excluded,
    /// Indicates a match for an expected pattern, but the match is on the
    /// wrong line (error).
    WrongLine,
    /// Indicates a discarded match for an expected pattern (not an error).
    Discarded
  };

private:
  StatusTy Status;
  SMRange MatchRange;

public:
  MatchFoundDiag(const Check::FileCheckType &CheckTy, SMLoc CheckLoc,
                 StatusTy Status, SMRange MatchRange, SMRange SearchRange)
      : MatchResultDiag(FileCheckDiag::MatchFoundDiag, CheckTy, CheckLoc,
                        SearchRange),
        Status(Status), MatchRange(MatchRange) {}
  /// Is \p FCD an instance of \c MatchFoundDiag?
  static bool classof(const FileCheckDiag *FCD) {
    return FCD->getKind() == FileCheckDiag::MatchFoundDiag;
  }
  /// Does this match produce an error?
  ///
  /// This is not always the same as \c getStatus()!=Success.  For example,
  /// \c CHECK-DAG discarded matches are neither successful matches nor errors.
  bool isError() const override {
    return Status != Success && Status != Discarded;
  }
  /// Was this a successful match?  If not, why not?
  ///
  /// See \c isError comments for the relationship between the two.
  StatusTy getStatus() const { return Status; }
  /// Adjust a successful status to a non-successful status.
  ///
  /// This is designed to be called while emitting diagnostics.  It is not
  /// designed to be called by a diagnostic presentation layer like
  /// `-dump-input`.
  ///
  /// For example, a match that was originally thought to be successful might
  /// later be discarded, or it might be determined that it violates a matching
  /// constraint (e.g., wrong line).
  void markUnsuccessful(StatusTy S) {
    assert(Status == Success && S != Success &&
           "expected to change successful status to unsuccessful");
    Status = S;
  }
  /// Return the match's input range, never \c std::nullopt.
  std::optional<SMRange> getMatchRange() const override { return MatchRange; }
};

/// \c MatchResultDiag for a pattern that did not match the input.
class MatchNoneDiag : public MatchResultDiag {
public:
  enum StatusTy {
    /// Indicates no match for an excluded pattern.
    Success,
    /// Indicates no match due to an expected or excluded pattern that has
    /// proven to be invalid at match time (error).  The exact problems are
    /// usually reported in subsequent \c MatchNoteDiag objects.
    InvalidPattern,
    /// Indicates no match for an expected pattern (error).  In some cases, it
    /// follows good matches (because multiple matches are expected) or
    /// discarded matches for the pattern.
    Expected
  };

private:
  StatusTy Status;

public:
  MatchNoneDiag(const Check::FileCheckType &CheckTy, SMLoc CheckLoc,
                StatusTy Status, SMRange SearchRange)
      : MatchResultDiag(FileCheckDiag::MatchNoneDiag, CheckTy, CheckLoc,
                        SearchRange),
        Status(Status) {}
  /// Is \p FCD an instance of \c MatchNoneDiag?
  static bool classof(const FileCheckDiag *FCD) {
    return FCD->getKind() == FileCheckDiag::MatchNoneDiag;
  }
  /// Does the lack of match represent an error?
  bool isError() const override { return Status != Success; }
  /// Does the lack of a match indicate a success?  If not, why not?
  StatusTy getStatus() const { return Status; }
  /// Return \c std::nullopt.
  std::optional<SMRange> getMatchRange() const override { return std::nullopt; }
};

/// Abstract base class for recording a FileCheck diagnostic that provides an
/// additional note (possibly a new error) about the most recent
/// \c MatchResultDiag.
class MatchNoteDiag : public FileCheckDiag {
private:
  MatchResultDiag *MRD;

public:
  MatchNoteDiag(FileCheckDiagKind Kind) : FileCheckDiag(Kind), MRD(nullptr) {}
  /// Destructor is purely virtual to ensure this remains an abstract class.
  virtual ~MatchNoteDiag() = 0;
  /// Is \p FCD an instance of \c MatchNoteDiag?
  static bool classof(const FileCheckDiag *FCD) {
    FileCheckDiagKind Kind = FCD->getKind();
    return MatchNoteDiag_First <= Kind && Kind <= MatchNoteDiag_Last;
  }
  /// Get the note's associated \c MatchResultDiag.
  const MatchResultDiag &getMatchResultDiag() const override { return *MRD; }
  /// Set the note's associated \c MatchResultDiag.
  void setMatchResultDiag(MatchResultDiag *MRDNew) {
    assert(!MRD && "expected setMatchResultDiag to be called only once");
    MRD = MRDNew;
  }
};

/// \c MatchNoteDiag for a fuzzy match that serves as a suggestion for the next
/// intended match for an expected pattern with too few or no good matches.
class MatchFuzzyDiag : public MatchNoteDiag {
private:
  SMLoc MatchStart;

public:
  MatchFuzzyDiag(SMLoc MatchStart)
      : MatchNoteDiag(FileCheckDiag::MatchFuzzyDiag), MatchStart(MatchStart) {}
  /// Is \p FCD an instance of \c MatchFuzzyDiag?
  static bool classof(const FileCheckDiag *FCD) {
    return FCD->getKind() == FileCheckDiag::MatchFuzzyDiag;
  }
  /// Always false.  A fuzzy match is not an error even though it is performed
  /// due to an error.
  bool isError() const override { return false; }
  /// Return an input range (never \c std::nullopt) starting and ending at the
  /// match start.  The actual match end is not computed.
  std::optional<SMRange> getMatchRange() const override {
    return SMRange(MatchStart, MatchStart);
  }
};

/// \c MatchNoteDiag with a custom note not described by any other class derived
/// from \c MatchNoteDiag.
class MatchCustomNoteDiag : public MatchNoteDiag {
private:
  std::string Note;
  bool AddsError;
  std::optional<SMRange> MatchRange;

public:
  /// If \p MatchRange is specified, it is a range for input text that was
  /// matched in some way (e.g., variable capture) and that is described by
  /// this note.  Either way, as usual, the associated \c MatchResultDiag has
  /// any full match range for the pattern.
  ///
  /// If \p AddsError is true, then this note indicates a \a new error that is
  /// distinct from any error indicated by the associated \c MatchResultDiag.
  /// The error is described by \c Note, which must be worded appropriately for
  /// prepending "error: " when presented later.  For example, the associated
  /// \c MatchResultDiag might indicate a match to either an expected pattern
  /// (success) or an excluded pattern (error), and \c Note might be "unable to
  /// represent numeric value" to indicate the match could not be processed
  /// afterward.
  ///
  /// If \p AddsError is false, then this note merely provides additional
  /// information about the associated \c MatchResultDiag.  That information
  /// might be something harmless (e.g., variable substitution), or it might be
  /// one of potentially many problems summarized as an error by the
  /// \c MatchResultDiag (e.g., one way in which the pattern was invalid).
  ///@{
  MatchCustomNoteDiag(SMRange MatchRange, StringRef Note,
                      bool AddsError = false)
      : MatchNoteDiag(FileCheckDiag::MatchCustomNoteDiag), Note(Note),
        AddsError(AddsError), MatchRange(MatchRange) {}
  MatchCustomNoteDiag(StringRef Note)
      : MatchNoteDiag(FileCheckDiag::MatchCustomNoteDiag), Note(Note),
        AddsError(false) {}
  ///@}
  /// Is \p FCD an instance of \c MatchCustomNoteDiag?
  static bool classof(const FileCheckDiag *FCD) {
    return FCD->getKind() == FileCheckDiag::MatchCustomNoteDiag;
  }
  const std::string &getNote() const { return Note; }
  /// Does this note indicate an \a additional error not indicated by the
  /// associated \c MatchResultDiag?
  ///
  /// For details, see the \c MatchCustomNoteDiag::MatchCustomNoteDiag comments
  /// for its \c AddsError parameter.
  bool isError() const override { return AddsError; }
  /// Return the match range described by the note, or \c std::nullopt if none.
  std::optional<SMRange> getMatchRange() const override { return MatchRange; }
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
  /// Adjust the previous \c MatchResultDiag, which must be a \c MatchFoundDiag,
  /// from successful status to unsuccessful status.
  void adjustPrevMatchFoundDiag(MatchFoundDiag::StatusTy Status) {
    cast<MatchFoundDiag>(CurMatchResultDiag)->markUnsuccessful(Status);
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
