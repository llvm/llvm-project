//===-- SpecialCaseList.h - special case list for sanitizers ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file implements a Special Case List for code sanitizers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_SPECIALCASELIST_H
#define LLVM_SUPPORT_SPECIALCASELIST_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/GlobPattern.h"
#include "llvm/Support/Regex.h"
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace llvm {
class MemoryBuffer;
class StringRef;

namespace vfs {
class FileSystem;
}

/// This is a utility class used to parse user-provided text files with
/// "special case lists" for code sanitizers. Such files are used to
/// define an "ABI list" for DataFlowSanitizer and allow/exclusion lists for
/// sanitizers like AddressSanitizer or UndefinedBehaviorSanitizer.
///
/// Empty lines and lines starting with "#" are ignored. Sections are defined
/// using a '[section_name]' header and can be used to specify sanitizers the
/// entries below it apply to. Section names are globs, and
/// entries without a section header match all sections (e.g. an '[*]' header
/// is assumed.)
/// The remaining lines should have the form:
///   prefix:glob_pattern[=category]
/// If category is not specified, it is assumed to be empty string.
/// Definitions of "prefix" and "category" are sanitizer-specific. For example,
/// sanitizer exclusion support prefixes "src", "mainfile", "fun" and "global".
/// "glob_pattern" defines source files, main files, functions or globals which
/// shouldn't be instrumented.
/// Examples of categories:
///   "functional": used in DFSan to list functions with pure functional
///                 semantics.
///   "init": used in ASan exclusion list to disable initialization-order bugs
///           detection for certain globals or source files.
/// Full special case list file example:
/// ---
/// [address]
/// # Excluded items:
/// fun:*_ZN4base6subtle*
/// global:*global_with_bad_access_or_initialization*
/// global:*global_with_initialization_issues*=init
/// type:*Namespace::ClassName*=init
/// src:file_with_tricky_code.cc
/// src:ignore-global-initializers-issues.cc=init
/// mainfile:main_file.cc
///
/// [dataflow]
/// # Functions with pure functional semantics:
/// fun:cos=functional
/// fun:sin=functional
/// ---
class SpecialCaseList {
public:
  static constexpr std::pair<unsigned, unsigned> NotFound = {0, 0};
  /// Parses the special case list entries from files. On failure, returns
  /// 0 and writes an error message to string.
  LLVM_ABI static std::unique_ptr<SpecialCaseList>
  create(const std::vector<std::string> &Paths, llvm::vfs::FileSystem &FS,
         std::string &Error);
  /// Parses the special case list from a memory buffer. On failure, returns
  /// 0 and writes an error message to string.
  LLVM_ABI static std::unique_ptr<SpecialCaseList>
  create(const MemoryBuffer *MB, std::string &Error);
  /// Parses the special case list entries from files. On failure, reports a
  /// fatal error.
  LLVM_ABI static std::unique_ptr<SpecialCaseList>
  createOrDie(const std::vector<std::string> &Paths, llvm::vfs::FileSystem &FS);

  LLVM_ABI ~SpecialCaseList();

  /// Returns true, if special case list contains a line
  /// \code
  ///   @Prefix:<E>=@Category
  /// \endcode
  /// where @Query satisfies the glob <E> in a given @Section.
  LLVM_ABI bool inSection(StringRef Section, StringRef Prefix, StringRef Query,
                          StringRef Category = StringRef()) const;

  /// Returns the file index and the line number <FileIdx, LineNo> corresponding
  /// to the special case list entry if the special case list contains a line
  /// \code
  ///   @Prefix:<E>=@Category
  /// \endcode
  /// where @Query satisfies the glob <E> in a given @Section.
  /// Returns (zero, zero) if there is no exclusion entry corresponding to this
  /// expression.
  LLVM_ABI std::pair<unsigned, unsigned>
  inSectionBlame(StringRef Section, StringRef Prefix, StringRef Query,
                 StringRef Category = StringRef()) const;

protected:
  // Implementations of the create*() functions that can also be used by derived
  // classes.
  LLVM_ABI bool createInternal(const std::vector<std::string> &Paths,
                               vfs::FileSystem &VFS, std::string &Error);
  LLVM_ABI bool createInternal(const MemoryBuffer *MB, std::string &Error);

  SpecialCaseList() = default;
  SpecialCaseList(SpecialCaseList const &) = delete;
  SpecialCaseList &operator=(SpecialCaseList const &) = delete;

private:
  // Lagacy v1 matcher.
  class RegexMatcher {
  public:
    LLVM_ABI Error insert(StringRef Pattern, unsigned LineNumber);
    LLVM_ABI void
    match(StringRef Query,
          llvm::function_ref<void(StringRef Rule, unsigned LineNo)> Cb) const;

    struct Reg {
      Reg(StringRef Name, unsigned LineNo, Regex &&Rg)
          : Name(Name), LineNo(LineNo), Rg(std::move(Rg)) {}
      std::string Name;
      unsigned LineNo;
      Regex Rg;
      Reg(Reg &&) = delete;
      Reg() = default;
    };

    std::vector<std::unique_ptr<Reg>> RegExes;
  };

  class GlobMatcher {
  public:
    LLVM_ABI Error insert(StringRef Pattern, unsigned LineNumber);
    LLVM_ABI void
    match(StringRef Query,
          llvm::function_ref<void(StringRef Rule, unsigned LineNo)> Cb) const;

    struct Glob {
      Glob(StringRef Name, unsigned LineNo) : Name(Name), LineNo(LineNo) {}
      std::string Name;
      unsigned LineNo;
      GlobPattern Pattern;
      // neither copyable nor movable because GlobPattern contains
      // Glob::StringRef that points to Glob::Name.
      Glob(Glob &&) = delete;
      Glob() = default;
    };

    std::vector<std::unique_ptr<Glob>> Globs;
  };

  /// Represents a set of patterns and their line numbers
  class Matcher {
  public:
    LLVM_ABI Matcher(bool UseGlobs, bool RemoveDotSlash);

    LLVM_ABI void
    match(StringRef Query,
          llvm::function_ref<void(StringRef Rule, unsigned LineNo)> Cb) const;

    LLVM_ABI bool matchAny(StringRef Query) const {
      bool R = false;
      match(Query, [&](StringRef, unsigned) { R = true; });
      return R;
    }

    LLVM_ABI Error insert(StringRef Pattern, unsigned LineNumber);

    std::variant<RegexMatcher, GlobMatcher> M;
    bool RemoveDotSlash;
  };

  using SectionEntries = StringMap<StringMap<Matcher>>;

protected:
  struct Section {
    Section(StringRef Str, unsigned FileIdx, bool UseGlobs)
        : SectionMatcher(UseGlobs, /*RemoveDotSlash=*/false), SectionStr(Str),
          FileIdx(FileIdx) {}

    Section(Section &&) = default;

    Matcher SectionMatcher;
    SectionEntries Entries;
    std::string SectionStr;
    unsigned FileIdx;

    // Helper method to search by Prefix, Query, and Category. Returns
    // 1-based line number on which rule is defined, or 0 if there is no match.
    LLVM_ABI unsigned getLastMatch(StringRef Prefix, StringRef Query,
                                   StringRef Category) const;

    // Helper method to search by Prefix, Query, and Category. Returns
    // matching rule, or empty string if there is no match.
    LLVM_ABI StringRef getLongestMatch(StringRef Prefix, StringRef Query,
                                       StringRef Category) const;

  private:
    LLVM_ABI const SpecialCaseList::Matcher *
    findMatcher(StringRef Prefix, StringRef Category) const;
  };

  ArrayRef<const Section> sections() const { return Sections; }

private:
  std::vector<Section> Sections;

  LLVM_ABI Expected<Section *> addSection(StringRef SectionStr,
                                          unsigned FileIdx, unsigned LineNo,
                                          bool UseGlobs);

  /// Parses just-constructed SpecialCaseList entries from a memory buffer.
  LLVM_ABI bool parse(unsigned FileIdx, const MemoryBuffer *MB,
                      std::string &Error);
};

} // namespace llvm

#endif // LLVM_SUPPORT_SPECIALCASELIST_H
