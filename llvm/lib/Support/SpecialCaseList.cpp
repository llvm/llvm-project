//===-- SpecialCaseList.cpp - special case list for sanitizers ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a utility class for instrumentation passes (like AddressSanitizer
// or ThreadSanitizer) to avoid instrumenting some functions or global
// variables, or to instrument some functions or global variables in a specific
// way, based on a user-supplied list.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SpecialCaseList.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <algorithm>
#include <limits>
#include <memory>
#include <stdio.h>
#include <string>
#include <system_error>
#include <utility>

namespace llvm {

Error SpecialCaseList::RegexMatcher::insert(StringRef Pattern,
                                            unsigned LineNumber) {
  if (Pattern.empty())
    return createStringError(errc::invalid_argument,
                             "Supplied regex was blank");

  // Replace * with .*
  auto Regexp = Pattern.str();
  for (size_t pos = 0; (pos = Regexp.find('*', pos)) != std::string::npos;
       pos += strlen(".*")) {
    Regexp.replace(pos, strlen("*"), ".*");
  }

  Regexp = (Twine("^(") + StringRef(Regexp) + ")$").str();

  // Check that the regexp is valid.
  Regex CheckRE(Regexp);
  std::string REError;
  if (!CheckRE.isValid(REError))
    return createStringError(errc::invalid_argument, REError);

  RegExes.emplace_back(Pattern, LineNumber, std::move(CheckRE));
  return Error::success();
}

void SpecialCaseList::RegexMatcher::preprocess(bool BySize) {
  if (BySize) {
    llvm::stable_sort(RegExes, [](const Reg &A, const Reg &B) {
      return A.Name.size() < B.Name.size();
    });
  }
}

void SpecialCaseList::RegexMatcher::match(
    StringRef Query,
    llvm::function_ref<void(StringRef Rule, unsigned LineNo)> Cb) const {
  for (const auto &R : reverse(RegExes))
    if (R.Rg.match(Query))
      return Cb(R.Name, R.LineNo);
}

Error SpecialCaseList::GlobMatcher::insert(StringRef Pattern,
                                           unsigned LineNumber) {
  if (Pattern.empty())
    return createStringError(errc::invalid_argument, "Supplied glob was blank");

  auto Res = GlobPattern::create(Pattern, /*MaxSubPatterns=*/1024);
  if (auto Err = Res.takeError())
    return Err;
  Globs.emplace_back(Pattern, LineNumber, std::move(Res.get()));
  return Error::success();
}

void SpecialCaseList::GlobMatcher::preprocess(bool BySize) {
  if (BySize) {
    llvm::stable_sort(Globs, [](const Glob &A, const Glob &B) {
      return A.Name.size() < B.Name.size();
    });
  }
}

void SpecialCaseList::GlobMatcher::match(
    StringRef Query,
    llvm::function_ref<void(StringRef Rule, unsigned LineNo)> Cb) const {
  for (const auto &G : reverse(Globs))
    if (G.Pattern.match(Query))
      return Cb(G.Name, G.LineNo);
}

SpecialCaseList::Matcher::Matcher(bool UseGlobs, bool RemoveDotSlash)
    : RemoveDotSlash(RemoveDotSlash) {
  if (UseGlobs)
    M.emplace<GlobMatcher>();
  else
    M.emplace<RegexMatcher>();
}

Error SpecialCaseList::Matcher::insert(StringRef Pattern, unsigned LineNumber) {
  return std::visit([&](auto &V) { return V.insert(Pattern, LineNumber); }, M);
}

LLVM_ABI void SpecialCaseList::Matcher::preprocess(bool BySize) {
  return std::visit([&](auto &V) { return V.preprocess(BySize); }, M);
}

void SpecialCaseList::Matcher::match(
    StringRef Query,
    llvm::function_ref<void(StringRef Rule, unsigned LineNo)> Cb) const {
  if (RemoveDotSlash)
    Query = llvm::sys::path::remove_leading_dotslash(Query);
  return std::visit([&](auto &V) { return V.match(Query, Cb); }, M);
}

// TODO: Refactor this to return Expected<...>
std::unique_ptr<SpecialCaseList>
SpecialCaseList::create(const std::vector<std::string> &Paths,
                        llvm::vfs::FileSystem &FS, std::string &Error) {
  std::unique_ptr<SpecialCaseList> SCL(new SpecialCaseList());
  if (SCL->createInternal(Paths, FS, Error))
    return SCL;
  return nullptr;
}

std::unique_ptr<SpecialCaseList> SpecialCaseList::create(const MemoryBuffer *MB,
                                                         std::string &Error) {
  std::unique_ptr<SpecialCaseList> SCL(new SpecialCaseList());
  if (SCL->createInternal(MB, Error))
    return SCL;
  return nullptr;
}

std::unique_ptr<SpecialCaseList>
SpecialCaseList::createOrDie(const std::vector<std::string> &Paths,
                             llvm::vfs::FileSystem &FS) {
  std::string Error;
  if (auto SCL = create(Paths, FS, Error))
    return SCL;
  report_fatal_error(Twine(Error));
}

bool SpecialCaseList::createInternal(const std::vector<std::string> &Paths,
                                     vfs::FileSystem &VFS, std::string &Error) {
  for (size_t i = 0; i < Paths.size(); ++i) {
    const auto &Path = Paths[i];
    ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
        VFS.getBufferForFile(Path);
    if (std::error_code EC = FileOrErr.getError()) {
      Error = (Twine("can't open file '") + Path + "': " + EC.message()).str();
      return false;
    }
    std::string ParseError;
    if (!parse(i, FileOrErr.get().get(), ParseError, /*OrderBySize=*/false)) {
      Error = (Twine("error parsing file '") + Path + "': " + ParseError).str();
      return false;
    }
  }
  return true;
}

bool SpecialCaseList::createInternal(const MemoryBuffer *MB, std::string &Error,
                                     bool OrderBySize) {
  if (!parse(0, MB, Error, OrderBySize))
    return false;
  return true;
}

Expected<SpecialCaseList::Section *>
SpecialCaseList::addSection(StringRef SectionStr, unsigned FileNo,
                            unsigned LineNo, bool UseGlobs) {
  Sections.emplace_back(SectionStr, FileNo, UseGlobs);
  auto &Section = Sections.back();

  SectionStr = SectionStr.copy(StrAlloc);
  if (auto Err = Section.SectionMatcher.insert(SectionStr, LineNo)) {
    return createStringError(errc::invalid_argument,
                             "malformed section at line " + Twine(LineNo) +
                                 ": '" + SectionStr +
                                 "': " + toString(std::move(Err)));
  }

  return &Section;
}

bool SpecialCaseList::parse(unsigned FileIdx, const MemoryBuffer *MB,
                            std::string &Error, bool OrderBySize) {
  unsigned long long Version = 2;

  StringRef Header = MB->getBuffer();
  if (Header.consume_front("#!special-case-list-v"))
    consumeUnsignedInteger(Header, 10, Version);

  // In https://reviews.llvm.org/D154014 we added glob support and planned
  // to remove regex support in patterns. We temporarily support the
  // original behavior using regexes if "#!special-case-list-v1" is the
  // first line of the file. For more details, see
  // https://discourse.llvm.org/t/use-glob-instead-of-regex-for-specialcaselists/71666
  bool UseGlobs = Version > 1;

  bool RemoveDotSlash = Version > 2;

  Section *CurrentSection;
  if (auto Err = addSection("*", FileIdx, 1, true).moveInto(CurrentSection)) {
    Error = toString(std::move(Err));
    return false;
  }

  // This is the current list of prefixes for all existing users matching file
  // path. We may need parametrization in constructor in future.
  constexpr StringRef PathPrefixes[] = {"src", "!src", "mainfile", "source"};

  for (line_iterator LineIt(*MB, /*SkipBlanks=*/true, /*CommentMarker=*/'#');
       !LineIt.is_at_eof(); LineIt++) {
    unsigned LineNo = LineIt.line_number();
    StringRef Line = LineIt->trim();
    if (Line.empty())
      continue;

    // Save section names
    if (Line.starts_with("[")) {
      if (!Line.ends_with("]")) {
        Error =
            ("malformed section header on line " + Twine(LineNo) + ": " + Line)
                .str();
        return false;
      }

      if (auto Err = addSection(Line.drop_front().drop_back(), FileIdx, LineNo,
                                UseGlobs)
                         .moveInto(CurrentSection)) {
        Error = toString(std::move(Err));
        return false;
      }
      continue;
    }

    // Get our prefix and unparsed glob.
    auto [Prefix, Postfix] = Line.split(":");
    if (Postfix.empty()) {
      // Missing ':' in the line.
      Error = ("malformed line " + Twine(LineNo) + ": '" + Line + "'").str();
      return false;
    }

    auto [Pattern, Category] = Postfix.split("=");
    auto [It, _] = CurrentSection->Entries[Prefix].try_emplace(
        Category, UseGlobs,
        RemoveDotSlash && llvm::is_contained(PathPrefixes, Prefix));
    Pattern = Pattern.copy(StrAlloc);
    if (auto Err = It->second.insert(Pattern, LineNo)) {
      Error =
          (Twine("malformed ") + (UseGlobs ? "glob" : "regex") + " in line " +
           Twine(LineNo) + ": '" + Pattern + "': " + toString(std::move(Err)))
              .str();
      return false;
    }
  }

  for (Section &S : Sections)
    S.preprocess(OrderBySize);

  return true;
}

SpecialCaseList::~SpecialCaseList() = default;

bool SpecialCaseList::inSection(StringRef Section, StringRef Prefix,
                                StringRef Query, StringRef Category) const {
  auto [FileIdx, LineNo] = inSectionBlame(Section, Prefix, Query, Category);
  return LineNo;
}

std::pair<unsigned, unsigned>
SpecialCaseList::inSectionBlame(StringRef Section, StringRef Prefix,
                                StringRef Query, StringRef Category) const {
  for (const auto &S : reverse(Sections)) {
    if (S.SectionMatcher.matchAny(Section)) {
      unsigned Blame = S.getLastMatch(Prefix, Query, Category);
      if (Blame)
        return {S.FileIdx, Blame};
    }
  }
  return NotFound;
}

const SpecialCaseList::Matcher *
SpecialCaseList::Section::findMatcher(StringRef Prefix,
                                      StringRef Category) const {
  SectionEntries::const_iterator I = Entries.find(Prefix);
  if (I == Entries.end())
    return nullptr;
  StringMap<Matcher>::const_iterator II = I->second.find(Category);
  if (II == I->second.end())
    return nullptr;

  return &II->second;
}

LLVM_ABI void SpecialCaseList::Section::preprocess(bool OrderBySize) {
  SectionMatcher.preprocess(false);
  for (auto &[K1, E] : Entries)
    for (auto &[K2, M] : E)
      M.preprocess(OrderBySize);
}

unsigned SpecialCaseList::Section::getLastMatch(StringRef Prefix,
                                                StringRef Query,
                                                StringRef Category) const {
  unsigned LastLine = 0;
  if (const Matcher *M = findMatcher(Prefix, Category)) {
    M->match(Query, [&](StringRef, unsigned LineNo) {
      LastLine = std::max(LastLine, LineNo);
    });
  }
  return LastLine;
}

StringRef SpecialCaseList::Section::getLongestMatch(StringRef Prefix,
                                                    StringRef Query,
                                                    StringRef Category) const {
  StringRef LongestRule;
  if (const Matcher *M = findMatcher(Prefix, Category)) {
    M->match(Query, [&](StringRef Rule, unsigned) {
      if (LongestRule.size() < Rule.size())
        LongestRule = Rule;
    });
  }
  return LongestRule;
}

} // namespace llvm
