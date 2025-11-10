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
#include "llvm/ADT/RadixTree.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/GlobPattern.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/WithColor.h"
#include <cassert>
#include <memory>
#include <mutex>
#include <stdio.h>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace llvm {

namespace {

// Lagacy v1 matcher.
class RegexMatcher {
public:
  Error insert(StringRef Pattern, unsigned LineNumber);
  unsigned match(StringRef Query) const;
  StringRef findRule(unsigned LineNo) const;

private:
  struct Reg {
    Reg(StringRef Name, unsigned LineNo, Regex &&Rg)
        : Name(Name), LineNo(LineNo), Rg(std::move(Rg)) {}
    StringRef Name;
    unsigned LineNo;
    Regex Rg;
  };

  std::vector<Reg> RegExes;
};

class GlobMatcher {
public:
  Error insert(StringRef Pattern, unsigned LineNumber);
  unsigned match(StringRef Query) const;
  StringRef findRule(unsigned LineNo) const;

private:
  struct Glob {
    Glob(StringRef Name, unsigned LineNo, GlobPattern &&Pattern)
        : Name(Name), LineNo(LineNo), Pattern(std::move(Pattern)) {}
    StringRef Name;
    unsigned LineNo;
    GlobPattern Pattern;
  };

  void LazyInit() const;

  std::vector<GlobMatcher::Glob> Globs;

  mutable RadixTree<iterator_range<StringRef::const_iterator>,
                    RadixTree<iterator_range<StringRef::const_reverse_iterator>,
                              SmallVector<int, 1>>>
      PrefixSuffixToGlob;

  mutable RadixTree<iterator_range<StringRef::const_iterator>,
                    SmallVector<int, 1>>
      SubstrToGlob;

  mutable bool Initialized = false;
};

/// Represents a set of patterns and their line numbers
class Matcher {
public:
  enum class QueryOptions {
    kUnchanged,
    kNoSlashDot,
    kNoSlashDotWithFallback,
  };

  Matcher(bool UseGlobs, QueryOptions QOpts = QueryOptions::kUnchanged);

  Error insert(StringRef Pattern, unsigned LineNumber);
  unsigned match(StringRef Query) const;

  bool matchAny(StringRef Query) const { return match(Query); }

private:
  unsigned matchInternal(StringRef Query) const;
  StringRef findRule(unsigned LineNo) const;

  std::variant<RegexMatcher, GlobMatcher> M;
  QueryOptions Options;
};

Error RegexMatcher::insert(StringRef Pattern, unsigned LineNumber) {
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

unsigned RegexMatcher::match(StringRef Query) const {
  for (const auto &R : reverse(RegExes))
    if (R.Rg.match(Query))
      return R.LineNo;
  return 0;
}

StringRef RegexMatcher::findRule(unsigned LineNo) const {
  for (const auto &R : RegExes)
    if (R.LineNo == LineNo)
      return R.Name;
  assert(!"`findRule` should be called only with correct `LineNo`");
  return {};
}

Error GlobMatcher::insert(StringRef Pattern, unsigned LineNumber) {
  if (Pattern.empty())
    return createStringError(errc::invalid_argument, "Supplied glob was blank");

  auto Res = GlobPattern::create(Pattern, /*MaxSubPatterns=*/1024);
  if (auto Err = Res.takeError())
    return Err;
  Globs.emplace_back(Pattern, LineNumber, std::move(Res.get()));
  return Error::success();
}

void GlobMatcher::LazyInit() const {
  if (LLVM_LIKELY(Initialized))
    return;
  Initialized = true;
  for (const auto &[Idx, G] : enumerate(Globs)) {
    StringRef Prefix = G.Pattern.prefix();
    StringRef Suffix = G.Pattern.suffix();

    if (Suffix.empty() && Prefix.empty()) {
      // If both prefix and suffix are empty put into special tree to search by
      // substring in a middle.
      StringRef Substr = G.Pattern.longest_substr();
      if (!Substr.empty()) {
        // But only if substring is not empty. Searching this tree is more
        // expensive.
        auto &V = SubstrToGlob.emplace(Substr).first->second;
        V.emplace_back(Idx);
        continue;
      }
    }

    auto &SToGlob = PrefixSuffixToGlob.emplace(Prefix).first->second;
    auto &V = SToGlob.emplace(reverse(Suffix)).first->second;
    V.emplace_back(Idx);
  }
}

unsigned GlobMatcher::match(StringRef Query) const {
  LazyInit();

  int Best = -1;
  if (!PrefixSuffixToGlob.empty()) {
    for (const auto &[_, SToGlob] : PrefixSuffixToGlob.find_prefixes(Query)) {
      for (const auto &[_, V] : SToGlob.find_prefixes(reverse(Query))) {
        for (int Idx : reverse(V)) {
          if (Best > Idx)
            break;
          const GlobMatcher::Glob &G = Globs[Idx];
          if (G.Pattern.match(Query)) {
            Best = Idx;
            // As soon as we find a match in the vector, we can break for this
            // vector, since the globs are already sorted by priority within the
            // prefix group. However, we continue searching other prefix groups
            // in the map, as they may contain a better match overall.
            break;
          }
        }
      }
    }
  }

  if (!SubstrToGlob.empty()) {
    // As we don't know when substring exactly starts, we will try all
    // possibilities. In most cases search will fail on first characters.
    for (StringRef Q = Query; !Q.empty(); Q = Q.drop_front()) {
      for (const auto &[_, V] : SubstrToGlob.find_prefixes(Q)) {
        for (int Idx : reverse(V)) {
          if (Best > Idx)
            break;
          const GlobMatcher::Glob &G = Globs[Idx];
          if (G.Pattern.match(Query)) {
            Best = Idx;
            // As soon as we find a match in the vector, we can break for this
            // vector, since the globs are already sorted by priority within the
            // prefix group. However, we continue searching other prefix groups
            // in the map, as they may contain a better match overall.
            break;
          }
        }
      }
    }
  }
  return Best < 0 ? 0 : Globs[Best].LineNo;
}

StringRef GlobMatcher::findRule(unsigned LineNo) const {
  for (const auto &G : Globs)
    if (G.LineNo == LineNo)
      return G.Name;
  assert(!"`findRule` should be called only with correct `LineNo`");
  return {};
}

Matcher::Matcher(bool UseGlobs, QueryOptions QOpts) : Options(QOpts) {
  if (UseGlobs)
    M.emplace<GlobMatcher>();
  else
    M.emplace<RegexMatcher>();
}

Error Matcher::insert(StringRef Pattern, unsigned LineNumber) {
  return std::visit([&](auto &V) { return V.insert(Pattern, LineNumber); }, M);
}

unsigned Matcher::match(StringRef Query) const {
  switch (Options) {
  case QueryOptions::kUnchanged:
    return matchInternal(Query);
  case QueryOptions::kNoSlashDot:
    return matchInternal(llvm::sys::path::remove_leading_dotslash(Query));
  case QueryOptions::kNoSlashDotWithFallback:
    break;
  }

  StringRef FixedQuery = llvm::sys::path::remove_leading_dotslash(Query);
  unsigned FixedMatched = matchInternal(FixedQuery);
  if (FixedQuery == Query)
    return FixedMatched;

  unsigned OriginalMatch = matchInternal(Query);
  if (OriginalMatch > FixedMatched) {
    static std::once_flag Warned;
    std::call_once(Warned, [&]() {
      WithColor::warning() << "Deprecated behaviour: '"
                           << findRule(OriginalMatch) << "' matches '" << Query
                           << "', but it should match '" << FixedQuery
                           << "'.\n";
    });
  }
  return std::max(OriginalMatch, FixedMatched);
}

unsigned Matcher::matchInternal(StringRef Query) const {
  return std::visit([&](auto &V) -> unsigned { return V.match(Query); }, M);
}

StringRef Matcher::findRule(unsigned LineNo) const {
  return std::visit([&](auto &V) -> StringRef { return V.findRule(LineNo); },
                    M);
}

} // namespace

class SpecialCaseList::Section::SectionImpl {
  friend class SpecialCaseList;
  const Matcher *findMatcher(StringRef Prefix, StringRef Category) const;

public:
  using SectionEntries = StringMap<StringMap<Matcher>>;

  SectionImpl(StringRef Str, bool UseGlobs) : SectionMatcher(UseGlobs) {}

  Matcher SectionMatcher;
  SectionEntries Entries;

  // Helper method to search by Prefix, Query, and Category. Returns
  // 1-based line number on which rule is defined, or 0 if there is no match.
  unsigned getLastMatch(StringRef Prefix, StringRef Query,
                        StringRef Category) const;

  // Helper method to search by Prefix, Query, and Category. Returns
  // matching rule, or empty string if there is no match.
  StringRef getLongestMatch(StringRef Prefix, StringRef Query,
                            StringRef Category) const;
};

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
    if (!parse(i, FileOrErr.get().get(), ParseError)) {
      Error = (Twine("error parsing file '") + Path + "': " + ParseError).str();
      return false;
    }
  }
  return true;
}

bool SpecialCaseList::createInternal(const MemoryBuffer *MB,
                                     std::string &Error) {
  if (!parse(0, MB, Error))
    return false;
  return true;
}

const std::vector<SpecialCaseList::Section> &SpecialCaseList::sections() const {
  return Sections;
}

Expected<SpecialCaseList::Section *>
SpecialCaseList::addSection(StringRef SectionStr, unsigned FileNo,
                            unsigned LineNo, bool UseGlobs) {
  SectionStr = SectionStr.copy(StrAlloc);
  Sections.emplace_back(SectionStr, FileNo, UseGlobs);
  auto &Section = Sections.back();

  if (auto Err = Section.Impl->SectionMatcher.insert(SectionStr, LineNo)) {
    return createStringError(errc::invalid_argument,
                             "malformed section at line " + Twine(LineNo) +
                                 ": '" + SectionStr +
                                 "': " + toString(std::move(Err)));
  }

  return &Section;
}

bool SpecialCaseList::parse(unsigned FileIdx, const MemoryBuffer *MB,
                            std::string &Error) {
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

  auto ErrOrSection = addSection("*", FileIdx, 1, true);
  if (auto Err = ErrOrSection.takeError()) {
    Error = toString(std::move(Err));
    return false;
  }
  Section::SectionImpl *CurrentImpl = ErrOrSection.get()->Impl.get();

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

      auto ErrOrSection =
          addSection(Line.drop_front().drop_back(), FileIdx, LineNo, UseGlobs);
      if (auto Err = ErrOrSection.takeError()) {
        Error = toString(std::move(Err));
        return false;
      }
      CurrentImpl = ErrOrSection.get()->Impl.get();
      continue;
    }

    // Get our prefix and unparsed glob.
    auto [Prefix, Postfix] = Line.split(":");
    if (Postfix.empty()) {
      // Missing ':' in the line.
      Error = ("malformed line " + Twine(LineNo) + ": '" + Line + "'").str();
      return false;
    }

    Matcher::QueryOptions QOpts = Matcher::QueryOptions::kUnchanged;
    if (llvm::is_contained(PathPrefixes, Prefix)) {
      if (Version == 3)
        QOpts = Matcher::QueryOptions::kNoSlashDotWithFallback;
      else if (Version > 4)
        QOpts = Matcher::QueryOptions::kNoSlashDot;
    }

    auto [Pattern, Category] = Postfix.split("=");
    auto [It, _] =
        CurrentImpl->Entries[Prefix].try_emplace(Category, UseGlobs, QOpts);
    Pattern = Pattern.copy(StrAlloc);
    if (auto Err = It->second.insert(Pattern, LineNo)) {
      Error =
          (Twine("malformed ") + (UseGlobs ? "glob" : "regex") + " in line " +
           Twine(LineNo) + ": '" + Pattern + "': " + toString(std::move(Err)))
              .str();
      return false;
    }
  }

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
    if (S.Impl->SectionMatcher.matchAny(Section)) {
      unsigned Blame = S.getLastMatch(Prefix, Query, Category);
      if (Blame)
        return {S.FileIdx, Blame};
    }
  }
  return NotFound;
}

SpecialCaseList::Section::Section(StringRef Str, unsigned FileIdx,
                                  bool UseGlobs)
    : Name(Str), FileIdx(FileIdx),
      Impl(std::make_unique<SectionImpl>(Str, UseGlobs)) {}

SpecialCaseList::Section::Section(Section &&) = default;
SpecialCaseList::Section::~Section() = default;

bool SpecialCaseList::Section::matchName(StringRef Name) const {
  return Impl->SectionMatcher.matchAny(Name);
}

const Matcher *
SpecialCaseList::Section::SectionImpl::findMatcher(StringRef Prefix,
                                                   StringRef Category) const {
  SectionEntries::const_iterator I = Entries.find(Prefix);
  if (I == Entries.end())
    return nullptr;
  StringMap<Matcher>::const_iterator II = I->second.find(Category);
  if (II == I->second.end())
    return nullptr;

  return &II->second;
}

unsigned SpecialCaseList::Section::getLastMatch(StringRef Prefix,
                                                StringRef Query,
                                                StringRef Category) const {
  if (const Matcher *M = Impl->findMatcher(Prefix, Category))
    return M->match(Query);
  return 0;
}

bool SpecialCaseList::Section::hasPrefix(StringRef Prefix) const {
  return Impl->Entries.find(Prefix) != Impl->Entries.end();
}

} // namespace llvm
