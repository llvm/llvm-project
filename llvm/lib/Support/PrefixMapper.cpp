//===- PrefixMapper.cpp - Prefix mapping utility --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/PrefixMapper.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace llvm;

std::optional<MappedPrefix> MappedPrefix::getFromJoined(StringRef JoinedMapping) {
  auto Equals = JoinedMapping.find('=');
  if (Equals == StringRef::npos)
    return std::nullopt;
  StringRef Old = JoinedMapping.substr(0, Equals);
  StringRef New = JoinedMapping.substr(Equals + 1);
  return MappedPrefix{Old, New};
}

template <bool StopOnInvalid, class StringT>
static std::optional<StringRef>
transformJoinedImpl(ArrayRef<StringT> JoinedMappings,
                    SmallVectorImpl<MappedPrefix> &Mappings) {
  size_t OriginalSize = Mappings.size();
  for (StringRef Joined : JoinedMappings) {
    if (std::optional<MappedPrefix> Split = MappedPrefix::getFromJoined(Joined)) {
      Mappings.push_back(*Split);
      continue;
    }
    if (!StopOnInvalid)
      continue;
    Mappings.resize(OriginalSize);
    return Joined;
  }
  return std::nullopt;
}

static Error makeErrorForInvalidJoin(std::optional<StringRef> Joined) {
  if (!Joined)
    return Error::success();
  return createStringError(inconvertibleErrorCode(),
                           "invalid prefix map: '" + *Joined + "'");
}

Error MappedPrefix::transformJoined(ArrayRef<StringRef> Joined,
                                    SmallVectorImpl<MappedPrefix> &Split) {
  return makeErrorForInvalidJoin(
      transformJoinedImpl</*StopOnInvalid*/ true>(Joined, Split));
}

Error MappedPrefix::transformJoined(ArrayRef<std::string> Joined,
                                    SmallVectorImpl<MappedPrefix> &Split) {
  return makeErrorForInvalidJoin(
      transformJoinedImpl</*StopOnInvalid*/ true>(Joined, Split));
}

void MappedPrefix::transformJoinedIfValid(
    ArrayRef<StringRef> Joined, SmallVectorImpl<MappedPrefix> &Split) {
  transformJoinedImpl</*StopOnInvalid*/ false>(Joined, Split);
}

void MappedPrefix::transformJoinedIfValid(
    ArrayRef<std::string> Joined, SmallVectorImpl<MappedPrefix> &Split) {
  transformJoinedImpl</*StopOnInvalid*/ false>(Joined, Split);
}

/// FIXME: Copy/pasted from llvm/lib/Support/Path.cpp.
static bool startsWith(StringRef Path, StringRef Prefix,
                       sys::path::Style PathStyle) {
  if (PathStyle == sys::path::Style::posix ||
      (PathStyle == sys::path::Style::native &&
       sys::path::is_style_posix(sys::path::Style::native)))
    return Path.startswith(Prefix);

  if (Path.size() < Prefix.size())
    return false;

  // Windows prefix matching : case and separator insensitive
  for (size_t I = 0, E = Prefix.size(); I != E; ++I) {
    bool SepPath = sys::path::is_separator(Path[I], PathStyle);
    bool SepPrefix = sys::path::is_separator(Prefix[I], PathStyle);
    if (SepPath != SepPrefix)
      return false;
    if (SepPath)
      continue;
    if (toLower(Path[I]) != toLower(Prefix[I]))
      return false;
  }
  return true;
}

std::optional<StringRef> PrefixMapper::mapImpl(StringRef Path,
                                          SmallVectorImpl<char> &Storage) {
  for (const MappedPrefix &Map : Mappings) {
    StringRef Old = Map.Old;
    StringRef New = Map.New;
    if (!startsWith(Path, Old, PathStyle))
      continue;
    StringRef Suffix = Path.drop_front(Old.size());
    if (Suffix.empty())
      return New; // Exact match.

    // Don't remap "/old-suffix" with mapping "/old=/new".
    if (!llvm::sys::path::is_separator(Suffix.front(), PathStyle))
      continue;

    // Drop the separator, append, and return.
    Storage.assign(New.begin(), New.end());
    llvm::sys::path::append(Storage, PathStyle, Suffix.drop_front());
    return StringRef(Storage.begin(), Storage.size());
  }
  return std::nullopt;
}

bool PrefixMapper::map(StringRef Path, SmallVectorImpl<char> &NewPath) {
  NewPath.clear();
  std::optional<StringRef> Mapped = mapImpl(Path, NewPath);
  if (!NewPath.empty())
    return true;
  bool Modified = Mapped.has_value();
  if (!Mapped)
    Mapped = Path;
  NewPath.assign(Mapped->begin(), Mapped->end());
  return Modified;
}

bool PrefixMapper::map(StringRef Path, std::string &NewPath) {
  SmallString<256> Storage;
  std::optional<StringRef> Mapped = mapImpl(Path, Storage);
  NewPath = Mapped ? Mapped->str() : Path.str();
  return Mapped.has_value();
}

std::string PrefixMapper::mapToString(StringRef Path) {
  SmallString<256> Storage;
  std::optional<StringRef> Mapped = mapImpl(Path, Storage);
  return Mapped ? Mapped->str() : Path.str();
}

bool PrefixMapper::mapInPlace(SmallVectorImpl<char> &Path) {
  SmallString<256> Storage;
  std::optional<StringRef> Mapped =
      mapImpl(StringRef(Path.begin(), Path.size()), Storage);
  if (!Mapped)
    return false;
  if (Storage.empty())
    Path.assign(Mapped->begin(), Mapped->end());
  else
    Storage.swap(Path);
  return true;
}

bool PrefixMapper::mapInPlace(std::string &Path) {
  SmallString<256> Storage;
  std::optional<StringRef> Mapped = mapImpl(Path, Storage);
  if (!Mapped)
    return false;
  Path.assign(Mapped->begin(), Mapped->size());
  return true;
}

void PrefixMapper::sort() {
  // FIXME: Only works for posix right now since it doesn't handle case- and
  // separator-insensitivity.
  std::stable_sort(Mappings.begin(), Mappings.end(),
                   [](const MappedPrefix &LHS, const MappedPrefix &RHS) {
                     return LHS.Old > RHS.Old;
                   });
}

TreePathPrefixMapper::TreePathPrefixMapper(
    IntrusiveRefCntPtr<vfs::FileSystem> FS, sys::path::Style PathStyle)
    : PrefixMapper(PathStyle), FS(std::move(FS)) {}

TreePathPrefixMapper::~TreePathPrefixMapper() = default;

std::optional<StringRef>
TreePathPrefixMapper::mapImpl(StringRef Path, SmallVectorImpl<char> &Storage) {
  StringRef TreePath = getTreePath(Path);
  std::optional<StringRef> Mapped = PrefixMapper::mapImpl(TreePath, Storage);
  if (Mapped)
    return *Mapped;
  if (TreePath != Path)
    return TreePath;
  return std::nullopt;
}

StringRef TreePathPrefixMapper::getTreePath(StringRef Path) {
  if (Path.empty())
    return Path;
  auto Entry = FS->getDirectoryEntry(Path, /*FollowSymlinks=*/false);
  if (!Entry) {
    consumeError(Entry.takeError());
    return Path;
  }
  return (*Entry)->getTreePath();
}

void TreePathPrefixMapper::add(const MappedPrefix &Mapping) {
  // Add the original mapping. If it contains a non-canonical path, this will
  // only affect the behaviour when later mapping a path that cannot be
  // canonicalized, since a non-canonical prefix cannot match a canonical path.
  PrefixMapper::add(Mapping);
  StringRef Old = getTreePath(Mapping.Old);
  StringRef New = Mapping.New;
  // Add the canonical prefix mapping, if different.
  if (Old != Mapping.Old)
    PrefixMapper::add(MappedPrefix{Old, New});
}

StringRef
TreePathPrefixMapper::mapDirEntry(const vfs::CachedDirectoryEntry &Entry,
                                  SmallVectorImpl<char> &Storage) {
  StringRef TreePath = Entry.getTreePath();
  std::optional<StringRef> Mapped = PrefixMapper::mapImpl(TreePath, Storage);
  return Mapped ? *Mapped : TreePath;
}
