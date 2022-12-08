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

Optional<MappedPrefix> MappedPrefix::getFromJoined(StringRef JoinedMapping) {
  auto Equals = JoinedMapping.find('=');
  if (Equals == StringRef::npos)
    return std::nullopt;
  StringRef Old = JoinedMapping.substr(0, Equals);
  StringRef New = JoinedMapping.substr(Equals + 1);
  return MappedPrefix{Old, New};
}

template <bool StopOnInvalid, class StringT>
static Optional<StringRef>
transformJoinedImpl(ArrayRef<StringT> JoinedMappings,
                    SmallVectorImpl<MappedPrefix> &Mappings) {
  size_t OriginalSize = Mappings.size();
  for (StringRef Joined : JoinedMappings) {
    if (Optional<MappedPrefix> Split = MappedPrefix::getFromJoined(Joined)) {
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

static Error makeErrorForInvalidJoin(Optional<StringRef> Joined) {
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

Expected<Optional<StringRef>>
PrefixMapper::mapImpl(StringRef Path, SmallVectorImpl<char> &Storage) {
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

Error PrefixMapper::map(StringRef Path, SmallVectorImpl<char> &NewPath) {
  NewPath.clear();
  Optional<StringRef> Mapped;
  if (Error E = mapImpl(Path, NewPath).moveInto(Mapped))
    return E;
  if (!NewPath.empty())
    return Error::success();
  if (!Mapped)
    Mapped = Path;
  NewPath.assign(Mapped->begin(), Mapped->end());
  return Error::success();
}

Error PrefixMapper::map(StringRef Path, std::string &NewPath) {
  return mapToString(Path).moveInto(NewPath);
}

Expected<std::string> PrefixMapper::mapToString(StringRef Path) {
  SmallString<256> Storage;
  Optional<StringRef> Mapped;
  if (Error E = mapImpl(Path, Storage).moveInto(Mapped))
    return std::move(E);
  return Mapped ? Mapped->str() : Path.str();
}

Error PrefixMapper::mapInPlace(SmallVectorImpl<char> &Path) {
  SmallString<256> Storage;
  Optional<StringRef> Mapped;
  if (Error E = mapImpl(StringRef(Path.begin(), Path.size()), Storage)
                    .moveInto(Mapped))
    return E;
  if (!Mapped)
    return Error::success();
  if (Storage.empty())
    Path.assign(Mapped->begin(), Mapped->end());
  else
    Storage.swap(Path);
  return Error::success();
}

Error PrefixMapper::mapInPlace(std::string &Path) {
  SmallString<256> Storage;
  Optional<StringRef> Mapped;
  if (Error E = mapImpl(Path, Storage).moveInto(Mapped))
    return E;
  if (!Mapped)
    return Error::success();
  Path.assign(Mapped->begin(), Mapped->size());
  return Error::success();
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

Expected<Optional<StringRef>>
TreePathPrefixMapper::mapImpl(StringRef Path, SmallVectorImpl<char> &Storage) {
  StringRef TreePath;
  if (Error E = getTreePath(Path).moveInto(TreePath))
    return std::move(E);
  Optional<StringRef> Mapped =
      cantFail(PrefixMapper::mapImpl(TreePath, Storage));
  if (Mapped)
    return *Mapped;
  if (TreePath != Path)
    return TreePath;
  return std::nullopt;
}

Expected<StringRef> TreePathPrefixMapper::getTreePath(StringRef Path) {
  if (Path.empty())
    return Path;
  const vfs::CachedDirectoryEntry *Entry = nullptr;
  if (Error E =
          FS->getDirectoryEntry(Path, /*FollowSymlinks=*/false).moveInto(Entry))
    return std::move(E);
  return Entry->getTreePath();
}

Error TreePathPrefixMapper::canonicalizePrefix(StringRef &Prefix) {
  StringRef TreePath;
  if (Error E = getTreePath(Prefix).moveInto(TreePath))
    return E;
  if (TreePath != Prefix)
    Prefix = TreePath;
  return Error::success();
}

Error TreePathPrefixMapper::add(const MappedPrefix &Mapping) {
  StringRef Old = Mapping.Old;
  StringRef New = Mapping.New;
  if (Error E = canonicalizePrefix(Old))
    return E;
  return PrefixMapper::add(MappedPrefix{Old, New});
}

StringRef
TreePathPrefixMapper::mapDirEntry(const vfs::CachedDirectoryEntry &Entry,
                                  StringSaver &Saver) {
  StringRef TreePath = Entry.getTreePath();
  SmallString<256> PathBuf;
  Optional<StringRef> Mapped =
      cantFail(PrefixMapper::mapImpl(TreePath, PathBuf));
  return Mapped ? Saver.save(*Mapped) : TreePath;
}
