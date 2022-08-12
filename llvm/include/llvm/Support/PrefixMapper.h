//===- llvm/Support/PrefixMapper.h - Prefix mapping utility -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_PREFIXMAPPER_H
#define LLVM_TABLEGEN_PREFIXMAPPER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"

namespace llvm {

namespace vfs {
class FileSystem;
class CachedDirectoryEntry;
} // end namespace vfs

struct MappedPrefix {
  StringRef Old;
  StringRef New;

  MappedPrefix getInverse() const { return MappedPrefix{New, Old}; }

  bool operator==(const MappedPrefix &RHS) const {
    return Old == RHS.Old && New == RHS.New;
  }
  bool operator!=(const MappedPrefix &RHS) const { return !(*this == RHS); }

  static Optional<MappedPrefix> getFromJoined(StringRef JoinedMapping);

  static Error transformJoined(ArrayRef<StringRef> Joined,
                               SmallVectorImpl<MappedPrefix> &Split);
  static Error transformJoined(ArrayRef<std::string> Joined,
                               SmallVectorImpl<MappedPrefix> &Split);
  static void transformJoinedIfValid(ArrayRef<StringRef> Joined,
                                     SmallVectorImpl<MappedPrefix> &Split);
  static void transformJoinedIfValid(ArrayRef<std::string> Joined,
                                     SmallVectorImpl<MappedPrefix> &Split);
};

/// Remap path prefixes.
///
/// FIXME: The StringSaver should be optional. Only APIs returning StringRef
/// need it, and those could assert/crash if one is not configured.
class PrefixMapper {
public:
  /// Map \p Path, and saving the new (or existing) path in \p NewPath.
  ///
  /// \pre \p Path is not a reference into \p NewPath.
  void map(StringRef Path, SmallVectorImpl<char> &NewPath);
  void map(StringRef Path, std::string &NewPath);

  /// Map \p Path, returning \a StringSaver::save() for new paths that aren't
  /// exact matches.
  StringRef map(StringRef Path);

  /// Map \p Path, returning \a std::string.
  std::string mapToString(StringRef Path);

  /// Map \p Path in place.
  void mapInPlace(SmallVectorImpl<char> &Path);
  void mapInPlace(std::string &Path);

private:
  /// Map (or unmap) \p Path. On a match, fills \p Storage with the mapped path
  /// unless it's an exact match.
  ///
  /// \pre \p Path is not a reference into \p Storage.
  Optional<StringRef> mapImpl(StringRef Path, SmallVectorImpl<char> &Storage);

public:
  void add(const MappedPrefix &MP) { Mappings.push_back(MP); }

  /// A path-based reverse lexicographic sort, putting deeper paths first so
  /// that deeper paths are prioritized over their parent paths. For example,
  /// if both the source and build directories are remapped and one is nested
  /// inside the other, the nested one will come first.
  ///
  /// FIXME: Doubtful that this would work correctly on windows, since it's
  /// case- and separator-sensitive.
  ///
  /// FIXME: Should probably be done implicitly, maybe by moving to a std::set
  /// or std::map.
  ///
  /// TODO: Test.
  void sort();

  template <class RangeT> void addRange(const RangeT &Mappings) {
    this->Mappings.append(Mappings.begin(), Mappings.end());
  }

  template <class RangeT> void addInverseRange(const RangeT &Mappings) {
    for (const MappedPrefix &M : Mappings)
      add(M.getInverse());
  }

  ArrayRef<MappedPrefix> getMappings() const { return Mappings; }

  StringSaver &getStringSaver() { return Saver; }
  sys::path::Style getPathStyle() const { return PathStyle; }

  PrefixMapper(sys::path::Style PathStyle = sys::path::Style::native)
      : PathStyle(PathStyle), Alloc(std::in_place), Saver(*Alloc) {}

  PrefixMapper(BumpPtrAllocator &Alloc,
               sys::path::Style PathStyle = sys::path::Style::native)
      : PathStyle(PathStyle), Saver(Alloc) {}

private:
  sys::path::Style PathStyle;
  Optional<BumpPtrAllocator> Alloc;
  StringSaver Saver;
  SmallVector<MappedPrefix> Mappings;
};

/// Wrapper for \a PrefixMapper that remaps paths that contain symlinks correctly.
///
/// This compares paths (included the prefix) using a "tree" path (like a real
/// path that does not follow symlinks in the basename). That is, the path to
/// the named filesystem object.
///
/// For example, given:
///
///     /a/sym -> b
///     /a/b/c/d
///
/// Paths are canonicalized in the following way:
///
/// - "/a/sym"   => "/a/sym"
/// - "/a/sym/"  => "/a/b"
/// - "/a/sym/c" => "/a/b/c"
///
/// If you have a prefix map "/a/sym/c=/new", then "/a/sym/c/d" and "/a/b/c/d"
/// are both remapped to "/new/c/d". However, a prefix map "/a/sym=/new" will
/// not remap anything under "/a/b"; instead, the symlink "/a/sym" is moved to
/// "/new".
///
/// Relative paths are also made absolute.
///
/// The implementation relies on \a vfs::CachedDirectoryEntry::getTreePath(),
/// which is only available in some filesystems.
///
/// Returns an error if an input cannot be found, except that an empty string
/// always maps to itself.
class TreePathPrefixMapper {
public:
  void map(const vfs::CachedDirectoryEntry &Entry,
           SmallVectorImpl<char> &NewPath);
  void map(const vfs::CachedDirectoryEntry &Entry, std::string &NewPath);
  StringRef map(const vfs::CachedDirectoryEntry &Entry);
  std::string mapToString(const vfs::CachedDirectoryEntry &Entry);

  Error map(StringRef Path, SmallVectorImpl<char> &NewPath);
  Error map(StringRef Path, std::string &NewPath) {
    return mapToString(Path).moveInto(NewPath);
  }
  Expected<StringRef> map(StringRef Path);
  Expected<std::string> mapToString(StringRef Path);
  Error mapInPlace(SmallVectorImpl<char> &Path);
  Error mapInPlace(std::string &Path);

  void mapOrOriginal(StringRef Path, SmallVectorImpl<char> &NewPath) {
    if (errorToBool(map(Path, NewPath)))
      NewPath.assign(Path.begin(), Path.end());
  }
  void mapOrOriginal(StringRef Path, std::string &NewPath) {
    if (errorToBool(map(Path, NewPath)))
      NewPath.assign(Path.begin(), Path.end());
  }
  Optional<StringRef> mapOrNone(StringRef Path) {
    return expectedToOptional(map(Path));
  }
  StringRef mapOrOriginal(StringRef Path) {
    Optional<StringRef> Mapped = mapOrNone(Path);
    return Mapped ? *Mapped : Path;
  }
  Optional<std::string> mapToStringOrNone(StringRef Path) {
    return expectedToOptional(mapToString(Path));
  }
  void mapInPlaceOrClear(SmallVectorImpl<char> &Path) {
    if (errorToBool(mapInPlace(Path)))
      Path.clear();
  }
  void mapInPlaceOrClear(std::string &Path) {
    if (errorToBool(mapInPlace(Path)))
      Path.clear();
  }

private:
  /// Find the tree path for \p Path, getting the real path for its parent
  /// directory but not following symlinks in \a sys::path::filename().
  Expected<StringRef> getTreePath(StringRef Path);
  Error getTreePath(StringRef Path, SmallVectorImpl<char> &TreePath);
  Error canonicalizePrefix(StringRef &Prefix);

public:
  ArrayRef<MappedPrefix> getMappings() const { return PM.getMappings(); }
  sys::path::Style getPathStyle() const { return PM.getPathStyle(); }

  Error add(const MappedPrefix &Mapping);

  template <class RangeT> Error addRange(const RangeT &Mappings) {
    for (const MappedPrefix &M : Mappings)
      if (Error E = add(M))
        return E;
    return Error::success();
  }

  template <class RangeT> Error addInverseRange(const RangeT &Mappings) {
    for (const MappedPrefix &M : Mappings)
      if (Error E = add(M.getInverse()))
        return E;
    return Error::success();
  }

  template <class RangeT> void addRangeIfValid(const RangeT &Mappings) {
    for (const MappedPrefix &M : Mappings)
      consumeError(add(M));
  }

  template <class RangeT> void addInverseRangeIfValid(const RangeT &Mappings) {
    for (const MappedPrefix &M : Mappings)
      consumeError(add(M.getInverse()));
  }

  void sort() { PM.sort(); }

  TreePathPrefixMapper(IntrusiveRefCntPtr<vfs::FileSystem> FS,
                       sys::path::Style PathStyle = sys::path::Style::native);
  TreePathPrefixMapper(IntrusiveRefCntPtr<vfs::FileSystem> FS,
                       BumpPtrAllocator &Alloc,
                       sys::path::Style PathStyle = sys::path::Style::native);
  ~TreePathPrefixMapper();

private:
  PrefixMapper PM;
  IntrusiveRefCntPtr<vfs::FileSystem> FS;
};

} // end namespace llvm

#endif // LLVM_TABLEGEN_PREFIXMAPPER_H
