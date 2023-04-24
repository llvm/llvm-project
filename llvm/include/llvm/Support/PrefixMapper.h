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
#include <optional>

namespace llvm {
class StringSaver;

namespace vfs {
class FileSystem;
class CachedDirectoryEntry;
} // end namespace vfs

struct MappedPrefix {
  std::string Old;
  std::string New;

  MappedPrefix() = default;
  MappedPrefix(StringRef Old_, StringRef New_)
      : Old(Old_.str()), New(New_.str()) {}

  MappedPrefix getInverse() const { return MappedPrefix{New, Old}; }

  bool operator==(const MappedPrefix &RHS) const {
    return Old == RHS.Old && New == RHS.New;
  }
  bool operator!=(const MappedPrefix &RHS) const { return !(*this == RHS); }

  static std::optional<MappedPrefix> getFromJoined(StringRef JoinedMapping);

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
  virtual ~PrefixMapper() = default;

  /// Map \p Path, and saving the new (or existing) path in \p NewPath.
  ///
  /// \pre \p Path is not a reference into \p NewPath.
  /// \returns true if \c NewPath was mapped.
  bool map(StringRef Path, SmallVectorImpl<char> &NewPath);
  /// Map \p Path, and saving the new (or existing) path in \p NewPath.
  ///
  /// \pre \p Path is not a reference into \p NewPath.
  /// \returns true if \c NewPath was mapped.
  bool map(StringRef Path, std::string &NewPath);

  /// Map \p Path, returning \a std::string.
  std::string mapToString(StringRef Path);

  /// Map \p Path in place.
  /// \returns true if the path was modified.
  bool mapInPlace(SmallVectorImpl<char> &Path);

  /// Map \p Path in place.
  /// \returns true if the path was modified.
  bool mapInPlace(std::string &Path);

protected:
  /// Map (or unmap) \p Path. On a match, fills \p Storage with the mapped path
  /// unless it's an exact match.
  ///
  /// \pre \p Path is not a reference into \p Storage.
  virtual std::optional<StringRef> mapImpl(StringRef Path,
                                      SmallVectorImpl<char> &Storage);

public:
  virtual void add(const MappedPrefix &MP) { Mappings.push_back(MP); }

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
    for (const MappedPrefix &M : Mappings)
      add(M);
  }

  template <class RangeT> void addInverseRange(const RangeT &Mappings) {
    for (const MappedPrefix &M : Mappings)
      add(M.getInverse());
  }

  bool empty() const { return getMappings().empty(); }

  ArrayRef<MappedPrefix> getMappings() const { return Mappings; }

  sys::path::Style getPathStyle() const { return PathStyle; }

  PrefixMapper(sys::path::Style PathStyle = sys::path::Style::native)
      : PathStyle(PathStyle) {}

private:
  sys::path::Style PathStyle;
  SmallVector<MappedPrefix> Mappings;
};

/// Wrapper for \a PrefixMapper that remaps paths that contain symlinks
/// correctly.
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
/// Falls back to a simple path prefix map if an input cannot be found, and
/// an empty string always maps to itself.
class TreePathPrefixMapper : public PrefixMapper {
private:
  std::optional<StringRef> mapImpl(StringRef Path,
                              SmallVectorImpl<char> &Storage) override;

  /// Find the tree path for \p Path, getting the real path for its parent
  /// directory but not following symlinks in \a sys::path::filename().
  ///
  /// \returns The tree path, or the original path if there are any errors.
  StringRef getTreePath(StringRef Path);

public:
  void add(const MappedPrefix &Mapping) override;

  StringRef mapDirEntry(const vfs::CachedDirectoryEntry &Entry,
                        SmallVectorImpl<char> &Storage);

  TreePathPrefixMapper(IntrusiveRefCntPtr<vfs::FileSystem> FS,
                       sys::path::Style PathStyle = sys::path::Style::native);
  ~TreePathPrefixMapper();

private:
  IntrusiveRefCntPtr<vfs::FileSystem> FS;
};

} // end namespace llvm

#endif // LLVM_TABLEGEN_PREFIXMAPPER_H
