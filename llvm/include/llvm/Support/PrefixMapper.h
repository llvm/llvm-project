//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Prefix mapping utility.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_PREFIXMAPPER_H
#define LLVM_SUPPORT_PREFIXMAPPER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include <optional>

namespace llvm {
class StringSaver;

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

  LLVM_ABI static std::optional<MappedPrefix>
  getFromJoined(StringRef JoinedMapping);

  LLVM_ABI static Error transformJoined(ArrayRef<StringRef> Joined,
                                        SmallVectorImpl<MappedPrefix> &Split);
  LLVM_ABI static Error transformJoined(ArrayRef<std::string> Joined,
                                        SmallVectorImpl<MappedPrefix> &Split);
  LLVM_ABI static void
  transformJoinedIfValid(ArrayRef<StringRef> Joined,
                         SmallVectorImpl<MappedPrefix> &Split);
  LLVM_ABI static void
  transformJoinedIfValid(ArrayRef<std::string> Joined,
                         SmallVectorImpl<MappedPrefix> &Split);
  LLVM_ABI static void
  transformPairs(ArrayRef<std::pair<std::string, std::string>> Pairs,
                 SmallVectorImpl<MappedPrefix> &Split);
};

/// Remap path prefixes.
///
/// FIXME: The StringSaver should be optional. Only APIs returning StringRef
/// need it, and those could assert/crash if one is not configured.
class LLVM_ABI PrefixMapper {
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

} // end namespace llvm

#endif // LLVM_SUPPORT_PREFIXMAPPER_H
