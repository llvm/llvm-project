//===- llvm/CAS/HierarchicalTreeBuilder.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_HIERARCHICALTREEBUILDER_H
#define LLVM_CAS_HIERARCHICALTREEBUILDER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/CAS/CASReference.h"
#include "llvm/CAS/TreeEntry.h"
#include "llvm/CAS/TreeSchema.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h" // FIXME: Split out sys::fs::file_status.
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <cstddef>

namespace llvm {
namespace cas {

class ObjectStore;

/// Structure to facilitating building full tree hierarchies.
class HierarchicalTreeBuilder {
  struct HierarchicalEntry {
  public:
    StringRef getPath() const { return Path; }
    std::optional<ObjectRef> getRef() const { return Ref; }
    TreeEntry::EntryKind getKind() const { return Kind; }

    HierarchicalEntry(std::optional<ObjectRef> Ref, TreeEntry::EntryKind Kind,
                      StringRef Path)
        : Ref(Ref), Kind(Kind), Path(Path.str()) {
      assert(Ref || Kind == TreeEntry::Tree);
    }

  private:
    std::optional<ObjectRef> Ref;
    TreeEntry::EntryKind Kind;
    std::string Path;
  };

  sys::path::Style PathStyle;

  /// Preallocate space for small trees, common when creating cache keys.
  SmallVector<HierarchicalEntry, 8> Entries;
  SmallVector<HierarchicalEntry, 0> TreeContents;

  void pushImpl(std::optional<ObjectRef> Ref, TreeEntry::EntryKind Kind,
                const Twine &Path);

public:
  HierarchicalTreeBuilder(sys::path::Style PathStyle = sys::path::Style::native)
      : PathStyle(PathStyle) {}

  /// Add a hierarchical entry at \p Path, which is expected to be from the
  /// top-level (otherwise, the caller should prepend a working directory).
  ///
  /// All ".." components will be squashed by eating the parent. Paths through
  /// symlinks will not work, and should be resolved ahead of time. Paths must
  /// be POSIX-style.
  void push(ObjectRef Ref, TreeEntry::EntryKind Kind, const Twine &Path) {
    return pushImpl(Ref, Kind, Path);
  }

  /// Add a directory. Ensures the directory will exist even if there are no
  /// files pushed from within it.
  void pushDirectory(const Twine &Path) {
    return pushImpl(std::nullopt, TreeEntry::Tree, Path);
  }

  /// Add a directory with specific contents. It is functionally equivalent to:
  ///   * Calling pushDirectory() for every tree
  ///   * Calling push() for every non-tree
  ///
  /// Allows merging the contents of multiple directories.
  void pushTreeContent(ObjectRef Ref, const Twine &Path);

  /// Drop all entries.
  void clear() { Entries.clear(); }

  /// Recursively create the trees implied by calls to \a push(), return the
  /// top-level \a CASID.
  Expected<ObjectProxy> create(ObjectStore &CAS);
};

} // namespace cas
} // namespace llvm

#endif // LLVM_CAS_HIERARCHICALTREEBUILDER_H
