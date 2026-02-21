//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_CASFSBUILDER_H
#define LLVM_CAS_CASFSBUILDER_H

#include "llvm/CAS/HierarchicalTreeBuilder.h"
#include "llvm/Support/PrefixMapper.h"

namespace llvm::cas {
class CachingOnDiskFileSystem;

/// Builds a CAS file-system tree that can be used with \c createCASFileSystem.
/// Combines ingestion of paths from the on-disk file-system along with merging
/// other CAS file-system roots.
///
/// Not thread-safe.
class CASFSBuilder {
public:
  /// \param Prefix maps used for file system ingestion.
  explicit CASFSBuilder(ObjectStore &DB,
                        ArrayRef<MappedPrefix> PrefixMaps = {});

  ~CASFSBuilder();

  /// Ingest contents from the on-disk file-system. For a directory it will
  /// recursively ingest its contents. Symlinks are not followed. Emits an error
  /// if the path doesn't exist.
  Error ingestFileSystemPath(const Twine &Path);

  /// Merge a prior constructed CAS file-system tree root.
  ///
  /// \param Path the path to place the root at; can be empty.
  void mergeCASFSRoot(ObjectRef Root, const Twine &Path = "");

  /// Produce the merged CAS file-system tree root. \c CASFSBuilder should not
  /// be used after calling this.
  Expected<ObjectProxy> finish();

private:
  ObjectStore &DB;
  SmallVector<MappedPrefix, 2> PrefixMaps;
  HierarchicalTreeBuilder Builder;
  IntrusiveRefCntPtr<CachingOnDiskFileSystem> FS;
};

} // namespace llvm::cas

#endif // LLVM_CAS_CASFSBUILDER_H
