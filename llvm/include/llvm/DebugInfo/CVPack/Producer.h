//===- Producer.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Packs COFF objects against a CodeView pack: the debug info that the objects
// share moves to the pack, and each object keeps what is its own with a
// reference to the pack.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CVPACK_PRODUCER_H
#define LLVM_DEBUGINFO_CVPACK_PRODUCER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/DebugInfo/CodeView/CVPacks.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace llvm {
namespace codeview {
namespace cvpack {

struct PackOptions {
  /// The directories searched for packs. A new pack is written to the first.
  std::vector<std::string> PackPaths;
  /// The directory that receives the packed objects and cvpacks.json, which
  /// lists the pack that each of them needs.
  std::string StagingDirectory;
  /// Write a new base pack from the inputs, without reading the packs of
  /// PackPaths.
  bool Rebase = false;
  /// Start over as a rebase when the base no longer fits the inputs: when a
  /// rebase would pack more than MaxMissingPercent of the size of the base
  /// that the base lacks, or the inputs use less than MinUsedPercent of it.
  bool AutoRebase = true;
  unsigned MaxMissingPercent = 25;
  unsigned MinUsedPercent = 50;
  /// The base to pack against, instead of the best pack of PackPaths. The run
  /// then never rebases.
  std::optional<PackId> Base;
  /// Delete the packs of the first directory that no packed object needs.
  bool Prune = false;
  /// Called with each warning, and with each progress message. Either may be
  /// empty. The calls come from one thread at a time.
  std::function<void(const Twine &)> Warn;
  std::function<void(const Twine &)> Log;
};

/// Pack the objects at InputPaths. Objects are read and written on the threads
/// of llvm::parallel. An object that cannot be packed is copied unchanged, and
/// every packed object is checked to restore to its input byte for byte.
LLVM_ABI Error packObjects(const PackOptions &Options,
                           ArrayRef<StringRef> InputPaths);

} // namespace cvpack
} // namespace codeview
} // namespace llvm

#endif // LLVM_DEBUGINFO_CVPACK_PRODUCER_H
