//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares shared functionality for linking static libraries
// (archives) in offloading tools. It provides a symbol-driven fixed-point
// archive member selection algorithm used by both clang-nvlink-wrapper and
// clang-sycl-linker.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_OFFLOADING_ARCHIVELINKER_H
#define LLVM_FRONTEND_OFFLOADING_ARCHIVELINKER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/IRSymtab.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"
#include <cstdint>
#include <memory>

namespace llvm {
class MemoryBuffer;

namespace object {
class SymbolRef;
} // namespace object

namespace offloading {

/// A minimum symbol interface that provides the necessary information to
/// extract archive members and resolve LTO symbols.
struct Symbol {
  enum Flags {
    None = 0,
    Undefined = 1 << 0,
    Weak = 1 << 1,
  };

  Symbol() : File(), SymFlags(None), UsedInRegularObj(false) {}
  Symbol(Symbol::Flags F) : File(), SymFlags(F), UsedInRegularObj(true) {}

  Symbol(MemoryBufferRef File, const irsymtab::Reader::SymbolRef Sym)
      : File(File), SymFlags(0), UsedInRegularObj(false) {
    if (Sym.isUndefined())
      SymFlags |= Undefined;
    if (Sym.isWeak())
      SymFlags |= Weak;
  }

  /// Create a Symbol from an object file symbol reference.
  /// Returns an error if symbol flags cannot be retrieved.
  static Expected<Symbol> createFromObject(MemoryBufferRef File,
                                           const object::SymbolRef &Sym);

  bool isWeak() const { return SymFlags & Weak; }
  bool isUndefined() const { return SymFlags & Undefined; }

  MemoryBufferRef File;
  uint32_t SymFlags;
  bool UsedInRegularObj;
};

/// Description of a single input (file or library).
struct InputDesc {
  enum class Kind { File, Library };

  StringRef Value; // File path, or library name for -l (the value after -l).
  Kind InputKind;
  bool WholeArchive; // --whole-archive state in effect at this input.
};

/// Result of archive member resolution.
struct ResolvedInputs {
  SmallVector<std::unique_ptr<MemoryBuffer>>
      Buffers;              // Members to link, in order.
  StringMap<Symbol> SymTab; // Symbol table (for LTO resolution).
};

/// Resolve archive members from the given inputs using a symbol-driven
/// fixed-point algorithm. For each input:
/// - If it's a Library, search for lib<name>.a or :<name> in SearchPaths
/// - If it's a File, use the path directly
/// - Archives are expanded and members are lazily extracted based on symbol
///   references unless WholeArchive is true
/// - Non-archive inputs (bitcode, ELF objects) are always included
///
/// Returns the buffers to link and the symbol table for LTO resolution.
///
/// \param Order Positional inputs + -l libraries in order.
/// \param SearchPaths -L paths for library search.
/// \param ForcedUndefs -u symbols (may be empty).
/// \param Root Sysroot for "=" prefixed paths ("" if none).
/// \param IsFatBinary Optional predicate to identify "fat binary" inputs that
///        should be passed through without symbol scanning (e.g., nvlink's
///        cubin detection). If null, all inputs are scanned normally.
Expected<ResolvedInputs> resolveArchiveMembers(
    ArrayRef<InputDesc> Order, ArrayRef<StringRef> SearchPaths,
    ArrayRef<StringRef> ForcedUndefs = {}, StringRef Root = "",
    function_ref<bool(MemoryBufferRef)> IsFatBinary = nullptr);

} // namespace offloading
} // namespace llvm

#endif // LLVM_FRONTEND_OFFLOADING_ARCHIVELINKER_H
