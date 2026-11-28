//===------- COFF.h - Generic JIT link function for COFF ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic jit-link functions for COFF.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_COFF_H
#define LLVM_EXECUTIONENGINE_JITLINK_COFF_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

namespace llvm {
namespace object {
class COFFObjectFile;
class COFFSymbolRef;
} // namespace object

namespace jitlink {

/// Create a LinkGraph from an COFF relocatable object.
///
/// Note: The graph does not take ownership of the underlying buffer, nor copy
/// its contents. The caller is responsible for ensuring that the object buffer
/// outlives the graph.
LLVM_ABI Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromCOFFObject(MemoryBufferRef ObjectBuffer,
                              std::shared_ptr<orc::SymbolStringPool> SSP);

/// Link the given graph.
///
/// Uses conservative defaults for GOT and stub handling based on the target
/// platform.
LLVM_ABI void link_COFF(std::unique_ptr<LinkGraph> G,
                        std::unique_ptr<JITLinkContext> Ctx);

/// GetImageBaseSymbol is a function object that finds the __ImageBase symbol
/// in the given graph if one is present.
///
/// The result is cached across calls, and can be reset by calling the reset
/// method.
class GetImageBaseSymbol {
public:
  GetImageBaseSymbol(StringRef ImageBaseName = "__ImageBase")
      : ImageBaseName(ImageBaseName) {}
  LLVM_ABI Symbol *operator()(LinkGraph &G);
  void reset(std::optional<Symbol *> CacheValue = std::nullopt) {
    ImageBase = CacheValue;
  }

private:
  StringRef ImageBaseName;
  std::optional<Symbol *> ImageBase;
};

/// Returns true if Name is the ".weak.<WeakName>.default[.<suffix>]" helper
/// that MC emits as the default of a SEARCH_ALIAS weak external WeakName.
LLVM_ABI bool isCOFFWeakDefaultHelper(StringRef Name, StringRef WeakName,
                                      uint32_t Characteristics);

/// Returns the name of Sym's default if Sym is a weak external whose default
/// is MC's helper, and std::nullopt otherwise.
LLVM_ABI Expected<std::optional<StringRef>>
getCOFFWeakDefaultHelper(const object::COFFObjectFile &Obj,
                         object::COFFSymbolRef Sym);

} // end namespace jitlink
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_COFF_H
