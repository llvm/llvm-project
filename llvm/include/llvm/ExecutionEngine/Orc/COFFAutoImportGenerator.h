//===- COFFAutoImportGenerator.h - COFF dllimport auto-import -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares COFFAutoImportGenerator, which synthesizes COFF dllimport __imp_
// symbols and jump-thunks for the symbols exported by a single dynamic
// library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_COFFAUTOIMPORTGENERATOR_H
#define LLVM_EXECUTIONENGINE_ORC_COFFAUTOIMPORTGENERATOR_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/DylibManager.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/Support/Compiler.h"

namespace llvm::orc {

/// A utility class that synthesizes COFF dllimport __imp_ symbols and PLT
/// stubs for the symbols exported by a single dynamic library ("easy mode"
/// auto-import).
///
/// Unlike DLLImportDefinitionGenerator, which resolves the underlying symbol
/// through the JITDylib's link order, this generator is bound to one dynamic
/// library: that library's export table is the authority on what may be
/// synthesized. Any requested symbol the library does not export is left
/// unresolved, so the link fails exactly as a static link against the
/// corresponding import library would.
///
/// Synthesis is lazy (driven by JITLink external-symbol lookups) and assumes
/// every import is a function: for each resolved import X it creates an __imp_X
/// pointer slot holding X's address in the library and an X thunk that jumps
/// through that slot. Data imports are not distinguished from code and will
/// misbehave; clients with data imports must supply an import library or use
/// __declspec(dllimport). Note also that &X resolves to the synthesized thunk,
/// not to X's address inside the library.
///
/// All synthesized stubs share a single ResourceTracker; see
/// getImportStubsResourceTracker() to reclaim them.
///
/// Supports whichever architectures JITLink has a pointer / pointer-jump-stub
/// creator registered for (see jitlink::getAnonymousPointerCreator and
/// jitlink::getPointerJumpStubCreator); Load() fails for any other
/// architecture.
class LLVM_ABI COFFAutoImportGenerator : public DefinitionGenerator {
public:
  /// Loads the dynamic library at the given path in the executor (via the given
  /// DylibManager) and, on success, returns a COFFAutoImportGenerator that
  /// synthesizes imports for the symbols it exports. On failure returns the
  /// reason the library failed to load. Resolving imports through DylibManager
  /// means this works for both in-process and out-of-process execution.
  static Expected<std::unique_ptr<COFFAutoImportGenerator>>
  Load(ExecutionSession &ES, ObjectLinkingLayer &L, DylibManager &DylibMgr,
       const char *LibraryPath);

  Error tryToGenerate(LookupState &LS, LookupKind K, JITDylib &JD,
                      JITDylibLookupFlags JDLookupFlags,
                      const SymbolLookupSet &Symbols) override;

  /// Returns the ResourceTracker that owns the stubs synthesized by this
  /// generator, or null if none have been synthesized yet. Calling remove() on
  /// it reclaims every synthesized __imp_ slot and thunk without affecting
  /// other definitions in the JITDylib; synthesis afterwards transparently
  /// starts a fresh tracker. Not thread-safe with respect to lookups that may
  /// concurrently trigger synthesis -- reclaim at a quiescent point.
  ResourceTrackerSP getImportStubsResourceTracker() const {
    return ImportStubsRT;
  }

private:
  COFFAutoImportGenerator(ExecutionSession &ES, ObjectLinkingLayer &L,
                          DylibManager &DylibMgr,
                          tpctypes::DylibHandle LibHandle,
                          jitlink::AnonymousPointerCreator CreatePointer,
                          jitlink::PointerJumpStubCreator CreateStub)
      : ES(ES), L(L), DylibMgr(DylibMgr), LibHandle(LibHandle),
        CreatePointer(std::move(CreatePointer)),
        CreateStub(std::move(CreateStub)) {}

  Expected<std::unique_ptr<jitlink::LinkGraph>>
  createStubsGraph(const SymbolMap &Resolved);

  static constexpr StringLiteral getImpPrefix() { return "__imp_"; }
  static constexpr StringLiteral getSectionName() {
    return "$__AUTOIMPORT_STUBS";
  }

  ExecutionSession &ES;
  ObjectLinkingLayer &L;
  DylibManager &DylibMgr;
  tpctypes::DylibHandle LibHandle;

  /// Cached at Load() time so unsupported architectures are rejected eagerly,
  /// rather than later inside tryToGenerate's asynchronous lookup callback.
  jitlink::AnonymousPointerCreator CreatePointer;
  jitlink::PointerJumpStubCreator CreateStub;

  /// Owns the synthesized stubs; (re)created lazily on first synthesis.
  ResourceTrackerSP ImportStubsRT;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_COFFAUTOIMPORTGENERATOR_H
