//===------ JITLinkGeneric.h - Generic JIT linker utilities -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic JITLinker utilities. E.g. graph pruning, eh-frame parsing.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_EXECUTIONENGINE_JITLINK_JITLINKGENERIC_H
#define LIB_EXECUTIONENGINE_JITLINK_JITLINKGENERIC_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"

#define DEBUG_TYPE "jitlink"

namespace llvm {

class MemoryBufferRef;

namespace jitlink {

/// Base class for a JIT linker.
///
/// A JITLinkerBase instance links one object file into an ongoing JIT
/// session. Symbol resolution and finalization operations are pluggable,
/// and called using continuation passing (passing a continuation for the
/// remaining linker work) to allow them to be performed asynchronously.
class JITLinkerBase {
public:
  JITLinkerBase(std::unique_ptr<JITLinkContext> Ctx,
                std::unique_ptr<LinkGraph> G, PassConfiguration Passes)
      : Ctx(std::move(Ctx)), G(std::move(G)), Passes(std::move(Passes)) {
    assert(this->Ctx && "Ctx can not be null");
    assert(this->G && "G can not be null");
  }

  virtual ~JITLinkerBase();

protected:
  struct SegmentLayout {
    using BlocksList = std::vector<Block *>;

    BlocksList ContentBlocks;
    BlocksList ZeroFillBlocks;
  };

  using SegmentLayoutMap = DenseMap<unsigned, SegmentLayout>;

  // Phase 1:
  //   1.1: Run pre-prune passes
  //   1.2: Prune graph
  //   1.3: Run post-prune passes
  //   1.4: Sort blocks into segments
  //   1.5: Allocate segment memory, update node vmaddrs to target vmaddrs
  //   1.6: Run post-allocation passes
  //   1.7: Notify context of final assigned symbol addresses
  //   1.8: Identify external symbols and make an async call to resolve
  void linkPhase1(std::unique_ptr<JITLinkerBase> Self);

  // Phase 2:
  //   2.1: Apply resolution results
  //   2.2: Run pre-fixup passes
  //   2.3: Fix up block contents
  //   2.4: Run post-fixup passes
  //   2.5: Make an async call to transfer and finalize memory.
  void linkPhase2(std::unique_ptr<JITLinkerBase> Self,
                  Expected<AsyncLookupResult> LookupResult,
                  SegmentLayoutMap Layout);

  // Phase 3:
  //   3.1: Call OnFinalized callback, handing off allocation.
  void linkPhase3(std::unique_ptr<JITLinkerBase> Self, Error Err);

  // Align a JITTargetAddress to conform with block alignment requirements.
  static JITTargetAddress alignToBlock(JITTargetAddress Addr, Block &B) {
    uint64_t Delta = (B.getAlignmentOffset() - Addr) % B.getAlignment();
    return Addr + Delta;
  }

  // Align a pointer to conform with block alignment requirements.
  static char *alignToBlock(char *P, Block &B) {
    uint64_t PAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(P));
    uint64_t Delta = (B.getAlignmentOffset() - PAddr) % B.getAlignment();
    return P + Delta;
  }

private:
  // Run all passes in the given pass list, bailing out immediately if any pass
  // returns an error.
  Error runPasses(LinkGraphPassList &Passes);

  // Copy block contents and apply relocations.
  // Implemented in JITLinker.
  virtual Error fixUpBlocks(LinkGraph &G) const = 0;

  SegmentLayoutMap layOutBlocks();
  Error allocateSegments(const SegmentLayoutMap &Layout);
  JITLinkContext::LookupMap getExternalSymbolNames() const;
  void applyLookupResult(AsyncLookupResult LR);
  void copyBlockContentToWorkingMemory(const SegmentLayoutMap &Layout,
                                       JITLinkMemoryManager::Allocation &Alloc);
  void deallocateAndBailOut(Error Err);

  std::unique_ptr<JITLinkContext> Ctx;
  std::unique_ptr<LinkGraph> G;
  PassConfiguration Passes;
  std::unique_ptr<JITLinkMemoryManager::Allocation> Alloc;
};

template <typename LinkerImpl> class JITLinker : public JITLinkerBase {
public:
  using JITLinkerBase::JITLinkerBase;

  /// Link constructs a LinkerImpl instance and calls linkPhase1.
  /// Link should be called with the constructor arguments for LinkerImpl, which
  /// will be forwarded to the constructor.
  template <typename... ArgTs> static void link(ArgTs &&... Args) {
    auto L = std::make_unique<LinkerImpl>(std::forward<ArgTs>(Args)...);

    // Ownership of the linker is passed into the linker's doLink function to
    // allow it to be passed on to async continuations.
    //
    // FIXME: Remove LTmp once we have c++17.
    // C++17 sequencing rules guarantee that function name expressions are
    // sequenced before arguments, so L->linkPhase1(std::move(L), ...) will be
    // well formed.
    auto &LTmp = *L;
    LTmp.linkPhase1(std::move(L));
  }

private:
  const LinkerImpl &impl() const {
    return static_cast<const LinkerImpl &>(*this);
  }

  Error fixUpBlocks(LinkGraph &G) const override {
    LLVM_DEBUG(dbgs() << "Fixing up blocks:\n");

    for (auto *B : G.blocks()) {
      LLVM_DEBUG(dbgs() << "  " << *B << ":\n");

      // Copy Block data and apply fixups.
      LLVM_DEBUG(dbgs() << "    Applying fixups.\n");
      for (auto &E : B->edges()) {

        // Skip non-relocation edges.
        if (!E.isRelocation())
          continue;

        // Dispatch to LinkerImpl for fixup.
        auto *BlockData = const_cast<char *>(B->getContent().data());
        if (auto Err = impl().applyFixup(*B, E, BlockData))
          return Err;
      }
    }

    return Error::success();
  }
};

/// Removes dead symbols/blocks/addressables.
///
/// Finds the set of symbols and addressables reachable from any symbol
/// initially marked live. All symbols/addressables not marked live at the end
/// of this process are removed.
void prune(LinkGraph &G);

} // end namespace jitlink
} // end namespace llvm

#undef DEBUG_TYPE // "jitlink"

#endif // LLVM_EXECUTIONENGINE_JITLINK_JITLINKGENERIC_H
