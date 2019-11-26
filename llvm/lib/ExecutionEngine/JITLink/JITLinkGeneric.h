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
  JITLinkerBase(std::unique_ptr<JITLinkContext> Ctx, PassConfiguration Passes)
      : Ctx(std::move(Ctx)), Passes(std::move(Passes)) {
    assert(this->Ctx && "Ctx can not be null");
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
  //   1.1: Build link graph
  //   1.2: Run pre-prune passes
  //   1.2: Prune graph
  //   1.3: Run post-prune passes
  //   1.4: Sort blocks into segments
  //   1.5: Allocate segment memory
  //   1.6: Identify externals and make an async call to resolve function
  void linkPhase1(std::unique_ptr<JITLinkerBase> Self);

  // Phase 2:
  //   2.1: Apply resolution results
  //   2.2: Fix up block contents
  //   2.3: Call OnResolved callback
  //   2.3: Make an async call to transfer and finalize memory.
  void linkPhase2(std::unique_ptr<JITLinkerBase> Self,
                  Expected<AsyncLookupResult> LookupResult,
                  SegmentLayoutMap Layout);

  // Phase 3:
  //   3.1: Call OnFinalized callback, handing off allocation.
  void linkPhase3(std::unique_ptr<JITLinkerBase> Self, Error Err);

  // Build a graph from the given object buffer.
  // To be implemented by the client.
  virtual Expected<std::unique_ptr<LinkGraph>>
  buildGraph(MemoryBufferRef ObjBuffer) = 0;

  // For debug dumping of the link graph.
  virtual StringRef getEdgeKindName(Edge::Kind K) const = 0;

  // Alight a JITTargetAddress to conform with block alignment requirements.
  static JITTargetAddress alignToBlock(JITTargetAddress Addr, Block &B) {
    uint64_t Delta = (B.getAlignmentOffset() - Addr) % B.getAlignment();
    return Addr + Delta;
  }

  // Alight a pointer to conform with block alignment requirements.
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
  virtual Error
  copyAndFixUpBlocks(const SegmentLayoutMap &Layout,
                     JITLinkMemoryManager::Allocation &Alloc) const = 0;

  SegmentLayoutMap layOutBlocks();
  Error allocateSegments(const SegmentLayoutMap &Layout);
  JITLinkContext::LookupMap getExternalSymbolNames() const;
  void applyLookupResult(AsyncLookupResult LR);
  void deallocateAndBailOut(Error Err);

  void dumpGraph(raw_ostream &OS);

  std::unique_ptr<JITLinkContext> Ctx;
  PassConfiguration Passes;
  std::unique_ptr<LinkGraph> G;
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

  Error
  copyAndFixUpBlocks(const SegmentLayoutMap &Layout,
                     JITLinkMemoryManager::Allocation &Alloc) const override {
    LLVM_DEBUG(dbgs() << "Copying and fixing up blocks:\n");
    for (auto &KV : Layout) {
      auto &Prot = KV.first;
      auto &SegLayout = KV.second;

      auto SegMem = Alloc.getWorkingMemory(
          static_cast<sys::Memory::ProtectionFlags>(Prot));
      char *LastBlockEnd = SegMem.data();
      char *BlockDataPtr = LastBlockEnd;

      LLVM_DEBUG({
        dbgs() << "  Processing segment "
               << static_cast<sys::Memory::ProtectionFlags>(Prot) << " [ "
               << (const void *)SegMem.data() << " .. "
               << (const void *)((char *)SegMem.data() + SegMem.size())
               << " ]\n    Processing content sections:\n";
      });

      for (auto *B : SegLayout.ContentBlocks) {
        LLVM_DEBUG(dbgs() << "    " << *B << ":\n");

        // Pad to alignment/alignment-offset.
        BlockDataPtr = alignToBlock(BlockDataPtr, *B);

        LLVM_DEBUG({
          dbgs() << "      Bumped block pointer to "
                 << (const void *)BlockDataPtr << " to meet block alignment "
                 << B->getAlignment() << " and alignment offset "
                 << B->getAlignmentOffset() << "\n";
        });

        // Zero pad up to alignment.
        LLVM_DEBUG({
          if (LastBlockEnd != BlockDataPtr)
            dbgs() << "      Zero padding from " << (const void *)LastBlockEnd
                   << " to " << (const void *)BlockDataPtr << "\n";
        });

        while (LastBlockEnd != BlockDataPtr)
          *LastBlockEnd++ = 0;

        // Copy initial block content.
        LLVM_DEBUG({
          dbgs() << "      Copying block " << *B << " content, "
                 << B->getContent().size() << " bytes, from "
                 << (const void *)B->getContent().data() << " to "
                 << (const void *)BlockDataPtr << "\n";
        });
        memcpy(BlockDataPtr, B->getContent().data(), B->getContent().size());

        // Copy Block data and apply fixups.
        LLVM_DEBUG(dbgs() << "      Applying fixups.\n");
        for (auto &E : B->edges()) {

          // Skip non-relocation edges.
          if (!E.isRelocation())
            continue;

          // Dispatch to LinkerImpl for fixup.
          if (auto Err = impl().applyFixup(*B, E, BlockDataPtr))
            return Err;
        }

        // Point the block's content to the fixed up buffer.
        B->setContent(StringRef(BlockDataPtr, B->getContent().size()));

        // Update block end pointer.
        LastBlockEnd = BlockDataPtr + B->getContent().size();
        BlockDataPtr = LastBlockEnd;
      }

      // Zero pad the rest of the segment.
      LLVM_DEBUG({
        dbgs() << "    Zero padding end of segment from "
               << (const void *)LastBlockEnd << " to "
               << (const void *)((char *)SegMem.data() + SegMem.size()) << "\n";
      });
      while (LastBlockEnd != SegMem.data() + SegMem.size())
        *LastBlockEnd++ = 0;
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
