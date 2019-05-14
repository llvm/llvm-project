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
    using SectionAtomsList = std::vector<DefinedAtom *>;
    struct SectionLayout {
      SectionLayout(Section &S) : S(&S) {}

      Section *S;
      SectionAtomsList Atoms;
    };

    using SectionLayoutList = std::vector<SectionLayout>;

    SectionLayoutList ContentSections;
    SectionLayoutList ZeroFillSections;
  };

  using SegmentLayoutMap = DenseMap<unsigned, SegmentLayout>;

  // Phase 1:
  //   1.1: Build atom graph
  //   1.2: Run pre-prune passes
  //   1.2: Prune graph
  //   1.3: Run post-prune passes
  //   1.4: Sort atoms into segments
  //   1.5: Allocate segment memory
  //   1.6: Identify externals and make an async call to resolve function
  void linkPhase1(std::unique_ptr<JITLinkerBase> Self);

  // Phase 2:
  //   2.1: Apply resolution results
  //   2.2: Fix up atom contents
  //   2.3: Call OnResolved callback
  //   2.3: Make an async call to transfer and finalize memory.
  void linkPhase2(std::unique_ptr<JITLinkerBase> Self,
                  Expected<AsyncLookupResult> LookupResult);

  // Phase 3:
  //   3.1: Call OnFinalized callback, handing off allocation.
  void linkPhase3(std::unique_ptr<JITLinkerBase> Self, Error Err);

  // Build a graph from the given object buffer.
  // To be implemented by the client.
  virtual Expected<std::unique_ptr<AtomGraph>>
  buildGraph(MemoryBufferRef ObjBuffer) = 0;

  // For debug dumping of the atom graph.
  virtual StringRef getEdgeKindName(Edge::Kind K) const = 0;

private:
  // Run all passes in the given pass list, bailing out immediately if any pass
  // returns an error.
  Error runPasses(AtomGraphPassList &Passes, AtomGraph &G);

  // Copy atom contents and apply relocations.
  // Implemented in JITLinker.
  virtual Error
  copyAndFixUpAllAtoms(const SegmentLayoutMap &Layout,
                       JITLinkMemoryManager::Allocation &Alloc) const = 0;

  void layOutAtoms();
  Error allocateSegments(const SegmentLayoutMap &Layout);
  DenseSet<StringRef> getExternalSymbolNames() const;
  void applyLookupResult(AsyncLookupResult LR);
  void deallocateAndBailOut(Error Err);

  void dumpGraph(raw_ostream &OS);

  std::unique_ptr<JITLinkContext> Ctx;
  PassConfiguration Passes;
  std::unique_ptr<AtomGraph> G;
  SegmentLayoutMap Layout;
  std::unique_ptr<JITLinkMemoryManager::Allocation> Alloc;
};

template <typename LinkerImpl> class JITLinker : public JITLinkerBase {
public:
  using JITLinkerBase::JITLinkerBase;

  /// Link constructs a LinkerImpl instance and calls linkPhase1.
  /// Link should be called with the constructor arguments for LinkerImpl, which
  /// will be forwarded to the constructor.
  template <typename... ArgTs> static void link(ArgTs &&... Args) {
    auto L = llvm::make_unique<LinkerImpl>(std::forward<ArgTs>(Args)...);

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
  copyAndFixUpAllAtoms(const SegmentLayoutMap &Layout,
                       JITLinkMemoryManager::Allocation &Alloc) const override {
    LLVM_DEBUG(dbgs() << "Copying and fixing up atoms:\n");
    for (auto &KV : Layout) {
      auto &Prot = KV.first;
      auto &SegLayout = KV.second;

      auto SegMem = Alloc.getWorkingMemory(
          static_cast<sys::Memory::ProtectionFlags>(Prot));
      char *LastAtomEnd = SegMem.data();
      char *AtomDataPtr = nullptr;

      LLVM_DEBUG({
        dbgs() << "  Processing segment "
               << static_cast<sys::Memory::ProtectionFlags>(Prot) << " [ "
               << (const void *)SegMem.data() << " .. "
               << (const void *)((char *)SegMem.data() + SegMem.size())
               << " ]\n    Processing content sections:\n";
      });

      for (auto &SI : SegLayout.ContentSections) {
        LLVM_DEBUG(dbgs() << "    " << SI.S->getName() << ":\n");
        for (auto *DA : SI.Atoms) {
          AtomDataPtr = LastAtomEnd;

          // Align.
          AtomDataPtr += alignmentAdjustment(AtomDataPtr, DA->getAlignment());
          LLVM_DEBUG({
            dbgs() << "      Bumped atom pointer to "
                   << (const void *)AtomDataPtr << " to meet alignment of "
                   << DA->getAlignment() << "\n";
          });

          // Zero pad up to alignment.
          LLVM_DEBUG({
            if (LastAtomEnd != AtomDataPtr)
              dbgs() << "      Zero padding from " << (const void *)LastAtomEnd
                     << " to " << (const void *)AtomDataPtr << "\n";
          });
          while (LastAtomEnd != AtomDataPtr)
            *LastAtomEnd++ = 0;

          // Copy initial atom content.
          LLVM_DEBUG({
            dbgs() << "      Copying atom " << *DA << " content, "
                   << DA->getContent().size() << " bytes, from "
                   << (const void *)DA->getContent().data() << " to "
                   << (const void *)AtomDataPtr << "\n";
          });
          memcpy(AtomDataPtr, DA->getContent().data(), DA->getContent().size());

          // Copy atom data and apply fixups.
          LLVM_DEBUG(dbgs() << "      Applying fixups.\n");
          for (auto &E : DA->edges()) {

            // Skip non-relocation edges.
            if (!E.isRelocation())
              continue;

            // Dispatch to LinkerImpl for fixup.
            if (auto Err = impl().applyFixup(*DA, E, AtomDataPtr))
              return Err;
          }

          // Point the atom's content to the fixed up buffer.
          DA->setContent(StringRef(AtomDataPtr, DA->getContent().size()));

          // Update atom end pointer.
          LastAtomEnd = AtomDataPtr + DA->getContent().size();
        }
      }

      // Zero pad the rest of the segment.
      LLVM_DEBUG({
        dbgs() << "    Zero padding end of segment from "
               << (const void *)LastAtomEnd << " to "
               << (const void *)((char *)SegMem.data() + SegMem.size()) << "\n";
      });
      while (LastAtomEnd != SegMem.data() + SegMem.size())
        *LastAtomEnd++ = 0;
    }

    return Error::success();
  }
};

/// Dead strips and replaces discarded definitions with external atoms.
///
/// Finds the set of nodes reachable from any node initially marked live
/// (nodes marked should-discard are treated as not live, even if they are
/// reachable). All nodes not marked as live at the end of this process,
/// are deleted. Nodes that are live, but marked should-discard are replaced
/// with external atoms and all edges to them are re-written.
void prune(AtomGraph &G);

Error addEHFrame(AtomGraph &G, Section &EHFrameSection,
                 StringRef EHFrameContent, JITTargetAddress EHFrameAddress,
                 Edge::Kind FDEToCIERelocKind, Edge::Kind FDEToTargetRelocKind);

} // end namespace jitlink
} // end namespace llvm

#undef DEBUG_TYPE // "jitlink"

#endif // LLVM_EXECUTIONENGINE_JITLINK_JITLINKGENERIC_H
