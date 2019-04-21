//===--------- JITLinkGeneric.cpp - Generic JIT linker utilities ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic JITLinker utility class.
//
//===----------------------------------------------------------------------===//

#include "JITLinkGeneric.h"
#include "JITLink_EHFrameSupportImpl.h"

#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/MemoryBuffer.h"

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

JITLinkerBase::~JITLinkerBase() {}

void JITLinkerBase::linkPhase1(std::unique_ptr<JITLinkerBase> Self) {

  // Build the atom graph.
  if (auto GraphOrErr = buildGraph(Ctx->getObjectBuffer()))
    G = std::move(*GraphOrErr);
  else
    return Ctx->notifyFailed(GraphOrErr.takeError());
  assert(G && "Graph should have been created by buildGraph above");

  // Prune and optimize the graph.
  if (auto Err = runPasses(Passes.PrePrunePasses, *G))
    return Ctx->notifyFailed(std::move(Err));

  LLVM_DEBUG({
    dbgs() << "Atom graph \"" << G->getName() << "\" pre-pruning:\n";
    dumpGraph(dbgs());
  });

  prune(*G);

  LLVM_DEBUG({
    dbgs() << "Atom graph \"" << G->getName() << "\" post-pruning:\n";
    dumpGraph(dbgs());
  });

  // Run post-pruning passes.
  if (auto Err = runPasses(Passes.PostPrunePasses, *G))
    return Ctx->notifyFailed(std::move(Err));

  // Sort atoms into segments.
  layOutAtoms();

  // Allocate memory for segments.
  if (auto Err = allocateSegments(Layout))
    return Ctx->notifyFailed(std::move(Err));

  // Notify client that the defined atoms have been assigned addresses.
  Ctx->notifyResolved(*G);

  auto ExternalSymbols = getExternalSymbolNames();

  // We're about to hand off ownership of ourself to the continuation. Grab a
  // pointer to the context so that we can call it to initiate the lookup.
  //
  // FIXME: Once callee expressions are defined to be sequenced before argument
  // expressions (c++17) we can simplify all this to:
  //
  // Ctx->lookup(std::move(UnresolvedExternals),
  //             [Self=std::move(Self)](Expected<AsyncLookupResult> Result) {
  //               Self->linkPhase2(std::move(Self), std::move(Result));
  //             });
  //
  // FIXME: Use move capture once we have c++14.
  auto *TmpCtx = Ctx.get();
  auto *UnownedSelf = Self.release();
  auto Phase2Continuation =
      [UnownedSelf](Expected<AsyncLookupResult> LookupResult) {
        std::unique_ptr<JITLinkerBase> Self(UnownedSelf);
        UnownedSelf->linkPhase2(std::move(Self), std::move(LookupResult));
      };
  TmpCtx->lookup(std::move(ExternalSymbols), std::move(Phase2Continuation));
}

void JITLinkerBase::linkPhase2(std::unique_ptr<JITLinkerBase> Self,
                               Expected<AsyncLookupResult> LR) {
  // If the lookup failed, bail out.
  if (!LR)
    return Ctx->notifyFailed(LR.takeError());

  // Assign addresses to external atoms.
  applyLookupResult(*LR);

  LLVM_DEBUG({
    dbgs() << "Atom graph \"" << G->getName() << "\" before copy-and-fixup:\n";
    dumpGraph(dbgs());
  });

  // Copy atom content to working memory and fix up.
  if (auto Err = copyAndFixUpAllAtoms(Layout, *Alloc))
    return Ctx->notifyFailed(std::move(Err));

  LLVM_DEBUG({
    dbgs() << "Atom graph \"" << G->getName() << "\" after copy-and-fixup:\n";
    dumpGraph(dbgs());
  });

  if (auto Err = runPasses(Passes.PostFixupPasses, *G))
    return Ctx->notifyFailed(std::move(Err));

  // FIXME: Use move capture once we have c++14.
  auto *UnownedSelf = Self.release();
  auto Phase3Continuation = [UnownedSelf](Error Err) {
    std::unique_ptr<JITLinkerBase> Self(UnownedSelf);
    UnownedSelf->linkPhase3(std::move(Self), std::move(Err));
  };

  Alloc->finalizeAsync(std::move(Phase3Continuation));
}

void JITLinkerBase::linkPhase3(std::unique_ptr<JITLinkerBase> Self, Error Err) {
  if (Err)
    return Ctx->notifyFailed(std::move(Err));
  Ctx->notifyFinalized(std::move(Alloc));
}

Error JITLinkerBase::runPasses(AtomGraphPassList &Passes, AtomGraph &G) {
  for (auto &P : Passes)
    if (auto Err = P(G))
      return Err;
  return Error::success();
}

void JITLinkerBase::layOutAtoms() {
  // Group sections by protections, and whether or not they're zero-fill.
  for (auto &S : G->sections()) {

    // Skip empty sections.
    if (S.atoms_empty())
      continue;

    auto &SL = Layout[S.getProtectionFlags()];
    if (S.isZeroFill())
      SL.ZeroFillSections.push_back(SegmentLayout::SectionLayout(S));
    else
      SL.ContentSections.push_back(SegmentLayout::SectionLayout(S));
  }

  // Sort sections within the layout by ordinal.
  {
    auto CompareByOrdinal = [](const SegmentLayout::SectionLayout &LHS,
                               const SegmentLayout::SectionLayout &RHS) {
      return LHS.S->getSectionOrdinal() < RHS.S->getSectionOrdinal();
    };
    for (auto &KV : Layout) {
      auto &SL = KV.second;
      std::sort(SL.ContentSections.begin(), SL.ContentSections.end(),
                CompareByOrdinal);
      std::sort(SL.ZeroFillSections.begin(), SL.ZeroFillSections.end(),
                CompareByOrdinal);
    }
  }

  // Add atoms to the sections.
  for (auto &KV : Layout) {
    auto &SL = KV.second;
    for (auto *SIList : {&SL.ContentSections, &SL.ZeroFillSections}) {
      for (auto &SI : *SIList) {
        std::vector<DefinedAtom *> LayoutHeads;
        LayoutHeads.reserve(SI.S->atoms_size());

        // First build the list of layout-heads (i.e. "heads" of layout-next
        // chains).
        DenseSet<DefinedAtom *> AlreadyLayedOut;
        for (auto *DA : SI.S->atoms()) {
          if (AlreadyLayedOut.count(DA))
            continue;
          LayoutHeads.push_back(DA);
          while (DA->hasLayoutNext()) {
            auto &Next = DA->getLayoutNext();
            AlreadyLayedOut.insert(&Next);
            DA = &Next;
          }
        }

        // Now sort the list of layout heads by address.
        std::sort(LayoutHeads.begin(), LayoutHeads.end(),
                  [](const DefinedAtom *LHS, const DefinedAtom *RHS) {
                    return LHS->getAddress() < RHS->getAddress();
                  });

        // Now populate the SI.Atoms field by appending each of the chains.
        for (auto *DA : LayoutHeads) {
          SI.Atoms.push_back(DA);
          while (DA->hasLayoutNext()) {
            auto &Next = DA->getLayoutNext();
            SI.Atoms.push_back(&Next);
            DA = &Next;
          }
        }
      }
    }
  }

  LLVM_DEBUG({
    dbgs() << "Segment ordering:\n";
    for (auto &KV : Layout) {
      dbgs() << "  Segment "
             << static_cast<sys::Memory::ProtectionFlags>(KV.first) << ":\n";
      auto &SL = KV.second;
      for (auto &SIEntry :
           {std::make_pair(&SL.ContentSections, "content sections"),
            std::make_pair(&SL.ZeroFillSections, "zero-fill sections")}) {
        auto &SIList = *SIEntry.first;
        dbgs() << "    " << SIEntry.second << ":\n";
        for (auto &SI : SIList) {
          dbgs() << "      " << SI.S->getName() << ":\n";
          for (auto *DA : SI.Atoms)
            dbgs() << "        " << *DA << "\n";
        }
      }
    }
  });
}

Error JITLinkerBase::allocateSegments(const SegmentLayoutMap &Layout) {

  // Compute segment sizes and allocate memory.
  LLVM_DEBUG(dbgs() << "JIT linker requesting: { ");
  JITLinkMemoryManager::SegmentsRequestMap Segments;
  for (auto &KV : Layout) {
    auto &Prot = KV.first;
    auto &SegLayout = KV.second;

    // Calculate segment content size.
    size_t SegContentSize = 0;
    for (auto &SI : SegLayout.ContentSections) {
      assert(!SI.S->atoms_empty() && "Sections in layout must not be empty");
      assert(!SI.Atoms.empty() && "Section layouts must not be empty");
      for (auto *DA : SI.Atoms) {
        SegContentSize = alignTo(SegContentSize, DA->getAlignment());
        SegContentSize += DA->getSize();
      }
    }

    // Get segment content alignment.
    unsigned SegContentAlign = 1;
    if (!SegLayout.ContentSections.empty())
      SegContentAlign =
          SegLayout.ContentSections.front().Atoms.front()->getAlignment();

    // Calculate segment zero-fill size.
    uint64_t SegZeroFillSize = 0;
    for (auto &SI : SegLayout.ZeroFillSections) {
      assert(!SI.S->atoms_empty() && "Sections in layout must not be empty");
      assert(!SI.Atoms.empty() && "Section layouts must not be empty");
      for (auto *DA : SI.Atoms) {
        SegZeroFillSize = alignTo(SegZeroFillSize, DA->getAlignment());
        SegZeroFillSize += DA->getSize();
      }
    }

    // Calculate segment zero-fill alignment.
    uint32_t SegZeroFillAlign = 1;
    if (!SegLayout.ZeroFillSections.empty())
      SegZeroFillAlign =
          SegLayout.ZeroFillSections.front().Atoms.front()->getAlignment();

    if (SegContentSize == 0)
      SegContentAlign = SegZeroFillAlign;

    if (SegContentAlign % SegZeroFillAlign != 0)
      return make_error<JITLinkError>("First content atom alignment does not "
                                      "accommodate first zero-fill atom "
                                      "alignment");

    Segments[Prot] = {SegContentSize, SegContentAlign, SegZeroFillSize,
                      SegZeroFillAlign};

    LLVM_DEBUG({
      dbgs() << (&KV == &*Layout.begin() ? "" : "; ")
             << static_cast<sys::Memory::ProtectionFlags>(Prot) << ": "
             << SegContentSize << " content bytes (alignment "
             << SegContentAlign << ") + " << SegZeroFillSize
             << " zero-fill bytes (alignment " << SegZeroFillAlign << ")";
    });
  }
  LLVM_DEBUG(dbgs() << " }\n");

  if (auto AllocOrErr = Ctx->getMemoryManager().allocate(Segments))
    Alloc = std::move(*AllocOrErr);
  else
    return AllocOrErr.takeError();

  LLVM_DEBUG({
    dbgs() << "JIT linker got working memory:\n";
    for (auto &KV : Layout) {
      auto Prot = static_cast<sys::Memory::ProtectionFlags>(KV.first);
      dbgs() << "  " << Prot << ": "
             << (const void *)Alloc->getWorkingMemory(Prot).data() << "\n";
    }
  });

  // Update atom target addresses.
  for (auto &KV : Layout) {
    auto &Prot = KV.first;
    auto &SL = KV.second;

    JITTargetAddress AtomTargetAddr =
        Alloc->getTargetMemory(static_cast<sys::Memory::ProtectionFlags>(Prot));

    for (auto *SIList : {&SL.ContentSections, &SL.ZeroFillSections})
      for (auto &SI : *SIList)
        for (auto *DA : SI.Atoms) {
          AtomTargetAddr = alignTo(AtomTargetAddr, DA->getAlignment());
          DA->setAddress(AtomTargetAddr);
          AtomTargetAddr += DA->getSize();
        }
  }

  return Error::success();
}

DenseSet<StringRef> JITLinkerBase::getExternalSymbolNames() const {
  // Identify unresolved external atoms.
  DenseSet<StringRef> UnresolvedExternals;
  for (auto *DA : G->external_atoms()) {
    assert(DA->getAddress() == 0 &&
           "External has already been assigned an address");
    assert(DA->getName() != StringRef() && DA->getName() != "" &&
           "Externals must be named");
    UnresolvedExternals.insert(DA->getName());
  }
  return UnresolvedExternals;
}

void JITLinkerBase::applyLookupResult(AsyncLookupResult Result) {
  for (auto &KV : Result) {
    Atom &A = G->getAtomByName(KV.first);
    assert(A.getAddress() == 0 && "Atom already resolved");
    A.setAddress(KV.second.getAddress());
  }

  assert(llvm::all_of(G->external_atoms(),
                      [](Atom *A) { return A->getAddress() != 0; }) &&
         "All atoms should have been resolved by this point");
}

void JITLinkerBase::dumpGraph(raw_ostream &OS) {
  assert(G && "Graph is not set yet");
  G->dump(dbgs(), [this](Edge::Kind K) { return getEdgeKindName(K); });
}

void prune(AtomGraph &G) {
  std::vector<DefinedAtom *> Worklist;
  DenseMap<DefinedAtom *, std::vector<Edge *>> EdgesToUpdate;

  // Build the initial worklist from all atoms initially live.
  for (auto *DA : G.defined_atoms()) {
    if (!DA->isLive() || DA->shouldDiscard())
      continue;

    for (auto &E : DA->edges()) {
      if (!E.getTarget().isDefined())
        continue;

      auto &EDT = static_cast<DefinedAtom &>(E.getTarget());

      if (EDT.shouldDiscard())
        EdgesToUpdate[&EDT].push_back(&E);
      else if (E.isKeepAlive() && !EDT.isLive())
        Worklist.push_back(&EDT);
    }
  }

  // Propagate live flags to all atoms reachable from the initial live set.
  while (!Worklist.empty()) {
    DefinedAtom &NextLive = *Worklist.back();
    Worklist.pop_back();

    assert(!NextLive.shouldDiscard() &&
           "should-discard nodes should never make it into the worklist");

    // If this atom has already been marked as live, or is marked to be
    // discarded, then skip it.
    if (NextLive.isLive())
      continue;

    // Otherwise set it as live and add any non-live atoms that it points to
    // to the worklist.
    NextLive.setLive(true);

    for (auto &E : NextLive.edges()) {
      if (!E.getTarget().isDefined())
        continue;

      auto &EDT = static_cast<DefinedAtom &>(E.getTarget());

      if (EDT.shouldDiscard())
        EdgesToUpdate[&EDT].push_back(&E);
      else if (E.isKeepAlive() && !EDT.isLive())
        Worklist.push_back(&EDT);
    }
  }

  // Collect atoms to remove, then remove them from the graph.
  std::vector<DefinedAtom *> AtomsToRemove;
  for (auto *DA : G.defined_atoms())
    if (DA->shouldDiscard() || !DA->isLive())
      AtomsToRemove.push_back(DA);

  LLVM_DEBUG(dbgs() << "Pruning atoms:\n");
  for (auto *DA : AtomsToRemove) {
    LLVM_DEBUG(dbgs() << "  " << *DA << "... ");

    // Check whether we need to replace this atom with an external atom.
    //
    // We replace if all of the following hold:
    //   (1) The atom is marked should-discard,
    //   (2) it is live, and
    //   (3) it has edges pointing to it.
    //
    // Otherwise we simply delete the atom.
    bool ReplaceWithExternal = DA->isLive() && DA->shouldDiscard();
    std::vector<Edge *> *EdgesToUpdateForDA = nullptr;
    if (ReplaceWithExternal) {
      auto ETUItr = EdgesToUpdate.find(DA);
      if (ETUItr == EdgesToUpdate.end())
        ReplaceWithExternal = false;
      else
        EdgesToUpdateForDA = &ETUItr->second;
    }

    G.removeDefinedAtom(*DA);

    if (ReplaceWithExternal) {
      assert(EdgesToUpdateForDA &&
             "Replacing atom: There should be edges to update");

      auto &ExternalReplacement = G.addExternalAtom(DA->getName());
      for (auto *EdgeToUpdate : *EdgesToUpdateForDA)
        EdgeToUpdate->setTarget(ExternalReplacement);
      LLVM_DEBUG(dbgs() << "replaced with " << ExternalReplacement << "\n");
    } else
      LLVM_DEBUG(dbgs() << "deleted\n");
  }

  // Finally, discard any absolute symbols that were marked should-discard.
  {
    std::vector<Atom *> AbsoluteAtomsToRemove;
    for (auto *A : G.absolute_atoms())
      if (A->shouldDiscard() || A->isLive())
        AbsoluteAtomsToRemove.push_back(A);
    for (auto *A : AbsoluteAtomsToRemove)
      G.removeAbsoluteAtom(*A);
  }
}

} // end namespace jitlink
} // end namespace llvm
