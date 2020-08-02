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

#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/MemoryBuffer.h"

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

JITLinkerBase::~JITLinkerBase() {}

void JITLinkerBase::linkPhase1(std::unique_ptr<JITLinkerBase> Self) {

  LLVM_DEBUG({
    dbgs() << "Building jitlink graph for new input "
           << Ctx->getObjectBuffer().getBufferIdentifier() << "...\n";
  });

  // Build the link graph.
  if (auto GraphOrErr = buildGraph(Ctx->getObjectBuffer()))
    G = std::move(*GraphOrErr);
  else
    return Ctx->notifyFailed(GraphOrErr.takeError());
  assert(G && "Graph should have been created by buildGraph above");

  LLVM_DEBUG({
    dbgs() << "Starting link phase 1 for graph " << G->getName() << "\n";
  });

  // Prune and optimize the graph.
  if (auto Err = runPasses(Passes.PrePrunePasses))
    return Ctx->notifyFailed(std::move(Err));

  LLVM_DEBUG({
    dbgs() << "Link graph \"" << G->getName() << "\" pre-pruning:\n";
    dumpGraph(dbgs());
  });

  prune(*G);

  LLVM_DEBUG({
    dbgs() << "Link graph \"" << G->getName() << "\" post-pruning:\n";
    dumpGraph(dbgs());
  });

  // Run post-pruning passes.
  if (auto Err = runPasses(Passes.PostPrunePasses))
    return Ctx->notifyFailed(std::move(Err));

  // Sort blocks into segments.
  auto Layout = layOutBlocks();

  // Allocate memory for segments.
  if (auto Err = allocateSegments(Layout))
    return Ctx->notifyFailed(std::move(Err));

  // Notify client that the defined symbols have been assigned addresses.
  LLVM_DEBUG(
      { dbgs() << "Resolving symbols defined in " << G->getName() << "\n"; });

  if (auto Err = Ctx->notifyResolved(*G))
    return Ctx->notifyFailed(std::move(Err));

  auto ExternalSymbols = getExternalSymbolNames();

  LLVM_DEBUG({
    dbgs() << "Issuing lookup for external symbols for " << G->getName()
           << " (may trigger materialization/linking of other graphs)...\n";
  });

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
  auto *TmpCtx = Ctx.get();
  TmpCtx->lookup(std::move(ExternalSymbols),
                 createLookupContinuation(
                     [S = std::move(Self), L = std::move(Layout)](
                         Expected<AsyncLookupResult> LookupResult) mutable {
                       auto &TmpSelf = *S;
                       TmpSelf.linkPhase2(std::move(S), std::move(LookupResult),
                                          std::move(L));
                     }));
}

void JITLinkerBase::linkPhase2(std::unique_ptr<JITLinkerBase> Self,
                               Expected<AsyncLookupResult> LR,
                               SegmentLayoutMap Layout) {

  LLVM_DEBUG({
    dbgs() << "Starting link phase 2 for graph " << G->getName() << "\n";
  });

  // If the lookup failed, bail out.
  if (!LR)
    return deallocateAndBailOut(LR.takeError());

  // Assign addresses to external addressables.
  applyLookupResult(*LR);

  // Copy block content to working memory.
  copyBlockContentToWorkingMemory(Layout, *Alloc);

  LLVM_DEBUG({
    dbgs() << "Link graph \"" << G->getName()
           << "\" before post-allocation passes:\n";
    dumpGraph(dbgs());
  });

  if (auto Err = runPasses(Passes.PostAllocationPasses))
    return deallocateAndBailOut(std::move(Err));

  LLVM_DEBUG({
    dbgs() << "Link graph \"" << G->getName() << "\" before copy-and-fixup:\n";
    dumpGraph(dbgs());
  });

  // Fix up block content.
  if (auto Err = fixUpBlocks(*G))
    return deallocateAndBailOut(std::move(Err));

  LLVM_DEBUG({
    dbgs() << "Link graph \"" << G->getName() << "\" after copy-and-fixup:\n";
    dumpGraph(dbgs());
  });

  if (auto Err = runPasses(Passes.PostFixupPasses))
    return deallocateAndBailOut(std::move(Err));

  // FIXME: Use move capture once we have c++14.
  auto *UnownedSelf = Self.release();
  auto Phase3Continuation = [UnownedSelf](Error Err) {
    std::unique_ptr<JITLinkerBase> Self(UnownedSelf);
    UnownedSelf->linkPhase3(std::move(Self), std::move(Err));
  };

  Alloc->finalizeAsync(std::move(Phase3Continuation));
}

void JITLinkerBase::linkPhase3(std::unique_ptr<JITLinkerBase> Self, Error Err) {

  LLVM_DEBUG({
    dbgs() << "Starting link phase 3 for graph " << G->getName() << "\n";
  });

  if (Err)
    return deallocateAndBailOut(std::move(Err));
  Ctx->notifyFinalized(std::move(Alloc));

  LLVM_DEBUG({ dbgs() << "Link of graph " << G->getName() << " complete\n"; });
}

Error JITLinkerBase::runPasses(LinkGraphPassList &Passes) {
  for (auto &P : Passes)
    if (auto Err = P(*G))
      return Err;
  return Error::success();
}

JITLinkerBase::SegmentLayoutMap JITLinkerBase::layOutBlocks() {

  SegmentLayoutMap Layout;

  /// Partition blocks based on permissions and content vs. zero-fill.
  for (auto *B : G->blocks()) {
    auto &SegLists = Layout[B->getSection().getProtectionFlags()];
    if (!B->isZeroFill())
      SegLists.ContentBlocks.push_back(B);
    else
      SegLists.ZeroFillBlocks.push_back(B);
  }

  /// Sort blocks within each list.
  for (auto &KV : Layout) {

    auto CompareBlocks = [](const Block *LHS, const Block *RHS) {
      // Sort by section, address and size
      if (LHS->getSection().getOrdinal() != RHS->getSection().getOrdinal())
        return LHS->getSection().getOrdinal() < RHS->getSection().getOrdinal();
      if (LHS->getAddress() != RHS->getAddress())
        return LHS->getAddress() < RHS->getAddress();
      return LHS->getSize() < RHS->getSize();
    };

    auto &SegLists = KV.second;
    llvm::sort(SegLists.ContentBlocks, CompareBlocks);
    llvm::sort(SegLists.ZeroFillBlocks, CompareBlocks);
  }

  LLVM_DEBUG({
    dbgs() << "Computed segment ordering:\n";
    for (auto &KV : Layout) {
      dbgs() << "  Segment "
             << static_cast<sys::Memory::ProtectionFlags>(KV.first) << ":\n";
      auto &SL = KV.second;
      for (auto &SIEntry :
           {std::make_pair(&SL.ContentBlocks, "content block"),
            std::make_pair(&SL.ZeroFillBlocks, "zero-fill block")}) {
        dbgs() << "    " << SIEntry.second << ":\n";
        for (auto *B : *SIEntry.first)
          dbgs() << "      " << *B << "\n";
      }
    }
  });

  return Layout;
}

Error JITLinkerBase::allocateSegments(const SegmentLayoutMap &Layout) {

  // Compute segment sizes and allocate memory.
  LLVM_DEBUG(dbgs() << "JIT linker requesting: { ");
  JITLinkMemoryManager::SegmentsRequestMap Segments;
  for (auto &KV : Layout) {
    auto &Prot = KV.first;
    auto &SegLists = KV.second;

    uint64_t SegAlign = 1;

    // Calculate segment content size.
    size_t SegContentSize = 0;
    for (auto *B : SegLists.ContentBlocks) {
      SegAlign = std::max(SegAlign, B->getAlignment());
      SegContentSize = alignToBlock(SegContentSize, *B);
      SegContentSize += B->getSize();
    }

    uint64_t SegZeroFillStart = SegContentSize;
    uint64_t SegZeroFillEnd = SegZeroFillStart;

    for (auto *B : SegLists.ZeroFillBlocks) {
      SegAlign = std::max(SegAlign, B->getAlignment());
      SegZeroFillEnd = alignToBlock(SegZeroFillEnd, *B);
      SegZeroFillEnd += B->getSize();
    }

    Segments[Prot] = {SegAlign, SegContentSize,
                      SegZeroFillEnd - SegZeroFillStart};

    LLVM_DEBUG({
      dbgs() << (&KV == &*Layout.begin() ? "" : "; ")
             << static_cast<sys::Memory::ProtectionFlags>(Prot)
             << ": alignment = " << SegAlign
             << ", content size = " << SegContentSize
             << ", zero-fill size = " << (SegZeroFillEnd - SegZeroFillStart);
    });
  }
  LLVM_DEBUG(dbgs() << " }\n");

  if (auto AllocOrErr = Ctx->getMemoryManager().allocate(Segments))
    Alloc = std::move(*AllocOrErr);
  else
    return AllocOrErr.takeError();

  LLVM_DEBUG({
    dbgs() << "JIT linker got memory (working -> target):\n";
    for (auto &KV : Layout) {
      auto Prot = static_cast<sys::Memory::ProtectionFlags>(KV.first);
      dbgs() << "  " << Prot << ": "
             << (const void *)Alloc->getWorkingMemory(Prot).data() << " -> "
             << formatv("{0:x16}", Alloc->getTargetMemory(Prot)) << "\n";
    }
  });

  // Update block target addresses.
  for (auto &KV : Layout) {
    auto &Prot = KV.first;
    auto &SL = KV.second;

    JITTargetAddress NextBlockAddr =
        Alloc->getTargetMemory(static_cast<sys::Memory::ProtectionFlags>(Prot));

    for (auto *SIList : {&SL.ContentBlocks, &SL.ZeroFillBlocks})
      for (auto *B : *SIList) {
        NextBlockAddr = alignToBlock(NextBlockAddr, *B);
        B->setAddress(NextBlockAddr);
        NextBlockAddr += B->getSize();
      }
  }

  return Error::success();
}

JITLinkContext::LookupMap JITLinkerBase::getExternalSymbolNames() const {
  // Identify unresolved external symbols.
  JITLinkContext::LookupMap UnresolvedExternals;
  for (auto *Sym : G->external_symbols()) {
    assert(Sym->getAddress() == 0 &&
           "External has already been assigned an address");
    assert(Sym->getName() != StringRef() && Sym->getName() != "" &&
           "Externals must be named");
    SymbolLookupFlags LookupFlags =
        Sym->getLinkage() == Linkage::Weak
            ? SymbolLookupFlags::WeaklyReferencedSymbol
            : SymbolLookupFlags::RequiredSymbol;
    UnresolvedExternals[Sym->getName()] = LookupFlags;
  }
  return UnresolvedExternals;
}

void JITLinkerBase::applyLookupResult(AsyncLookupResult Result) {
  for (auto *Sym : G->external_symbols()) {
    assert(Sym->getOffset() == 0 &&
           "External symbol is not at the start of its addressable block");
    assert(Sym->getAddress() == 0 && "Symbol already resolved");
    assert(!Sym->isDefined() && "Symbol being resolved is already defined");
    auto ResultI = Result.find(Sym->getName());
    if (ResultI != Result.end())
      Sym->getAddressable().setAddress(ResultI->second.getAddress());
    else
      assert(Sym->getLinkage() == Linkage::Weak &&
             "Failed to resolve non-weak reference");
  }

  LLVM_DEBUG({
    dbgs() << "Externals after applying lookup result:\n";
    for (auto *Sym : G->external_symbols())
      dbgs() << "  " << Sym->getName() << ": "
             << formatv("{0:x16}", Sym->getAddress()) << "\n";
  });
}

void JITLinkerBase::copyBlockContentToWorkingMemory(
    const SegmentLayoutMap &Layout, JITLinkMemoryManager::Allocation &Alloc) {

  LLVM_DEBUG(dbgs() << "Copying block content:\n");
  for (auto &KV : Layout) {
    auto &Prot = KV.first;
    auto &SegLayout = KV.second;

    auto SegMem =
        Alloc.getWorkingMemory(static_cast<sys::Memory::ProtectionFlags>(Prot));
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
        dbgs() << "      Bumped block pointer to " << (const void *)BlockDataPtr
               << " to meet block alignment " << B->getAlignment()
               << " and alignment offset " << B->getAlignmentOffset() << "\n";
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
}

void JITLinkerBase::deallocateAndBailOut(Error Err) {
  assert(Err && "Should not be bailing out on success value");
  assert(Alloc && "can not call deallocateAndBailOut before allocation");
  Ctx->notifyFailed(joinErrors(std::move(Err), Alloc->deallocate()));
}

void JITLinkerBase::dumpGraph(raw_ostream &OS) {
  assert(G && "Graph is not set yet");
  G->dump(dbgs(), [this](Edge::Kind K) { return getEdgeKindName(K); });
}

void prune(LinkGraph &G) {
  std::vector<Symbol *> Worklist;
  DenseSet<Block *> VisitedBlocks;

  // Build the initial worklist from all symbols initially live.
  for (auto *Sym : G.defined_symbols())
    if (Sym->isLive())
      Worklist.push_back(Sym);

  // Propagate live flags to all symbols reachable from the initial live set.
  while (!Worklist.empty()) {
    auto *Sym = Worklist.back();
    Worklist.pop_back();

    auto &B = Sym->getBlock();

    // Skip addressables that we've visited before.
    if (VisitedBlocks.count(&B))
      continue;

    VisitedBlocks.insert(&B);

    for (auto &E : Sym->getBlock().edges()) {
      // If the edge target is a defined symbol that is being newly marked live
      // then add it to the worklist.
      if (E.getTarget().isDefined() && !E.getTarget().isLive())
        Worklist.push_back(&E.getTarget());

      // Mark the target live.
      E.getTarget().setLive(true);
    }
  }

  // Collect all defined symbols to remove, then remove them.
  {
    LLVM_DEBUG(dbgs() << "Dead-stripping defined symbols:\n");
    std::vector<Symbol *> SymbolsToRemove;
    for (auto *Sym : G.defined_symbols())
      if (!Sym->isLive())
        SymbolsToRemove.push_back(Sym);
    for (auto *Sym : SymbolsToRemove) {
      LLVM_DEBUG(dbgs() << "  " << *Sym << "...\n");
      G.removeDefinedSymbol(*Sym);
    }
  }

  // Delete any unused blocks.
  {
    LLVM_DEBUG(dbgs() << "Dead-stripping blocks:\n");
    std::vector<Block *> BlocksToRemove;
    for (auto *B : G.blocks())
      if (!VisitedBlocks.count(B))
        BlocksToRemove.push_back(B);
    for (auto *B : BlocksToRemove) {
      LLVM_DEBUG(dbgs() << "  " << *B << "...\n");
      G.removeBlock(*B);
    }
  }

  // Collect all external symbols to remove, then remove them.
  {
    LLVM_DEBUG(dbgs() << "Removing unused external symbols:\n");
    std::vector<Symbol *> SymbolsToRemove;
    for (auto *Sym : G.external_symbols())
      if (!Sym->isLive())
        SymbolsToRemove.push_back(Sym);
    for (auto *Sym : SymbolsToRemove) {
      LLVM_DEBUG(dbgs() << "  " << *Sym << "...\n");
      G.removeExternalSymbol(*Sym);
    }
  }
}

} // end namespace jitlink
} // end namespace llvm
