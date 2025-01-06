//===------------- JITLink.cpp - Core Run-time JIT linker APIs ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/ExecutionEngine/JITLink/COFF.h"
#include "llvm/ExecutionEngine/JITLink/ELF.h"
#include "llvm/ExecutionEngine/JITLink/MachO.h"
#include "llvm/ExecutionEngine/JITLink/aarch64.h"
#include "llvm/ExecutionEngine/JITLink/i386.h"
#include "llvm/ExecutionEngine/JITLink/loongarch.h"
#include "llvm/ExecutionEngine/JITLink/x86_64.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

#define DEBUG_TYPE "jitlink"

namespace {

enum JITLinkErrorCode { GenericJITLinkError = 1 };

// FIXME: This class is only here to support the transition to llvm::Error. It
// will be removed once this transition is complete. Clients should prefer to
// deal with the Error value directly, rather than converting to error_code.
class JITLinkerErrorCategory : public std::error_category {
public:
  const char *name() const noexcept override { return "runtimedyld"; }

  std::string message(int Condition) const override {
    switch (static_cast<JITLinkErrorCode>(Condition)) {
    case GenericJITLinkError:
      return "Generic JITLink error";
    }
    llvm_unreachable("Unrecognized JITLinkErrorCode");
  }
};

} // namespace

namespace llvm {
namespace jitlink {

char JITLinkError::ID = 0;

void JITLinkError::log(raw_ostream &OS) const { OS << ErrMsg; }

std::error_code JITLinkError::convertToErrorCode() const {
  static JITLinkerErrorCategory TheJITLinkerErrorCategory;
  return std::error_code(GenericJITLinkError, TheJITLinkerErrorCategory);
}

const char *getGenericEdgeKindName(Edge::Kind K) {
  switch (K) {
  case Edge::Invalid:
    return "INVALID RELOCATION";
  case Edge::KeepAlive:
    return "Keep-Alive";
  default:
    return "<Unrecognized edge kind>";
  }
}

const char *getLinkageName(Linkage L) {
  switch (L) {
  case Linkage::Strong:
    return "strong";
  case Linkage::Weak:
    return "weak";
  }
  llvm_unreachable("Unrecognized llvm.jitlink.Linkage enum");
}

const char *getScopeName(Scope S) {
  switch (S) {
  case Scope::Default:
    return "default";
  case Scope::Hidden:
    return "hidden";
  case Scope::SideEffectsOnly:
    return "side-effects-only";
  case Scope::Local:
    return "local";
  }
  llvm_unreachable("Unrecognized llvm.jitlink.Scope enum");
}

bool isCStringBlock(Block &B) {
  if (B.getSize() == 0) // Empty blocks are not valid C-strings.
    return false;

  // Zero-fill blocks of size one are valid empty strings.
  if (B.isZeroFill())
    return B.getSize() == 1;

  for (size_t I = 0; I != B.getSize() - 1; ++I)
    if (B.getContent()[I] == '\0')
      return false;

  return B.getContent()[B.getSize() - 1] == '\0';
}

raw_ostream &operator<<(raw_ostream &OS, const Block &B) {
  return OS << B.getAddress() << " -- " << (B.getAddress() + B.getSize())
            << ": "
            << "size = " << formatv("{0:x8}", B.getSize()) << ", "
            << (B.isZeroFill() ? "zero-fill" : "content")
            << ", align = " << B.getAlignment()
            << ", align-ofs = " << B.getAlignmentOffset()
            << ", section = " << B.getSection().getName();
}

raw_ostream &operator<<(raw_ostream &OS, const Symbol &Sym) {
  OS << Sym.getAddress() << " (" << (Sym.isDefined() ? "block" : "addressable")
     << " + " << formatv("{0:x8}", Sym.getOffset())
     << "): size: " << formatv("{0:x8}", Sym.getSize())
     << ", linkage: " << formatv("{0:6}", getLinkageName(Sym.getLinkage()))
     << ", scope: " << formatv("{0:8}", getScopeName(Sym.getScope())) << ", "
     << (Sym.isLive() ? "live" : "dead") << "  -   "
     << (Sym.hasName() ? *Sym.getName() : "<anonymous symbol>");
  return OS;
}

void printEdge(raw_ostream &OS, const Block &B, const Edge &E,
               StringRef EdgeKindName) {
  OS << "edge@" << B.getAddress() + E.getOffset() << ": " << B.getAddress()
     << " + " << formatv("{0:x}", E.getOffset()) << " -- " << EdgeKindName
     << " -> ";

  auto &TargetSym = E.getTarget();
  if (TargetSym.hasName())
    OS << TargetSym.getName();
  else {
    auto &TargetBlock = TargetSym.getBlock();
    auto &TargetSec = TargetBlock.getSection();
    orc::ExecutorAddr SecAddress(~uint64_t(0));
    for (auto *B : TargetSec.blocks())
      if (B->getAddress() < SecAddress)
        SecAddress = B->getAddress();

    orc::ExecutorAddrDiff SecDelta = TargetSym.getAddress() - SecAddress;
    OS << TargetSym.getAddress() << " (section " << TargetSec.getName();
    if (SecDelta)
      OS << " + " << formatv("{0:x}", SecDelta);
    OS << " / block " << TargetBlock.getAddress();
    if (TargetSym.getOffset())
      OS << " + " << formatv("{0:x}", TargetSym.getOffset());
    OS << ")";
  }

  if (E.getAddend() != 0)
    OS << " + " << E.getAddend();
}

Section::~Section() {
  for (auto *Sym : Symbols)
    Sym->~Symbol();
  for (auto *B : Blocks)
    B->~Block();
}

LinkGraph::~LinkGraph() {
  for (auto *Sym : AbsoluteSymbols) {
    Sym->~Symbol();
  }
  for (auto *Sym : external_symbols()) {
    Sym->~Symbol();
  }
  ExternalSymbols.clear();
}

std::vector<Block *> LinkGraph::splitBlockImpl(std::vector<Block *> Blocks,
                                               SplitBlockCache *Cache) {
  assert(!Blocks.empty() && "Blocks must at least contain the original block");

  // Fix up content of all blocks.
  ArrayRef<char> Content = Blocks.front()->getContent();
  for (size_t I = 0; I != Blocks.size() - 1; ++I) {
    Blocks[I]->setContent(
        Content.slice(Blocks[I]->getAddress() - Blocks[0]->getAddress(),
                      Blocks[I + 1]->getAddress() - Blocks[I]->getAddress()));
  }
  Blocks.back()->setContent(
      Content.slice(Blocks.back()->getAddress() - Blocks[0]->getAddress()));
  bool IsMutable = Blocks[0]->ContentMutable;
  for (auto *B : Blocks)
    B->ContentMutable = IsMutable;

  // Transfer symbols.
  {
    SplitBlockCache LocalBlockSymbolsCache;
    if (!Cache)
      Cache = &LocalBlockSymbolsCache;

    // Build cache if required.
    if (*Cache == std::nullopt) {
      *Cache = SplitBlockCache::value_type();

      for (auto *Sym : Blocks[0]->getSection().symbols())
        if (&Sym->getBlock() == Blocks[0])
          (*Cache)->push_back(Sym);
      llvm::sort(**Cache, [](const Symbol *LHS, const Symbol *RHS) {
        return LHS->getAddress() > RHS->getAddress();
      });
    }

    auto TransferSymbol = [](Symbol &Sym, Block &B) {
      Sym.setOffset(Sym.getAddress() - B.getAddress());
      Sym.setBlock(B);
      if (Sym.getSize() > B.getSize())
        Sym.setSize(B.getSize() - Sym.getOffset());
    };

    // Transfer symbols to all blocks except the last one.
    for (size_t I = 0; I != Blocks.size() - 1; ++I) {
      if ((*Cache)->empty())
        break;
      while (!(*Cache)->empty() &&
             (*Cache)->back()->getAddress() < Blocks[I + 1]->getAddress()) {
        TransferSymbol(*(*Cache)->back(), *Blocks[I]);
        (*Cache)->pop_back();
      }
    }
    // Transfer symbols to the last block, checking that all are in-range.
    while (!(*Cache)->empty()) {
      auto &Sym = *(*Cache)->back();
      (*Cache)->pop_back();
      assert(Sym.getAddress() >= Blocks.back()->getAddress() &&
             "Symbol address preceeds block");
      assert(Sym.getAddress() <= Blocks.back()->getRange().End &&
             "Symbol address starts past end of block");
      TransferSymbol(Sym, *Blocks.back());
    }
  }

  // Transfer edges.
  auto &Edges = Blocks[0]->Edges;
  llvm::sort(Edges, [](const Edge &LHS, const Edge &RHS) {
    return LHS.getOffset() < RHS.getOffset();
  });

  for (size_t I = Blocks.size() - 1; I != 0; --I) {

    // If all edges have been transferred then bail out.
    if (Edges.empty())
      break;

    Edge::OffsetT Delta = Blocks[I]->getAddress() - Blocks[0]->getAddress();

    // If no edges to move for this block then move to the next one.
    if (Edges.back().getOffset() < Delta)
      continue;

    size_t EI = Edges.size() - 1;
    while (EI != 0 && Edges[EI - 1].getOffset() >= Delta)
      --EI;

    for (size_t J = EI; J != Edges.size(); ++J) {
      Blocks[I]->Edges.push_back(std::move(Edges[J]));
      Blocks[I]->Edges.back().setOffset(Blocks[I]->Edges.back().getOffset() -
                                        Delta);
    }

    while (Edges.size() > EI)
      Edges.pop_back();
  }

  return Blocks;
}

void LinkGraph::dump(raw_ostream &OS) {
  DenseMap<Block *, std::vector<Symbol *>> BlockSymbols;

  // Map from blocks to the symbols pointing at them.
  for (auto *Sym : defined_symbols())
    BlockSymbols[&Sym->getBlock()].push_back(Sym);

  // For each block, sort its symbols by something approximating
  // relevance.
  for (auto &KV : BlockSymbols)
    llvm::sort(KV.second, [](const Symbol *LHS, const Symbol *RHS) {
      if (LHS->getOffset() != RHS->getOffset())
        return LHS->getOffset() < RHS->getOffset();
      if (LHS->getLinkage() != RHS->getLinkage())
        return LHS->getLinkage() < RHS->getLinkage();
      if (LHS->getScope() != RHS->getScope())
        return LHS->getScope() < RHS->getScope();
      if (LHS->hasName()) {
        if (!RHS->hasName())
          return true;
        return LHS->getName() < RHS->getName();
      }
      return false;
    });

  std::vector<Section *> SortedSections;
  for (auto &Sec : sections())
    SortedSections.push_back(&Sec);
  llvm::sort(SortedSections, [](const Section *LHS, const Section *RHS) {
    return LHS->getName() < RHS->getName();
  });

  for (auto *Sec : SortedSections) {
    OS << "section " << Sec->getName() << ":\n\n";

    std::vector<Block *> SortedBlocks;
    llvm::copy(Sec->blocks(), std::back_inserter(SortedBlocks));
    llvm::sort(SortedBlocks, [](const Block *LHS, const Block *RHS) {
      return LHS->getAddress() < RHS->getAddress();
    });

    for (auto *B : SortedBlocks) {
      OS << "  block " << B->getAddress()
         << " size = " << formatv("{0:x8}", B->getSize())
         << ", align = " << B->getAlignment()
         << ", alignment-offset = " << B->getAlignmentOffset();
      if (B->isZeroFill())
        OS << ", zero-fill";
      OS << "\n";

      auto BlockSymsI = BlockSymbols.find(B);
      if (BlockSymsI != BlockSymbols.end()) {
        OS << "    symbols:\n";
        auto &Syms = BlockSymsI->second;
        for (auto *Sym : Syms)
          OS << "      " << *Sym << "\n";
      } else
        OS << "    no symbols\n";

      if (!B->edges_empty()) {
        OS << "    edges:\n";
        std::vector<Edge> SortedEdges;
        llvm::copy(B->edges(), std::back_inserter(SortedEdges));
        llvm::sort(SortedEdges, [](const Edge &LHS, const Edge &RHS) {
          return LHS.getOffset() < RHS.getOffset();
        });
        for (auto &E : SortedEdges) {
          OS << "      " << B->getFixupAddress(E) << " (block + "
             << formatv("{0:x8}", E.getOffset()) << "), addend = ";
          if (E.getAddend() >= 0)
            OS << formatv("+{0:x8}", E.getAddend());
          else
            OS << formatv("-{0:x8}", -E.getAddend());
          OS << ", kind = " << getEdgeKindName(E.getKind()) << ", target = ";
          if (E.getTarget().hasName())
            OS << E.getTarget().getName();
          else
            OS << "addressable@"
               << formatv("{0:x16}", E.getTarget().getAddress()) << "+"
               << formatv("{0:x8}", E.getTarget().getOffset());
          OS << "\n";
        }
      } else
        OS << "    no edges\n";
      OS << "\n";
    }
  }

  OS << "Absolute symbols:\n";
  if (!absolute_symbols().empty()) {
    for (auto *Sym : absolute_symbols())
      OS << "  " << Sym->getAddress() << ": " << *Sym << "\n";
  } else
    OS << "  none\n";

  OS << "\nExternal symbols:\n";
  if (!external_symbols().empty()) {
    for (auto *Sym : external_symbols())
      OS << "  " << Sym->getAddress() << ": " << *Sym
         << (Sym->isWeaklyReferenced() ? " (weakly referenced)" : "") << "\n";
  } else
    OS << "  none\n";
}

raw_ostream &operator<<(raw_ostream &OS, const SymbolLookupFlags &LF) {
  switch (LF) {
  case SymbolLookupFlags::RequiredSymbol:
    return OS << "RequiredSymbol";
  case SymbolLookupFlags::WeaklyReferencedSymbol:
    return OS << "WeaklyReferencedSymbol";
  }
  llvm_unreachable("Unrecognized lookup flags");
}

void JITLinkAsyncLookupContinuation::anchor() {}

JITLinkContext::~JITLinkContext() = default;

bool JITLinkContext::shouldAddDefaultTargetPasses(const Triple &TT) const {
  return true;
}

LinkGraphPassFunction JITLinkContext::getMarkLivePass(const Triple &TT) const {
  return LinkGraphPassFunction();
}

Error JITLinkContext::modifyPassConfig(LinkGraph &G,
                                       PassConfiguration &Config) {
  return Error::success();
}

Error markAllSymbolsLive(LinkGraph &G) {
  for (auto *Sym : G.defined_symbols())
    Sym->setLive(true);
  return Error::success();
}

Error makeTargetOutOfRangeError(const LinkGraph &G, const Block &B,
                                const Edge &E) {
  std::string ErrMsg;
  {
    raw_string_ostream ErrStream(ErrMsg);
    Section &Sec = B.getSection();
    ErrStream << "In graph " << G.getName() << ", section " << Sec.getName()
              << ": relocation target ";
    if (E.getTarget().hasName()) {
      ErrStream << "\"" << E.getTarget().getName() << "\"";
    } else
      ErrStream << E.getTarget().getBlock().getSection().getName() << " + "
                << formatv("{0:x}", E.getOffset());
    ErrStream << " at address " << formatv("{0:x}", E.getTarget().getAddress())
              << " is out of range of " << G.getEdgeKindName(E.getKind())
              << " fixup at " << formatv("{0:x}", B.getFixupAddress(E)) << " (";

    Symbol *BestSymbolForBlock = nullptr;
    for (auto *Sym : Sec.symbols())
      if (&Sym->getBlock() == &B && Sym->hasName() && Sym->getOffset() == 0 &&
          (!BestSymbolForBlock ||
           Sym->getScope() < BestSymbolForBlock->getScope() ||
           Sym->getLinkage() < BestSymbolForBlock->getLinkage()))
        BestSymbolForBlock = Sym;

    if (BestSymbolForBlock)
      ErrStream << BestSymbolForBlock->getName() << ", ";
    else
      ErrStream << "<anonymous block> @ ";

    ErrStream << formatv("{0:x}", B.getAddress()) << " + "
              << formatv("{0:x}", E.getOffset()) << ")";
  }
  return make_error<JITLinkError>(std::move(ErrMsg));
}

Error makeAlignmentError(llvm::orc::ExecutorAddr Loc, uint64_t Value, int N,
                         const Edge &E) {
  return make_error<JITLinkError>("0x" + llvm::utohexstr(Loc.getValue()) +
                                  " improper alignment for relocation " +
                                  formatv("{0:d}", E.getKind()) + ": 0x" +
                                  llvm::utohexstr(Value) +
                                  " is not aligned to " + Twine(N) + " bytes");
}

AnonymousPointerCreator getAnonymousPointerCreator(const Triple &TT) {
  switch (TT.getArch()) {
  case Triple::aarch64:
    return aarch64::createAnonymousPointer;
  case Triple::x86_64:
    return x86_64::createAnonymousPointer;
  case Triple::x86:
    return i386::createAnonymousPointer;
  case Triple::loongarch32:
  case Triple::loongarch64:
    return loongarch::createAnonymousPointer;
  default:
    return nullptr;
  }
}

PointerJumpStubCreator getPointerJumpStubCreator(const Triple &TT) {
  switch (TT.getArch()) {
  case Triple::aarch64:
    return aarch64::createAnonymousPointerJumpStub;
  case Triple::x86_64:
    return x86_64::createAnonymousPointerJumpStub;
  case Triple::x86:
    return i386::createAnonymousPointerJumpStub;
  case Triple::loongarch32:
  case Triple::loongarch64:
    return loongarch::createAnonymousPointerJumpStub;
  default:
    return nullptr;
  }
}

Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromObject(MemoryBufferRef ObjectBuffer,
                          std::shared_ptr<orc::SymbolStringPool> SSP) {
  auto Magic = identify_magic(ObjectBuffer.getBuffer());
  switch (Magic) {
  case file_magic::macho_object:
    return createLinkGraphFromMachOObject(ObjectBuffer, std::move(SSP));
  case file_magic::elf_relocatable:
    return createLinkGraphFromELFObject(ObjectBuffer, std::move(SSP));
  case file_magic::coff_object:
    return createLinkGraphFromCOFFObject(ObjectBuffer, std::move(SSP));
  default:
    return make_error<JITLinkError>("Unsupported file format");
  };
}

std::unique_ptr<LinkGraph>
absoluteSymbolsLinkGraph(const Triple &TT,
                         std::shared_ptr<orc::SymbolStringPool> SSP,
                         orc::SymbolMap Symbols) {
  unsigned PointerSize;
  endianness Endianness =
      TT.isLittleEndian() ? endianness::little : endianness::big;
  switch (TT.getArch()) {
  case Triple::aarch64:
  case llvm::Triple::riscv64:
  case Triple::x86_64:
    PointerSize = 8;
    break;
  case llvm::Triple::arm:
  case llvm::Triple::riscv32:
  case llvm::Triple::x86:
    PointerSize = 4;
    break;
  default:
    llvm::report_fatal_error("unhandled target architecture");
  }

  static std::atomic<uint64_t> Counter = {0};
  auto Index = Counter.fetch_add(1, std::memory_order_relaxed);
  auto G = std::make_unique<LinkGraph>(
      "<Absolute Symbols " + std::to_string(Index) + ">", std::move(SSP), TT,
      PointerSize, Endianness, /*GetEdgeKindName=*/nullptr);
  for (auto &[Name, Def] : Symbols) {
    auto &Sym =
        G->addAbsoluteSymbol(*Name, Def.getAddress(), /*Size=*/0,
                             Linkage::Strong, Scope::Default, /*IsLive=*/true);
    Sym.setCallable(Def.getFlags().isCallable());
  }

  return G;
}

void link(std::unique_ptr<LinkGraph> G, std::unique_ptr<JITLinkContext> Ctx) {
  switch (G->getTargetTriple().getObjectFormat()) {
  case Triple::MachO:
    return link_MachO(std::move(G), std::move(Ctx));
  case Triple::ELF:
    return link_ELF(std::move(G), std::move(Ctx));
  case Triple::COFF:
    return link_COFF(std::move(G), std::move(Ctx));
  default:
    Ctx->notifyFailed(make_error<JITLinkError>("Unsupported object format"));
  };
}

} // end namespace jitlink
} // end namespace llvm
