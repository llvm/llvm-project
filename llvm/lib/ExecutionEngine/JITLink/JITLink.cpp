//===------------- JITLink.cpp - Core Run-time JIT linker APIs ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

#include "llvm/BinaryFormat/Magic.h"
#include "llvm/ExecutionEngine/JITLink/ELF.h"
#include "llvm/ExecutionEngine/JITLink/MachO.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
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

static ManagedStatic<JITLinkerErrorCategory> JITLinkerErrorCategory;

} // namespace

namespace llvm {
namespace jitlink {

char JITLinkError::ID = 0;

void JITLinkError::log(raw_ostream &OS) const { OS << ErrMsg << "\n"; }

std::error_code JITLinkError::convertToErrorCode() const {
  return std::error_code(GenericJITLinkError, *JITLinkerErrorCategory);
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
  case Scope::Local:
    return "local";
  }
  llvm_unreachable("Unrecognized llvm.jitlink.Scope enum");
}

raw_ostream &operator<<(raw_ostream &OS, const Block &B) {
  return OS << formatv("{0:x16}", B.getAddress()) << " -- "
            << formatv("{0:x16}", B.getAddress() + B.getSize()) << ": "
            << "size = " << formatv("{0:x}", B.getSize()) << ", "
            << (B.isZeroFill() ? "zero-fill" : "content")
            << ", align = " << B.getAlignment()
            << ", align-ofs = " << B.getAlignmentOffset()
            << ", section = " << B.getSection().getName();
}

raw_ostream &operator<<(raw_ostream &OS, const Symbol &Sym) {
  OS << "<";
  if (Sym.getName().empty())
    OS << "*anon*";
  else
    OS << Sym.getName();
  OS << ": flags = ";
  switch (Sym.getLinkage()) {
  case Linkage::Strong:
    OS << 'S';
    break;
  case Linkage::Weak:
    OS << 'W';
    break;
  }
  switch (Sym.getScope()) {
  case Scope::Default:
    OS << 'D';
    break;
  case Scope::Hidden:
    OS << 'H';
    break;
  case Scope::Local:
    OS << 'L';
    break;
  }
  OS << (Sym.isLive() ? '+' : '-')
     << ", size = " << formatv("{0:x}", Sym.getSize())
     << ", addr = " << formatv("{0:x16}", Sym.getAddress()) << " ("
     << formatv("{0:x16}", Sym.getAddressable().getAddress()) << " + "
     << formatv("{0:x}", Sym.getOffset());
  if (Sym.isDefined())
    OS << " " << Sym.getBlock().getSection().getName();
  OS << ")>";
  return OS;
}

void printEdge(raw_ostream &OS, const Block &B, const Edge &E,
               StringRef EdgeKindName) {
  OS << "edge@" << formatv("{0:x16}", B.getAddress() + E.getOffset()) << ": "
     << formatv("{0:x16}", B.getAddress()) << " + "
     << formatv("{0:x}", E.getOffset()) << " -- " << EdgeKindName << " -> ";

  auto &TargetSym = E.getTarget();
  if (TargetSym.hasName())
    OS << TargetSym.getName();
  else {
    auto &TargetBlock = TargetSym.getBlock();
    auto &TargetSec = TargetBlock.getSection();
    JITTargetAddress SecAddress = ~JITTargetAddress(0);
    for (auto *B : TargetSec.blocks())
      if (B->getAddress() < SecAddress)
        SecAddress = B->getAddress();

    JITTargetAddress SecDelta = TargetSym.getAddress() - SecAddress;
    OS << formatv("{0:x16}", TargetSym.getAddress()) << " (section "
       << TargetSec.getName();
    if (SecDelta)
      OS << " + " << formatv("{0:x}", SecDelta);
    OS << " / block " << formatv("{0:x16}", TargetBlock.getAddress());
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

Block &LinkGraph::splitBlock(Block &B, size_t SplitIndex,
                             SplitBlockCache *Cache) {

  assert(SplitIndex > 0 && "splitBlock can not be called with SplitIndex == 0");

  // If the split point covers all of B then just return B.
  if (SplitIndex == B.getSize())
    return B;

  assert(SplitIndex < B.getSize() && "SplitIndex out of range");

  // Create the new block covering [ 0, SplitIndex ).
  auto &NewBlock =
      B.isZeroFill()
          ? createZeroFillBlock(B.getSection(), SplitIndex, B.getAddress(),
                                B.getAlignment(), B.getAlignmentOffset())
          : createContentBlock(
                B.getSection(), B.getContent().substr(0, SplitIndex),
                B.getAddress(), B.getAlignment(), B.getAlignmentOffset());

  // Modify B to cover [ SplitIndex, B.size() ).
  B.setAddress(B.getAddress() + SplitIndex);
  B.setContent(B.getContent().substr(SplitIndex));
  B.setAlignmentOffset((B.getAlignmentOffset() + SplitIndex) %
                       B.getAlignment());

  // Handle edge transfer/update.
  {
    // Copy edges to NewBlock (recording their iterators so that we can remove
    // them from B), and update of Edges remaining on B.
    std::vector<Block::edge_iterator> EdgesToRemove;
    for (auto I = B.edges().begin(); I != B.edges().end();) {
      if (I->getOffset() < SplitIndex) {
        NewBlock.addEdge(*I);
        I = B.removeEdge(I);
      } else {
        I->setOffset(I->getOffset() - SplitIndex);
        ++I;
      }
    }
  }

  // Handle symbol transfer/update.
  {
    // Initialize the symbols cache if necessary.
    SplitBlockCache LocalBlockSymbolsCache;
    if (!Cache)
      Cache = &LocalBlockSymbolsCache;
    if (*Cache == None) {
      *Cache = SplitBlockCache::value_type();
      for (auto *Sym : B.getSection().symbols())
        if (&Sym->getBlock() == &B)
          (*Cache)->push_back(Sym);

      llvm::sort(**Cache, [](const Symbol *LHS, const Symbol *RHS) {
        return LHS->getOffset() > RHS->getOffset();
      });
    }
    auto &BlockSymbols = **Cache;

    // Transfer all symbols with offset less than SplitIndex to NewBlock.
    while (!BlockSymbols.empty() &&
           BlockSymbols.back()->getOffset() < SplitIndex) {
      BlockSymbols.back()->setBlock(NewBlock);
      BlockSymbols.pop_back();
    }

    // Update offsets for all remaining symbols in B.
    for (auto *Sym : BlockSymbols)
      Sym->setOffset(Sym->getOffset() - SplitIndex);
  }

  return NewBlock;
}

void LinkGraph::dump(raw_ostream &OS,
                     std::function<StringRef(Edge::Kind)> EdgeKindToName) {
  if (!EdgeKindToName)
    EdgeKindToName = [](Edge::Kind K) { return StringRef(); };

  OS << "Symbols:\n";
  for (auto *Sym : defined_symbols()) {
    OS << "  " << format("0x%016" PRIx64, Sym->getAddress()) << ": " << *Sym
       << "\n";
    if (Sym->isDefined()) {
      for (auto &E : Sym->getBlock().edges()) {
        OS << "    ";
        StringRef EdgeName = (E.getKind() < Edge::FirstRelocation
                                  ? getGenericEdgeKindName(E.getKind())
                                  : EdgeKindToName(E.getKind()));

        if (!EdgeName.empty())
          printEdge(OS, Sym->getBlock(), E, EdgeName);
        else {
          auto EdgeNumberString = std::to_string(E.getKind());
          printEdge(OS, Sym->getBlock(), E, EdgeNumberString);
        }
        OS << "\n";
      }
    }
  }

  OS << "Absolute symbols:\n";
  for (auto *Sym : absolute_symbols())
    OS << "  " << format("0x%016" PRIx64, Sym->getAddress()) << ": " << *Sym
       << "\n";

  OS << "External symbols:\n";
  for (auto *Sym : external_symbols())
    OS << "  " << format("0x%016" PRIx64, Sym->getAddress()) << ": " << *Sym
       << "\n";
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

JITLinkContext::~JITLinkContext() {}

bool JITLinkContext::shouldAddDefaultTargetPasses(const Triple &TT) const {
  return true;
}

LinkGraphPassFunction JITLinkContext::getMarkLivePass(const Triple &TT) const {
  return LinkGraphPassFunction();
}

Error JITLinkContext::modifyPassConfig(const Triple &TT,
                                       PassConfiguration &Config) {
  return Error::success();
}

Error markAllSymbolsLive(LinkGraph &G) {
  for (auto *Sym : G.defined_symbols())
    Sym->setLive(true);
  return Error::success();
}

Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromObject(MemoryBufferRef ObjectBuffer) {
  auto Magic = identify_magic(ObjectBuffer.getBuffer());
  switch (Magic) {
  case file_magic::macho_object:
    return createLinkGraphFromMachOObject(ObjectBuffer);
  case file_magic::elf_relocatable:
    return createLinkGraphFromELFObject(ObjectBuffer);
  default:
    return make_error<JITLinkError>("Unsupported file format");
  };
}

void link(std::unique_ptr<LinkGraph> G, std::unique_ptr<JITLinkContext> Ctx) {
  switch (G->getTargetTriple().getObjectFormat()) {
  case Triple::MachO:
    return link_MachO(std::move(G), std::move(Ctx));
  case Triple::ELF:
    return link_ELF(std::move(G), std::move(Ctx));
  default:
    Ctx->notifyFailed(make_error<JITLinkError>("Unsupported object format"));
  };
}

} // end namespace jitlink
} // end namespace llvm
