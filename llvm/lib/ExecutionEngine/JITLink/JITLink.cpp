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
#include "llvm/ExecutionEngine/JITLink/JITLink_MachO.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
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

JITLinkMemoryManager::~JITLinkMemoryManager() = default;

JITLinkMemoryManager::Allocation::~Allocation() = default;

const StringRef getGenericEdgeKindName(Edge::Kind K) {
  switch (K) {
  case Edge::Invalid:
    return "INVALID RELOCATION";
  case Edge::KeepAlive:
    return "Keep-Alive";
  case Edge::LayoutNext:
    return "Layout-Next";
  default:
    llvm_unreachable("Unrecognized relocation kind");
  }
}

raw_ostream &operator<<(raw_ostream &OS, const Atom &A) {
  OS << "<";
  if (A.getName().empty())
    OS << "anon@" << format("0x%016" PRIx64, A.getAddress());
  else
    OS << A.getName();
  OS << " [";
  if (A.isDefined()) {
    auto &DA = static_cast<const DefinedAtom &>(A);
    OS << " section=" << DA.getSection().getName();
    if (DA.isLive())
      OS << " live";
    if (DA.shouldDiscard())
      OS << " should-discard";
  } else
    OS << " external";
  OS << " ]>";
  return OS;
}

void printEdge(raw_ostream &OS, const Atom &FixupAtom, const Edge &E,
               StringRef EdgeKindName) {
  OS << "edge@" << formatv("{0:x16}", FixupAtom.getAddress() + E.getOffset())
     << ": " << FixupAtom << " + " << E.getOffset() << " -- " << EdgeKindName
     << " -> " << E.getTarget() << " + " << E.getAddend();
}

Section::~Section() {
  for (auto *DA : DefinedAtoms)
    DA->~DefinedAtom();
}

void AtomGraph::dump(raw_ostream &OS,
                     std::function<StringRef(Edge::Kind)> EdgeKindToName) {
  if (!EdgeKindToName)
    EdgeKindToName = [](Edge::Kind K) { return StringRef(); };

  OS << "Defined atoms:\n";
  for (auto *DA : defined_atoms()) {
    OS << "  " << format("0x%016" PRIx64, DA->getAddress()) << ": " << *DA
       << "\n";
    for (auto &E : DA->edges()) {
      OS << "    ";
      StringRef EdgeName = (E.getKind() < Edge::FirstRelocation
                                ? getGenericEdgeKindName(E.getKind())
                                : EdgeKindToName(E.getKind()));

      if (!EdgeName.empty())
        printEdge(OS, *DA, E, EdgeName);
      else {
        auto EdgeNumberString = std::to_string(E.getKind());
        printEdge(OS, *DA, E, EdgeNumberString);
      }
      OS << "\n";
    }
  }

  OS << "Absolute atoms:\n";
  for (auto *A : absolute_atoms())
    OS << "  " << format("0x%016" PRIx64, A->getAddress()) << ": " << *A
       << "\n";

  OS << "External atoms:\n";
  for (auto *A : external_atoms())
    OS << "  " << format("0x%016" PRIx64, A->getAddress()) << ": " << *A
       << "\n";
}

Expected<std::unique_ptr<JITLinkMemoryManager::Allocation>>
InProcessMemoryManager::allocate(const SegmentsRequestMap &Request) {

  using AllocationMap = DenseMap<unsigned, sys::MemoryBlock>;

  // Local class for allocation.
  class IPMMAlloc : public Allocation {
  public:
    IPMMAlloc(AllocationMap SegBlocks) : SegBlocks(std::move(SegBlocks)) {}
    MutableArrayRef<char> getWorkingMemory(ProtectionFlags Seg) override {
      assert(SegBlocks.count(Seg) && "No allocation for segment");
      return {static_cast<char *>(SegBlocks[Seg].base()),
              SegBlocks[Seg].size()};
    }
    JITTargetAddress getTargetMemory(ProtectionFlags Seg) override {
      assert(SegBlocks.count(Seg) && "No allocation for segment");
      return reinterpret_cast<JITTargetAddress>(SegBlocks[Seg].base());
    }
    void finalizeAsync(FinalizeContinuation OnFinalize) override {
      OnFinalize(applyProtections());
    }
    Error deallocate() override {
      for (auto &KV : SegBlocks)
        if (auto EC = sys::Memory::releaseMappedMemory(KV.second))
          return errorCodeToError(EC);
      return Error::success();
    }

  private:
    Error applyProtections() {
      for (auto &KV : SegBlocks) {
        auto &Prot = KV.first;
        auto &Block = KV.second;
        if (auto EC = sys::Memory::protectMappedMemory(Block, Prot))
          return errorCodeToError(EC);
        if (Prot & sys::Memory::MF_EXEC)
          sys::Memory::InvalidateInstructionCache(Block.base(), Block.size());
      }
      return Error::success();
    }

    AllocationMap SegBlocks;
  };

  AllocationMap Blocks;
  const sys::Memory::ProtectionFlags ReadWrite =
      static_cast<sys::Memory::ProtectionFlags>(sys::Memory::MF_READ |
                                                sys::Memory::MF_WRITE);

  for (auto &KV : Request) {
    auto &Seg = KV.second;

    if (Seg.getContentAlignment() > sys::Process::getPageSize())
      return make_error<StringError>("Cannot request higher than page "
                                     "alignment",
                                     inconvertibleErrorCode());

    if (sys::Process::getPageSize() % Seg.getContentAlignment() != 0)
      return make_error<StringError>("Page size is not a multiple of "
                                     "alignment",
                                     inconvertibleErrorCode());

    uint64_t ZeroFillStart =
        alignTo(Seg.getContentSize(), Seg.getZeroFillAlignment());
    uint64_t SegmentSize = ZeroFillStart + Seg.getZeroFillSize();

    std::error_code EC;
    auto SegMem =
        sys::Memory::allocateMappedMemory(SegmentSize, nullptr, ReadWrite, EC);

    if (EC)
      return errorCodeToError(EC);

    // Zero out the zero-fill memory.
    memset(static_cast<char *>(SegMem.base()) + ZeroFillStart, 0,
           Seg.getZeroFillSize());

    // Record the block for this segment.
    Blocks[KV.first] = std::move(SegMem);
  }
  return std::unique_ptr<InProcessMemoryManager::Allocation>(
      new IPMMAlloc(std::move(Blocks)));
}

JITLinkContext::~JITLinkContext() {}

bool JITLinkContext::shouldAddDefaultTargetPasses(const Triple &TT) const {
  return true;
}

AtomGraphPassFunction JITLinkContext::getMarkLivePass(const Triple &TT) const {
  return AtomGraphPassFunction();
}

Error JITLinkContext::modifyPassConfig(const Triple &TT,
                                       PassConfiguration &Config) {
  return Error::success();
}

Error markAllAtomsLive(AtomGraph &G) {
  for (auto *DA : G.defined_atoms())
    DA->setLive(true);
  return Error::success();
}

void jitLink(std::unique_ptr<JITLinkContext> Ctx) {
  auto Magic = identify_magic(Ctx->getObjectBuffer().getBuffer());
  switch (Magic) {
  case file_magic::macho_object:
    return jitLink_MachO(std::move(Ctx));
  default:
    Ctx->notifyFailed(make_error<JITLinkError>("Unsupported file format"));
  };
}

} // end namespace jitlink
} // end namespace llvm
