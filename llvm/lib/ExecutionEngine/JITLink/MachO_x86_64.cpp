//===---- MachO_x86_64.cpp -JIT linker implementation for MachO/x86-64 ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MachO/x86-64 jit-link implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/MachO_x86_64.h"

#include "BasicGOTAndStubsBuilder.h"
#include "MachOLinkGraphBuilder.h"

#define DEBUG_TYPE "jitlink"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::jitlink::MachO_x86_64_Edges;

namespace {

class MachOLinkGraphBuilder_x86_64 : public MachOLinkGraphBuilder {
public:
  MachOLinkGraphBuilder_x86_64(const object::MachOObjectFile &Obj)
      : MachOLinkGraphBuilder(Obj) {}

private:
  static Expected<MachOX86RelocationKind>
  getRelocationKind(const MachO::relocation_info &RI) {
    switch (RI.r_type) {
    case MachO::X86_64_RELOC_UNSIGNED:
      if (!RI.r_pcrel) {
        if (RI.r_length == 3)
          return RI.r_extern ? Pointer64 : Pointer64Anon;
        else if (RI.r_extern && RI.r_length == 2)
          return Pointer32;
      }
      break;
    case MachO::X86_64_RELOC_SIGNED:
      if (RI.r_pcrel && RI.r_length == 2)
        return RI.r_extern ? PCRel32 : PCRel32Anon;
      break;
    case MachO::X86_64_RELOC_BRANCH:
      if (RI.r_pcrel && RI.r_extern && RI.r_length == 2)
        return Branch32;
      break;
    case MachO::X86_64_RELOC_GOT_LOAD:
      if (RI.r_pcrel && RI.r_extern && RI.r_length == 2)
        return PCRel32GOTLoad;
      break;
    case MachO::X86_64_RELOC_GOT:
      if (RI.r_pcrel && RI.r_extern && RI.r_length == 2)
        return PCRel32GOT;
      break;
    case MachO::X86_64_RELOC_SUBTRACTOR:
      // SUBTRACTOR must be non-pc-rel, extern, with length 2 or 3.
      // Initially represent SUBTRACTOR relocations with 'Delta<W>'. They may
      // be turned into NegDelta<W> by parsePairRelocation.
      if (!RI.r_pcrel && RI.r_extern) {
        if (RI.r_length == 2)
          return Delta32;
        else if (RI.r_length == 3)
          return Delta64;
      }
      break;
    case MachO::X86_64_RELOC_SIGNED_1:
      if (RI.r_pcrel && RI.r_length == 2)
        return RI.r_extern ? PCRel32Minus1 : PCRel32Minus1Anon;
      break;
    case MachO::X86_64_RELOC_SIGNED_2:
      if (RI.r_pcrel && RI.r_length == 2)
        return RI.r_extern ? PCRel32Minus2 : PCRel32Minus2Anon;
      break;
    case MachO::X86_64_RELOC_SIGNED_4:
      if (RI.r_pcrel && RI.r_length == 2)
        return RI.r_extern ? PCRel32Minus4 : PCRel32Minus4Anon;
      break;
    case MachO::X86_64_RELOC_TLV:
      if (RI.r_pcrel && RI.r_extern && RI.r_length == 2)
        return PCRel32TLV;
      break;
    }

    return make_error<JITLinkError>(
        "Unsupported x86-64 relocation: address=" +
        formatv("{0:x8}", RI.r_address) +
        ", symbolnum=" + formatv("{0:x6}", RI.r_symbolnum) +
        ", kind=" + formatv("{0:x1}", RI.r_type) +
        ", pc_rel=" + (RI.r_pcrel ? "true" : "false") +
        ", extern=" + (RI.r_extern ? "true" : "false") +
        ", length=" + formatv("{0:d}", RI.r_length));
  }

  MachO::relocation_info
  getRelocationInfo(const object::relocation_iterator RelItr) {
    MachO::any_relocation_info ARI =
        getObject().getRelocation(RelItr->getRawDataRefImpl());
    MachO::relocation_info RI;
    memcpy(&RI, &ARI, sizeof(MachO::relocation_info));
    return RI;
  }

  using PairRelocInfo = std::tuple<MachOX86RelocationKind, Symbol *, uint64_t>;

  // Parses paired SUBTRACTOR/UNSIGNED relocations and, on success,
  // returns the edge kind and addend to be used.
  Expected<PairRelocInfo>
  parsePairRelocation(Block &BlockToFix, Edge::Kind SubtractorKind,
                      const MachO::relocation_info &SubRI,
                      JITTargetAddress FixupAddress, const char *FixupContent,
                      object::relocation_iterator &UnsignedRelItr,
                      object::relocation_iterator &RelEnd) {
    using namespace support;

    assert(((SubtractorKind == Delta32 && SubRI.r_length == 2) ||
            (SubtractorKind == Delta64 && SubRI.r_length == 3)) &&
           "Subtractor kind should match length");
    assert(SubRI.r_extern && "SUBTRACTOR reloc symbol should be extern");
    assert(!SubRI.r_pcrel && "SUBTRACTOR reloc should not be PCRel");

    if (UnsignedRelItr == RelEnd)
      return make_error<JITLinkError>("x86_64 SUBTRACTOR without paired "
                                      "UNSIGNED relocation");

    auto UnsignedRI = getRelocationInfo(UnsignedRelItr);

    if (SubRI.r_address != UnsignedRI.r_address)
      return make_error<JITLinkError>("x86_64 SUBTRACTOR and paired UNSIGNED "
                                      "point to different addresses");

    if (SubRI.r_length != UnsignedRI.r_length)
      return make_error<JITLinkError>("length of x86_64 SUBTRACTOR and paired "
                                      "UNSIGNED reloc must match");

    Symbol *FromSymbol;
    if (auto FromSymbolOrErr = findSymbolByIndex(SubRI.r_symbolnum))
      FromSymbol = FromSymbolOrErr->GraphSymbol;
    else
      return FromSymbolOrErr.takeError();

    // Read the current fixup value.
    uint64_t FixupValue = 0;
    if (SubRI.r_length == 3)
      FixupValue = *(const little64_t *)FixupContent;
    else
      FixupValue = *(const little32_t *)FixupContent;

    // Find 'ToSymbol' using symbol number or address, depending on whether the
    // paired UNSIGNED relocation is extern.
    Symbol *ToSymbol = nullptr;
    if (UnsignedRI.r_extern) {
      // Find target symbol by symbol index.
      if (auto ToSymbolOrErr = findSymbolByIndex(UnsignedRI.r_symbolnum))
        ToSymbol = ToSymbolOrErr->GraphSymbol;
      else
        return ToSymbolOrErr.takeError();
    } else {
      if (auto ToSymbolOrErr = findSymbolByAddress(FixupValue))
        ToSymbol = &*ToSymbolOrErr;
      else
        return ToSymbolOrErr.takeError();
      FixupValue -= ToSymbol->getAddress();
    }

    MachOX86RelocationKind DeltaKind;
    Symbol *TargetSymbol;
    uint64_t Addend;
    if (&BlockToFix == &FromSymbol->getAddressable()) {
      TargetSymbol = ToSymbol;
      DeltaKind = (SubRI.r_length == 3) ? Delta64 : Delta32;
      Addend = FixupValue + (FixupAddress - FromSymbol->getAddress());
      // FIXME: handle extern 'from'.
    } else if (&BlockToFix == &ToSymbol->getAddressable()) {
      TargetSymbol = FromSymbol;
      DeltaKind = (SubRI.r_length == 3) ? NegDelta64 : NegDelta32;
      Addend = FixupValue - (FixupAddress - ToSymbol->getAddress());
    } else {
      // BlockToFix was neither FromSymbol nor ToSymbol.
      return make_error<JITLinkError>("SUBTRACTOR relocation must fix up "
                                      "either 'A' or 'B' (or a symbol in one "
                                      "of their alt-entry chains)");
    }

    return PairRelocInfo(DeltaKind, TargetSymbol, Addend);
  }

  Error addRelocations() override {
    using namespace support;
    auto &Obj = getObject();

    for (auto &S : Obj.sections()) {

      JITTargetAddress SectionAddress = S.getAddress();

      if (S.isVirtual()) {
        if (S.relocation_begin() != S.relocation_end())
          return make_error<JITLinkError>("Virtual section contains "
                                          "relocations");
        continue;
      }

      for (auto RelItr = S.relocation_begin(), RelEnd = S.relocation_end();
           RelItr != RelEnd; ++RelItr) {

        MachO::relocation_info RI = getRelocationInfo(RelItr);

        // Sanity check the relocation kind.
        auto Kind = getRelocationKind(RI);
        if (!Kind)
          return Kind.takeError();

        // Find the address of the value to fix up.
        JITTargetAddress FixupAddress = SectionAddress + (uint32_t)RI.r_address;

        LLVM_DEBUG({
          dbgs() << "Processing relocation at "
                 << format("0x%016" PRIx64, FixupAddress) << "\n";
        });

        // Find the block that the fixup points to.
        Block *BlockToFix = nullptr;
        {
          auto SymbolToFixOrErr = findSymbolByAddress(FixupAddress);
          if (!SymbolToFixOrErr)
            return SymbolToFixOrErr.takeError();
          BlockToFix = &SymbolToFixOrErr->getBlock();
        }

        if (FixupAddress + static_cast<JITTargetAddress>(1ULL << RI.r_length) >
            BlockToFix->getAddress() + BlockToFix->getContent().size())
          return make_error<JITLinkError>(
              "Relocation extends past end of fixup block");

        // Get a pointer to the fixup content.
        const char *FixupContent = BlockToFix->getContent().data() +
                                   (FixupAddress - BlockToFix->getAddress());

        // The target symbol and addend will be populated by the switch below.
        Symbol *TargetSymbol = nullptr;
        uint64_t Addend = 0;

        switch (*Kind) {
        case Branch32:
        case PCRel32:
        case PCRel32GOTLoad:
        case PCRel32GOT:
          if (auto TargetSymbolOrErr = findSymbolByIndex(RI.r_symbolnum))
            TargetSymbol = TargetSymbolOrErr->GraphSymbol;
          else
            return TargetSymbolOrErr.takeError();
          Addend = *(const little32_t *)FixupContent;
          break;
        case Pointer32:
          if (auto TargetSymbolOrErr = findSymbolByIndex(RI.r_symbolnum))
            TargetSymbol = TargetSymbolOrErr->GraphSymbol;
          else
            return TargetSymbolOrErr.takeError();
          Addend = *(const ulittle32_t *)FixupContent;
          break;
        case Pointer64:
          if (auto TargetSymbolOrErr = findSymbolByIndex(RI.r_symbolnum))
            TargetSymbol = TargetSymbolOrErr->GraphSymbol;
          else
            return TargetSymbolOrErr.takeError();
          Addend = *(const ulittle64_t *)FixupContent;
          break;
        case Pointer64Anon: {
          JITTargetAddress TargetAddress = *(const ulittle64_t *)FixupContent;
          if (auto TargetSymbolOrErr = findSymbolByAddress(TargetAddress))
            TargetSymbol = &*TargetSymbolOrErr;
          else
            return TargetSymbolOrErr.takeError();
          Addend = TargetAddress - TargetSymbol->getAddress();
          break;
        }
        case PCRel32Minus1:
        case PCRel32Minus2:
        case PCRel32Minus4:
          if (auto TargetSymbolOrErr = findSymbolByIndex(RI.r_symbolnum))
            TargetSymbol = TargetSymbolOrErr->GraphSymbol;
          else
            return TargetSymbolOrErr.takeError();
          Addend = *(const little32_t *)FixupContent +
                   (1 << (*Kind - PCRel32Minus1));
          break;
        case PCRel32Anon: {
          JITTargetAddress TargetAddress =
              FixupAddress + 4 + *(const little32_t *)FixupContent;
          if (auto TargetSymbolOrErr = findSymbolByAddress(TargetAddress))
            TargetSymbol = &*TargetSymbolOrErr;
          else
            return TargetSymbolOrErr.takeError();
          Addend = TargetAddress - TargetSymbol->getAddress();
          break;
        }
        case PCRel32Minus1Anon:
        case PCRel32Minus2Anon:
        case PCRel32Minus4Anon: {
          JITTargetAddress Delta =
              static_cast<JITTargetAddress>(1ULL << (*Kind - PCRel32Minus1Anon));
          JITTargetAddress TargetAddress =
              FixupAddress + 4 + Delta + *(const little32_t *)FixupContent;
          if (auto TargetSymbolOrErr = findSymbolByAddress(TargetAddress))
            TargetSymbol = &*TargetSymbolOrErr;
          else
            return TargetSymbolOrErr.takeError();
          Addend = TargetAddress - TargetSymbol->getAddress();
          break;
        }
        case Delta32:
        case Delta64: {
          // We use Delta32/Delta64 to represent SUBTRACTOR relocations.
          // parsePairRelocation handles the paired reloc, and returns the
          // edge kind to be used (either Delta32/Delta64, or
          // NegDelta32/NegDelta64, depending on the direction of the
          // subtraction) along with the addend.
          auto PairInfo =
              parsePairRelocation(*BlockToFix, *Kind, RI, FixupAddress,
                                  FixupContent, ++RelItr, RelEnd);
          if (!PairInfo)
            return PairInfo.takeError();
          std::tie(*Kind, TargetSymbol, Addend) = *PairInfo;
          assert(TargetSymbol && "No target symbol from parsePairRelocation?");
          break;
        }
        default:
          llvm_unreachable("Special relocation kind should not appear in "
                           "mach-o file");
        }

        LLVM_DEBUG({
          Edge GE(*Kind, FixupAddress - BlockToFix->getAddress(), *TargetSymbol,
                  Addend);
          printEdge(dbgs(), *BlockToFix, GE,
                    getMachOX86RelocationKindName(*Kind));
          dbgs() << "\n";
        });
        BlockToFix->addEdge(*Kind, FixupAddress - BlockToFix->getAddress(),
                            *TargetSymbol, Addend);
      }
    }
    return Error::success();
  }
};

class MachO_x86_64_GOTAndStubsBuilder
    : public BasicGOTAndStubsBuilder<MachO_x86_64_GOTAndStubsBuilder> {
public:
  static const uint8_t NullGOTEntryContent[8];
  static const uint8_t StubContent[6];

  MachO_x86_64_GOTAndStubsBuilder(LinkGraph &G)
      : BasicGOTAndStubsBuilder<MachO_x86_64_GOTAndStubsBuilder>(G) {}

  bool isGOTEdge(Edge &E) const {
    return E.getKind() == PCRel32GOT || E.getKind() == PCRel32GOTLoad;
  }

  Symbol &createGOTEntry(Symbol &Target) {
    auto &GOTEntryBlock = G.createContentBlock(
        getGOTSection(), getGOTEntryBlockContent(), 0, 8, 0);
    GOTEntryBlock.addEdge(Pointer64, 0, Target, 0);
    return G.addAnonymousSymbol(GOTEntryBlock, 0, 8, false, false);
  }

  void fixGOTEdge(Edge &E, Symbol &GOTEntry) {
    assert((E.getKind() == PCRel32GOT || E.getKind() == PCRel32GOTLoad) &&
           "Not a GOT edge?");
    // If this is a PCRel32GOT then change it to an ordinary PCRel32. If it is
    // a PCRel32GOTLoad then leave it as-is for now. We will use the kind to
    // check for GOT optimization opportunities in the
    // optimizeMachO_x86_64_GOTAndStubs pass below.
    if (E.getKind() == PCRel32GOT)
      E.setKind(PCRel32);

    E.setTarget(GOTEntry);
    // Leave the edge addend as-is.
  }

  bool isExternalBranchEdge(Edge &E) {
    return E.getKind() == Branch32 && !E.getTarget().isDefined();
  }

  Symbol &createStub(Symbol &Target) {
    auto &StubContentBlock =
        G.createContentBlock(getStubsSection(), getStubBlockContent(), 0, 1, 0);
    // Re-use GOT entries for stub targets.
    auto &GOTEntrySymbol = getGOTEntrySymbol(Target);
    StubContentBlock.addEdge(PCRel32, 2, GOTEntrySymbol, 0);
    return G.addAnonymousSymbol(StubContentBlock, 0, 6, true, false);
  }

  void fixExternalBranchEdge(Edge &E, Symbol &Stub) {
    assert(E.getKind() == Branch32 && "Not a Branch32 edge?");
    assert(E.getAddend() == 0 && "Branch32 edge has non-zero addend?");

    // Set the edge kind to Branch32ToStub. We will use this to check for stub
    // optimization opportunities in the optimizeMachO_x86_64_GOTAndStubs pass
    // below.
    E.setKind(Branch32ToStub);
    E.setTarget(Stub);
  }

private:
  Section &getGOTSection() {
    if (!GOTSection)
      GOTSection = &G.createSection("$__GOT", sys::Memory::MF_READ);
    return *GOTSection;
  }

  Section &getStubsSection() {
    if (!StubsSection) {
      auto StubsProt = static_cast<sys::Memory::ProtectionFlags>(
          sys::Memory::MF_READ | sys::Memory::MF_EXEC);
      StubsSection = &G.createSection("$__STUBS", StubsProt);
    }
    return *StubsSection;
  }

  StringRef getGOTEntryBlockContent() {
    return StringRef(reinterpret_cast<const char *>(NullGOTEntryContent),
                     sizeof(NullGOTEntryContent));
  }

  StringRef getStubBlockContent() {
    return StringRef(reinterpret_cast<const char *>(StubContent),
                     sizeof(StubContent));
  }

  Section *GOTSection = nullptr;
  Section *StubsSection = nullptr;
};

const uint8_t MachO_x86_64_GOTAndStubsBuilder::NullGOTEntryContent[8] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
const uint8_t MachO_x86_64_GOTAndStubsBuilder::StubContent[6] = {
    0xFF, 0x25, 0x00, 0x00, 0x00, 0x00};
} // namespace

static Error optimizeMachO_x86_64_GOTAndStubs(LinkGraph &G) {
  LLVM_DEBUG(dbgs() << "Optimizing GOT entries and stubs:\n");

  for (auto *B : G.blocks())
    for (auto &E : B->edges())
      if (E.getKind() == PCRel32GOTLoad) {
        assert(E.getOffset() >= 3 && "GOT edge occurs too early in block");

        // Switch the edge kind to PCRel32: Whether we change the edge target
        // or not this will be the desired kind.
        E.setKind(PCRel32);

        // Optimize GOT references.
        auto &GOTBlock = E.getTarget().getBlock();
        assert(GOTBlock.getSize() == G.getPointerSize() &&
               "GOT entry block should be pointer sized");
        assert(GOTBlock.edges_size() == 1 &&
               "GOT entry should only have one outgoing edge");

        auto &GOTTarget = GOTBlock.edges().begin()->getTarget();
        JITTargetAddress EdgeAddr = B->getAddress() + E.getOffset();
        JITTargetAddress TargetAddr = GOTTarget.getAddress();

        // Check that this is a recognized MOV instruction.
        // FIXME: Can we assume this?
        constexpr uint8_t MOVQRIPRel[] = {0x48, 0x8b};
        if (strncmp(B->getContent().data() + E.getOffset() - 3,
                    reinterpret_cast<const char *>(MOVQRIPRel), 2) != 0)
          continue;

        int64_t Displacement = TargetAddr - EdgeAddr + 4;
        if (Displacement >= std::numeric_limits<int32_t>::min() &&
            Displacement <= std::numeric_limits<int32_t>::max()) {
          E.setTarget(GOTTarget);
          auto *BlockData = reinterpret_cast<uint8_t *>(
              const_cast<char *>(B->getContent().data()));
          BlockData[E.getOffset() - 2] = 0x8d;
          LLVM_DEBUG({
            dbgs() << "  Replaced GOT load wih LEA:\n    ";
            printEdge(dbgs(), *B, E,
                      getMachOX86RelocationKindName(E.getKind()));
            dbgs() << "\n";
          });
        }
      } else if (E.getKind() == Branch32ToStub) {

        // Switch the edge kind to PCRel32: Whether we change the edge target
        // or not this will be the desired kind.
        E.setKind(Branch32);

        auto &StubBlock = E.getTarget().getBlock();
        assert(StubBlock.getSize() ==
                   sizeof(MachO_x86_64_GOTAndStubsBuilder::StubContent) &&
               "Stub block should be stub sized");
        assert(StubBlock.edges_size() == 1 &&
               "Stub block should only have one outgoing edge");

        auto &GOTBlock = StubBlock.edges().begin()->getTarget().getBlock();
        assert(GOTBlock.getSize() == G.getPointerSize() &&
               "GOT block should be pointer sized");
        assert(GOTBlock.edges_size() == 1 &&
               "GOT block should only have one outgoing edge");

        auto &GOTTarget = GOTBlock.edges().begin()->getTarget();
        JITTargetAddress EdgeAddr = B->getAddress() + E.getOffset();
        JITTargetAddress TargetAddr = GOTTarget.getAddress();

        int64_t Displacement = TargetAddr - EdgeAddr + 4;
        if (Displacement >= std::numeric_limits<int32_t>::min() &&
            Displacement <= std::numeric_limits<int32_t>::max()) {
          E.setTarget(GOTTarget);
          LLVM_DEBUG({
            dbgs() << "  Replaced stub branch with direct branch:\n    ";
            printEdge(dbgs(), *B, E,
                      getMachOX86RelocationKindName(E.getKind()));
            dbgs() << "\n";
          });
        }
      }

  return Error::success();
}

namespace llvm {
namespace jitlink {

class MachOJITLinker_x86_64 : public JITLinker<MachOJITLinker_x86_64> {
  friend class JITLinker<MachOJITLinker_x86_64>;

public:
  MachOJITLinker_x86_64(std::unique_ptr<JITLinkContext> Ctx,
                        PassConfiguration PassConfig)
      : JITLinker(std::move(Ctx), std::move(PassConfig)) {}

private:
  StringRef getEdgeKindName(Edge::Kind R) const override {
    return getMachOX86RelocationKindName(R);
  }

  Expected<std::unique_ptr<LinkGraph>>
  buildGraph(MemoryBufferRef ObjBuffer) override {
    auto MachOObj = object::ObjectFile::createMachOObjectFile(ObjBuffer);
    if (!MachOObj)
      return MachOObj.takeError();
    return MachOLinkGraphBuilder_x86_64(**MachOObj).buildGraph();
  }

  static Error targetOutOfRangeError(const Block &B, const Edge &E) {
    std::string ErrMsg;
    {
      raw_string_ostream ErrStream(ErrMsg);
      ErrStream << "Relocation target out of range: ";
      printEdge(ErrStream, B, E, getMachOX86RelocationKindName(E.getKind()));
      ErrStream << "\n";
    }
    return make_error<JITLinkError>(std::move(ErrMsg));
  }

  Error applyFixup(Block &B, const Edge &E, char *BlockWorkingMem) const {

    using namespace support;

    char *FixupPtr = BlockWorkingMem + E.getOffset();
    JITTargetAddress FixupAddress = B.getAddress() + E.getOffset();

    switch (E.getKind()) {
    case Branch32:
    case PCRel32:
    case PCRel32Anon: {
      int64_t Value =
          E.getTarget().getAddress() - (FixupAddress + 4) + E.getAddend();
      if (Value < std::numeric_limits<int32_t>::min() ||
          Value > std::numeric_limits<int32_t>::max())
        return targetOutOfRangeError(B, E);
      *(little32_t *)FixupPtr = Value;
      break;
    }
    case Pointer64:
    case Pointer64Anon: {
      uint64_t Value = E.getTarget().getAddress() + E.getAddend();
      *(ulittle64_t *)FixupPtr = Value;
      break;
    }
    case PCRel32Minus1:
    case PCRel32Minus2:
    case PCRel32Minus4: {
      int Delta = 4 + (1 << (E.getKind() - PCRel32Minus1));
      int64_t Value =
          E.getTarget().getAddress() - (FixupAddress + Delta) + E.getAddend();
      if (Value < std::numeric_limits<int32_t>::min() ||
          Value > std::numeric_limits<int32_t>::max())
        return targetOutOfRangeError(B, E);
      *(little32_t *)FixupPtr = Value;
      break;
    }
    case PCRel32Minus1Anon:
    case PCRel32Minus2Anon:
    case PCRel32Minus4Anon: {
      int Delta = 4 + (1 << (E.getKind() - PCRel32Minus1Anon));
      int64_t Value =
          E.getTarget().getAddress() - (FixupAddress + Delta) + E.getAddend();
      if (Value < std::numeric_limits<int32_t>::min() ||
          Value > std::numeric_limits<int32_t>::max())
        return targetOutOfRangeError(B, E);
      *(little32_t *)FixupPtr = Value;
      break;
    }
    case Delta32:
    case Delta64:
    case NegDelta32:
    case NegDelta64: {
      int64_t Value;
      if (E.getKind() == Delta32 || E.getKind() == Delta64)
        Value = E.getTarget().getAddress() - FixupAddress + E.getAddend();
      else
        Value = FixupAddress - E.getTarget().getAddress() + E.getAddend();

      if (E.getKind() == Delta32 || E.getKind() == NegDelta32) {
        if (Value < std::numeric_limits<int32_t>::min() ||
            Value > std::numeric_limits<int32_t>::max())
          return targetOutOfRangeError(B, E);
        *(little32_t *)FixupPtr = Value;
      } else
        *(little64_t *)FixupPtr = Value;
      break;
    }
    case Pointer32: {
      uint64_t Value = E.getTarget().getAddress() + E.getAddend();
      if (Value > std::numeric_limits<uint32_t>::max())
        return targetOutOfRangeError(B, E);
      *(ulittle32_t *)FixupPtr = Value;
      break;
    }
    default:
      llvm_unreachable("Unrecognized edge kind");
    }

    return Error::success();
  }

  uint64_t NullValue = 0;
};

void jitLink_MachO_x86_64(std::unique_ptr<JITLinkContext> Ctx) {
  PassConfiguration Config;
  Triple TT("x86_64-apple-macosx");

  if (Ctx->shouldAddDefaultTargetPasses(TT)) {
    // Add eh-frame passses.
    Config.PrePrunePasses.push_back(EHFrameSplitter("__eh_frame"));
    Config.PrePrunePasses.push_back(
        EHFrameEdgeFixer("__eh_frame", NegDelta32, Delta64, Delta64));

    // Add a mark-live pass.
    if (auto MarkLive = Ctx->getMarkLivePass(TT))
      Config.PrePrunePasses.push_back(std::move(MarkLive));
    else
      Config.PrePrunePasses.push_back(markAllSymbolsLive);

    // Add an in-place GOT/Stubs pass.
    Config.PostPrunePasses.push_back([](LinkGraph &G) -> Error {
      MachO_x86_64_GOTAndStubsBuilder(G).run();
      return Error::success();
    });

    // Add GOT/Stubs optimizer pass.
    Config.PostAllocationPasses.push_back(optimizeMachO_x86_64_GOTAndStubs);
  }

  if (auto Err = Ctx->modifyPassConfig(TT, Config))
    return Ctx->notifyFailed(std::move(Err));

  // Construct a JITLinker and run the link function.
  MachOJITLinker_x86_64::link(std::move(Ctx), std::move(Config));
}

StringRef getMachOX86RelocationKindName(Edge::Kind R) {
  switch (R) {
  case Branch32:
    return "Branch32";
  case Branch32ToStub:
    return "Branch32ToStub";
  case Pointer32:
    return "Pointer32";
  case Pointer64:
    return "Pointer64";
  case Pointer64Anon:
    return "Pointer64Anon";
  case PCRel32:
    return "PCRel32";
  case PCRel32Minus1:
    return "PCRel32Minus1";
  case PCRel32Minus2:
    return "PCRel32Minus2";
  case PCRel32Minus4:
    return "PCRel32Minus4";
  case PCRel32Anon:
    return "PCRel32Anon";
  case PCRel32Minus1Anon:
    return "PCRel32Minus1Anon";
  case PCRel32Minus2Anon:
    return "PCRel32Minus2Anon";
  case PCRel32Minus4Anon:
    return "PCRel32Minus4Anon";
  case PCRel32GOTLoad:
    return "PCRel32GOTLoad";
  case PCRel32GOT:
    return "PCRel32GOT";
  case PCRel32TLV:
    return "PCRel32TLV";
  case Delta32:
    return "Delta32";
  case Delta64:
    return "Delta64";
  case NegDelta32:
    return "NegDelta32";
  case NegDelta64:
    return "NegDelta64";
  default:
    return getGenericEdgeKindName(static_cast<Edge::Kind>(R));
  }
}

} // end namespace jitlink
} // end namespace llvm
