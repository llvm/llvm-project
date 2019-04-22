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
#include "MachOAtomGraphBuilder.h"

#define DEBUG_TYPE "jitlink"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::jitlink::MachO_x86_64_Edges;

namespace {

class MachOAtomGraphBuilder_x86_64 : public MachOAtomGraphBuilder {
public:
  MachOAtomGraphBuilder_x86_64(const object::MachOObjectFile &Obj)
      : MachOAtomGraphBuilder(Obj),
        NumSymbols(Obj.getSymtabLoadCommand().nsyms) {
    addCustomAtomizer("__eh_frame", [this](MachOSection &EHFrameSection) {
      return addEHFrame(getGraph(), EHFrameSection.getGenericSection(),
                        EHFrameSection.getContent(),
                        EHFrameSection.getAddress(), NegDelta32, Delta64);
    });
  }

private:
  static Expected<MachOX86RelocationKind>
  getRelocationKind(const MachO::relocation_info &RI) {
    switch (RI.r_type) {
    case MachO::X86_64_RELOC_UNSIGNED:
      if (!RI.r_pcrel && RI.r_length == 3)
        return RI.r_extern ? Pointer64 : Pointer64Anon;
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
        ", extern= " + (RI.r_extern ? "true" : "false") +
        ", length=" + formatv("{0:d}", RI.r_length));
  }

  Expected<Atom &> findAtomBySymbolIndex(const MachO::relocation_info &RI) {
    auto &Obj = getObject();
    if (RI.r_symbolnum >= NumSymbols)
      return make_error<JITLinkError>("Symbol index out of range");
    auto SymI = Obj.getSymbolByIndex(RI.r_symbolnum);
    auto Name = SymI->getName();
    if (!Name)
      return Name.takeError();
    return getGraph().getAtomByName(*Name);
  }

  MachO::relocation_info
  getRelocationInfo(const object::relocation_iterator RelItr) {
    MachO::any_relocation_info ARI =
        getObject().getRelocation(RelItr->getRawDataRefImpl());
    MachO::relocation_info RI;
    memcpy(&RI, &ARI, sizeof(MachO::relocation_info));
    return RI;
  }

  using PairRelocInfo = std::tuple<MachOX86RelocationKind, Atom *, uint64_t>;

  // Parses paired SUBTRACTOR/UNSIGNED relocations and, on success,
  // returns the edge kind and addend to be used.
  Expected<PairRelocInfo>
  parsePairRelocation(DefinedAtom &AtomToFix, Edge::Kind SubtractorKind,
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

    auto FromAtom = findAtomBySymbolIndex(SubRI);
    if (!FromAtom)
      return FromAtom.takeError();

    // Read the current fixup value.
    uint64_t FixupValue = 0;
    if (SubRI.r_length == 3)
      FixupValue = *(const ulittle64_t *)FixupContent;
    else
      FixupValue = *(const ulittle32_t *)FixupContent;

    // Find 'ToAtom' using symbol number or address, depending on whether the
    // paired UNSIGNED relocation is extern.
    Atom *ToAtom = nullptr;
    if (UnsignedRI.r_extern) {
      // Find target atom by symbol index.
      if (auto ToAtomOrErr = findAtomBySymbolIndex(UnsignedRI))
        ToAtom = &*ToAtomOrErr;
      else
        return ToAtomOrErr.takeError();
    } else {
      if (auto ToAtomOrErr = getGraph().findAtomByAddress(FixupValue))
        ToAtom = &*ToAtomOrErr;
      else
        return ToAtomOrErr.takeError();
      FixupValue -= ToAtom->getAddress();
    }

    MachOX86RelocationKind DeltaKind;
    Atom *TargetAtom;
    uint64_t Addend;
    if (&AtomToFix == &*FromAtom) {
      TargetAtom = ToAtom;
      DeltaKind = (SubRI.r_length == 3) ? Delta64 : Delta32;
      Addend = FixupValue + (FixupAddress - FromAtom->getAddress());
      // FIXME: handle extern 'from'.
    } else if (&AtomToFix == ToAtom) {
      TargetAtom = &*FromAtom;
      DeltaKind = (SubRI.r_length == 3) ? NegDelta64 : NegDelta32;
      Addend = FixupValue - (FixupAddress - ToAtom->getAddress());
    } else {
      // AtomToFix was neither FromAtom nor ToAtom.
      return make_error<JITLinkError>("SUBTRACTOR relocation must fix up "
                                      "either 'A' or 'B'");
    }

    return PairRelocInfo(DeltaKind, TargetAtom, Addend);
  }

  Error addRelocations() override {
    using namespace support;
    auto &G = getGraph();
    auto &Obj = getObject();

    for (auto &S : Obj.sections()) {

      JITTargetAddress SectionAddress = S.getAddress();

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

        // Find the atom that the fixup points to.
        DefinedAtom *AtomToFix = nullptr;
        {
          auto AtomToFixOrErr = G.findAtomByAddress(FixupAddress);
          if (!AtomToFixOrErr)
            return AtomToFixOrErr.takeError();
          AtomToFix = &*AtomToFixOrErr;
        }

        if (FixupAddress + static_cast<JITTargetAddress>(1 << RI.r_length) >
            AtomToFix->getAddress() + AtomToFix->getContent().size())
          return make_error<JITLinkError>(
              "Relocation content extends past end of fixup atom");

        // Get a pointer to the fixup content.
        const char *FixupContent = AtomToFix->getContent().data() +
                                   (FixupAddress - AtomToFix->getAddress());

        // The target atom and addend will be populated by the switch below.
        Atom *TargetAtom = nullptr;
        uint64_t Addend = 0;

        switch (*Kind) {
        case Branch32:
        case PCRel32:
        case PCRel32GOTLoad:
        case PCRel32GOT:
          if (auto TargetAtomOrErr = findAtomBySymbolIndex(RI))
            TargetAtom = &*TargetAtomOrErr;
          else
            return TargetAtomOrErr.takeError();
          Addend = *(const ulittle32_t *)FixupContent;
          break;
        case Pointer64:
          if (auto TargetAtomOrErr = findAtomBySymbolIndex(RI))
            TargetAtom = &*TargetAtomOrErr;
          else
            return TargetAtomOrErr.takeError();
          Addend = *(const ulittle64_t *)FixupContent;
          break;
        case Pointer64Anon: {
          JITTargetAddress TargetAddress = *(const ulittle64_t *)FixupContent;
          if (auto TargetAtomOrErr = G.findAtomByAddress(TargetAddress))
            TargetAtom = &*TargetAtomOrErr;
          else
            return TargetAtomOrErr.takeError();
          Addend = TargetAddress - TargetAtom->getAddress();
          break;
        }
        case PCRel32Minus1:
        case PCRel32Minus2:
        case PCRel32Minus4:
          if (auto TargetAtomOrErr = findAtomBySymbolIndex(RI))
            TargetAtom = &*TargetAtomOrErr;
          else
            return TargetAtomOrErr.takeError();
          Addend = *(const ulittle32_t *)FixupContent +
                   (1 << (*Kind - PCRel32Minus1));
          break;
        case PCRel32Anon: {
          JITTargetAddress TargetAddress =
              FixupAddress + 4 + *(const ulittle32_t *)FixupContent;
          if (auto TargetAtomOrErr = G.findAtomByAddress(TargetAddress))
            TargetAtom = &*TargetAtomOrErr;
          else
            return TargetAtomOrErr.takeError();
          Addend = TargetAddress - TargetAtom->getAddress();
          break;
        }
        case PCRel32Minus1Anon:
        case PCRel32Minus2Anon:
        case PCRel32Minus4Anon: {
          JITTargetAddress Delta =
              static_cast<JITTargetAddress>(1 << (*Kind - PCRel32Minus1Anon));
          JITTargetAddress TargetAddress =
              FixupAddress + 4 + Delta + *(const ulittle32_t *)FixupContent;
          if (auto TargetAtomOrErr = G.findAtomByAddress(TargetAddress))
            TargetAtom = &*TargetAtomOrErr;
          else
            return TargetAtomOrErr.takeError();
          Addend = TargetAddress - TargetAtom->getAddress();
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
              parsePairRelocation(*AtomToFix, *Kind, RI, FixupAddress,
                                  FixupContent, ++RelItr, RelEnd);
          if (!PairInfo)
            return PairInfo.takeError();
          std::tie(*Kind, TargetAtom, Addend) = *PairInfo;
          assert(TargetAtom && "No target atom from parsePairRelocation?");
          break;
        }
        default:
          llvm_unreachable("Special relocation kind should not appear in "
                           "mach-o file");
        }

        LLVM_DEBUG({
          Edge GE(*Kind, FixupAddress - AtomToFix->getAddress(), *TargetAtom,
                  Addend);
          printEdge(dbgs(), *AtomToFix, GE,
                    getMachOX86RelocationKindName(*Kind));
          dbgs() << "\n";
        });
        AtomToFix->addEdge(*Kind, FixupAddress - AtomToFix->getAddress(),
                           *TargetAtom, Addend);
      }
    }
    return Error::success();
  }

  unsigned NumSymbols = 0;
};

class MachO_x86_64_GOTAndStubsBuilder
    : public BasicGOTAndStubsBuilder<MachO_x86_64_GOTAndStubsBuilder> {
public:
  MachO_x86_64_GOTAndStubsBuilder(AtomGraph &G)
      : BasicGOTAndStubsBuilder<MachO_x86_64_GOTAndStubsBuilder>(G) {}

  bool isGOTEdge(Edge &E) const {
    return E.getKind() == PCRel32GOT || E.getKind() == PCRel32GOTLoad;
  }

  DefinedAtom &createGOTEntry(Atom &Target) {
    auto &GOTEntryAtom = G.addAnonymousAtom(getGOTSection(), 0x0, 8);
    GOTEntryAtom.setContent(
        StringRef(reinterpret_cast<const char *>(NullGOTEntryContent), 8));
    GOTEntryAtom.addEdge(Pointer64, 0, Target, 0);
    return GOTEntryAtom;
  }

  void fixGOTEdge(Edge &E, Atom &GOTEntry) {
    assert((E.getKind() == PCRel32GOT || E.getKind() == PCRel32GOTLoad) &&
           "Not a GOT edge?");
    E.setKind(PCRel32);
    E.setTarget(GOTEntry);
    // Leave the edge addend as-is.
  }

  bool isExternalBranchEdge(Edge &E) {
    return E.getKind() == Branch32 && !E.getTarget().isDefined();
  }

  DefinedAtom &createStub(Atom &Target) {
    auto &StubAtom = G.addAnonymousAtom(getStubsSection(), 0x0, 2);
    StubAtom.setContent(
        StringRef(reinterpret_cast<const char *>(StubContent), 6));

    // Re-use GOT entries for stub targets.
    auto &GOTEntryAtom = getGOTEntryAtom(Target);
    StubAtom.addEdge(PCRel32, 2, GOTEntryAtom, 0);

    return StubAtom;
  }

  void fixExternalBranchEdge(Edge &E, Atom &Stub) {
    assert(E.getKind() == Branch32 && "Not a Branch32 edge?");
    assert(E.getAddend() == 0 && "Branch32 edge has non-zero addend?");
    E.setTarget(Stub);
  }

private:
  Section &getGOTSection() {
    if (!GOTSection)
      GOTSection = &G.createSection("$__GOT", sys::Memory::MF_READ, false);
    return *GOTSection;
  }

  Section &getStubsSection() {
    if (!StubsSection) {
      auto StubsProt = static_cast<sys::Memory::ProtectionFlags>(
          sys::Memory::MF_READ | sys::Memory::MF_EXEC);
      StubsSection = &G.createSection("$__STUBS", StubsProt, false);
    }
    return *StubsSection;
  }

  static const uint8_t NullGOTEntryContent[8];
  static const uint8_t StubContent[6];
  Section *GOTSection = nullptr;
  Section *StubsSection = nullptr;
};

const uint8_t MachO_x86_64_GOTAndStubsBuilder::NullGOTEntryContent[8] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
const uint8_t MachO_x86_64_GOTAndStubsBuilder::StubContent[6] = {
    0xFF, 0x25, 0x00, 0x00, 0x00, 0x00};
} // namespace

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

  Expected<std::unique_ptr<AtomGraph>>
  buildGraph(MemoryBufferRef ObjBuffer) override {
    auto MachOObj = object::ObjectFile::createMachOObjectFile(ObjBuffer);
    if (!MachOObj)
      return MachOObj.takeError();
    return MachOAtomGraphBuilder_x86_64(**MachOObj).buildGraph();
  }

  static Error targetOutOfRangeError(const Edge &E) {
    std::string ErrMsg;
    {
      raw_string_ostream ErrStream(ErrMsg);
      ErrStream << "Target \"" << E.getTarget() << "\" out of range";
    }
    return make_error<JITLinkError>(std::move(ErrMsg));
  }

  Error applyFixup(DefinedAtom &A, const Edge &E, char *AtomWorkingMem) const {
    using namespace support;

    char *FixupPtr = AtomWorkingMem + E.getOffset();
    JITTargetAddress FixupAddress = A.getAddress() + E.getOffset();

    switch (E.getKind()) {
    case Branch32:
    case PCRel32:
    case PCRel32Anon: {
      int64_t Value =
          E.getTarget().getAddress() - (FixupAddress + 4) + E.getAddend();
      if (Value < std::numeric_limits<int32_t>::min() ||
          Value > std::numeric_limits<int32_t>::max())
        return targetOutOfRangeError(E);
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
        return targetOutOfRangeError(E);
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
        return targetOutOfRangeError(E);
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
          return targetOutOfRangeError(E);
        *(little32_t *)FixupPtr = Value;
      } else
        *(little64_t *)FixupPtr = Value;
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
    // Add a mark-live pass.
    if (auto MarkLive = Ctx->getMarkLivePass(TT))
      Config.PrePrunePasses.push_back(std::move(MarkLive));
    else
      Config.PrePrunePasses.push_back(markAllAtomsLive);

    // Add an in-place GOT/Stubs pass.
    Config.PostPrunePasses.push_back([](AtomGraph &G) -> Error {
      MachO_x86_64_GOTAndStubsBuilder(G).run();
      return Error::success();
    });
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
