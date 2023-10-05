//===- MC/MCCASObjectV1.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MCCAS/MCCASObjectV1.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/CAS/CASID.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAbbrev.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/MCCAS/MCCASDebugV1.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"
#include <memory>

// FIXME: Fix dependency here.
#include "llvm/CASObjectFormats/Encoding.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace llvm::mccasformats;
using namespace llvm::mccasformats::v1;
using namespace llvm::mccasformats::reader;

using namespace llvm::casobjectformats::encoding;

constexpr StringLiteral MCAssemblerRef::KindString;
constexpr StringLiteral PaddingRef::KindString;

#define CASV1_SIMPLE_DATA_REF(RefName, IdentifierName)                         \
  constexpr StringLiteral RefName::KindString;
#define CASV1_SIMPLE_GROUP_REF(RefName, IdentifierName)                        \
  constexpr StringLiteral RefName::KindString;
#define MCFRAGMENT_NODE_REF(MCFragmentName, MCEnumName, MCEnumIdentifier)      \
  constexpr StringLiteral MCFragmentName##Ref::KindString;
#include "llvm/MCCAS/MCCASObjectV1.def"
constexpr StringLiteral DebugInfoSectionRef::KindString;

void MCSchema::anchor() {}
char MCSchema::ID = 0;

cl::opt<unsigned>
    MCDataMergeThreshold("mc-cas-data-merge-threshold",
                         cl::desc("MCDataFragment merge threshold"),
                         cl::init(1024));
cl::opt<bool>
    DebugInfoUnopt("debug-info-unopt",
                   cl::desc("Whether debug info storage should be optimized or "
                            "just stored as one cas block per section"),
                   cl::init(false));

enum RelEncodeLoc {
  Atom,
  Section,
  CompileUnit,
};

cl::opt<RelEncodeLoc> RelocLocation(
    "mc-cas-reloc-encode-in", cl::desc("Where to put relocation in encoding"),
    cl::values(clEnumVal(Atom, "In atom"), clEnumVal(Section, "In section"),
               clEnumVal(CompileUnit, "In compile unit")),
    cl::init(Atom));

class AbbrevSetWriter;

/// A DWARFObject implementation that can be used to dwarfdump CAS-formatted
/// debug info.
class InMemoryCASDWARFObject : public DWARFObject {
  ArrayRef<char> DebugAbbrevSection;
  DWARFSection DebugStringOffsetsSection;
  bool IsLittleEndian;

public:
  InMemoryCASDWARFObject(ArrayRef<char> AbbrevContents,
                         ArrayRef<char> StringOffsetsContents,
                         bool IsLittleEndian)
      : DebugAbbrevSection(AbbrevContents),
        DebugStringOffsetsSection({toStringRef(StringOffsetsContents)}),
        IsLittleEndian(IsLittleEndian) {}
  bool isLittleEndian() const override { return IsLittleEndian; }

  StringRef getAbbrevSection() const override {
    return toStringRef(DebugAbbrevSection);
  }

  const DWARFSection &getStrOffsetsSection() const override {
    return DebugStringOffsetsSection;
  }

  std::optional<RelocAddrEntry> find(const DWARFSection &Sec,
                                     uint64_t Pos) const override {
    return {};
  }

  /// Create a DwarfCompileUnit that represents the compile unit at \p CUOffset
  /// in the debug info section, and iterate over the individual DIEs to
  /// identify and separate the Forms that do not deduplicate in
  /// PartitionedDebugInfoSection::FormsToPartition and those that do
  /// deduplicate. Store both kinds of Forms in their own buffers per compile
  /// unit.
  Error partitionCUData(ArrayRef<char> DebugInfoData, uint64_t AbbrevOffset,
                        DWARFContext *Ctx, MCCASBuilder &Builder,
                        AbbrevSetWriter &AbbrevWriter, uint16_t DwarfVersion);
};

struct CUInfo {
  uint64_t CUSize;
  uint32_t AbbrevOffset;
  uint16_t DwarfVersion;
};
static Expected<CUInfo> getAndSetDebugAbbrevOffsetAndSkip(
    MutableArrayRef<char> CUData, support::endianness Endian,
    std::optional<uint32_t> NewOffset = std::nullopt);
Expected<cas::ObjectProxy>
MCSchema::createFromMCAssemblerImpl(MachOCASWriter &ObjectWriter,
                                    MCAssembler &Asm, const MCAsmLayout &Layout,
                                    raw_ostream *DebugOS) const {
  return MCAssemblerRef::create(*this, ObjectWriter, Asm, Layout, DebugOS);
}

Error MCSchema::serializeObjectFile(cas::ObjectProxy RootNode,
                                    raw_ostream &OS) const {
  if (!isRootNode(RootNode))
    return createStringError(inconvertibleErrorCode(), "invalid root node");
  auto Asm = MCAssemblerRef::get(*this, RootNode.getRef());
  if (!Asm)
    return Asm.takeError();

  return Asm->materialize(OS);
}

// Helper function to load the list of references inside an ObjectProxy.
SmallVector<cas::ObjectRef> loadReferences(const cas::ObjectProxy &Proxy) {
  SmallVector<cas::ObjectRef> Refs;
  if (auto E = Proxy.forEachReference([&](cas::ObjectRef ID) -> Error {
        Refs.push_back(ID);
        return Error::success();
      }))
    llvm_unreachable("Callback never returns an error");
  return Refs;
}

MCSchema::MCSchema(cas::ObjectStore &CAS) : MCSchema::RTTIExtends(CAS) {
  // Fill the cache immediately to preserve thread-safety.
  if (Error E = fillCache())
    report_fatal_error(std::move(E));
}

Error MCSchema::fillCache() {
  std::optional<cas::ObjectRef> RootKindID;
  const unsigned Version = 0; // Bump this to error on old object files.
  if (Error E = CAS.storeFromString(std::nullopt,
                                    "mc:v1:schema:" + Twine(Version).str())
                    .moveInto(RootKindID))
    return E;

  StringRef AllKindStrings[] = {
      PaddingRef::KindString,
      MCAssemblerRef::KindString,
      DebugInfoSectionRef::KindString,
      DIEDataRef::KindString,
#define CASV1_SIMPLE_DATA_REF(RefName, IdentifierName) RefName::KindString,
#define CASV1_SIMPLE_GROUP_REF(RefName, IdentifierName) RefName::KindString,
#define MCFRAGMENT_NODE_REF(MCFragmentName, MCEnumName, MCEnumIdentifier)      \
  MCFragmentName##Ref::KindString,
#include "llvm/MCCAS/MCCASObjectV1.def"
  };
  cas::ObjectRef Refs[] = {*RootKindID};
  SmallVector<cas::ObjectRef> IDs = {*RootKindID};
  for (StringRef KS : AllKindStrings) {
    auto ExpectedID = CAS.storeFromString(Refs, KS);
    if (!ExpectedID)
      return ExpectedID.takeError();
    IDs.push_back(*ExpectedID);
    KindStrings.push_back(std::make_pair(KindStrings.size(), KS));
    assert(KindStrings.size() < UCHAR_MAX &&
           "Ran out of bits for kind strings");
  }

  return CAS.storeFromString(IDs, "mc:v1:root").moveInto(RootNodeTypeID);
}

std::optional<StringRef>
MCSchema::getKindString(const cas::ObjectProxy &Node) const {
  assert(&Node.getCAS() == &CAS);
  StringRef Data = Node.getData();
  if (Data.empty())
    return std::nullopt;

  unsigned char ID = Data[0];
  for (auto &I : KindStrings)
    if (I.first == ID)
      return I.second;
  return std::nullopt;
}

bool MCSchema::isRootNode(const cas::ObjectProxy &Node) const {
  if (Node.getNumReferences() < 1)
    return false;
  return Node.getReference(0) == *RootNodeTypeID;
}

bool MCSchema::isNode(const cas::ObjectProxy &Node) const {
  // This is a very weak check!
  return bool(getKindString(Node));
}

Expected<MCObjectProxy::Builder>
MCObjectProxy::Builder::startRootNode(const MCSchema &Schema,
                                      StringRef KindString) {
  Builder B(Schema);
  B.Refs.push_back(Schema.getRootNodeTypeID());

  if (Error E = B.startNodeImpl(KindString))
    return std::move(E);
  return std::move(B);
}

Error MCObjectProxy::Builder::startNodeImpl(StringRef KindString) {
  std::optional<unsigned char> TypeID = Schema->getKindStringID(KindString);
  if (!TypeID)
    return createStringError(inconvertibleErrorCode(),
                             "invalid mc format kind string: " + KindString);
  Data.push_back(*TypeID);
  return Error::success();
}

Expected<MCObjectProxy::Builder>
MCObjectProxy::Builder::startNode(const MCSchema &Schema,
                                  StringRef KindString) {
  Builder B(Schema);
  if (Error E = B.startNodeImpl(KindString))
    return std::move(E);
  return std::move(B);
}

Expected<MCObjectProxy> MCObjectProxy::Builder::build() {
  return MCObjectProxy::get(*Schema, Schema->CAS.createProxy(Refs, Data));
}

StringRef MCObjectProxy::getKindString() const {
  std::optional<StringRef> KS = getSchema().getKindString(*this);
  assert(KS && "Expected valid kind string");
  return *KS;
}

std::optional<unsigned char>
MCSchema::getKindStringID(StringRef KindString) const {
  for (auto &I : KindStrings)
    if (I.second == KindString)
      return I.first;
  return std::nullopt;
}

Expected<MCObjectProxy> MCObjectProxy::get(const MCSchema &Schema,
                                           Expected<cas::ObjectProxy> Ref) {
  if (!Ref)
    return Ref.takeError();
  if (!Schema.isNode(*Ref))
    return createStringError(inconvertibleErrorCode(),
                             "invalid kind-string for node in mc-cas-schema");
  return MCObjectProxy(Schema, *Ref);
}

static Expected<StringRef> consumeDataOfSize(StringRef &Data, unsigned Size) {
  if (Data.size() < Size)
    return createStringError(inconvertibleErrorCode(),
                             "Requested data go beyond the buffer");

  auto Ret = Data.take_front(Size);
  Data = Data.drop_front(Size);

  return Ret;
}

#define CASV1_SIMPLE_DATA_REF(RefName, IdentifierName)                         \
  Expected<RefName> RefName::create(MCCASBuilder &MB, StringRef Name) {        \
    auto B = Builder::startNode(MB.Schema, KindString);                        \
    if (!B)                                                                    \
      return B.takeError();                                                    \
    B->Data.append(Name);                                                      \
    return get(B->build());                                                    \
  }                                                                            \
  Expected<RefName> RefName::get(Expected<MCObjectProxy> Ref) {                \
    auto Specific = SpecificRefT::getSpecific(std::move(Ref));                 \
    if (!Specific)                                                             \
      return Specific.takeError();                                             \
    return RefName(*Specific);                                                 \
  }
#include "llvm/MCCAS/MCCASObjectV1.def"

Expected<PaddingRef> PaddingRef::create(MCCASBuilder &MB, uint64_t Size) {
  // Fake a FT_Fill Fragment that is zero filled.
  Expected<Builder> B = Builder::startNode(MB.Schema, KindString);
  if (!B)
    return B.takeError();

  writeVBR8(Size, B->Data);
  return get(B->build());
}

Expected<uint64_t> PaddingRef::materialize(raw_ostream &OS) const {
  StringRef Remaining = getData();
  uint64_t Size;
  if (auto E = consumeVBR8(Remaining, Size))
    return std::move(E);
  OS.write_zeros(Size);
  return Size;
}

Expected<PaddingRef> PaddingRef::get(Expected<MCObjectProxy> Ref) {
  auto Specific = SpecificRefT::getSpecific(std::move(Ref));
  if (!Specific)
    return Specific.takeError();

  return PaddingRef(*Specific);
}

static void writeRelocations(ArrayRef<MachO::any_relocation_info> Rels,
                             SmallVectorImpl<char> &Data) {
  for (auto Rel : Rels) {
    // FIXME: Might be better just encode raw data?
    writeVBR8(Rel.r_word0, Data);
    writeVBR8(Rel.r_word1, Data);
  }
}

static Error decodeRelocations(MCCASReader &Reader, StringRef Data) {
  while (!Data.empty()) {
    MachO::any_relocation_info Rel;
    if (auto E = consumeVBR8(Data, Rel.r_word0))
      return E;
    if (auto E = consumeVBR8(Data, Rel.r_word1))
      return E;
    Reader.Relocations.back().push_back(Rel);
  }
  return Error::success();
}

Error MCObjectProxy::encodeReferences(ArrayRef<cas::ObjectRef> Refs,
                                      SmallVectorImpl<char> &Data,
                                      SmallVectorImpl<cas::ObjectRef> &IDs) {
  DenseMap<cas::ObjectRef, unsigned> RefMap;
  SmallVector<cas::ObjectRef> CompactRefs;
  for (const auto &ID : Refs) {
    auto I = RefMap.try_emplace(ID, CompactRefs.size());
    if (I.second)
      CompactRefs.push_back(ID);
  }

  // Guess the size of the encoding. Made an assumption that VBR8 encoding is
  // 1 byte (the minimal).
  size_t ReferenceSize = Refs.size() * sizeof(void *);
  size_t CompactSize = CompactRefs.size() * sizeof(void *) + Refs.size();
  if (ReferenceSize <= CompactSize) {
    writeVBR8(0, Data);
    IDs.append(Refs.begin(), Refs.end());
    return Error::success();
  }

  writeVBR8(Refs.size(), Data);
  for (const auto &ID : Refs) {
    auto Idx = RefMap.find(ID);
    assert(Idx != RefMap.end() && "ID must be in the map");
    writeVBR8(Idx->second, Data);
  }

  IDs.append(CompactRefs.begin(), CompactRefs.end());
  return Error::success();
}

Expected<SmallVector<cas::ObjectRef>>
MCObjectProxy::decodeReferences(const MCObjectProxy &Node,
                                StringRef &Remaining) {
  SmallVector<cas::ObjectRef> Refs = loadReferences(Node);

  unsigned Size = 0;
  if (auto E = consumeVBR8(Remaining, Size))
    return std::move(E);

  if (!Size)
    return Refs;

  SmallVector<cas::ObjectRef> CompactRefs;
  for (unsigned I = 0; I < Size; ++I) {
    unsigned Idx = 0;
    if (auto E = consumeVBR8(Remaining, Idx))
      return std::move(E);

    if (Idx >= Refs.size())
      return createStringError(inconvertibleErrorCode(), "invalid ref index");

    CompactRefs.push_back(Refs[Idx]);
  }

  return CompactRefs;
}

Expected<GroupRef> GroupRef::create(MCCASBuilder &MB,
                                    ArrayRef<cas::ObjectRef> Fragments) {
  Expected<Builder> B = Builder::startNode(MB.Schema, KindString);
  if (!B)
    return B.takeError();

  if (auto E = encodeReferences(Fragments, B->Data, B->Refs))
    return std::move(E);

  return get(B->build());
}

template <typename RefTy>
static Expected<SmallVector<RefTy, 0>> findRefs(MCCASReader &Reader,
                                                ArrayRef<cas::ObjectRef> Refs) {
  SmallVector<RefTy, 0> TopRefs;
  for (auto ID : Refs) {
    auto Node = Reader.getObjectProxy(ID);
    if (!Node)
      return Node.takeError();
    if (auto TopRef = RefTy::Cast(*Node))
      TopRefs.push_back(*TopRef);
  }
  if (TopRefs.size())
    return std::move(TopRefs);
  return createStringError(inconvertibleErrorCode(),
                           "failed to find reference");
}

template <typename RefTy>
static Expected<RefTy> findRef(MCCASReader &Reader,
                               ArrayRef<cas::ObjectRef> Refs) {
  auto FoundRefs = findRefs<RefTy>(Reader, Refs);
  if (!FoundRefs) return FoundRefs.takeError();
  return FoundRefs->front();
}

Expected<uint64_t> materializeAbbrevFromTagImpl(MCCASReader &Reader,
                                                DebugAbbrevSectionRef AbbrevRef,
                                                ArrayRef<cas::ObjectRef> Refs) {
  auto MaybeDebugInfoSectionRef = findRef<DebugInfoSectionRef>(Reader, Refs);
  if (!MaybeDebugInfoSectionRef)
    return MaybeDebugInfoSectionRef.takeError();
  SmallVector<cas::ObjectRef> DebugInfoSectionRefs =
      loadReferences(*MaybeDebugInfoSectionRef);

  auto TopRefs = findRefs<DIETopLevelRef>(Reader, DebugInfoSectionRefs);
  if (!TopRefs)
    return TopRefs.takeError();

  uint64_t Size = 0;
  uint64_t MaxDIEAbbrevCount = 1;
  for (auto TopRef : *TopRefs) {
    auto LoadedTopRef = loadDIETopLevel(TopRef);
    if (!LoadedTopRef)
      return LoadedTopRef.takeError();
    Size += reconstructAbbrevSection(Reader.OS, LoadedTopRef->AbbrevEntries,
                                     MaxDIEAbbrevCount);
  }

  // FIXME: Currently, one DIELevelTopRef corresponds to one Compile Unit, but
  // multiple compile units could refer to the same abbreviation contribution,
  // such is the case with swift, where both Compile Units have the abbr_offset
  // of 0.
  // Dwarf 5: Section 7.5.3: The abbreviations for a given compilation
  // unit end with an entry consisting of a 0 byte for the abbreviation code.
  Reader.OS.write_zeros(1);
  Size += 1;

  auto MaybePaddingRef = findRef<PaddingRef>(Reader, loadReferences(AbbrevRef));
  if (!MaybePaddingRef)
    return MaybePaddingRef.takeError();

  Expected<uint64_t> MaybePaddingSize = MaybePaddingRef->materialize(Reader.OS);
  if (!MaybePaddingSize)
    return MaybePaddingSize.takeError();

  return Size + *MaybePaddingSize;
}

static Error materializeDebugInfoOpt(MCCASReader &Reader,
                                     ArrayRef<cas::ObjectRef> Refs,
                                     raw_ostream *SectionStream) {

  auto MaybeTopRefs = findRefs<DIETopLevelRef>(Reader, Refs);
  auto HeaderCallback = [&](StringRef HeaderData) {
    *SectionStream << HeaderData;
  };

  auto StartTagCallback = [&](dwarf::Tag, uint64_t AbbrevIdx) {
    encodeULEB128(decodeAbbrevIndexAsDwarfAbbrevIdx(AbbrevIdx), *SectionStream);
  };

  auto AttrCallback = [&](dwarf::Attribute, dwarf::Form Form,
                          StringRef FormData, bool) {
    if (Form == dwarf::Form::DW_FORM_ref4_cas ||
        Form == dwarf::Form::DW_FORM_strp_cas) {
      auto Reader = BinaryStreamReader(FormData, support::endianness::little);
      uint64_t Data64;
      if (auto Err = Reader.readULEB128(Data64))
        handleAllErrors(std::move(Err));
      uint32_t Data32 = Data64;
      assert(Data32 == Data64 && Reader.empty());
      SectionStream->write(reinterpret_cast<char *>(&Data32), sizeof(Data32));
    } else
      *SectionStream << FormData;
  };

  auto EndTagCallback = [&](bool HadChildren) {
    SectionStream->write_zeros(HadChildren);
  };

  if (!MaybeTopRefs)
    return MaybeTopRefs.takeError();

  SmallVector<StringRef, 0> TotAbbrevEntries;
  for (auto MaybeTopRef : *MaybeTopRefs) {

    if (auto E = visitDebugInfo(TotAbbrevEntries, std::move(MaybeTopRef),
                                HeaderCallback, StartTagCallback, AttrCallback,
                                EndTagCallback))
      return E;
  }
  return Error::success();
}

static Error materializeDebugInfoUnopt(MCCASReader &Reader,
                                       ArrayRef<cas::ObjectRef> Refs,
                                       SmallVectorImpl<char> &SectionContents) {

  for (auto Ref : Refs) {
    auto Node = Reader.getObjectProxy(Ref);
    if (!Node)
      return Node.takeError();
    if (auto F = DebugInfoUnoptRef::Cast(*Node)) {
      append_range(SectionContents, F->getData());
      continue;
    }
    if (auto F = PaddingRef::Cast(*Node)) {
      raw_svector_ostream OS(SectionContents);
      auto Size = F->materialize(OS);
      if (!Size)
        return Size.takeError();
      continue;
    }
    llvm_unreachable("Incorrect CAS Object in SectionContents");
  }
  return Error::success();
}

static Expected<uint64_t>
materializeDebugInfoFromTagImpl(MCCASReader &Reader,
                                DebugInfoSectionRef SectionRef) {
  SmallVector<cas::ObjectRef> Refs = loadReferences(SectionRef);
  SmallVector<char, 0> SectionContents;
  raw_svector_ostream SectionStream(SectionContents);
  auto Node = Reader.getObjectProxy(Refs[0]);
  if (!Node)
    return Node.takeError();
  if (auto UnoptRef = DebugInfoUnoptRef::Cast(*Node)) {
    if (Refs.size() > 2)
      return createStringError(
          inconvertibleErrorCode(),
          "If a DebugInfoUnoptRef is seen, there should be no more than 2 "
          "CAS objects under the DebugInfoSectionRef!");
    if (Error E = materializeDebugInfoUnopt(Reader, Refs, SectionContents))
      return std::move(E);
  } else {
    if (Error E = materializeDebugInfoOpt(Reader, Refs, &SectionStream))
      return std::move(E);
  }
  Reader.Relocations.emplace_back();
  if (auto E = decodeRelocations(Reader, SectionRef.getData()))
    return std::move(E);

  auto MaybePaddingRef = findRef<PaddingRef>(Reader, Refs);
  if (!MaybePaddingRef)
    return MaybePaddingRef.takeError();

  Expected<uint64_t> Size = MaybePaddingRef->materialize(SectionStream);
  if (!Size)
    return Size.takeError();

  Reader.OS << SectionContents;
  return SectionContents.size();
}

Expected<uint64_t> GroupRef::materialize(MCCASReader &Reader,
                                         raw_ostream *Stream) const {
  unsigned Size = 0;
  StringRef Remaining = getData();
  auto Refs = decodeReferences(*this, Remaining);
  if (!Refs)
    return Refs.takeError();

  for (auto ID : *Refs) {
    auto Node = Reader.getObjectProxy(ID);
    uint64_t FragSize = 0;
    if (!Node)
      return Node.takeError();
    if (auto AbbrevRef = DebugAbbrevSectionRef::Cast(*Node)) {
      auto AbbrevRefs = loadReferences(*Node);
      auto Obj = Reader.getObjectProxy(AbbrevRefs[0]);
      if (!Obj)
        return Obj.takeError();
      if (auto MaybeUnoptRef = DebugAbbrevUnoptRef::Cast(*Obj)) {
        if (Refs->size() > 2)
          return createStringError(
              inconvertibleErrorCode(),
              "If a DebugAbbrevUnoptRef is seen, there should be no more than "
              "2 CAS objects under the DebugAbbrevSectionRef!");
        auto FragmentSize =
            Reader.materializeDebugAbbrevUnopt(ArrayRef(AbbrevRefs));
        if (!FragmentSize)
          return FragmentSize.takeError();
        FragSize = *FragmentSize;
      } else {
        auto FragmentSize =
            materializeAbbrevFromTagImpl(Reader, *AbbrevRef, ArrayRef(*Refs));
        if (!FragmentSize)
          return FragmentSize.takeError();
        FragSize = *FragmentSize;
      }
      Size += FragSize;
      continue;
    }
    auto FragmentSize = Reader.materializeGroup(ID);
    if (!FragmentSize)
      return FragmentSize.takeError();
    Size += *FragmentSize;
  }

  if (!Remaining.empty())
    return createStringError(inconvertibleErrorCode(),
                             "Group should not have relocations");

  return Size;
}

Expected<SymbolTableRef>
SymbolTableRef::create(MCCASBuilder &MB, ArrayRef<cas::ObjectRef> Fragments) {
  Expected<Builder> B = Builder::startNode(MB.Schema, KindString);
  if (!B)
    return B.takeError();

  if (auto E = encodeReferences(Fragments, B->Data, B->Refs))
    return std::move(E);

  return get(B->build());
}

Expected<uint64_t> SymbolTableRef::materialize(MCCASReader &Reader,
                                               raw_ostream *Stream) const {
  unsigned Size = 0;
  StringRef Remaining = getData();
  auto Refs = decodeReferences(*this, Remaining);
  if (!Refs)
    return Refs.takeError();

  for (auto ID : *Refs) {
    auto FragmentSize = Reader.materializeGroup(ID);
    if (!FragmentSize)
      return FragmentSize.takeError();
    Size += *FragmentSize;
  }

  return Size;
}

Expected<SectionRef> SectionRef::create(MCCASBuilder &MB,
                                        ArrayRef<cas::ObjectRef> Fragments) {
  Expected<Builder> B = Builder::startNode(MB.Schema, KindString);
  if (!B)
    return B.takeError();

  if (auto E = encodeReferences(Fragments, B->Data, B->Refs))
    return std::move(E);

  writeRelocations(MB.getSectionRelocs(), B->Data);

  return get(B->build());
}

Expected<DebugInfoSectionRef>
DebugInfoSectionRef::create(MCCASBuilder &MB,
                            ArrayRef<cas::ObjectRef> ChildrenNode) {
  Expected<Builder> B = Builder::startNode(MB.Schema, KindString);
  if (!B)
    return B.takeError();

  append_range(B->Refs, ChildrenNode);
  writeRelocations(MB.getSectionRelocs(), B->Data);
  return get(B->build());
}

Expected<DebugAbbrevSectionRef>
DebugAbbrevSectionRef::create(MCCASBuilder &MB,
                              ArrayRef<cas::ObjectRef> Fragments) {
  Expected<Builder> B = Builder::startNode(MB.Schema, KindString);
  if (!B)
    return B.takeError();

  if (auto E = encodeReferences(Fragments, B->Data, B->Refs))
    return std::move(E);

  writeRelocations(MB.getSectionRelocs(), B->Data);
  return get(B->build());
}

Expected<DebugLineSectionRef>
DebugLineSectionRef::create(MCCASBuilder &MB,
                            ArrayRef<cas::ObjectRef> Fragments) {
  Expected<Builder> B = Builder::startNode(MB.Schema, KindString);
  if (!B)
    return B.takeError();

  if (auto E = encodeReferences(Fragments, B->Data, B->Refs))
    return std::move(E);

  writeRelocations(MB.getSectionRelocs(), B->Data);
  return get(B->build());
}

Expected<DebugStringSectionRef>
DebugStringSectionRef::create(MCCASBuilder &MB,
                              ArrayRef<cas::ObjectRef> Fragments) {
  Expected<Builder> B = Builder::startNode(MB.Schema, KindString);
  if (!B)
    return B.takeError();

  if (auto E = encodeReferences(Fragments, B->Data, B->Refs))
    return std::move(E);

  writeRelocations(MB.getSectionRelocs(), B->Data);
  return get(B->build());
}

Expected<uint64_t> SectionRef::materialize(MCCASReader &Reader,
                                           raw_ostream *Stream) const {
  // Start a new section for relocations.
  Reader.Relocations.emplace_back();
  SmallVector<char, 0> SectionContents;
  raw_svector_ostream SectionStream(SectionContents);

  unsigned Size = 0;
  StringRef Remaining = getData();
  auto Refs = decodeReferences(*this, Remaining);
  if (!Refs)
    return Refs.takeError();

  if (auto E = Reader.checkIfAddendRefExistsAndCopy(*Refs))
    return std::move(E);

  if (auto E = decodeRelocations(Reader, Remaining))
    return std::move(E);

  SmallVector<char, 0> FragmentContents;
  raw_svector_ostream FragmentStream(FragmentContents);

  for (auto ID : *Refs) {
    auto FragmentSize = Reader.materializeSection(ID, &FragmentStream);
    if (!FragmentSize)
      return FragmentSize.takeError();
    Size += *FragmentSize;
  }

  auto AddendSize =
      Reader.reconstructSection(SectionContents, FragmentContents);
  if (!AddendSize)
    return AddendSize.takeError();

  Size += *AddendSize;

  Reader.OS << SectionContents;
  // Reset the state for section materialization.
  Reader.AddendBufferIndex = 0;
  Reader.Addends.clear();

  return Size;
}

struct LineTablePrologue {
  uint64_t Length;
  uint16_t Version;
  uint32_t PrologueLength;
  uint64_t Offset;
  uint8_t OpcodeBase;
  dwarf::DwarfFormat Format;
};

static Expected<LineTablePrologue>
getLineTableLengthInfoAndVersion(DWARFDataExtractor &LineTableDataReader,
                                 uint64_t *OffsetPtr) {
  LineTablePrologue Prologue;
  Error Err = Error::success();
  // From DWARF 5 section 7.4:
  // In the 32-bit DWARF format, an initial length field [...] is an unsigned
  // 4-byte integer (which must be less than 0xfffffff0);
  auto Length = LineTableDataReader.getU32(OffsetPtr, &Err);
  if (Err)
    return std::move(Err);
  if (Length >= 0xfffffff0)
    return createStringError(inconvertibleErrorCode(),
                             "DWARF input is not in the 32-bit format");
  Prologue.Length = Length;
  Prologue.Format = llvm::dwarf::DWARF32;
  auto Version = LineTableDataReader.getU16(OffsetPtr, &Err);
  if (Err)
    return std::move(Err);
  if (Version >= 5) {
    // Dwarf 5 Section 6.2.4:
    // Line Table Header Format is now changed with an address_size and
    // segment_selector_size after the version. Parse both values from the
    // header.
    auto AddressSize = LineTableDataReader.getU8(OffsetPtr, &Err);
    if (Err)
      return std::move(Err);
    if (AddressSize != 8)
      return createStringError(
          inconvertibleErrorCode(),
          "Address size is not 8 bytes, unsupported architecture for MCCAS!");
    LineTableDataReader.getU8(OffsetPtr, &Err);
    if (Err)
      return std::move(Err);
  }

  Prologue.Version = Version;
  // Since we do not support 64 bit DWARF, the prologue length is 4 bytes in
  // size.
  auto PrologueLength = LineTableDataReader.getU32(OffsetPtr, &Err);
  if (Err)
    return std::move(Err);

  Prologue.PrologueLength = PrologueLength;
  Prologue.Offset = *OffsetPtr + PrologueLength;
  return Prologue;
}

static Expected<LineTablePrologue>
parseLineTableHeaderAndSkip(DWARFDataExtractor &LineTableDataReader) {
  uint64_t Offset = 0;
  uint64_t *OffsetPtr = &Offset;
  auto Prologue =
      getLineTableLengthInfoAndVersion(LineTableDataReader, OffsetPtr);
  if (!Prologue)
    return Prologue.takeError();
  Error Err = Error::success();
  // Parse Minimum instruction length.
  LineTableDataReader.getU8(OffsetPtr, &Err);
  // Parse Maximum Operands Per Instruction, if it exists.
  if (Prologue->Version >= 4)
    LineTableDataReader.getU8(OffsetPtr, &Err);
  // Parse DefaultIsStmt, LineBase, and LineRange.
  LineTableDataReader.getU8(OffsetPtr, &Err);
  LineTableDataReader.getU8(OffsetPtr, &Err);
  LineTableDataReader.getU8(OffsetPtr, &Err);
  // Parse OpcodeBase.
  Prologue->OpcodeBase = LineTableDataReader.getU8(OffsetPtr, &Err);
  if (Err)
    return std::move(Err);
  return Prologue;
}

static Error
handleExtendedOpcodesForLineTable(DWARFDataExtractor &LineTableDataReader,
                                  DWARFDataExtractor::Cursor &LineTableCursor,
                                  uint8_t SubOpcode, uint64_t Len,
                                  bool &IsEndSequence, bool &IsRelocation) {
  switch (SubOpcode) {
  case dwarf::DW_LNE_end_sequence: {
    // Takes no operand, it needs to be handled specially when materializing and
    // creating the CAS.
    IsEndSequence = true;
  } break;
  case dwarf::DW_LNE_set_address: {
    // Takes a relocatable address size, move cursor to the end of the
    // address.
    if (LineTableDataReader.getAddressSize() != Len - 1)
      return createStringError(inconvertibleErrorCode(),
                               "Address size mismatch");
    IsRelocation = true;
  } break;
  case dwarf::DW_LNE_define_file: {
    // Takes 4 arguments. The first is a null terminated string containing
    // a source file name. The second is an unsigned LEB128 number
    // representing the directory index of the directory in which the file
    // was found. The third is an unsigned LEB128 number representing the
    // time of last modification of the file. The fourth is an unsigned
    // LEB128 number representing the length in bytes of the file. Move
    // cursor to the end of the arguments.
    LineTableDataReader.getCStr(LineTableCursor);
    LineTableDataReader.getULEB128(LineTableCursor);
    LineTableDataReader.getULEB128(LineTableCursor);
    LineTableDataReader.getULEB128(LineTableCursor);
  } break;
  case dwarf::DW_LNE_set_discriminator:
    // Takes one operand, a ULEB128 value. Move cursor to end of operand.
    LineTableDataReader.getULEB128(LineTableCursor);
    break;
  default:
    llvm_unreachable("Unknown special opcode for line table");
    break;
  }
  return Error::success();
}

static Error
handleStandardOpcodesForLineTable(DWARFDataExtractor &LineTableDataReader,
                                  DWARFDataExtractor::Cursor &LineTableCursor,
                                  uint8_t Opcode, bool &IsSetFile,
                                  bool &IsRelocation) {
  switch (Opcode) {
  case dwarf::DW_LNS_copy:
  case dwarf::DW_LNS_negate_stmt:
  case dwarf::DW_LNS_set_basic_block:
  case dwarf::DW_LNS_const_add_pc:
  case dwarf::DW_LNS_set_prologue_end:
  case dwarf::DW_LNS_set_epilogue_begin:
    // Takes no arguments, move on
    break;
  case dwarf::DW_LNS_advance_pc:
  case dwarf::DW_LNS_advance_line:
  case dwarf::DW_LNS_set_column:
  case dwarf::DW_LNS_set_isa: {
    // Takes a single unsigned LEB128 operand, move cursor to the end of
    // operand.
    LineTableDataReader.getULEB128(LineTableCursor);
  } break;
  case dwarf::DW_LNS_set_file: {
    // Takes a single unsigned LEB128 operand, it needs to be handled specially
    // when materializing and creating the CAS.
    IsSetFile = true;
  } break;
  case dwarf::DW_LNS_fixed_advance_pc: {
    // Takes a single uhalf operand, move cursor to the end of operand.
    IsRelocation = true;
  } break;
  default:
    llvm_unreachable("Unknown standard opcode for line table");
    break;
  }
  return Error::success();
}

static Expected<std::pair<uint64_t, uint64_t>>
getOpcodeAndOperandSize(StringRef DistinctData, StringRef LineTableData,
                        uint64_t DistinctOffset, uint64_t LineTableOffset,
                        bool IsLittleEndian, uint8_t OpcodeBase) {
  DWARFDataExtractor LineTableDataReader(LineTableData, IsLittleEndian, 8);
  DWARFDataExtractor DistinctDataReader(DistinctData, IsLittleEndian, 8);
  DWARFDataExtractor::Cursor LineTableCursor(LineTableOffset);
  DWARFDataExtractor::Cursor DistinctCursor(DistinctOffset);

  auto Opcode = LineTableDataReader.getU8(LineTableCursor);
  if (Opcode == 0) {
    // Extended Opcodes always start with a zero opcode followed by
    // a uleb128 length so you can skip ones you don't know about
    uint64_t Len = LineTableDataReader.getULEB128(LineTableCursor);
    if (Len == 0)
      return createStringError(inconvertibleErrorCode(),
                               "0 Length for an extended opcode is wrong");

    uint8_t SubOpcode = LineTableDataReader.getU8(LineTableCursor);
    bool IsEndSequence = false;
    bool IsRelocation = false;
    auto Err = handleExtendedOpcodesForLineTable(
        LineTableDataReader, LineTableCursor, SubOpcode, Len, IsEndSequence,
        IsRelocation);
    if (Err)
      return std::move(Err);
    if (IsRelocation)
      DistinctDataReader.getRelocatedAddress(DistinctCursor);

    if (IsEndSequence) {
      // The SubOpcode is a DW_LNE_end_sequence, it takes no operand, but check
      // if this is the end of the line table and return.
      assert(LineTableData.size() == LineTableCursor.tell() &&
             "Malformed Line Table, data exists after a DW_LNE_end_sequence");
    }
  } else if (Opcode < OpcodeBase) {
    bool IsSetFile = false;
    bool IsRelocation = false;
    auto Err = handleStandardOpcodesForLineTable(
        LineTableDataReader, LineTableCursor, Opcode, IsSetFile, IsRelocation);
    if (Err)
      return std::move(Err);
    if (IsRelocation)
      DistinctDataReader.getRelocatedValue(DistinctCursor, 2);

    if (IsSetFile) {
      // The Opcode is DW_LNS_set_file, this means we need to get the file
      // number from the DistinctData, which is stored as a ULEB.
      DistinctDataReader.getULEB128(DistinctCursor);
    }
  } else {
    // Special Opcodes, do nothing.
  }

  if (!LineTableCursor)
    return LineTableCursor.takeError();
  if (!DistinctCursor)
    return DistinctCursor.takeError();

  return std::make_pair(LineTableCursor.tell() - LineTableOffset,
                        DistinctCursor.tell() - DistinctOffset);
}

static Expected<SmallVector<char, 0>>
materializeDebugLineSection(MCCASReader &Reader,
                            ArrayRef<cas::ObjectRef> Refs) {
  SmallVector<char, 0> DistinctData;
  uint64_t DistinctOffset = 0;
  uint8_t OpcodeBase = 0;
  SmallVector<char, 0> DebugLineSection;
  bool DistinctDebugLineRefSeen = false;
  bool DebugLineUnoptRefSeen = false;
  for (auto Ref : Refs) {
    auto Node = Reader.getObjectProxy(Ref);
    if (!Node)
      return Node.takeError();
    if (auto PadRef = PaddingRef::Cast(*Node)) {
      if (!DistinctDebugLineRefSeen && !DebugLineUnoptRefSeen)
        return createStringError(
            inconvertibleErrorCode(),
            "Line Table layout is incorrect, unexpected "
            "PaddingRef before a DistinctDebugLineRef or a DebugLineUnoptRef");
      raw_svector_ostream OS(DebugLineSection);
      auto Size = PadRef->materialize(OS);
      if (!Size)
        return Size.takeError();
      continue;
    }

    if (DebugLineUnoptRefSeen)
      return createStringError(
          inconvertibleErrorCode(),
          "DebugLineUnoptRef seen, only block allowed after is a PaddingRef");

    if (auto LineUnoptRef = DebugLineUnoptRef::Cast(*Node)) {

      DebugLineUnoptRefSeen = true;
      auto Data = LineUnoptRef->getData();
      DebugLineSection.append(Data.begin(), Data.end());
      continue;
    }
    if (auto DistinctRef = DistinctDebugLineRef::Cast(*Node)) {
      if (DistinctDebugLineRefSeen) {
        // This is the start of a new line table.
        DistinctOffset = 0;
        DistinctData.clear();
      }

      DistinctDebugLineRefSeen = true;
      auto Data = DistinctRef->getData();
      DistinctData.append(Data.begin(), Data.end());
      DWARFDataExtractor LineTableDataReader(Data, Reader.getEndian(), 8);
      auto Prologue = parseLineTableHeaderAndSkip(LineTableDataReader);
      if (!Prologue)
        return Prologue.takeError();
      DistinctOffset = Prologue->Offset;
      OpcodeBase = Prologue->OpcodeBase;
      // Copy line table prologue into final debug line section.
      DebugLineSection.append(DistinctData.begin(),
                              DistinctData.begin() + DistinctOffset);
      continue;
    }
    if (auto LineRef = DebugLineRef::Cast(*Node)) {

      if (!DistinctDebugLineRefSeen)
        return createStringError(inconvertibleErrorCode(),
                                 "Line Table layout is incorrect, unexpected "
                                 "DebugLineRef before a DistinctDebugLineRef");
      auto Data = LineRef->getData();
      uint64_t LineTableOffset = 0;
      while (LineTableOffset < Data.size()) {
        auto Sizes = getOpcodeAndOperandSize(toStringRef(DistinctData), Data,
                                             DistinctOffset, LineTableOffset,
                                             Reader.getEndian(), OpcodeBase);
        if (!Sizes)
          return Sizes.takeError();
        // Copy opcode and operand, only in the case of DW_LNS_set_file, the
        // operand will be in the DistinctData.
        DebugLineSection.append(Data.begin() + LineTableOffset,
                                Data.begin() + LineTableOffset + Sizes->first);
        LineTableOffset += Sizes->first;
        if (Sizes->second) {
          DebugLineSection.append(DistinctData.begin() + DistinctOffset,
                                  DistinctData.begin() + DistinctOffset +
                                      Sizes->second);
          DistinctOffset += Sizes->second;
        }
      }
      continue;
    }
    return createStringError(inconvertibleErrorCode(),
                             "Unknown cas node type for debug line section");
  }
  return DebugLineSection;
}

Expected<uint64_t> DebugLineSectionRef::materialize(MCCASReader &Reader,
                                                    raw_ostream *Stream) const {
  // Start a new section for relocations.
  Reader.Relocations.emplace_back();

  StringRef Remaining = getData();
  auto Refs = decodeReferences(*this, Remaining);
  if (!Refs)
    return Refs.takeError();

  auto SectionContents = materializeDebugLineSection(Reader, *Refs);
  if (!SectionContents)
    return SectionContents.takeError();

  if (auto E = decodeRelocations(Reader, Remaining))
    return std::move(E);
  Reader.OS << *SectionContents;

  return SectionContents->size();
}

Expected<uint64_t>
DebugStringSectionRef::materialize(MCCASReader &Reader,
                                   raw_ostream *Stream) const {
  // Start a new section for relocations.
  Reader.Relocations.emplace_back();
  SmallVector<char, 0> SectionContents;
  raw_svector_ostream SectionStream(SectionContents);

  unsigned Size = 0;
  StringRef Remaining = getData();
  auto Refs = decodeReferences(*this, Remaining);
  if (!Refs)
    return Refs.takeError();

  for (auto ID : *Refs) {
    auto FragmentSize = Reader.materializeSection(ID, &SectionStream);
    if (!FragmentSize)
      return FragmentSize.takeError();
    Size += *FragmentSize;
  }

  if (auto E = decodeRelocations(Reader, Remaining))
    return std::move(E);
  Reader.OS << SectionContents;

  return Size;
}

Expected<AtomRef> AtomRef::create(MCCASBuilder &MB,
                                  ArrayRef<cas::ObjectRef> Fragments) {
  Expected<Builder> B = Builder::startNode(MB.Schema, KindString);
  if (!B)
    return B.takeError();

  if (auto E = encodeReferences(Fragments, B->Data, B->Refs))
    return std::move(E);

  writeRelocations(MB.getAtomRelocs(), B->Data);

  return get(B->build());
}

Expected<uint64_t> AtomRef::materialize(MCCASReader &Reader,
                                        raw_ostream *Stream) const {
  unsigned Size = 0;
  StringRef Remaining = getData();
  auto Refs = decodeReferences(*this, Remaining);
  if (!Refs)
    return Refs.takeError();

  if (auto E = decodeRelocations(Reader, Remaining))
    return std::move(E);

  for (auto ID : *Refs) {
    auto FragmentSize = Reader.materializeAtom(ID, Stream);
    if (!FragmentSize)
      return FragmentSize.takeError();

    Size += *FragmentSize;
  }

  return Size;
}

Expected<MCAlignFragmentRef>
MCAlignFragmentRef::create(MCCASBuilder &MB, const MCAlignFragment &F,
                           unsigned FragmentSize,
                           ArrayRef<char> FragmentContents) {
  Expected<Builder> B = Builder::startNode(MB.Schema, KindString);
  if (!B)
    return B.takeError();

  uint64_t Count = FragmentSize / F.getValueSize();
  if (F.hasEmitNops()) {
    // Write 0 as size and use backend to emit nop.
    writeVBR8(0, B->Data);
    if (!MB.Asm.getBackend().writeNopData(MB.FragmentOS, Count,
                                          F.getSubtargetInfo()))
      report_fatal_error("unable to write nop sequence of " + Twine(Count) +
                         " bytes");
    B->Data.append(MB.FragmentData);
    return get(B->build());
  }
  writeVBR8(Count, B->Data);
  writeVBR8(F.getValue(), B->Data);
  writeVBR8(F.getValueSize(), B->Data);
  return get(B->build());
}

Expected<uint64_t> MCAlignFragmentRef::materialize(MCCASReader &Reader,
                                                   raw_ostream *Stream) const {
  uint64_t Count;
  auto Remaining = getData();
  auto Endian = Reader.getEndian();
  if (auto E = consumeVBR8(Remaining, Count))
    return std::move(E);

  // hasEmitNops.
  if (!Count) {
    *Stream << Remaining;
    return Remaining.size();
  }
  int64_t Value;
  unsigned ValueSize;
  if (auto E = consumeVBR8(Remaining, Value))
    return std::move(E);
  if (auto E = consumeVBR8(Remaining, ValueSize))
    return std::move(E);

  for (uint64_t I = 0; I != Count; ++I) {
    switch (ValueSize) {
    default:
      llvm_unreachable("Invalid size!");
    case 1:
      *Stream << char(Value);
      break;
    case 2:
      support::endian::write<uint16_t>(*Stream, Value, Endian);
      break;
    case 4:
      support::endian::write<uint32_t>(*Stream, Value, Endian);
      break;
    case 8:
      support::endian::write<uint64_t>(*Stream, Value, Endian);
      break;
    }
  }
  return Count * ValueSize;
}

Expected<MCBoundaryAlignFragmentRef> MCBoundaryAlignFragmentRef::create(
    MCCASBuilder &MB, const MCBoundaryAlignFragment &F, unsigned FragmentSize,
    ArrayRef<char> FragmentContents) {
  Expected<Builder> B = Builder::startNode(MB.Schema, KindString);
  if (!B)
    return B.takeError();
  if (!MB.Asm.getBackend().writeNopData(MB.FragmentOS, FragmentSize,
                                        F.getSubtargetInfo()))
    report_fatal_error("unable to write nop sequence of " +
                       Twine(FragmentSize) + " bytes");
  B->Data.append(MB.FragmentData);
  return get(B->build());
}

Expected<uint64_t>
MCBoundaryAlignFragmentRef::materialize(MCCASReader &Reader,
                                        raw_ostream *Stream) const {
  *Stream << getData();
  return getData().size();
}

Expected<MCCVInlineLineTableFragmentRef> MCCVInlineLineTableFragmentRef::create(
    MCCASBuilder &MB, const MCCVInlineLineTableFragment &F,
    unsigned FragmentSize, ArrayRef<char> FragmentContents) {
  Expected<Builder> B = Builder::startNode(MB.Schema, KindString);
  if (!B)
    return B.takeError();
  B->Data.append(F.getContents());
  return get(B->build());
}

Expected<uint64_t>
MCCVInlineLineTableFragmentRef::materialize(MCCASReader &Reader,
                                            raw_ostream *Stream) const {
  *Stream << getData();
  return getData().size();
}

Expected<MCDummyFragmentRef>
MCDummyFragmentRef::create(MCCASBuilder &MB, const MCDummyFragment &F,
                           unsigned FragmentSize,
                           ArrayRef<char> FragmentContents) {
  llvm_unreachable("Should not have been added");
}

Expected<uint64_t> MCDummyFragmentRef::materialize(MCCASReader &Reader,
                                                   raw_ostream *Stream) const {
  llvm_unreachable("Should not have been added");
}

Expected<MCFillFragmentRef>
MCFillFragmentRef::create(MCCASBuilder &MB, const MCFillFragment &F,
                          unsigned FragmentSize,
                          ArrayRef<char> FragmentContents) {
  Expected<Builder> B = Builder::startNode(MB.Schema, KindString);
  if (!B)
    return B.takeError();
  writeVBR8(FragmentSize, B->Data);
  writeVBR8(F.getValue(), B->Data);
  writeVBR8(F.getValueSize(), B->Data);
  return get(B->build());
}

Expected<uint64_t> MCFillFragmentRef::materialize(MCCASReader &Reader,
                                                  raw_ostream *Stream) const {
  StringRef Remaining = getData();
  uint64_t Size;
  uint64_t Value;
  unsigned ValueSize;
  if (auto E = consumeVBR8(Remaining, Size))
    return std::move(E);
  if (auto E = consumeVBR8(Remaining, Value))
    return std::move(E);
  if (auto E = consumeVBR8(Remaining, ValueSize))
    return std::move(E);

  // FIXME: Code duplication from writeFragment.
  const unsigned MaxChunkSize = 16;
  char Data[MaxChunkSize];
  for (unsigned I = 0; I != ValueSize; ++I) {
    unsigned Index =
        Reader.getEndian() == support::little ? I : (ValueSize - I - 1);
    Data[I] = uint8_t(Value >> (Index * 8));
  }
  for (unsigned I = ValueSize; I < MaxChunkSize; ++I)
    Data[I] = Data[I - ValueSize];

  const unsigned NumPerChunk = MaxChunkSize / ValueSize;
  const unsigned ChunkSize = ValueSize * NumPerChunk;

  StringRef Ref(Data, ChunkSize);
  for (uint64_t I = 0, E = Size / ChunkSize; I != E; ++I)
    *Stream << Ref;

  unsigned TrailingCount = Size % ChunkSize;
  if (TrailingCount)
    Stream->write(Data, TrailingCount);
  return Size;
}

Expected<MCLEBFragmentRef>
MCLEBFragmentRef::create(MCCASBuilder &MB, const MCLEBFragment &F,
                         unsigned FragmentSize,
                         ArrayRef<char> FragmentContents) {
  Expected<Builder> B = Builder::startNode(MB.Schema, KindString);
  if (!B)
    return B.takeError();
  B->Data.append(F.getContents());
  return get(B->build());
}

Expected<uint64_t> MCLEBFragmentRef::materialize(MCCASReader &Reader,
                                                 raw_ostream *Stream) const {
  *Stream << getData();
  return getData().size();
}

Expected<MCNopsFragmentRef>
MCNopsFragmentRef::create(MCCASBuilder &MB, const MCNopsFragment &F,
                          unsigned FragmentSize,
                          ArrayRef<char> FragmentContents) {
  Expected<Builder> B = Builder::startNode(MB.Schema, KindString);
  if (!B)
    return B.takeError();
  int64_t NumBytes = F.getNumBytes();
  int64_t ControlledNopLength = F.getControlledNopLength();
  int64_t MaximumNopLength =
      MB.Asm.getBackend().getMaximumNopSize(*F.getSubtargetInfo());
  if (ControlledNopLength > MaximumNopLength)
    ControlledNopLength = MaximumNopLength;
  if (!ControlledNopLength)
    ControlledNopLength = MaximumNopLength;
  while (NumBytes) {
    uint64_t NumBytesToEmit = (uint64_t)std::min(NumBytes, ControlledNopLength);
    assert(NumBytesToEmit && "try to emit empty NOP instruction");
    if (!MB.Asm.getBackend().writeNopData(MB.FragmentOS, NumBytesToEmit,
                                          F.getSubtargetInfo())) {
      report_fatal_error("unable to write nop sequence of the remaining " +
                         Twine(NumBytesToEmit) + " bytes");
      break;
    }
    NumBytes -= NumBytesToEmit;
  }
  B->Data.append(MB.FragmentData);
  return get(B->build());
}

Expected<uint64_t> MCNopsFragmentRef::materialize(MCCASReader &Reader,
                                                  raw_ostream *Stream) const {
  *Stream << getData();
  return getData().size();
}

Expected<MCOrgFragmentRef>
MCOrgFragmentRef::create(MCCASBuilder &MB, const MCOrgFragment &F,
                         unsigned FragmentSize,
                         ArrayRef<char> FragmentContents) {
  Expected<Builder> B = Builder::startNode(MB.Schema, KindString);
  if (!B)
    return B.takeError();
  writeVBR8(FragmentSize, B->Data);
  writeVBR8((char)F.getValue(), B->Data);
  return get(B->build());
}

Expected<uint64_t> MCOrgFragmentRef::materialize(MCCASReader &Reader,
                                                 raw_ostream *Stream) const {
  *Stream << getData();
  return getData().size();
}

Expected<MCSymbolIdFragmentRef>
MCSymbolIdFragmentRef::create(MCCASBuilder &MB, const MCSymbolIdFragment &F,
                              unsigned FragmentSize,
                              ArrayRef<char> FragmentContents) {
  Expected<Builder> B = Builder::startNode(MB.Schema, KindString);
  if (!B)
    return B.takeError();
  writeVBR8(F.getSymbol()->getIndex(), B->Data);
  return get(B->build());
}

Expected<uint64_t>
MCSymbolIdFragmentRef::materialize(MCCASReader &Reader,
                                   raw_ostream *Stream) const {
  *Stream << getData();
  return getData().size();
}

#define MCFRAGMENT_NODE_REF(MCFragmentName, MCEnumName, MCEnumIdentifier)      \
  Expected<MCFragmentName##Ref> MCFragmentName##Ref::create(                   \
      MCCASBuilder &MB, const MCFragmentName &F, unsigned FragmentSize,        \
      ArrayRef<char> FragmentContents) {                                       \
    Expected<Builder> B = Builder::startNode(MB.Schema, KindString);           \
    if (!B)                                                                    \
      return B.takeError();                                                    \
    MB.Asm.writeFragmentPadding(MB.FragmentOS, F, FragmentSize);               \
    ArrayRef<char> Contents = F.getContents();                                 \
    B->Data.append(MB.FragmentData);                                           \
    B->Data.append(FragmentContents.begin(), FragmentContents.end());          \
    assert(((MB.FragmentData.empty() && Contents.empty()) ||                   \
            (MB.FragmentData.size() + Contents.size() == FragmentSize)) &&     \
           "Size should match");                                               \
    return get(B->build());                                                    \
  }                                                                            \
  Expected<uint64_t> MCFragmentName##Ref::materialize(                         \
      MCCASReader &Reader, raw_ostream *Stream) const {                        \
    *Stream << getData();                                                      \
    return getData().size();                                                   \
  }
#define MCFRAGMENT_ENCODED_FRAGMENT_ONLY
#include "llvm/MCCAS/MCCASObjectV1.def"

Expected<MCAssemblerRef> MCAssemblerRef::get(Expected<MCObjectProxy> Ref) {
  auto Specific = SpecificRefT::getSpecific(std::move(Ref));
  if (!Specific)
    return Specific.takeError();

  return MCAssemblerRef(*Specific);
}

DwarfSectionsCache mccasformats::v1::getDwarfSections(MCAssembler &Asm) {
  return DwarfSectionsCache{
      Asm.getContext().getObjectFileInfo()->getDwarfInfoSection(),
      Asm.getContext().getObjectFileInfo()->getDwarfLineSection(),
      Asm.getContext().getObjectFileInfo()->getDwarfStrSection(),
      Asm.getContext().getObjectFileInfo()->getDwarfAbbrevSection(),
      Asm.getContext().getObjectFileInfo()->getDwarfStrOffSection()};
}

Error MCCASBuilder::prepare() {
  ObjectWriter.resetBuffer();
  ObjectWriter.prepareObject(Asm, Layout);
  assert(ObjectWriter.getContent().empty() &&
         "prepare stage writes no content");
  return Error::success();
}

Error MCCASBuilder::buildMachOHeader() {
  ObjectWriter.resetBuffer();
  ObjectWriter.writeMachOHeader(Asm, Layout);
  auto Header = HeaderRef::create(*this, ObjectWriter.getContent());
  if (!Header)
    return Header.takeError();

  addNode(*Header);
  return Error::success();
}

Error MCCASBuilder::buildFragment(const MCFragment &F, unsigned Size,
                                  ArrayRef<char> FragmentContents) {
  FragmentData.clear();
  switch (F.getKind()) {
#define MCFRAGMENT_NODE_REF(MCFragmentName, MCEnumName, MCEnumIdentifier)      \
  case MCFragment::MCEnumName: {                                               \
    const MCFragmentName &SF = cast<MCFragmentName>(F);                        \
    auto FN = MCFragmentName##Ref::create(*this, SF, Size, FragmentContents);  \
    if (!FN)                                                                   \
      return FN.takeError();                                                   \
    addNode(*FN);                                                              \
    return Error::success();                                                   \
  }
#include "llvm/MCCAS/MCCASObjectV1.def"
  }
  llvm_unreachable("unknown fragment");
}

class MCDataFragmentMerger {
public:
  MCDataFragmentMerger(MCCASBuilder &Builder, const MCSection *Sec)
      : Builder(Builder) {}
  ~MCDataFragmentMerger() { assert(MergeCandidates.empty() && "Not flushed"); }

  Error tryMerge(const MCFragment &F, unsigned Size,
                 ArrayRef<char> FinalFragmentContents);
  Error flush() { return emitMergedFragments(); }
  SmallVector<SmallVector<char, 0>> MergeCandidatesContents;

private:
  Error emitMergedFragments();
  void reset();

  MCCASBuilder &Builder;
  unsigned CurrentSize = 0;
  std::vector<std::pair<const MCFragment *, unsigned>> MergeCandidates;
};

Error MCDataFragmentMerger::tryMerge(const MCFragment &F, unsigned Size,
                                     ArrayRef<char> FinalFragmentContents) {
  bool IsSameAtom = Builder.getCurrentAtom() == F.getAtom();
  bool Oversized = CurrentSize + Size > MCDataMergeThreshold;
  // TODO: Try merge align fragment?
  bool IsMergeableFragment =
      isa<MCEncodedFragment>(F) || isa<MCAlignFragment>(F);
  // If not the same atom, flush merge candidate and return false.
  if (!IsSameAtom || !IsMergeableFragment || Oversized) {
    if (auto E = emitMergedFragments())
      return E;

    // If it is a new Atom, start a new sub-section.
    if (!IsSameAtom) {
      if (auto E = Builder.finalizeAtom())
        return E;
      Builder.startAtom(F.getAtom());
    }
  }

  // Emit none Data segments.
  if (!IsMergeableFragment) {
    if (auto E = Builder.buildFragment(F, Size, FinalFragmentContents))
      return E;

    return Error::success();
  }

  // Add the fragment to the merge candidate.
  CurrentSize += Size;
  MergeCandidates.emplace_back(&F, Size);
  MergeCandidatesContents.push_back({});
  MergeCandidatesContents.back().append(FinalFragmentContents.begin(),
                                        FinalFragmentContents.end());

  return Error::success();
}

static Error writeAlignFragment(MCCASBuilder &Builder,
                                const MCAlignFragment &AF, raw_ostream &OS,
                                unsigned FragmentSize) {
  uint64_t Count = FragmentSize / AF.getValueSize();
  if (AF.hasEmitNops()) {
    if (!Builder.Asm.getBackend().writeNopData(OS, Count,
                                               AF.getSubtargetInfo()))
      return createStringError(inconvertibleErrorCode(),
                               "unable to write nop sequence of " +
                                   Twine(Count) + " bytes");
    return Error::success();
  }
  auto Endian = Builder.ObjectWriter.Target.isLittleEndian() ? support::little
                                                             : support::big;
  for (uint64_t I = 0; I != Count; ++I) {
    switch (AF.getValueSize()) {
    default:
      llvm_unreachable("Invalid size!");
    case 1:
      OS << char(AF.getValue());
      break;
    case 2:
      support::endian::write<uint16_t>(OS, AF.getValue(), Endian);
      break;
    case 4:
      support::endian::write<uint32_t>(OS, AF.getValue(), Endian);
      break;
    case 8:
      support::endian::write<uint64_t>(OS, AF.getValue(), Endian);
      break;
    }
  }
  return Error::success();
}

Error MCDataFragmentMerger::emitMergedFragments() {
  if (MergeCandidates.empty())
    return Error::success();

  // Use normal node to store the node.
  if (MergeCandidates.size() == 1) {
    auto E = Builder.buildFragment(*MergeCandidates.front().first,
                                   MergeCandidates.front().second,
                                   MergeCandidatesContents.back());
    reset();
    return E;
  }
  SmallString<8> FragmentData;
  raw_svector_ostream FragmentOS(FragmentData);
  for (const auto &[Candidate, CandidateContents] :
       llvm::zip(MergeCandidates, MergeCandidatesContents)) {
    switch (Candidate.first->getKind()) {
#define MCFRAGMENT_NODE_REF(MCFragmentName, MCEnumName, MCEnumIdentifier)      \
  case MCFragment::MCEnumName: {                                               \
    const MCFragmentName *SF = cast<MCFragmentName>(Candidate.first);          \
    Builder.Asm.writeFragmentPadding(FragmentOS, *SF, Candidate.second);       \
    FragmentData.append(CandidateContents);                                    \
    break;                                                                     \
  }
#define MCFRAGMENT_ENCODED_FRAGMENT_ONLY
#include "llvm/MCCAS/MCCASObjectV1.def"
    case MCFragment::FT_Align: {
      const MCAlignFragment *AF = cast<MCAlignFragment>(Candidate.first);
      if (auto E =
              writeAlignFragment(Builder, *AF, FragmentOS, Candidate.second))
        return E;
      break;
    }
    default:
      llvm_unreachable("other framgents should not be added");
    }
  }

  auto FN = MergedFragmentRef::create(Builder, FragmentData);
  if (!FN)
    return FN.takeError();
  Builder.addNode(*FN);

  // Clear state.
  reset();
  return Error::success();
}

void MCDataFragmentMerger::reset() {
  CurrentSize = 0;
  MergeCandidates.clear();
  MergeCandidatesContents.clear();
}

Error MCCASBuilder::createPaddingRef(const MCSection *Sec) {
  uint64_t Pad = ObjectWriter.getPaddingSize(Sec, Layout);
  auto Fill = PaddingRef::create(*this, Pad);
  if (!Fill)
    return Fill.takeError();
  addNode(*Fill);
  return Error::success();
}

Error MCCASBuilder::createStringSection(
    StringRef S, std::function<Error(StringRef)> CreateFn) {
  assert(S.endswith("\0") && "String sections are null terminated");
  if (DebugInfoUnopt)
    // Drop the null terminator at the end when not splitting the debug_string
    // section as it is always added when materializing.
    return CreateFn(S.drop_back());

  while (!S.empty()) {
    auto SplitSym = S.split('\0');
    if (auto E = CreateFn(SplitSym.first))
      return E;

    S = SplitSym.second;
  }
  return Error::success();
}

/// Reads and returns the length field of a dwarf header contained in Reader,
/// assuming Reader is positioned at the beginning of the header. The Reader's
/// state is advanced to the first byte after the header.
static Expected<size_t> getSizeFromDwarfHeader(BinaryStreamReader &Reader) {
  // From DWARF 5 section 7.4:
  // In the 32-bit DWARF format, an initial length field [...] is an unsigned
  // 4-byte integer (which must be less than 0xfffffff0);
  uint32_t Word1;
  if (auto E = Reader.readInteger(Word1))
    return std::move(E);

  // TODO: handle 64-bit DWARF format.
  if (Word1 >= 0xfffffff0)
    return createStringError(inconvertibleErrorCode(),
                             "DWARF input is not in the 32-bit format");

  return Word1;
}

// TODO: Remove
Expected<size_t>
mccasformats::v1::getSizeFromDwarfHeaderAndSkip(BinaryStreamReader &Reader) {
  Expected<size_t> Size = getSizeFromDwarfHeader(Reader);
  if (!Size)
    return Size.takeError();
  if (auto E = Reader.skip(*Size))
    return std::move(E);
  return Size;
}

/// Returns the Abbreviation Offset field of a Dwarf Compilation Unit (CU)
/// contained in CUData, as well as the total number of bytes taken by the CU.
/// Note: this is different from the length field of the Dwarf header, which
/// does not account for the header size.
static Expected<CUInfo>
getAndSetDebugAbbrevOffsetAndSkip(MutableArrayRef<char> CUData,
                                  support::endianness Endian,
                                  std::optional<uint32_t> NewOffset) {
  BinaryStreamReader Reader(toStringRef(CUData), Endian);
  Expected<size_t> Size = getSizeFromDwarfHeader(Reader);
  if (!Size)
    return Size.takeError();

  size_t AfterSizeOffset = Reader.getOffset();

  // 2-byte Dwarf version identifier.
  uint16_t DwarfVersion;
  if (auto E = Reader.readInteger(DwarfVersion))
    return std::move(E);

  if (DwarfVersion >= 5) {
    // From Dwarf 5 Section 7.5.1.1:
    // Compile Unit Header Format is now changed with unit_type and address_size
    // after the version. Parse both values from the header.
    uint8_t UnitType;
    if (auto E = Reader.readInteger(UnitType))
      return std::move(E);
    if (UnitType != dwarf::DW_UT_compile)
      return createStringError(
          inconvertibleErrorCode(),
          "Unit type is not DW_UT_compile, and is incompatible with MCCAS!");
    uint8_t AddressSize;
    if (auto E = Reader.readInteger(AddressSize))
      return std::move(E);
    if (AddressSize != 8)
      return createStringError(
          inconvertibleErrorCode(),
          "Address size is not 8 bytes, unsupported architecture for MCCAS!");
  }

  // TODO: Handle Dwarf 64 format, which uses 8 bytes.
  size_t AbbrevPosition = Reader.getOffset();
  uint32_t AbbrevOffset;
  if (auto E = Reader.readInteger(AbbrevOffset))
    return std::move(E);

  if (NewOffset.has_value()) {
    // FIXME: safe but ugly cast. Similar to: llvm::arrayRefFromStringRef.
    auto UnsignedData = MutableArrayRef(
        reinterpret_cast<uint8_t *>(CUData.data()), CUData.size());
    BinaryStreamWriter Writer(UnsignedData, Endian);
    Writer.setOffset(AbbrevPosition);
    if (auto E = Writer.writeInteger(*NewOffset))
      return std::move(E);
  }

  Reader.setOffset(AfterSizeOffset);
  if (auto E = Reader.skip(*Size))
    return std::move(E);

  return CUInfo{Reader.getOffset(), AbbrevOffset, DwarfVersion};
}

/// Given a list of MCFragments, return a vector with the concatenation of their
/// data contents.
/// If any fragment is not an MCDataFragment, or the fragment is an
/// MCDwarfLineAddrFragment and the section containing that fragment is not a
/// debug_line section, an error is returned.
Expected<SmallVector<char, 0>> MCCASBuilder::mergeMCFragmentContents(
    const MCSection::FragmentListType &FragmentList, bool IsDebugLineSection) {
  SmallVector<char, 0> mergedData;
  for (const MCFragment &Fragment : FragmentList) {
    if (const auto *DataFragment = dyn_cast<MCDataFragment>(&Fragment))
      llvm::append_range(mergedData, DataFragment->getContents());
    else if (const auto *DwarfLineAddrFrag =
                 dyn_cast<MCDwarfLineAddrFragment>(&Fragment))
      if (IsDebugLineSection)
        llvm::append_range(mergedData, DwarfLineAddrFrag->getContents());
      else
        return createStringError(
            inconvertibleErrorCode(),
            "Invalid MCDwarfLineAddrFragment in a non debug line section");
    else
      return createStringError(inconvertibleErrorCode(),
                               "Invalid fragment type");
  }
  return mergedData;
}

Expected<MCCASBuilder::CUSplit>
MCCASBuilder::splitDebugInfoSectionData(MutableArrayRef<char> DebugInfoData) {
  CUSplit Split;
  // CU splitting loop.
  while (!DebugInfoData.empty()) {
    Expected<CUInfo> Info = getAndSetDebugAbbrevOffsetAndSkip(
        DebugInfoData, Asm.getBackend().Endian, /*NewOffset*/ 0);
    if (!Info)
      return Info.takeError();
    Split.SplitCUData.push_back(DebugInfoData.take_front(Info->CUSize));
    Split.AbbrevOffsets.push_back(Info->AbbrevOffset);
    Split.DwarfVersions.push_back(Info->DwarfVersion);
    DebugInfoData = DebugInfoData.drop_front(Info->CUSize);
  }

  return Split;
}

Error MCCASBuilder::createDebugInfoSection() {
  startSection(DwarfSections.DebugInfo);

  if (DebugInfoUnopt) {
    Expected<SmallVector<char, 0>> DebugInfoData = mergeMCFragmentContents(
        DwarfSections.DebugInfo->getFragmentList(), true);
    if (!DebugInfoData)
      return DebugInfoData.takeError();
    auto DbgInfoUnoptRef =
        DebugInfoUnoptRef::create(*this, toStringRef(*DebugInfoData));
    if (!DbgInfoUnoptRef)
      return DbgInfoUnoptRef.takeError();
    addNode(*DbgInfoUnoptRef);
  } else {
    if (auto E = splitDebugInfoAndAbbrevSections())
      return E;
  }

  if (auto E = createPaddingRef(DwarfSections.DebugInfo))
    return E;
  return finalizeSection<DebugInfoSectionRef>();
}

Error MCCASBuilder::createDebugAbbrevSection() {
  startSection(DwarfSections.Abbrev);
  if (DebugInfoUnopt) {
    Expected<SmallVector<char, 0>> DebugAbbrevData =
        mergeMCFragmentContents(DwarfSections.Abbrev->getFragmentList(), true);
    if (!DebugAbbrevData)
      return DebugAbbrevData.takeError();
    auto DbgAbbrevUnoptRef =
        DebugAbbrevUnoptRef::create(*this, toStringRef(*DebugAbbrevData));
    if (!DbgAbbrevUnoptRef)
      return DbgAbbrevUnoptRef.takeError();
    addNode(*DbgAbbrevUnoptRef);
  }
  if (auto E = createPaddingRef(DwarfSections.Abbrev))
    return E;
  return finalizeSection<DebugAbbrevSectionRef>();
}

/// Helper class to create DIEDataRefs by keeping track of references to
/// children blocks.
struct DIEDataWriter : public DataWriter {
  /// Add `CASObj` to the list of children of the DIE being created.
  void addRef(DIEDataRef CASObj) { Children.push_back(CASObj.getRef()); }

  /// Saves the main data stream and any children to a new DIEDataRef node.
  Expected<DIEDataRef> getCASNode(MCCASBuilder &CASBuilder) {
    return DIEDataRef::create(CASBuilder, Children, Data);
  }

private:
  SmallVector<cas::ObjectRef> Children;
};

/// Helper class to create DIEDistinctDataRefs.
/// These nodes contain raw data that is not expected to deduplicate. This data
/// is described by some DIEAbbrevRef block.
struct DistinctDataWriter : public DataWriter {
  Expected<DIEDistinctDataRef> getCASNode(MCCASBuilder &CASBuilder) {
    return DIEDistinctDataRef::create(CASBuilder, toStringRef(Data));
  }
};

/// Helper class to create DIEAbbrevSetRefs and DIEAbbrevRefs.
/// A DIEAbbrevSetRef has no data, only references to DIEAbbrevRefs.
/// A DIEAbbrevRef has no references, and its data follow the format of a DWARF
/// abbreviation entry exactly.
class AbbrevSetWriter : public AbbrevEntryWriter {
  DenseMap<cas::ObjectRef, unsigned> Children;
  uint32_t PrevSize = 0;

public:
  Expected<unsigned> createAbbrevEntry(DWARFDie DIE, MCCASBuilder &CASBuilder) {
    writeAbbrevEntry(DIE);
    auto MaybeAbbrev = DIEAbbrevRef::create(CASBuilder, toStringRef(Data));
    if (!MaybeAbbrev)
      return MaybeAbbrev.takeError();
    Data.clear();

    // Assign the smallest possible index to the new Abbrev.
    auto [it, inserted] =
        Children.try_emplace(MaybeAbbrev->getRef(), Children.size());
    return it->getSecond();
  }

  /// Creates a DIEAbbrevSetRef with all the DIEAbbrevRefs created so far.
  Expected<DIEAbbrevSetRef> endAbbrevSet(MCCASBuilder &CASBuilder) {
    if (Children.empty())
      return createStringError(inconvertibleErrorCode(),
                               "Abbrev Set cannot be empty");

    // Initialize the vector with copies of an arbitrary element, because we
    // need to assign elements in a random order and ObjectRefs can't be
    // default constructed.
    SmallVector<cas::ObjectRef, 0> ChildrenArray(Children.size() - PrevSize,
                                                 Children.begin()->getFirst());

    // Order the DIEAbbrevRefs based on their creation order.
    for (auto [Obj, Idx] : Children)
      if ((Idx + 1) > PrevSize)
        ChildrenArray[Idx - PrevSize] = Obj;
    auto DIEAbbrevRef = DIEAbbrevSetRef::create(CASBuilder, ChildrenArray);
    ChildrenArray.clear();
    PrevSize = Children.size();
    return DIEAbbrevRef;
  }
};

/// Helper class to convert DIEs into CAS objects.
struct DIEToCASConverter {
  DIEToCASConverter(ArrayRef<char> DebugInfoData, MCCASBuilder &CASBuilder)
      : DebugInfoData(DebugInfoData), CASBuilder(CASBuilder) {}

  /// Converts a DIE into three types of CAS objects:
  /// 1. A tree of DIEDataRefs, containing data expected to deduplicate.
  /// 2. A single DIEAbbrevSetRef, containing all the abbreviations used by the
  /// DIE as DIEAbbrevRef objects.
  /// 3. A single DIEDistinctDataRef, containing data that is not expected to
  /// deduplicate.
  /// These nodes are wrapped into a DIETopLevelRef for convenience.
  Expected<DIETopLevelRef> convert(DWARFDie DIE, ArrayRef<char> HeaderData,
                                   AbbrevSetWriter &AbbrevWriter);

private:
  ArrayRef<char> DebugInfoData;
  MCCASBuilder &CASBuilder;

  Expected<DIEDataRef> convertInNewDIEBlock(DWARFDie DIE,
                                            DistinctDataWriter &DistinctWriter,
                                            AbbrevSetWriter &AbbrevWriter);

  Error convertImpl(DWARFDie &DIE, DIEDataWriter &DIEWriter,
                    DistinctDataWriter &DistinctWriter,
                    AbbrevSetWriter &AbbrevWriter);
};

Error InMemoryCASDWARFObject::partitionCUData(ArrayRef<char> DebugInfoData,
                                              uint64_t AbbrevOffset,
                                              DWARFContext *Ctx,
                                              MCCASBuilder &Builder,
                                              AbbrevSetWriter &AbbrevWriter,
                                              uint16_t DwarfVersion) {
  StringRef AbbrevSectionContribution =
      getAbbrevSection().drop_front(AbbrevOffset);
  DataExtractor Data(AbbrevSectionContribution, isLittleEndian(), 8);
  DWARFDebugAbbrev Abbrev(Data);
  uint64_t OffsetPtr = 0;
  DWARFUnitHeader Header;
  DWARFSection Section = {toStringRef(DebugInfoData), 0 /*Address*/};
  Header.extract(*Ctx, DWARFDataExtractor(*this, Section, isLittleEndian(), 8),
                 &OffsetPtr, DWARFSectionKind::DW_SECT_INFO);

  DWARFUnitVector UV;
  DWARFCompileUnit DCU(*Ctx, Section, Header, &Abbrev, &getRangesSection(),
                       &getLocSection(), getStrSection(),
                       getStrOffsetsSection(), &getAddrSection(),
                       getLocSection(), isLittleEndian(), false, UV);

  DWARFDie CUDie = DCU.getUnitDIE(false);
  assert(CUDie);
  ArrayRef<char> HeaderData;
  if (DwarfVersion >= 5) {
    // Copy 12 bytes which represents the 32-bit DWARF Header for DWARF5.
    if (DebugInfoData.size() < Dwarf5HeaderSize32Bit)
      return createStringError(inconvertibleErrorCode(),
                               "DebugInfoData is too small, it doesn't even "
                               "contain a 32-bit DWARF5 Header");

    HeaderData = DebugInfoData.take_front(Dwarf5HeaderSize32Bit);
  } else {
    // Copy 11 bytes which represents the 32-bit DWARF Header for DWARF4.
    if (DebugInfoData.size() < Dwarf4HeaderSize32Bit)
      return createStringError(inconvertibleErrorCode(),
                               "DebugInfoData is too small, it doesn't even "
                               "contain a 32-bit DWARF4 Header");

    HeaderData = DebugInfoData.take_front(Dwarf4HeaderSize32Bit);
  }
  Expected<DIETopLevelRef> Converted =
      DIEToCASConverter(DebugInfoData, Builder)
          .convert(CUDie, HeaderData, AbbrevWriter);
  if (!Converted)
    return Converted.takeError();
  Builder.addNode(*Converted);
  return Error::success();
}

Error MCCASBuilder::splitDebugInfoAndAbbrevSections() {
  if (!DwarfSections.DebugInfo)
    return Error::success();

  const MCSection::FragmentListType &FragmentList =
      DwarfSections.DebugInfo->getFragmentList();
  Expected<SmallVector<char, 0>> DebugInfoData =
      mergeMCFragmentContents(FragmentList);
  if (!DebugInfoData)
    return DebugInfoData.takeError();

  Expected<CUSplit> SplitInfo = splitDebugInfoSectionData(*DebugInfoData);
  if (!SplitInfo)
    return SplitInfo.takeError();

  const MCSection::FragmentListType &AbbrevFragmentList =
      DwarfSections.Abbrev->getFragmentList();

  Expected<SmallVector<char, 0>> FullAbbrevData =
      mergeMCFragmentContents(AbbrevFragmentList);

  if (!FullAbbrevData)
    return FullAbbrevData.takeError();

  const MCSection::FragmentListType &StringOffsetsFragmentList =
      DwarfSections.StrOffsets->getFragmentList();

  Expected<SmallVector<char, 0>> FullStringOffsetsData =
      mergeMCFragmentContents(StringOffsetsFragmentList);

  if (!FullStringOffsetsData)
    return FullStringOffsetsData.takeError();

  InMemoryCASDWARFObject CASObj(*FullAbbrevData, *FullStringOffsetsData,
                                Asm.getBackend().Endian ==
                                    support::endianness::little);
  auto DWARFObj = std::make_unique<InMemoryCASDWARFObject>(CASObj);
  auto DWARFContextHolder = std::make_unique<DWARFContext>(std::move(DWARFObj));
  auto *DWARFCtx = DWARFContextHolder.get();

  AbbrevSetWriter AbbrevWriter;
  for (auto [CUData, AbbrevOffset, DwarfVersion] :
       llvm::zip(SplitInfo->SplitCUData, SplitInfo->AbbrevOffsets,
                 SplitInfo->DwarfVersions)) {
    if (auto E = CASObj.partitionCUData(CUData, AbbrevOffset, DWARFCtx, *this,
                                        AbbrevWriter, DwarfVersion))
      return E;
  }
  return Error::success();
}

inline void copyData(SmallVector<char, 0> &Data, StringRef DebugLineStrRef,
                     uint64_t &CurrOffset, DWARFDataExtractor::Cursor &Cursor) {
  Data.append(DebugLineStrRef.data() + CurrOffset,
              DebugLineStrRef.data() + Cursor.tell());
  CurrOffset = Cursor.tell();
}

Expected<uint64_t>
MCCASBuilder::createOptimizedLineSection(StringRef DebugLineStrRef) {
  DWARFDataExtractor LineTableDataReader(DebugLineStrRef,
                                         Asm.getBackend().Endian, 8);
  auto Prologue = parseLineTableHeaderAndSkip(LineTableDataReader);
  if (!Prologue)
    return Prologue.takeError();
  SmallVector<char, 0> DistinctData;
  // Copy line table prologue into the DistinctData buffer.
  DistinctData.append(DebugLineStrRef.data(),
                      DebugLineStrRef.data() + Prologue->Offset);
  SmallVector<DebugLineRef, 0> LineTableVector;
  SmallVector<char, 0> LineTableData;
  uint64_t *OffsetPtr = &Prologue->Offset;
  uint64_t End =
      Prologue->Length + (Prologue->Format == dwarf::DWARF32 ? 4 : 8);
  while (*OffsetPtr < End) {
    DWARFDataExtractor::Cursor Cursor(*OffsetPtr);
    auto Opcode = LineTableDataReader.getU8(Cursor);
    LineTableData.push_back(Opcode);
    if (Opcode == 0) {
      // Extended Opcodes always start with a zero opcode followed by
      // a uleb128 length so unknown opcodes can be skipped.
      uint64_t CurrOffset = Cursor.tell();
      uint64_t Len = LineTableDataReader.getULEB128(Cursor);
      if (Len == 0)
        return createStringError(inconvertibleErrorCode(),
                                 "0 Length for an extended opcode is wrong");
      copyData(LineTableData, DebugLineStrRef, CurrOffset, Cursor);

      uint8_t SubOpcode = LineTableDataReader.getU8(Cursor);
      LineTableData.push_back(SubOpcode);
      bool IsEndSequence = false;
      bool IsRelocation = false;
      CurrOffset = Cursor.tell();
      auto Err = handleExtendedOpcodesForLineTable(LineTableDataReader, Cursor,
                                                   SubOpcode, Len,
                                                   IsEndSequence, IsRelocation);
      if (Err)
        return std::move(Err);
      if (IsRelocation) {
        // Move cursor to end of relocation and copy.
        LineTableDataReader.getRelocatedAddress(Cursor);
        copyData(DistinctData, DebugLineStrRef, CurrOffset, Cursor);
      } else
        copyData(LineTableData, DebugLineStrRef, CurrOffset, Cursor);

      if (IsEndSequence) {
        // The current Opcode is a DW_LNE_end_sequence. It takes no operand,
        // create a cas block here.
        auto LineTable =
            DebugLineRef::create(*this, toStringRef(LineTableData));
        if (!LineTable)
          return LineTable.takeError();
        LineTableVector.push_back(*LineTable);
        LineTableData.clear();
      }
    } else if (Opcode < Prologue->OpcodeBase) {
      bool IsSetFile = false;
      bool IsRelocation = false;
      uint64_t CurrOffset = Cursor.tell();
      auto Err = handleStandardOpcodesForLineTable(
          LineTableDataReader, Cursor, Opcode, IsSetFile, IsRelocation);
      if (Err)
        return std::move(Err);
      if (IsRelocation) {
        // Move cursor to end of relocation and copy.
        LineTableDataReader.getRelocatedValue(Cursor, 2);
        copyData(DistinctData, DebugLineStrRef, CurrOffset, Cursor);
      } else
        copyData(LineTableData, DebugLineStrRef, CurrOffset, Cursor);

      if (IsSetFile) {
        // The current Opcode is a DW_LNS_set_file. Store file numbers in the
        // distinct data.
        uint64_t CurrOffset = Cursor.tell();
        LineTableDataReader.getULEB128(Cursor);
        // The file number doesn't dedupe, so store that in the DistinctData.
        copyData(DistinctData, DebugLineStrRef, CurrOffset, Cursor);
      }
    } else {
      // Special Opcodes. Do nothing, move on.
    }
    if (!Cursor)
      return Cursor.takeError();
    *OffsetPtr = Cursor.tell();
  }

  // Create DistinctDebugLineRef.
  auto DistinctRef =
      DistinctDebugLineRef::create(*this, toStringRef(DistinctData));
  if (!DistinctRef)
    return DistinctRef.takeError();

  // Add Nodes in order. DistinctDebugLineRef first, then the DebugLineRefs,
  // then a PaddingRef if needed.
  addNode(*DistinctRef);
  for (auto &Node : LineTableVector)
    addNode(Node);
  return *OffsetPtr;
}

Error MCCASBuilder::createLineSection() {
  if (!DwarfSections.Line)
    return Error::success();

  Expected<SmallVector<char, 0>> DebugLineData =
      mergeMCFragmentContents(DwarfSections.Line->getFragmentList(), true);
  if (!DebugLineData)
    return DebugLineData.takeError();

  startSection(DwarfSections.Line);

  if (DebugInfoUnopt) {
    auto DbgLineUnoptRef =
        DebugLineUnoptRef::create(*this, toStringRef(*DebugLineData));
    if (!DbgLineUnoptRef)
      return DbgLineUnoptRef.takeError();
    addNode(*DbgLineUnoptRef);
  } else {
    StringRef DebugLineStrRef(DebugLineData->data(), DebugLineData->size());
    while (DebugLineStrRef.size()) {
      auto BytesWritten = createOptimizedLineSection(DebugLineStrRef);
      if (!BytesWritten)
        return BytesWritten.takeError();
      DebugLineStrRef = DebugLineStrRef.drop_front(*BytesWritten);
    }
  }

  if (auto E = createPaddingRef(DwarfSections.Line))
    return E;
  return finalizeSection<DebugLineSectionRef>();
}

Error MCCASBuilder::createDebugStrSection() {

  auto DebugStringRefs = createDebugStringRefs();
  if (!DebugStringRefs)
    return DebugStringRefs.takeError();

  startSection(DwarfSections.Str);
  for (auto DebugStringRef : *DebugStringRefs)
    addNode(DebugStringRef);
  if (auto E = createPaddingRef(DwarfSections.Str))
    return E;
  return finalizeSection<DebugStringSectionRef>();
}

Expected<SmallVector<DebugStrRef, 0>> MCCASBuilder::createDebugStringRefs() {
  if (!DwarfSections.Str || !DwarfSections.Str->getFragmentList().size())
    return SmallVector<DebugStrRef, 0>();

  assert(DwarfSections.Str->getFragmentList().size() == 1 &&
         "One fragment in debug str section");

  SmallVector<DebugStrRef, 0> DebugStringRefs;
  ArrayRef<char> DebugStrData =
      cast<MCDataFragment>(*DwarfSections.Str->begin()).getContents();
  StringRef S(DebugStrData.data(), DebugStrData.size());
  if (auto E = createStringSection(S, [&](StringRef S) -> Error {
        auto Sym = DebugStrRef::create(*this, S);
        if (!Sym)
          return Sym.takeError();
        DebugStringRefs.push_back(*Sym);
        return Error::success();
      }))
    return std::move(E);
  return DebugStringRefs;
}

static ArrayRef<char> getFragmentContents(const MCFragment &Fragment) {
  switch (Fragment.getKind()) {
#define MCFRAGMENT_NODE_REF(MCFragmentName, MCEnumName, MCEnumIdentifier)      \
  case MCFragment::MCEnumName: {                                               \
    const MCFragmentName &SF = cast<MCFragmentName>(Fragment);                 \
    return SF.getContents();                                                   \
  }
#define MCFRAGMENT_ENCODED_FRAGMENT_ONLY
#include "llvm/MCCAS/MCCASObjectV1.def"
  case MCFragment::FT_CVInlineLines: {
    const MCCVInlineLineTableFragment &SF =
        cast<MCCVInlineLineTableFragment>(Fragment);
    return SF.getContents();
  }
  case MCFragment::FT_LEB: {
    const MCLEBFragment &SF = cast<MCLEBFragment>(Fragment);
    return SF.getContents();
  }
  default:
    return ArrayRef<char>();
  }
}

static unsigned getRelocationSize(const MachO::any_relocation_info &RE,
                                  bool IsLittleEndian) {
  if (IsLittleEndian)
    return 1 << ((RE.r_word1 >> 25) & 3);
  return 1 << ((RE.r_word1 >> 5) & 3);
}

static uint32_t getRelocationOffset(const MachO::any_relocation_info &RE) {
  return RE.r_word0;
}

/// This helper function is used to partition a Fragment into 2 parts, the \p
/// FinalFragmentContents will contain the contents of the Fragment that will
/// deduplicate and will be stored as part of a *FragmentRef or a
/// MergedFragmentRef, the \p Addends stores all the relocation
/// addends that will not deduplicate in the CAS and must be stored separately,
/// there is one AddendRef per section in the CAS. Depending on the situation,
/// the \p RelocationBuffer may have the relocations for the entire section,
/// thereforem there may not be a way to know whether a relocation belongs to
/// the current fragment being processed. The \p RelocationBufferIndex is an out
/// parameter and is the index into the \p RelocationBuffer, which is used to
/// access the current relocation that has to be partitioned.
static void
partitionFragment(const MCAsmLayout &Layout, SmallVector<char, 0> &Addends,
                  SmallVector<char, 0> &FinalFragmentContents,
                  ArrayRef<MachO::any_relocation_info> RelocationBuffer,
                  const MCFragment &Fragment, uint64_t &RelocationBufferIndex,
                  bool IsLittleEndian) {
  auto FragmentContents = getFragmentContents(Fragment);
  /// FragmentIndex: It denotes the index into the FragmentContents that is used
  /// to copy the data that deduplicates in the \p FinalFragmentContents.
  uint64_t FragmentIndex = 0;
  /// PrevOffset: A relocation can sometimes be divided into multiple parts, all
  /// of them share the same offset, but describe only one Addend, if we already
  /// copied the addend out of the FragmentContents at a particular offset, we
  /// should skip all relocations that matches the same offset.
  int64_t PrevOffset = -1;
  // Relocations are stored in the Section.
  for (; RelocationBufferIndex < RelocationBuffer.size();
       RelocationBufferIndex++) {
    auto Reloc = RelocationBuffer[RelocationBufferIndex];
    uint32_t RelocOffset = getRelocationOffset(Reloc);
    if (PrevOffset == RelocOffset)
      continue;
    uint64_t FragmentOffset = Layout.getFragmentOffset(&Fragment);
    if (RelocOffset < FragmentOffset + FragmentContents.size()) {
      /// RelocOffsetInFragment: This is used to denote the offset of the
      /// relocation in the current Fragment. Relocation offsets are always from
      /// the start of the section, we need to normalize this value to start at
      /// the start of the Fragment instead.
      uint32_t RelocOffsetInFragment = RelocOffset - FragmentOffset;
      // Copy contents of fragment upto relocation addend data in the
      // FinalFragmentContents.
      FinalFragmentContents.append(FragmentContents.data() + FragmentIndex,
                                   FragmentContents.data() +
                                       RelocOffsetInFragment);
      // Addend belongs to the current fragment, copy the addend into the
      // Addend vector.
      Addends.append(FragmentContents.data() + RelocOffsetInFragment,
                     FragmentContents.data() + RelocOffsetInFragment +
                         getRelocationSize(Reloc, IsLittleEndian));
      FragmentIndex =
          RelocOffsetInFragment + getRelocationSize(Reloc, IsLittleEndian);
      PrevOffset = RelocOffset;
    } else
      break;
  }
  // Copy any leftover fragment contents into the FinalFragmentContents.
  FinalFragmentContents.append(FragmentContents.data() + FragmentIndex,
                               FragmentContents.data() +
                                   FragmentContents.size());
}

Error MCCASBuilder::buildFragments() {
  startGroup();

  for (const MCSection &Sec : Asm) {
    if (Sec.isVirtualSection() || Sec.getFragmentList().empty())
      continue;

    // Handle Debug Info sections separately.
    if (&Sec == DwarfSections.DebugInfo) {
      if (auto E = createDebugInfoSection())
        return E;
      continue;
    }

    // Handle Debug Abbrev sections separately.
    if (&Sec == DwarfSections.Abbrev) {
      if (auto E = createDebugAbbrevSection())
        return E;
      continue;
    }

    // Handle Debug Line sections separately.
    if (&Sec == DwarfSections.Line) {
      if (auto E = createLineSection())
        return E;
      continue;
    }

    // Handle Debug Str sections separately.
    if (&Sec == DwarfSections.Str) {
      if (auto E = createDebugStrSection())
        return E;
      continue;
    }

    // Start Subsection for one section.
    startSection(&Sec);

    // Start subsection for first Atom.
    startAtom(Sec.getFragmentList().front().getAtom());

    SmallVector<char, 0> Addends;
    ArrayRef<MachO::any_relocation_info> RelocationBuffer;
    MCDataFragmentMerger Merger(*this, &Sec);
    uint64_t RelocationBufferIndex = 0;
    for (const MCFragment &F : Sec) {
      auto Relocs = RelMap.find(&F);
      if (RelocLocation == Atom) {
        if (Relocs != RelMap.end()) {
          AtomRelocs.append(Relocs->second.begin(), Relocs->second.end());
          // Reset RelocationBufferIndex if Relocations exist per Fragment. This
          // is done because if the relocations are stored in the RelMap, and
          // not with the SectionContents, we need to reset the
          // RelocationBufferIndex to start at the beginning of the Fragments
          // relocation buffer.
          RelocationBufferIndex = 0;
        }
      }

      // Relocations are in the RelMap or in the SectionRelocs. If they are in
      // the RelMap, they are accessible per fragment. Choose the correct buffer
      // where they are stored. Initialize RelocationBuffer to an empty
      // ArrayRef, if no Relocations are present.
      if (Relocs != RelMap.end())
        RelocationBuffer = Relocs->second;
      else if (SectionRelocs.empty())
        RelocationBuffer = ArrayRef<MachO::any_relocation_info>();
      else
        RelocationBuffer = SectionRelocs;

      auto Size = Asm.computeFragmentSize(Layout, F);
      // Don't need to encode the fragment if it doesn't contribute anything.
      if (!Size)
        continue;

      SmallVector<char, 0> FinalFragmentContents;
      partitionFragment(Layout, Addends, FinalFragmentContents,
                        RelocationBuffer, F, RelocationBufferIndex,
                        ObjectWriter.Target.isLittleEndian());

      if (auto E = Merger.tryMerge(F, Size, FinalFragmentContents))
        return E;
    }
    if (auto E = Merger.flush())
      return E;

    // End last subsection for late Atom.
    if (auto E = finalizeAtom())
      return E;

    if (auto E = createPaddingRef(&Sec))
      return E;

    // Do not create an AddendsRef if there were no relocations in this section.
    if (!Addends.empty()) {
      auto AddendRef = AddendsRef::create(*this, toStringRef(Addends));
      if (!AddendRef)
        return AddendRef.takeError();
      addNode(*AddendRef);
    }

    if (auto E = finalizeSection())
      return E;
  }
  return finalizeGroup();
}

Error MCCASBuilder::buildRelocations() {
  ObjectWriter.resetBuffer();
  if (ObjectWriter.Mode == CASBackendMode::Verify ||
      RelocLocation == CompileUnit)
    ObjectWriter.writeRelocations(Asm, Layout);

  if (RelocLocation == CompileUnit) {
    auto Relocs = RelocationsRef::create(*this, ObjectWriter.getContent());
    if (!Relocs)
      return Relocs.takeError();

    addNode(*Relocs);
  }

  return Error::success();
}

Error MCCASBuilder::buildDataInCodeRegion() {
  ObjectWriter.resetBuffer();
  ObjectWriter.writeDataInCodeRegion(Asm, Layout);
  auto Data = DataInCodeRef::create(*this, ObjectWriter.getContent());
  if (!Data)
    return Data.takeError();

  addNode(*Data);
  return Error::success();
}

Error MCCASBuilder::buildSymbolTable() {
  ObjectWriter.resetBuffer();
  ObjectWriter.writeSymbolTable(Asm, Layout);
  StringRef S = ObjectWriter.getContent();
  std::vector<cas::ObjectRef> CStrings;
  if (auto E = createStringSection(S, [&](StringRef S) -> Error {
        auto Sym = CStringRef::create(*this, S);
        if (!Sym)
          return Sym.takeError();
        CStrings.push_back(Sym->getRef());
        return Error::success();
      }))
    return E;

  auto Ref = SymbolTableRef::create(*this, CStrings);
  if (!Ref)
    return Ref.takeError();
  addNode(*Ref);

  return Error::success();
}

void MCCASBuilder::startGroup() {
  assert(GroupContext.empty() && "GroupContext is not empty");
  CurrentContext = &GroupContext;
}

Error MCCASBuilder::finalizeGroup() {
  auto Ref = GroupRef::create(*this, GroupContext);
  if (!Ref)
    return Ref.takeError();
  GroupContext.clear();
  CurrentContext = &Sections;
  addNode(*Ref);
  return Error::success();
}

void MCCASBuilder::startSection(const MCSection *Sec) {
  assert(SectionContext.empty() && !CurrentSection && RelMap.empty() &&
         SectionRelocs.empty() && "SectionContext is not empty");

  CurrentSection = Sec;
  CurrentContext = &SectionContext;

  if (RelocLocation == Atom) {
    // Build a map for lookup.
    for (auto R : ObjectWriter.getRelocations()[Sec]) {
      // For the Dwarf Sections, just append the relocations to the
      // SectionRelocs. No Atoms are considered for this section.
      if (R.F && Sec != DwarfSections.Line && Sec != DwarfSections.DebugInfo &&
          Sec != DwarfSections.Abbrev)
        RelMap[R.F].push_back(R.MRE);
      else
        // If the fragment is nullptr, it should a section with only relocation
        // in section. Encode in section.
        // DebugInfo sections are also encoded in a single section.
        SectionRelocs.push_back(R.MRE);
    }
  }

  if (RelocLocation == Section) {
    for (auto R : ObjectWriter.getRelocations()[Sec])
      SectionRelocs.push_back(R.MRE);
  }
}

template <typename SectionRefTy>
Error MCCASBuilder::finalizeSection() {
  auto Ref = SectionRefTy::create(*this, SectionContext);
  if (!Ref)
    return Ref.takeError();

  SectionContext.clear();
  SectionRelocs.clear();
  RelMap.clear();
  CurrentSection = nullptr;
  CurrentContext = &GroupContext;
  addNode(*Ref);

  return Error::success();
}

void MCCASBuilder::startAtom(const MCSymbol *Atom) {
  assert(AtomContext.empty() && AtomRelocs.empty() && !CurrentAtom &&
         "AtomContext is not empty");

  CurrentAtom = Atom;
  CurrentContext = &AtomContext;
}

Error MCCASBuilder::finalizeAtom() {
  auto Ref = AtomRef::create(*this, AtomContext);
  if (!Ref)
    return Ref.takeError();

  AtomContext.clear();
  AtomRelocs.clear();
  CurrentAtom = nullptr;
  CurrentContext = &SectionContext;
  addNode(*Ref);

  return Error::success();
}

void MCCASBuilder::addNode(cas::ObjectProxy Node) {
  CurrentContext->push_back(Node.getRef());
}

Expected<MCAssemblerRef> MCAssemblerRef::create(const MCSchema &Schema,
                                                MachOCASWriter &ObjectWriter,
                                                MCAssembler &Asm,
                                                const MCAsmLayout &Layout,
                                                raw_ostream *DebugOS) {
  MCCASBuilder Builder(Schema, ObjectWriter, Asm, Layout, DebugOS);

  if (auto E = Builder.prepare())
    return std::move(E);

  if (auto E = Builder.buildMachOHeader())
    return std::move(E);

  if (auto E = Builder.buildFragments())
    return std::move(E);

  // Only need to do this for verify mode so we compare the output byte by
  // byte.
  if (ObjectWriter.Mode == CASBackendMode::Verify) {
    ObjectWriter.writeSectionData(Asm, Layout);
  }

  if (auto E = Builder.buildRelocations())
    return std::move(E);

  if (auto E = Builder.buildDataInCodeRegion())
    return std::move(E);

  if (auto E = Builder.buildSymbolTable())
    return std::move(E);

  auto B = Builder::startRootNode(Schema, KindString);
  if (!B)
    return B.takeError();

  // Put Header, Relocations, SymbolTable, etc. in the front.
  B->Refs.append(Builder.Sections.begin(), Builder.Sections.end());

  std::string NormalizedTriple = ObjectWriter.Target.normalize();
  writeVBR8(uint32_t(NormalizedTriple.size()), B->Data);
  B->Data.append(NormalizedTriple);

  return get(B->build());
}

template <typename T>
static Expected<T> findSectionFromAsm(const MCAssemblerRef &Asm) {
  for (unsigned I = 1; I < Asm.getNumReferences(); ++I) {
    auto Node = MCObjectProxy::get(
        Asm.getSchema(), Asm.getSchema().CAS.getProxy(Asm.getReference(I)));
    if (!Node)
      return Node.takeError();
    if (auto Ref = T::Cast(*Node))
      return *Ref;
  }

  return createStringError(inconvertibleErrorCode(),
                           "cannot locate the requested section");
}

template <typename T>
static Expected<uint64_t> materializeData(raw_ostream &OS,
                                          const MCAssemblerRef &Asm) {
  auto Node = findSectionFromAsm<T>(Asm);
  if (!Node)
    return Node.takeError();

  return Node->materialize(OS);
}

Error MCAssemblerRef::materialize(raw_ostream &OS) const {
  // Read the triple first.
  StringRef Remaining = getData();
  uint32_t NormalizedTripleSize;
  if (auto E = consumeVBR8(Remaining, NormalizedTripleSize))
    return E;
  auto TripleStr = consumeDataOfSize(Remaining, NormalizedTripleSize);
  if (!TripleStr)
    return TripleStr.takeError();
  Triple Target(*TripleStr);

  MCCASReader Reader(OS, Target, getSchema());
  uint64_t Written = 0;
  // MachOHeader.
  auto HeaderSize = materializeData<HeaderRef>(OS, *this);
  if (!HeaderSize)
    return HeaderSize.takeError();
  Written += *HeaderSize;

  // SectionData.
  auto SectionDataRef = findSectionFromAsm<GroupRef>(*this);
  if (!SectionDataRef)
    return SectionDataRef.takeError();
  auto SectionDataSize = SectionDataRef->materialize(Reader);
  if (!SectionDataSize)
    return SectionDataSize.takeError();
  Written += *SectionDataSize;

  // Add padding to pointer size.
  auto SectionDataPad =
      offsetToAlignment(Written, Target.isArch64Bit() ? Align(8) : Align(4));
  OS.write_zeros(SectionDataPad);

  for (auto &Sec : Reader.Relocations) {
    for (auto &Entry : llvm::reverse(Sec)) {
      support::endian::write<uint32_t>(OS, Entry.r_word0, Reader.getEndian());
      support::endian::write<uint32_t>(OS, Entry.r_word1, Reader.getEndian());
    }
  }

  if (auto Relocations = findSectionFromAsm<RelocationsRef>(*this)) {
    auto RelocSize = Relocations->materialize(OS);
    if (!RelocSize)
      return RelocSize.takeError();
  } else
    consumeError(Relocations.takeError()); // Relocations can be missing.

  auto DCOrErr = materializeData<DataInCodeRef>(OS, *this);
  if (!DCOrErr)
    return DCOrErr.takeError();

  auto SymTableRef = findSectionFromAsm<SymbolTableRef>(*this);
  if (!SymTableRef)
    return SymTableRef.takeError();
  auto SymbolTableSize = SymTableRef->materialize(Reader);
  if (!SymbolTableSize)
    return SymbolTableSize.takeError();

  return Error::success();
}

MCCASReader::MCCASReader(raw_ostream &OS, const Triple &Target,
                         const MCSchema &Schema)
    : OS(OS), Target(Target), Schema(Schema) {}

Expected<uint64_t> MCCASReader::materializeGroup(cas::ObjectRef ID) {
  auto Node = MCObjectProxy::get(Schema, Schema.CAS.getProxy(ID));
  if (!Node)
    return Node.takeError();

  // Group can have sections, symbol table strs.
  if (auto F = SectionRef::Cast(*Node))
    return F->materialize(*this);
  if (auto F = DebugStringSectionRef::Cast(*Node))
    return F->materialize(*this);
  if (auto F = CStringRef::Cast(*Node)) {
    auto Size = F->materialize(OS);
    if (!Size)
      return Size.takeError();
    // Write null between strings.
    OS.write_zeros(1);
    return *Size + 1;
  }
  if (auto F = DebugInfoSectionRef::Cast(*Node))
    return materializeDebugInfoFromTagImpl(*this, *F);
  if (auto F = DebugLineSectionRef::Cast(*Node))
    return F->materialize(*this);
  return createStringError(inconvertibleErrorCode(),
                           "unsupported CAS node for group");
}

Expected<uint64_t>
MCCASReader::materializeDebugAbbrevUnopt(ArrayRef<cas::ObjectRef> Refs) {

  SmallVector<char, 0> DebugAbbrevSection;
  bool DebugAbbrevUnoptRefSeen = false;
  for (auto Ref : Refs) {
    auto Obj = getObjectProxy(Ref);
    if (!Obj)
      return Obj.takeError();
    if (auto F = DebugAbbrevUnoptRef::Cast(*Obj)) {
      DebugAbbrevUnoptRefSeen = true;
      append_range(DebugAbbrevSection, F->getData());
      continue;
    }
    if (auto F = PaddingRef::Cast(*Obj)) {
      if (!DebugAbbrevUnoptRefSeen)
        return 0;
      raw_svector_ostream OS(DebugAbbrevSection);
      auto Size = F->materialize(OS);
      if (!Size)
        return Size.takeError();
      continue;
    }
    llvm_unreachable("Incorrect CAS Object in DebugAbbrevSection");
  }
  OS << DebugAbbrevSection;
  return DebugAbbrevSection.size();
}

Expected<uint64_t> MCCASReader::materializeSection(cas::ObjectRef ID,
                                                   raw_ostream *Stream) {
  auto Node = MCObjectProxy::get(Schema, Schema.CAS.getProxy(ID));
  if (!Node)
    return Node.takeError();

  // Section can have atoms, padding, debug_strs.
  if (auto F = AtomRef::Cast(*Node))
    return F->materialize(*this, Stream);
  if (auto F = PaddingRef::Cast(*Node))
    return F->materialize(*Stream);
  if (auto F = DebugStrRef::Cast(*Node)) {
    auto Size = F->materialize(*Stream);
    if (!Size)
      return Size.takeError();
    // Write null between strings.
    Stream->write_zeros(1);
    return *Size + 1;
  }
  if (auto F = AddendsRef::Cast(*Node))
    // AddendsRef is already handled when materializing Atoms, skip.
    return 0;
  return createStringError(inconvertibleErrorCode(),
                           "unsupported CAS node for atom");
}

/// This helper function reconstructs the current Section as it was before the
/// CAS was created. In the CAS, each section is split into multiple
/// *FragmentRef or MergedFragmentRef blocks, and an AddendsRef block that
/// contains the relocation addends for that section. The \p SectionBuffer is an
/// out parameter that will contain the contents of the entire section after it
/// is properly reconstructed. The \p FragmentBuffer contains the contents of
/// all the Fragments for that section.
Expected<uint64_t>
MCCASReader::reconstructSection(SmallVectorImpl<char> &SectionBuffer,
                                ArrayRef<char> FragmentBuffer) {
  /// FragmentIndex: It denotes the index into the \p FragmentBuffer that is
  /// used to copy the Fragment contents into the \p SectionBuffer.
  uint64_t FragmentIndex = 0;
  /// PrevOffset: A relocation can sometimes be divided into multiple parts, all
  /// of them share the same offset, but describe only one Addend, if we already
  /// copied the addend out of the Addends at a particular offset, we should
  /// skip all relocations that matches the same offset.
  int64_t PrevOffset = -1;
  for (auto Reloc : Relocations.back()) {
    auto RelocationOffsetInSection = getRelocationOffset(Reloc);
    if (PrevOffset == RelocationOffsetInSection)
      continue;
    auto RelocationSize =
        getRelocationSize(Reloc, getEndian() == support::little);
    /// NumOfBytesToReloc: This denotes the number of bytes needed to be copied
    /// into the \p SectionBuffer before we copy the next addend.
    auto NumOfBytesToReloc = RelocationOffsetInSection - SectionBuffer.size();
    // Copy the contents of the fragment till the next relocation.
    SectionBuffer.append(FragmentBuffer.begin() + FragmentIndex,
                         FragmentBuffer.begin() + FragmentIndex +
                             NumOfBytesToReloc);
    FragmentIndex += NumOfBytesToReloc;
    // Copy the relocation addend.
    SectionBuffer.append(Addends.begin() + AddendBufferIndex,
                         Addends.begin() + AddendBufferIndex + RelocationSize);
    AddendBufferIndex += RelocationSize;
    PrevOffset = RelocationOffsetInSection;
  }
  // Copy any remaining bytes of the fragment into the SectionBuffer.
  SectionBuffer.append(FragmentBuffer.begin() + FragmentIndex,
                       FragmentBuffer.end());
  assert(AddendBufferIndex == Addends.size() &&
         "All addends for section not copied into final section buffer");
  return AddendBufferIndex;
}

Expected<uint64_t> MCCASReader::materializeAtom(cas::ObjectRef ID,
                                                raw_ostream *Stream) {
  auto Node = MCObjectProxy::get(Schema, Schema.CAS.getProxy(ID));
  if (!Node)
    return Node.takeError();

#define MCFRAGMENT_NODE_REF(MCFragmentName, MCEnumName, MCEnumIdentifier)      \
  if (auto F = MCFragmentName##Ref::Cast(*Node))                               \
    return F->materialize(*this, Stream);
#include "llvm/MCCAS/MCCASObjectV1.def"
  if (auto F = PaddingRef::Cast(*Node))
    return F->materialize(*Stream);
  if (auto F = MergedFragmentRef::Cast(*Node))
    return F->materialize(*Stream);

  return createStringError(inconvertibleErrorCode(),
                           "unsupported CAS node for fragment");
}

/// If the AddendsRef exists in the current section, copy its contents into the
/// Addends buffer.
Error MCCASReader::checkIfAddendRefExistsAndCopy(
    ArrayRef<cas::ObjectRef> Refs) {
  auto ID = Refs.back();
  auto Node = getObjectProxy(ID);
  if (!Node)
    return Node.takeError();

  if (auto F = AddendsRef::Cast(*Node))
    Addends.append(F->getData().begin(), F->getData().end());
  return Error::success();
}

Expected<DIEAbbrevSetRef>
DIEAbbrevSetRef::create(MCCASBuilder &MB, ArrayRef<cas::ObjectRef> Abbrevs) {
  Expected<Builder> B = Builder::startNode(MB.Schema, KindString);
  if (!B)
    return B.takeError();
  append_range(B->Refs, Abbrevs);
  return get(B->build());
}

Expected<DIETopLevelRef>
DIETopLevelRef::create(MCCASBuilder &MB, ArrayRef<cas::ObjectRef> Children) {
  Expected<Builder> B = Builder::startNode(MB.Schema, KindString);
  if (!B)
    return B.takeError();
  append_range(B->Refs, Children);
  return get(B->build());
}

Expected<DIEDataRef> DIEDataRef::create(MCCASBuilder &MB,
                                        ArrayRef<cas::ObjectRef> Children,
                                        ArrayRef<char> Contents) {
  Expected<Builder> B = Builder::startNode(MB.Schema, KindString);
  if (!B)
    return B.takeError();
  append_range(B->Refs, Children);
  append_range(B->Data, toStringRef(Contents));
  return get(B->build());
}

// Returns true if DIE should be placed in a separate CAS block.
static bool shouldCreateSeparateBlockFor(DWARFDie &DIE) {
  dwarf::Tag Tag = DIE.getTag();

  if (Tag == dwarf::Tag::DW_TAG_subprogram) {
    // Only split on subprogram definitions, so look for low_pc.
    for (const DWARFAttribute &AttrValue : DIE.attributes())
      if (AttrValue.Attr == dwarf::Attribute::DW_AT_low_pc)
        return true;
    return false;
  }

  if (Tag == dwarf::Tag::DW_TAG_enumeration_type)
    return true;

  if (Tag == dwarf::Tag::DW_TAG_structure_type ||
      Tag == dwarf::Tag::DW_TAG_class_type) {
    // Don't split on declarations, as these are short
    for (const DWARFAttribute &AttrValue : DIE.attributes())
      if (AttrValue.Attr == dwarf::Attribute::DW_AT_declaration)
        return false;
    return true;
  }

  return false;
}

static void writeDIEAttrs(DWARFDie &DIE, ArrayRef<char> DebugInfoData,
                          DIEDataWriter &DIEWriter,
                          DistinctDataWriter &DistinctWriter) {
  for (const DWARFAttribute &AttrValue : DIE.attributes()) {
    dwarf::Attribute Attr = AttrValue.Attr;
    dwarf::Form Form = AttrValue.Value.getForm();
    ArrayRef<char> FormData =
        DebugInfoData.slice(AttrValue.Offset, AttrValue.ByteSize);
    auto &WriterToUse = doesntDedup(Form, Attr)
                            ? static_cast<DataWriter &>(DistinctWriter)
                            : DIEWriter;
    if (Form == dwarf::Form::DW_FORM_ref4 || Form == dwarf::Form::DW_FORM_strp)
      convertFourByteFormDataToULEB(FormData, WriterToUse);
    else
      WriterToUse.writeData(FormData);
  }
}

/// Creates an abbreviation for DIE using AbbrevWriter.
/// Stores the contents of the DIE using DistinctWriter and DIEWriter following
/// the format:
///   [abbreviation_index, raw_data?]+
/// * abbreviation_index is always written with DistinctWriter.
/// * raw_data, when present, may be written to either DistinctWriter or
/// DIEWriter, depending on deduplication choices made.
/// * If abbreviation_index is getEndOfDIESiblingsMarker(), raw_data is empty and
/// this denotes the end of a sequence of sibling DIEs.
/// * If abbreviation_index is getDIEInAnotherBlockMarker(), raw_data is empty
/// and this denotes a DIE placed in a child CAS block.
/// * Otherwise, abbreviation_index is an index into the list of references of a
/// DIEAbbrevSetRef block. In this case, raw_data should be interpreted
/// according to the corresponding DIEAbbrevRefs block.
Error DIEToCASConverter::convertImpl(DWARFDie &DIE, DIEDataWriter &DIEWriter,
                                     DistinctDataWriter &DistinctWriter,
                                     AbbrevSetWriter &AbbrevWriter) {
  Expected<unsigned> MaybeAbbrevIndex =
      AbbrevWriter.createAbbrevEntry(DIE, CASBuilder);
  if (!MaybeAbbrevIndex)
    return MaybeAbbrevIndex.takeError();

  DistinctWriter.writeULEB128(encodeAbbrevIndex(*MaybeAbbrevIndex));
  writeDIEAttrs(DIE, DebugInfoData, DIEWriter, DistinctWriter);

  for (DWARFDie Child = DIE.getFirstChild(); Child;
       Child = Child.getSibling()) {
    dwarf::Tag ChildTag = Child.getTag();
    if (ChildTag == dwarf::Tag::DW_TAG_null) {
      DistinctWriter.writeULEB128(getEndOfDIESiblingsMarker());
      break;
    }

    // FIXME: don't use recursion.
    if (shouldCreateSeparateBlockFor(Child)) {
      DistinctWriter.writeULEB128(getDIEInAnotherBlockMarker());
      auto MaybeNode =
          convertInNewDIEBlock(Child, DistinctWriter, AbbrevWriter);
      if (!MaybeNode)
        return MaybeNode.takeError();
      DIEWriter.addRef(*MaybeNode);
      continue;
    }

    if (auto E = convertImpl(Child, DIEWriter, DistinctWriter, AbbrevWriter))
      return E;
  }
  return Error::success();
}

Expected<DIEDataRef>
DIEToCASConverter::convertInNewDIEBlock(DWARFDie DIE,
                                        DistinctDataWriter &DistinctWriter,
                                        AbbrevSetWriter &AbbrevWriter) {
  DIEDataWriter DIEWriter;
  if (auto E = convertImpl(DIE, DIEWriter, DistinctWriter, AbbrevWriter))
    return std::move(E);
  return DIEWriter.getCASNode(CASBuilder);
}

Expected<DIETopLevelRef>
DIEToCASConverter::convert(DWARFDie DIE, ArrayRef<char> HeaderData,
                           AbbrevSetWriter &AbbrevWriter) {
  DistinctDataWriter DistinctWriter;
  DistinctWriter.writeData(HeaderData);
  Expected<DIEDataRef> MaybeDIE =
      convertInNewDIEBlock(DIE, DistinctWriter, AbbrevWriter);
  if (!MaybeDIE)
    return MaybeDIE.takeError();
  Expected<DIEAbbrevSetRef> MaybeAbbrevSet =
      AbbrevWriter.endAbbrevSet(CASBuilder);
  if (!MaybeAbbrevSet)
    return MaybeAbbrevSet.takeError();
  Expected<DIEDistinctDataRef> MaybeDistinct =
      DistinctWriter.getCASNode(CASBuilder);
  if (!MaybeDistinct)
    return MaybeDistinct.takeError();
  SmallVector<cas::ObjectRef, 3> Refs{
      MaybeDIE->getRef(), MaybeAbbrevSet->getRef(), MaybeDistinct->getRef()};
  return DIETopLevelRef::create(CASBuilder, Refs);
}

Expected<LoadedDIETopLevel>
mccasformats::v1::loadDIETopLevel(DIETopLevelRef TopLevelRef) {
  if (TopLevelRef.getNumReferences() != 3)
    return createStringError(
        inconvertibleErrorCode(),
        "TopLevelRef is expected to have three references");

  const MCSchema &Schema = TopLevelRef.getSchema();
  Expected<DIEDataRef> RootDIE =
      DIEDataRef::get(Schema, TopLevelRef.getReference(0));
  Expected<DIEAbbrevSetRef> AbbrevSet =
      DIEAbbrevSetRef::get(Schema, TopLevelRef.getReference(1));
  Expected<DIEDistinctDataRef> DistinctData =
      DIEDistinctDataRef::get(Schema, TopLevelRef.getReference(2));
  if (!RootDIE)
    return RootDIE.takeError();
  if (!AbbrevSet)
    return AbbrevSet.takeError();
  if (!DistinctData)
    return DistinctData.takeError();

  auto MaybeAbbrevEntries = loadAllRefs<DIEAbbrevRef>(*AbbrevSet);
  if (!MaybeAbbrevEntries)
    return MaybeAbbrevEntries.takeError();
  SmallVector<StringRef, 0> AbbrevData(map_range(
      *MaybeAbbrevEntries, [](DIEAbbrevRef Ref) { return Ref.getData(); }));
  return LoadedDIETopLevel{std::move(AbbrevData), *DistinctData,
                           *RootDIE};
}

struct DIEVisitor {
  Error visitDIERef(DIEDataRef Ref);
  Error visitDIERef(BinaryStreamReader &DataReader, unsigned AbbrevIdx,
                    StringRef DIEData, ArrayRef<DIEDataRef> &DIEChildrenStack);
  Error visitDIEAttrs(AbbrevEntryReader &AbbrevReader,
                      BinaryStreamReader &Reader, StringRef DIEData);

  ArrayRef<StringRef> AbbrevEntries;
  BinaryStreamReader DistinctReader;
  StringRef DistinctData;

  std::function<void(StringRef)> HeaderCallback;
  std::function<void(dwarf::Tag, uint64_t)> StartTagCallback;
  std::function<void(dwarf::Attribute, dwarf::Form, StringRef, bool)>
      AttrCallback;
  std::function<void(bool)> EndTagCallback;
  std::function<void(StringRef)> NewBlockCallback;
};

Error DIEVisitor::visitDIEAttrs(AbbrevEntryReader &AbbrevReader,
                                BinaryStreamReader &DataReader,
                                StringRef DIEData) {
  constexpr auto IsLittleEndian = true;
  constexpr auto AddrSize = 8;
  constexpr auto FormParams =
      dwarf::FormParams{4 /*Version*/, AddrSize, dwarf::DwarfFormat::DWARF32};

  while (true) {
    Expected<dwarf::Attribute> Attr = AbbrevReader.readAttr();
    if (!Attr)
      return Attr.takeError();
    if (*Attr == getEndOfAttributesMarker())
      break;

    Expected<dwarf::Form> Form = AbbrevReader.readForm();
    if (!Form)
      return Form.takeError();

    bool DataInDistinct = doesntDedup(*Form, *Attr);
    auto &ReaderForData = DataInDistinct ? DistinctReader : DataReader;
    StringRef DataToUse = DataInDistinct ? DistinctData : DIEData;
    Expected<uint64_t> FormSize =
        getFormSize(*Form, FormParams, DataToUse, ReaderForData.getOffset(),
                    IsLittleEndian, AddrSize);
    if (!FormSize)
      return FormSize.takeError();

    ArrayRef<char> RawBytes;
    if (auto E = ReaderForData.readArray(RawBytes, *FormSize))
      return E;
    AttrCallback(*Attr, *Form, toStringRef(RawBytes), DataInDistinct);
  }
  return Error::success();
}

static Expected<uint64_t> readAbbrevIdx(BinaryStreamReader &Reader) {
  uint64_t Idx;
  if (auto E = Reader.readULEB128(Idx))
    return std::move(E);
  return Idx;
}

static AbbrevEntryReader getAbbrevEntryReader(ArrayRef<StringRef> AbbrevEntries,
                                              unsigned AbbrevIdx) {
  StringRef AbbrevData =
      AbbrevEntries[decodeAbbrevIndexAsAbbrevSetIdx(AbbrevIdx)];
  return AbbrevEntryReader(AbbrevData);
}

Error DIEVisitor::visitDIERef(BinaryStreamReader &DataReader,
                              unsigned AbbrevIdx, StringRef DIEData,
                              ArrayRef<DIEDataRef> &DIEChildrenStack) {
  AbbrevEntryReader AbbrevReader =
      getAbbrevEntryReader(AbbrevEntries, AbbrevIdx);

  if (Expected<dwarf::Tag> MaybeTag = AbbrevReader.readTag())
    StartTagCallback(*MaybeTag, AbbrevIdx);
  else
    return MaybeTag.takeError();

  Expected<bool> MaybeHasChildren = AbbrevReader.readHasChildren();
  if (!MaybeHasChildren)
    return MaybeHasChildren.takeError();

  if (auto E = visitDIEAttrs(AbbrevReader, DataReader, DIEData))
    return E;

  if (!*MaybeHasChildren) {
    EndTagCallback(false /*HadChildren*/);
    return Error::success();
  }

  while (true) {
    Expected<uint64_t> ChildAbbrevIdx = readAbbrevIdx(DistinctReader);
    if (!ChildAbbrevIdx)
      return ChildAbbrevIdx.takeError();

    if (*ChildAbbrevIdx == getEndOfDIESiblingsMarker())
      break;

    if (*ChildAbbrevIdx == getDIEInAnotherBlockMarker()) {
      if (auto E = visitDIERef(DIEChildrenStack.front()))
        return E;
      DIEChildrenStack = DIEChildrenStack.drop_front();
      continue;
    }

    if (auto E =
            visitDIERef(DataReader, *ChildAbbrevIdx, DIEData, DIEChildrenStack))
      return E;
  }

  EndTagCallback(true /*HadChildren*/);
  return Error::success();
}

Error DIEVisitor::visitDIERef(DIEDataRef StartDIERef) {
  StringRef DIEData = StartDIERef.getData();
  BinaryStreamReader DataReader(DIEData, support::endianness::little);

  Expected<uint64_t> MaybeAbbrevIdx = readAbbrevIdx(DistinctReader);
  if (!MaybeAbbrevIdx)
    return MaybeAbbrevIdx.takeError();
  auto AbbrevIdx = *MaybeAbbrevIdx;

  // The tag of a fresh block must be meaningful, otherwise we wouldn't have
  // made a new block.
  assert(AbbrevIdx != getEndOfDIESiblingsMarker() &&
         AbbrevIdx != getDIEInAnotherBlockMarker());

  NewBlockCallback(StartDIERef.getID().toString());

  Expected<SmallVector<DIEDataRef>> MaybeChildren =
      loadAllRefs<DIEDataRef>(StartDIERef);
  if (!MaybeChildren)
    return MaybeChildren.takeError();
  ArrayRef<DIEDataRef> Children = *MaybeChildren;

  return visitDIERef(DataReader, AbbrevIdx, DIEData, Children);
}

Error mccasformats::v1::visitDebugInfo(
    SmallVectorImpl<StringRef> &TotAbbrevEntries,
    Expected<DIETopLevelRef> MaybeTopLevelRef,
    std::function<void(StringRef)> HeaderCallback,
    std::function<void(dwarf::Tag, uint64_t)> StartTagCallback,
    std::function<void(dwarf::Attribute, dwarf::Form, StringRef, bool)>
        AttrCallback,
    std::function<void(bool)> EndTagCallback,
    std::function<void(StringRef)> NewBlockCallback) {

  Expected<LoadedDIETopLevel> LoadedTopRef =
      loadDIETopLevel(std::move(MaybeTopLevelRef));
  if (!LoadedTopRef)
    return LoadedTopRef.takeError();

  StringRef DistinctData = LoadedTopRef->DistinctData.getData();
  BinaryStreamReader DistinctReader(DistinctData, support::endianness::little);
  ArrayRef<char> HeaderData;

  auto BeginOffset = DistinctReader.getOffset();
  auto Size = getSizeFromDwarfHeader(DistinctReader);
  if (!Size)
    return Size.takeError();

  // 2-byte Dwarf version identifier.
  uint16_t DwarfVersion;
  if (auto E = DistinctReader.readInteger(DwarfVersion))
    return E;

  DistinctReader.setOffset(BeginOffset);

  if (auto E = DistinctReader.readArray(
          HeaderData,
          DwarfVersion >= 5 ? Dwarf5HeaderSize32Bit : Dwarf4HeaderSize32Bit))
    return E;
  HeaderCallback(toStringRef(HeaderData));

  append_range(TotAbbrevEntries, LoadedTopRef->AbbrevEntries);
  DIEVisitor Visitor{TotAbbrevEntries, DistinctReader,   DistinctData,
                     HeaderCallback,   StartTagCallback, AttrCallback,
                     EndTagCallback,   NewBlockCallback};
  return Visitor.visitDIERef(LoadedTopRef->RootDIE);
}
