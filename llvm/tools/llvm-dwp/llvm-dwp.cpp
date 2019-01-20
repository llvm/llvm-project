//===-- llvm-dwp.cpp - Split DWARF merging tool for llvm ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A utility for merging DWARF 5 Split DWARF .dwo files into .dwp (DWARF
// package files).
//
//===----------------------------------------------------------------------===//
#include "DWPError.h"
#include "DWPStringPool.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFUnitIndex.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.inc"
#include "llvm/Object/Decompressor.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

cl::OptionCategory DwpCategory("Specific Options");
static cl::list<std::string> InputFiles(cl::Positional, cl::ZeroOrMore,
                                        cl::desc("<input files>"),
                                        cl::cat(DwpCategory));

static cl::list<std::string> ExecFilenames(
    "e", cl::ZeroOrMore,
    cl::desc("Specify the executable/library files to get the list of *.dwo from"),
    cl::value_desc("filename"), cl::cat(DwpCategory));

static cl::opt<std::string> OutputFilename(cl::Required, "o",
                                           cl::desc("Specify the output file."),
                                           cl::value_desc("filename"),
                                           cl::cat(DwpCategory));

static void writeStringsAndOffsets(MCStreamer &Out, DWPStringPool &Strings,
                                   MCSection *StrOffsetSection,
                                   StringRef CurStrSection,
                                   StringRef CurStrOffsetSection) {
  // Could possibly produce an error or warning if one of these was non-null but
  // the other was null.
  if (CurStrSection.empty() || CurStrOffsetSection.empty())
    return;

  DenseMap<uint32_t, uint32_t> OffsetRemapping;

  DataExtractor Data(CurStrSection, true, 0);
  uint32_t LocalOffset = 0;
  uint32_t PrevOffset = 0;
  while (const char *s = Data.getCStr(&LocalOffset)) {
    OffsetRemapping[PrevOffset] =
        Strings.getOffset(s, LocalOffset - PrevOffset);
    PrevOffset = LocalOffset;
  }

  Data = DataExtractor(CurStrOffsetSection, true, 0);

  Out.SwitchSection(StrOffsetSection);

  uint32_t Offset = 0;
  uint64_t Size = CurStrOffsetSection.size();
  while (Offset < Size) {
    auto OldOffset = Data.getU32(&Offset);
    auto NewOffset = OffsetRemapping[OldOffset];
    Out.EmitIntValue(NewOffset, 4);
  }
}

static uint32_t getCUAbbrev(StringRef Abbrev, uint64_t AbbrCode) {
  uint64_t CurCode;
  uint32_t Offset = 0;
  DataExtractor AbbrevData(Abbrev, true, 0);
  while ((CurCode = AbbrevData.getULEB128(&Offset)) != AbbrCode) {
    // Tag
    AbbrevData.getULEB128(&Offset);
    // DW_CHILDREN
    AbbrevData.getU8(&Offset);
    // Attributes
    while (AbbrevData.getULEB128(&Offset) | AbbrevData.getULEB128(&Offset))
      ;
  }
  return Offset;
}

struct CompileUnitIdentifiers {
  uint64_t Signature = 0;
  const char *Name = "";
  const char *DWOName = "";
};

static Expected<const char *>
getIndexedString(dwarf::Form Form, DataExtractor InfoData,
                 uint32_t &InfoOffset, StringRef StrOffsets, StringRef Str) {
  if (Form == dwarf::DW_FORM_string)
    return InfoData.getCStr(&InfoOffset);
  if (Form != dwarf::DW_FORM_GNU_str_index)
    return make_error<DWPError>(
        "string field encoded without DW_FORM_string or DW_FORM_GNU_str_index");
  auto StrIndex = InfoData.getULEB128(&InfoOffset);
  DataExtractor StrOffsetsData(StrOffsets, true, 0);
  uint32_t StrOffsetsOffset = 4 * StrIndex;
  uint32_t StrOffset = StrOffsetsData.getU32(&StrOffsetsOffset);
  DataExtractor StrData(Str, true, 0);
  return StrData.getCStr(&StrOffset);
}

static Expected<CompileUnitIdentifiers> getCUIdentifiers(StringRef Abbrev,
                                                         StringRef Info,
                                                         StringRef StrOffsets,
                                                         StringRef Str) {
  uint32_t Offset = 0;
  DataExtractor InfoData(Info, true, 0);
  dwarf::DwarfFormat Format = dwarf::DwarfFormat::DWARF32;
  uint64_t Length = InfoData.getU32(&Offset);
  // If the length is 0xffffffff, then this indictes that this is a DWARF 64
  // stream and the length is actually encoded into a 64 bit value that follows.
  if (Length == 0xffffffffU) {
    Format = dwarf::DwarfFormat::DWARF64;
    Length = InfoData.getU64(&Offset);
  }
  uint16_t Version = InfoData.getU16(&Offset);
  InfoData.getU32(&Offset); // Abbrev offset (should be zero)
  uint8_t AddrSize = InfoData.getU8(&Offset);

  uint32_t AbbrCode = InfoData.getULEB128(&Offset);

  DataExtractor AbbrevData(Abbrev, true, 0);
  uint32_t AbbrevOffset = getCUAbbrev(Abbrev, AbbrCode);
  auto Tag = static_cast<dwarf::Tag>(AbbrevData.getULEB128(&AbbrevOffset));
  if (Tag != dwarf::DW_TAG_compile_unit)
    return make_error<DWPError>("top level DIE is not a compile unit");
  // DW_CHILDREN
  AbbrevData.getU8(&AbbrevOffset);
  uint32_t Name;
  dwarf::Form Form;
  CompileUnitIdentifiers ID;
  while ((Name = AbbrevData.getULEB128(&AbbrevOffset)) |
         (Form = static_cast<dwarf::Form>(AbbrevData.getULEB128(&AbbrevOffset))) &&
         (Name != 0 || Form != 0)) {
    switch (Name) {
    case dwarf::DW_AT_name: {
      Expected<const char *> EName =
          getIndexedString(Form, InfoData, Offset, StrOffsets, Str);
      if (!EName)
        return EName.takeError();
      ID.Name = *EName;
      break;
    }
    case dwarf::DW_AT_GNU_dwo_name: {
      Expected<const char *> EName =
          getIndexedString(Form, InfoData, Offset, StrOffsets, Str);
      if (!EName)
        return EName.takeError();
      ID.DWOName = *EName;
      break;
    }
    case dwarf::DW_AT_GNU_dwo_id:
      ID.Signature = InfoData.getU64(&Offset);
      break;
    default:
      DWARFFormValue::skipValue(Form, InfoData, &Offset,
                                dwarf::FormParams({Version, AddrSize, Format}));
    }
  }
  return ID;
}

struct UnitIndexEntry {
  DWARFUnitIndex::Entry::SectionContribution Contributions[8];
  std::string Name;
  std::string DWOName;
  StringRef DWPName;
};

static StringRef getSubsection(StringRef Section,
                               const DWARFUnitIndex::Entry &Entry,
                               DWARFSectionKind Kind) {
  const auto *Off = Entry.getOffset(Kind);
  if (!Off)
    return StringRef();
  return Section.substr(Off->Offset, Off->Length);
}

static void addAllTypesFromDWP(
    MCStreamer &Out, MapVector<uint64_t, UnitIndexEntry> &TypeIndexEntries,
    const DWARFUnitIndex &TUIndex, MCSection *OutputTypes, StringRef Types,
    const UnitIndexEntry &TUEntry, uint32_t &TypesOffset) {
  Out.SwitchSection(OutputTypes);
  for (const DWARFUnitIndex::Entry &E : TUIndex.getRows()) {
    auto *I = E.getOffsets();
    if (!I)
      continue;
    auto P = TypeIndexEntries.insert(std::make_pair(E.getSignature(), TUEntry));
    if (!P.second)
      continue;
    auto &Entry = P.first->second;
    // Zero out the debug_info contribution
    Entry.Contributions[0] = {};
    for (auto Kind : TUIndex.getColumnKinds()) {
      auto &C = Entry.Contributions[Kind - DW_SECT_INFO];
      C.Offset += I->Offset;
      C.Length = I->Length;
      ++I;
    }
    auto &C = Entry.Contributions[DW_SECT_TYPES - DW_SECT_INFO];
    Out.EmitBytes(Types.substr(
        C.Offset - TUEntry.Contributions[DW_SECT_TYPES - DW_SECT_INFO].Offset,
        C.Length));
    C.Offset = TypesOffset;
    TypesOffset += C.Length;
  }
}

static void addAllTypes(MCStreamer &Out,
                        MapVector<uint64_t, UnitIndexEntry> &TypeIndexEntries,
                        MCSection *OutputTypes,
                        const std::vector<StringRef> &TypesSections,
                        const UnitIndexEntry &CUEntry, uint32_t &TypesOffset) {
  for (StringRef Types : TypesSections) {
    Out.SwitchSection(OutputTypes);
    uint32_t Offset = 0;
    DataExtractor Data(Types, true, 0);
    while (Data.isValidOffset(Offset)) {
      UnitIndexEntry Entry = CUEntry;
      // Zero out the debug_info contribution
      Entry.Contributions[0] = {};
      auto &C = Entry.Contributions[DW_SECT_TYPES - DW_SECT_INFO];
      C.Offset = TypesOffset;
      auto PrevOffset = Offset;
      // Length of the unit, including the 4 byte length field.
      C.Length = Data.getU32(&Offset) + 4;

      Data.getU16(&Offset); // Version
      Data.getU32(&Offset); // Abbrev offset
      Data.getU8(&Offset);  // Address size
      auto Signature = Data.getU64(&Offset);
      Offset = PrevOffset + C.Length;

      auto P = TypeIndexEntries.insert(std::make_pair(Signature, Entry));
      if (!P.second)
        continue;

      Out.EmitBytes(Types.substr(PrevOffset, C.Length));
      TypesOffset += C.Length;
    }
  }
}

static void
writeIndexTable(MCStreamer &Out, ArrayRef<unsigned> ContributionOffsets,
                const MapVector<uint64_t, UnitIndexEntry> &IndexEntries,
                uint32_t DWARFUnitIndex::Entry::SectionContribution::*Field) {
  for (const auto &E : IndexEntries)
    for (size_t i = 0; i != array_lengthof(E.second.Contributions); ++i)
      if (ContributionOffsets[i])
        Out.EmitIntValue(E.second.Contributions[i].*Field, 4);
}

static void
writeIndex(MCStreamer &Out, MCSection *Section,
           ArrayRef<unsigned> ContributionOffsets,
           const MapVector<uint64_t, UnitIndexEntry> &IndexEntries) {
  if (IndexEntries.empty())
    return;

  unsigned Columns = 0;
  for (auto &C : ContributionOffsets)
    if (C)
      ++Columns;

  std::vector<unsigned> Buckets(NextPowerOf2(3 * IndexEntries.size() / 2));
  uint64_t Mask = Buckets.size() - 1;
  size_t i = 0;
  for (const auto &P : IndexEntries) {
    auto S = P.first;
    auto H = S & Mask;
    auto HP = ((S >> 32) & Mask) | 1;
    while (Buckets[H]) {
      assert(S != IndexEntries.begin()[Buckets[H] - 1].first &&
             "Duplicate unit");
      H = (H + HP) & Mask;
    }
    Buckets[H] = i + 1;
    ++i;
  }

  Out.SwitchSection(Section);
  Out.EmitIntValue(2, 4);                   // Version
  Out.EmitIntValue(Columns, 4);             // Columns
  Out.EmitIntValue(IndexEntries.size(), 4); // Num Units
  Out.EmitIntValue(Buckets.size(), 4);      // Num Buckets

  // Write the signatures.
  for (const auto &I : Buckets)
    Out.EmitIntValue(I ? IndexEntries.begin()[I - 1].first : 0, 8);

  // Write the indexes.
  for (const auto &I : Buckets)
    Out.EmitIntValue(I, 4);

  // Write the column headers (which sections will appear in the table)
  for (size_t i = 0; i != ContributionOffsets.size(); ++i)
    if (ContributionOffsets[i])
      Out.EmitIntValue(i + DW_SECT_INFO, 4);

  // Write the offsets.
  writeIndexTable(Out, ContributionOffsets, IndexEntries,
                  &DWARFUnitIndex::Entry::SectionContribution::Offset);

  // Write the lengths.
  writeIndexTable(Out, ContributionOffsets, IndexEntries,
                  &DWARFUnitIndex::Entry::SectionContribution::Length);
}

std::string buildDWODescription(StringRef Name, StringRef DWPName, StringRef DWOName) {
  std::string Text = "\'";
  Text += Name;
  Text += '\'';
  if (!DWPName.empty()) {
    Text += " (from ";
    if (!DWOName.empty()) {
      Text += '\'';
      Text += DWOName;
      Text += "' in ";
    }
    Text += '\'';
    Text += DWPName;
    Text += "')";
  }
  return Text;
}

static Error createError(StringRef Name, Error E) {
  return make_error<DWPError>(
      ("failure while decompressing compressed section: '" + Name + "', " +
       llvm::toString(std::move(E)))
          .str());
}

static Error
handleCompressedSection(std::deque<SmallString<32>> &UncompressedSections,
                        StringRef &Name, StringRef &Contents) {
  if (!Decompressor::isGnuStyle(Name))
    return Error::success();

  Expected<Decompressor> Dec =
      Decompressor::create(Name, Contents, false /*IsLE*/, false /*Is64Bit*/);
  if (!Dec)
    return createError(Name, Dec.takeError());

  UncompressedSections.emplace_back();
  if (Error E = Dec->resizeAndDecompress(UncompressedSections.back()))
    return createError(Name, std::move(E));

  Name = Name.substr(2); // Drop ".z"
  Contents = UncompressedSections.back();
  return Error::success();
}

static Error handleSection(
    const StringMap<std::pair<MCSection *, DWARFSectionKind>> &KnownSections,
    const MCSection *StrSection, const MCSection *StrOffsetSection,
    const MCSection *TypesSection, const MCSection *CUIndexSection,
    const MCSection *TUIndexSection, const SectionRef &Section, MCStreamer &Out,
    std::deque<SmallString<32>> &UncompressedSections,
    uint32_t (&ContributionOffsets)[8], UnitIndexEntry &CurEntry,
    StringRef &CurStrSection, StringRef &CurStrOffsetSection,
    std::vector<StringRef> &CurTypesSection, StringRef &InfoSection,
    StringRef &AbbrevSection, StringRef &CurCUIndexSection,
    StringRef &CurTUIndexSection) {
  if (Section.isBSS())
    return Error::success();

  if (Section.isVirtual())
    return Error::success();

  StringRef Name;
  if (std::error_code Err = Section.getName(Name))
    return errorCodeToError(Err);

  StringRef Contents;
  if (auto Err = Section.getContents(Contents))
    return errorCodeToError(Err);

  if (auto Err = handleCompressedSection(UncompressedSections, Name, Contents))
    return Err;

  Name = Name.substr(Name.find_first_not_of("._"));

  auto SectionPair = KnownSections.find(Name);
  if (SectionPair == KnownSections.end())
    return Error::success();

  if (DWARFSectionKind Kind = SectionPair->second.second) {
    auto Index = Kind - DW_SECT_INFO;
    if (Kind != DW_SECT_TYPES) {
      CurEntry.Contributions[Index].Offset = ContributionOffsets[Index];
      ContributionOffsets[Index] +=
          (CurEntry.Contributions[Index].Length = Contents.size());
    }

    switch (Kind) {
    case DW_SECT_INFO:
      InfoSection = Contents;
      break;
    case DW_SECT_ABBREV:
      AbbrevSection = Contents;
      break;
    default:
      break;
    }
  }

  MCSection *OutSection = SectionPair->second.first;
  if (OutSection == StrOffsetSection)
    CurStrOffsetSection = Contents;
  else if (OutSection == StrSection)
    CurStrSection = Contents;
  else if (OutSection == TypesSection)
    CurTypesSection.push_back(Contents);
  else if (OutSection == CUIndexSection)
    CurCUIndexSection = Contents;
  else if (OutSection == TUIndexSection)
    CurTUIndexSection = Contents;
  else {
    Out.SwitchSection(OutSection);
    Out.EmitBytes(Contents);
  }
  return Error::success();
}

static Error
buildDuplicateError(const std::pair<uint64_t, UnitIndexEntry> &PrevE,
                    const CompileUnitIdentifiers &ID, StringRef DWPName) {
  return make_error<DWPError>(
      std::string("Duplicate DWO ID (") + utohexstr(PrevE.first) + ") in " +
      buildDWODescription(PrevE.second.Name, PrevE.second.DWPName,
                          PrevE.second.DWOName) +
      " and " + buildDWODescription(ID.Name, DWPName, ID.DWOName));
}

static Expected<SmallVector<std::string, 16>>
getDWOFilenames(StringRef ExecFilename) {
  auto ErrOrObj = object::ObjectFile::createObjectFile(ExecFilename);
  if (!ErrOrObj)
    return ErrOrObj.takeError();

  const ObjectFile &Obj = *ErrOrObj.get().getBinary();
  std::unique_ptr<DWARFContext> DWARFCtx = DWARFContext::create(Obj);

  SmallVector<std::string, 16> DWOPaths;
  for (const auto &CU : DWARFCtx->compile_units()) {
    const DWARFDie &Die = CU->getUnitDIE();
    std::string DWOName = dwarf::toString(
        Die.find({dwarf::DW_AT_dwo_name, dwarf::DW_AT_GNU_dwo_name}), "");
    if (DWOName.empty())
      continue;
    std::string DWOCompDir =
        dwarf::toString(Die.find(dwarf::DW_AT_comp_dir), "");
    if (!DWOCompDir.empty()) {
      SmallString<16> DWOPath;
      sys::path::append(DWOPath, DWOCompDir, DWOName);
      DWOPaths.emplace_back(DWOPath.data(), DWOPath.size());
    } else {
      DWOPaths.push_back(std::move(DWOName));
    }
  }
  return std::move(DWOPaths);
}

static Error write(MCStreamer &Out, ArrayRef<std::string> Inputs) {
  const auto &MCOFI = *Out.getContext().getObjectFileInfo();
  MCSection *const StrSection = MCOFI.getDwarfStrDWOSection();
  MCSection *const StrOffsetSection = MCOFI.getDwarfStrOffDWOSection();
  MCSection *const TypesSection = MCOFI.getDwarfTypesDWOSection();
  MCSection *const CUIndexSection = MCOFI.getDwarfCUIndexSection();
  MCSection *const TUIndexSection = MCOFI.getDwarfTUIndexSection();
  const StringMap<std::pair<MCSection *, DWARFSectionKind>> KnownSections = {
      {"debug_info.dwo", {MCOFI.getDwarfInfoDWOSection(), DW_SECT_INFO}},
      {"debug_types.dwo", {MCOFI.getDwarfTypesDWOSection(), DW_SECT_TYPES}},
      {"debug_str_offsets.dwo", {StrOffsetSection, DW_SECT_STR_OFFSETS}},
      {"debug_str.dwo", {StrSection, static_cast<DWARFSectionKind>(0)}},
      {"debug_loc.dwo", {MCOFI.getDwarfLocDWOSection(), DW_SECT_LOC}},
      {"debug_line.dwo", {MCOFI.getDwarfLineDWOSection(), DW_SECT_LINE}},
      {"debug_abbrev.dwo", {MCOFI.getDwarfAbbrevDWOSection(), DW_SECT_ABBREV}},
      {"debug_cu_index", {CUIndexSection, static_cast<DWARFSectionKind>(0)}},
      {"debug_tu_index", {TUIndexSection, static_cast<DWARFSectionKind>(0)}}};

  MapVector<uint64_t, UnitIndexEntry> IndexEntries;
  MapVector<uint64_t, UnitIndexEntry> TypeIndexEntries;

  uint32_t ContributionOffsets[8] = {};

  DWPStringPool Strings(Out, StrSection);

  SmallVector<OwningBinary<object::ObjectFile>, 128> Objects;
  Objects.reserve(Inputs.size());

  std::deque<SmallString<32>> UncompressedSections;

  for (const auto &Input : Inputs) {
    auto ErrOrObj = object::ObjectFile::createObjectFile(Input);
    if (!ErrOrObj)
      return ErrOrObj.takeError();

    auto &Obj = *ErrOrObj->getBinary();
    Objects.push_back(std::move(*ErrOrObj));

    UnitIndexEntry CurEntry = {};

    StringRef CurStrSection;
    StringRef CurStrOffsetSection;
    std::vector<StringRef> CurTypesSection;
    StringRef InfoSection;
    StringRef AbbrevSection;
    StringRef CurCUIndexSection;
    StringRef CurTUIndexSection;

    for (const auto &Section : Obj.sections())
      if (auto Err = handleSection(
              KnownSections, StrSection, StrOffsetSection, TypesSection,
              CUIndexSection, TUIndexSection, Section, Out,
              UncompressedSections, ContributionOffsets, CurEntry,
              CurStrSection, CurStrOffsetSection, CurTypesSection, InfoSection,
              AbbrevSection, CurCUIndexSection, CurTUIndexSection))
        return Err;

    if (InfoSection.empty())
      continue;

    writeStringsAndOffsets(Out, Strings, StrOffsetSection, CurStrSection,
                           CurStrOffsetSection);

    if (CurCUIndexSection.empty()) {
      Expected<CompileUnitIdentifiers> EID = getCUIdentifiers(
          AbbrevSection, InfoSection, CurStrOffsetSection, CurStrSection);
      if (!EID)
        return EID.takeError();
      const auto &ID = *EID;
      auto P = IndexEntries.insert(std::make_pair(ID.Signature, CurEntry));
      if (!P.second)
        return buildDuplicateError(*P.first, ID, "");
      P.first->second.Name = ID.Name;
      P.first->second.DWOName = ID.DWOName;
      addAllTypes(Out, TypeIndexEntries, TypesSection, CurTypesSection,
                  CurEntry, ContributionOffsets[DW_SECT_TYPES - DW_SECT_INFO]);
      continue;
    }

    DWARFUnitIndex CUIndex(DW_SECT_INFO);
    DataExtractor CUIndexData(CurCUIndexSection, Obj.isLittleEndian(), 0);
    if (!CUIndex.parse(CUIndexData))
      return make_error<DWPError>("Failed to parse cu_index");

    for (const DWARFUnitIndex::Entry &E : CUIndex.getRows()) {
      auto *I = E.getOffsets();
      if (!I)
        continue;
      auto P = IndexEntries.insert(std::make_pair(E.getSignature(), CurEntry));
      Expected<CompileUnitIdentifiers> EID = getCUIdentifiers(
          getSubsection(AbbrevSection, E, DW_SECT_ABBREV),
          getSubsection(InfoSection, E, DW_SECT_INFO),
          getSubsection(CurStrOffsetSection, E, DW_SECT_STR_OFFSETS),
          CurStrSection);
      if (!EID)
        return EID.takeError();
      const auto &ID = *EID;
      if (!P.second)
        return buildDuplicateError(*P.first, ID, Input);
      auto &NewEntry = P.first->second;
      NewEntry.Name = ID.Name;
      NewEntry.DWOName = ID.DWOName;
      NewEntry.DWPName = Input;
      for (auto Kind : CUIndex.getColumnKinds()) {
        auto &C = NewEntry.Contributions[Kind - DW_SECT_INFO];
        C.Offset += I->Offset;
        C.Length = I->Length;
        ++I;
      }
    }

    if (!CurTypesSection.empty()) {
      if (CurTypesSection.size() != 1)
        return make_error<DWPError>("multiple type unit sections in .dwp file");
      DWARFUnitIndex TUIndex(DW_SECT_TYPES);
      DataExtractor TUIndexData(CurTUIndexSection, Obj.isLittleEndian(), 0);
      if (!TUIndex.parse(TUIndexData))
        return make_error<DWPError>("Failed to parse tu_index");
      addAllTypesFromDWP(Out, TypeIndexEntries, TUIndex, TypesSection,
                         CurTypesSection.front(), CurEntry,
                         ContributionOffsets[DW_SECT_TYPES - DW_SECT_INFO]);
    }
  }

  // Lie about there being no info contributions so the TU index only includes
  // the type unit contribution
  ContributionOffsets[0] = 0;
  writeIndex(Out, MCOFI.getDwarfTUIndexSection(), ContributionOffsets,
             TypeIndexEntries);

  // Lie about the type contribution
  ContributionOffsets[DW_SECT_TYPES - DW_SECT_INFO] = 0;
  // Unlie about the info contribution
  ContributionOffsets[0] = 1;

  writeIndex(Out, MCOFI.getDwarfCUIndexSection(), ContributionOffsets,
             IndexEntries);

  return Error::success();
}

static int error(const Twine &Error, const Twine &Context) {
  errs() << Twine("while processing ") + Context + ":\n";
  errs() << Twine("error: ") + Error + "\n";
  return 1;
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  cl::ParseCommandLineOptions(argc, argv, "merge split dwarf (.dwo) files\n");

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();

  std::string ErrorStr;
  StringRef Context = "dwarf streamer init";

  Triple TheTriple("x86_64-linux-gnu");

  // Get the target.
  const Target *TheTarget =
      TargetRegistry::lookupTarget("", TheTriple, ErrorStr);
  if (!TheTarget)
    return error(ErrorStr, Context);
  std::string TripleName = TheTriple.getTriple();

  // Create all the MC Objects.
  std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TripleName));
  if (!MRI)
    return error(Twine("no register info for target ") + TripleName, Context);

  std::unique_ptr<MCAsmInfo> MAI(TheTarget->createMCAsmInfo(*MRI, TripleName));
  if (!MAI)
    return error("no asm info for target " + TripleName, Context);

  MCObjectFileInfo MOFI;
  MCContext MC(MAI.get(), MRI.get(), &MOFI);
  MOFI.InitMCObjectFileInfo(TheTriple, /*PIC*/ false, MC);

  std::unique_ptr<MCSubtargetInfo> MSTI(
      TheTarget->createMCSubtargetInfo(TripleName, "", ""));
  if (!MSTI)
    return error("no subtarget info for target " + TripleName, Context);

  MCTargetOptions Options;
  auto MAB = TheTarget->createMCAsmBackend(*MSTI, *MRI, Options);
  if (!MAB)
    return error("no asm backend for target " + TripleName, Context);

  std::unique_ptr<MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  if (!MII)
    return error("no instr info info for target " + TripleName, Context);

  MCCodeEmitter *MCE = TheTarget->createMCCodeEmitter(*MII, *MRI, MC);
  if (!MCE)
    return error("no code emitter for target " + TripleName, Context);

  // Create the output file.
  std::error_code EC;
  raw_fd_ostream OutFile(OutputFilename, EC, sys::fs::F_None);
  Optional<buffer_ostream> BOS;
  raw_pwrite_stream *OS;
  if (EC)
    return error(Twine(OutputFilename) + ": " + EC.message(), Context);
  if (OutFile.supportsSeeking()) {
    OS = &OutFile;
  } else {
    BOS.emplace(OutFile);
    OS = BOS.getPointer();
  }

  MCTargetOptions MCOptions = InitMCTargetOptionsFromFlags();
  std::unique_ptr<MCStreamer> MS(TheTarget->createMCObjectStreamer(
      TheTriple, MC, std::unique_ptr<MCAsmBackend>(MAB),
      MAB->createObjectWriter(*OS), std::unique_ptr<MCCodeEmitter>(MCE),
      *MSTI, MCOptions.MCRelaxAll, MCOptions.MCIncrementalLinkerCompatible,
      /*DWARFMustBeAtTheEnd*/ false));
  if (!MS)
    return error("no object streamer for target " + TripleName, Context);

  std::vector<std::string> DWOFilenames = InputFiles;
  for (const auto &ExecFilename : ExecFilenames) {
    auto DWOs = getDWOFilenames(ExecFilename);
    if (!DWOs) {
      logAllUnhandledErrors(DWOs.takeError(), errs(), "error: ");
      return 1;
    }
    DWOFilenames.insert(DWOFilenames.end(),
                        std::make_move_iterator(DWOs->begin()),
                        std::make_move_iterator(DWOs->end()));
  }

  if (auto Err = write(*MS, DWOFilenames)) {
    logAllUnhandledErrors(std::move(Err), errs(), "error: ");
    return 1;
  }

  MS->Finish();
}
