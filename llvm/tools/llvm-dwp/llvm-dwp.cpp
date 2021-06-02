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
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
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
#include "llvm/MC/MCTargetOptionsCommandFlags.h"
#include "llvm/Object/Decompressor.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

static mc::RegisterMCTargetOptionsFlags MCTargetOptionsFlags;

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

// Returns the size of debug_str_offsets section headers in bytes.
static uint64_t debugStrOffsetsHeaderSize(DataExtractor StrOffsetsData,
                                          uint16_t DwarfVersion) {
  if (DwarfVersion <= 4)
    return 0; // There is no header before dwarf 5.
  uint64_t Offset = 0;
  uint64_t Length = StrOffsetsData.getU32(&Offset);
  if (Length == llvm::dwarf::DW_LENGTH_DWARF64)
    return 16; // unit length: 12 bytes, version: 2 bytes, padding: 2 bytes.
  return 8;    // unit length: 4 bytes, version: 2 bytes, padding: 2 bytes.
}

// Holds data for Skeleton, Split Compilation, and Type Unit Headers (only in
// v5) as defined in Dwarf 5 specification, 7.5.1.2, 7.5.1.3 and Dwarf 4
// specification 7.5.1.1.
struct InfoSectionUnitHeader {
  // unit_length field. Note that the type is uint64_t even in 32-bit dwarf.
  uint64_t Length = 0;

  // version field.
  uint16_t Version = 0;

  // unit_type field. Initialized only if Version >= 5.
  uint8_t UnitType = 0;

  // address_size field.
  uint8_t AddrSize = 0;

  // debug_abbrev_offset field. Note that the type is uint64_t even in 32-bit
  // dwarf. It is assumed to be 0.
  uint64_t DebugAbbrevOffset = 0;

  // dwo_id field. This resides in the header only if Version >= 5.
  // In earlier versions, it is read from DW_AT_GNU_dwo_id.
  Optional<uint64_t> Signature = None;

  // Derived from the length of Length field.
  dwarf::DwarfFormat Format = dwarf::DwarfFormat::DWARF32;

  // The size of the Header in bytes. This is derived while parsing the header,
  // and is stored as a convenience.
  uint8_t HeaderSize = 0;
};

// Parse and return the header of an info section compile/type unit.
static Expected<InfoSectionUnitHeader>
parseInfoSectionUnitHeader(StringRef Info) {
  InfoSectionUnitHeader Header;
  Error Err = Error::success();
  uint64_t Offset = 0;
  DWARFDataExtractor InfoData(Info, true, 0);
  std::tie(Header.Length, Header.Format) =
      InfoData.getInitialLength(&Offset, &Err);
  if (Err)
    return make_error<DWPError>("cannot parse compile unit length: " +
                                llvm::toString(std::move(Err)));

  if (!InfoData.isValidOffset(Offset + (Header.Length - 1))) {
    return make_error<DWPError>(
        "compile unit exceeds .debug_info section range: " +
        utostr(Offset + Header.Length) + " >= " + utostr(InfoData.size()));
  }

  Header.Version = InfoData.getU16(&Offset, &Err);
  if (Err)
    return make_error<DWPError>("cannot parse compile unit version: " +
                                llvm::toString(std::move(Err)));

  uint64_t MinHeaderLength;
  if (Header.Version >= 5) {
    // Size: Version (2), UnitType (1), AddrSize (1), DebugAbbrevOffset (4),
    // Signature (8)
    MinHeaderLength = 16;
  } else {
    // Size: Version (2), DebugAbbrevOffset (4), AddrSize (1)
    MinHeaderLength = 7;
  }
  if (Header.Length < MinHeaderLength) {
    return make_error<DWPError>("unit length is too small: expected at least " +
                                utostr(MinHeaderLength) + " got " +
                                utostr(Header.Length) + ".");
  }
  if (Header.Version >= 5) {
    Header.UnitType = InfoData.getU8(&Offset);
    Header.AddrSize = InfoData.getU8(&Offset);
    Header.DebugAbbrevOffset = InfoData.getU32(&Offset);
    Header.Signature = InfoData.getU64(&Offset);
    if (Header.UnitType == dwarf::DW_UT_split_type) {
      // Type offset.
      MinHeaderLength += 4;
      if (Header.Length < MinHeaderLength)
        return make_error<DWPError>("type unit is missing type offset");
      InfoData.getU32(&Offset);
    }
  } else {
    // Note that, address_size and debug_abbrev_offset fields have switched
    // places between dwarf version 4 and 5.
    Header.DebugAbbrevOffset = InfoData.getU32(&Offset);
    Header.AddrSize = InfoData.getU8(&Offset);
  }

  Header.HeaderSize = Offset;
  return Header;
}

static void writeStringsAndOffsets(MCStreamer &Out, DWPStringPool &Strings,
                                   MCSection *StrOffsetSection,
                                   StringRef CurStrSection,
                                   StringRef CurStrOffsetSection,
                                   uint16_t Version) {
  // Could possibly produce an error or warning if one of these was non-null but
  // the other was null.
  if (CurStrSection.empty() || CurStrOffsetSection.empty())
    return;

  DenseMap<uint64_t, uint32_t> OffsetRemapping;

  DataExtractor Data(CurStrSection, true, 0);
  uint64_t LocalOffset = 0;
  uint64_t PrevOffset = 0;
  while (const char *s = Data.getCStr(&LocalOffset)) {
    OffsetRemapping[PrevOffset] =
        Strings.getOffset(s, LocalOffset - PrevOffset);
    PrevOffset = LocalOffset;
  }

  Data = DataExtractor(CurStrOffsetSection, true, 0);

  Out.SwitchSection(StrOffsetSection);

  uint64_t HeaderSize = debugStrOffsetsHeaderSize(Data, Version);
  uint64_t Offset = 0;
  uint64_t Size = CurStrOffsetSection.size();
  // FIXME: This can be caused by bad input and should be handled as such.
  assert(HeaderSize <= Size && "StrOffsetSection size is less than its header");
  // Copy the header to the output.
  Out.emitBytes(Data.getBytes(&Offset, HeaderSize));
  while (Offset < Size) {
    auto OldOffset = Data.getU32(&Offset);
    auto NewOffset = OffsetRemapping[OldOffset];
    Out.emitIntValue(NewOffset, 4);
  }
}

static uint64_t getCUAbbrev(StringRef Abbrev, uint64_t AbbrCode) {
  uint64_t Offset = 0;
  DataExtractor AbbrevData(Abbrev, true, 0);
  while (AbbrevData.getULEB128(&Offset) != AbbrCode) {
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
getIndexedString(dwarf::Form Form, DataExtractor InfoData, uint64_t &InfoOffset,
                 StringRef StrOffsets, StringRef Str, uint16_t Version) {
  if (Form == dwarf::DW_FORM_string)
    return InfoData.getCStr(&InfoOffset);
  uint64_t StrIndex;
  switch (Form) {
  case dwarf::DW_FORM_strx1:
    StrIndex = InfoData.getU8(&InfoOffset);
    break;
  case dwarf::DW_FORM_strx2:
    StrIndex = InfoData.getU16(&InfoOffset);
    break;
  case dwarf::DW_FORM_strx3:
    StrIndex = InfoData.getU24(&InfoOffset);
    break;
  case dwarf::DW_FORM_strx4:
    StrIndex = InfoData.getU32(&InfoOffset);
    break;
  case dwarf::DW_FORM_strx:
  case dwarf::DW_FORM_GNU_str_index:
    StrIndex = InfoData.getULEB128(&InfoOffset);
    break;
  default:
    return make_error<DWPError>(
        "string field must be encoded with one of the following: "
        "DW_FORM_string, DW_FORM_strx, DW_FORM_strx1, DW_FORM_strx2, "
        "DW_FORM_strx3, DW_FORM_strx4, or DW_FORM_GNU_str_index.");
  }
  DataExtractor StrOffsetsData(StrOffsets, true, 0);
  uint64_t StrOffsetsOffset = 4 * StrIndex;
  StrOffsetsOffset += debugStrOffsetsHeaderSize(StrOffsetsData, Version);

  uint64_t StrOffset = StrOffsetsData.getU32(&StrOffsetsOffset);
  DataExtractor StrData(Str, true, 0);
  return StrData.getCStr(&StrOffset);
}

static Expected<CompileUnitIdentifiers>
getCUIdentifiers(InfoSectionUnitHeader &Header, StringRef Abbrev,
                 StringRef Info, StringRef StrOffsets, StringRef Str) {
  DataExtractor InfoData(Info, true, 0);
  uint64_t Offset = Header.HeaderSize;
  if (Header.Version >= 5 && Header.UnitType != dwarf::DW_UT_split_compile)
    return make_error<DWPError>(
        std::string("unit type DW_UT_split_compile type not found in "
                    "debug_info header. Unexpected unit type 0x" +
                    utostr(Header.UnitType) + " found"));

  CompileUnitIdentifiers ID;

  uint32_t AbbrCode = InfoData.getULEB128(&Offset);
  DataExtractor AbbrevData(Abbrev, true, 0);
  uint64_t AbbrevOffset = getCUAbbrev(Abbrev, AbbrCode);
  auto Tag = static_cast<dwarf::Tag>(AbbrevData.getULEB128(&AbbrevOffset));
  if (Tag != dwarf::DW_TAG_compile_unit)
    return make_error<DWPError>("top level DIE is not a compile unit");
  // DW_CHILDREN
  AbbrevData.getU8(&AbbrevOffset);
  uint32_t Name;
  dwarf::Form Form;
  while ((Name = AbbrevData.getULEB128(&AbbrevOffset)) |
         (Form = static_cast<dwarf::Form>(AbbrevData.getULEB128(&AbbrevOffset))) &&
         (Name != 0 || Form != 0)) {
    switch (Name) {
    case dwarf::DW_AT_name: {
      Expected<const char *> EName = getIndexedString(
          Form, InfoData, Offset, StrOffsets, Str, Header.Version);
      if (!EName)
        return EName.takeError();
      ID.Name = *EName;
      break;
    }
    case dwarf::DW_AT_GNU_dwo_name:
    case dwarf::DW_AT_dwo_name: {
      Expected<const char *> EName = getIndexedString(
          Form, InfoData, Offset, StrOffsets, Str, Header.Version);
      if (!EName)
        return EName.takeError();
      ID.DWOName = *EName;
      break;
    }
    case dwarf::DW_AT_GNU_dwo_id:
      Header.Signature = InfoData.getU64(&Offset);
      break;
    default:
      DWARFFormValue::skipValue(
          Form, InfoData, &Offset,
          dwarf::FormParams({Header.Version, Header.AddrSize, Header.Format}));
    }
  }
  if (!Header.Signature)
    return make_error<DWPError>("compile unit missing dwo_id");
  ID.Signature = *Header.Signature;
  return ID;
}

struct UnitIndexEntry {
  DWARFUnitIndex::Entry::SectionContribution Contributions[8];
  std::string Name;
  std::string DWOName;
  StringRef DWPName;
};

static bool isSupportedSectionKind(DWARFSectionKind Kind) {
  return Kind != DW_SECT_EXT_unknown;
}

// Convert an internal section identifier into the index to use with
// UnitIndexEntry::Contributions.
static unsigned getContributionIndex(DWARFSectionKind Kind,
                                     uint32_t IndexVersion) {
  assert(serializeSectionKind(Kind, IndexVersion) >= DW_SECT_INFO);
  return serializeSectionKind(Kind, IndexVersion) - DW_SECT_INFO;
}

// Convert a UnitIndexEntry::Contributions index to the corresponding on-disk
// value of the section identifier.
static unsigned getOnDiskSectionId(unsigned Index) {
  return Index + DW_SECT_INFO;
}

static StringRef getSubsection(StringRef Section,
                               const DWARFUnitIndex::Entry &Entry,
                               DWARFSectionKind Kind) {
  const auto *Off = Entry.getContribution(Kind);
  if (!Off)
    return StringRef();
  return Section.substr(Off->Offset, Off->Length);
}

static void
addAllTypesFromDWP(MCStreamer &Out,
                   MapVector<uint64_t, UnitIndexEntry> &TypeIndexEntries,
                   const DWARFUnitIndex &TUIndex, MCSection *OutputTypes,
                   StringRef Types, const UnitIndexEntry &TUEntry,
                   uint32_t &TypesOffset, unsigned TypesContributionIndex) {
  Out.SwitchSection(OutputTypes);
  for (const DWARFUnitIndex::Entry &E : TUIndex.getRows()) {
    auto *I = E.getContributions();
    if (!I)
      continue;
    auto P = TypeIndexEntries.insert(std::make_pair(E.getSignature(), TUEntry));
    if (!P.second)
      continue;
    auto &Entry = P.first->second;
    // Zero out the debug_info contribution
    Entry.Contributions[0] = {};
    for (auto Kind : TUIndex.getColumnKinds()) {
      if (!isSupportedSectionKind(Kind))
        continue;
      auto &C =
          Entry.Contributions[getContributionIndex(Kind, TUIndex.getVersion())];
      C.Offset += I->Offset;
      C.Length = I->Length;
      ++I;
    }
    auto &C = Entry.Contributions[TypesContributionIndex];
    Out.emitBytes(Types.substr(
        C.Offset - TUEntry.Contributions[TypesContributionIndex].Offset,
        C.Length));
    C.Offset = TypesOffset;
    TypesOffset += C.Length;
  }
}

static void addAllTypesFromTypesSection(
    MCStreamer &Out, MapVector<uint64_t, UnitIndexEntry> &TypeIndexEntries,
    MCSection *OutputTypes, const std::vector<StringRef> &TypesSections,
    const UnitIndexEntry &CUEntry, uint32_t &TypesOffset) {
  for (StringRef Types : TypesSections) {
    Out.SwitchSection(OutputTypes);
    uint64_t Offset = 0;
    DataExtractor Data(Types, true, 0);
    while (Data.isValidOffset(Offset)) {
      UnitIndexEntry Entry = CUEntry;
      // Zero out the debug_info contribution
      Entry.Contributions[0] = {};
      auto &C = Entry.Contributions[getContributionIndex(DW_SECT_EXT_TYPES, 2)];
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

      Out.emitBytes(Types.substr(PrevOffset, C.Length));
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
        Out.emitIntValue(E.second.Contributions[i].*Field, 4);
}

static void writeIndex(MCStreamer &Out, MCSection *Section,
                       ArrayRef<unsigned> ContributionOffsets,
                       const MapVector<uint64_t, UnitIndexEntry> &IndexEntries,
                       uint32_t IndexVersion) {
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
  Out.emitIntValue(IndexVersion, 4);        // Version
  Out.emitIntValue(Columns, 4);             // Columns
  Out.emitIntValue(IndexEntries.size(), 4); // Num Units
  Out.emitIntValue(Buckets.size(), 4);      // Num Buckets

  // Write the signatures.
  for (const auto &I : Buckets)
    Out.emitIntValue(I ? IndexEntries.begin()[I - 1].first : 0, 8);

  // Write the indexes.
  for (const auto &I : Buckets)
    Out.emitIntValue(I, 4);

  // Write the column headers (which sections will appear in the table)
  for (size_t i = 0; i != ContributionOffsets.size(); ++i)
    if (ContributionOffsets[i])
      Out.emitIntValue(getOnDiskSectionId(i), 4);

  // Write the offsets.
  writeIndexTable(Out, ContributionOffsets, IndexEntries,
                  &DWARFUnitIndex::Entry::SectionContribution::Offset);

  // Write the lengths.
  writeIndexTable(Out, ContributionOffsets, IndexEntries,
                  &DWARFUnitIndex::Entry::SectionContribution::Length);
}

static std::string buildDWODescription(StringRef Name, StringRef DWPName,
                                       StringRef DWOName) {
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
    const MCSection *TUIndexSection, const MCSection *InfoSection,
    const SectionRef &Section, MCStreamer &Out,
    std::deque<SmallString<32>> &UncompressedSections,
    uint32_t (&ContributionOffsets)[8], UnitIndexEntry &CurEntry,
    StringRef &CurStrSection, StringRef &CurStrOffsetSection,
    std::vector<StringRef> &CurTypesSection,
    std::vector<StringRef> &CurInfoSection, StringRef &AbbrevSection,
    StringRef &CurCUIndexSection, StringRef &CurTUIndexSection,
    std::vector<std::pair<DWARFSectionKind, uint32_t>> &SectionLength) {
  if (Section.isBSS())
    return Error::success();

  if (Section.isVirtual())
    return Error::success();

  Expected<StringRef> NameOrErr = Section.getName();
  if (!NameOrErr)
    return NameOrErr.takeError();
  StringRef Name = *NameOrErr;

  Expected<StringRef> ContentsOrErr = Section.getContents();
  if (!ContentsOrErr)
    return ContentsOrErr.takeError();
  StringRef Contents = *ContentsOrErr;

  if (auto Err = handleCompressedSection(UncompressedSections, Name, Contents))
    return Err;

  Name = Name.substr(Name.find_first_not_of("._"));

  auto SectionPair = KnownSections.find(Name);
  if (SectionPair == KnownSections.end())
    return Error::success();

  if (DWARFSectionKind Kind = SectionPair->second.second) {
    if (Kind != DW_SECT_EXT_TYPES && Kind != DW_SECT_INFO) {
      SectionLength.push_back(std::make_pair(Kind, Contents.size()));
    }

    if (Kind == DW_SECT_ABBREV) {
      AbbrevSection = Contents;
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
  else if (OutSection == InfoSection)
    CurInfoSection.push_back(Contents);
  else {
    Out.SwitchSection(OutSection);
    Out.emitBytes(Contents);
  }
  return Error::success();
}

static Error
buildDuplicateError(const std::pair<uint64_t, UnitIndexEntry> &PrevE,
                    const CompileUnitIdentifiers &ID, StringRef DWPName) {
  return make_error<DWPError>(
      std::string("duplicate DWO ID (") + utohexstr(PrevE.first) + ") in " +
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
      SmallString<16> DWOPath(std::move(DWOName));
      sys::fs::make_absolute(DWOCompDir, DWOPath);
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
  MCSection *const InfoSection = MCOFI.getDwarfInfoDWOSection();
  const StringMap<std::pair<MCSection *, DWARFSectionKind>> KnownSections = {
      {"debug_info.dwo", {InfoSection, DW_SECT_INFO}},
      {"debug_types.dwo", {MCOFI.getDwarfTypesDWOSection(), DW_SECT_EXT_TYPES}},
      {"debug_str_offsets.dwo", {StrOffsetSection, DW_SECT_STR_OFFSETS}},
      {"debug_str.dwo", {StrSection, static_cast<DWARFSectionKind>(0)}},
      {"debug_loc.dwo", {MCOFI.getDwarfLocDWOSection(), DW_SECT_EXT_LOC}},
      {"debug_line.dwo", {MCOFI.getDwarfLineDWOSection(), DW_SECT_LINE}},
      {"debug_macro.dwo", {MCOFI.getDwarfMacroDWOSection(), DW_SECT_MACRO}},
      {"debug_abbrev.dwo", {MCOFI.getDwarfAbbrevDWOSection(), DW_SECT_ABBREV}},
      {"debug_loclists.dwo",
       {MCOFI.getDwarfLoclistsDWOSection(), DW_SECT_LOCLISTS}},
      {"debug_rnglists.dwo",
       {MCOFI.getDwarfRnglistsDWOSection(), DW_SECT_RNGLISTS}},
      {"debug_cu_index", {CUIndexSection, static_cast<DWARFSectionKind>(0)}},
      {"debug_tu_index", {TUIndexSection, static_cast<DWARFSectionKind>(0)}}};

  MapVector<uint64_t, UnitIndexEntry> IndexEntries;
  MapVector<uint64_t, UnitIndexEntry> TypeIndexEntries;

  uint32_t ContributionOffsets[8] = {};
  uint16_t Version = 0;
  uint32_t IndexVersion = 0;

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
    std::vector<StringRef> CurInfoSection;
    StringRef AbbrevSection;
    StringRef CurCUIndexSection;
    StringRef CurTUIndexSection;

    // This maps each section contained in this file to its length.
    // This information is later on used to calculate the contributions,
    // i.e. offset and length, of each compile/type unit to a section.
    std::vector<std::pair<DWARFSectionKind, uint32_t>> SectionLength;

    for (const auto &Section : Obj.sections())
      if (auto Err = handleSection(
              KnownSections, StrSection, StrOffsetSection, TypesSection,
              CUIndexSection, TUIndexSection, InfoSection, Section, Out,
              UncompressedSections, ContributionOffsets, CurEntry,
              CurStrSection, CurStrOffsetSection, CurTypesSection,
              CurInfoSection, AbbrevSection, CurCUIndexSection,
              CurTUIndexSection, SectionLength))
        return Err;

    if (CurInfoSection.empty())
      continue;

    Expected<InfoSectionUnitHeader> HeaderOrErr =
        parseInfoSectionUnitHeader(CurInfoSection.front());
    if (!HeaderOrErr)
      return HeaderOrErr.takeError();
    InfoSectionUnitHeader &Header = *HeaderOrErr;

    if (Version == 0) {
      Version = Header.Version;
      IndexVersion = Version < 5 ? 2 : 5;
    } else if (Version != Header.Version) {
      return make_error<DWPError>("incompatible DWARF compile unit versions.");
    }

    writeStringsAndOffsets(Out, Strings, StrOffsetSection, CurStrSection,
                           CurStrOffsetSection, Header.Version);

    for (auto Pair : SectionLength) {
      auto Index = getContributionIndex(Pair.first, IndexVersion);
      CurEntry.Contributions[Index].Offset = ContributionOffsets[Index];
      ContributionOffsets[Index] +=
          (CurEntry.Contributions[Index].Length = Pair.second);
    }

    uint32_t &InfoSectionOffset =
        ContributionOffsets[getContributionIndex(DW_SECT_INFO, IndexVersion)];
    if (CurCUIndexSection.empty()) {
      bool FoundCUUnit = false;
      Out.SwitchSection(InfoSection);
      for (StringRef Info : CurInfoSection) {
        uint64_t UnitOffset = 0;
        while (Info.size() > UnitOffset) {
          Expected<InfoSectionUnitHeader> HeaderOrError =
              parseInfoSectionUnitHeader(Info.substr(UnitOffset, Info.size()));
          if (!HeaderOrError)
            return HeaderOrError.takeError();
          InfoSectionUnitHeader &Header = *HeaderOrError;

          UnitIndexEntry Entry = CurEntry;
          auto &C = Entry.Contributions[getContributionIndex(DW_SECT_INFO,
                                                             IndexVersion)];
          C.Offset = InfoSectionOffset;
          C.Length = Header.Length + 4;
          UnitOffset += C.Length;
          if (Header.Version < 5 ||
              Header.UnitType == dwarf::DW_UT_split_compile) {
            Expected<CompileUnitIdentifiers> EID =
                getCUIdentifiers(Header, AbbrevSection,
                                 Info.substr(UnitOffset - C.Length, C.Length),
                                 CurStrOffsetSection, CurStrSection);

            if (!EID)
              return createFileError(Input, EID.takeError());
            const auto &ID = *EID;
            auto P = IndexEntries.insert(std::make_pair(ID.Signature, Entry));
            if (!P.second)
              return buildDuplicateError(*P.first, ID, "");
            P.first->second.Name = ID.Name;
            P.first->second.DWOName = ID.DWOName;

            FoundCUUnit = true;
          } else if (Header.UnitType == dwarf::DW_UT_split_type) {
            auto P = TypeIndexEntries.insert(
                std::make_pair(Header.Signature.getValue(), Entry));
            if (!P.second)
              continue;
          }
          Out.emitBytes(Info.substr(UnitOffset - C.Length, C.Length));
          InfoSectionOffset += C.Length;
        }
      }

      if (!FoundCUUnit)
        return make_error<DWPError>("no compile unit found in file: " + Input);

      if (IndexVersion == 2) {
        // Add types from the .debug_types section from DWARF < 5.
        addAllTypesFromTypesSection(
            Out, TypeIndexEntries, TypesSection, CurTypesSection, CurEntry,
            ContributionOffsets[getContributionIndex(DW_SECT_EXT_TYPES, 2)]);
      }
      continue;
    }

    if (CurInfoSection.size() != 1)
      return make_error<DWPError>("expected exactly one occurrence of a debug "
                                  "info section in a .dwp file");
    StringRef DwpSingleInfoSection = CurInfoSection.front();

    DWARFUnitIndex CUIndex(DW_SECT_INFO);
    DataExtractor CUIndexData(CurCUIndexSection, Obj.isLittleEndian(), 0);
    if (!CUIndex.parse(CUIndexData))
      return make_error<DWPError>("failed to parse cu_index");
    if (CUIndex.getVersion() != IndexVersion)
      return make_error<DWPError>("incompatible cu_index versions, found " +
                                  utostr(CUIndex.getVersion()) +
                                  " and expecting " + utostr(IndexVersion));

    Out.SwitchSection(InfoSection);
    for (const DWARFUnitIndex::Entry &E : CUIndex.getRows()) {
      auto *I = E.getContributions();
      if (!I)
        continue;
      auto P = IndexEntries.insert(std::make_pair(E.getSignature(), CurEntry));
      StringRef CUInfoSection =
          getSubsection(DwpSingleInfoSection, E, DW_SECT_INFO);
      Expected<InfoSectionUnitHeader> HeaderOrError =
          parseInfoSectionUnitHeader(CUInfoSection);
      if (!HeaderOrError)
        return HeaderOrError.takeError();
      InfoSectionUnitHeader &Header = *HeaderOrError;

      Expected<CompileUnitIdentifiers> EID = getCUIdentifiers(
          Header, getSubsection(AbbrevSection, E, DW_SECT_ABBREV),
          CUInfoSection,
          getSubsection(CurStrOffsetSection, E, DW_SECT_STR_OFFSETS),
          CurStrSection);
      if (!EID)
        return createFileError(Input, EID.takeError());
      const auto &ID = *EID;
      if (!P.second)
        return buildDuplicateError(*P.first, ID, Input);
      auto &NewEntry = P.first->second;
      NewEntry.Name = ID.Name;
      NewEntry.DWOName = ID.DWOName;
      NewEntry.DWPName = Input;
      for (auto Kind : CUIndex.getColumnKinds()) {
        if (!isSupportedSectionKind(Kind))
          continue;
        auto &C =
            NewEntry.Contributions[getContributionIndex(Kind, IndexVersion)];
        C.Offset += I->Offset;
        C.Length = I->Length;
        ++I;
      }
      unsigned Index = getContributionIndex(DW_SECT_INFO, IndexVersion);
      auto &C = NewEntry.Contributions[Index];
      Out.emitBytes(CUInfoSection);
      C.Offset = InfoSectionOffset;
      InfoSectionOffset += C.Length;
    }

    if (!CurTUIndexSection.empty()) {
      llvm::DWARFSectionKind TUSectionKind;
      MCSection *OutSection;
      StringRef TypeInputSection;
      // Write type units into debug info section for DWARFv5.
      if (Version >= 5) {
        TUSectionKind = DW_SECT_INFO;
        OutSection = InfoSection;
        TypeInputSection = DwpSingleInfoSection;
      } else {
        // Write type units into debug types section for DWARF < 5.
        if (CurTypesSection.size() != 1)
          return make_error<DWPError>(
              "multiple type unit sections in .dwp file");

        TUSectionKind = DW_SECT_EXT_TYPES;
        OutSection = TypesSection;
        TypeInputSection = CurTypesSection.front();
      }

      DWARFUnitIndex TUIndex(TUSectionKind);
      DataExtractor TUIndexData(CurTUIndexSection, Obj.isLittleEndian(), 0);
      if (!TUIndex.parse(TUIndexData))
        return make_error<DWPError>("failed to parse tu_index");
      if (TUIndex.getVersion() != IndexVersion)
        return make_error<DWPError>("incompatible tu_index versions, found " +
                                    utostr(TUIndex.getVersion()) +
                                    " and expecting " + utostr(IndexVersion));

      unsigned TypesContributionIndex =
          getContributionIndex(TUSectionKind, IndexVersion);
      addAllTypesFromDWP(Out, TypeIndexEntries, TUIndex, OutSection,
                         TypeInputSection, CurEntry,
                         ContributionOffsets[TypesContributionIndex],
                         TypesContributionIndex);
    }
  }

  if (Version < 5) {
    // Lie about there being no info contributions so the TU index only includes
    // the type unit contribution for DWARF < 5. In DWARFv5 the TU index has a
    // contribution to the info section, so we do not want to lie about it.
    ContributionOffsets[0] = 0;
  }
  writeIndex(Out, MCOFI.getDwarfTUIndexSection(), ContributionOffsets,
             TypeIndexEntries, IndexVersion);

  if (Version < 5) {
    // Lie about the type contribution for DWARF < 5. In DWARFv5 the type
    // section does not exist, so no need to do anything about this.
    ContributionOffsets[getContributionIndex(DW_SECT_EXT_TYPES, 2)] = 0;
    // Unlie about the info contribution
    ContributionOffsets[0] = 1;
  }

  writeIndex(Out, MCOFI.getDwarfCUIndexSection(), ContributionOffsets,
             IndexEntries, IndexVersion);

  return Error::success();
}

static int error(const Twine &Error, const Twine &Context) {
  errs() << Twine("while processing ") + Context + ":\n";
  errs() << Twine("error: ") + Error + "\n";
  return 1;
}

static Expected<Triple> readTargetTriple(StringRef FileName) {
  auto ErrOrObj = object::ObjectFile::createObjectFile(FileName);
  if (!ErrOrObj)
    return ErrOrObj.takeError();

  return ErrOrObj->getBinary()->makeTriple();
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  cl::ParseCommandLineOptions(argc, argv, "merge split dwarf (.dwo) files\n");

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();

  std::vector<std::string> DWOFilenames = InputFiles;
  for (const auto &ExecFilename : ExecFilenames) {
    auto DWOs = getDWOFilenames(ExecFilename);
    if (!DWOs) {
      logAllUnhandledErrors(DWOs.takeError(), WithColor::error());
      return 1;
    }
    DWOFilenames.insert(DWOFilenames.end(),
                        std::make_move_iterator(DWOs->begin()),
                        std::make_move_iterator(DWOs->end()));
  }

  if (DWOFilenames.empty())
    return 0;

  std::string ErrorStr;
  StringRef Context = "dwarf streamer init";

  auto ErrOrTriple = readTargetTriple(DWOFilenames.front());
  if (!ErrOrTriple) {
    logAllUnhandledErrors(ErrOrTriple.takeError(), WithColor::error());
    return 1;
  }

  // Get the target.
  const Target *TheTarget =
      TargetRegistry::lookupTarget("", *ErrOrTriple, ErrorStr);
  if (!TheTarget)
    return error(ErrorStr, Context);
  std::string TripleName = ErrOrTriple->getTriple();

  // Create all the MC Objects.
  std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TripleName));
  if (!MRI)
    return error(Twine("no register info for target ") + TripleName, Context);

  MCTargetOptions MCOptions = llvm::mc::InitMCTargetOptionsFromFlags();
  std::unique_ptr<MCAsmInfo> MAI(
      TheTarget->createMCAsmInfo(*MRI, TripleName, MCOptions));
  if (!MAI)
    return error("no asm info for target " + TripleName, Context);

  std::unique_ptr<MCSubtargetInfo> MSTI(
      TheTarget->createMCSubtargetInfo(TripleName, "", ""));
  if (!MSTI)
    return error("no subtarget info for target " + TripleName, Context);

  MCContext MC(*ErrOrTriple, MAI.get(), MRI.get(), MSTI.get());
  std::unique_ptr<MCObjectFileInfo> MOFI(
      TheTarget->createMCObjectFileInfo(MC, /*PIC=*/false));
  MC.setObjectFileInfo(MOFI.get());

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
  ToolOutputFile OutFile(OutputFilename, EC, sys::fs::OF_None);
  Optional<buffer_ostream> BOS;
  raw_pwrite_stream *OS;
  if (EC)
    return error(Twine(OutputFilename) + ": " + EC.message(), Context);
  if (OutFile.os().supportsSeeking()) {
    OS = &OutFile.os();
  } else {
    BOS.emplace(OutFile.os());
    OS = BOS.getPointer();
  }

  std::unique_ptr<MCStreamer> MS(TheTarget->createMCObjectStreamer(
      *ErrOrTriple, MC, std::unique_ptr<MCAsmBackend>(MAB),
      MAB->createObjectWriter(*OS), std::unique_ptr<MCCodeEmitter>(MCE), *MSTI,
      MCOptions.MCRelaxAll, MCOptions.MCIncrementalLinkerCompatible,
      /*DWARFMustBeAtTheEnd*/ false));
  if (!MS)
    return error("no object streamer for target " + TripleName, Context);

  if (auto Err = write(*MS, DWOFilenames)) {
    logAllUnhandledErrors(std::move(Err), WithColor::error());
    return 1;
  }

  MS->Finish();
  OutFile.keep();
  return 0;
}
