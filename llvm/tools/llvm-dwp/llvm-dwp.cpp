//===-- llvm-dwp.cpp - Split DWARF merging tool for llvm ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A utility for merging DWARF 5 Split DWARF .dwo files into .dwp (DWARF
// package files).
//
//===----------------------------------------------------------------------===//
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFUnitIndex.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Options.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <list>
#include <iostream>
#include <memory>
#include <system_error>
#include <unordered_set>

using namespace llvm;
using namespace llvm::object;
using namespace cl;

OptionCategory DwpCategory("Specific Options");
static list<std::string> InputFiles(Positional, OneOrMore,
                                    desc("<input files>"), cat(DwpCategory));

static opt<std::string> OutputFilename(Required, "o",
                                       desc("Specify the output file."),
                                       value_desc("filename"),
                                       cat(DwpCategory));

static int error(const Twine &Error, const Twine &Context) {
  errs() << Twine("while processing ") + Context + ":\n";
  errs() << Twine("error: ") + Error + "\n";
  return 1;
}

static std::error_code
writeStringsAndOffsets(MCStreamer &Out, StringMap<uint32_t> &Strings,
                       uint32_t &StringOffset, MCSection *StrSection,
                       MCSection *StrOffsetSection, StringRef CurStrSection,
                       StringRef CurStrOffsetSection) {
  // Could possibly produce an error or warning if one of these was non-null but
  // the other was null.
  if (CurStrSection.empty() || CurStrOffsetSection.empty())
    return std::error_code();

  DenseMap<uint32_t, uint32_t> OffsetRemapping;

  DataExtractor Data(CurStrSection, true, 0);
  uint32_t LocalOffset = 0;
  uint32_t PrevOffset = 0;
  while (const char *s = Data.getCStr(&LocalOffset)) {
    StringRef Str(s, LocalOffset - PrevOffset - 1);
    auto Pair = Strings.insert(std::make_pair(Str, StringOffset));
    if (Pair.second) {
      Out.SwitchSection(StrSection);
      Out.EmitBytes(
          StringRef(Pair.first->getKeyData(), Pair.first->getKeyLength() + 1));
      StringOffset += Str.size() + 1;
    }
    OffsetRemapping[PrevOffset] = Pair.first->second;
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

  return std::error_code();
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

static const char *getIndexedString(uint32_t Form, DataExtractor InfoData,
                                    uint32_t &InfoOffset, StringRef StrOffsets,
                                    StringRef Str) {
  if (Form == dwarf::DW_FORM_string)
    return InfoData.getCStr(&InfoOffset);
  assert(Form == dwarf::DW_FORM_GNU_str_index);
  auto StrIndex = InfoData.getULEB128(&InfoOffset);
  DataExtractor StrOffsetsData(StrOffsets, true, 0);
  uint32_t StrOffsetsOffset = 4 * StrIndex;
  uint32_t StrOffset = StrOffsetsData.getU32(&StrOffsetsOffset);
  DataExtractor StrData(Str, true, 0);
  return StrData.getCStr(&StrOffset);
}

static CompileUnitIdentifiers getCUIdentifiers(StringRef Abbrev, StringRef Info,
                                               StringRef StrOffsets,
                                               StringRef Str) {
  uint32_t Offset = 0;
  DataExtractor InfoData(Info, true, 0);
  InfoData.getU32(&Offset); // Length
  uint16_t Version = InfoData.getU16(&Offset);
  InfoData.getU32(&Offset); // Abbrev offset (should be zero)
  uint8_t AddrSize = InfoData.getU8(&Offset);

  uint32_t AbbrCode = InfoData.getULEB128(&Offset);

  DataExtractor AbbrevData(Abbrev, true, 0);
  uint32_t AbbrevOffset = getCUAbbrev(Abbrev, AbbrCode);
  uint64_t Tag = AbbrevData.getULEB128(&AbbrevOffset);
  (void)Tag;
  // FIXME: Real error handling
  assert(Tag == dwarf::DW_TAG_compile_unit);
  // DW_CHILDREN
  AbbrevData.getU8(&AbbrevOffset);
  uint32_t Name;
  uint32_t Form;
  CompileUnitIdentifiers ID;
  while ((Name = AbbrevData.getULEB128(&AbbrevOffset)) |
             (Form = AbbrevData.getULEB128(&AbbrevOffset)) &&
         (Name != 0 || Form != 0)) {
    switch (Name) {
    case dwarf::DW_AT_name: {
      ID.Name = getIndexedString(Form, InfoData, Offset, StrOffsets, Str);
      break;
    }
    case dwarf::DW_AT_GNU_dwo_name: {
      ID.DWOName = getIndexedString(Form, InfoData, Offset, StrOffsets, Str);
      break;
    }
    case dwarf::DW_AT_GNU_dwo_id:
      ID.Signature = InfoData.getU64(&Offset);
      break;
    default:
      DWARFFormValue::skipValue(Form, InfoData, &Offset, Version, AddrSize);
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

StringRef getSubsection(StringRef Section, const DWARFUnitIndex::Entry &Entry, DWARFSectionKind Kind) {
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
static bool consumeCompressedDebugSectionHeader(StringRef &data,
                                                uint64_t &OriginalSize) {
  // Consume "ZLIB" prefix.
  if (!data.startswith("ZLIB"))
    return false;
  data = data.substr(4);
  // Consume uncompressed section size (big-endian 8 bytes).
  DataExtractor extractor(data, false, 8);
  uint32_t Offset = 0;
  OriginalSize = extractor.getU64(&Offset);
  if (Offset == 0)
    return false;
  data = data.substr(Offset);
  return true;
}

void printDuplicateError(const std::pair<uint64_t, UnitIndexEntry> &PrevE,
                         const CompileUnitIdentifiers &ID, StringRef DWPName) {
  errs() << "Duplicate DWO ID (" << PrevE.first << ") in '" << PrevE.second.Name
         << '\'';
  if (!PrevE.second.DWPName.empty()) {
    errs() << " (from ";
    if (!PrevE.second.DWOName.empty())
      errs() << '\'' << PrevE.second.DWOName << "' in ";
    errs() << "'" << PrevE.second.DWPName.str() << "')";
  }
  errs() << " and '" << ID.Name << '\'';
  if (!DWPName.empty()) {
    errs() << " (from ";
    if (*ID.DWOName)
      errs() << '\'' << ID.DWOName << "\' in ";
    errs() << '\'' << DWPName << "')";
  }
  errs() << '\n';
}
static std::error_code write(MCStreamer &Out, ArrayRef<std::string> Inputs) {
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

  StringMap<uint32_t> Strings;
  uint32_t StringOffset = 0;

  uint32_t ContributionOffsets[8] = {};

  for (const auto &Input : Inputs) {
    auto ErrOrObj = object::ObjectFile::createObjectFile(Input);
    if (!ErrOrObj)
      return ErrOrObj.getError();

    UnitIndexEntry CurEntry = {};

    StringRef CurStrSection;
    StringRef CurStrOffsetSection;
    std::vector<StringRef> CurTypesSection;
    StringRef InfoSection;
    StringRef AbbrevSection;
    StringRef CurCUIndexSection;
    StringRef CurTUIndexSection;

    SmallVector<SmallString<32>, 4> UncompressedSections;

    for (const auto &Section : ErrOrObj->getBinary()->sections()) {
      if (Section.isBSS())
        continue;
      if (Section.isVirtual())
        continue;

      StringRef Name;
      if (std::error_code Err = Section.getName(Name))
        return Err;

      Name = Name.substr(Name.find_first_not_of("._"));

      StringRef Contents;
      if (auto Err = Section.getContents(Contents))
        return Err;

      if (Name.startswith("zdebug_")) {
        uint64_t OriginalSize;
        if (!zlib::isAvailable() ||
            !consumeCompressedDebugSectionHeader(Contents, OriginalSize))
          return make_error_code(std::errc::invalid_argument);
        UncompressedSections.resize(UncompressedSections.size() + 1);
        if (zlib::uncompress(Contents, UncompressedSections.back(), OriginalSize) !=
            zlib::StatusOK) {
          UncompressedSections.pop_back();
          continue;
        }
        Name = Name.substr(1);
        Contents = UncompressedSections.back();
      }

      auto SectionPair = KnownSections.find(Name);
      if (SectionPair == KnownSections.end())
        continue;

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
    }

    if (InfoSection.empty())
      continue;

    if (!CurCUIndexSection.empty()) {
      DWARFUnitIndex CUIndex(DW_SECT_INFO);
      DataExtractor CUIndexData(CurCUIndexSection,
                                ErrOrObj->getBinary()->isLittleEndian(), 0);
      if (!CUIndex.parse(CUIndexData))
        return make_error_code(std::errc::invalid_argument);

      for (const DWARFUnitIndex::Entry &E : CUIndex.getRows()) {
        auto *I = E.getOffsets();
        if (!I)
          continue;
        auto P =
            IndexEntries.insert(std::make_pair(E.getSignature(), CurEntry));
        CompileUnitIdentifiers ID = getCUIdentifiers(
            getSubsection(AbbrevSection, E, DW_SECT_ABBREV),
            getSubsection(InfoSection, E, DW_SECT_INFO),
            getSubsection(CurStrOffsetSection, E, DW_SECT_STR_OFFSETS),
            CurStrSection);
        if (!P.second) {
          printDuplicateError(*P.first, ID, Input);
          return make_error_code(std::errc::invalid_argument);
        }
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
        assert(CurTypesSection.size() == 1);
        if (CurTUIndexSection.empty())
          return make_error_code(std::errc::invalid_argument);
        DWARFUnitIndex TUIndex(DW_SECT_TYPES);
        DataExtractor TUIndexData(CurTUIndexSection,
                                  ErrOrObj->getBinary()->isLittleEndian(), 0);
        if (!TUIndex.parse(TUIndexData))
          return make_error_code(std::errc::invalid_argument);
        addAllTypesFromDWP(Out, TypeIndexEntries, TUIndex, TypesSection,
                           CurTypesSection.front(), CurEntry,
                           ContributionOffsets[DW_SECT_TYPES - DW_SECT_INFO]);
      }
    } else {
      CompileUnitIdentifiers ID = getCUIdentifiers(
          AbbrevSection, InfoSection, CurStrOffsetSection, CurStrSection);
      auto P = IndexEntries.insert(std::make_pair(ID.Signature, CurEntry));
      if (!P.second) {
        printDuplicateError(*P.first, ID, "");
        return make_error_code(std::errc::invalid_argument);
      }
      P.first->second.Name = ID.Name;
      P.first->second.DWOName = ID.DWOName;
      addAllTypes(Out, TypeIndexEntries, TypesSection, CurTypesSection,
                  CurEntry, ContributionOffsets[DW_SECT_TYPES - DW_SECT_INFO]);
    }

    if (auto Err = writeStringsAndOffsets(Out, Strings, StringOffset,
                                          StrSection, StrOffsetSection,
                                          CurStrSection, CurStrOffsetSection))
      return Err;
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

  return std::error_code();
}

int main(int argc, char **argv) {

  ParseCommandLineOptions(argc, argv, "merge split dwarf (.dwo) files");

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
  MOFI.InitMCObjectFileInfo(TheTriple, Reloc::Default, CodeModel::Default, MC);

  auto MAB = TheTarget->createMCAsmBackend(*MRI, TripleName, "");
  if (!MAB)
    return error("no asm backend for target " + TripleName, Context);

  std::unique_ptr<MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  if (!MII)
    return error("no instr info info for target " + TripleName, Context);

  std::unique_ptr<MCSubtargetInfo> MSTI(
      TheTarget->createMCSubtargetInfo(TripleName, "", ""));
  if (!MSTI)
    return error("no subtarget info for target " + TripleName, Context);

  MCCodeEmitter *MCE = TheTarget->createMCCodeEmitter(*MII, *MRI, MC);
  if (!MCE)
    return error("no code emitter for target " + TripleName, Context);

  // Create the output file.
  std::error_code EC;
  raw_fd_ostream OutFile(OutputFilename, EC, sys::fs::F_None);
  if (EC)
    return error(Twine(OutputFilename) + ": " + EC.message(), Context);

  MCTargetOptions MCOptions = InitMCTargetOptionsFromFlags();
  std::unique_ptr<MCStreamer> MS(TheTarget->createMCObjectStreamer(
      TheTriple, MC, *MAB, OutFile, MCE, *MSTI, MCOptions.MCRelaxAll,
      MCOptions.MCIncrementalLinkerCompatible,
      /*DWARFMustBeAtTheEnd*/ false));
  if (!MS)
    return error("no object streamer for target " + TripleName, Context);

  if (auto Err = write(*MS, InputFiles))
    return error(Err.message(), "Writing DWP file");

  MS->Finish();
}
