//===- yaml2xcoff - Convert YAML to a xcoff object file -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The xcoff component of yaml2obj.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/BinaryFormat/XCOFF.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/XCOFFObjectFile.h"
#include "llvm/ObjectYAML/ObjectYAML.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

constexpr unsigned DefaultSectionAlign = 4;
constexpr int16_t MaxSectionIndex = INT16_MAX;
constexpr uint32_t MaxRawDataSize = UINT32_MAX;

class XCOFFWriter {
public:
  XCOFFWriter(XCOFFYAML::Object &Obj, raw_ostream &OS, yaml::ErrorHandler EH)
      : Obj(Obj), W(OS, support::big), ErrHandler(EH),
        StrTblBuilder(StringTableBuilder::XCOFF) {
    Is64Bit = Obj.Header.Magic == (llvm::yaml::Hex16)XCOFF::XCOFF64;
  }
  bool writeXCOFF();

private:
  bool nameShouldBeInStringTable(StringRef SymbolName);
  bool initFileHeader(uint64_t CurrentOffset);
  void initAuxFileHeader();
  bool initSectionHeader(uint64_t &CurrentOffset);
  bool initRelocations(uint64_t &CurrentOffset);
  bool initStringTable();
  bool assignAddressesAndIndices();

  void writeFileHeader();
  void writeAuxFileHeader();
  void writeSectionHeader();
  bool writeSectionData();
  bool writeRelocations();
  bool writeSymbols();
  void writeStringTable();

  void writeAuxSymbol(const XCOFFYAML::CsectAuxEnt &AuxSym);
  void writeAuxSymbol(const XCOFFYAML::FileAuxEnt &AuxSym);
  void writeAuxSymbol(const XCOFFYAML::FunctionAuxEnt &AuxSym);
  void writeAuxSymbol(const XCOFFYAML::ExcpetionAuxEnt &AuxSym);
  void writeAuxSymbol(const XCOFFYAML::BlockAuxEnt &AuxSym);
  void writeAuxSymbol(const XCOFFYAML::SectAuxEntForDWARF &AuxSym);
  void writeAuxSymbol(const XCOFFYAML::SectAuxEntForStat &AuxSym);
  void writeAuxSymbol(const std::unique_ptr<XCOFFYAML::AuxSymbolEnt> &AuxSym);

  XCOFFYAML::Object &Obj;
  bool Is64Bit = false;
  support::endian::Writer W;
  yaml::ErrorHandler ErrHandler;
  StringTableBuilder StrTblBuilder;
  uint64_t StartOffset;
  // Map the section name to its corrresponding section index.
  DenseMap<StringRef, int16_t> SectionIndexMap = {
      {StringRef("N_DEBUG"), XCOFF::N_DEBUG},
      {StringRef("N_ABS"), XCOFF::N_ABS},
      {StringRef("N_UNDEF"), XCOFF::N_UNDEF}};
  XCOFFYAML::FileHeader InitFileHdr = Obj.Header;
  XCOFFYAML::AuxiliaryHeader InitAuxFileHdr;
  std::vector<XCOFFYAML::Section> InitSections = Obj.Sections;
};

static void writeName(StringRef StrName, support::endian::Writer W) {
  char Name[XCOFF::NameSize];
  memset(Name, 0, XCOFF::NameSize);
  char SrcName[] = "";
  memcpy(Name, StrName.size() ? StrName.data() : SrcName, StrName.size());
  ArrayRef<char> NameRef(Name, XCOFF::NameSize);
  W.write(NameRef);
}

bool XCOFFWriter::nameShouldBeInStringTable(StringRef SymbolName) {
  // For XCOFF64: The symbol name is always in the string table.
  return (SymbolName.size() > XCOFF::NameSize) || Is64Bit;
}

bool XCOFFWriter::initRelocations(uint64_t &CurrentOffset) {
  for (XCOFFYAML::Section &InitSection : InitSections) {
    if (!InitSection.Relocations.empty()) {
      InitSection.NumberOfRelocations = InitSection.Relocations.size();
      InitSection.FileOffsetToRelocations = CurrentOffset;
      uint64_t RelSize = Is64Bit ? XCOFF::RelocationSerializationSize64
                                 : XCOFF::RelocationSerializationSize32;
      CurrentOffset += InitSection.NumberOfRelocations * RelSize;
      if (CurrentOffset > MaxRawDataSize) {
        ErrHandler("maximum object size of" + Twine(MaxRawDataSize) +
                   "exceeded when writing relocation data");
        return false;
      }
    }
  }
  return true;
}

bool XCOFFWriter::initSectionHeader(uint64_t &CurrentOffset) {
  uint64_t CurrentSecAddr = 0;
  for (uint16_t I = 0, E = InitSections.size(); I < E; ++I) {
    if (CurrentOffset > MaxRawDataSize) {
      ErrHandler("maximum object size of" + Twine(MaxRawDataSize) +
                 "exceeded when writing section data");
      return false;
    }

    // Assign indices for sections.
    if (InitSections[I].SectionName.size() &&
        !SectionIndexMap[InitSections[I].SectionName]) {
      // The section index starts from 1.
      SectionIndexMap[InitSections[I].SectionName] = I + 1;
      if ((I + 1) > MaxSectionIndex) {
        ErrHandler("exceeded the maximum permitted section index of " +
                   Twine(MaxSectionIndex));
        return false;
      }
    }

    // Calculate the physical/virtual address. This field should contain 0 for
    // all sections except the text, data and bss sections.
    if (InitSections[I].Flags != XCOFF::STYP_TEXT &&
        InitSections[I].Flags != XCOFF::STYP_DATA &&
        InitSections[I].Flags != XCOFF::STYP_BSS)
      InitSections[I].Address = 0;
    else
      InitSections[I].Address = CurrentSecAddr;

    // Calculate the FileOffsetToData and data size for sections.
    if (InitSections[I].SectionData.binary_size()) {
      InitSections[I].FileOffsetToData = CurrentOffset;
      CurrentOffset += InitSections[I].SectionData.binary_size();
      // Ensure the offset is aligned to DefaultSectionAlign.
      CurrentOffset = alignTo(CurrentOffset, DefaultSectionAlign);
      InitSections[I].Size = CurrentOffset - InitSections[I].FileOffsetToData;
      CurrentSecAddr += InitSections[I].Size;
    }
  }
  return initRelocations(CurrentOffset);
}

bool XCOFFWriter::initStringTable() {
  if (Obj.StrTbl.RawContent) {
    size_t RawSize = Obj.StrTbl.RawContent->binary_size();
    if (Obj.StrTbl.Strings || Obj.StrTbl.Length) {
      ErrHandler(
          "can't specify Strings or Length when RawContent is specified");
      return false;
    }
    if (Obj.StrTbl.ContentSize && *Obj.StrTbl.ContentSize < RawSize) {
      ErrHandler("specified ContentSize (" + Twine(*Obj.StrTbl.ContentSize) +
                 ") is less than the RawContent data size (" + Twine(RawSize) +
                 ")");
      return false;
    }
    return true;
  }
  if (Obj.StrTbl.ContentSize && *Obj.StrTbl.ContentSize <= 3) {
    ErrHandler("ContentSize shouldn't be less than 4 without RawContent");
    return false;
  }

  // Build the string table.
  StrTblBuilder.clear();

  if (Obj.StrTbl.Strings) {
    // All specified strings should be added to the string table.
    for (StringRef StringEnt : *Obj.StrTbl.Strings)
      StrTblBuilder.add(StringEnt);

    size_t StrTblIdx = 0;
    size_t NumOfStrings = Obj.StrTbl.Strings->size();
    for (XCOFFYAML::Symbol &YamlSym : Obj.Symbols) {
      if (nameShouldBeInStringTable(YamlSym.SymbolName)) {
        if (StrTblIdx < NumOfStrings) {
          // Overwrite the symbol name with the specified string.
          YamlSym.SymbolName = (*Obj.StrTbl.Strings)[StrTblIdx];
          ++StrTblIdx;
        } else
          // Names that are not overwritten are still stored in the string
          // table.
          StrTblBuilder.add(YamlSym.SymbolName);
      }
    }
  } else {
    for (const XCOFFYAML::Symbol &YamlSym : Obj.Symbols) {
      if (nameShouldBeInStringTable(YamlSym.SymbolName))
        StrTblBuilder.add(YamlSym.SymbolName);
    }
  }

  // Check if the file name in the File Auxiliary Entry should be added to the
  // string table.
  for (const XCOFFYAML::Symbol &YamlSym : Obj.Symbols) {
    for (const std::unique_ptr<XCOFFYAML::AuxSymbolEnt> &AuxSym :
         YamlSym.AuxEntries) {
      if (auto AS = dyn_cast<XCOFFYAML::FileAuxEnt>(AuxSym.get()))
        if (nameShouldBeInStringTable(AS->FileNameOrString.getValueOr("")))
          StrTblBuilder.add(AS->FileNameOrString.getValueOr(""));
    }
  }

  StrTblBuilder.finalize();

  size_t StrTblSize = StrTblBuilder.getSize();
  if (Obj.StrTbl.ContentSize && *Obj.StrTbl.ContentSize < StrTblSize) {
    ErrHandler("specified ContentSize (" + Twine(*Obj.StrTbl.ContentSize) +
               ") is less than the size of the data that would otherwise be "
               "written (" +
               Twine(StrTblSize) + ")");
    return false;
  }

  return true;
}

bool XCOFFWriter::initFileHeader(uint64_t CurrentOffset) {
  // The default format of the object file is XCOFF32.
  InitFileHdr.Magic = XCOFF::XCOFF32;
  InitFileHdr.NumberOfSections = Obj.Sections.size();
  InitFileHdr.NumberOfSymTableEntries = Obj.Symbols.size();

  for (XCOFFYAML::Symbol &YamlSym : Obj.Symbols) {
    uint32_t AuxCount = YamlSym.AuxEntries.size();
    if (YamlSym.NumberOfAuxEntries && *YamlSym.NumberOfAuxEntries < AuxCount) {
      ErrHandler("specified NumberOfAuxEntries " +
                 Twine(static_cast<uint32_t>(*YamlSym.NumberOfAuxEntries)) +
                 " is less than the actual number "
                 "of auxiliary entries " +
                 Twine(AuxCount));
      return false;
    }
    YamlSym.NumberOfAuxEntries =
        YamlSym.NumberOfAuxEntries.getValueOr(AuxCount);
    // Add the number of auxiliary symbols to the total number.
    InitFileHdr.NumberOfSymTableEntries += *YamlSym.NumberOfAuxEntries;
  }

  // Calculate SymbolTableOffset for the file header.
  if (InitFileHdr.NumberOfSymTableEntries) {
    InitFileHdr.SymbolTableOffset = CurrentOffset;
    CurrentOffset +=
        InitFileHdr.NumberOfSymTableEntries * XCOFF::SymbolTableEntrySize;
    if (CurrentOffset > MaxRawDataSize) {
      ErrHandler("maximum object size of" + Twine(MaxRawDataSize) +
                 "exceeded when writing symbols");
      return false;
    }
  }
  // TODO: Calculate FileOffsetToLineNumbers when line number supported.
  return true;
}

void XCOFFWriter::initAuxFileHeader() {
  InitAuxFileHdr = *Obj.AuxHeader;
  // In general, an object file might contain multiple sections of a given type,
  // but in a loadable module, there must be exactly one .text, .data, .bss, and
  // .loader section. A loadable object might also have one .tdata section and
  // one .tbss section.
  // Set these section-related values if not set explicitly. We assume that the
  // input YAML matches the format of the loadable object, but if multiple input
  // sections still have the same type, the first section with that type
  // prevails.
  for (uint16_t I = 0, E = InitSections.size(); I < E; ++I) {
    switch (InitSections[I].Flags) {
    case XCOFF::STYP_TEXT:
      if (!InitAuxFileHdr.TextSize)
        InitAuxFileHdr.TextSize = InitSections[I].Size;
      if (!InitAuxFileHdr.TextStartAddr)
        InitAuxFileHdr.TextStartAddr = InitSections[I].Address;
      if (!InitAuxFileHdr.SecNumOfText)
        InitAuxFileHdr.SecNumOfText = I + 1;
      break;
    case XCOFF::STYP_DATA:
      if (!InitAuxFileHdr.InitDataSize)
        InitAuxFileHdr.InitDataSize = InitSections[I].Size;
      if (!InitAuxFileHdr.DataStartAddr)
        InitAuxFileHdr.DataStartAddr = InitSections[I].Address;
      if (!InitAuxFileHdr.SecNumOfData)
        InitAuxFileHdr.SecNumOfData = I + 1;
      break;
    case XCOFF::STYP_BSS:
      if (!InitAuxFileHdr.BssDataSize)
        InitAuxFileHdr.BssDataSize = InitSections[I].Size;
      if (!InitAuxFileHdr.SecNumOfBSS)
        InitAuxFileHdr.SecNumOfBSS = I + 1;
      break;
    case XCOFF::STYP_TDATA:
      if (!InitAuxFileHdr.SecNumOfTData)
        InitAuxFileHdr.SecNumOfTData = I + 1;
      break;
    case XCOFF::STYP_TBSS:
      if (!InitAuxFileHdr.SecNumOfTBSS)
        InitAuxFileHdr.SecNumOfTBSS = I + 1;
      break;
    case XCOFF::STYP_LOADER:
      if (!InitAuxFileHdr.SecNumOfLoader)
        InitAuxFileHdr.SecNumOfLoader = I + 1;
      break;
    default:
      break;
    }
  }
}

bool XCOFFWriter::assignAddressesAndIndices() {
  uint64_t FileHdrSize =
      Is64Bit ? XCOFF::FileHeaderSize64 : XCOFF::FileHeaderSize32;
  uint64_t AuxFileHdrSize = 0;
  if (Obj.AuxHeader)
    AuxFileHdrSize = Obj.Header.AuxHeaderSize
                         ? Obj.Header.AuxHeaderSize
                         : (Is64Bit ? XCOFF::AuxFileHeaderSize64
                                    : XCOFF::AuxFileHeaderSize32);
  uint64_t SecHdrSize =
      Is64Bit ? XCOFF::SectionHeaderSize64 : XCOFF::SectionHeaderSize32;
  uint64_t CurrentOffset =
      FileHdrSize + AuxFileHdrSize + InitSections.size() * SecHdrSize;

  // Calculate section header info.
  if (!initSectionHeader(CurrentOffset))
    return false;
  InitFileHdr.AuxHeaderSize = AuxFileHdrSize;

  // Calculate file header info.
  if (!initFileHeader(CurrentOffset))
    return false;

  // Initialize the auxiliary file header.
  if (Obj.AuxHeader)
    initAuxFileHeader();

  // Initialize the string table.
  return initStringTable();
}

void XCOFFWriter::writeFileHeader() {
  W.write<uint16_t>(Obj.Header.Magic ? Obj.Header.Magic : InitFileHdr.Magic);
  W.write<uint16_t>(Obj.Header.NumberOfSections ? Obj.Header.NumberOfSections
                                                : InitFileHdr.NumberOfSections);
  W.write<int32_t>(Obj.Header.TimeStamp);
  if (Is64Bit) {
    W.write<uint64_t>(Obj.Header.SymbolTableOffset
                          ? Obj.Header.SymbolTableOffset
                          : InitFileHdr.SymbolTableOffset);
    W.write<uint16_t>(InitFileHdr.AuxHeaderSize);
    W.write<uint16_t>(Obj.Header.Flags);
    W.write<int32_t>(Obj.Header.NumberOfSymTableEntries
                         ? Obj.Header.NumberOfSymTableEntries
                         : InitFileHdr.NumberOfSymTableEntries);
  } else {
    W.write<uint32_t>(Obj.Header.SymbolTableOffset
                          ? Obj.Header.SymbolTableOffset
                          : InitFileHdr.SymbolTableOffset);
    W.write<int32_t>(Obj.Header.NumberOfSymTableEntries
                         ? Obj.Header.NumberOfSymTableEntries
                         : InitFileHdr.NumberOfSymTableEntries);
    W.write<uint16_t>(InitFileHdr.AuxHeaderSize);
    W.write<uint16_t>(Obj.Header.Flags);
  }
}

void XCOFFWriter::writeAuxFileHeader() {
  W.write<uint16_t>(InitAuxFileHdr.Magic.getValueOr(yaml::Hex16(1)));
  W.write<uint16_t>(InitAuxFileHdr.Version.getValueOr(yaml::Hex16(1)));
  if (Is64Bit) {
    W.OS.write_zeros(4); // Reserved for debugger.
    W.write<uint64_t>(InitAuxFileHdr.TextStartAddr.getValueOr(yaml::Hex64(0)));
    W.write<uint64_t>(InitAuxFileHdr.DataStartAddr.getValueOr(yaml::Hex64(0)));
    W.write<uint64_t>(InitAuxFileHdr.TOCAnchorAddr.getValueOr(yaml::Hex64(0)));
  } else {
    W.write<uint32_t>(InitAuxFileHdr.TextSize.getValueOr(yaml::Hex64(0)));
    W.write<uint32_t>(InitAuxFileHdr.InitDataSize.getValueOr(yaml::Hex64(0)));
    W.write<uint32_t>(InitAuxFileHdr.BssDataSize.getValueOr(yaml::Hex64(0)));
    W.write<uint32_t>(InitAuxFileHdr.EntryPointAddr.getValueOr(yaml::Hex64(0)));
    W.write<uint32_t>(InitAuxFileHdr.TextStartAddr.getValueOr(yaml::Hex64(0)));
    W.write<uint32_t>(InitAuxFileHdr.DataStartAddr.getValueOr(yaml::Hex64(0)));
    W.write<uint32_t>(InitAuxFileHdr.TOCAnchorAddr.getValueOr(yaml::Hex64(0)));
  }
  W.write<uint16_t>(InitAuxFileHdr.SecNumOfEntryPoint.getValueOr(0));
  W.write<uint16_t>(InitAuxFileHdr.SecNumOfText.getValueOr(0));
  W.write<uint16_t>(InitAuxFileHdr.SecNumOfData.getValueOr(0));
  W.write<uint16_t>(InitAuxFileHdr.SecNumOfTOC.getValueOr(0));
  W.write<uint16_t>(InitAuxFileHdr.SecNumOfLoader.getValueOr(0));
  W.write<uint16_t>(InitAuxFileHdr.SecNumOfBSS.getValueOr(0));
  W.write<uint16_t>(InitAuxFileHdr.MaxAlignOfText.getValueOr(yaml::Hex16(0)));
  W.write<uint16_t>(InitAuxFileHdr.MaxAlignOfData.getValueOr(yaml::Hex16(0)));
  W.write<uint16_t>(InitAuxFileHdr.ModuleType.getValueOr(yaml::Hex16(0)));
  W.write<uint8_t>(InitAuxFileHdr.CpuFlag.getValueOr(yaml::Hex8(0)));
  W.write<uint8_t>(0); // Reserved for CPU type.
  if (Is64Bit) {
    W.write<uint8_t>(InitAuxFileHdr.TextPageSize.getValueOr(yaml::Hex8(0)));
    W.write<uint8_t>(InitAuxFileHdr.DataPageSize.getValueOr(yaml::Hex8(0)));
    W.write<uint8_t>(InitAuxFileHdr.StackPageSize.getValueOr(yaml::Hex8(0)));
    W.write<uint8_t>(
        InitAuxFileHdr.FlagAndTDataAlignment.getValueOr(yaml::Hex8(0x80)));
    W.write<uint64_t>(InitAuxFileHdr.TextSize.getValueOr(yaml::Hex64(0)));
    W.write<uint64_t>(InitAuxFileHdr.InitDataSize.getValueOr(yaml::Hex64(0)));
    W.write<uint64_t>(InitAuxFileHdr.BssDataSize.getValueOr(yaml::Hex64(0)));
    W.write<uint64_t>(InitAuxFileHdr.EntryPointAddr.getValueOr(yaml::Hex64(0)));
    W.write<uint64_t>(InitAuxFileHdr.MaxStackSize.getValueOr(yaml::Hex64(0)));
    W.write<uint64_t>(InitAuxFileHdr.MaxDataSize.getValueOr(yaml::Hex64(0)));
  } else {
    W.write<uint32_t>(InitAuxFileHdr.MaxStackSize.getValueOr(yaml::Hex64(0)));
    W.write<uint32_t>(InitAuxFileHdr.MaxDataSize.getValueOr(yaml::Hex64(0)));
    W.OS.write_zeros(4); // Reserved for debugger.
    W.write<uint8_t>(InitAuxFileHdr.TextPageSize.getValueOr(yaml::Hex8(0)));
    W.write<uint8_t>(InitAuxFileHdr.DataPageSize.getValueOr(yaml::Hex8(0)));
    W.write<uint8_t>(InitAuxFileHdr.StackPageSize.getValueOr(yaml::Hex8(0)));
    W.write<uint8_t>(
        InitAuxFileHdr.FlagAndTDataAlignment.getValueOr(yaml::Hex8(0)));
  }
  W.write<uint16_t>(InitAuxFileHdr.SecNumOfTData.getValueOr(0));
  W.write<uint16_t>(InitAuxFileHdr.SecNumOfTBSS.getValueOr(0));
  if (Is64Bit) {
    W.write<uint16_t>(InitAuxFileHdr.Flag.getValueOr(yaml::Hex16(XCOFF::SHR_SYMTAB)));
    if (InitFileHdr.AuxHeaderSize > XCOFF::AuxFileHeaderSize64)
      W.OS.write_zeros(InitFileHdr.AuxHeaderSize - XCOFF::AuxFileHeaderSize64);
  } else if (InitFileHdr.AuxHeaderSize > XCOFF::AuxFileHeaderSize32) {
    W.OS.write_zeros(InitFileHdr.AuxHeaderSize - XCOFF::AuxFileHeaderSize32);
  }
}

void XCOFFWriter::writeSectionHeader() {
  for (uint16_t I = 0, E = Obj.Sections.size(); I < E; ++I) {
    XCOFFYAML::Section YamlSec = Obj.Sections[I];
    XCOFFYAML::Section DerivedSec = InitSections[I];
    writeName(YamlSec.SectionName, W);
    // Virtual address is the same as physical address.
    uint64_t SectionAddress =
        YamlSec.Address ? YamlSec.Address : DerivedSec.Address;
    if (Is64Bit) {
      W.write<uint64_t>(SectionAddress); // Physical address
      W.write<uint64_t>(SectionAddress); // Virtual address
      W.write<uint64_t>(YamlSec.Size ? YamlSec.Size : DerivedSec.Size);
      W.write<uint64_t>(YamlSec.FileOffsetToData ? YamlSec.FileOffsetToData
                                                 : DerivedSec.FileOffsetToData);
      W.write<uint64_t>(YamlSec.FileOffsetToRelocations
                            ? YamlSec.FileOffsetToRelocations
                            : DerivedSec.FileOffsetToRelocations);
      W.write<uint64_t>(YamlSec.FileOffsetToLineNumbers);
      W.write<uint32_t>(YamlSec.NumberOfRelocations
                            ? YamlSec.NumberOfRelocations
                            : DerivedSec.NumberOfRelocations);
      W.write<uint32_t>(YamlSec.NumberOfLineNumbers);
      W.write<int32_t>(YamlSec.Flags);
      W.OS.write_zeros(4);
    } else {
      W.write<uint32_t>(SectionAddress); // Physical address
      W.write<uint32_t>(SectionAddress); // Virtual address
      W.write<uint32_t>(YamlSec.Size ? YamlSec.Size : DerivedSec.Size);
      W.write<uint32_t>(YamlSec.FileOffsetToData ? YamlSec.FileOffsetToData
                                                 : DerivedSec.FileOffsetToData);
      W.write<uint32_t>(YamlSec.FileOffsetToRelocations
                            ? YamlSec.FileOffsetToRelocations
                            : DerivedSec.FileOffsetToRelocations);
      W.write<uint32_t>(YamlSec.FileOffsetToLineNumbers);
      W.write<uint16_t>(YamlSec.NumberOfRelocations
                            ? YamlSec.NumberOfRelocations
                            : DerivedSec.NumberOfRelocations);
      W.write<uint16_t>(YamlSec.NumberOfLineNumbers);
      W.write<int32_t>(YamlSec.Flags);
    }
  }
}

bool XCOFFWriter::writeSectionData() {
  for (uint16_t I = 0, E = Obj.Sections.size(); I < E; ++I) {
    XCOFFYAML::Section YamlSec = Obj.Sections[I];
    if (YamlSec.SectionData.binary_size()) {
      // Fill the padding size with zeros.
      int64_t PaddingSize =
          InitSections[I].FileOffsetToData - (W.OS.tell() - StartOffset);
      if (PaddingSize < 0) {
        ErrHandler("redundant data was written before section data");
        return false;
      }
      W.OS.write_zeros(PaddingSize);
      YamlSec.SectionData.writeAsBinary(W.OS);
    }
  }
  return true;
}

bool XCOFFWriter::writeRelocations() {
  for (uint16_t I = 0, E = Obj.Sections.size(); I < E; ++I) {
    XCOFFYAML::Section YamlSec = Obj.Sections[I];
    if (!YamlSec.Relocations.empty()) {
      int64_t PaddingSize =
          InitSections[I].FileOffsetToRelocations - (W.OS.tell() - StartOffset);
      if (PaddingSize < 0) {
        ErrHandler("redundant data was written before relocations");
        return false;
      }
      W.OS.write_zeros(PaddingSize);
      for (const XCOFFYAML::Relocation &YamlRel : YamlSec.Relocations) {
        if (Is64Bit)
          W.write<uint64_t>(YamlRel.VirtualAddress);
        else
          W.write<uint32_t>(YamlRel.VirtualAddress);
        W.write<uint32_t>(YamlRel.SymbolIndex);
        W.write<uint8_t>(YamlRel.Info);
        W.write<uint8_t>(YamlRel.Type);
      }
    }
  }
  return true;
}

void XCOFFWriter::writeAuxSymbol(const XCOFFYAML::CsectAuxEnt &AuxSym) {
  if (Is64Bit) {
    W.write<uint32_t>(AuxSym.SectionOrLengthLo.getValueOr(0));
    W.write<uint32_t>(AuxSym.ParameterHashIndex.getValueOr(0));
    W.write<uint16_t>(AuxSym.TypeChkSectNum.getValueOr(0));
    W.write<uint8_t>(AuxSym.SymbolAlignmentAndType.getValueOr(0));
    W.write<uint8_t>(AuxSym.StorageMappingClass.getValueOr(XCOFF::XMC_PR));
    W.write<uint32_t>(AuxSym.SectionOrLengthHi.getValueOr(0));
    W.write<uint8_t>(0);
    W.write<uint8_t>(XCOFF::AUX_CSECT);
  } else {
    W.write<uint32_t>(AuxSym.SectionOrLength.getValueOr(0));
    W.write<uint32_t>(AuxSym.ParameterHashIndex.getValueOr(0));
    W.write<uint16_t>(AuxSym.TypeChkSectNum.getValueOr(0));
    W.write<uint8_t>(AuxSym.SymbolAlignmentAndType.getValueOr(0));
    W.write<uint8_t>(AuxSym.StorageMappingClass.getValueOr(XCOFF::XMC_PR));
    W.write<uint32_t>(AuxSym.StabInfoIndex.getValueOr(0));
    W.write<uint16_t>(AuxSym.StabSectNum.getValueOr(0));
  }
}

void XCOFFWriter::writeAuxSymbol(const XCOFFYAML::ExcpetionAuxEnt &AuxSym) {
  assert(Is64Bit && "can't write the exception auxiliary symbol for XCOFF32");
  W.write<uint64_t>(AuxSym.OffsetToExceptionTbl.getValueOr(0));
  W.write<uint32_t>(AuxSym.SizeOfFunction.getValueOr(0));
  W.write<uint32_t>(AuxSym.SymIdxOfNextBeyond.getValueOr(0));
  W.write<uint8_t>(0);
  W.write<uint8_t>(XCOFF::AUX_EXCEPT);
}

void XCOFFWriter::writeAuxSymbol(const XCOFFYAML::FunctionAuxEnt &AuxSym) {
  if (Is64Bit) {
    W.write<uint64_t>(AuxSym.PtrToLineNum.getValueOr(0));
    W.write<uint32_t>(AuxSym.SizeOfFunction.getValueOr(0));
    W.write<uint32_t>(AuxSym.SymIdxOfNextBeyond.getValueOr(0));
    W.write<uint8_t>(0);
    W.write<uint8_t>(XCOFF::AUX_FCN);
  } else {
    W.write<uint32_t>(AuxSym.OffsetToExceptionTbl.getValueOr(0));
    W.write<uint32_t>(AuxSym.SizeOfFunction.getValueOr(0));
    W.write<uint32_t>(AuxSym.PtrToLineNum.getValueOr(0));
    W.write<uint32_t>(AuxSym.SymIdxOfNextBeyond.getValueOr(0));
    W.OS.write_zeros(2);
  }
}

void XCOFFWriter::writeAuxSymbol(const XCOFFYAML::FileAuxEnt &AuxSym) {
  StringRef FileName = AuxSym.FileNameOrString.getValueOr("");
  if (nameShouldBeInStringTable(FileName)) {
    W.write<int32_t>(0);
    W.write<uint32_t>(StrTblBuilder.getOffset(FileName));
  } else {
    writeName(FileName, W);
  }
  W.OS.write_zeros(XCOFF::FileNamePadSize);
  W.write<uint8_t>(AuxSym.FileStringType.getValueOr(XCOFF::XFT_FN));
  if (Is64Bit) {
    W.OS.write_zeros(2);
    W.write<uint8_t>(XCOFF::AUX_FILE);
  } else {
    W.OS.write_zeros(3);
  }
}

void XCOFFWriter::writeAuxSymbol(const XCOFFYAML::BlockAuxEnt &AuxSym) {
  if (Is64Bit) {
    W.write<uint32_t>(AuxSym.LineNum.getValueOr(0));
    W.OS.write_zeros(13);
    W.write<uint8_t>(XCOFF::AUX_SYM);
  } else {
    W.OS.write_zeros(2);
    W.write<uint16_t>(AuxSym.LineNumHi.getValueOr(0));
    W.write<uint16_t>(AuxSym.LineNumLo.getValueOr(0));
    W.OS.write_zeros(12);
  }
}

void XCOFFWriter::writeAuxSymbol(const XCOFFYAML::SectAuxEntForDWARF &AuxSym) {
  if (Is64Bit) {
    W.write<uint64_t>(AuxSym.LengthOfSectionPortion.getValueOr(0));
    W.write<uint64_t>(AuxSym.NumberOfRelocEnt.getValueOr(0));
    W.write<uint8_t>(0);
    W.write<uint8_t>(XCOFF::AUX_SECT);
  } else {
    W.write<uint32_t>(AuxSym.LengthOfSectionPortion.getValueOr(0));
    W.OS.write_zeros(4);
    W.write<uint32_t>(AuxSym.NumberOfRelocEnt.getValueOr(0));
    W.OS.write_zeros(6);
  }
}

void XCOFFWriter::writeAuxSymbol(const XCOFFYAML::SectAuxEntForStat &AuxSym) {
  assert(!Is64Bit && "can't write the stat auxiliary symbol for XCOFF64");
  W.write<uint32_t>(AuxSym.SectionLength.getValueOr(0));
  W.write<uint16_t>(AuxSym.NumberOfRelocEnt.getValueOr(0));
  W.write<uint16_t>(AuxSym.NumberOfLineNum.getValueOr(0));
  W.OS.write_zeros(10);
}

void XCOFFWriter::writeAuxSymbol(
    const std::unique_ptr<XCOFFYAML::AuxSymbolEnt> &AuxSym) {
  if (auto AS = dyn_cast<XCOFFYAML::CsectAuxEnt>(AuxSym.get()))
    writeAuxSymbol(*AS);
  else if (auto AS = dyn_cast<XCOFFYAML::FunctionAuxEnt>(AuxSym.get()))
    writeAuxSymbol(*AS);
  else if (auto AS = dyn_cast<XCOFFYAML::ExcpetionAuxEnt>(AuxSym.get()))
    writeAuxSymbol(*AS);
  else if (auto AS = dyn_cast<XCOFFYAML::FileAuxEnt>(AuxSym.get()))
    writeAuxSymbol(*AS);
  else if (auto AS = dyn_cast<XCOFFYAML::BlockAuxEnt>(AuxSym.get()))
    writeAuxSymbol(*AS);
  else if (auto AS = dyn_cast<XCOFFYAML::SectAuxEntForDWARF>(AuxSym.get()))
    writeAuxSymbol(*AS);
  else if (auto AS = dyn_cast<XCOFFYAML::SectAuxEntForStat>(AuxSym.get()))
    writeAuxSymbol(*AS);
  else
    llvm_unreachable("unknown auxiliary symbol type");
}

bool XCOFFWriter::writeSymbols() {
  int64_t PaddingSize =
      (uint64_t)InitFileHdr.SymbolTableOffset - (W.OS.tell() - StartOffset);
  if (PaddingSize < 0) {
    ErrHandler("redundant data was written before symbols");
    return false;
  }
  W.OS.write_zeros(PaddingSize);
  for (const XCOFFYAML::Symbol &YamlSym : Obj.Symbols) {
    if (Is64Bit) {
      W.write<uint64_t>(YamlSym.Value);
      W.write<uint32_t>(StrTblBuilder.getOffset(YamlSym.SymbolName));
    } else {
      if (nameShouldBeInStringTable(YamlSym.SymbolName)) {
        // For XCOFF32: A value of 0 indicates that the symbol name is in the
        // string table.
        W.write<int32_t>(0);
        W.write<uint32_t>(StrTblBuilder.getOffset(YamlSym.SymbolName));
      } else {
        writeName(YamlSym.SymbolName, W);
      }
      W.write<uint32_t>(YamlSym.Value);
    }
    if (YamlSym.SectionName) {
      if (!SectionIndexMap.count(*YamlSym.SectionName)) {
        ErrHandler("the SectionName " + *YamlSym.SectionName +
                   " specified in the symbol does not exist");
        return false;
      }
      if (YamlSym.SectionIndex &&
          SectionIndexMap[*YamlSym.SectionName] != *YamlSym.SectionIndex) {
        ErrHandler("the SectionName " + *YamlSym.SectionName +
                   " and the SectionIndex (" + Twine(*YamlSym.SectionIndex) +
                   ") refer to different sections");
        return false;
      }
      W.write<int16_t>(SectionIndexMap[*YamlSym.SectionName]);
    } else {
      W.write<int16_t>(YamlSym.SectionIndex ? *YamlSym.SectionIndex : 0);
    }
    W.write<uint16_t>(YamlSym.Type);
    W.write<uint8_t>(YamlSym.StorageClass);

    uint8_t NumOfAuxSym = YamlSym.NumberOfAuxEntries.getValueOr(0);
    W.write<uint8_t>(NumOfAuxSym);

    if (!NumOfAuxSym && !YamlSym.AuxEntries.size())
      continue;

    // Now write auxiliary entries.
    if (!YamlSym.AuxEntries.size()) {
      W.OS.write_zeros(XCOFF::SymbolTableEntrySize * NumOfAuxSym);
    } else {
      for (const std::unique_ptr<XCOFFYAML::AuxSymbolEnt> &AuxSym :
           YamlSym.AuxEntries) {
        writeAuxSymbol(AuxSym);
      }
      // Pad with zeros.
      if (NumOfAuxSym > YamlSym.AuxEntries.size())
        W.OS.write_zeros(XCOFF::SymbolTableEntrySize *
                         (NumOfAuxSym - YamlSym.AuxEntries.size()));
    }
  }
  return true;
}

void XCOFFWriter::writeStringTable() {
  if (Obj.StrTbl.RawContent) {
    Obj.StrTbl.RawContent->writeAsBinary(W.OS);
    if (Obj.StrTbl.ContentSize) {
      assert(*Obj.StrTbl.ContentSize >= Obj.StrTbl.RawContent->binary_size() &&
             "Specified ContentSize is less than the RawContent size.");
      W.OS.write_zeros(*Obj.StrTbl.ContentSize -
                       Obj.StrTbl.RawContent->binary_size());
    }
    return;
  }

  size_t StrTblBuilderSize = StrTblBuilder.getSize();
  // If neither Length nor ContentSize is specified, write the StrTblBuilder
  // directly, which contains the auto-generated Length value.
  if (!Obj.StrTbl.Length && !Obj.StrTbl.ContentSize) {
    if (StrTblBuilderSize <= 4)
      return;
    StrTblBuilder.write(W.OS);
    return;
  }

  // Serialize the string table's content to a temporary buffer.
  std::unique_ptr<WritableMemoryBuffer> Buf =
      WritableMemoryBuffer::getNewMemBuffer(StrTblBuilderSize);
  uint8_t *Ptr = reinterpret_cast<uint8_t *>(Buf->getBufferStart());
  StrTblBuilder.write(Ptr);
  // Replace the first 4 bytes, which contain the auto-generated Length value,
  // with the specified value.
  memset(Ptr, 0, 4);
  support::endian::write32be(Ptr, Obj.StrTbl.Length ? *Obj.StrTbl.Length
                                                    : *Obj.StrTbl.ContentSize);
  // Copy the buffer content to the actual output stream.
  W.OS.write(Buf->getBufferStart(), Buf->getBufferSize());
  // Add zeros as padding after strings.
  if (Obj.StrTbl.ContentSize) {
    assert(*Obj.StrTbl.ContentSize >= StrTblBuilderSize &&
           "Specified ContentSize is less than the StringTableBuilder size.");
    W.OS.write_zeros(*Obj.StrTbl.ContentSize - StrTblBuilderSize);
  }
}

bool XCOFFWriter::writeXCOFF() {
  if (!assignAddressesAndIndices())
    return false;
  StartOffset = W.OS.tell();
  writeFileHeader();
  if (Obj.AuxHeader)
    writeAuxFileHeader();
  if (!Obj.Sections.empty()) {
    writeSectionHeader();
    if (!writeSectionData())
      return false;
    if (!writeRelocations())
      return false;
  }
  if (!Obj.Symbols.empty() && !writeSymbols())
    return false;
  writeStringTable();
  return true;
}

} // end anonymous namespace

namespace llvm {
namespace yaml {

bool yaml2xcoff(XCOFFYAML::Object &Doc, raw_ostream &Out, ErrorHandler EH) {
  XCOFFWriter Writer(Doc, Out, EH);
  return Writer.writeXCOFF();
}

} // namespace yaml
} // namespace llvm
