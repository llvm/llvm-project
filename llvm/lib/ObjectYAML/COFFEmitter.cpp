//===- yaml2coff - Convert YAML to a COFF object file ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The COFF component of yaml2obj.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/DebugInfo/CodeView/StringsAndChecksums.h"
#include "llvm/ObjectYAML/ContiguousBlobAccumulator.h"
#include "llvm/ObjectYAML/ObjectYAML.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <vector>

using namespace llvm;
using llvm::yaml::ContiguousBlobAccumulator;

namespace {

constexpr auto LittleEndian = llvm::endianness::little;

/// This parses a yaml stream that represents a COFF object file.
/// See docs/yaml2obj for the yaml scheema.
struct COFFParser {
  COFFParser(COFFYAML::Object &Obj, yaml::ErrorHandler EH)
      : Obj(Obj), ErrHandler(EH) {
    // A COFF string table always starts with a 4 byte size field. Offsets into
    // it include this size, so allocate it now.
    StringTable.append(4, char(0));
  }

  bool useBigObj() const {
    return static_cast<int32_t>(Obj.Sections.size()) >
           COFF::MaxNumberOfSections16;
  }

  bool isPE() const { return Obj.OptionalHeader.has_value(); }
  bool is64Bit() const { return COFF::is64Bit(Obj.Header.Machine); }

  uint32_t getFileAlignment() const {
    return Obj.OptionalHeader->Header.FileAlignment;
  }

  unsigned getSymbolSize() const {
    return useBigObj() ? COFF::Symbol32Size : COFF::Symbol16Size;
  }

  bool parseSections() {
    for (COFFYAML::Section &Sec : Obj.Sections) {
      // If the name is less than 8 bytes, store it in place, otherwise
      // store it in the string table.
      StringRef Name = Sec.Name;

      if (Name.size() <= COFF::NameSize) {
        llvm::copy(Name, Sec.Header.Name);
      } else {
        // Add string to the string table and format the index for output.
        unsigned Index = getStringIndex(Name);
        std::string str = utostr(Index);
        if (str.size() > 7) {
          ErrHandler("string table got too large");
          return false;
        }
        Sec.Header.Name[0] = '/';
        llvm::copy(str, Sec.Header.Name + 1);
      }

      if (Sec.Alignment) {
        if (Sec.Alignment > 8192) {
          ErrHandler("section alignment is too large");
          return false;
        }
        if (!isPowerOf2_32(Sec.Alignment)) {
          ErrHandler("section alignment is not a power of 2");
          return false;
        }
        Sec.Header.Characteristics |= (Log2_32(Sec.Alignment) + 1) << 20;
      }
    }
    return true;
  }

  bool parseSymbols() {
    for (COFFYAML::Symbol &Sym : Obj.Symbols) {
      // If the name is less than 8 bytes, store it in place, otherwise
      // store it in the string table.
      StringRef Name = Sym.Name;
      if (Name.size() <= COFF::NameSize) {
        llvm::copy(Name, Sym.Header.Name);
      } else {
        // Add string to the string table and format the index for output.
        unsigned Index = getStringIndex(Name);
        *reinterpret_cast<support::aligned_ulittle32_t *>(Sym.Header.Name + 4) =
            Index;
      }

      Sym.Header.Type = Sym.SimpleType;
      Sym.Header.Type |= Sym.ComplexType << COFF::SCT_COMPLEX_TYPE_SHIFT;
    }
    return true;
  }

  bool parse() {
    if (!parseSections())
      return false;
    if (!parseSymbols())
      return false;
    return true;
  }

  unsigned getStringIndex(StringRef Str) {
    auto [It, Inserted] = StringTableMap.try_emplace(Str, StringTable.size());
    if (Inserted) {
      StringTable.append(Str.begin(), Str.end());
      StringTable.push_back(0);
    }
    return It->second;
  }

  COFFYAML::Object &Obj;

  codeview::StringsAndChecksums StringsAndChecksums;
  BumpPtrAllocator Allocator;
  StringMap<unsigned> StringTableMap;
  std::string StringTable;
  uint32_t SectionTableStart;
  uint32_t SectionTableSize;

  yaml::ErrorHandler ErrHandler;
};

enum { DOSStubSize = 128 };

} // end anonymous namespace

static yaml::BinaryRef
toDebugS(ArrayRef<CodeViewYAML::YAMLDebugSubsection> Subsections,
         const codeview::StringsAndChecksums &SC, BumpPtrAllocator &Allocator) {
  using namespace codeview;
  ExitOnError Err("Error occurred writing .debug$S section");
  auto CVSS =
      Err(CodeViewYAML::toCodeViewSubsectionList(Allocator, Subsections, SC));

  std::vector<DebugSubsectionRecordBuilder> Builders;
  uint32_t Size = sizeof(uint32_t);
  for (auto &SS : CVSS) {
    DebugSubsectionRecordBuilder B(SS);
    Size += B.calculateSerializedLength();
    Builders.push_back(std::move(B));
  }
  uint8_t *Buffer = Allocator.Allocate<uint8_t>(Size);
  MutableArrayRef<uint8_t> Output(Buffer, Size);
  BinaryStreamWriter Writer(Output, llvm::endianness::little);

  Err(Writer.writeInteger<uint32_t>(COFF::DEBUG_SECTION_MAGIC));
  for (const auto &B : Builders) {
    Err(B.commit(Writer, CodeViewContainer::ObjectFile));
  }
  return {Output};
}

// Write the content of a section and fill in the header fields locating it.
// Returns whether the section has any content.
static bool writeSectionContent(COFFParser &CP, COFFYAML::Section &S,
                                ContiguousBlobAccumulator &CBA) {
  if (S.SectionData.binary_size() == 0) {
    if (S.Name == ".debug$S") {
      assert(CP.StringsAndChecksums.hasStrings() &&
             "Object file does not have debug string table!");
      S.SectionData = toDebugS(S.DebugS, CP.StringsAndChecksums, CP.Allocator);
    } else if (S.Name == ".debug$T") {
      S.SectionData = CodeViewYAML::toDebugT(S.DebugT, CP.Allocator, S.Name);
    } else if (S.Name == ".debug$P") {
      S.SectionData = CodeViewYAML::toDebugT(S.DebugP, CP.Allocator, S.Name);
    } else if (S.Name == ".debug$H" && S.DebugH) {
      S.SectionData = CodeViewYAML::toDebugH(*S.DebugH, CP.Allocator);
    }
  }

  bool HasContent = S.SectionData.binary_size() != 0;
  for (const auto &E : S.StructuredData)
    HasContent |= E.size() != 0;

  if (!HasContent) {
    // Leave SizeOfRawData unaltered. For .bss sections in object files, it
    // carries the section size.
    S.Header.PointerToRawData = 0;
    return false;
  }

  CBA.padToAlignment(CP.isPE() ? CP.getFileAlignment() : 4);
  S.Header.PointerToRawData = CBA.getOffset();
  for (const auto &E : S.StructuredData)
    E.writeAsBinary(CBA);
  CBA.writeAsBinary(S.SectionData);
  if (CP.isPE())
    CBA.padToAlignment(CP.getFileAlignment());
  S.Header.SizeOfRawData = CBA.getOffset() - S.Header.PointerToRawData;
  return true;
}

template <typename T>
static uint32_t initializeOptionalHeader(COFFParser &CP, uint16_t Magic,
                                         T Header) {
  memset(Header, 0, sizeof(*Header));
  Header->Magic = Magic;
  Header->SectionAlignment = CP.Obj.OptionalHeader->Header.SectionAlignment;
  Header->FileAlignment = CP.Obj.OptionalHeader->Header.FileAlignment;
  uint32_t SizeOfCode = 0, SizeOfInitializedData = 0,
           SizeOfUninitializedData = 0;
  uint32_t SizeOfHeaders = alignTo(CP.SectionTableStart + CP.SectionTableSize,
                                   Header->FileAlignment);
  uint32_t SizeOfImage = alignTo(SizeOfHeaders, Header->SectionAlignment);
  uint32_t BaseOfData = 0;
  for (const COFFYAML::Section &S : CP.Obj.Sections) {
    if (S.Header.Characteristics & COFF::IMAGE_SCN_CNT_CODE)
      SizeOfCode += S.Header.SizeOfRawData;
    if (S.Header.Characteristics & COFF::IMAGE_SCN_CNT_INITIALIZED_DATA)
      SizeOfInitializedData += S.Header.SizeOfRawData;
    if (S.Header.Characteristics & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA)
      SizeOfUninitializedData += S.Header.SizeOfRawData;
    if (S.Name == ".text")
      Header->BaseOfCode = S.Header.VirtualAddress; // RVA
    else if (S.Name == ".data")
      BaseOfData = S.Header.VirtualAddress; // RVA
    if (S.Header.VirtualAddress)
      SizeOfImage += alignTo(S.Header.VirtualSize, Header->SectionAlignment);
  }
  Header->SizeOfCode = SizeOfCode;
  Header->SizeOfInitializedData = SizeOfInitializedData;
  Header->SizeOfUninitializedData = SizeOfUninitializedData;
  Header->AddressOfEntryPoint =
      CP.Obj.OptionalHeader->Header.AddressOfEntryPoint; // RVA
  Header->ImageBase = CP.Obj.OptionalHeader->Header.ImageBase;
  Header->MajorOperatingSystemVersion =
      CP.Obj.OptionalHeader->Header.MajorOperatingSystemVersion;
  Header->MinorOperatingSystemVersion =
      CP.Obj.OptionalHeader->Header.MinorOperatingSystemVersion;
  Header->MajorImageVersion = CP.Obj.OptionalHeader->Header.MajorImageVersion;
  Header->MinorImageVersion = CP.Obj.OptionalHeader->Header.MinorImageVersion;
  Header->MajorSubsystemVersion =
      CP.Obj.OptionalHeader->Header.MajorSubsystemVersion;
  Header->MinorSubsystemVersion =
      CP.Obj.OptionalHeader->Header.MinorSubsystemVersion;
  Header->SizeOfImage = SizeOfImage;
  Header->SizeOfHeaders = SizeOfHeaders;
  Header->Subsystem = CP.Obj.OptionalHeader->Header.Subsystem;
  Header->DLLCharacteristics = CP.Obj.OptionalHeader->Header.DLLCharacteristics;
  Header->SizeOfStackReserve = CP.Obj.OptionalHeader->Header.SizeOfStackReserve;
  Header->SizeOfStackCommit = CP.Obj.OptionalHeader->Header.SizeOfStackCommit;
  Header->SizeOfHeapReserve = CP.Obj.OptionalHeader->Header.SizeOfHeapReserve;
  Header->SizeOfHeapCommit = CP.Obj.OptionalHeader->Header.SizeOfHeapCommit;
  Header->NumberOfRvaAndSize = CP.Obj.OptionalHeader->Header.NumberOfRvaAndSize;
  return BaseOfData;
}

static bool writeCOFF(COFFParser &CP, ContiguousBlobAccumulator &CBA) {
  // Calculate number of symbols.
  CP.Obj.Header.NumberOfSymbols = 0;
  for (COFFYAML::Symbol &Sym : CP.Obj.Symbols) {
    uint32_t NumberOfAuxSymbols = 0;
    if (Sym.FunctionDefinition)
      NumberOfAuxSymbols += 1;
    if (Sym.bfAndefSymbol)
      NumberOfAuxSymbols += 1;
    if (Sym.WeakExternal)
      NumberOfAuxSymbols += 1;
    if (!Sym.File.empty())
      NumberOfAuxSymbols +=
          (Sym.File.size() + CP.getSymbolSize() - 1) / CP.getSymbolSize();
    if (Sym.SectionDefinition)
      NumberOfAuxSymbols += 1;
    if (Sym.CLRToken)
      NumberOfAuxSymbols += 1;
    Sym.Header.NumberOfAuxSymbols = NumberOfAuxSymbols;
    CP.Obj.Header.NumberOfSymbols += 1 + NumberOfAuxSymbols;
  }

  CP.Obj.Header.NumberOfSections = CP.Obj.Sections.size();

  unsigned PEHeaderSize = CP.is64Bit() ? sizeof(object::pe32plus_header)
                                       : sizeof(object::pe32_header);
  if (CP.isPE())
    CP.Obj.Header.SizeOfOptionalHeader =
        PEHeaderSize + sizeof(object::data_directory) *
                           CP.Obj.OptionalHeader->Header.NumberOfRvaAndSize;

  // Save field offsets for writing back their final values.
  uint64_t PointerToSymbolTableOffset;

  if (CP.isPE()) {
    // PE files start with a DOS stub.
    object::dos_header DH;
    memset(&DH, 0, sizeof(DH));

    // DOS EXEs start with "MZ" magic.
    DH.Magic[0] = 'M';
    DH.Magic[1] = 'Z';
    // Initializing the AddressOfRelocationTable is strictly optional but
    // mollifies certain tools which expect it to have a value greater than
    // 0x40.
    DH.AddressOfRelocationTable = sizeof(DH);
    // This is the address of the PE signature.
    DH.AddressOfNewExeHeader = DOSStubSize;

    // Write out our DOS stub.
    CBA.write(reinterpret_cast<const char *>(&DH), sizeof(DH));
    // Write padding until we reach the position of where our PE signature
    // should live.
    CBA.writeZeros(DOSStubSize - sizeof(DH));
    // Write out the PE signature.
    CBA.write(COFF::PEMagic, sizeof(COFF::PEMagic));
  }
  if (CP.useBigObj()) {
    CBA.write(static_cast<uint16_t>(COFF::IMAGE_FILE_MACHINE_UNKNOWN),
              LittleEndian);
    CBA.write(static_cast<uint16_t>(0xffff), LittleEndian);
    CBA.write(static_cast<uint16_t>(COFF::BigObjHeader::MinBigObjectVersion),
              LittleEndian);
    CBA.write(CP.Obj.Header.Machine, LittleEndian);
    CBA.write(CP.Obj.Header.TimeDateStamp, LittleEndian);
    CBA.write(COFF::BigObjMagic, sizeof(COFF::BigObjMagic));
    CBA.writeZeros(4 * sizeof(uint32_t));
    CBA.write(CP.Obj.Header.NumberOfSections, LittleEndian);
    PointerToSymbolTableOffset = CBA.getOffset();
    // The final symbol table offset is written after the section data.
    CBA.writeZeros(sizeof(CP.Obj.Header.PointerToSymbolTable));
    CBA.write(CP.Obj.Header.NumberOfSymbols, LittleEndian);
  } else {
    CBA.write(CP.Obj.Header.Machine, LittleEndian);
    CBA.write(static_cast<int16_t>(CP.Obj.Header.NumberOfSections),
              LittleEndian);
    CBA.write(CP.Obj.Header.TimeDateStamp, LittleEndian);
    PointerToSymbolTableOffset = CBA.getOffset();
    // The final symbol table offset is written after the section data.
    CBA.writeZeros(sizeof(CP.Obj.Header.PointerToSymbolTable));
    CBA.write(CP.Obj.Header.NumberOfSymbols, LittleEndian);
    CBA.write(CP.Obj.Header.SizeOfOptionalHeader, LittleEndian);
    CBA.write(CP.Obj.Header.Characteristics, LittleEndian);
  }

  // The optional header, if present, immediately follows the COFF file header.
  uint64_t OptionalHeaderOffset = CBA.getOffset();
  if (CP.isPE()) {
    // Reserve space for the PE header, whose fields depend on the final section
    // layout. The data directories that follow it are already final.
    CBA.writeZeros(PEHeaderSize);
    for (uint32_t I = 0; I < CP.Obj.OptionalHeader->Header.NumberOfRvaAndSize;
         ++I) {
      const std::optional<COFF::DataDirectory> *DataDirectories =
          CP.Obj.OptionalHeader->DataDirectories;
      uint32_t NumDataDir = std::size(CP.Obj.OptionalHeader->DataDirectories);
      if (I >= NumDataDir || !DataDirectories[I]) {
        CBA.writeZeros(2 * sizeof(uint32_t));
      } else {
        CBA.write(DataDirectories[I]->RelativeVirtualAddress, LittleEndian);
        CBA.write(DataDirectories[I]->Size, LittleEndian);
      }
    }
  }

  CP.SectionTableStart = CBA.getOffset();
  CP.SectionTableSize = COFF::SectionSize * CP.Obj.Sections.size();
  // Reserve space for the section table. Section offsets and sizes are filled
  // in later.
  CBA.writeZeros(CP.SectionTableSize);

  unsigned CurSymbol = 0;
  StringMap<unsigned> SymbolTableIndexMap;
  for (const COFFYAML::Symbol &Sym : CP.Obj.Symbols) {
    SymbolTableIndexMap[Sym.Name] = CurSymbol;
    CurSymbol += 1 + Sym.Header.NumberOfAuxSymbols;
  }

  // Collect the CodeView strings and checksums shared by all .debug$S sections.
  for (COFFYAML::Section &S : CP.Obj.Sections) {
    // We support specifying exactly one of SectionData or Subsections. So if
    // there is already some SectionData, then we don't need to do any of this.
    if (S.Name == ".debug$S" && S.SectionData.binary_size() == 0) {
      CodeViewYAML::initializeStringsAndChecksums(S.DebugS,
                                                  CP.StringsAndChecksums);
      if (CP.StringsAndChecksums.hasChecksums() &&
          CP.StringsAndChecksums.hasStrings())
        break;
    }
  }

  // Output section data.
  for (COFFYAML::Section &S : CP.Obj.Sections) {
    bool HasContent = writeSectionContent(CP, S, CBA);
    if (!HasContent || S.Relocations.empty())
      continue;

    S.Header.PointerToRelocations = CBA.getOffset();
    if (S.Header.Characteristics & COFF::IMAGE_SCN_LNK_NRELOC_OVFL) {
      S.Header.NumberOfRelocations = 0xffff;
      CBA.write<uint32_t>(/*VirtualAddress=*/S.Relocations.size() + 1,
                          LittleEndian);
      CBA.write<uint32_t>(/*SymbolTableIndex=*/0, LittleEndian);
      CBA.write<uint16_t>(/*Type=*/0, LittleEndian);
    } else {
      S.Header.NumberOfRelocations = S.Relocations.size();
    }

    for (const COFFYAML::Relocation &R : S.Relocations) {
      uint32_t SymbolTableIndex;
      if (R.SymbolTableIndex) {
        if (!R.SymbolName.empty())
          WithColor::error()
              << "Both SymbolName and SymbolTableIndex specified\n";
        SymbolTableIndex = *R.SymbolTableIndex;
      } else {
        SymbolTableIndex = SymbolTableIndexMap[R.SymbolName];
      }
      CBA.write(R.VirtualAddress, LittleEndian);
      CBA.write(SymbolTableIndex, LittleEndian);
      CBA.write(R.Type, LittleEndian);
    }
  }

  // Fill in the optional header now that the section layout is final.
  if (CP.isPE()) {
    if (CP.is64Bit()) {
      object::pe32plus_header PEH;
      initializeOptionalHeader(CP, COFF::PE32Header::PE32_PLUS, &PEH);
      CBA.updateDataAt(OptionalHeaderOffset, &PEH, sizeof(PEH));
    } else {
      object::pe32_header PEH;
      uint32_t BaseOfData =
          initializeOptionalHeader(CP, COFF::PE32Header::PE32, &PEH);
      PEH.BaseOfData = BaseOfData;
      CBA.updateDataAt(OptionalHeaderOffset, &PEH, sizeof(PEH));
    }
  }

  // Fill in the section table.
  static_assert(sizeof(object::coff_section) == COFF::SectionSize,
                "unexpected COFF section header size");
  uint64_t SectionHeaderOffset = CP.SectionTableStart;
  for (const COFFYAML::Section &S : CP.Obj.Sections) {
    object::coff_section Header{};
    memcpy(Header.Name, S.Header.Name, COFF::NameSize);
    Header.VirtualSize = S.Header.VirtualSize;
    Header.VirtualAddress = S.Header.VirtualAddress;
    Header.SizeOfRawData = S.Header.SizeOfRawData;
    Header.PointerToRawData = S.Header.PointerToRawData;
    Header.PointerToRelocations = S.Header.PointerToRelocations;
    Header.PointerToLinenumbers = S.Header.PointerToLineNumbers;
    Header.NumberOfRelocations = S.Header.NumberOfRelocations;
    Header.NumberOfLinenumbers = S.Header.NumberOfLineNumbers;
    Header.Characteristics = S.Header.Characteristics;
    CBA.updateDataAt(SectionHeaderOffset, &Header, sizeof(Header));
    SectionHeaderOffset += sizeof(Header);
  }

  // Output symbol table.
  if (CP.Obj.Header.NumberOfSymbols || CP.StringTable.size() > 4)
    CP.Obj.Header.PointerToSymbolTable = CBA.getOffset();
  else
    CP.Obj.Header.PointerToSymbolTable = 0;

  CBA.updateDataAt(PointerToSymbolTableOffset,
                   CP.Obj.Header.PointerToSymbolTable, LittleEndian);

  for (std::vector<COFFYAML::Symbol>::const_iterator i = CP.Obj.Symbols.begin(),
                                                     e = CP.Obj.Symbols.end();
       i != e; ++i) {
    CBA.write(i->Header.Name, COFF::NameSize);
    CBA.write(i->Header.Value, LittleEndian);
    if (CP.useBigObj())
      CBA.write(i->Header.SectionNumber, LittleEndian);
    else
      CBA.write(static_cast<int16_t>(i->Header.SectionNumber), LittleEndian);
    CBA.write(i->Header.Type, LittleEndian);
    CBA.write(i->Header.StorageClass, LittleEndian);
    CBA.write(i->Header.NumberOfAuxSymbols, LittleEndian);

    if (i->FunctionDefinition) {
      CBA.write(i->FunctionDefinition->TagIndex, LittleEndian);
      CBA.write(i->FunctionDefinition->TotalSize, LittleEndian);
      CBA.write(i->FunctionDefinition->PointerToLinenumber, LittleEndian);
      CBA.write(i->FunctionDefinition->PointerToNextFunction, LittleEndian);
      CBA.writeZeros(sizeof(i->FunctionDefinition->unused));
      CBA.writeZeros(CP.getSymbolSize() - COFF::Symbol16Size);
    }
    if (i->bfAndefSymbol) {
      CBA.writeZeros(sizeof(i->bfAndefSymbol->unused1));
      CBA.write(i->bfAndefSymbol->Linenumber, LittleEndian);
      CBA.writeZeros(sizeof(i->bfAndefSymbol->unused2));
      CBA.write(i->bfAndefSymbol->PointerToNextFunction, LittleEndian);
      CBA.writeZeros(sizeof(i->bfAndefSymbol->unused3));
      CBA.writeZeros(CP.getSymbolSize() - COFF::Symbol16Size);
    }
    if (i->WeakExternal) {
      CBA.write(i->WeakExternal->TagIndex, LittleEndian);
      CBA.write(i->WeakExternal->Characteristics, LittleEndian);
      CBA.writeZeros(sizeof(i->WeakExternal->unused));
      CBA.writeZeros(CP.getSymbolSize() - COFF::Symbol16Size);
    }
    if (!i->File.empty()) {
      unsigned SymbolSize = CP.getSymbolSize();
      uint32_t NumberOfAuxRecords =
          (i->File.size() + SymbolSize - 1) / SymbolSize;
      uint32_t NumberOfAuxBytes = NumberOfAuxRecords * SymbolSize;
      uint32_t NumZeros = NumberOfAuxBytes - i->File.size();
      CBA.write(i->File.data(), i->File.size());
      CBA.writeZeros(NumZeros);
    }
    if (i->SectionDefinition) {
      CBA.write(i->SectionDefinition->Length, LittleEndian);
      CBA.write(i->SectionDefinition->NumberOfRelocations, LittleEndian);
      CBA.write(i->SectionDefinition->NumberOfLinenumbers, LittleEndian);
      CBA.write(i->SectionDefinition->CheckSum, LittleEndian);
      CBA.write(static_cast<int16_t>(i->SectionDefinition->Number),
                LittleEndian);
      CBA.write(i->SectionDefinition->Selection, LittleEndian);
      CBA.writeZeros(sizeof(i->SectionDefinition->unused));
      CBA.write(static_cast<int16_t>(i->SectionDefinition->Number >> 16),
                LittleEndian);
      CBA.writeZeros(CP.getSymbolSize() - COFF::Symbol16Size);
    }
    if (i->CLRToken) {
      CBA.write(i->CLRToken->AuxType, LittleEndian);
      CBA.writeZeros(sizeof(i->CLRToken->unused1));
      CBA.write(i->CLRToken->SymbolTableIndex, LittleEndian);
      CBA.writeZeros(sizeof(i->CLRToken->unused2));
      CBA.writeZeros(CP.getSymbolSize() - COFF::Symbol16Size);
    }
  }

  // Output string table.
  if (CP.Obj.Header.PointerToSymbolTable) {
    *reinterpret_cast<support::ulittle32_t *>(CP.StringTable.data()) =
        CP.StringTable.size();
    CBA.write(CP.StringTable.data(), CP.StringTable.size());
  }
  return true;
}

size_t COFFYAML::SectionDataEntry::size() const {
  size_t Size = Binary.binary_size();
  if (UInt32)
    Size += sizeof(*UInt32);
  if (LoadConfig32)
    Size += LoadConfig32->Size;
  if (LoadConfig64)
    Size += LoadConfig64->Size;
  return Size;
}

template <typename T>
static void writeLoadConfig(T &S, ContiguousBlobAccumulator &CBA) {
  CBA.write(reinterpret_cast<const char *>(&S),
            std::min(sizeof(S), static_cast<size_t>(S.Size)));
  if (sizeof(S) < S.Size)
    CBA.writeZeros(S.Size - sizeof(S));
}

void COFFYAML::SectionDataEntry::writeAsBinary(
    ContiguousBlobAccumulator &CBA) const {
  if (UInt32)
    CBA.write(*UInt32, LittleEndian);
  CBA.writeAsBinary(Binary);
  if (LoadConfig32)
    writeLoadConfig(*LoadConfig32, CBA);
  if (LoadConfig64)
    writeLoadConfig(*LoadConfig64, CBA);
}

namespace llvm {
namespace yaml {

bool yaml2coff(llvm::COFFYAML::Object &Doc, raw_ostream &Out,
               ErrorHandler ErrHandler, uint64_t MaxSize) {
  COFFParser CP(Doc, ErrHandler);
  if (!CP.parse()) {
    ErrHandler("failed to parse YAML file");
    return false;
  }

  // Limit the output size to guard against a runaway YAML description.
  ContiguousBlobAccumulator CBA(/*BaseOffset=*/0, MaxSize);
  if (!writeCOFF(CP, CBA)) {
    ErrHandler("failed to write COFF file");
    return false;
  }
  if (Error E = CBA.takeLimitError()) {
    // Match ELF by reporting a custom error message instead below.
    consumeError(std::move(E));
    ErrHandler("the desired output size is greater than permitted. Use the "
               "--max-size option to change the limit");
    return false;
  }

  CBA.writeBlobToStream(Out);
  return true;
}

} // namespace yaml
} // namespace llvm
