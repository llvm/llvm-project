//===-- llvm/MC/WinCOFFObjectWriter.cpp -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains an implementation of a Win32 COFF object file writer.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCWinCOFFObjectWriter.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TimeValue.h"
#include <cstdio>

using namespace llvm;

#define DEBUG_TYPE "WinCOFFObjectWriter"

namespace {
typedef SmallString<COFF::NameSize> name;

enum AuxiliaryType {
  ATFunctionDefinition,
  ATbfAndefSymbol,
  ATWeakExternal,
  ATFile,
  ATSectionDefinition
};

struct AuxSymbol {
  AuxiliaryType   AuxType;
  COFF::Auxiliary Aux;
};

class COFFSymbol;
class COFFSection;

class COFFSymbol {
public:
  COFF::symbol Data;

  typedef SmallVector<AuxSymbol, 1> AuxiliarySymbols;

  name             Name;
  int              Index;
  AuxiliarySymbols Aux;
  COFFSymbol      *Other;
  COFFSection     *Section;
  int              Relocations;

  MCSymbolData const *MCData;

  COFFSymbol(StringRef name);
  void set_name_offset(uint32_t Offset);

  bool should_keep() const;
};

// This class contains staging data for a COFF relocation entry.
struct COFFRelocation {
  COFF::relocation Data;
  COFFSymbol          *Symb;

  COFFRelocation() : Symb(nullptr) {}
  static size_t size() { return COFF::RelocationSize; }
};

typedef std::vector<COFFRelocation> relocations;

class COFFSection {
public:
  COFF::section Header;

  std::string          Name;
  int                  Number;
  MCSectionData const *MCData;
  COFFSymbol          *Symbol;
  relocations          Relocations;

  COFFSection(StringRef name);
  static size_t size();
};

class WinCOFFObjectWriter : public MCObjectWriter {
public:

  typedef std::vector<std::unique_ptr<COFFSymbol>>  symbols;
  typedef std::vector<std::unique_ptr<COFFSection>> sections;

  typedef DenseMap<MCSymbol  const *, COFFSymbol *>   symbol_map;
  typedef DenseMap<MCSection const *, COFFSection *> section_map;

  std::unique_ptr<MCWinCOFFObjectTargetWriter> TargetObjectWriter;

  // Root level file contents.
  COFF::header Header;
  sections     Sections;
  symbols      Symbols;
  StringTableBuilder Strings;

  // Maps used during object file creation.
  section_map SectionMap;
  symbol_map  SymbolMap;

  bool UseBigObj;

  WinCOFFObjectWriter(MCWinCOFFObjectTargetWriter *MOTW, raw_ostream &OS);
  
  void reset() override {
    memset(&Header, 0, sizeof(Header));
    Header.Machine = TargetObjectWriter->getMachine();
    Sections.clear();
    Symbols.clear();
    Strings.clear();
    SectionMap.clear();
    SymbolMap.clear();
    MCObjectWriter::reset();
  }

  COFFSymbol *createSymbol(StringRef Name);
  COFFSymbol *GetOrCreateCOFFSymbol(const MCSymbol * Symbol);
  COFFSection *createSection(StringRef Name);

  template <typename object_t, typename list_t>
  object_t *createCOFFEntity(StringRef Name, list_t &List);

  void DefineSection(MCSectionData const &SectionData);
  void DefineSymbol(MCSymbolData const &SymbolData, MCAssembler &Assembler,
                    const MCAsmLayout &Layout);

  void SetSymbolName(COFFSymbol &S);
  void SetSectionName(COFFSection &S);

  bool ExportSymbol(const MCSymbol &Symbol, MCAssembler &Asm);

  bool IsPhysicalSection(COFFSection *S);

  // Entity writing methods.

  void WriteFileHeader(const COFF::header &Header);
  void WriteSymbol(const COFFSymbol &S);
  void WriteAuxiliarySymbols(const COFFSymbol::AuxiliarySymbols &S);
  void WriteSectionHeader(const COFF::section &S);
  void WriteRelocation(const COFF::relocation &R);

  // MCObjectWriter interface implementation.

  void ExecutePostLayoutBinding(MCAssembler &Asm,
                                const MCAsmLayout &Layout) override;

  bool IsSymbolRefDifferenceFullyResolvedImpl(const MCAssembler &Asm,
                                              const MCSymbolData &DataA,
                                              const MCFragment &FB, bool InSet,
                                              bool IsPCRel) const override;

  void RecordRelocation(const MCAssembler &Asm, const MCAsmLayout &Layout,
                        const MCFragment *Fragment, const MCFixup &Fixup,
                        MCValue Target, bool &IsPCRel,
                        uint64_t &FixedValue) override;

  void WriteObject(MCAssembler &Asm, const MCAsmLayout &Layout) override;
};
}

static inline void write_uint32_le(void *Data, uint32_t Value) {
  support::endian::write<uint32_t, support::little, support::unaligned>(Data,
                                                                        Value);
}

//------------------------------------------------------------------------------
// Symbol class implementation

COFFSymbol::COFFSymbol(StringRef name)
  : Name(name.begin(), name.end())
  , Other(nullptr)
  , Section(nullptr)
  , Relocations(0)
  , MCData(nullptr) {
  memset(&Data, 0, sizeof(Data));
}

// In the case that the name does not fit within 8 bytes, the offset
// into the string table is stored in the last 4 bytes instead, leaving
// the first 4 bytes as 0.
void COFFSymbol::set_name_offset(uint32_t Offset) {
  write_uint32_le(Data.Name + 0, 0);
  write_uint32_le(Data.Name + 4, Offset);
}

/// logic to decide if the symbol should be reported in the symbol table
bool COFFSymbol::should_keep() const {
  // no section means its external, keep it
  if (!Section)
    return true;

  // if it has relocations pointing at it, keep it
  if (Relocations > 0)   {
    assert(Section->Number != -1 && "Sections with relocations must be real!");
    return true;
  }

  // if the section its in is being droped, drop it
  if (Section->Number == -1)
      return false;

  // if it is the section symbol, keep it
  if (Section->Symbol == this)
    return true;

  // if its temporary, drop it
  if (MCData && MCData->getSymbol().isTemporary())
      return false;

  // otherwise, keep it
  return true;
}

//------------------------------------------------------------------------------
// Section class implementation

COFFSection::COFFSection(StringRef name)
  : Name(name)
  , MCData(nullptr)
  , Symbol(nullptr) {
  memset(&Header, 0, sizeof(Header));
}

size_t COFFSection::size() {
  return COFF::SectionSize;
}

//------------------------------------------------------------------------------
// WinCOFFObjectWriter class implementation

WinCOFFObjectWriter::WinCOFFObjectWriter(MCWinCOFFObjectTargetWriter *MOTW,
                                         raw_ostream &OS)
    : MCObjectWriter(OS, true), TargetObjectWriter(MOTW) {
  memset(&Header, 0, sizeof(Header));

  Header.Machine = TargetObjectWriter->getMachine();
}

COFFSymbol *WinCOFFObjectWriter::createSymbol(StringRef Name) {
  return createCOFFEntity<COFFSymbol>(Name, Symbols);
}

COFFSymbol *WinCOFFObjectWriter::GetOrCreateCOFFSymbol(const MCSymbol *Symbol) {
  symbol_map::iterator i = SymbolMap.find(Symbol);
  if (i != SymbolMap.end())
    return i->second;
  COFFSymbol *RetSymbol =
      createCOFFEntity<COFFSymbol>(Symbol->getName(), Symbols);
  SymbolMap[Symbol] = RetSymbol;
  return RetSymbol;
}

COFFSection *WinCOFFObjectWriter::createSection(StringRef Name) {
  return createCOFFEntity<COFFSection>(Name, Sections);
}

/// A template used to lookup or create a symbol/section, and initialize it if
/// needed.
template <typename object_t, typename list_t>
object_t *WinCOFFObjectWriter::createCOFFEntity(StringRef Name,
                                                list_t &List) {
  List.push_back(make_unique<object_t>(Name));

  return List.back().get();
}

/// This function takes a section data object from the assembler
/// and creates the associated COFF section staging object.
void WinCOFFObjectWriter::DefineSection(MCSectionData const &SectionData) {
  assert(SectionData.getSection().getVariant() == MCSection::SV_COFF
    && "Got non-COFF section in the COFF backend!");
  // FIXME: Not sure how to verify this (at least in a debug build).
  MCSectionCOFF const &Sec =
    static_cast<MCSectionCOFF const &>(SectionData.getSection());

  COFFSection *coff_section = createSection(Sec.getSectionName());
  COFFSymbol  *coff_symbol = createSymbol(Sec.getSectionName());
  if (Sec.getSelection() != COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE) {
    if (const MCSymbol *S = Sec.getCOMDATSymbol()) {
      COFFSymbol *COMDATSymbol = GetOrCreateCOFFSymbol(S);
      if (COMDATSymbol->Section)
        report_fatal_error("two sections have the same comdat");
      COMDATSymbol->Section = coff_section;
    }
  }

  coff_section->Symbol = coff_symbol;
  coff_symbol->Section = coff_section;
  coff_symbol->Data.StorageClass = COFF::IMAGE_SYM_CLASS_STATIC;

  // In this case the auxiliary symbol is a Section Definition.
  coff_symbol->Aux.resize(1);
  memset(&coff_symbol->Aux[0], 0, sizeof(coff_symbol->Aux[0]));
  coff_symbol->Aux[0].AuxType = ATSectionDefinition;
  coff_symbol->Aux[0].Aux.SectionDefinition.Selection = Sec.getSelection();

  coff_section->Header.Characteristics = Sec.getCharacteristics();

  uint32_t &Characteristics = coff_section->Header.Characteristics;
  switch (SectionData.getAlignment()) {
  case 1:    Characteristics |= COFF::IMAGE_SCN_ALIGN_1BYTES;    break;
  case 2:    Characteristics |= COFF::IMAGE_SCN_ALIGN_2BYTES;    break;
  case 4:    Characteristics |= COFF::IMAGE_SCN_ALIGN_4BYTES;    break;
  case 8:    Characteristics |= COFF::IMAGE_SCN_ALIGN_8BYTES;    break;
  case 16:   Characteristics |= COFF::IMAGE_SCN_ALIGN_16BYTES;   break;
  case 32:   Characteristics |= COFF::IMAGE_SCN_ALIGN_32BYTES;   break;
  case 64:   Characteristics |= COFF::IMAGE_SCN_ALIGN_64BYTES;   break;
  case 128:  Characteristics |= COFF::IMAGE_SCN_ALIGN_128BYTES;  break;
  case 256:  Characteristics |= COFF::IMAGE_SCN_ALIGN_256BYTES;  break;
  case 512:  Characteristics |= COFF::IMAGE_SCN_ALIGN_512BYTES;  break;
  case 1024: Characteristics |= COFF::IMAGE_SCN_ALIGN_1024BYTES; break;
  case 2048: Characteristics |= COFF::IMAGE_SCN_ALIGN_2048BYTES; break;
  case 4096: Characteristics |= COFF::IMAGE_SCN_ALIGN_4096BYTES; break;
  case 8192: Characteristics |= COFF::IMAGE_SCN_ALIGN_8192BYTES; break;
  default:
    llvm_unreachable("unsupported section alignment");
  }

  // Bind internal COFF section to MC section.
  coff_section->MCData = &SectionData;
  SectionMap[&SectionData.getSection()] = coff_section;
}

static uint64_t getSymbolValue(const MCSymbolData &Data,
                               const MCAsmLayout &Layout) {
  if (Data.isCommon() && Data.isExternal())
    return Data.getCommonSize();

  uint64_t Res;
  if (!Layout.getSymbolOffset(&Data, Res))
    return 0;

  return Res;
}

/// This function takes a symbol data object from the assembler
/// and creates the associated COFF symbol staging object.
void WinCOFFObjectWriter::DefineSymbol(MCSymbolData const &SymbolData,
                                       MCAssembler &Assembler,
                                       const MCAsmLayout &Layout) {
  MCSymbol const &Symbol = SymbolData.getSymbol();
  COFFSymbol *coff_symbol = GetOrCreateCOFFSymbol(&Symbol);
  SymbolMap[&Symbol] = coff_symbol;

  if (SymbolData.getFlags() & COFF::SF_WeakExternal) {
    coff_symbol->Data.StorageClass = COFF::IMAGE_SYM_CLASS_WEAK_EXTERNAL;

    if (Symbol.isVariable()) {
      const MCSymbolRefExpr *SymRef =
        dyn_cast<MCSymbolRefExpr>(Symbol.getVariableValue());

      if (!SymRef)
        report_fatal_error("Weak externals may only alias symbols");

      coff_symbol->Other = GetOrCreateCOFFSymbol(&SymRef->getSymbol());
    } else {
      std::string WeakName = std::string(".weak.")
                           +  Symbol.getName().str()
                           + ".default";
      COFFSymbol *WeakDefault = createSymbol(WeakName);
      WeakDefault->Data.SectionNumber = COFF::IMAGE_SYM_ABSOLUTE;
      WeakDefault->Data.StorageClass  = COFF::IMAGE_SYM_CLASS_EXTERNAL;
      WeakDefault->Data.Type          = 0;
      WeakDefault->Data.Value         = 0;
      coff_symbol->Other = WeakDefault;
    }

    // Setup the Weak External auxiliary symbol.
    coff_symbol->Aux.resize(1);
    memset(&coff_symbol->Aux[0], 0, sizeof(coff_symbol->Aux[0]));
    coff_symbol->Aux[0].AuxType = ATWeakExternal;
    coff_symbol->Aux[0].Aux.WeakExternal.TagIndex = 0;
    coff_symbol->Aux[0].Aux.WeakExternal.Characteristics =
      COFF::IMAGE_WEAK_EXTERN_SEARCH_LIBRARY;

    coff_symbol->MCData = &SymbolData;
  } else {
    const MCSymbolData &ResSymData = Assembler.getSymbolData(Symbol);
    const MCSymbol *Base = Layout.getBaseSymbol(Symbol);
    coff_symbol->Data.Value = getSymbolValue(ResSymData, Layout);

    coff_symbol->Data.Type         = (ResSymData.getFlags() & 0x0000FFFF) >>  0;
    coff_symbol->Data.StorageClass = (ResSymData.getFlags() & 0x00FF0000) >> 16;

    // If no storage class was specified in the streamer, define it here.
    if (coff_symbol->Data.StorageClass == 0) {
      bool IsExternal =
          ResSymData.isExternal() ||
          (!ResSymData.getFragment() && !ResSymData.getSymbol().isVariable());

      coff_symbol->Data.StorageClass = IsExternal
                                           ? COFF::IMAGE_SYM_CLASS_EXTERNAL
                                           : COFF::IMAGE_SYM_CLASS_STATIC;
    }

    if (!Base) {
      coff_symbol->Data.SectionNumber = COFF::IMAGE_SYM_ABSOLUTE;
    } else {
      const MCSymbolData &BaseData = Assembler.getSymbolData(*Base);
      if (BaseData.getFragment()) {
        COFFSection *Sec =
            SectionMap[&BaseData.getFragment()->getParent()->getSection()];

        if (coff_symbol->Section && coff_symbol->Section != Sec)
          report_fatal_error("conflicting sections for symbol");

        coff_symbol->Section = Sec;
      }
    }

    coff_symbol->MCData = &ResSymData;
  }
}

// Maximum offsets for different string table entry encodings.
static const unsigned Max6DecimalOffset = 999999;
static const unsigned Max7DecimalOffset = 9999999;
static const uint64_t MaxBase64Offset = 0xFFFFFFFFFULL; // 64^6, including 0

// Encode a string table entry offset in base 64, padded to 6 chars, and
// prefixed with a double slash: '//AAAAAA', '//AAAAAB', ...
// Buffer must be at least 8 bytes large. No terminating null appended.
static void encodeBase64StringEntry(char* Buffer, uint64_t Value) {
  assert(Value > Max7DecimalOffset && Value <= MaxBase64Offset &&
         "Illegal section name encoding for value");

  static const char Alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                 "abcdefghijklmnopqrstuvwxyz"
                                 "0123456789+/";

  Buffer[0] = '/';
  Buffer[1] = '/';

  char* Ptr = Buffer + 7;
  for (unsigned i = 0; i < 6; ++i) {
    unsigned Rem = Value % 64;
    Value /= 64;
    *(Ptr--) = Alphabet[Rem];
  }
}

void WinCOFFObjectWriter::SetSectionName(COFFSection &S) {
  if (S.Name.size() > COFF::NameSize) {
    uint64_t StringTableEntry = Strings.getOffset(S.Name);

    if (StringTableEntry <= Max6DecimalOffset) {
      std::sprintf(S.Header.Name, "/%d", unsigned(StringTableEntry));
    } else if (StringTableEntry <= Max7DecimalOffset) {
      // With seven digits, we have to skip the terminating null. Because
      // sprintf always appends it, we use a larger temporary buffer.
      char buffer[9] = { };
      std::sprintf(buffer, "/%d", unsigned(StringTableEntry));
      std::memcpy(S.Header.Name, buffer, 8);
    } else if (StringTableEntry <= MaxBase64Offset) {
      // Starting with 10,000,000, offsets are encoded as base64.
      encodeBase64StringEntry(S.Header.Name, StringTableEntry);
    } else {
      report_fatal_error("COFF string table is greater than 64 GB.");
    }
  } else
    std::memcpy(S.Header.Name, S.Name.c_str(), S.Name.size());
}

void WinCOFFObjectWriter::SetSymbolName(COFFSymbol &S) {
  if (S.Name.size() > COFF::NameSize)
    S.set_name_offset(Strings.getOffset(S.Name));
  else
    std::memcpy(S.Data.Name, S.Name.c_str(), S.Name.size());
}

bool WinCOFFObjectWriter::ExportSymbol(const MCSymbol &Symbol,
                                       MCAssembler &Asm) {
  // This doesn't seem to be right. Strings referred to from the .data section
  // need symbols so they can be linked to code in the .text section right?

  // return Asm.isSymbolLinkerVisible(Symbol);

  // Non-temporary labels should always be visible to the linker.
  if (!Symbol.isTemporary())
    return true;

  // Absolute temporary labels are never visible.
  if (!Symbol.isInSection())
    return false;

  // For now, all non-variable symbols are exported,
  // the linker will sort the rest out for us.
  return !Symbol.isVariable();
}

bool WinCOFFObjectWriter::IsPhysicalSection(COFFSection *S) {
  return (S->Header.Characteristics
         & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA) == 0;
}

//------------------------------------------------------------------------------
// entity writing methods

void WinCOFFObjectWriter::WriteFileHeader(const COFF::header &Header) {
  if (UseBigObj) {
    WriteLE16(COFF::IMAGE_FILE_MACHINE_UNKNOWN);
    WriteLE16(0xFFFF);
    WriteLE16(COFF::BigObjHeader::MinBigObjectVersion);
    WriteLE16(Header.Machine);
    WriteLE32(Header.TimeDateStamp);
    for (uint8_t MagicChar : COFF::BigObjMagic)
      Write8(MagicChar);
    WriteLE32(0);
    WriteLE32(0);
    WriteLE32(0);
    WriteLE32(0);
    WriteLE32(Header.NumberOfSections);
    WriteLE32(Header.PointerToSymbolTable);
    WriteLE32(Header.NumberOfSymbols);
  } else {
    WriteLE16(Header.Machine);
    WriteLE16(static_cast<int16_t>(Header.NumberOfSections));
    WriteLE32(Header.TimeDateStamp);
    WriteLE32(Header.PointerToSymbolTable);
    WriteLE32(Header.NumberOfSymbols);
    WriteLE16(Header.SizeOfOptionalHeader);
    WriteLE16(Header.Characteristics);
  }
}

void WinCOFFObjectWriter::WriteSymbol(const COFFSymbol &S) {
  WriteBytes(StringRef(S.Data.Name, COFF::NameSize));
  WriteLE32(S.Data.Value);
  if (UseBigObj)
    WriteLE32(S.Data.SectionNumber);
  else
    WriteLE16(static_cast<int16_t>(S.Data.SectionNumber));
  WriteLE16(S.Data.Type);
  Write8(S.Data.StorageClass);
  Write8(S.Data.NumberOfAuxSymbols);
  WriteAuxiliarySymbols(S.Aux);
}

void WinCOFFObjectWriter::WriteAuxiliarySymbols(
                                        const COFFSymbol::AuxiliarySymbols &S) {
  for(COFFSymbol::AuxiliarySymbols::const_iterator i = S.begin(), e = S.end();
      i != e; ++i) {
    switch(i->AuxType) {
    case ATFunctionDefinition:
      WriteLE32(i->Aux.FunctionDefinition.TagIndex);
      WriteLE32(i->Aux.FunctionDefinition.TotalSize);
      WriteLE32(i->Aux.FunctionDefinition.PointerToLinenumber);
      WriteLE32(i->Aux.FunctionDefinition.PointerToNextFunction);
      WriteZeros(sizeof(i->Aux.FunctionDefinition.unused));
      if (UseBigObj)
        WriteZeros(COFF::Symbol32Size - COFF::Symbol16Size);
      break;
    case ATbfAndefSymbol:
      WriteZeros(sizeof(i->Aux.bfAndefSymbol.unused1));
      WriteLE16(i->Aux.bfAndefSymbol.Linenumber);
      WriteZeros(sizeof(i->Aux.bfAndefSymbol.unused2));
      WriteLE32(i->Aux.bfAndefSymbol.PointerToNextFunction);
      WriteZeros(sizeof(i->Aux.bfAndefSymbol.unused3));
      if (UseBigObj)
        WriteZeros(COFF::Symbol32Size - COFF::Symbol16Size);
      break;
    case ATWeakExternal:
      WriteLE32(i->Aux.WeakExternal.TagIndex);
      WriteLE32(i->Aux.WeakExternal.Characteristics);
      WriteZeros(sizeof(i->Aux.WeakExternal.unused));
      if (UseBigObj)
        WriteZeros(COFF::Symbol32Size - COFF::Symbol16Size);
      break;
    case ATFile:
      WriteBytes(
          StringRef(reinterpret_cast<const char *>(&i->Aux),
                    UseBigObj ? COFF::Symbol32Size : COFF::Symbol16Size));
      break;
    case ATSectionDefinition:
      WriteLE32(i->Aux.SectionDefinition.Length);
      WriteLE16(i->Aux.SectionDefinition.NumberOfRelocations);
      WriteLE16(i->Aux.SectionDefinition.NumberOfLinenumbers);
      WriteLE32(i->Aux.SectionDefinition.CheckSum);
      WriteLE16(static_cast<int16_t>(i->Aux.SectionDefinition.Number));
      Write8(i->Aux.SectionDefinition.Selection);
      WriteZeros(sizeof(i->Aux.SectionDefinition.unused));
      WriteLE16(static_cast<int16_t>(i->Aux.SectionDefinition.Number >> 16));
      if (UseBigObj)
        WriteZeros(COFF::Symbol32Size - COFF::Symbol16Size);
      break;
    }
  }
}

void WinCOFFObjectWriter::WriteSectionHeader(const COFF::section &S) {
  WriteBytes(StringRef(S.Name, COFF::NameSize));

  WriteLE32(S.VirtualSize);
  WriteLE32(S.VirtualAddress);
  WriteLE32(S.SizeOfRawData);
  WriteLE32(S.PointerToRawData);
  WriteLE32(S.PointerToRelocations);
  WriteLE32(S.PointerToLineNumbers);
  WriteLE16(S.NumberOfRelocations);
  WriteLE16(S.NumberOfLineNumbers);
  WriteLE32(S.Characteristics);
}

void WinCOFFObjectWriter::WriteRelocation(const COFF::relocation &R) {
  WriteLE32(R.VirtualAddress);
  WriteLE32(R.SymbolTableIndex);
  WriteLE16(R.Type);
}

////////////////////////////////////////////////////////////////////////////////
// MCObjectWriter interface implementations

void WinCOFFObjectWriter::ExecutePostLayoutBinding(MCAssembler &Asm,
                                                   const MCAsmLayout &Layout) {
  // "Define" each section & symbol. This creates section & symbol
  // entries in the staging area.
  for (const auto &Section : Asm)
    DefineSection(Section);

  for (MCSymbolData &SD : Asm.symbols())
    if (ExportSymbol(SD.getSymbol(), Asm))
      DefineSymbol(SD, Asm, Layout);
}

bool WinCOFFObjectWriter::IsSymbolRefDifferenceFullyResolvedImpl(
    const MCAssembler &Asm, const MCSymbolData &DataA, const MCFragment &FB,
    bool InSet, bool IsPCRel) const {
  // MS LINK expects to be able to replace all references to a function with a
  // thunk to implement their /INCREMENTAL feature.  Make sure we don't optimize
  // away any relocations to functions.
  if ((((DataA.getFlags() & COFF::SF_TypeMask) >> COFF::SF_TypeShift) >>
       COFF::SCT_COMPLEX_TYPE_SHIFT) == COFF::IMAGE_SYM_DTYPE_FUNCTION)
    return false;
  return MCObjectWriter::IsSymbolRefDifferenceFullyResolvedImpl(Asm, DataA, FB,
                                                                InSet, IsPCRel);
}

void WinCOFFObjectWriter::RecordRelocation(const MCAssembler &Asm,
                                           const MCAsmLayout &Layout,
                                           const MCFragment *Fragment,
                                           const MCFixup &Fixup,
                                           MCValue Target,
                                           bool &IsPCRel,
                                           uint64_t &FixedValue) {
  assert(Target.getSymA() && "Relocation must reference a symbol!");

  const MCSymbol &Symbol = Target.getSymA()->getSymbol();
  const MCSymbol &A = Symbol.AliasedSymbol();
  if (!Asm.hasSymbolData(A))
    Asm.getContext().FatalError(
        Fixup.getLoc(),
        Twine("symbol '") + A.getName() + "' can not be undefined");

  const MCSymbolData &A_SD = Asm.getSymbolData(A);

  MCSectionData const *SectionData = Fragment->getParent();

  // Mark this symbol as requiring an entry in the symbol table.
  assert(SectionMap.find(&SectionData->getSection()) != SectionMap.end() &&
         "Section must already have been defined in ExecutePostLayoutBinding!");
  assert(SymbolMap.find(&A_SD.getSymbol()) != SymbolMap.end() &&
         "Symbol must already have been defined in ExecutePostLayoutBinding!");

  COFFSection *coff_section = SectionMap[&SectionData->getSection()];
  COFFSymbol *coff_symbol = SymbolMap[&A_SD.getSymbol()];
  const MCSymbolRefExpr *SymB = Target.getSymB();
  bool CrossSection = false;

  if (SymB) {
    const MCSymbol *B = &SymB->getSymbol();
    const MCSymbolData &B_SD = Asm.getSymbolData(*B);
    if (!B_SD.getFragment())
      Asm.getContext().FatalError(
          Fixup.getLoc(),
          Twine("symbol '") + B->getName() +
              "' can not be undefined in a subtraction expression");

    if (!A_SD.getFragment())
      Asm.getContext().FatalError(
          Fixup.getLoc(),
          Twine("symbol '") + Symbol.getName() +
              "' can not be undefined in a subtraction expression");

    CrossSection = &Symbol.getSection() != &B->getSection();

    // Offset of the symbol in the section
    int64_t OffsetOfB = Layout.getSymbolOffset(&B_SD);

    // In the case where we have SymbA and SymB, we just need to store the delta
    // between the two symbols.  Update FixedValue to account for the delta, and
    // skip recording the relocation.
    if (!CrossSection) {
      int64_t OffsetOfA = Layout.getSymbolOffset(&A_SD);
      FixedValue = (OffsetOfA - OffsetOfB) + Target.getConstant();
      return;
    }

    // Offset of the relocation in the section
    int64_t OffsetOfRelocation =
        Layout.getFragmentOffset(Fragment) + Fixup.getOffset();

    FixedValue = OffsetOfRelocation - OffsetOfB;
  } else {
    FixedValue = Target.getConstant();
  }

  COFFRelocation Reloc;

  Reloc.Data.SymbolTableIndex = 0;
  Reloc.Data.VirtualAddress = Layout.getFragmentOffset(Fragment);

  // Turn relocations for temporary symbols into section relocations.
  if (coff_symbol->MCData->getSymbol().isTemporary() || CrossSection) {
    Reloc.Symb = coff_symbol->Section->Symbol;
    FixedValue += Layout.getFragmentOffset(coff_symbol->MCData->getFragment()) +
                  coff_symbol->MCData->getOffset();
  } else
    Reloc.Symb = coff_symbol;

  ++Reloc.Symb->Relocations;

  Reloc.Data.VirtualAddress += Fixup.getOffset();
  Reloc.Data.Type = TargetObjectWriter->getRelocType(Target, Fixup,
                                                     CrossSection);

  // FIXME: Can anyone explain what this does other than adjust for the size
  // of the offset?
  if ((Header.Machine == COFF::IMAGE_FILE_MACHINE_AMD64 &&
       Reloc.Data.Type == COFF::IMAGE_REL_AMD64_REL32) ||
      (Header.Machine == COFF::IMAGE_FILE_MACHINE_I386 &&
       Reloc.Data.Type == COFF::IMAGE_REL_I386_REL32))
    FixedValue += 4;

  if (Header.Machine == COFF::IMAGE_FILE_MACHINE_ARMNT) {
    switch (Reloc.Data.Type) {
    case COFF::IMAGE_REL_ARM_ABSOLUTE:
    case COFF::IMAGE_REL_ARM_ADDR32:
    case COFF::IMAGE_REL_ARM_ADDR32NB:
    case COFF::IMAGE_REL_ARM_TOKEN:
    case COFF::IMAGE_REL_ARM_SECTION:
    case COFF::IMAGE_REL_ARM_SECREL:
      break;
    case COFF::IMAGE_REL_ARM_BRANCH11:
    case COFF::IMAGE_REL_ARM_BLX11:
      // IMAGE_REL_ARM_BRANCH11 and IMAGE_REL_ARM_BLX11 are only used for
      // pre-ARMv7, which implicitly rules it out of ARMNT (it would be valid
      // for Windows CE).
    case COFF::IMAGE_REL_ARM_BRANCH24:
    case COFF::IMAGE_REL_ARM_BLX24:
    case COFF::IMAGE_REL_ARM_MOV32A:
      // IMAGE_REL_ARM_BRANCH24, IMAGE_REL_ARM_BLX24, IMAGE_REL_ARM_MOV32A are
      // only used for ARM mode code, which is documented as being unsupported
      // by Windows on ARM.  Empirical proof indicates that masm is able to
      // generate the relocations however the rest of the MSVC toolchain is
      // unable to handle it.
      llvm_unreachable("unsupported relocation");
      break;
    case COFF::IMAGE_REL_ARM_MOV32T:
      break;
    case COFF::IMAGE_REL_ARM_BRANCH20T:
    case COFF::IMAGE_REL_ARM_BRANCH24T:
    case COFF::IMAGE_REL_ARM_BLX23T:
      // IMAGE_REL_BRANCH20T, IMAGE_REL_ARM_BRANCH24T, IMAGE_REL_ARM_BLX23T all
      // perform a 4 byte adjustment to the relocation.  Relative branches are
      // offset by 4 on ARM, however, because there is no RELA relocations, all
      // branches are offset by 4.
      FixedValue = FixedValue + 4;
      break;
    }
  }

  if (TargetObjectWriter->recordRelocation(Fixup))
    coff_section->Relocations.push_back(Reloc);
}

void WinCOFFObjectWriter::WriteObject(MCAssembler &Asm,
                                      const MCAsmLayout &Layout) {
  size_t SectionsSize = Sections.size();
  if (SectionsSize > static_cast<size_t>(INT32_MAX))
    report_fatal_error(
        "PE COFF object files can't have more than 2147483647 sections");

  // Assign symbol and section indexes and offsets.
  int32_t NumberOfSections = static_cast<int32_t>(SectionsSize);

  UseBigObj = NumberOfSections > COFF::MaxNumberOfSections16;

  DenseMap<COFFSection *, int32_t> SectionIndices(
      NextPowerOf2(NumberOfSections));

  // Assign section numbers.
  size_t Number = 1;
  for (const auto &Section : Sections) {
    SectionIndices[Section.get()] = Number;
    Section->Number = Number;
    Section->Symbol->Data.SectionNumber = Number;
    Section->Symbol->Aux[0].Aux.SectionDefinition.Number = Number;
    ++Number;
  }

  Header.NumberOfSections = NumberOfSections;
  Header.NumberOfSymbols = 0;

  for (auto FI = Asm.file_names_begin(), FE = Asm.file_names_end();
       FI != FE; ++FI) {
    // round up to calculate the number of auxiliary symbols required
    unsigned SymbolSize = UseBigObj ? COFF::Symbol32Size : COFF::Symbol16Size;
    unsigned Count = (FI->size() + SymbolSize - 1) / SymbolSize;

    COFFSymbol *file = createSymbol(".file");
    file->Data.SectionNumber = COFF::IMAGE_SYM_DEBUG;
    file->Data.StorageClass = COFF::IMAGE_SYM_CLASS_FILE;
    file->Aux.resize(Count);

    unsigned Offset = 0;
    unsigned Length = FI->size();
    for (auto &Aux : file->Aux) {
      Aux.AuxType = ATFile;

      if (Length > SymbolSize) {
        memcpy(&Aux.Aux, FI->c_str() + Offset, SymbolSize);
        Length = Length - SymbolSize;
      } else {
        memcpy(&Aux.Aux, FI->c_str() + Offset, Length);
        memset((char *)&Aux.Aux + Length, 0, SymbolSize - Length);
        break;
      }

      Offset += SymbolSize;
    }
  }

  for (auto &Symbol : Symbols) {
    // Update section number & offset for symbols that have them.
    if (Symbol->Section)
      Symbol->Data.SectionNumber = Symbol->Section->Number;
    if (Symbol->should_keep()) {
      Symbol->Index = Header.NumberOfSymbols++;
      // Update auxiliary symbol info.
      Symbol->Data.NumberOfAuxSymbols = Symbol->Aux.size();
      Header.NumberOfSymbols += Symbol->Data.NumberOfAuxSymbols;
    } else
      Symbol->Index = -1;
  }

  // Build string table.
  for (const auto &S : Sections)
    if (S->Name.size() > COFF::NameSize)
      Strings.add(S->Name);
  for (const auto &S : Symbols)
    if (S->should_keep() && S->Name.size() > COFF::NameSize)
      Strings.add(S->Name);
  Strings.finalize(StringTableBuilder::WinCOFF);

  // Set names.
  for (const auto &S : Sections)
    SetSectionName(*S);
  for (auto &S : Symbols)
    if (S->should_keep())
      SetSymbolName(*S);

  // Fixup weak external references.
  for (auto &Symbol : Symbols) {
    if (Symbol->Other) {
      assert(Symbol->Index != -1);
      assert(Symbol->Aux.size() == 1 && "Symbol must contain one aux symbol!");
      assert(Symbol->Aux[0].AuxType == ATWeakExternal &&
             "Symbol's aux symbol must be a Weak External!");
      Symbol->Aux[0].Aux.WeakExternal.TagIndex = Symbol->Other->Index;
    }
  }

  // Fixup associative COMDAT sections.
  for (auto &Section : Sections) {
    if (Section->Symbol->Aux[0].Aux.SectionDefinition.Selection !=
        COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE)
      continue;

    const MCSectionCOFF &MCSec =
      static_cast<const MCSectionCOFF &>(Section->MCData->getSection());

    const MCSymbol *COMDAT = MCSec.getCOMDATSymbol();
    assert(COMDAT);
    COFFSymbol *COMDATSymbol = GetOrCreateCOFFSymbol(COMDAT);
    assert(COMDATSymbol);
    COFFSection *Assoc = COMDATSymbol->Section;
    if (!Assoc)
      report_fatal_error(
          Twine("Missing associated COMDAT section for section ") +
          MCSec.getSectionName());

    // Skip this section if the associated section is unused.
    if (Assoc->Number == -1)
      continue;

    Section->Symbol->Aux[0].Aux.SectionDefinition.Number = SectionIndices[Assoc];
  }


  // Assign file offsets to COFF object file structures.

  unsigned offset = 0;

  if (UseBigObj)
    offset += COFF::Header32Size;
  else
    offset += COFF::Header16Size;
  offset += COFF::SectionSize * Header.NumberOfSections;

  for (const auto &Section : Asm) {
    COFFSection *Sec = SectionMap[&Section.getSection()];

    if (Sec->Number == -1)
      continue;

    Sec->Header.SizeOfRawData = Layout.getSectionAddressSize(&Section);

    if (IsPhysicalSection(Sec)) {
      Sec->Header.PointerToRawData = offset;

      offset += Sec->Header.SizeOfRawData;
    }

    if (Sec->Relocations.size() > 0) {
      bool RelocationsOverflow = Sec->Relocations.size() >= 0xffff;

      if (RelocationsOverflow) {
        // Signal overflow by setting NumberOfRelocations to max value. Actual
        // size is found in reloc #0. Microsoft tools understand this.
        Sec->Header.NumberOfRelocations = 0xffff;
      } else {
        Sec->Header.NumberOfRelocations = Sec->Relocations.size();
      }
      Sec->Header.PointerToRelocations = offset;

      if (RelocationsOverflow) {
        // Reloc #0 will contain actual count, so make room for it.
        offset += COFF::RelocationSize;
      }

      offset += COFF::RelocationSize * Sec->Relocations.size();

      for (auto &Relocation : Sec->Relocations) {
        assert(Relocation.Symb->Index != -1);
        Relocation.Data.SymbolTableIndex = Relocation.Symb->Index;
      }
    }

    assert(Sec->Symbol->Aux.size() == 1 &&
           "Section's symbol must have one aux!");
    AuxSymbol &Aux = Sec->Symbol->Aux[0];
    assert(Aux.AuxType == ATSectionDefinition &&
           "Section's symbol's aux symbol must be a Section Definition!");
    Aux.Aux.SectionDefinition.Length = Sec->Header.SizeOfRawData;
    Aux.Aux.SectionDefinition.NumberOfRelocations =
                                                Sec->Header.NumberOfRelocations;
    Aux.Aux.SectionDefinition.NumberOfLinenumbers =
                                                Sec->Header.NumberOfLineNumbers;
  }

  Header.PointerToSymbolTable = offset;

  // We want a deterministic output. It looks like GNU as also writes 0 in here.
  Header.TimeDateStamp = 0;

  // Write it all to disk...
  WriteFileHeader(Header);

  {
    sections::iterator i, ie;
    MCAssembler::const_iterator j, je;

    for (auto &Section : Sections) {
      if (Section->Number != -1) {
        if (Section->Relocations.size() >= 0xffff)
          Section->Header.Characteristics |= COFF::IMAGE_SCN_LNK_NRELOC_OVFL;
        WriteSectionHeader(Section->Header);
      }
    }

    for (i = Sections.begin(), ie = Sections.end(),
         j = Asm.begin(), je = Asm.end();
         (i != ie) && (j != je); ++i, ++j) {

      if ((*i)->Number == -1)
        continue;

      if ((*i)->Header.PointerToRawData != 0) {
        assert(OS.tell() == (*i)->Header.PointerToRawData &&
               "Section::PointerToRawData is insane!");

        Asm.writeSectionData(j, Layout);
      }

      if ((*i)->Relocations.size() > 0) {
        assert(OS.tell() == (*i)->Header.PointerToRelocations &&
               "Section::PointerToRelocations is insane!");

        if ((*i)->Relocations.size() >= 0xffff) {
          // In case of overflow, write actual relocation count as first
          // relocation. Including the synthetic reloc itself (+ 1).
          COFF::relocation r;
          r.VirtualAddress = (*i)->Relocations.size() + 1;
          r.SymbolTableIndex = 0;
          r.Type = 0;
          WriteRelocation(r);
        }

        for (const auto &Relocation : (*i)->Relocations)
          WriteRelocation(Relocation.Data);
      } else
        assert((*i)->Header.PointerToRelocations == 0 &&
               "Section::PointerToRelocations is insane!");
    }
  }

  assert(OS.tell() == Header.PointerToSymbolTable &&
         "Header::PointerToSymbolTable is insane!");

  for (auto &Symbol : Symbols)
    if (Symbol->Index != -1)
      WriteSymbol(*Symbol);

  OS.write(Strings.data().data(), Strings.data().size());
}

MCWinCOFFObjectTargetWriter::MCWinCOFFObjectTargetWriter(unsigned Machine_) :
  Machine(Machine_) {
}

// Pin the vtable to this file.
void MCWinCOFFObjectTargetWriter::anchor() {}

//------------------------------------------------------------------------------
// WinCOFFObjectWriter factory function

namespace llvm {
  MCObjectWriter *createWinCOFFObjectWriter(MCWinCOFFObjectTargetWriter *MOTW,
                                            raw_ostream &OS) {
    return new WinCOFFObjectWriter(MOTW, OS);
  }
}
