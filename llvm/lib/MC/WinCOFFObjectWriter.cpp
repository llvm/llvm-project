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

#define DEBUG_TYPE "WinCOFFObjectWriter"

#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCWinCOFFObjectWriter.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "llvm/Support/COFF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include "llvm/Support/TimeValue.h"

#include <cstdio>

using namespace llvm;

namespace {
typedef llvm::SmallString<COFF::NameSize> name;

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

  typedef llvm::SmallVector<AuxSymbol, 1> AuxiliarySymbols;

  name             Name;
  int              Index;
  AuxiliarySymbols Aux;
  COFFSymbol      *Other;
  COFFSection     *Section;
  int              Relocations;

  MCSymbolData const *MCData;

  COFFSymbol(llvm::StringRef name);
  size_t size() const;
  void set_name_offset(uint32_t Offset);

  bool should_keep() const;
};

// This class contains staging data for a COFF relocation entry.
struct COFFRelocation {
  COFF::relocation Data;
  COFFSymbol          *Symb;

  COFFRelocation() : Symb(NULL) {}
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

  COFFSection(llvm::StringRef name);
  static size_t size();
};

// This class holds the COFF string table.
class StringTable {
  typedef llvm::StringMap<size_t> map;
  map Map;

  void update_length();
public:
  std::vector<char> Data;

  StringTable();
  size_t size() const;
  size_t insert(llvm::StringRef String);
};

class WinCOFFObjectWriter : public MCObjectWriter {
public:

  typedef std::vector<COFFSymbol*>  symbols;
  typedef std::vector<COFFSection*> sections;

  typedef DenseMap<MCSymbol  const *, COFFSymbol *>   symbol_map;
  typedef DenseMap<MCSection const *, COFFSection *> section_map;

  llvm::OwningPtr<MCWinCOFFObjectTargetWriter> TargetObjectWriter;

  // Root level file contents.
  COFF::header Header;
  sections     Sections;
  symbols      Symbols;
  StringTable  Strings;

  // Maps used during object file creation.
  section_map SectionMap;
  symbol_map  SymbolMap;

  WinCOFFObjectWriter(MCWinCOFFObjectTargetWriter *MOTW, raw_ostream &OS);
  ~WinCOFFObjectWriter();

  COFFSymbol *createSymbol(StringRef Name);
  COFFSymbol *GetOrCreateCOFFSymbol(const MCSymbol * Symbol);
  COFFSection *createSection(StringRef Name);

  template <typename object_t, typename list_t>
  object_t *createCOFFEntity(llvm::StringRef Name, list_t &List);

  void DefineSection(MCSectionData const &SectionData);
  void DefineSymbol(MCSymbolData const &SymbolData, MCAssembler &Assembler);

  void MakeSymbolReal(COFFSymbol &S, size_t Index);
  void MakeSectionReal(COFFSection &S, size_t Number);

  bool ExportSection(COFFSection const *S);
  bool ExportSymbol(MCSymbolData const &SymbolData, MCAssembler &Asm);

  bool IsPhysicalSection(COFFSection *S);

  // Entity writing methods.

  void WriteFileHeader(const COFF::header &Header);
  void WriteSymbol(const COFFSymbol *S);
  void WriteAuxiliarySymbols(const COFFSymbol::AuxiliarySymbols &S);
  void WriteSectionHeader(const COFF::section &S);
  void WriteRelocation(const COFF::relocation &R);

  // MCObjectWriter interface implementation.

  void ExecutePostLayoutBinding(MCAssembler &Asm, const MCAsmLayout &Layout);

  void RecordRelocation(const MCAssembler &Asm,
                        const MCAsmLayout &Layout,
                        const MCFragment *Fragment,
                        const MCFixup &Fixup,
                        MCValue Target,
                        uint64_t &FixedValue);

  void WriteObject(MCAssembler &Asm, const MCAsmLayout &Layout);
};
}

static inline void write_uint32_le(void *Data, uint32_t const &Value) {
  uint8_t *Ptr = reinterpret_cast<uint8_t *>(Data);
  Ptr[0] = (Value & 0x000000FF) >>  0;
  Ptr[1] = (Value & 0x0000FF00) >>  8;
  Ptr[2] = (Value & 0x00FF0000) >> 16;
  Ptr[3] = (Value & 0xFF000000) >> 24;
}

static inline void write_uint16_le(void *Data, uint16_t const &Value) {
  uint8_t *Ptr = reinterpret_cast<uint8_t *>(Data);
  Ptr[0] = (Value & 0x00FF) >> 0;
  Ptr[1] = (Value & 0xFF00) >> 8;
}

static inline void write_uint8_le(void *Data, uint8_t const &Value) {
  uint8_t *Ptr = reinterpret_cast<uint8_t *>(Data);
  Ptr[0] = (Value & 0xFF) >> 0;
}

//------------------------------------------------------------------------------
// Symbol class implementation

COFFSymbol::COFFSymbol(llvm::StringRef name)
  : Name(name.begin(), name.end())
  , Other(NULL)
  , Section(NULL)
  , Relocations(0)
  , MCData(NULL) {
  memset(&Data, 0, sizeof(Data));
}

size_t COFFSymbol::size() const {
  return COFF::SymbolSize + (Data.NumberOfAuxSymbols * COFF::SymbolSize);
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
  if (Section == NULL)
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

COFFSection::COFFSection(llvm::StringRef name)
  : Name(name)
  , MCData(NULL)
  , Symbol(NULL) {
  memset(&Header, 0, sizeof(Header));
}

size_t COFFSection::size() {
  return COFF::SectionSize;
}

//------------------------------------------------------------------------------
// StringTable class implementation

/// Write the length of the string table into Data.
/// The length of the string table includes uint32 length header.
void StringTable::update_length() {
  write_uint32_le(&Data.front(), Data.size());
}

StringTable::StringTable() {
  // The string table data begins with the length of the entire string table
  // including the length header. Allocate space for this header.
  Data.resize(4);
  update_length();
}

size_t StringTable::size() const {
  return Data.size();
}

/// Add String to the table iff it is not already there.
/// @returns the index into the string table where the string is now located.
size_t StringTable::insert(llvm::StringRef String) {
  map::iterator i = Map.find(String);

  if (i != Map.end())
    return i->second;

  size_t Offset = Data.size();

  // Insert string data into string table.
  Data.insert(Data.end(), String.begin(), String.end());
  Data.push_back('\0');

  // Put a reference to it in the map.
  Map[String] = Offset;

  // Update the internal length field.
  update_length();

  return Offset;
}

//------------------------------------------------------------------------------
// WinCOFFObjectWriter class implementation

WinCOFFObjectWriter::WinCOFFObjectWriter(MCWinCOFFObjectTargetWriter *MOTW,
                                         raw_ostream &OS)
  : MCObjectWriter(OS, true)
  , TargetObjectWriter(MOTW) {
  memset(&Header, 0, sizeof(Header));

  Header.Machine = TargetObjectWriter->getMachine();
}

WinCOFFObjectWriter::~WinCOFFObjectWriter() {
  for (symbols::iterator I = Symbols.begin(), E = Symbols.end(); I != E; ++I)
    delete *I;
  for (sections::iterator I = Sections.begin(), E = Sections.end(); I != E; ++I)
    delete *I;
}

COFFSymbol *WinCOFFObjectWriter::createSymbol(StringRef Name) {
  return createCOFFEntity<COFFSymbol>(Name, Symbols);
}

COFFSymbol *WinCOFFObjectWriter::GetOrCreateCOFFSymbol(const MCSymbol * Symbol){
  symbol_map::iterator i = SymbolMap.find(Symbol);
  if (i != SymbolMap.end())
    return i->second;
  COFFSymbol *RetSymbol
    = createCOFFEntity<COFFSymbol>(Symbol->getName(), Symbols);
  SymbolMap[Symbol] = RetSymbol;
  return RetSymbol;
}

COFFSection *WinCOFFObjectWriter::createSection(llvm::StringRef Name) {
  return createCOFFEntity<COFFSection>(Name, Sections);
}

/// A template used to lookup or create a symbol/section, and initialize it if
/// needed.
template <typename object_t, typename list_t>
object_t *WinCOFFObjectWriter::createCOFFEntity(llvm::StringRef Name,
                                                list_t &List) {
  object_t *Object = new object_t(Name);

  List.push_back(Object);

  return Object;
}

/// This function takes a section data object from the assembler
/// and creates the associated COFF section staging object.
void WinCOFFObjectWriter::DefineSection(MCSectionData const &SectionData) {
  assert(SectionData.getSection().getVariant() == MCSection::SV_COFF
    && "Got non COFF section in the COFF backend!");
  // FIXME: Not sure how to verify this (at least in a debug build).
  MCSectionCOFF const &Sec =
    static_cast<MCSectionCOFF const &>(SectionData.getSection());

  COFFSection *coff_section = createSection(Sec.getSectionName());
  COFFSymbol  *coff_symbol = createSymbol(Sec.getSectionName());

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

/// This function takes a section data object from the assembler
/// and creates the associated COFF symbol staging object.
void WinCOFFObjectWriter::DefineSymbol(MCSymbolData const &SymbolData,
                                       MCAssembler &Assembler) {
  COFFSymbol *coff_symbol = GetOrCreateCOFFSymbol(&SymbolData.getSymbol());

  coff_symbol->Data.Type         = (SymbolData.getFlags() & 0x0000FFFF) >>  0;
  coff_symbol->Data.StorageClass = (SymbolData.getFlags() & 0x00FF0000) >> 16;

  if (SymbolData.getFlags() & COFF::SF_WeakExternal) {
    coff_symbol->Data.StorageClass = COFF::IMAGE_SYM_CLASS_WEAK_EXTERNAL;

    if (SymbolData.getSymbol().isVariable()) {
      coff_symbol->Data.StorageClass = COFF::IMAGE_SYM_CLASS_WEAK_EXTERNAL;
      const MCExpr *Value = SymbolData.getSymbol().getVariableValue();

      // FIXME: This assert message isn't very good.
      assert(Value->getKind() == MCExpr::SymbolRef &&
              "Value must be a SymbolRef!");

      const MCSymbolRefExpr *SymbolRef =
        static_cast<const MCSymbolRefExpr *>(Value);
      coff_symbol->Other = GetOrCreateCOFFSymbol(&SymbolRef->getSymbol());
    } else {
      std::string WeakName = std::string(".weak.")
                           +  SymbolData.getSymbol().getName().str()
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
  }

  // If no storage class was specified in the streamer, define it here.
  if (coff_symbol->Data.StorageClass == 0) {
    bool external = SymbolData.isExternal() || (SymbolData.Fragment == NULL);

    coff_symbol->Data.StorageClass =
      external ? COFF::IMAGE_SYM_CLASS_EXTERNAL : COFF::IMAGE_SYM_CLASS_STATIC;
  }

  if (SymbolData.Fragment != NULL)
    coff_symbol->Section =
      SectionMap[&SymbolData.Fragment->getParent()->getSection()];

  // Bind internal COFF symbol to MC symbol.
  coff_symbol->MCData = &SymbolData;
  SymbolMap[&SymbolData.getSymbol()] = coff_symbol;
}

/// making a section real involves assigned it a number and putting
/// name into the string table if needed
void WinCOFFObjectWriter::MakeSectionReal(COFFSection &S, size_t Number) {
  if (S.Name.size() > COFF::NameSize) {
    size_t StringTableEntry = Strings.insert(S.Name.c_str());

    // FIXME: Why is this number 999999? This number is never mentioned in the
    // spec. I'm assuming this is due to the printed value needing to fit into
    // the S.Header.Name field. In which case why not 9999999 (7 9's instead of
    // 6)? The spec does not state if this entry should be null terminated in
    // this case, and thus this seems to be the best way to do it. I think I
    // just solved my own FIXME...
    if (StringTableEntry > 999999)
      report_fatal_error("COFF string table is greater than 999999 bytes.");

    std::sprintf(S.Header.Name, "/%d", unsigned(StringTableEntry));
  } else
    std::memcpy(S.Header.Name, S.Name.c_str(), S.Name.size());

  S.Number = Number;
  S.Symbol->Data.SectionNumber = S.Number;
  S.Symbol->Aux[0].Aux.SectionDefinition.Number = S.Number;
}

void WinCOFFObjectWriter::MakeSymbolReal(COFFSymbol &S, size_t Index) {
  if (S.Name.size() > COFF::NameSize) {
    size_t StringTableEntry = Strings.insert(S.Name.c_str());

    S.set_name_offset(StringTableEntry);
  } else
    std::memcpy(S.Data.Name, S.Name.c_str(), S.Name.size());
  S.Index = Index;
}

bool WinCOFFObjectWriter::ExportSection(COFFSection const *S) {
  return !S->MCData->getFragmentList().empty();
}

bool WinCOFFObjectWriter::ExportSymbol(MCSymbolData const &SymbolData,
                                       MCAssembler &Asm) {
  // This doesn't seem to be right. Strings referred to from the .data section
  // need symbols so they can be linked to code in the .text section right?

  // return Asm.isSymbolLinkerVisible (&SymbolData);

  // For now, all non-variable symbols are exported,
  // the linker will sort the rest out for us.
  return SymbolData.isExternal() || !SymbolData.getSymbol().isVariable();
}

bool WinCOFFObjectWriter::IsPhysicalSection(COFFSection *S) {
  return (S->Header.Characteristics
         & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA) == 0;
}

//------------------------------------------------------------------------------
// entity writing methods

void WinCOFFObjectWriter::WriteFileHeader(const COFF::header &Header) {
  WriteLE16(Header.Machine);
  WriteLE16(Header.NumberOfSections);
  WriteLE32(Header.TimeDateStamp);
  WriteLE32(Header.PointerToSymbolTable);
  WriteLE32(Header.NumberOfSymbols);
  WriteLE16(Header.SizeOfOptionalHeader);
  WriteLE16(Header.Characteristics);
}

void WinCOFFObjectWriter::WriteSymbol(const COFFSymbol *S) {
  WriteBytes(StringRef(S->Data.Name, COFF::NameSize));
  WriteLE32(S->Data.Value);
  WriteLE16(S->Data.SectionNumber);
  WriteLE16(S->Data.Type);
  Write8(S->Data.StorageClass);
  Write8(S->Data.NumberOfAuxSymbols);
  WriteAuxiliarySymbols(S->Aux);
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
      break;
    case ATbfAndefSymbol:
      WriteZeros(sizeof(i->Aux.bfAndefSymbol.unused1));
      WriteLE16(i->Aux.bfAndefSymbol.Linenumber);
      WriteZeros(sizeof(i->Aux.bfAndefSymbol.unused2));
      WriteLE32(i->Aux.bfAndefSymbol.PointerToNextFunction);
      WriteZeros(sizeof(i->Aux.bfAndefSymbol.unused3));
      break;
    case ATWeakExternal:
      WriteLE32(i->Aux.WeakExternal.TagIndex);
      WriteLE32(i->Aux.WeakExternal.Characteristics);
      WriteZeros(sizeof(i->Aux.WeakExternal.unused));
      break;
    case ATFile:
      WriteBytes(StringRef(reinterpret_cast<const char *>(i->Aux.File.FileName),
                 sizeof(i->Aux.File.FileName)));
      break;
    case ATSectionDefinition:
      WriteLE32(i->Aux.SectionDefinition.Length);
      WriteLE16(i->Aux.SectionDefinition.NumberOfRelocations);
      WriteLE16(i->Aux.SectionDefinition.NumberOfLinenumbers);
      WriteLE32(i->Aux.SectionDefinition.CheckSum);
      WriteLE16(i->Aux.SectionDefinition.Number);
      Write8(i->Aux.SectionDefinition.Selection);
      WriteZeros(sizeof(i->Aux.SectionDefinition.unused));
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

  for (MCAssembler::const_iterator i = Asm.begin(), e = Asm.end(); i != e; i++)
    DefineSection(*i);

  for (MCAssembler::const_symbol_iterator i = Asm.symbol_begin(),
                                          e = Asm.symbol_end(); i != e; i++) {
    if (ExportSymbol(*i, Asm))
      DefineSymbol(*i, Asm);
  }
}

void WinCOFFObjectWriter::RecordRelocation(const MCAssembler &Asm,
                                           const MCAsmLayout &Layout,
                                           const MCFragment *Fragment,
                                           const MCFixup &Fixup,
                                           MCValue Target,
                                           uint64_t &FixedValue) {
  assert(Target.getSymA() != NULL && "Relocation must reference a symbol!");

  const MCSymbol *A = &Target.getSymA()->getSymbol();
  MCSymbolData &A_SD = Asm.getSymbolData(*A);

  MCSectionData const *SectionData = Fragment->getParent();

  // Mark this symbol as requiring an entry in the symbol table.
  assert(SectionMap.find(&SectionData->getSection()) != SectionMap.end() &&
         "Section must already have been defined in ExecutePostLayoutBinding!");
  assert(SymbolMap.find(&A_SD.getSymbol()) != SymbolMap.end() &&
         "Symbol must already have been defined in ExecutePostLayoutBinding!");

  COFFSection *coff_section = SectionMap[&SectionData->getSection()];
  COFFSymbol *coff_symbol = SymbolMap[&A_SD.getSymbol()];
  const MCSymbolRefExpr *SymA = Target.getSymA();
  const MCSymbolRefExpr *SymB = Target.getSymB();
  const bool CrossSection = SymB &&
    &SymA->getSymbol().getSection() != &SymB->getSymbol().getSection();

  if (Target.getSymB()) {
    const MCSymbol *B = &Target.getSymB()->getSymbol();
    MCSymbolData &B_SD = Asm.getSymbolData(*B);

    // Offset of the symbol in the section
    int64_t a = Layout.getSymbolOffset(&B_SD);

    // Ofeset of the relocation in the section
    int64_t b = Layout.getFragmentOffset(Fragment) + Fixup.getOffset();

    FixedValue = b - a;
    // In the case where we have SymbA and SymB, we just need to store the delta
    // between the two symbols.  Update FixedValue to account for the delta, and
    // skip recording the relocation.
    if (!CrossSection)
      return;
  } else {
    FixedValue = Target.getConstant();
  }

  COFFRelocation Reloc;

  Reloc.Data.SymbolTableIndex = 0;
  Reloc.Data.VirtualAddress = Layout.getFragmentOffset(Fragment);

  // Turn relocations for temporary symbols into section relocations.
  if (coff_symbol->MCData->getSymbol().isTemporary() || CrossSection) {
    Reloc.Symb = coff_symbol->Section->Symbol;
    FixedValue += Layout.getFragmentOffset(coff_symbol->MCData->Fragment)
                + coff_symbol->MCData->getOffset();
  } else
    Reloc.Symb = coff_symbol;

  ++Reloc.Symb->Relocations;

  Reloc.Data.VirtualAddress += Fixup.getOffset();

  unsigned FixupKind = Fixup.getKind();

  if (CrossSection)
    FixupKind = FK_PCRel_4;

  Reloc.Data.Type = TargetObjectWriter->getRelocType(FixupKind);

  // FIXME: Can anyone explain what this does other than adjust for the size
  // of the offset?
  if (Reloc.Data.Type == COFF::IMAGE_REL_AMD64_REL32 ||
      Reloc.Data.Type == COFF::IMAGE_REL_I386_REL32)
    FixedValue += 4;

  coff_section->Relocations.push_back(Reloc);
}

void WinCOFFObjectWriter::WriteObject(MCAssembler &Asm,
                                      const MCAsmLayout &Layout) {
  // Assign symbol and section indexes and offsets.
  Header.NumberOfSections = 0;

  for (sections::iterator i = Sections.begin(),
                          e = Sections.end(); i != e; i++) {
    if (Layout.getSectionAddressSize((*i)->MCData) > 0) {
      MakeSectionReal(**i, ++Header.NumberOfSections);
    } else {
      (*i)->Number = -1;
    }
  }

  Header.NumberOfSymbols = 0;

  for (symbols::iterator i = Symbols.begin(), e = Symbols.end(); i != e; i++) {
    COFFSymbol *coff_symbol = *i;
    MCSymbolData const *SymbolData = coff_symbol->MCData;

    // Update section number & offset for symbols that have them.
    if ((SymbolData != NULL) && (SymbolData->Fragment != NULL)) {
      assert(coff_symbol->Section != NULL);

      coff_symbol->Data.SectionNumber = coff_symbol->Section->Number;
      coff_symbol->Data.Value = Layout.getFragmentOffset(SymbolData->Fragment)
                              + SymbolData->Offset;
    }

    if (coff_symbol->should_keep()) {
      MakeSymbolReal(*coff_symbol, Header.NumberOfSymbols++);

      // Update auxiliary symbol info.
      coff_symbol->Data.NumberOfAuxSymbols = coff_symbol->Aux.size();
      Header.NumberOfSymbols += coff_symbol->Data.NumberOfAuxSymbols;
    } else
      coff_symbol->Index = -1;
  }

  // Fixup weak external references.
  for (symbols::iterator i = Symbols.begin(), e = Symbols.end(); i != e; i++) {
    COFFSymbol *coff_symbol = *i;
    if (coff_symbol->Other != NULL) {
      assert(coff_symbol->Index != -1);
      assert(coff_symbol->Aux.size() == 1 &&
             "Symbol must contain one aux symbol!");
      assert(coff_symbol->Aux[0].AuxType == ATWeakExternal &&
             "Symbol's aux symbol must be a Weak External!");
      coff_symbol->Aux[0].Aux.WeakExternal.TagIndex = coff_symbol->Other->Index;
    }
  }

  // Assign file offsets to COFF object file structures.

  unsigned offset = 0;

  offset += COFF::HeaderSize;
  offset += COFF::SectionSize * Header.NumberOfSections;

  for (MCAssembler::const_iterator i = Asm.begin(),
                                   e = Asm.end();
                                   i != e; i++) {
    COFFSection *Sec = SectionMap[&i->getSection()];

    if (Sec->Number == -1)
      continue;

    Sec->Header.SizeOfRawData = Layout.getSectionAddressSize(i);

    if (IsPhysicalSection(Sec)) {
      Sec->Header.PointerToRawData = offset;

      offset += Sec->Header.SizeOfRawData;
    }

    if (Sec->Relocations.size() > 0) {
      bool RelocationsOverflow = Sec->Relocations.size() >= 0xffff;

      if (RelocationsOverflow) {
        // Signal overflow by setting NumberOfSections to max value. Actual
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

      for (relocations::iterator cr = Sec->Relocations.begin(),
                                 er = Sec->Relocations.end();
                                 cr != er; ++cr) {
        assert((*cr).Symb->Index != -1);
        (*cr).Data.SymbolTableIndex = (*cr).Symb->Index;
      }
    }

    assert(Sec->Symbol->Aux.size() == 1
      && "Section's symbol must have one aux!");
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

  Header.TimeDateStamp = sys::TimeValue::now().toEpochTime();

  // Write it all to disk...
  WriteFileHeader(Header);

  {
    sections::iterator i, ie;
    MCAssembler::const_iterator j, je;

    for (i = Sections.begin(), ie = Sections.end(); i != ie; i++)
      if ((*i)->Number != -1) {
        if ((*i)->Relocations.size() >= 0xffff) {
          (*i)->Header.Characteristics |= COFF::IMAGE_SCN_LNK_NRELOC_OVFL;
        }
        WriteSectionHeader((*i)->Header);
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

        for (relocations::const_iterator k = (*i)->Relocations.begin(),
                                               ke = (*i)->Relocations.end();
                                               k != ke; k++) {
          WriteRelocation(k->Data);
        }
      } else
        assert((*i)->Header.PointerToRelocations == 0 &&
               "Section::PointerToRelocations is insane!");
    }
  }

  assert(OS.tell() == Header.PointerToSymbolTable &&
         "Header::PointerToSymbolTable is insane!");

  for (symbols::iterator i = Symbols.begin(), e = Symbols.end(); i != e; i++)
    if ((*i)->Index != -1)
      WriteSymbol(*i);

  OS.write((char const *)&Strings.Data.front(), Strings.Data.size());
}

MCWinCOFFObjectTargetWriter::MCWinCOFFObjectTargetWriter(unsigned Machine_) :
  Machine(Machine_) {
}

//------------------------------------------------------------------------------
// WinCOFFObjectWriter factory function

namespace llvm {
  MCObjectWriter *createWinCOFFObjectWriter(MCWinCOFFObjectTargetWriter *MOTW,
                                            raw_ostream &OS) {
    return new WinCOFFObjectWriter(MOTW, OS);
  }
}
