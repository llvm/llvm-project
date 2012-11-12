//===- COFFObjectFile.cpp - COFF object file implementation -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the COFFObjectFile class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/COFF.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"

using namespace llvm;
using namespace object;

namespace {
using support::ulittle8_t;
using support::ulittle16_t;
using support::ulittle32_t;
using support::little16_t;
}

namespace {
// Returns false if size is greater than the buffer size. And sets ec.
bool checkSize(const MemoryBuffer *m, error_code &ec, uint64_t size) {
  if (m->getBufferSize() < size) {
    ec = object_error::unexpected_eof;
    return false;
  }
  return true;
}

// Returns false if any bytes in [addr, addr + size) fall outsize of m.
bool checkAddr(const MemoryBuffer *m,
               error_code &ec,
               uintptr_t addr,
               uint64_t size) {
  if (addr + size < addr ||
      addr + size < size ||
      addr + size > uintptr_t(m->getBufferEnd())) {
    ec = object_error::unexpected_eof;
    return false;
  }
  return true;
}
}

const coff_symbol *COFFObjectFile::toSymb(DataRefImpl Symb) const {
  const coff_symbol *addr = reinterpret_cast<const coff_symbol*>(Symb.p);

# ifndef NDEBUG
  // Verify that the symbol points to a valid entry in the symbol table.
  uintptr_t offset = uintptr_t(addr) - uintptr_t(base());
  if (offset < Header->PointerToSymbolTable
      || offset >= Header->PointerToSymbolTable
         + (Header->NumberOfSymbols * sizeof(coff_symbol)))
    report_fatal_error("Symbol was outside of symbol table.");

  assert((offset - Header->PointerToSymbolTable) % sizeof(coff_symbol)
         == 0 && "Symbol did not point to the beginning of a symbol");
# endif

  return addr;
}

const coff_section *COFFObjectFile::toSec(DataRefImpl Sec) const {
  const coff_section *addr = reinterpret_cast<const coff_section*>(Sec.p);

# ifndef NDEBUG
  // Verify that the section points to a valid entry in the section table.
  if (addr < SectionTable
      || addr >= (SectionTable + Header->NumberOfSections))
    report_fatal_error("Section was outside of section table.");

  uintptr_t offset = uintptr_t(addr) - uintptr_t(SectionTable);
  assert(offset % sizeof(coff_section) == 0 &&
         "Section did not point to the beginning of a section");
# endif

  return addr;
}

error_code COFFObjectFile::getSymbolNext(DataRefImpl Symb,
                                         SymbolRef &Result) const {
  const coff_symbol *symb = toSymb(Symb);
  symb += 1 + symb->NumberOfAuxSymbols;
  Symb.p = reinterpret_cast<uintptr_t>(symb);
  Result = SymbolRef(Symb, this);
  return object_error::success;
}

 error_code COFFObjectFile::getSymbolName(DataRefImpl Symb,
                                          StringRef &Result) const {
  const coff_symbol *symb = toSymb(Symb);
  return getSymbolName(symb, Result);
}

error_code COFFObjectFile::getSymbolFileOffset(DataRefImpl Symb,
                                            uint64_t &Result) const {
  const coff_symbol *symb = toSymb(Symb);
  const coff_section *Section = NULL;
  if (error_code ec = getSection(symb->SectionNumber, Section))
    return ec;
  char Type;
  if (error_code ec = getSymbolNMTypeChar(Symb, Type))
    return ec;
  if (Type == 'U' || Type == 'w')
    Result = UnknownAddressOrSize;
  else if (Section)
    Result = Section->PointerToRawData + symb->Value;
  else
    Result = symb->Value;
  return object_error::success;
}

error_code COFFObjectFile::getSymbolAddress(DataRefImpl Symb,
                                            uint64_t &Result) const {
  const coff_symbol *symb = toSymb(Symb);
  const coff_section *Section = NULL;
  if (error_code ec = getSection(symb->SectionNumber, Section))
    return ec;
  char Type;
  if (error_code ec = getSymbolNMTypeChar(Symb, Type))
    return ec;
  if (Type == 'U' || Type == 'w')
    Result = UnknownAddressOrSize;
  else if (Section)
    Result = Section->VirtualAddress + symb->Value;
  else
    Result = symb->Value;
  return object_error::success;
}

error_code COFFObjectFile::getSymbolType(DataRefImpl Symb,
                                         SymbolRef::Type &Result) const {
  const coff_symbol *symb = toSymb(Symb);
  Result = SymbolRef::ST_Other;
  if (symb->StorageClass == COFF::IMAGE_SYM_CLASS_EXTERNAL &&
      symb->SectionNumber == COFF::IMAGE_SYM_UNDEFINED) {
    Result = SymbolRef::ST_Unknown;
  } else {
    if (symb->getComplexType() == COFF::IMAGE_SYM_DTYPE_FUNCTION) {
      Result = SymbolRef::ST_Function;
    } else {
      char Type;
      if (error_code ec = getSymbolNMTypeChar(Symb, Type))
        return ec;
      if (Type == 'r' || Type == 'R') {
        Result = SymbolRef::ST_Data;
      }
    }
  }
  return object_error::success;
}

error_code COFFObjectFile::getSymbolFlags(DataRefImpl Symb,
                                          uint32_t &Result) const {
  const coff_symbol *symb = toSymb(Symb);
  Result = SymbolRef::SF_None;

  // TODO: Correctly set SF_FormatSpecific, SF_ThreadLocal, SF_Common

  if (symb->StorageClass == COFF::IMAGE_SYM_CLASS_EXTERNAL &&
      symb->SectionNumber == COFF::IMAGE_SYM_UNDEFINED)
    Result |= SymbolRef::SF_Undefined;

  // TODO: This are certainly too restrictive.
  if (symb->StorageClass == COFF::IMAGE_SYM_CLASS_EXTERNAL)
    Result |= SymbolRef::SF_Global;

  if (symb->StorageClass == COFF::IMAGE_SYM_CLASS_WEAK_EXTERNAL)
    Result |= SymbolRef::SF_Weak;

  if (symb->SectionNumber == COFF::IMAGE_SYM_ABSOLUTE)
    Result |= SymbolRef::SF_Absolute;

  return object_error::success;
}

error_code COFFObjectFile::getSymbolSize(DataRefImpl Symb,
                                         uint64_t &Result) const {
  // FIXME: Return the correct size. This requires looking at all the symbols
  //        in the same section as this symbol, and looking for either the next
  //        symbol, or the end of the section.
  const coff_symbol *symb = toSymb(Symb);
  const coff_section *Section = NULL;
  if (error_code ec = getSection(symb->SectionNumber, Section))
    return ec;
  char Type;
  if (error_code ec = getSymbolNMTypeChar(Symb, Type))
    return ec;
  if (Type == 'U' || Type == 'w')
    Result = UnknownAddressOrSize;
  else if (Section)
    Result = Section->SizeOfRawData - symb->Value;
  else
    Result = 0;
  return object_error::success;
}

error_code COFFObjectFile::getSymbolNMTypeChar(DataRefImpl Symb,
                                               char &Result) const {
  const coff_symbol *symb = toSymb(Symb);
  StringRef name;
  if (error_code ec = getSymbolName(Symb, name))
    return ec;
  char ret = StringSwitch<char>(name)
    .StartsWith(".debug", 'N')
    .StartsWith(".sxdata", 'N')
    .Default('?');

  if (ret != '?') {
    Result = ret;
    return object_error::success;
  }

  uint32_t Characteristics = 0;
  if (symb->SectionNumber > 0) {
    const coff_section *Section = NULL;
    if (error_code ec = getSection(symb->SectionNumber, Section))
      return ec;
    Characteristics = Section->Characteristics;
  }

  switch (symb->SectionNumber) {
  case COFF::IMAGE_SYM_UNDEFINED:
    // Check storage classes.
    if (symb->StorageClass == COFF::IMAGE_SYM_CLASS_WEAK_EXTERNAL) {
      Result = 'w';
      return object_error::success; // Don't do ::toupper.
    } else if (symb->Value != 0) // Check for common symbols.
      ret = 'c';
    else
      ret = 'u';
    break;
  case COFF::IMAGE_SYM_ABSOLUTE:
    ret = 'a';
    break;
  case COFF::IMAGE_SYM_DEBUG:
    ret = 'n';
    break;
  default:
    // Check section type.
    if (Characteristics & COFF::IMAGE_SCN_CNT_CODE)
      ret = 't';
    else if (  Characteristics & COFF::IMAGE_SCN_MEM_READ
            && ~Characteristics & COFF::IMAGE_SCN_MEM_WRITE) // Read only.
      ret = 'r';
    else if (Characteristics & COFF::IMAGE_SCN_CNT_INITIALIZED_DATA)
      ret = 'd';
    else if (Characteristics & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA)
      ret = 'b';
    else if (Characteristics & COFF::IMAGE_SCN_LNK_INFO)
      ret = 'i';

    // Check for section symbol.
    else if (  symb->StorageClass == COFF::IMAGE_SYM_CLASS_STATIC
            && symb->Value == 0)
       ret = 's';
  }

  if (symb->StorageClass == COFF::IMAGE_SYM_CLASS_EXTERNAL)
    ret = ::toupper(ret);

  Result = ret;
  return object_error::success;
}

error_code COFFObjectFile::getSymbolSection(DataRefImpl Symb,
                                            section_iterator &Result) const {
  const coff_symbol *symb = toSymb(Symb);
  if (symb->SectionNumber <= COFF::IMAGE_SYM_UNDEFINED)
    Result = end_sections();
  else {
    const coff_section *sec = 0;
    if (error_code ec = getSection(symb->SectionNumber, sec)) return ec;
    DataRefImpl Sec;
    Sec.p = reinterpret_cast<uintptr_t>(sec);
    Result = section_iterator(SectionRef(Sec, this));
  }
  return object_error::success;
}

error_code COFFObjectFile::getSymbolValue(DataRefImpl Symb,
                                          uint64_t &Val) const {
  report_fatal_error("getSymbolValue unimplemented in COFFObjectFile");
}

error_code COFFObjectFile::getSectionNext(DataRefImpl Sec,
                                          SectionRef &Result) const {
  const coff_section *sec = toSec(Sec);
  sec += 1;
  Sec.p = reinterpret_cast<uintptr_t>(sec);
  Result = SectionRef(Sec, this);
  return object_error::success;
}

error_code COFFObjectFile::getSectionName(DataRefImpl Sec,
                                          StringRef &Result) const {
  const coff_section *sec = toSec(Sec);
  return getSectionName(sec, Result);
}

error_code COFFObjectFile::getSectionAddress(DataRefImpl Sec,
                                             uint64_t &Result) const {
  const coff_section *sec = toSec(Sec);
  Result = sec->VirtualAddress;
  return object_error::success;
}

error_code COFFObjectFile::getSectionSize(DataRefImpl Sec,
                                          uint64_t &Result) const {
  const coff_section *sec = toSec(Sec);
  Result = sec->SizeOfRawData;
  return object_error::success;
}

error_code COFFObjectFile::getSectionContents(DataRefImpl Sec,
                                              StringRef &Result) const {
  const coff_section *sec = toSec(Sec);
  ArrayRef<uint8_t> Res;
  error_code EC = getSectionContents(sec, Res);
  Result = StringRef(reinterpret_cast<const char*>(Res.data()), Res.size());
  return EC;
}

error_code COFFObjectFile::getSectionAlignment(DataRefImpl Sec,
                                               uint64_t &Res) const {
  const coff_section *sec = toSec(Sec);
  if (!sec)
    return object_error::parse_failed;
  Res = uint64_t(1) << (((sec->Characteristics & 0x00F00000) >> 20) - 1);
  return object_error::success;
}

error_code COFFObjectFile::isSectionText(DataRefImpl Sec,
                                         bool &Result) const {
  const coff_section *sec = toSec(Sec);
  Result = sec->Characteristics & COFF::IMAGE_SCN_CNT_CODE;
  return object_error::success;
}

error_code COFFObjectFile::isSectionData(DataRefImpl Sec,
                                         bool &Result) const {
  const coff_section *sec = toSec(Sec);
  Result = sec->Characteristics & COFF::IMAGE_SCN_CNT_INITIALIZED_DATA;
  return object_error::success;
}

error_code COFFObjectFile::isSectionBSS(DataRefImpl Sec,
                                        bool &Result) const {
  const coff_section *sec = toSec(Sec);
  Result = sec->Characteristics & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA;
  return object_error::success;
}

error_code COFFObjectFile::isSectionRequiredForExecution(DataRefImpl Sec,
                                                         bool &Result) const {
  // FIXME: Unimplemented
  Result = true;
  return object_error::success;
}

error_code COFFObjectFile::isSectionVirtual(DataRefImpl Sec,
                                           bool &Result) const {
  const coff_section *sec = toSec(Sec);
  Result = sec->Characteristics & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA;
  return object_error::success;
}

error_code COFFObjectFile::isSectionZeroInit(DataRefImpl Sec,
                                             bool &Result) const {
  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

error_code COFFObjectFile::isSectionReadOnlyData(DataRefImpl Sec,
                                                bool &Result) const {
  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

error_code COFFObjectFile::sectionContainsSymbol(DataRefImpl Sec,
                                                 DataRefImpl Symb,
                                                 bool &Result) const {
  const coff_section *sec = toSec(Sec);
  const coff_symbol *symb = toSymb(Symb);
  const coff_section *symb_sec = 0;
  if (error_code ec = getSection(symb->SectionNumber, symb_sec)) return ec;
  if (symb_sec == sec)
    Result = true;
  else
    Result = false;
  return object_error::success;
}

relocation_iterator COFFObjectFile::getSectionRelBegin(DataRefImpl Sec) const {
  const coff_section *sec = toSec(Sec);
  DataRefImpl ret;
  if (sec->NumberOfRelocations == 0)
    ret.p = 0;
  else
    ret.p = reinterpret_cast<uintptr_t>(base() + sec->PointerToRelocations);

  return relocation_iterator(RelocationRef(ret, this));
}

relocation_iterator COFFObjectFile::getSectionRelEnd(DataRefImpl Sec) const {
  const coff_section *sec = toSec(Sec);
  DataRefImpl ret;
  if (sec->NumberOfRelocations == 0)
    ret.p = 0;
  else
    ret.p = reinterpret_cast<uintptr_t>(
              reinterpret_cast<const coff_relocation*>(
                base() + sec->PointerToRelocations)
              + sec->NumberOfRelocations);

  return relocation_iterator(RelocationRef(ret, this));
}

COFFObjectFile::COFFObjectFile(MemoryBuffer *Object, error_code &ec)
  : ObjectFile(Binary::ID_COFF, Object, ec)
  , Header(0)
  , SectionTable(0)
  , SymbolTable(0)
  , StringTable(0)
  , StringTableSize(0) {
  // Check that we at least have enough room for a header.
  if (!checkSize(Data, ec, sizeof(coff_file_header))) return;

  // The actual starting location of the COFF header in the file. This can be
  // non-zero in PE/COFF files.
  uint64_t HeaderStart = 0;

  // Check if this is a PE/COFF file.
  if (base()[0] == 0x4d && base()[1] == 0x5a) {
    // PE/COFF, seek through MS-DOS compatibility stub and 4-byte
    // PE signature to find 'normal' COFF header.
    if (!checkSize(Data, ec, 0x3c + 8)) return;
    HeaderStart = *reinterpret_cast<const ulittle16_t *>(base() + 0x3c);
    // Check the PE header. ("PE\0\0")
    if (std::memcmp(base() + HeaderStart, "PE\0\0", 4) != 0) {
      ec = object_error::parse_failed;
      return;
    }
    HeaderStart += 4; // Skip the PE Header.
  }

  Header = reinterpret_cast<const coff_file_header *>(base() + HeaderStart);
  if (!checkAddr(Data, ec, uintptr_t(Header), sizeof(coff_file_header)))
    return;

  SectionTable =
    reinterpret_cast<const coff_section *>( base()
                                          + HeaderStart
                                          + sizeof(coff_file_header)
                                          + Header->SizeOfOptionalHeader);
  if (!checkAddr(Data, ec, uintptr_t(SectionTable),
                 Header->NumberOfSections * sizeof(coff_section)))
    return;

  if (Header->PointerToSymbolTable != 0) {
    SymbolTable =
      reinterpret_cast<const coff_symbol *>(base()
                                            + Header->PointerToSymbolTable);
    if (!checkAddr(Data, ec, uintptr_t(SymbolTable),
                   Header->NumberOfSymbols * sizeof(coff_symbol)))
      return;

    // Find string table.
    StringTable = reinterpret_cast<const char *>(base())
                  + Header->PointerToSymbolTable
                  + Header->NumberOfSymbols * sizeof(coff_symbol);
    if (!checkAddr(Data, ec, uintptr_t(StringTable), sizeof(ulittle32_t)))
      return;

    StringTableSize = *reinterpret_cast<const ulittle32_t *>(StringTable);
    if (!checkAddr(Data, ec, uintptr_t(StringTable), StringTableSize))
      return;
    // Check that the string table is null terminated if has any in it.
    if (StringTableSize < 4
        || (StringTableSize > 4 && StringTable[StringTableSize - 1] != 0)) {
      ec = object_error::parse_failed;
      return;
    }
  }

  ec = object_error::success;
}

symbol_iterator COFFObjectFile::begin_symbols() const {
  DataRefImpl ret;
  ret.p = reinterpret_cast<intptr_t>(SymbolTable);
  return symbol_iterator(SymbolRef(ret, this));
}

symbol_iterator COFFObjectFile::end_symbols() const {
  // The symbol table ends where the string table begins.
  DataRefImpl ret;
  ret.p = reinterpret_cast<intptr_t>(StringTable);
  return symbol_iterator(SymbolRef(ret, this));
}

symbol_iterator COFFObjectFile::begin_dynamic_symbols() const {
  // TODO: implement
  report_fatal_error("Dynamic symbols unimplemented in COFFObjectFile");
}

symbol_iterator COFFObjectFile::end_dynamic_symbols() const {
  // TODO: implement
  report_fatal_error("Dynamic symbols unimplemented in COFFObjectFile");
}

library_iterator COFFObjectFile::begin_libraries_needed() const {
  // TODO: implement
  report_fatal_error("Libraries needed unimplemented in COFFObjectFile");
}

library_iterator COFFObjectFile::end_libraries_needed() const {
  // TODO: implement
  report_fatal_error("Libraries needed unimplemented in COFFObjectFile");
}

StringRef COFFObjectFile::getLoadName() const {
  // COFF does not have this field.
  return "";
}


section_iterator COFFObjectFile::begin_sections() const {
  DataRefImpl ret;
  ret.p = reinterpret_cast<intptr_t>(SectionTable);
  return section_iterator(SectionRef(ret, this));
}

section_iterator COFFObjectFile::end_sections() const {
  DataRefImpl ret;
  ret.p = reinterpret_cast<intptr_t>(SectionTable + Header->NumberOfSections);
  return section_iterator(SectionRef(ret, this));
}

uint8_t COFFObjectFile::getBytesInAddress() const {
  return getArch() == Triple::x86_64 ? 8 : 4;
}

StringRef COFFObjectFile::getFileFormatName() const {
  switch(Header->Machine) {
  case COFF::IMAGE_FILE_MACHINE_I386:
    return "COFF-i386";
  case COFF::IMAGE_FILE_MACHINE_AMD64:
    return "COFF-x86-64";
  default:
    return "COFF-<unknown arch>";
  }
}

unsigned COFFObjectFile::getArch() const {
  switch(Header->Machine) {
  case COFF::IMAGE_FILE_MACHINE_I386:
    return Triple::x86;
  case COFF::IMAGE_FILE_MACHINE_AMD64:
    return Triple::x86_64;
  default:
    return Triple::UnknownArch;
  }
}

error_code COFFObjectFile::getHeader(const coff_file_header *&Res) const {
  Res = Header;
  return object_error::success;
}

error_code COFFObjectFile::getSection(int32_t index,
                                      const coff_section *&Result) const {
  // Check for special index values.
  if (index == COFF::IMAGE_SYM_UNDEFINED ||
      index == COFF::IMAGE_SYM_ABSOLUTE ||
      index == COFF::IMAGE_SYM_DEBUG)
    Result = NULL;
  else if (index > 0 && index <= Header->NumberOfSections)
    // We already verified the section table data, so no need to check again.
    Result = SectionTable + (index - 1);
  else
    return object_error::parse_failed;
  return object_error::success;
}

error_code COFFObjectFile::getString(uint32_t offset,
                                     StringRef &Result) const {
  if (StringTableSize <= 4)
    // Tried to get a string from an empty string table.
    return object_error::parse_failed;
  if (offset >= StringTableSize)
    return object_error::unexpected_eof;
  Result = StringRef(StringTable + offset);
  return object_error::success;
}

error_code COFFObjectFile::getSymbol(uint32_t index,
                                     const coff_symbol *&Result) const {
  if (index < Header->NumberOfSymbols)
    Result = SymbolTable + index;
  else
    return object_error::parse_failed;
  return object_error::success;
}

error_code COFFObjectFile::getSymbolName(const coff_symbol *symbol,
                                         StringRef &Res) const {
  // Check for string table entry. First 4 bytes are 0.
  if (symbol->Name.Offset.Zeroes == 0) {
    uint32_t Offset = symbol->Name.Offset.Offset;
    if (error_code ec = getString(Offset, Res))
      return ec;
    return object_error::success;
  }

  if (symbol->Name.ShortName[7] == 0)
    // Null terminated, let ::strlen figure out the length.
    Res = StringRef(symbol->Name.ShortName);
  else
    // Not null terminated, use all 8 bytes.
    Res = StringRef(symbol->Name.ShortName, 8);
  return object_error::success;
}

ArrayRef<uint8_t> COFFObjectFile::getSymbolAuxData(
                                  const coff_symbol *symbol) const {
  const uint8_t *aux = NULL;
  
  if ( symbol->NumberOfAuxSymbols > 0 ) {
  // AUX data comes immediately after the symbol in COFF
    aux = reinterpret_cast<const uint8_t *>(symbol + 1);
# ifndef NDEBUG
    // Verify that the aux symbol points to a valid entry in the symbol table.
    uintptr_t offset = uintptr_t(aux) - uintptr_t(base());
    if (offset < Header->PointerToSymbolTable
        || offset >= Header->PointerToSymbolTable
           + (Header->NumberOfSymbols * sizeof(coff_symbol)))
      report_fatal_error("Aux Symbol data was outside of symbol table.");

    assert((offset - Header->PointerToSymbolTable) % sizeof(coff_symbol)
         == 0 && "Aux Symbol data did not point to the beginning of a symbol");
# endif
  }
  return ArrayRef<uint8_t>(aux, symbol->NumberOfAuxSymbols * sizeof(coff_symbol));
}

error_code COFFObjectFile::getSectionName(const coff_section *Sec,
                                          StringRef &Res) const {
  StringRef Name;
  if (Sec->Name[7] == 0)
    // Null terminated, let ::strlen figure out the length.
    Name = Sec->Name;
  else
    // Not null terminated, use all 8 bytes.
    Name = StringRef(Sec->Name, 8);

  // Check for string table entry. First byte is '/'.
  if (Name[0] == '/') {
    uint32_t Offset;
    if (Name.substr(1).getAsInteger(10, Offset))
      return object_error::parse_failed;
    if (error_code ec = getString(Offset, Name))
      return ec;
  }

  Res = Name;
  return object_error::success;
}

error_code COFFObjectFile::getSectionContents(const coff_section *Sec,
                                              ArrayRef<uint8_t> &Res) const {
  // The only thing that we need to verify is that the contents is contained
  // within the file bounds. We don't need to make sure it doesn't cover other
  // data, as there's nothing that says that is not allowed.
  uintptr_t ConStart = uintptr_t(base()) + Sec->PointerToRawData;
  uintptr_t ConEnd = ConStart + Sec->SizeOfRawData;
  if (ConEnd > uintptr_t(Data->getBufferEnd()))
    return object_error::parse_failed;
  Res = ArrayRef<uint8_t>(reinterpret_cast<const unsigned char*>(ConStart),
                          Sec->SizeOfRawData);
  return object_error::success;
}

const coff_relocation *COFFObjectFile::toRel(DataRefImpl Rel) const {
  return reinterpret_cast<const coff_relocation*>(Rel.p);
}
error_code COFFObjectFile::getRelocationNext(DataRefImpl Rel,
                                             RelocationRef &Res) const {
  Rel.p = reinterpret_cast<uintptr_t>(
            reinterpret_cast<const coff_relocation*>(Rel.p) + 1);
  Res = RelocationRef(Rel, this);
  return object_error::success;
}
error_code COFFObjectFile::getRelocationAddress(DataRefImpl Rel,
                                                uint64_t &Res) const {
  Res = toRel(Rel)->VirtualAddress;
  return object_error::success;
}
error_code COFFObjectFile::getRelocationOffset(DataRefImpl Rel,
                                               uint64_t &Res) const {
  Res = toRel(Rel)->VirtualAddress;
  return object_error::success;
}
error_code COFFObjectFile::getRelocationSymbol(DataRefImpl Rel,
                                               SymbolRef &Res) const {
  const coff_relocation* R = toRel(Rel);
  DataRefImpl Symb;
  Symb.p = reinterpret_cast<uintptr_t>(SymbolTable + R->SymbolTableIndex);
  Res = SymbolRef(Symb, this);
  return object_error::success;
}
error_code COFFObjectFile::getRelocationType(DataRefImpl Rel,
                                             uint64_t &Res) const {
  const coff_relocation* R = toRel(Rel);
  Res = R->Type;
  return object_error::success;
}

const coff_section *COFFObjectFile::getCOFFSection(section_iterator &It) const {
  return toSec(It->getRawDataRefImpl());
}

const coff_symbol *COFFObjectFile::getCOFFSymbol(symbol_iterator &It) const {
  return toSymb(It->getRawDataRefImpl());
}

const coff_relocation *COFFObjectFile::getCOFFRelocation(
                                             relocation_iterator &It) const {
  return toRel(It->getRawDataRefImpl());
}


#define LLVM_COFF_SWITCH_RELOC_TYPE_NAME(enum) \
  case COFF::enum: res = #enum; break;

error_code COFFObjectFile::getRelocationTypeName(DataRefImpl Rel,
                                          SmallVectorImpl<char> &Result) const {
  const coff_relocation *reloc = toRel(Rel);
  StringRef res;
  switch (Header->Machine) {
  case COFF::IMAGE_FILE_MACHINE_AMD64:
    switch (reloc->Type) {
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_ABSOLUTE);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_ADDR64);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_ADDR32);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_ADDR32NB);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_REL32);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_REL32_1);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_REL32_2);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_REL32_3);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_REL32_4);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_REL32_5);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_SECTION);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_SECREL);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_SECREL7);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_TOKEN);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_SREL32);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_PAIR);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_SSPAN32);
    default:
      res = "Unknown";
    }
    break;
  case COFF::IMAGE_FILE_MACHINE_I386:
    switch (reloc->Type) {
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_I386_ABSOLUTE);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_I386_DIR16);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_I386_REL16);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_I386_DIR32);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_I386_DIR32NB);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_I386_SEG12);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_I386_SECTION);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_I386_SECREL);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_I386_TOKEN);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_I386_SECREL7);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_I386_REL32);
    default:
      res = "Unknown";
    }
    break;
  default:
    res = "Unknown";
  }
  Result.append(res.begin(), res.end());
  return object_error::success;
}

#undef LLVM_COFF_SWITCH_RELOC_TYPE_NAME

error_code COFFObjectFile::getRelocationAdditionalInfo(DataRefImpl Rel,
                                                       int64_t &Res) const {
  Res = 0;
  return object_error::success;
}
error_code COFFObjectFile::getRelocationValueString(DataRefImpl Rel,
                                          SmallVectorImpl<char> &Result) const {
  const coff_relocation *reloc = toRel(Rel);
  const coff_symbol *symb = 0;
  if (error_code ec = getSymbol(reloc->SymbolTableIndex, symb)) return ec;
  DataRefImpl sym;
  sym.p = reinterpret_cast<uintptr_t>(symb);
  StringRef symname;
  if (error_code ec = getSymbolName(sym, symname)) return ec;
  Result.append(symname.begin(), symname.end());
  return object_error::success;
}

error_code COFFObjectFile::getLibraryNext(DataRefImpl LibData,
                                          LibraryRef &Result) const {
  report_fatal_error("getLibraryNext not implemented in COFFObjectFile");
}

error_code COFFObjectFile::getLibraryPath(DataRefImpl LibData,
                                          StringRef &Result) const {
  report_fatal_error("getLibraryPath not implemented in COFFObjectFile");
}

namespace llvm {

  ObjectFile *ObjectFile::createCOFFObjectFile(MemoryBuffer *Object) {
    error_code ec;
    return new COFFObjectFile(Object, ec);
  }

} // end namespace llvm
