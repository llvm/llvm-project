//===- COFF.h - COFF object file implementation -----------------*- C++ -*-===//
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

#ifndef LLVM_OBJECT_COFF_H
#define LLVM_OBJECT_COFF_H

#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Endian.h"

namespace llvm {
  template <typename T>
  class ArrayRef;

namespace object {

struct coff_file_header {
  support::ulittle16_t Machine;
  support::ulittle16_t NumberOfSections;
  support::ulittle32_t TimeDateStamp;
  support::ulittle32_t PointerToSymbolTable;
  support::ulittle32_t NumberOfSymbols;
  support::ulittle16_t SizeOfOptionalHeader;
  support::ulittle16_t Characteristics;
};

struct coff_symbol {
  struct StringTableOffset {
    support::ulittle32_t Zeroes;
    support::ulittle32_t Offset;
  };

  union {
    char ShortName[8];
    StringTableOffset Offset;
  } Name;

  support::ulittle32_t Value;
  support::little16_t SectionNumber;

  support::ulittle16_t Type;

  support::ulittle8_t  StorageClass;
  support::ulittle8_t  NumberOfAuxSymbols;

  uint8_t getBaseType() const {
    return Type & 0x0F;
  }

  uint8_t getComplexType() const {
    return (Type & 0xF0) >> 4;
  }
};

struct coff_section {
  char Name[8];
  support::ulittle32_t VirtualSize;
  support::ulittle32_t VirtualAddress;
  support::ulittle32_t SizeOfRawData;
  support::ulittle32_t PointerToRawData;
  support::ulittle32_t PointerToRelocations;
  support::ulittle32_t PointerToLinenumbers;
  support::ulittle16_t NumberOfRelocations;
  support::ulittle16_t NumberOfLinenumbers;
  support::ulittle32_t Characteristics;
};

struct coff_relocation {
  support::ulittle32_t VirtualAddress;
  support::ulittle32_t SymbolTableIndex;
  support::ulittle16_t Type;
};

struct coff_aux_section_definition {
  support::ulittle32_t Length;
  support::ulittle16_t NumberOfRelocations;
  support::ulittle16_t NumberOfLinenumbers;
  support::ulittle32_t CheckSum;
  support::ulittle16_t Number;
  support::ulittle8_t Selection;
  char Unused[3];
};

class COFFObjectFile : public ObjectFile {
private:
  const coff_file_header *Header;
  const coff_section     *SectionTable;
  const coff_symbol      *SymbolTable;
  const char             *StringTable;
        uint32_t          StringTableSize;

        error_code        getString(uint32_t offset, StringRef &Res) const;

  const coff_symbol      *toSymb(DataRefImpl Symb) const;
  const coff_section     *toSec(DataRefImpl Sec) const;
  const coff_relocation  *toRel(DataRefImpl Rel) const;

protected:
  virtual error_code getSymbolNext(DataRefImpl Symb, SymbolRef &Res) const;
  virtual error_code getSymbolName(DataRefImpl Symb, StringRef &Res) const;
  virtual error_code getSymbolFileOffset(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code getSymbolAddress(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code getSymbolSize(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code getSymbolNMTypeChar(DataRefImpl Symb, char &Res) const;
  virtual error_code getSymbolFlags(DataRefImpl Symb, uint32_t &Res) const;
  virtual error_code getSymbolType(DataRefImpl Symb, SymbolRef::Type &Res) const;
  virtual error_code getSymbolSection(DataRefImpl Symb,
                                      section_iterator &Res) const;
  virtual error_code getSymbolValue(DataRefImpl Symb, uint64_t &Val) const;

  virtual error_code getSectionNext(DataRefImpl Sec, SectionRef &Res) const;
  virtual error_code getSectionName(DataRefImpl Sec, StringRef &Res) const;
  virtual error_code getSectionAddress(DataRefImpl Sec, uint64_t &Res) const;
  virtual error_code getSectionSize(DataRefImpl Sec, uint64_t &Res) const;
  virtual error_code getSectionContents(DataRefImpl Sec, StringRef &Res) const;
  virtual error_code getSectionAlignment(DataRefImpl Sec, uint64_t &Res) const;
  virtual error_code isSectionText(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionData(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionBSS(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionVirtual(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionZeroInit(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionReadOnlyData(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionRequiredForExecution(DataRefImpl Sec,
                                                   bool &Res) const;
  virtual error_code sectionContainsSymbol(DataRefImpl Sec, DataRefImpl Symb,
                                           bool &Result) const;
  virtual relocation_iterator getSectionRelBegin(DataRefImpl Sec) const;
  virtual relocation_iterator getSectionRelEnd(DataRefImpl Sec) const;

  virtual error_code getRelocationNext(DataRefImpl Rel,
                                       RelocationRef &Res) const;
  virtual error_code getRelocationAddress(DataRefImpl Rel,
                                          uint64_t &Res) const;
  virtual error_code getRelocationOffset(DataRefImpl Rel,
                                         uint64_t &Res) const;
  virtual error_code getRelocationSymbol(DataRefImpl Rel,
                                         SymbolRef &Res) const;
  virtual error_code getRelocationType(DataRefImpl Rel,
                                       uint64_t &Res) const;
  virtual error_code getRelocationTypeName(DataRefImpl Rel,
                                           SmallVectorImpl<char> &Result) const;
  virtual error_code getRelocationAdditionalInfo(DataRefImpl Rel,
                                                 int64_t &Res) const;
  virtual error_code getRelocationValueString(DataRefImpl Rel,
                                           SmallVectorImpl<char> &Result) const;

  virtual error_code getLibraryNext(DataRefImpl LibData,
                                    LibraryRef &Result) const;
  virtual error_code getLibraryPath(DataRefImpl LibData,
                                    StringRef &Result) const;

public:
  COFFObjectFile(MemoryBuffer *Object, error_code &ec);
  virtual symbol_iterator begin_symbols() const;
  virtual symbol_iterator end_symbols() const;
  virtual symbol_iterator begin_dynamic_symbols() const;
  virtual symbol_iterator end_dynamic_symbols() const;
  virtual library_iterator begin_libraries_needed() const;
  virtual library_iterator end_libraries_needed() const;
  virtual section_iterator begin_sections() const;
  virtual section_iterator end_sections() const;

  const coff_section *getCOFFSection(section_iterator &It) const;
  const coff_symbol *getCOFFSymbol(symbol_iterator &It) const;
  const coff_relocation *getCOFFRelocation(relocation_iterator &It) const;
  
  virtual uint8_t getBytesInAddress() const;
  virtual StringRef getFileFormatName() const;
  virtual unsigned getArch() const;
  virtual StringRef getLoadName() const;

  error_code getHeader(const coff_file_header *&Res) const;
  error_code getSection(int32_t index, const coff_section *&Res) const;
  error_code getSymbol(uint32_t index, const coff_symbol *&Res) const;
  template <typename T>
  error_code getAuxSymbol(uint32_t index, const T *&Res) const {
    const coff_symbol *s;
    error_code ec = getSymbol(index, s);
    Res = reinterpret_cast<const T*>(s);
    return ec;
  }
  error_code getSymbolName(const coff_symbol *symbol, StringRef &Res) const;
  ArrayRef<uint8_t> getSymbolAuxData(const coff_symbol *symbol) const;

  error_code getSectionName(const coff_section *Sec, StringRef &Res) const;
  error_code getSectionContents(const coff_section *Sec,
                                ArrayRef<uint8_t> &Res) const;

  static inline bool classof(const Binary *v) {
    return v->isCOFF();
  }
};

}
}

#endif
