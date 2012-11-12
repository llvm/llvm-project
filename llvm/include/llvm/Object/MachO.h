//===- MachO.h - MachO object file implementation ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MachOObjectFile class, which binds the MachOObject
// class to the generic ObjectFile wrapper.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_MACHO_H
#define LLVM_OBJECT_MACHO_H

#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/MachOObject.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
namespace object {

typedef MachOObject::LoadCommandInfo LoadCommandInfo;

class MachOObjectFile : public ObjectFile {
public:
  MachOObjectFile(MemoryBuffer *Object, MachOObject *MOO, error_code &ec);

  virtual symbol_iterator begin_symbols() const;
  virtual symbol_iterator end_symbols() const;
  virtual symbol_iterator begin_dynamic_symbols() const;
  virtual symbol_iterator end_dynamic_symbols() const;
  virtual library_iterator begin_libraries_needed() const;
  virtual library_iterator end_libraries_needed() const;
  virtual section_iterator begin_sections() const;
  virtual section_iterator end_sections() const;

  virtual uint8_t getBytesInAddress() const;
  virtual StringRef getFileFormatName() const;
  virtual unsigned getArch() const;
  virtual StringRef getLoadName() const;

  MachOObject *getObject() { return MachOObj; }

  static inline bool classof(const Binary *v) {
    return v->isMachO();
  }

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
  virtual error_code isSectionRequiredForExecution(DataRefImpl Sec,
                                                   bool &Res) const;
  virtual error_code isSectionVirtual(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionZeroInit(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionReadOnlyData(DataRefImpl Sec, bool &Res) const;
  virtual error_code sectionContainsSymbol(DataRefImpl DRI, DataRefImpl S,
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
  virtual error_code getRelocationHidden(DataRefImpl Rel, bool &Result) const;

  virtual error_code getLibraryNext(DataRefImpl LibData, LibraryRef &Res) const;
  virtual error_code getLibraryPath(DataRefImpl LibData, StringRef &Res) const;

private:
  MachOObject *MachOObj;
  mutable uint32_t RegisteredStringTable;
  typedef SmallVector<DataRefImpl, 1> SectionList;
  SectionList Sections;


  void moveToNextSection(DataRefImpl &DRI) const;
  void getSymbolTableEntry(DataRefImpl DRI,
                           InMemoryStruct<macho::SymbolTableEntry> &Res) const;
  void getSymbol64TableEntry(DataRefImpl DRI,
                          InMemoryStruct<macho::Symbol64TableEntry> &Res) const;
  void moveToNextSymbol(DataRefImpl &DRI) const;
  void getSection(DataRefImpl DRI, InMemoryStruct<macho::Section> &Res) const;
  void getSection64(DataRefImpl DRI,
                    InMemoryStruct<macho::Section64> &Res) const;
  void getRelocation(DataRefImpl Rel,
                     InMemoryStruct<macho::RelocationEntry> &Res) const;
  std::size_t getSectionIndex(DataRefImpl Sec) const;

  void printRelocationTargetName(InMemoryStruct<macho::RelocationEntry>& RE,
                                 raw_string_ostream &fmt) const;
};

}
}

#endif

