//===- COFFImportFile.h - COFF short import file implementation -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// COFF short import file is a special kind of file which contains
// only symbol names for DLL-exported symbols. This class implements
// SymbolicFile interface for the file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_COFF_IMPORT_FILE_H
#define LLVM_OBJECT_COFF_IMPORT_FILE_H

#include "llvm/Object/COFF.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace object {

class COFFImportFile : public SymbolicFile {
public:
  COFFImportFile(MemoryBufferRef Source)
      : SymbolicFile(ID_COFFImportFile, Source) {}

  static inline bool classof(Binary const *V) { return V->isCOFFImportFile(); }

  void moveSymbolNext(DataRefImpl &Symb) const override { ++Symb.p; }

  std::error_code printSymbolName(raw_ostream &OS,
                                  DataRefImpl Symb) const override {
    if (Symb.p == 0)
      OS << "__imp_";
    OS << StringRef(Data.getBufferStart() + sizeof(coff_import_header));
    return std::error_code();
  }

  uint32_t getSymbolFlags(DataRefImpl Symb) const override {
    return SymbolRef::SF_Global;
  }

  basic_symbol_iterator symbol_begin() const override {
    return BasicSymbolRef(DataRefImpl(), this);
  }

  basic_symbol_iterator symbol_end() const override {
    DataRefImpl Symb;
    Symb.p = isCode() ? 2 : 1;
    return BasicSymbolRef(Symb, this);
  }

  const coff_import_header *getCOFFImportHeader() const {
    return reinterpret_cast<const object::coff_import_header *>(
        Data.getBufferStart());
  }

private:
  bool isCode() const {
    return getCOFFImportHeader()->getType() == COFF::IMPORT_CODE;
  }
};

} // namespace object
} // namespace llvm

#endif
