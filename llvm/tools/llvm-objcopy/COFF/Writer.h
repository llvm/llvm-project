//===- Writer.h -------------------------------------------------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_OBJCOPY_COFF_WRITER_H
#define LLVM_TOOLS_OBJCOPY_COFF_WRITER_H

#include "Buffer.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Support/Error.h"
#include <cstddef>
#include <utility>

namespace llvm {
namespace objcopy {
namespace coff {

struct Object;

class Writer {
protected:
  Object &Obj;
  Buffer &Buf;

public:
  virtual ~Writer();
  virtual Error write() = 0;

  Writer(Object &O, Buffer &B) : Obj(O), Buf(B) {}
};

class COFFWriter : public Writer {
  size_t FileSize;
  size_t FileAlignment;
  size_t SizeOfInitializedData;
  StringTableBuilder StrTabBuilder;

  void layoutSections();
  size_t finalizeStringTable();
  template <class SymbolTy> std::pair<size_t, size_t> finalizeSymbolTable();

  void finalize(bool IsBigObj);

  void writeHeaders(bool IsBigObj);
  void writeSections();
  template <class SymbolTy> void writeSymbolStringTables();

  Error write(bool IsBigObj);

  Error patchDebugDirectory();

public:
  virtual ~COFFWriter() {}
  Error write() override;

  COFFWriter(Object &Obj, Buffer &Buf)
      : Writer(Obj, Buf), StrTabBuilder(StringTableBuilder::WinCOFF) {}
};

} // end namespace coff
} // end namespace objcopy
} // end namespace llvm

#endif // LLVM_TOOLS_OBJCOPY_COFF_WRITER_H
