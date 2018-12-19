//===- Reader.h -------------------------------------------------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_OBJCOPY_COFF_READER_H
#define LLVM_TOOLS_OBJCOPY_COFF_READER_H

#include "Buffer.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/Object/COFF.h"

namespace llvm {
namespace objcopy {
namespace coff {

struct Object;

using object::COFFObjectFile;

class Reader {
public:
  virtual ~Reader();
  virtual std::unique_ptr<Object> create() const = 0;
};

class COFFReader : public Reader {
  const COFFObjectFile &COFFObj;

  void readExecutableHeaders(Object &Obj) const;
  void readSections(Object &Obj) const;
  void readSymbols(Object &Obj, bool IsBigObj) const;

public:
  explicit COFFReader(const COFFObjectFile &O) : COFFObj(O) {}
  std::unique_ptr<Object> create() const override;
};

} // end namespace coff
} // end namespace objcopy
} // end namespace llvm

#endif // LLVM_TOOLS_OBJCOPY_COFF_READER_H
