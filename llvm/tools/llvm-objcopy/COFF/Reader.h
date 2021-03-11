//===- Reader.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_OBJCOPY_COFF_READER_H
#define LLVM_TOOLS_OBJCOPY_COFF_READER_H

#include "llvm/BinaryFormat/COFF.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace objcopy {
namespace coff {

struct Object;

using object::COFFObjectFile;

class COFFReader {
  const COFFObjectFile &COFFObj;

  Error readExecutableHeaders(Object &Obj) const;
  Error readSections(Object &Obj) const;
  Error readSymbols(Object &Obj, bool IsBigObj) const;
  Error setSymbolTargets(Object &Obj) const;

public:
  explicit COFFReader(const COFFObjectFile &O) : COFFObj(O) {}
  Expected<std::unique_ptr<Object>> create() const;
};

} // end namespace coff
} // end namespace objcopy
} // end namespace llvm

#endif // LLVM_TOOLS_OBJCOPY_COFF_READER_H
