//===-- SymbolDumpDelegate.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_SYMBOLDUMPDELEGATE_H
#define LLVM_DEBUGINFO_CODEVIEW_SYMBOLDUMPDELEGATE_H

#include "SymbolVisitorDelegate.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <stdint.h>

namespace llvm {

namespace codeview {

class SymbolDumpDelegate : public SymbolVisitorDelegate {
public:
  virtual ~SymbolDumpDelegate() {}

  virtual void printRelocatedField(StringRef Label, uint32_t RelocOffset,
                                   uint32_t Offset,
                                   StringRef *RelocSym = nullptr) = 0;
  virtual void printBinaryBlockWithRelocs(StringRef Label,
                                          ArrayRef<uint8_t> Block) = 0;
};
} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_SYMBOLDUMPDELEGATE_H
