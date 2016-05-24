//===-- SymbolVisitorDelegate.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_SYMBOLVISITORDELEGATE_H
#define LLVM_DEBUGINFO_CODEVIEW_SYMBOLVISITORDELEGATE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <stdint.h>

namespace llvm {

namespace codeview {

class SymbolVisitorDelegate {
public:
  virtual ~SymbolVisitorDelegate() {}

  virtual uint32_t getRecordOffset(ArrayRef<uint8_t> Record) = 0;
  virtual StringRef getFileNameForFileOffset(uint32_t FileOffset) = 0;
  virtual StringRef getStringTable() = 0;
};
} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_SYMBOLVISITORDELEGATE_H
