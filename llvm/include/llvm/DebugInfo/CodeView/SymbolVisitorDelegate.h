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

#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace llvm {

namespace msf {
class StreamReader;
} // end namespace msf

namespace codeview {

class SymbolVisitorDelegate {
public:
  virtual ~SymbolVisitorDelegate() = default;

  virtual uint32_t getRecordOffset(msf::StreamReader Reader) = 0;
  virtual StringRef getFileNameForFileOffset(uint32_t FileOffset) = 0;
  virtual StringRef getStringTable() = 0;
};

} // end namespace codeview

} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_SYMBOLVISITORDELEGATE_H
