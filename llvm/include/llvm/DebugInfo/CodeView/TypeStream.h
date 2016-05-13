//===- TypeStream.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_TYPESTREAM_H
#define LLVM_DEBUGINFO_CODEVIEW_TYPESTREAM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/RecordIterator.h"
#include "llvm/Support/Endian.h"
#include <cstdint>
#include <system_error>

namespace llvm {

class APSInt;

namespace codeview {

typedef RecordIterator<TypeLeafKind> TypeIterator;

inline iterator_range<TypeIterator> makeTypeRange(ArrayRef<uint8_t> Data, bool *HadError) {
  return make_range(TypeIterator(Data, HadError), TypeIterator());
}

} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_TYPESTREAM_H
