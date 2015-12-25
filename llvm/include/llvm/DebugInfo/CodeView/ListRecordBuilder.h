//===- ListRecordBuilder.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_LISTRECORDBUILDER_H
#define LLVM_DEBUGINFO_CODEVIEW_LISTRECORDBUILDER_H

#include "llvm/DebugInfo/CodeView/TypeRecordBuilder.h"

namespace llvm {
namespace codeview {

class ListRecordBuilder {
private:
  ListRecordBuilder(const ListRecordBuilder &) = delete;
  ListRecordBuilder &operator=(const ListRecordBuilder &) = delete;

protected:
  const int MethodKindShift = 2;

  explicit ListRecordBuilder(TypeRecordKind Kind);

public:
  llvm::StringRef str() { return Builder.str(); }

protected:
  void finishSubRecord();

  TypeRecordBuilder &getBuilder() { return Builder; }

private:
  TypeRecordBuilder Builder;
  SmallVector<size_t, 4> ContinuationOffsets;
};
}
}

#endif
