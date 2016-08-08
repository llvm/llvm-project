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
class TypeTableBuilder;

class ListRecordBuilder {
private:
  ListRecordBuilder(const ListRecordBuilder &) = delete;
  ListRecordBuilder &operator=(const ListRecordBuilder &) = delete;

protected:
  const int MethodKindShift = 2;

  explicit ListRecordBuilder(TypeRecordKind Kind);

public:
  llvm::StringRef str() { return Builder.str(); }

  void reset() {
    Builder.reset(Kind);
    ContinuationOffsets.clear();
    SubrecordStart = 0;
  }

  void writeMemberType(const ListContinuationRecord &R);

  /// Writes this list record as a possible sequence of records.
  TypeIndex writeListRecord(TypeTableBuilder &Table);

protected:
  void finishSubRecord();

  TypeRecordBuilder &getBuilder() { return Builder; }

private:
  size_t getLastContinuationStart() const {
    return ContinuationOffsets.empty() ? 0 : ContinuationOffsets.back();
  }
  size_t getLastContinuationEnd() const { return Builder.size(); }
  size_t getLastContinuationSize() const {
    return getLastContinuationEnd() - getLastContinuationStart();
  }

  TypeRecordKind Kind;
  TypeRecordBuilder Builder;
  SmallVector<size_t, 4> ContinuationOffsets;
  size_t SubrecordStart = 0;
};
}
}

#endif
