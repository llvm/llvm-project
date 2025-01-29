//===- ContinuationRecordBuilder.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_CONTINUATIONRECORDBUILDER_H
#define LLVM_DEBUGINFO_CODEVIEW_CONTINUATIONRECORDBUILDER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/DebugInfo/CodeView/TypeRecordMapping.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include <cstdint>
#include <vector>

namespace llvm {
namespace codeview {
class TypeIndex;
enum class ContinuationRecordKind { FieldList, MethodOverloadList };

class ContinuationRecordBuilder {
  SmallVector<uint32_t, 4> SegmentOffsets;
  std::optional<ContinuationRecordKind> Kind;
  AppendingBinaryByteStream Buffer;
  BinaryStreamWriter SegmentWriter;
  TypeRecordMapping Mapping;
  ArrayRef<uint8_t> InjectedSegmentBytes;

  uint32_t getCurrentSegmentLength() const;

  void insertSegmentEnd(uint32_t Offset);
  CVType createSegmentRecord(uint32_t OffBegin, uint32_t OffEnd,
                             std::optional<TypeIndex> RefersTo);

public:
  ContinuationRecordBuilder();
  ~ContinuationRecordBuilder();

  void begin(ContinuationRecordKind RecordKind);

  // This template is explicitly instantiated in the implementation file for all
  // supported types.  The method itself is ugly, so inlining it into the header
  // file clutters an otherwise straightforward interface.
  template <typename RecordType> void writeMemberType(RecordType &Record);

  std::vector<CVType> end(TypeIndex Index);
};

// Needed by RandomAccessVisitorTest.cpp
#define TYPE_RECORD(EnumName, EnumVal, Name)
#define TYPE_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#define MEMBER_RECORD(EnumName, EnumVal, Name)                                 \
  extern template LLVM_TEMPLATE_ABI void ContinuationRecordBuilder::writeMemberType(    \
      Name##Record &Record);
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "llvm/DebugInfo/CodeView/CodeViewTypes.def"
#undef TYPE_RECORD
#undef TYPE_RECORD_ALIAS
#undef MEMBER_RECORD
#undef MEMBER_RECORD_ALIAS

} // namespace codeview
} // namespace llvm

#endif
