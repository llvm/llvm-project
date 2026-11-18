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
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecordMapping.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Support/Compiler.h"
#include <cstdint>
#include <vector>

namespace llvm {
namespace codeview {
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
  void finishRecord();

public:
  LLVM_ABI ContinuationRecordBuilder();
  LLVM_ABI ~ContinuationRecordBuilder();

  LLVM_ABI void begin(ContinuationRecordKind RecordKind);

  // This template is explicitly instantiated in the implementation file for all
  // supported types.  The method itself is ugly, so inlining it into the header
  // file clutters an otherwise straightforward interface.
  template <typename RecordType> void writeMemberType(RecordType &Record);

  /// Finishes the record and returns its segments in the order they must be
  /// committed to the type stream, assuming the segments will be assigned the
  /// consecutive type indices \p Index, \p Index + 1, ... Only use this when
  /// the segments are guaranteed to receive those indices; a type table that
  /// deduplicates records should use the overload below instead.
  LLVM_ABI std::vector<CVType> end(TypeIndex Index);

  /// Finishes the record and passes each of its segments to \p InsertSegment,
  /// in the order they must be committed to the type stream. \p InsertSegment
  /// returns the type index the segment was assigned, and the continuation in
  /// the following segment is set to refer to that index. This keeps the
  /// chain of segments valid when a type table deduplicates a segment against
  /// an existing record and returns that record's index instead of the next
  /// one. Returns the type index of the last segment inserted, which is the
  /// index of the complete record.
  LLVM_ABI TypeIndex
  end(function_ref<TypeIndex(ArrayRef<uint8_t>)> InsertSegment);
};
} // namespace codeview
} // namespace llvm

#endif
