//===- TypeTableBuilderTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APSInt.h"
#include "llvm/DebugInfo/CodeView/AppendingTypeTableBuilder.h"
#include "llvm/DebugInfo/CodeView/ContinuationRecordBuilder.h"
#include "llvm/DebugInfo/CodeView/GlobalTypeTableBuilder.h"
#include "llvm/DebugInfo/CodeView/MergingTypeTableBuilder.h"
#include "llvm/DebugInfo/CodeView/TypeIndexDiscovery.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/Support/Allocator.h"

#include "gtest/gtest.h"

#include <string>
#include <vector>

using namespace llvm;
using namespace llvm::codeview;

namespace {

// Enough enumerators that the field list is split into at least three
// segments, so that there is a segment whose continuation refers to a segment
// that itself has a continuation.
constexpr unsigned NumEnumerators = 2500;

template <typename TableT> class TypeTableBuilderTest : public ::testing::Test {
protected:
  TypeTableBuilderTest() : Table(Storage) {
    for (unsigned I = 0; I < NumEnumerators; ++I)
      Names.push_back("ENUMERATOR_" + std::to_string(I) +
                      "_WITH_A_LONG_NAME_SO_THE_FIELD_LIST_IS_LARGE");
  }

  // Writes an enum field list that is too large for a single record. Only the
  // value of the first enumerator, which ends up in the first (head) segment,
  // depends on \p FirstValue.
  TypeIndex insertLargeFieldList(int64_t FirstValue) {
    ContinuationRecordBuilder CRB;
    CRB.begin(ContinuationRecordKind::FieldList);
    for (unsigned I = 0; I < NumEnumerators; ++I) {
      int64_t Value = I == 0 ? FirstValue : I;
      EnumeratorRecord E(MemberAccess::Public, APSInt(APInt(64, Value), false),
                         Names[I]);
      CRB.writeMemberType(E);
    }
    return Table.insertRecord(CRB);
  }

  // A type stream must be topologically sorted: every type index a record
  // refers to, including LF_INDEX continuations, must belong to an earlier
  // record.
  void expectTopologicallySorted() {
    ArrayRef<ArrayRef<uint8_t>> Records = Table.records();
    for (unsigned I = 0; I < Records.size(); ++I) {
      TypeIndex Self = TypeIndex::fromArrayIndex(I);
      SmallVector<TypeIndex, 8> Refs;
      discoverTypeIndices(Records[I], Refs);
      for (TypeIndex Ref : Refs) {
        if (Ref.isSimple() || Ref.isNoneType())
          continue;
        EXPECT_LT(Ref.getIndex(), Self.getIndex())
            << "record " << Self.getIndex() << " refers to "
            << Ref.getIndex();
      }
    }
  }

  BumpPtrAllocator Storage;
  TableT Table;
  std::vector<std::string> Names;
};

using DeduplicatingTables =
    ::testing::Types<GlobalTypeTableBuilder, MergingTypeTableBuilder>;
template <typename TableT>
using DeduplicatingTypeTableBuilderTest = TypeTableBuilderTest<TableT>;
TYPED_TEST_SUITE(DeduplicatingTypeTableBuilderTest, DeduplicatingTables, );

TYPED_TEST(DeduplicatingTypeTableBuilderTest, IdenticalContinuedRecord) {
  TypeIndex First = this->insertLargeFieldList(0);
  uint32_t NumSegments = this->Table.size();
  ASSERT_GE(NumSegments, 3u);

  // Every segment of an identical field list is deduplicated, so inserting it
  // again must return the original index and add no records.
  TypeIndex Second = this->insertLargeFieldList(0);
  EXPECT_EQ(First, Second);
  EXPECT_EQ(NumSegments, this->Table.size());
  this->expectTopologicallySorted();
}

TYPED_TEST(DeduplicatingTypeTableBuilderTest, ContinuedRecordWithSharedTail) {
  TypeIndex First = this->insertLargeFieldList(0);
  uint32_t NumSegments = this->Table.size();
  ASSERT_GE(NumSegments, 3u);

  // Only the head segment differs. The other segments are deduplicated, and
  // the new head segment must continue into the existing ones rather than
  // into the indices they would have had if they had been added.
  TypeIndex Second = this->insertLargeFieldList(-1);
  EXPECT_NE(First, Second);
  EXPECT_EQ(NumSegments + 1, this->Table.size());
  this->expectTopologicallySorted();
}

using AppendingTypeTableBuilderTest =
    TypeTableBuilderTest<AppendingTypeTableBuilder>;

TEST_F(AppendingTypeTableBuilderTest, ContinuedRecord) {
  TypeIndex First = insertLargeFieldList(0);
  uint32_t NumSegments = Table.size();
  ASSERT_GE(NumSegments, 3u);

  // Nothing is deduplicated, so the second copy gets all new records.
  TypeIndex Second = insertLargeFieldList(0);
  EXPECT_NE(First, Second);
  EXPECT_EQ(2 * NumSegments, Table.size());
  expectTopologicallySorted();
}

} // namespace
