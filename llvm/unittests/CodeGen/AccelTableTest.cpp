//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/AccelTable.h"
#include "TestAsmPrinter.h"
#include "llvm/CodeGen/DwarfStringPoolEntry.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class AccelTableTest : public testing::Test {
protected:
  void SetUp() override {
    auto ExpectedTestPrinter = TestAsmPrinter::create(
        "x86_64-pc-linux", /*DwarfVersion=*/5, dwarf::DWARF32);
    ASSERT_THAT_EXPECTED(ExpectedTestPrinter, Succeeded());
    TestPrinter = std::move(*ExpectedTestPrinter);
    if (!TestPrinter)
      GTEST_SKIP();
  }

  /// Builds a table holding \p Names and returns each bucket's contents in the
  /// order they would be emitted in.
  std::vector<std::vector<StringRef>>
  bucketOrder(ArrayRef<const DwarfStringPoolEntryWithExtString *> Names) {
    DWARF5AccelTable Table;
    for (const DwarfStringPoolEntryWithExtString *Name : Names)
      Table.addName(DwarfStringPoolEntryRef(*Name), /*DieOffset=*/0x11,
                    /*DefiningParentOffset=*/std::nullopt,
                    /*DieTag=*/dwarf::DW_TAG_variable, /*UnitID=*/0,
                    /*IsTU=*/false);
    Table.finalize(TestPrinter->getAP(), "names");

    std::vector<std::vector<StringRef>> Order;
    for (const AccelTableBase::HashList &Bucket : Table.getBuckets()) {
      Order.emplace_back();
      for (const AccelTableBase::HashData *Hash : Bucket)
        Order.back().push_back(Hash->Name.getString());
    }
    return Order;
  }

  std::unique_ptr<TestAsmPrinter> TestPrinter;
};

TEST_F(AccelTableTest, CollidingNamesOrderedIndependentlyOfInsertion) {
  // The DWARF v5 hash folds case, so these three names always collide.
  DwarfStringPoolEntryWithExtString Lower = {{}, "fixups"};
  DwarfStringPoolEntryWithExtString Mixed = {{}, "Fixups"};
  DwarfStringPoolEntryWithExtString Upper = {{}, "FIXUPS"};
  DwarfStringPoolEntryWithExtString Other = {{}, "gamma"};

  const uint32_t Hash = DWARF5AccelTableData::hash(Lower.String);
  ASSERT_EQ(Hash, DWARF5AccelTableData::hash(Mixed.String));
  ASSERT_EQ(Hash, DWARF5AccelTableData::hash(Upper.String));
  ASSERT_LT(DWARF5AccelTableData::hash(Other.String), Hash);

  const std::vector<std::vector<StringRef>> Order =
      bucketOrder({&Lower, &Mixed, &Upper, &Other});
  EXPECT_EQ(bucketOrder({&Other, &Upper, &Mixed, &Lower}), Order);
  EXPECT_EQ(bucketOrder({&Mixed, &Other, &Lower, &Upper}), Order);

  // The hash value orders before the name, so "gamma" comes first even though
  // it sorts after the names it shares a bucket with.
  EXPECT_THAT(Order, testing::Contains(testing::ElementsAre(
                         "gamma", "FIXUPS", "Fixups", "fixups")));
}

} // end namespace
