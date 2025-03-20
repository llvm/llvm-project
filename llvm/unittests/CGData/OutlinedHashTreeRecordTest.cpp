//===- OutlinedHashTreeRecordTest.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CGData/OutlinedHashTreeRecord.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(OutlinedHashTreeRecordTest, Empty) {
  OutlinedHashTreeRecord HashTreeRecord;
  ASSERT_TRUE(HashTreeRecord.empty());
}

TEST(OutlinedHashTreeRecordTest, Print) {
  OutlinedHashTreeRecord HashTreeRecord;
  HashTreeRecord.HashTree->insert({{1, 2}, 3});

  const char *ExpectedTreeStr = R"(---
0:
  Hash:            0x0
  Terminals:       0
  SuccessorIds:    [ 1 ]
1:
  Hash:            0x1
  Terminals:       0
  SuccessorIds:    [ 2 ]
2:
  Hash:            0x2
  Terminals:       3
  SuccessorIds:    [  ]
...
)";
  std::string TreeDump;
  raw_string_ostream OS(TreeDump);
  HashTreeRecord.print(OS);
  EXPECT_EQ(ExpectedTreeStr, TreeDump);
}

TEST(OutlinedHashTreeRecordTest, Stable) {
  OutlinedHashTreeRecord HashTreeRecord1;
  HashTreeRecord1.HashTree->insert({{1, 2}, 4});
  HashTreeRecord1.HashTree->insert({{1, 3}, 5});

  OutlinedHashTreeRecord HashTreeRecord2;
  HashTreeRecord2.HashTree->insert({{1, 3}, 5});
  HashTreeRecord2.HashTree->insert({{1, 2}, 4});

  // Output is stable regardless of insertion order.
  std::string TreeDump1;
  raw_string_ostream OS1(TreeDump1);
  HashTreeRecord1.print(OS1);
  std::string TreeDump2;
  raw_string_ostream OS2(TreeDump2);
  HashTreeRecord2.print(OS2);

  EXPECT_EQ(TreeDump1, TreeDump2);
}

TEST(OutlinedHashTreeRecordTest, Serialize) {
  OutlinedHashTreeRecord HashTreeRecord1;
  HashTreeRecord1.HashTree->insert({{1, 2}, 4});
  HashTreeRecord1.HashTree->insert({{1, 3}, 5});

  // Serialize and deserialize the tree.
  SmallVector<char> Out;
  raw_svector_ostream OS(Out);
  HashTreeRecord1.serialize(OS);

  OutlinedHashTreeRecord HashTreeRecord2;
  const uint8_t *Data = reinterpret_cast<const uint8_t *>(Out.data());
  HashTreeRecord2.deserialize(Data);

  // Two trees should be identical.
  std::string TreeDump1;
  raw_string_ostream OS1(TreeDump1);
  HashTreeRecord1.print(OS1);
  std::string TreeDump2;
  raw_string_ostream OS2(TreeDump2);
  HashTreeRecord2.print(OS2);

  EXPECT_EQ(TreeDump1, TreeDump2);
}

TEST(OutlinedHashTreeRecordTest, SerializeYAML) {
  OutlinedHashTreeRecord HashTreeRecord1;
  HashTreeRecord1.HashTree->insert({{1, 2}, 4});
  HashTreeRecord1.HashTree->insert({{1, 3}, 5});

  // Serialize and deserialize the tree in a YAML format.
  std::string Out;
  raw_string_ostream OS(Out);
  yaml::Output YOS(OS);
  HashTreeRecord1.serializeYAML(YOS);

  OutlinedHashTreeRecord HashTreeRecord2;
  yaml::Input YIS(StringRef(Out.data(), Out.size()));
  HashTreeRecord2.deserializeYAML(YIS);

  // Two trees should be identical.
  std::string TreeDump1;
  raw_string_ostream OS1(TreeDump1);
  HashTreeRecord1.print(OS1);
  std::string TreeDump2;
  raw_string_ostream OS2(TreeDump2);
  HashTreeRecord2.print(OS2);

  EXPECT_EQ(TreeDump1, TreeDump2);
}

} // end namespace
