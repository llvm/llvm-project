//===- llvm/unittest/DWARFLinkerParallel/StringTableTest.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DWARFLinkerParallel/StringTable.h"
#include "llvm/Support/Parallel.h"
#include "gtest/gtest.h"
#include <cstdlib>

using namespace llvm;
using namespace dwarflinker_parallel;

namespace {

TEST(StringPoolTest, TestStringTable) {
  struct StringDescription {
    const char *Str = nullptr;
    uint64_t Idx = 0;
    uint64_t Offset = 0;
  };

  SmallVector<StringDescription> InputStrings = {
      {"first", 0, 0}, {"second", 1, 6}, {"third", 2, 13}};

  StringPool Strings;
  StringTable OutStrings(Strings, nullptr);

  // Check string insertion.
  StringEntry *FirstPtr = Strings.insert(InputStrings[0].Str).first;
  StringEntry *SecondPtr = Strings.insert(InputStrings[1].Str).first;
  StringEntry *ThirdPtr = Strings.insert(InputStrings[2].Str).first;

  FirstPtr = OutStrings.add(FirstPtr);
  SecondPtr = OutStrings.add(SecondPtr);
  ThirdPtr = OutStrings.add(ThirdPtr);

  // Check fields of inserted strings.
  EXPECT_TRUE(FirstPtr->getKey() == InputStrings[0].Str);
  EXPECT_TRUE(FirstPtr->getValue()->Offset == InputStrings[0].Offset);
  EXPECT_TRUE(FirstPtr->getValue()->Index == InputStrings[0].Idx);

  EXPECT_TRUE(SecondPtr->getKey() == InputStrings[1].Str);
  EXPECT_TRUE(SecondPtr->getValue()->Offset == InputStrings[1].Offset);
  EXPECT_TRUE(SecondPtr->getValue()->Index == InputStrings[1].Idx);

  EXPECT_TRUE(ThirdPtr->getKey() == InputStrings[2].Str);
  EXPECT_TRUE(ThirdPtr->getValue()->Offset == InputStrings[2].Offset);
  EXPECT_TRUE(ThirdPtr->getValue()->Index == InputStrings[2].Idx);

  // Check order enumerated strings.
  uint64_t CurIdx = 0;
  std::function<void(DwarfStringPoolEntryRef)> checkStr =
      [&](DwarfStringPoolEntryRef Entry) {
        EXPECT_TRUE(Entry.getEntry().isIndexed());
        EXPECT_TRUE(Entry.getIndex() == CurIdx);
        EXPECT_TRUE(Entry.getOffset() == InputStrings[CurIdx].Offset);
        EXPECT_TRUE(Entry.getString() == InputStrings[CurIdx].Str);

        CurIdx++;
      };

  OutStrings.forEach(checkStr);
}

TEST(StringPoolTest, TestStringTableWithTranslator) {
  std::string Word;
  std::function<StringRef(StringRef)> TranslatorFunc =
      [&](StringRef InputString) -> StringRef {
    Word.clear();
    for (auto Sym : InputString)
      Word.insert(Word.begin(), Sym);
    Word += '0';
    return Word;
  };

  StringPool Strings;
  StringTable OutStrings(Strings, TranslatorFunc);

  StringEntry *FirstPtr = Strings.insert("first").first;
  StringEntry *SecondPtr = Strings.insert("second").first;
  StringEntry *ThirdPtr = Strings.insert("third").first;

  FirstPtr = OutStrings.add(FirstPtr);
  SecondPtr = OutStrings.add(SecondPtr);
  ThirdPtr = OutStrings.add(ThirdPtr);

  EXPECT_TRUE(FirstPtr->getKey() == "tsrif0");
  EXPECT_TRUE(FirstPtr->getValue()->Offset == 0);
  EXPECT_TRUE(FirstPtr->getValue()->Index == 0);

  EXPECT_TRUE(SecondPtr->getKey() == "dnoces0");
  EXPECT_TRUE(SecondPtr->getValue()->Offset == 7);
  EXPECT_TRUE(SecondPtr->getValue()->Index == 1);

  EXPECT_TRUE(ThirdPtr->getKey() == "driht0");
  EXPECT_TRUE(ThirdPtr->getValue()->Offset == 15);
  EXPECT_TRUE(ThirdPtr->getValue()->Index == 2);
}

} // anonymous namespace
