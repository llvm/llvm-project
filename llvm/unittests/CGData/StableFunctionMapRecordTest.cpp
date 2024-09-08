//===- StableFunctionMapRecordTest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CGData/StableFunctionMapRecord.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(StableFunctionMapRecordTest, Print) {
  StableFunctionMapRecord MapRecord;
  StableFunction Func1{1, "Func1", "Mod1", 2, {{{0, 1}, 3}}};
  MapRecord.FunctionMap->insert(Func1);

  const char *ExpectedMapStr = R"(---
- Hash:            1
  FunctionName:    Func1
  ModuleName:      Mod1
  InstCount:       2
  IndexOperandHashes:
    - InstIndex:       0
      OpndIndex:       1
      OpndHash:        3
...
)";
  std::string MapDump;
  raw_string_ostream OS(MapDump);
  MapRecord.print(OS);
  EXPECT_EQ(ExpectedMapStr, MapDump);
}

TEST(StableFunctionMapRecordTest, Stable) {
  StableFunction Func1{1, "Func2", "Mod1", 1, {}};
  StableFunction Func2{1, "Func3", "Mod1", 1, {}};
  StableFunction Func3{1, "Func1", "Mod2", 1, {}};
  StableFunction Func4{2, "Func4", "Mod3", 1, {}};

  StableFunctionMapRecord MapRecord1;
  MapRecord1.FunctionMap->insert(Func1);
  MapRecord1.FunctionMap->insert(Func2);
  MapRecord1.FunctionMap->insert(Func3);
  MapRecord1.FunctionMap->insert(Func4);

  StableFunctionMapRecord MapRecord2;
  MapRecord2.FunctionMap->insert(Func4);
  MapRecord2.FunctionMap->insert(Func3);
  MapRecord2.FunctionMap->insert(Func2);
  MapRecord2.FunctionMap->insert(Func1);

  // Output is sorted by hash (1 < 2), module name (Mod1 < Mod2), and function
  // name (Func2 < Func3).
  std::string MapDump1;
  raw_string_ostream OS1(MapDump1);
  MapRecord1.print(OS1);
  std::string MapDump2;
  raw_string_ostream OS2(MapDump2);
  MapRecord2.print(OS2);
  EXPECT_EQ(MapDump1, MapDump2);
}

TEST(StableFunctionMapRecordTest, Serialize) {
  StableFunctionMapRecord MapRecord1;
  StableFunction Func1{1, "Func1", "Mod1", 2, {{{0, 1}, 3}, {{1, 2}, 4}}};
  StableFunction Func2{2, "Func2", "Mod1", 3, {{{0, 1}, 2}}};
  StableFunction Func3{2, "Func3", "Mod1", 3, {{{0, 1}, 3}}};
  MapRecord1.FunctionMap->insert(Func1);
  MapRecord1.FunctionMap->insert(Func2);
  MapRecord1.FunctionMap->insert(Func3);

  // Serialize and deserialize the map.
  SmallVector<char> Out;
  raw_svector_ostream OS(Out);
  MapRecord1.serialize(OS);

  StableFunctionMapRecord MapRecord2;
  const uint8_t *Data = reinterpret_cast<const uint8_t *>(Out.data());
  MapRecord2.deserialize(Data);

  // Two maps should be identical.
  std::string MapDump1;
  raw_string_ostream OS1(MapDump1);
  MapRecord1.print(OS1);
  std::string MapDump2;
  raw_string_ostream OS2(MapDump2);
  MapRecord2.print(OS2);

  EXPECT_EQ(MapDump1, MapDump2);
}

TEST(StableFunctionMapRecordTest, SerializeYAML) {
  StableFunctionMapRecord MapRecord1;
  StableFunction Func1{1, "Func1", "Mod1", 2, {{{0, 1}, 3}, {{1, 2}, 4}}};
  StableFunction Func2{2, "Func2", "Mod1", 3, {{{0, 1}, 2}}};
  StableFunction Func3{2, "Func3", "Mod1", 3, {{{0, 1}, 3}}};
  MapRecord1.FunctionMap->insert(Func1);
  MapRecord1.FunctionMap->insert(Func2);
  MapRecord1.FunctionMap->insert(Func3);

  // Serialize and deserialize the map in a YAML format.
  std::string Out;
  raw_string_ostream OS(Out);
  yaml::Output YOS(OS);
  MapRecord1.serializeYAML(YOS);

  StableFunctionMapRecord MapRecord2;
  yaml::Input YIS(StringRef(Out.data(), Out.size()));
  MapRecord2.deserializeYAML(YIS);

  // Two maps should be identical.
  std::string MapDump1;
  raw_string_ostream OS1(MapDump1);
  MapRecord1.print(OS1);
  std::string MapDump2;
  raw_string_ostream OS2(MapDump2);
  MapRecord2.print(OS2);

  EXPECT_EQ(MapDump1, MapDump2);
}

} // end namespace
