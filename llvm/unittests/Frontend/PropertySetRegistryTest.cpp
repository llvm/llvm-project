//===- llvm/unittest/Frontend/PropertySetRegistry.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/Offloading/PropertySet.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

using namespace llvm::offloading;
using namespace llvm;

void checkSerialization(const PropertySetRegistry &PSR) {
  SmallString<0> Serialized;
  raw_svector_ostream OS(Serialized);
  writePropertiesToJSON(PSR, OS);
  auto PSR2 = readPropertiesFromJSON({Serialized, ""});
  ASSERT_EQ("", toString(PSR2.takeError()));
  EXPECT_EQ(PSR, *PSR2);
}

TEST(PropertySetRegistryTest, PropertySetRegistry) {
  PropertySetRegistry PSR;
  checkSerialization(PSR);

  PSR["Category1"]["Prop1"] = 42U;
  PSR["Category1"]["Prop2"] = ByteArray(StringRef("Hello").bytes());
  PSR["Category2"]["A"] = ByteArray{0, 4, 16, 32, 255};
  checkSerialization(PSR);

  PSR = PropertySetRegistry();
  PSR["ABC"]["empty_array"] = ByteArray();
  PSR["ABC"]["max_val"] = std::numeric_limits<uint32_t>::max();
  checkSerialization(PSR);
}

TEST(PropertySetRegistryTest, IllFormedJSON) {
  SmallString<0> Input;

  // Invalid json
  Input = "{ invalid }";
  auto Res = readPropertiesFromJSON({Input, ""});
  EXPECT_NE("", toString(Res.takeError()));

  Input = "";
  Res = readPropertiesFromJSON({Input, ""});
  EXPECT_NE("", toString(Res.takeError()));

  // Not a JSON object
  Input = "[1, 2, 3]";
  Res = readPropertiesFromJSON({Input, ""});
  EXPECT_NE("", toString(Res.takeError()));

  // Property set not an object
  Input = R"({ "Category": 42 })";
  Res = readPropertiesFromJSON({Input, ""});
  EXPECT_NE("", toString(Res.takeError()));

  // Property value has non string/non-integer type
  Input = R"({ "Category": { "Prop": [1, 2, 3] } })";
  Res = readPropertiesFromJSON({Input, ""});
  EXPECT_NE("", toString(Res.takeError()));

  // Property value is an invalid base64 string
  Input = R"({ "Category": { "Prop": ";" } })";
  Res = readPropertiesFromJSON({Input, ""});
  EXPECT_NE("", toString(Res.takeError()));

  Input = R"({ "Category": { "Prop": "!@#$" } })";
  Res = readPropertiesFromJSON({Input, ""});
  EXPECT_NE("", toString(Res.takeError()));
}
