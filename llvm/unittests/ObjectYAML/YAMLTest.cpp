//===- YAMLTest.cpp - Tests for Object YAML -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/YAML.h"
#include "llvm/Support/YAMLTraits.h"
#include "gtest/gtest.h"

using namespace llvm;

struct BinaryHolder {
  yaml::BinaryRef Binary;
};

namespace llvm {
namespace yaml {
template <>
struct MappingTraits<BinaryHolder> {
  static void mapping(IO &IO, BinaryHolder &BH) {
    IO.mapRequired("Binary", BH.Binary);
  }
};
} // end namespace yaml
} // end namespace llvm

TEST(ObjectYAML, BinaryRef) {
  BinaryHolder BH;
  SmallVector<char, 32> Buf;
  llvm::raw_svector_ostream OS(Buf);
  yaml::Output YOut(OS);
  YOut << BH;
  EXPECT_NE(OS.str().find("''"), StringRef::npos);
}

TEST(ObjectYAML, UnknownOption) {
  StringRef InputYAML = "InvalidKey: InvalidValue\n"
                        "Binary: AAAA\n";
  BinaryHolder BH;
  yaml::Input Input(InputYAML);
  // test 1: default in trying to parse invalid key is an error case.
  Input >> BH;
  EXPECT_EQ(Input.error().value(), 22);

  // test 2: only warn about invalid key if actively set.
  yaml::Input Input2(InputYAML);
  BinaryHolder BH2;
  Input2.setAllowUnknownKeys(true);
  Input2 >> BH2;
  EXPECT_EQ(BH2.Binary, yaml::BinaryRef("AAAA"));
  EXPECT_EQ(Input2.error().value(), 0);
}
