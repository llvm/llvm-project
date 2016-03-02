//===- YAMLTest.cpp - Tests for Object YAML -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
