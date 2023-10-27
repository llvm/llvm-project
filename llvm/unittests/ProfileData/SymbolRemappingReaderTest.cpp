//===- unittests/Support/SymbolRemappingReaderTest.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/SymbolRemappingReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
class SymbolRemappingReaderTest : public testing::Test {
public:
  std::unique_ptr<MemoryBuffer> Buffer;
  SymbolRemappingReader Reader;

  std::string readWithErrors(StringRef Text, StringRef BufferName) {
    Buffer = MemoryBuffer::getMemBuffer(Text, BufferName);
    Error E = Reader.read(*Buffer);
    EXPECT_TRUE((bool)E);
    return toString(std::move(E));
  }

  void read(StringRef Text, StringRef BufferName) {
    Buffer = MemoryBuffer::getMemBuffer(Text, BufferName);
    Error E = Reader.read(*Buffer);
    EXPECT_FALSE((bool)E);
  }
};
} // unnamed namespace

TEST_F(SymbolRemappingReaderTest, ParseErrors) {
  EXPECT_EQ(readWithErrors("error", "foo.map"),
            "foo.map:1: Expected 'kind mangled_name mangled_name', "
            "found 'error'");

  EXPECT_EQ(readWithErrors("error m1 m2", "foo.map"),
            "foo.map:1: Invalid kind, expected 'name', 'type', or 'encoding', "
            "found 'error'");
}

TEST_F(SymbolRemappingReaderTest, DemanglingErrors) {
  EXPECT_EQ(readWithErrors("type i banana", "foo.map"),
            "foo.map:1: Could not demangle 'banana' as a <type>; "
            "invalid mangling?");
  EXPECT_EQ(readWithErrors("name i 1X", "foo.map"),
            "foo.map:1: Could not demangle 'i' as a <name>; "
            "invalid mangling?");
  EXPECT_EQ(readWithErrors("name 1X 1fv", "foo.map"),
            "foo.map:1: Could not demangle '1fv' as a <name>; "
            "invalid mangling?");
  EXPECT_EQ(readWithErrors("encoding 1fv 1f1gE", "foo.map"),
            "foo.map:1: Could not demangle '1f1gE' as a <encoding>; "
            "invalid mangling?");
}

TEST_F(SymbolRemappingReaderTest, BadMappingOrder) {
  StringRef Map = R"(
    # N::foo == M::bar
    name N1N3fooE N1M3barE

    # N:: == M::
    name 1N 1M
  )";
  EXPECT_EQ(readWithErrors(Map, "foo.map"),
            "foo.map:6: Manglings '1N' and '1M' have both been used in prior "
            "remappings. Move this remapping earlier in the file.");
}

TEST_F(SymbolRemappingReaderTest, RemappingsAdded) {
  StringRef Map = R"(
    # A::foo == B::bar
    name N1A3fooE N1B3barE

    # int == long
    type i l

    # void f<int>() = void g<int>()
    encoding 1fIiEvv 1gIiEvv
  )";

  read(Map, "foo.map");
  auto Key = Reader.insert("_ZN1B3bar3bazIiEEvv");
  EXPECT_NE(Key, SymbolRemappingReader::Key());
  EXPECT_EQ(Key, Reader.lookup("_ZN1A3foo3bazIlEEvv"));
  EXPECT_NE(Key, Reader.lookup("_ZN1C3foo3bazIlEEvv"));

  Key = Reader.insert("_Z1fIiEvv");
  EXPECT_NE(Key, SymbolRemappingReader::Key());
  EXPECT_EQ(Key, Reader.lookup("_Z1gIlEvv"));
}
