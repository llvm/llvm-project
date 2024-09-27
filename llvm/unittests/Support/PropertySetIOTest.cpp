//===- llvm/unittest/Support/PropertySetIO.cpp - Property set I/O tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/PropertySetIO.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::util;

namespace {

TEST(PropertySet, IntValuesIO) {
  // '1' in '1|20' means 'integer property'
  auto Content = "[Staff/Ages]\n"
                 "person1=1|20\n"
                 "person2=1|25\n"
                 "[Staff/Experience]\n"
                 "person1=1|1\n"
                 "person2=1|2\n"
                 "person3=1|12\n";
  auto MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  auto PropSetsPtr = PropertySetRegistry::read(MemBuf.get());

  if (!PropSetsPtr)
    FAIL() << "PropertySetRegistry::read failed\n";

  std::string Serialized;
  {
    llvm::raw_string_ostream OS(Serialized);
    // Serialize
    PropSetsPtr->get()->write(OS);
  }
  // Check that the original and the serialized version are equal
  ASSERT_EQ(Serialized, Content);
}

TEST(PropertySet, ByteArrayValuesIO) {
  // '2' in '2|...' means 'byte array property', Base64-encoded
  // encodes the following byte arrays:
  //   { 8, 0, 0, 0, 0, 0, 0, 0, 0x1 };
  //   { 40, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0x7F, 0xFF, 0x70 };
  // first 8 bytes are the size in bits (40) of what follows (5 bytes).

  auto Content = "[Opt/Param]\n"
                 "kernel1=2|IAAAAAAAAAQA\n"
                 "kernel2=2|oAAAAAAAAAw///3/wB\n";
  auto MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  auto PropSetsPtr = PropertySetRegistry::read(MemBuf.get());

  if (!PropSetsPtr)
    FAIL() << "PropertySetRegistry::read failed\n";

  std::string Serialized;
  {
    llvm::raw_string_ostream OS(Serialized);
    // Serialize
    PropSetsPtr->get()->write(OS);
  }
  // Check that the original and the serialized version are equal
  ASSERT_EQ(Serialized, Content);
}
} // namespace
