// llvm/unittest/Support/SYCLPropertySetIOTest.cpp - SYCL Property set I/O tests
// //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SYCLPropertySetIO.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::util;

namespace {

TEST(SYCLPropertySet, IncorrectValuesIO) {
  auto Content = "Staff/Ages]\n";
  auto MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  auto PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());
  EXPECT_THAT_ERROR(std::move(PropSetsPtr.takeError()), Failed())
      << "Invalid line";

  Content = "[Staff/Ages\n";
  MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());
  EXPECT_THAT_ERROR(std::move(PropSetsPtr.takeError()), Failed())
      << "Invalid line";

  Content = "[Staff/Ages]\n"
            "person1=\n";
  MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());
  EXPECT_THAT_ERROR(std::move(PropSetsPtr.takeError()), Failed())
      << "Invalid property line";

  Content = "[Staff/Ages]\n"
            "person1=|10\n";
  MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());
  EXPECT_THAT_ERROR(std::move(PropSetsPtr.takeError()), Failed())
      << "Invalid property value";

  Content = "[Staff/Ages]\n"
            "person1=abc|10\n";
  MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());
  EXPECT_THAT_ERROR(std::move(PropSetsPtr.takeError()), Failed())
      << "Invalid property type";

  Content = "[Staff/Ages]\n"
            "person1=1|IAQ\n";
  MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());
  EXPECT_THAT_ERROR(std::move(PropSetsPtr.takeError()), Failed())
      << "Invalid property value";

  Content = "[Staff/Ages]\n"
            "person1=2|IAQ\n";
  MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());
  EXPECT_THAT_ERROR(std::move(PropSetsPtr.takeError()), Failed())
      << "Invalid property value";

  Content = "[Staff/Ages]\n"
            "person1=100|10\n";
  MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());
  EXPECT_THAT_ERROR(std::move(PropSetsPtr.takeError()), Failed())
      << "Invalid property type";
}

TEST(SYCLPropertySet, IntValuesIO) {
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
  auto PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());

  if (!PropSetsPtr)
    FAIL() << "SYCLPropertySetRegistry::read failed\n";

  std::string Serialized;
  {
    llvm::raw_string_ostream OS(Serialized);
    // Serialize
    PropSetsPtr->get()->write(OS);
  }
  // Check that the original and the serialized version are equal
  EXPECT_EQ(Serialized, Content);
}

TEST(SYCLPropertySet, ByteArrayValuesIO) {
  // '2' in '2|...' means 'byte array property', Base64-encoded
  // encodes the following byte arrays:
  //   { 8, 0, 0, 0, 0, 0, 0, 0, 0x1 };
  //   { 40, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0x7F, 0xFF, 0x70 };
  // first 8 bytes are the size in bits (40) of what follows (5 bytes).

  auto Content = "[Opt/Param]\n"
                 "kernel1=2|IAAAAAAAAAQA\n"
                 "kernel2=2|oAAAAAAAAAAAw///3/wB\n";
  auto MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  auto PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());

  if (!PropSetsPtr)
    FAIL() << "SYCLPropertySetRegistry::read failed\n";

  std::string Serialized;
  {
    llvm::raw_string_ostream OS(Serialized);
    // Serialize
    PropSetsPtr->get()->write(OS);
  }
  // Check that the original and the serialized version are equal
  EXPECT_EQ(Serialized, Content);
}
} // namespace
