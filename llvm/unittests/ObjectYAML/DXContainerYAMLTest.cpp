//===- DXContainerTest.cpp - Tests for DXContainerFile --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ObjectYAML/ObjectYAML.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

static bool convert(SmallVectorImpl<char> &Output, const char *YAML) {
  raw_svector_ostream OS(Output);
  yaml::Input YIn(YAML);
  return convertYAML(YIn, OS, [](const Twine &Err) { errs() << Err; });
}

TEST(DXCFile, ParseEmptyParts) {
  SmallString<128> Storage;

  // First read a fully explicit yaml with all sizes and offsets provided
  ASSERT_TRUE(convert(Storage, R"(--- !dxcontainer
Header:
  Hash:            [ 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 
                     0x0, 0x0, 0x0, 0x0, 0x0, 0x0 ]
  Version:
    Major:           1
    Minor:           0
  FileSize:        116
  PartCount:       7
  PartOffsets:     [ 60, 68, 76, 84, 92, 100, 108 ]
Parts:
  - Name:            FKE0
    Size:            0
  - Name:            FKE1
    Size:            0
  - Name:            FKE2
    Size:            0
  - Name:            FKE3
    Size:            0
  - Name:            FKE4
    Size:            0
  - Name:            FKE5
    Size:            0
  - Name:            FKE6
    Size:            0
...
  )"));

  // Result
  char Buffer[] = {
      0x44, 0x58, 0x42, 0x43, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x74, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x3C, 0x00, 0x00, 0x00,
      0x44, 0x00, 0x00, 0x00, 0x4C, 0x00, 0x00, 0x00, 0x54, 0x00, 0x00, 0x00,
      0x5C, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0x6C, 0x00, 0x00, 0x00,
      0x46, 0x4B, 0x45, 0x30, 0x00, 0x00, 0x00, 0x00, 0x46, 0x4B, 0x45, 0x31,
      0x00, 0x00, 0x00, 0x00, 0x46, 0x4B, 0x45, 0x32, 0x00, 0x00, 0x00, 0x00,
      0x46, 0x4B, 0x45, 0x33, 0x00, 0x00, 0x00, 0x00, 0x46, 0x4B, 0x45, 0x34,
      0x00, 0x00, 0x00, 0x00, 0x46, 0x4B, 0x45, 0x35, 0x00, 0x00, 0x00, 0x00,
      0x46, 0x4B, 0x45, 0x36, 0x00, 0x00, 0x00, 0x00,
  };

  EXPECT_EQ(Storage.size(), 116u);
  EXPECT_TRUE(memcmp(Buffer, Storage.data(), 116) == 0);

  Storage.clear();

  // Next, read the same file without the part offsets or file size. Both cases
  // should result in the same final output.
  ASSERT_TRUE(convert(Storage, R"(--- !dxcontainer
Header:
  Hash:            [ 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 
                     0x0, 0x0, 0x0, 0x0, 0x0, 0x0 ]
  Version:
    Major:           1
    Minor:           0
  PartCount:       7
Parts:
  - Name:            FKE0
    Size:            0
  - Name:            FKE1
    Size:            0
  - Name:            FKE2
    Size:            0
  - Name:            FKE3
    Size:            0
  - Name:            FKE4
    Size:            0
  - Name:            FKE5
    Size:            0
  - Name:            FKE6
    Size:            0
...
  )"));

  EXPECT_EQ(Storage.size(), 116u);
  EXPECT_TRUE(memcmp(Buffer, Storage.data(), 116) == 0);
}
