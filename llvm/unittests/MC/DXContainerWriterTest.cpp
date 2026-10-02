//===- DXContainerWriterTest.cpp - MCDXContainerWriter tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCDXContainerWriter.h"
#include "llvm/Object/DXContainer.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

namespace {

class TestDXContainerWriter : public MCDXContainerBaseWriter {
  SmallVector<StringRef> PartNames;
  SmallVector<StringRef> PartData;
  SmallVector<MCDXContainerPart> Parts;

protected:
  ArrayRef<MCDXContainerPart> collectParts() override { return Parts; }

public:
  void addPart(StringRef Name, ArrayRef<uint8_t> Data) {
    PartNames.emplace_back(Name);
    PartData.emplace_back(llvm::toStringRef(Data));
    Parts.push_back({PartNames.back(), PartData.back()});
  }
};

static Triple getTestTriple() {
  return Triple("dxilv1.3-pc-shadermodel6.3-library");
}

static std::string writeContainer(TestDXContainerWriter &Writer) {
  std::string Buffer;
  raw_string_ostream OS(Buffer);
  Writer.write(OS, getTestTriple());
  return Buffer;
}

static DXContainer parseContainer(StringRef Buffer) {
  return llvm::cantFail(DXContainer::create(MemoryBufferRef(Buffer, "")));
}

TEST(MCDXContainerWriterTest, PrivUnaligned) {
  TestDXContainerWriter Writer;
  const uint8_t PrivData[] = {0xDE, 0xAD, 0xBE, 0xEF, 0x42};
  Writer.addPart("PRIV", PrivData);

  std::string Buffer = writeContainer(Writer);
  DXContainer C = parseContainer(Buffer);

  EXPECT_EQ(C.getHeader().PartCount, 1u);
  EXPECT_EQ(C.getHeader().FileSize, 49u);
  EXPECT_EQ(C.getData().size(), 49u);

  ASSERT_TRUE(C.getPrivateData());
  EXPECT_EQ(C.getPrivateData()->size(), 5u);
  EXPECT_EQ(
      *C.getPrivateData(),
      StringRef(reinterpret_cast<const char *>(PrivData), sizeof(PrivData)));
}

TEST(MCDXContainerWriterTest, PrivMustBeLast) {
  TestDXContainerWriter Writer;
  const uint8_t PrivData[] = {0x42};
  const uint8_t DxilData[] = {0xBC, 0xC0, 0xDE, 0x00};
  Writer.addPart("PRIV", PrivData);
  Writer.addPart("DXIL", DxilData);

  EXPECT_DEATH(writeContainer(Writer),
               "PRIV must be the last section in a DXContainer");
}

} // namespace
