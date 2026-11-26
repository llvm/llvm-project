//===- DXContainerWriterTest.cpp - MCDXContainerWriter tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/DXContainerPSVInfo.h"
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

TEST(MCDXContainerWriterTest, SignaturePadding) {
  for (bool PadToDword : {false, true}) {
    mcdxbc::Signature Sig;
    Sig.addParam(0, "A", 0, dxbc::D3DSystemValue::Undefined,
                 dxbc::SigComponentType::Float32, 0, 1, 14,
                 dxbc::SigMinPrecision::Default);
    std::string SigData;
    raw_string_ostream OS(SigData);
    Sig.write(OS, PadToDword);
    EXPECT_EQ(SigData.size(), PadToDword ? 44u : 42u);

    for (StringRef Name : {"ISG1", "OSG1", "PSG1"}) {
      TestDXContainerWriter Writer;
      Writer.addPart(Name, {reinterpret_cast<const uint8_t *>(SigData.data()),
                            SigData.size()});
      const uint8_t Flags[] = {0, 0, 0, 0, 0, 0, 0, 0};
      Writer.addPart("SFI0", Flags);
      std::string Buffer = writeContainer(Writer);
      DXContainer C = parseContainer(Buffer);
      auto Part = C.begin();
      EXPECT_EQ(Part->Part.Size, SigData.size());
      uint32_t NextOffset =
          Part->Offset + sizeof(dxbc::PartHeader) + Part->Part.Size;
      ++Part;
      EXPECT_EQ(Part->Offset, NextOffset);
      EXPECT_EQ(Part->Part.Size, sizeof(Flags));
      EXPECT_EQ(C.getHeader().FileSize,
                Part->Offset + sizeof(dxbc::PartHeader) + sizeof(Flags));
    }
  }
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
