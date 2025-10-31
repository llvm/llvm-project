//===----------------------- ELFTypesTest.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/Object/ELFTypes.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <iostream>

using namespace llvm;
using namespace llvm::object;

template <typename ELFT> using Elf_Note = typename ELFT::Note;

template <class ELFT> struct NoteTestData {
  std::vector<uint8_t> Data;

  const Elf_Note_Impl<ELFT> getElfNote(StringRef Name, uint32_t Type,
                                       ArrayRef<uint8_t> Desc) {
    constexpr uint64_t Align = 4;
    Data.resize(alignTo(sizeof(Elf_Nhdr_Impl<ELFT>) + Name.size(), Align) +
                    alignTo(Desc.size(), Align),
                0);

    Elf_Nhdr_Impl<ELFT> *Nhdr =
        reinterpret_cast<Elf_Nhdr_Impl<ELFT> *>(Data.data());
    Nhdr->n_namesz = (Name == "") ? 0 : Name.size() + 1;
    Nhdr->n_descsz = Desc.size();
    Nhdr->n_type = Type;

    auto NameOffset = Data.begin() + sizeof(*Nhdr);
    std::copy(Name.begin(), Name.end(), NameOffset);

    auto DescOffset =
        Data.begin() + alignTo(sizeof(*Nhdr) + Nhdr->n_namesz, Align);
    std::copy(Desc.begin(), Desc.end(), DescOffset);

    return Elf_Note_Impl<ELFT>(*Nhdr);
  }
};

TEST(ELFTypesTest, NoteTest) {
  static const uint8_t Random[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  ArrayRef<uint8_t> RandomData = ArrayRef(Random);
  NoteTestData<ELF64LE> TestData;

  auto Note1 = TestData.getElfNote(StringRef("AMD"), ELF::NT_AMDGPU_METADATA,
                                   RandomData);
  EXPECT_EQ(Note1.getName(), "AMD");
  EXPECT_EQ(Note1.getType(), ELF::NT_AMDGPU_METADATA);
  EXPECT_EQ(Note1.getDesc(4), RandomData);
  EXPECT_EQ(Note1.getDescAsStringRef(4),
            StringRef(reinterpret_cast<const char *>(Random), sizeof(Random)));

  auto Note2 = TestData.getElfNote("", ELF::NT_AMDGPU_METADATA, RandomData);
  EXPECT_EQ(Note2.getName(), "");

  auto Note3 =
      TestData.getElfNote("AMD", ELF::NT_AMDGPU_METADATA, ArrayRef<uint8_t>());
  EXPECT_EQ(Note3.getDescAsStringRef(4), StringRef(""));
}

TEST(ELFTypesTest, BBEntryMetadataEncodingTest) {
  const std::array<BBAddrMap::BBEntry::Metadata, 7> Decoded = {
      {{false, false, false, false, false},
       {true, false, false, false, false},
       {false, true, false, false, false},
       {false, false, true, false, false},
       {false, false, false, true, false},
       {false, false, false, false, true},
       {true, true, true, true, true}}};
  const std::array<uint32_t, 7> Encoded = {{0, 1, 2, 4, 8, 16, 31}};
  for (size_t i = 0; i < Decoded.size(); ++i)
    EXPECT_EQ(Decoded[i].encode(), Encoded[i]);
  for (size_t i = 0; i < Encoded.size(); ++i) {
    Expected<BBAddrMap::BBEntry::Metadata> MetadataOrError =
        BBAddrMap::BBEntry::Metadata::decode(Encoded[i]);
    ASSERT_THAT_EXPECTED(MetadataOrError, Succeeded());
    EXPECT_EQ(*MetadataOrError, Decoded[i]);
  }
}

TEST(ELFTypesTest, BBEntryMetadataInvalidEncodingTest) {
  const std::array<std::string, 2> Errors = {
      "invalid encoding for BBEntry::Metadata: 0xffff",
      "invalid encoding for BBEntry::Metadata: 0x100001"};
  const std::array<uint32_t, 2> Values = {{0xFFFF, 0x100001}};
  for (size_t i = 0; i < Values.size(); ++i) {
    EXPECT_THAT_ERROR(
        BBAddrMap::BBEntry::Metadata::decode(Values[i]).takeError(),
        FailedWithMessage(Errors[i]));
  }
}

static_assert(
    std::is_same_v<decltype(PGOAnalysisMap::PGOBBEntry::SuccessorEntry::ID),
                   decltype(BBAddrMap::BBEntry::ID)>,
    "PGOAnalysisMap should use the same type for basic block ID as BBAddrMap");

TEST(ELFTypesTest, BBAddrMapFeaturesEncodingTest) {
  const std::array<BBAddrMap::Features, 12> Decoded = {
      {{false, false, false, false, false, false, false},
       {true, false, false, false, false, false, false},
       {false, true, false, false, false, false, false},
       {false, false, true, false, false, false, false},
       {false, false, false, true, false, false, false},
       {true, true, false, false, false, false, false},
       {false, true, true, false, false, false, false},
       {false, true, true, true, false, false, false},
       {true, true, true, true, false, false, false},
       {false, false, false, false, true, false, false},
       {false, false, false, false, false, true, false},
       {false, false, false, false, false, false, true}}};
  const std::array<uint8_t, 12> Encoded = {
      {0b0000, 0b0001, 0b0010, 0b0100, 0b1000, 0b0011, 0b0110, 0b1110, 0b1111,
       0b1'0000, 0b10'0000, 0b100'0000}};
  for (const auto &[Feat, EncodedVal] : llvm::zip(Decoded, Encoded))
    EXPECT_EQ(Feat.encode(), EncodedVal);
  for (const auto &[Feat, EncodedVal] : llvm::zip(Decoded, Encoded)) {
    Expected<BBAddrMap::Features> FeatEnableOrError =
        BBAddrMap::Features::decode(EncodedVal);
    ASSERT_THAT_EXPECTED(FeatEnableOrError, Succeeded());
    EXPECT_EQ(*FeatEnableOrError, Feat);
  }
}

TEST(ELFTypesTest, BBAddrMapFeaturesInvalidEncodingTest) {
  const std::array<std::string, 2> Errors = {
      "invalid encoding for BBAddrMap::Features: 0x80",
      "invalid encoding for BBAddrMap::Features: 0xf0"};
  const std::array<uint8_t, 2> Values = {{0b1000'0000, 0b1111'0000}};
  for (const auto &[Val, Error] : llvm::zip(Values, Errors)) {
    EXPECT_THAT_ERROR(BBAddrMap::Features::decode(Val).takeError(),
                      FailedWithMessage(Error));
  }
}
