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
  for (size_t i = 0; i < Decoded.size(); ++i) {
    EXPECT_EQ(Decoded[i].encode(), Encoded[i]);
    EXPECT_LT(Decoded[i].encode(),
              uint32_t{1} << BBAddrMap::BBEntry::Metadata::NumberOfBits)
        << "If a new bit was added, please update NumberOfBits.";
  }
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
static_assert(BBAddrMap::BBEntry::Metadata::NumberOfBits <
                  (sizeof(uint32_t) * 8) - 2,
              "currently PGOAnalysisMap relies on having two bits of space to "
              "encode number of successors, to add more we need increase the "
              "encoded size of Metadata");

TEST(ELFTypesTest, PGOBBEntryMetadataEncodingTest) {
  using ST = PGOAnalysisMap::PGOBBEntry::SuccessorsType;
  const std::array<std::pair<BBAddrMap::BBEntry::Metadata, ST>, 7> Decoded = {
      {{{false, false, false, false, false}, ST::None},
       {{true, false, false, false, false}, ST::One},
       {{false, true, false, false, false}, ST::Two},
       {{false, false, true, false, false}, ST::Multiple},
       {{false, false, false, true, false}, ST::One},
       {{false, false, false, false, true}, ST::Two},
       {{true, true, true, true, true}, ST::Multiple}}};
  const std::array<uint32_t, 7> Encoded = {{0b00'00000, 0b01'00001, 0b10'00010,
                                            0b11'00100, 0b01'01000, 0b10'10000,
                                            0b11'11111}};
  for (auto [Enc, Dec] : llvm::zip(Encoded, Decoded)) {
    auto [MD, SuccType] = Dec;
    EXPECT_EQ(PGOAnalysisMap::PGOBBEntry::encodeMD(MD, SuccType), Enc);
  }
  for (auto [Enc, Dec] : llvm::zip(Encoded, Decoded)) {
    Expected<std::pair<BBAddrMap::BBEntry::Metadata, ST>> MetadataOrError =
        PGOAnalysisMap::PGOBBEntry::decodeMD(Enc);
    ASSERT_THAT_EXPECTED(MetadataOrError, Succeeded());
    EXPECT_EQ(*MetadataOrError, Dec);
  }
}

TEST(ELFTypesTest, PGOBBEntryMetadataInvalidEncodingTest) {
  const std::array<std::string, 3> Errors = {
      "invalid encoding for BBEntry::Metadata: 0xff9f",
      "invalid encoding for BBEntry::Metadata: 0x100001",
      "invalid encoding for BBEntry::Metadata: 0x80"};
  const std::array<uint32_t, 3> Values = {0xFFFF, 0x100001, 0x00c0};
  for (auto [Val, Err] : llvm::zip(Values, Errors)) {
    EXPECT_THAT_ERROR(PGOAnalysisMap::PGOBBEntry::decodeMD(Val).takeError(),
                      FailedWithMessage(Err));
  }
}
