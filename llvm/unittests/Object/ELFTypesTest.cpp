//===----------------------- ELFTypesTest.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/Object/ELFTypes.h"
#include "gtest/gtest.h"
#include <iostream>

using namespace llvm;
using namespace llvm::object;

template <typename ELFT> using Elf_Note = typename ELFT::Note;

template <class ELFT> struct NoteTestData {
  std::vector<uint8_t> Data;

  const Elf_Note_Impl<ELFT> getElfNote(StringRef Name, uint32_t Type,
                                       ArrayRef<uint8_t> Desc) {
    Data.resize(sizeof(Elf_Nhdr_Impl<ELFT>) +
                    alignTo<Elf_Nhdr_Impl<ELFT>::Align>(Name.size()) +
                    alignTo<Elf_Nhdr_Impl<ELFT>::Align>(Desc.size()),
                0);

    Elf_Nhdr_Impl<ELFT> *Nhdr =
        reinterpret_cast<Elf_Nhdr_Impl<ELFT> *>(Data.data());
    Nhdr->n_namesz = (Name == "") ? 0 : Name.size() + 1;
    Nhdr->n_descsz = Desc.size();
    Nhdr->n_type = Type;

    auto NameOffset = Data.begin() + sizeof(*Nhdr);
    std::copy(Name.begin(), Name.end(), NameOffset);

    auto DescOffset =
        NameOffset + alignTo<Elf_Nhdr_Impl<ELFT>::Align>(Nhdr->n_namesz);
    std::copy(Desc.begin(), Desc.end(), DescOffset);

    return Elf_Note_Impl<ELFT>(*Nhdr);
  }
};

TEST(ELFTypesTest, NoteTest) {
  static const uint8_t Random[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  ArrayRef<uint8_t> RandomData = makeArrayRef(Random);
  NoteTestData<ELF64LE> TestData;

  auto Note1 = TestData.getElfNote(StringRef("AMD"), ELF::NT_AMDGPU_METADATA,
                                   RandomData);
  EXPECT_EQ(Note1.getName(), "AMD");
  EXPECT_EQ(Note1.getType(), ELF::NT_AMDGPU_METADATA);
  EXPECT_EQ(Note1.getDesc(), RandomData);
  EXPECT_EQ(Note1.getDescAsStringRef(),
            StringRef(reinterpret_cast<const char *>(Random), sizeof(Random)));

  auto Note2 = TestData.getElfNote("", ELF::NT_AMDGPU_METADATA, RandomData);
  EXPECT_EQ(Note2.getName(), "");

  auto Note3 =
      TestData.getElfNote("AMD", ELF::NT_AMDGPU_METADATA, ArrayRef<uint8_t>());
  EXPECT_EQ(Note3.getDescAsStringRef(), StringRef(""));
}
