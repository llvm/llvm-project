//===- BuildIDTest.cpp - Tests for getBuildID ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/BuildID.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

template <class ELFT>
static Expected<ELFObjectFile<ELFT>> toBinary(SmallVectorImpl<char> &Storage,
                                              StringRef Yaml) {
  raw_svector_ostream OS(Storage);
  yaml::Input YIn(Yaml);
  if (!yaml::convertYAML(YIn, OS, [](const Twine &Msg) {}))
    return createStringError(std::errc::invalid_argument,
                             "unable to convert YAML");
  return ELFObjectFile<ELFT>::create(MemoryBufferRef(OS.str(), "dummyELF"));
}

TEST(BuildIDTest, InvalidNoteFileSizeTest) {
  SmallString<0> Storage;
  Expected<ELFObjectFile<ELF64LE>> ElfOrErr = toBinary<ELF64LE>(Storage, R"(
--- !ELF
FileHeader:
  Class:          ELFCLASS64
  Data:           ELFDATA2LSB
  Type:           ET_EXEC
  Machine:        EM_X86_64
Sections:
  - Name:         .note.gnu.build-id
    Type:         SHT_NOTE
    AddressAlign: 0x04
    Notes:
      - Name:     "GNU"
        Desc:     "abb50d82b6bdc861"
        Type:     3
ProgramHeaders:
  - Type:         PT_NOTE
    FileSize:     0xffffffffffffffff
    Offset:       0x100
)");
  ASSERT_THAT_EXPECTED(ElfOrErr, Succeeded());
  BuildIDRef BuildID = getBuildID(&ElfOrErr.get());
  EXPECT_EQ(
      StringRef(reinterpret_cast<const char *>(BuildID.data()), BuildID.size()),
      "\xAB\xB5\x0D\x82\xB6\xBD\xC8\x61");
}