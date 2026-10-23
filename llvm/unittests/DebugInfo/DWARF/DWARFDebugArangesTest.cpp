//===- DWARFDebugArangesTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDebugAranges.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(DWARFDebugAranges, DataBeforeText) {
  // The aranges index also serves data lookups. Allocated data can precede
  // the first code section, so its address ranges must remain in the index.
  StringRef Yaml = R"(
!ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_X86_64
Sections:
  - Name:    .data
    Type:    SHT_PROGBITS
    Flags:   [ SHF_ALLOC, SHF_WRITE ]
    Address: 0x800
    Size:    0x10
  - Name:    .text
    Type:    SHT_PROGBITS
    Flags:   [ SHF_ALLOC, SHF_EXECINSTR ]
    Address: 0x1000
    Size:    0x10
DWARF:
  debug_aranges:
    - Version:  2
      CuOffset: 0
      Descriptors:
        - Address: 0x800
          Length:  0x10
    - Version:  2
      CuOffset: 1
      Descriptors:
        - Address: 0x1000
          Length:  0x10
)";
  SmallString<0> Storage;
  std::unique_ptr<object::ObjectFile> Obj = yaml::yaml2ObjectFile(
      Storage, Yaml, [](const Twine &Err) { errs() << Err; });
  ASSERT_TRUE(Obj);
  std::unique_ptr<DWARFContext> Ctx = DWARFContext::create(*Obj);
  const DWARFDebugAranges *Aranges = Ctx->getDebugAranges();
  EXPECT_EQ(Aranges->findAddress(0x804), 0u);
  EXPECT_EQ(Aranges->findAddress(0x1004), 1u);
}
