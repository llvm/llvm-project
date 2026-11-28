//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/SectionLoadList.h"
#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <optional>

using namespace lldb;
using namespace lldb_private;

namespace {

class SectionLoadListTest : public testing::Test {
  SubsystemRAII<FileSystem, HostInfo, ObjectFileELF> subsystems;

protected:
  void SetUp() override;

  /// Resolve a load address in a section load list.
  ///
  /// \param[in] list
  ///     The section load list to look the address up in.
  ///
  /// \param[in] load_addr
  ///     The load address to resolve.
  ///
  /// \return
  ///     The section that \a load_addr resolves to, or nullptr if it resolves
  ///     to none.
  static const Section *ResolvedSection(const SectionLoadList &list,
                                        addr_t load_addr);

  // The module reads its object file from this file's bytes without copying
  // them, so the file must outlive the module.
  std::optional<TestFile> m_file;
  ModuleSP m_module_sp;
  SectionSP m_text_sp;
  SectionSP m_data_sp;
};

} // namespace

void SectionLoadListTest::SetUp() {
  // Two sections of the same size, so that either one covers the same range
  // when it is loaded at an address the other one had.
  llvm::Expected<TestFile> file = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_X86_64
Sections:
  - Name:    .text
    Type:    SHT_PROGBITS
    Flags:   [ SHF_ALLOC, SHF_EXECINSTR ]
    Address: 0x1000
    Size:    0x100
  - Name:    .data
    Type:    SHT_PROGBITS
    Flags:   [ SHF_ALLOC, SHF_WRITE ]
    Address: 0x2000
    Size:    0x100
...
)");
  ASSERT_THAT_EXPECTED(file, llvm::Succeeded());
  m_file.emplace(std::move(*file));
  m_module_sp = std::make_shared<Module>(m_file->moduleSpec());
  SectionList *sections = m_module_sp->GetSectionList();
  ASSERT_NE(sections, nullptr);
  m_text_sp = sections->FindSectionByName(".text");
  ASSERT_NE(m_text_sp, nullptr);
  m_data_sp = sections->FindSectionByName(".data");
  ASSERT_NE(m_data_sp, nullptr);
}

const Section *SectionLoadListTest::ResolvedSection(const SectionLoadList &list,
                                                    addr_t load_addr) {
  Address addr;
  if (!list.ResolveLoadAddress(load_addr, addr))
    return nullptr;
  return addr.GetSection().get();
}

TEST_F(SectionLoadListTest, MoveSection) {
  SectionLoadList list;
  EXPECT_TRUE(list.SetSectionLoadAddress(m_text_sp, 0x10000));
  EXPECT_EQ(ResolvedSection(list, 0x10010), m_text_sp.get());
  // Loading a section where it already is changes nothing.
  EXPECT_FALSE(list.SetSectionLoadAddress(m_text_sp, 0x10000));

  EXPECT_TRUE(list.SetSectionLoadAddress(m_text_sp, 0x20000));
  EXPECT_EQ(list.GetSectionLoadAddress(m_text_sp), 0x20000u);
  EXPECT_EQ(ResolvedSection(list, 0x20010), m_text_sp.get());
  // The address the section left no longer resolves to it.
  EXPECT_EQ(ResolvedSection(list, 0x10010), nullptr);

  // The same holds for a move to a lower address.
  EXPECT_TRUE(list.SetSectionLoadAddress(m_text_sp, 0x10000));
  EXPECT_EQ(list.GetSectionLoadAddress(m_text_sp), 0x10000u);
  EXPECT_EQ(ResolvedSection(list, 0x10010), m_text_sp.get());
  EXPECT_EQ(ResolvedSection(list, 0x20010), nullptr);
}

TEST_F(SectionLoadListTest, MoveSectionOntoTakenAddress) {
  SectionLoadList list;
  ASSERT_TRUE(list.SetSectionLoadAddress(m_text_sp, 0x10000));
  ASSERT_TRUE(list.SetSectionLoadAddress(m_data_sp, 0x20000));

  // A section that moves to another section's address takes it over.
  EXPECT_TRUE(list.SetSectionLoadAddress(m_text_sp, 0x20000));
  EXPECT_EQ(ResolvedSection(list, 0x20010), m_text_sp.get());
  // The address the section left no longer resolves to it.
  EXPECT_EQ(ResolvedSection(list, 0x10010), nullptr);
}

TEST_F(SectionLoadListTest, SectionTakesOverAddress) {
  SectionLoadList list;
  ASSERT_TRUE(list.SetSectionLoadAddress(m_text_sp, 0x10000));
  // An address resolves to the section loaded there last.
  EXPECT_TRUE(list.SetSectionLoadAddress(m_data_sp, 0x10000));
  EXPECT_EQ(ResolvedSection(list, 0x10010), m_data_sp.get());

  // When the section that lost the address moves, the address stays with the
  // section that took it over.
  EXPECT_TRUE(list.SetSectionLoadAddress(m_text_sp, 0x20000));
  EXPECT_EQ(ResolvedSection(list, 0x20010), m_text_sp.get());
  EXPECT_EQ(ResolvedSection(list, 0x10010), m_data_sp.get());
  EXPECT_EQ(list.GetSectionLoadAddress(m_data_sp), 0x10000u);
}

TEST_F(SectionLoadListTest, UnloadSection) {
  SectionLoadList list;
  ASSERT_TRUE(list.SetSectionLoadAddress(m_text_sp, 0x10000));
  ASSERT_TRUE(list.SetSectionLoadAddress(m_data_sp, 0x20000));

  EXPECT_EQ(list.SetSectionUnloaded(m_text_sp), 1u);
  EXPECT_EQ(list.GetSectionLoadAddress(m_text_sp), LLDB_INVALID_ADDRESS);
  EXPECT_EQ(ResolvedSection(list, 0x10010), nullptr);
  EXPECT_EQ(ResolvedSection(list, 0x20010), m_data_sp.get());

  EXPECT_TRUE(list.SetSectionUnloaded(m_data_sp, 0x20000));
  EXPECT_EQ(list.GetSectionLoadAddress(m_data_sp), LLDB_INVALID_ADDRESS);
  EXPECT_EQ(ResolvedSection(list, 0x20010), nullptr);
  EXPECT_TRUE(list.IsEmpty());

  // An unloaded section can be loaded again, at another address.
  EXPECT_TRUE(list.SetSectionLoadAddress(m_text_sp, 0x30000));
  EXPECT_EQ(ResolvedSection(list, 0x30010), m_text_sp.get());
  EXPECT_EQ(ResolvedSection(list, 0x10010), nullptr);
}
