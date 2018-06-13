//===-- TestObjectFileELF.cpp -----------------------------------*- C++ -*-===//
//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/SymbolVendor/ELF/SymbolVendorELF.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/Section.h"
#include "lldb/Host/HostInfo.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;

class ObjectFileELFTest : public testing::Test {
public:
  void SetUp() override {
    HostInfo::Initialize();
    ObjectFileELF::Initialize();
    SymbolVendorELF::Initialize();
  }

  void TearDown() override {
    SymbolVendorELF::Terminate();
    ObjectFileELF::Terminate();
    HostInfo::Terminate();
  }

protected:
};

#define ASSERT_NO_ERROR(x)                                                     \
  if (std::error_code ASSERT_NO_ERROR_ec = x) {                                \
    llvm::SmallString<128> MessageStorage;                                     \
    llvm::raw_svector_ostream Message(MessageStorage);                         \
    Message << #x ": did not return errc::success.\n"                          \
            << "error number: " << ASSERT_NO_ERROR_ec.value() << "\n"          \
            << "error message: " << ASSERT_NO_ERROR_ec.message() << "\n";      \
    GTEST_FATAL_FAILURE_(MessageStorage.c_str());                              \
  } else {                                                                     \
  }

TEST_F(ObjectFileELFTest, SectionsResolveConsistently) {
  std::string yaml = GetInputFilePath("sections-resolve-consistently.yaml");
  llvm::SmallString<128> obj;
  ASSERT_NO_ERROR(llvm::sys::fs::createTemporaryFile(
      "sections-resolve-consistently-%%%%%%", "obj", obj));

  llvm::FileRemover remover(obj);
  llvm::StringRef args[] = {YAML2OBJ, yaml};
  llvm::StringRef obj_ref = obj;
  const llvm::Optional<llvm::StringRef> redirects[] = {llvm::None, obj_ref,
                                                       llvm::None};
  ASSERT_EQ(0,
            llvm::sys::ExecuteAndWait(YAML2OBJ, args, llvm::None, redirects));
  uint64_t size;
  ASSERT_NO_ERROR(llvm::sys::fs::file_size(obj, size));
  ASSERT_GT(size, 0u);

  ModuleSpec spec{FileSpec(obj, false)};
  spec.GetSymbolFileSpec().SetFile(obj, false, FileSpec::Style::native);
  auto module_sp = std::make_shared<Module>(spec);
  SectionList *list = module_sp->GetSectionList();
  ASSERT_NE(nullptr, list);

  auto bss_sp = list->FindSectionByName(ConstString(".bss"));
  ASSERT_NE(nullptr, bss_sp);
  auto data_sp = list->FindSectionByName(ConstString(".data"));
  ASSERT_NE(nullptr, data_sp);
  auto text_sp = list->FindSectionByName(ConstString(".text"));
  ASSERT_NE(nullptr, text_sp);

  const Symbol *X = module_sp->FindFirstSymbolWithNameAndType(ConstString("X"),
                                                              eSymbolTypeAny);
  ASSERT_NE(nullptr, X);
  EXPECT_EQ(bss_sp, X->GetAddress().GetSection());

  const Symbol *Y = module_sp->FindFirstSymbolWithNameAndType(ConstString("Y"),
                                                              eSymbolTypeAny);
  ASSERT_NE(nullptr, Y);
  EXPECT_EQ(data_sp, Y->GetAddress().GetSection());

  const Symbol *start = module_sp->FindFirstSymbolWithNameAndType(
      ConstString("_start"), eSymbolTypeAny);
  ASSERT_NE(nullptr, start);
  EXPECT_EQ(text_sp, start->GetAddress().GetSection());
}

// Test that GetModuleSpecifications works on an "atypical" object file which
// has section headers right after the ELF header (instead of the more common
// layout where the section headers are at the very end of the object file).
//
// Test file generated with yaml2obj (@svn rev 324254) from the following input:
/*
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_EXEC
  Machine:         EM_X86_64
  Entry:           0x00000000004003D0
Sections:
  - Name:            .note.gnu.build-id
    Type:            SHT_NOTE
    Flags:           [ SHF_ALLOC ]
    Address:         0x0000000000400274
    AddressAlign:    0x0000000000000004
    Content:         040000001400000003000000474E55001B8A73AC238390E32A7FF4AC8EBE4D6A41ECF5C9
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x00000000004003D0
    AddressAlign:    0x0000000000000010
    Content:         DEADBEEFBAADF00D
...
*/
TEST_F(ObjectFileELFTest, GetModuleSpecifications_EarlySectionHeaders) {
  std::string SO = GetInputFilePath("early-section-headers.so");
  ModuleSpecList Specs;
  ASSERT_EQ(1u, ObjectFile::GetModuleSpecifications(FileSpec(SO, false), 0, 0, Specs));
  ModuleSpec Spec;
  ASSERT_TRUE(Specs.GetModuleSpecAtIndex(0, Spec)) ;
  UUID Uuid;
  Uuid.SetFromStringRef("1b8a73ac238390e32a7ff4ac8ebe4d6a41ecf5c9", 20);
  EXPECT_EQ(Spec.GetUUID(), Uuid);
}
