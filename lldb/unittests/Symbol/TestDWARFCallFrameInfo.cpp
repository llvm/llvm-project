//===-- TestDWARFCallFrameInfo.cpp ------------------------------*- C++ -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/Process/Utility/RegisterContext_x86.h"
#include "TestingSupport/TestUtilities.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/Section.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/DWARFCallFrameInfo.h"
#include "lldb/Utility/StreamString.h"

#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

using namespace lldb_private;
using namespace lldb;

class DWARFCallFrameInfoTest : public testing::Test {
public:
  void SetUp() override {
    FileSystem::Initialize();
    HostInfo::Initialize();
    ObjectFileELF::Initialize();
  }

  void TearDown() override {
    ObjectFileELF::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
  }

protected:
  void TestBasic(DWARFCallFrameInfo::Type type, llvm::StringRef symbol);
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

namespace lldb_private {
static std::ostream &operator<<(std::ostream &OS, const UnwindPlan::Row &row) {
  StreamString SS;
  row.Dump(SS, nullptr, nullptr, 0);
  return OS << SS.GetData();
}
} // namespace lldb_private

static UnwindPlan::Row GetExpectedRow0() {
  UnwindPlan::Row row;
  row.SetOffset(0);
  row.GetCFAValue().SetIsRegisterPlusOffset(dwarf_rsp_x86_64, 8);
  row.SetRegisterLocationToAtCFAPlusOffset(dwarf_rip_x86_64, -8, false);
  return row;
}

static UnwindPlan::Row GetExpectedRow1() {
  UnwindPlan::Row row;
  row.SetOffset(1);
  row.GetCFAValue().SetIsRegisterPlusOffset(dwarf_rsp_x86_64, 16);
  row.SetRegisterLocationToAtCFAPlusOffset(dwarf_rip_x86_64, -8, false);
  row.SetRegisterLocationToAtCFAPlusOffset(dwarf_rbp_x86_64, -16, false);
  return row;
}

static UnwindPlan::Row GetExpectedRow2() {
  UnwindPlan::Row row;
  row.SetOffset(4);
  row.GetCFAValue().SetIsRegisterPlusOffset(dwarf_rbp_x86_64, 16);
  row.SetRegisterLocationToAtCFAPlusOffset(dwarf_rip_x86_64, -8, false);
  row.SetRegisterLocationToAtCFAPlusOffset(dwarf_rbp_x86_64, -16, false);
  return row;
}

void DWARFCallFrameInfoTest::TestBasic(DWARFCallFrameInfo::Type type,
                                       llvm::StringRef symbol) {
  std::string yaml = GetInputFilePath("basic-call-frame-info.yaml");
  llvm::SmallString<128> obj;

  ASSERT_NO_ERROR(llvm::sys::fs::createTemporaryFile(
      "basic-call-frame-info-%%%%%%", "obj", obj));
  llvm::FileRemover obj_remover(obj);

  llvm::StringRef args[] = {YAML2OBJ, yaml};
  llvm::StringRef obj_ref = obj;
  const llvm::Optional<llvm::StringRef> redirects[] = {llvm::None, obj_ref,
                                                       llvm::None};
  ASSERT_EQ(0,
            llvm::sys::ExecuteAndWait(YAML2OBJ, args, llvm::None, redirects));

  uint64_t size;
  ASSERT_NO_ERROR(llvm::sys::fs::file_size(obj, size));
  ASSERT_GT(size, 0u);

  auto module_sp = std::make_shared<Module>(ModuleSpec(FileSpec(obj)));
  SectionList *list = module_sp->GetSectionList();
  ASSERT_NE(nullptr, list);

  auto section_sp = list->FindSectionByType(type == DWARFCallFrameInfo::EH
                                                ? eSectionTypeEHFrame
                                                : eSectionTypeDWARFDebugFrame,
                                            false);
  ASSERT_NE(nullptr, section_sp);

  DWARFCallFrameInfo cfi(*module_sp->GetObjectFile(), section_sp, type);

  const Symbol *sym = module_sp->FindFirstSymbolWithNameAndType(
      ConstString(symbol), eSymbolTypeAny);
  ASSERT_NE(nullptr, sym);

  UnwindPlan plan(eRegisterKindGeneric);
  ASSERT_TRUE(cfi.GetUnwindPlan(sym->GetAddress(), plan));
  ASSERT_EQ(3, plan.GetRowCount());
  EXPECT_EQ(GetExpectedRow0(), *plan.GetRowAtIndex(0));
  EXPECT_EQ(GetExpectedRow1(), *plan.GetRowAtIndex(1));
  EXPECT_EQ(GetExpectedRow2(), *plan.GetRowAtIndex(2));
}

TEST_F(DWARFCallFrameInfoTest, Basic_dwarf3) {
  TestBasic(DWARFCallFrameInfo::DWARF, "debug_frame3");
}

TEST_F(DWARFCallFrameInfoTest, Basic_dwarf4) {
  TestBasic(DWARFCallFrameInfo::DWARF, "debug_frame4");
}

TEST_F(DWARFCallFrameInfoTest, Basic_eh) {
  TestBasic(DWARFCallFrameInfo::EH, "eh_frame");
}
