//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/ObjectFile/PECOFF/ObjectFilePECOFF.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Testing/Support/Error.h"

#include <chrono>
#include <future>
#include <mutex>
#include <thread>

using namespace lldb_private;
using namespace lldb;

class SectionListTest : public testing::Test {
  SubsystemRAII<FileSystem, ObjectFilePECOFF> subsystems;
};

static void
BuildWhileHoldingModuleMutex(llvm::function_ref<SectionList *(Module &)> get) {
  llvm::Expected<TestFile> file = TestFile::fromYaml(R"(
--- !COFF
OptionalHeader:
  SectionAlignment: 4096
  FileAlignment:   512
header:
  Machine:         IMAGE_FILE_MACHINE_AMD64
  Characteristics: [ IMAGE_FILE_EXECUTABLE_IMAGE, IMAGE_FILE_LARGE_ADDRESS_AWARE ]
sections:
  - Name:            .data
    VirtualSize:     496
    SizeOfRawData:   512
    Characteristics: [ IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_MEM_READ ]
    SectionData:     11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111110000

symbols:         []
...
)");
  ASSERT_THAT_EXPECTED(file, llvm::Succeeded());

  for (int i = 0; i < 10; ++i) {
    auto module_sp = std::make_shared<Module>(file->moduleSpec());
    ASSERT_NE(module_sp->GetObjectFile(), nullptr);

    std::promise<void> locked;
    std::future<void> locked_future = locked.get_future();
    std::thread builder([&] {
      locked_future.wait();
      get(*module_sp);
    });
    {
      std::lock_guard<std::recursive_mutex> guard(module_sp->GetMutex());
      locked.set_value();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      EXPECT_NE(get(*module_sp), nullptr);
    }
    builder.join();
  }
}

TEST_F(SectionListTest, ModuleSectionListWhileHoldingModuleMutex) {
  BuildWhileHoldingModuleMutex(
      [](Module &module) { return module.GetSectionList(); });
}

TEST_F(SectionListTest, ObjectFileSectionListWhileHoldingModuleMutex) {
  BuildWhileHoldingModuleMutex(
      [](Module &module) { return module.GetObjectFile()->GetSectionList(); });
}
