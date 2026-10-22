//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "Plugins/ScriptInterpreter/None/ScriptInterpreterNone.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Target.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include <optional>

using namespace lldb_private;
using namespace lldb;

namespace {

// An ELF shared library. A PIE executable is one too.
const char *k_shared_library_yaml = R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_DYN
  Machine:         EM_X86_64
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x1000
    AddressAlign:    0x10
    Content:         C3
...
)";

const char *k_executable_yaml = R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_EXEC
  Machine:         EM_X86_64
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x1000
    AddressAlign:    0x10
    Content:         C3
...
)";

class ExecutableModuleTest : public testing::Test {
public:
  void SetUp() override {
    llvm::Expected<TestFile> shared_library =
        TestFile::fromYaml(k_shared_library_yaml);
    ASSERT_THAT_EXPECTED(shared_library, llvm::Succeeded());
    m_shared_library.emplace(std::move(*shared_library));
    llvm::Expected<TestFile> executable = TestFile::fromYaml(k_executable_yaml);
    ASSERT_THAT_EXPECTED(executable, llvm::Succeeded());
    m_executable.emplace(std::move(*executable));

    ArchSpec arch("x86_64-pc-linux");
    std::call_once(TestUtilities::g_debugger_initialize_flag,
                   []() { Debugger::Initialize(nullptr); });
    Platform::SetHostPlatform(
        platform_linux::PlatformLinux::CreateInstance(true, &arch));
    m_debugger_sp = Debugger::CreateInstance();
    ASSERT_TRUE(m_debugger_sp);
    PlatformSP platform_sp;
    m_debugger_sp->GetTargetList().CreateTarget(
        *m_debugger_sp, "", arch, eLoadDependentsNo, platform_sp, m_target_sp);
    ASSERT_TRUE(m_target_sp);
  }

  void TearDown() override {
    m_target_sp.reset();
    if (m_debugger_sp)
      Debugger::Destroy(m_debugger_sp);
  }

  ModuleSP CreateSharedLibrary() {
    return std::make_shared<Module>(m_shared_library->moduleSpec());
  }

  ModuleSP CreateExecutable() {
    return std::make_shared<Module>(m_executable->moduleSpec());
  }

  SubsystemRAII<FileSystem, HostInfo, ObjectFileELF,
                platform_linux::PlatformLinux, ScriptInterpreterNone>
      subsystems;
  // These back the target's modules and must outlive them.
  std::optional<TestFile> m_shared_library;
  std::optional<TestFile> m_executable;
  DebuggerSP m_debugger_sp;
  TargetSP m_target_sp;
};

} // namespace

TEST_F(ExecutableModuleTest, NoExecutable) {
  m_target_sp->GetImages().Append(CreateSharedLibrary(), /*notify=*/false);
  EXPECT_EQ(m_target_sp->GetExecutableModule(), nullptr);
}

TEST_F(ExecutableModuleTest, SharedLibraryExecutable) {
  ModuleSP executable_sp = CreateSharedLibrary();
  m_target_sp->RebuildModuleListWithExecutable(executable_sp,
                                               eLoadDependentsNo);
  m_target_sp->GetImages().Append(CreateSharedLibrary(), /*notify=*/false);
  EXPECT_EQ(m_target_sp->GetExecutableModule(), executable_sp);
}

TEST_F(ExecutableModuleTest, MarkKeepsImages) {
  ModuleList &images = m_target_sp->GetImages();
  images.Append(CreateSharedLibrary(), /*notify=*/false);
  ModuleSP executable_sp = CreateSharedLibrary();
  images.Append(executable_sp, /*notify=*/false);

  m_target_sp->MarkExecutableModule(executable_sp);
  EXPECT_EQ(m_target_sp->GetExecutableModule(), executable_sp);
  EXPECT_EQ(images.GetSize(), 2u);
}

TEST_F(ExecutableModuleTest, RemovedExecutable) {
  ModuleSP executable_sp = CreateSharedLibrary();
  m_target_sp->RebuildModuleListWithExecutable(executable_sp,
                                               eLoadDependentsNo);
  ModuleList &images = m_target_sp->GetImages();
  images.Append(CreateSharedLibrary(), /*notify=*/false);

  images.Remove(executable_sp);
  EXPECT_EQ(m_target_sp->GetExecutableModule(), nullptr);
}

TEST_F(ExecutableModuleTest, ReplacedExecutable) {
  ModuleSP executable_sp = CreateSharedLibrary();
  m_target_sp->RebuildModuleListWithExecutable(executable_sp,
                                               eLoadDependentsNo);
  m_target_sp->GetImages().Append(CreateSharedLibrary(), /*notify=*/false);

  // A replacement only becomes the executable once the target is rebuilt
  // around it, which callers decide by comparing it against the executable.
  ModuleSP replacement_sp = CreateSharedLibrary();
  m_target_sp->GetImages().ReplaceModule(executable_sp, replacement_sp);
  EXPECT_EQ(m_target_sp->GetExecutableModule(), nullptr);
}

TEST_F(ExecutableModuleTest, ExecutableTypeFirst) {
  ModuleSP library_sp = CreateSharedLibrary();
  m_target_sp->RebuildModuleListWithExecutable(library_sp, eLoadDependentsNo);
  ModuleSP executable_sp = CreateExecutable();
  m_target_sp->GetImages().Append(executable_sp, /*notify=*/false);
  EXPECT_EQ(m_target_sp->GetExecutableModule(), executable_sp);
}
