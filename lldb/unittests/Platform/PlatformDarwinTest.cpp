//===-- PlatformDarwinTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/Platform/MacOSX/PlatformDarwin.h"
#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteMacOSX.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Interpreter/ScriptInterpreter.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"

#include <memory>
#include <tuple>

using namespace lldb;
using namespace lldb_private;

namespace {
class MockScriptInterpreterPython : public ScriptInterpreter {
public:
  MockScriptInterpreterPython(Debugger &debugger)
      : ScriptInterpreter(debugger,
                          lldb::ScriptLanguage::eScriptLanguagePython) {}

  ~MockScriptInterpreterPython() override = default;

  bool ExecuteOneLine(llvm::StringRef command, CommandReturnObject *,
                      const ExecuteScriptOptions &) override {
    return false;
  }

  void ExecuteInterpreterLoop() override {}

  static void Initialize() {
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(),
                                  lldb::eScriptLanguagePython, CreateInstance);
  }

  static void Terminate() {}

  static lldb::ScriptInterpreterSP CreateInstance(Debugger &debugger) {
    return std::make_shared<MockScriptInterpreterPython>(debugger);
  }

  static llvm::StringRef GetPluginNameStatic() {
    return "MockScriptInterpreterPython";
  }

  static llvm::StringRef GetPluginDescriptionStatic() {
    return "MockScriptInterpreterPython";
  }

  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }
};

LLDB_PLUGIN_DEFINE(MockScriptInterpreterPython)
} // namespace

struct PlatformDarwinLocateTest : public testing::Test {
protected:
  void SetUp() override {
    std::call_once(TestUtilities::g_debugger_initialize_flag,
                   []() { Debugger::Initialize(nullptr); });

    ArchSpec arch("x86_64-apple-macosx-");
    m_platform_sp = PlatformRemoteMacOSX::CreateInstance(true, &arch);
    Platform::SetHostPlatform(m_platform_sp);

    m_debugger_sp = Debugger::CreateInstance();

    m_debugger_sp->GetTargetList().CreateTarget(*m_debugger_sp, "", arch,
                                                lldb_private::eLoadDependentsNo,
                                                m_platform_sp, m_target_sp);

    ASSERT_TRUE(m_target_sp);
    ASSERT_TRUE(m_platform_sp);

    ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory(
        "locate-scripts-from-dsym-test", m_tmp_root_dir))
        << "Failed to create test directory.";

    // Create <test-root>/.dSYM/Contents/Resources
    llvm::SmallString<128> dsym_resource_dir(m_tmp_root_dir);
    llvm::sys::path::append(dsym_resource_dir, ".dSYM", "Contents",
                            "Resources");
    ASSERT_FALSE(llvm::sys::fs::create_directories(dsym_resource_dir))
        << "Failed to create test dSYM root directory.";

    // Create <test-root>/.dSYM/Contents/Resources/DWARF
    m_tmp_dsym_dwarf_dir = dsym_resource_dir;
    llvm::sys::path::append(m_tmp_dsym_dwarf_dir, "DWARF");
    ASSERT_FALSE(llvm::sys::fs::create_directory(m_tmp_dsym_dwarf_dir))
        << "Failed to create test dSYM DWARF directory.";

    // Create <test-root>/.dSYM/Contents/Resources/Python
    m_tmp_dsym_python_dir = dsym_resource_dir;
    llvm::sys::path::append(m_tmp_dsym_python_dir, "Python");
    ASSERT_FALSE(llvm::sys::fs::create_directory(m_tmp_dsym_python_dir))
        << "Failed to create test dSYM Python directory.";
  };

  DebuggerSP m_debugger_sp;
  PlatformSP m_platform_sp;
  TargetSP m_target_sp;

  /// Root directory for m_tmp_dsym_dwarf_dir and m_tmp_dsym_python_dir
  llvm::SmallString<128> m_tmp_root_dir;

  /// <test-root>/.dSYM/Contents/Resources/DWARF
  llvm::SmallString<128> m_tmp_dsym_dwarf_dir;

  /// <test-root>/.dSYM/Contents/Resources/Python
  llvm::SmallString<128> m_tmp_dsym_python_dir;

  SubsystemRAII<FileSystem, HostInfo, PlatformMacOSX,
                MockScriptInterpreterPython>
      subsystems;
};

static std::string CreateFile(llvm::StringRef filename,
                              llvm::SmallString<128> parent_dir) {
  llvm::SmallString<128> path(parent_dir);
  llvm::sys::path::append(path, filename);
  int fd;
  std::error_code ret = llvm::sys::fs::openFileForWrite(path, fd);
  assert(!ret && "Failed to create test file.");
  ::close(fd);

  return path.c_str();
}

TEST(PlatformDarwinTest, TestParseVersionBuildDir) {
  llvm::VersionTuple V;
  llvm::StringRef D;

  std::tie(V, D) = PlatformDarwin::ParseVersionBuildDir("1.2.3 (test1)");
  EXPECT_EQ(llvm::VersionTuple(1, 2, 3), V);
  EXPECT_EQ("test1", D);

  std::tie(V, D) = PlatformDarwin::ParseVersionBuildDir("2.3 (test2)");
  EXPECT_EQ(llvm::VersionTuple(2, 3), V);
  EXPECT_EQ("test2", D);

  std::tie(V, D) = PlatformDarwin::ParseVersionBuildDir("3 (test3)");
  EXPECT_EQ(llvm::VersionTuple(3), V);
  EXPECT_EQ("test3", D);

  std::tie(V, D) = PlatformDarwin::ParseVersionBuildDir("1.2.3 (test");
  EXPECT_EQ(llvm::VersionTuple(1, 2, 3), V);
  EXPECT_EQ("test", D);

  std::tie(V, D) = PlatformDarwin::ParseVersionBuildDir("2.3.4 test");
  EXPECT_EQ(llvm::VersionTuple(2, 3, 4), V);
  EXPECT_EQ("", D);

  std::tie(V, D) = PlatformDarwin::ParseVersionBuildDir("3.4.5");
  EXPECT_EQ(llvm::VersionTuple(3, 4, 5), V);
}

TEST_F(PlatformDarwinLocateTest, LocateExecutableScriptingResourcesFromDSYM) {
  // Create dummy module file at <test-root>/TestModule.o
  FileSpec module_fspec(CreateFile("TestModule.o", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  // Create dummy module file at
  // <test-root>/.dSYM/Contents/Resources/DWARF/TestModule.o
  FileSpec dsym_module_fpec(CreateFile("TestModule.o", m_tmp_dsym_dwarf_dir));
  ASSERT_TRUE(dsym_module_fpec);

  CreateFile("TestModule.py", m_tmp_dsym_python_dir);
  CreateFile("TestModule.txt", m_tmp_dsym_python_dir);
  CreateFile("TestModule.sh", m_tmp_dsym_python_dir);

  StreamString ss;
  FileSpecList fspecs =
      std::static_pointer_cast<PlatformDarwin>(m_platform_sp)
          ->LocateExecutableScriptingResourcesFromDSYM(
              ss, module_fspec, *m_target_sp, dsym_module_fpec);
  EXPECT_EQ(fspecs.GetSize(), 1u);
  EXPECT_EQ(fspecs.GetFileSpecAtIndex(0).GetFilename(), "TestModule.py");
}
