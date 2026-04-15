//===-- PlatformDarwinTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "TestUtils.h"

#include "Plugins/Platform/MacOSX/PlatformDarwin.h"
#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteMacOSX.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/HostInfo.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"

#include <memory>
#include <tuple>

using namespace lldb;
using namespace lldb_private;

struct PlatformDarwinLocateTest : public testing::Test {
protected:
  void SetUp() override {
#ifdef _WIN32
    GTEST_SKIP() << "PlatformDarwin tests are not supported on Windows";
#endif
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

  void TearDown() override {
    ASSERT_FALSE(llvm::sys::fs::remove_directories(m_tmp_root_dir));
  }

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

TEST_F(PlatformDarwinLocateTest,
       LocateExecutableScriptingResourcesFromDSYM_StripExtensions) {
  // Tests that LocateExecutableScriptingResourcesFromDSYM will strip module
  // names until the basename matches the Python script.

  // Create dummy module file at <test-root>/TestModule.1.o.ext
  FileSpec module_fspec(CreateFile("TestModule.o.1.ext", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  // Create dummy module file at
  // <test-root>/.dSYM/Contents/Resources/DWARF/TestModule.o.1.ext
  FileSpec dsym_module_fpec(
      CreateFile("TestModule.o.1.ext", m_tmp_dsym_dwarf_dir));
  ASSERT_TRUE(dsym_module_fpec);

  CreateFile("TestModule.py", m_tmp_dsym_python_dir);
  CreateFile("TestModule.txt", m_tmp_dsym_python_dir);
  CreateFile("TestModule.sh", m_tmp_dsym_python_dir);

  StreamString ss;
  auto fspecs = std::static_pointer_cast<PlatformDarwin>(m_platform_sp)
                    ->LocateExecutableScriptingResourcesFromDSYM(
                        ss, module_fspec, *m_target_sp, dsym_module_fpec);
  EXPECT_EQ(fspecs.size(), 1u);
  EXPECT_EQ(fspecs.begin()->getFirst().GetFilename(), "TestModule.py");
}

TEST_F(PlatformDarwinLocateTest,
       LocateExecutableScriptingResourcesFromDSYM_ModuleNoExtension) {
  // Tests case where module has no file extension.

  // Create dummy module file at <test-root>/TestModule
  FileSpec module_fspec(CreateFile("TestModule", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  // Create dummy module file at
  // <test-root>/.dSYM/Contents/Resources/DWARF/TestModule
  FileSpec dsym_module_fpec(CreateFile("TestModule", m_tmp_dsym_dwarf_dir));
  ASSERT_TRUE(dsym_module_fpec);

  CreateFile("TestModule.py", m_tmp_dsym_python_dir);
  CreateFile("TestModule.txt", m_tmp_dsym_python_dir);
  CreateFile("TestModule.sh", m_tmp_dsym_python_dir);

  StreamString ss;
  auto fspecs = std::static_pointer_cast<PlatformDarwin>(m_platform_sp)
                    ->LocateExecutableScriptingResourcesFromDSYM(
                        ss, module_fspec, *m_target_sp, dsym_module_fpec);
  EXPECT_EQ(fspecs.size(), 1u);
  EXPECT_EQ(fspecs.begin()->getFirst().GetFilename(), "TestModule.py");
}

TEST_F(PlatformDarwinLocateTest,
       LocateExecutableScriptingResourcesFromDSYM_StripExtension_NoMatch) {
  // Tests case where stripping the module's file extensions still doesn't
  // result in a match.

  // Create dummy module file at <test-root>/TestModule.dylib
  FileSpec module_fspec(CreateFile("TestModule.dylib", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  // Create dummy module file at
  // <test-root>/.dSYM/Contents/Resources/DWARF/TestModule
  FileSpec dsym_module_fpec(
      CreateFile("TestModule.dylib", m_tmp_dsym_dwarf_dir));
  ASSERT_TRUE(dsym_module_fpec);

  CreateFile("TestModule.1.py", m_tmp_dsym_python_dir);

  StreamString ss;
  auto fspecs = std::static_pointer_cast<PlatformDarwin>(m_platform_sp)
                    ->LocateExecutableScriptingResourcesFromDSYM(
                        ss, module_fspec, *m_target_sp, dsym_module_fpec);
  EXPECT_EQ(fspecs.size(), 0u);
}

TEST_F(PlatformDarwinLocateTest,
       LocateExecutableScriptingResourcesFromDSYM_NestedDir) {
  // Tests case where a nested directory within the dSYM Python directory
  // contains Python scripts. LLDB shouldn't pick those.

  // Create dummy module file at <test-root>/TestModule.o
  FileSpec module_fspec(CreateFile("TestModule.o", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  // Create dummy module file at
  // <test-root>/.dSYM/Contents/Resources/DWARF/TestModule.o
  FileSpec dsym_module_fpec(CreateFile("TestModule.o", m_tmp_dsym_dwarf_dir));
  ASSERT_TRUE(dsym_module_fpec);

  // Create nested directory at
  // <test-root>/.dSYM/Contents/Resources/Python/nested_dir
  llvm::SmallString<128> nested_dir(m_tmp_dsym_python_dir);
  llvm::sys::path::append(nested_dir, "nested_dir");
  ASSERT_FALSE(llvm::sys::fs::create_directory(nested_dir))
      << "Failed to create test nested directory in dSYM Python directory.";

  CreateFile("TestModule.py", nested_dir);

  StreamString ss;
  auto fspecs = std::static_pointer_cast<PlatformDarwin>(m_platform_sp)
                    ->LocateExecutableScriptingResourcesFromDSYM(
                        ss, module_fspec, *m_target_sp, dsym_module_fpec);
  EXPECT_EQ(fspecs.size(), 0u);
}

TEST_F(
    PlatformDarwinLocateTest,
    LocateExecutableScriptingResourcesFromDSYM_KeywordInModuleNameIsKeyword) {
  // Tests case where the module name has a Python reserved keyword in its name.
  // That should resolve fine.

  // Create dummy module file at <test-root>/TestModule_import.o
  FileSpec module_fspec(CreateFile("TestModule_import.o", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  // Create dummy module file at
  // <test-root>/.dSYM/Contents/Resources/DWARF/TestModule_import.o
  FileSpec dsym_module_fpec(
      CreateFile("TestModule_import.o", m_tmp_dsym_dwarf_dir));
  ASSERT_TRUE(dsym_module_fpec);

  // Keywords are not permitted in module names.
  // See MockScriptInterpreterPython::IsReservedWord
  CreateFile("TestModule_import.py", m_tmp_dsym_python_dir);

  StreamString ss;
  auto fspecs = std::static_pointer_cast<PlatformDarwin>(m_platform_sp)
                    ->LocateExecutableScriptingResourcesFromDSYM(
                        ss, module_fspec, *m_target_sp, dsym_module_fpec);
  EXPECT_EQ(fspecs.size(), 1u);
  EXPECT_EQ(fspecs.begin()->getFirst().GetFilename(), "TestModule_import.py");
  EXPECT_TRUE(ss.Empty());
}

TEST_F(PlatformDarwinLocateTest,
       LocateExecutableScriptingResourcesFromDSYM_ModuleNameIsKeyword_NoMatch) {
  // Tests case where the module name is a Python reserved keyword. That isn't
  // supported.

  // Create dummy module file at <test-root>/import
  FileSpec module_fspec(CreateFile("import", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  // Create dummy module file at
  // <test-root>/.dSYM/Contents/Resources/DWARF/import
  FileSpec dsym_module_fpec(CreateFile("import", m_tmp_dsym_dwarf_dir));
  ASSERT_TRUE(dsym_module_fpec);

  // Keywords are not permitted in module names.
  // See MockScriptInterpreterPython::IsReservedWord
  FileSpec script_fspec(CreateFile("import.py", m_tmp_dsym_python_dir));
  ASSERT_TRUE(script_fspec);

  StreamString ss;
  auto fspecs = std::static_pointer_cast<PlatformDarwin>(m_platform_sp)
                    ->LocateExecutableScriptingResourcesFromDSYM(
                        ss, module_fspec, *m_target_sp, dsym_module_fpec);
  EXPECT_EQ(fspecs.size(), 0u);

  std::string expected = llvm::formatv(
      "debug script '{0}' cannot be loaded because 'import.py' "
      "conflicts with the keyword 'import'. If you intend to have this script "
      "loaded, please rename it to '_import.py' and retry.\n",
      script_fspec.GetPath());
  EXPECT_EQ(ss.GetString(), expected);
}

TEST_F(PlatformDarwinLocateTest,
       LocateExecutableScriptingResourcesFromDSYM_ModuleNameIsKeyword_Match) {
  // Tests case where the module name is a Python reserved keyword but we are
  // able to match a script.

  // Create dummy module file at <test-root>/import
  FileSpec module_fspec(CreateFile("import", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  // Create dummy module file at
  // <test-root>/.dSYM/Contents/Resources/DWARF/import
  FileSpec dsym_module_fpec(CreateFile("import", m_tmp_dsym_dwarf_dir));
  ASSERT_TRUE(dsym_module_fpec);

  // Keywords are not permitted in module names.
  // See MockScriptInterpreterPython::IsReservedWord
  CreateFile("_import.py", m_tmp_dsym_python_dir);

  FileSpec orig_fspec(CreateFile("import.py", m_tmp_dsym_python_dir));
  ASSERT_TRUE(orig_fspec);

  StreamString ss;
  auto fspecs = std::static_pointer_cast<PlatformDarwin>(m_platform_sp)
                    ->LocateExecutableScriptingResourcesFromDSYM(
                        ss, module_fspec, *m_target_sp, dsym_module_fpec);
  EXPECT_EQ(fspecs.size(), 1u);
  EXPECT_EQ(fspecs.begin()->getFirst().GetFilename(), "_import.py");

  std::string expected = llvm::formatv(
      "debug script '{0}' cannot be loaded because 'import.py' "
      "conflicts with the keyword 'import'. Ignoring 'import.py' and loading "
      "'_import.py' instead.\n",
      orig_fspec.GetPath());
  EXPECT_EQ(ss.GetString(), expected);
}

TEST_F(
    PlatformDarwinLocateTest,
    LocateExecutableScriptingResourcesFromDSYM_ModuleNameIsKeyword_Match_NoWarning) {
  // Tests case where the module name is a Python reserved keyword but we are
  // able to match a script but don't print a warning.

  // Create dummy module file at <test-root>/import
  FileSpec module_fspec(CreateFile("import", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  // Create dummy module file at
  // <test-root>/.dSYM/Contents/Resources/DWARF/import
  FileSpec dsym_module_fpec(CreateFile("import", m_tmp_dsym_dwarf_dir));
  ASSERT_TRUE(dsym_module_fpec);

  // Keywords are not permitted in module names.
  // See MockScriptInterpreterPython::IsReservedWord
  CreateFile("_import.py", m_tmp_dsym_python_dir);

  StreamString ss;
  auto fspecs = std::static_pointer_cast<PlatformDarwin>(m_platform_sp)
                    ->LocateExecutableScriptingResourcesFromDSYM(
                        ss, module_fspec, *m_target_sp, dsym_module_fpec);
  EXPECT_EQ(fspecs.size(), 1u);
  EXPECT_EQ(fspecs.begin()->getFirst().GetFilename(), "_import.py");
  EXPECT_TRUE(ss.GetString().empty());
}

TEST_F(
    PlatformDarwinLocateTest,
    LocateExecutableScriptingResourcesFromDSYM_SpecialCharactersInModuleName_NoMatch) {
  // Tests case where the module name contains "special characters" that the
  // Python ScriptInterpreter can't handle when importing modules.

  // Create dummy module file at <test-root>/TestModule-1.1 1.o
  FileSpec module_fspec(CreateFile("TestModule-1.1 1.o", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  // Create dummy module file at
  // <test-root>/.dSYM/Contents/Resources/DWARF/TestModule-1.1 1.o
  FileSpec dsym_module_fpec(
      CreateFile("TestModule-1.1 1.o", m_tmp_dsym_dwarf_dir));
  ASSERT_TRUE(dsym_module_fpec);

  FileSpec script_fspec(
      CreateFile("TestModule-1.1 1.py", m_tmp_dsym_python_dir));
  ASSERT_TRUE(script_fspec);

  StreamString ss;
  auto fspecs = std::static_pointer_cast<PlatformDarwin>(m_platform_sp)
                    ->LocateExecutableScriptingResourcesFromDSYM(
                        ss, module_fspec, *m_target_sp, dsym_module_fpec);
  EXPECT_EQ(fspecs.size(), 0u);

  std::string expected = llvm::formatv(
      "debug script '{0}' cannot be loaded because 'TestModule-1.1 1.py' "
      "contains reserved characters. If you intend to have this script "
      "loaded, please rename it to 'TestModule_1_1_1.py' and retry.\n",
      script_fspec.GetPath());
  EXPECT_EQ(ss.GetString(), expected);
}

TEST_F(
    PlatformDarwinLocateTest,
    LocateExecutableScriptingResourcesFromDSYM_SpecialCharactersInModuleName_Match_Warning) {
  // Tests case where the module name contains "special characters" that the
  // Python ScriptInterpreter can't handle when importing modules. LLDB can
  // still match a script but it warns when two scripts conflict in naming.

  // Create dummy module file at <test-root>/TestModule-1.1 1.o
  FileSpec module_fspec(CreateFile("TestModule-1.1 1.o", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  // Create dummy module file at
  // <test-root>/.dSYM/Contents/Resources/DWARF/TestModule-1.1 1.o
  FileSpec dsym_module_fpec(
      CreateFile("TestModule-1.1 1.o", m_tmp_dsym_dwarf_dir));
  ASSERT_TRUE(dsym_module_fpec);

  FileSpec orig_fspec(CreateFile("TestModule-1.1 1.py", m_tmp_dsym_python_dir));
  ASSERT_TRUE(orig_fspec);

  CreateFile("TestModule_1_1_1.py", m_tmp_dsym_python_dir);

  StreamString ss;
  auto fspecs = std::static_pointer_cast<PlatformDarwin>(m_platform_sp)
                    ->LocateExecutableScriptingResourcesFromDSYM(
                        ss, module_fspec, *m_target_sp, dsym_module_fpec);
  EXPECT_EQ(fspecs.size(), 1u);
  EXPECT_EQ(fspecs.begin()->getFirst().GetFilename(), "TestModule_1_1_1.py");

  std::string expected = llvm::formatv(
      "debug script '{0}' cannot be loaded because"
      " 'TestModule-1.1 1.py' contains reserved characters. Ignoring"
      " 'TestModule-1.1 1.py' and loading 'TestModule_1_1_1.py' instead.\n",
      orig_fspec.GetPath());
  EXPECT_EQ(ss.GetString(), expected);
}

TEST_F(
    PlatformDarwinLocateTest,
    LocateExecutableScriptingResourcesFromDSYM_SpecialCharactersInModuleName_Match_NoWarning) {
  // Tests case where the module name contains "special characters" that the
  // Python ScriptInterpreter can't handle when importing modules. We can still
  // match appropriately named scripts.

  // Create dummy module file at <test-root>/TestModule-1.1 1.o
  FileSpec module_fspec(CreateFile("TestModule-1.1 1.o", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  // Create dummy module file at
  // <test-root>/.dSYM/Contents/Resources/DWARF/TestModule-1.1 1.o
  FileSpec dsym_module_fpec(
      CreateFile("TestModule-1.1 1.o", m_tmp_dsym_dwarf_dir));
  ASSERT_TRUE(dsym_module_fpec);

  CreateFile("TestModule_1_1_1.py", m_tmp_dsym_python_dir);

  StreamString ss;
  auto fspecs = std::static_pointer_cast<PlatformDarwin>(m_platform_sp)
                    ->LocateExecutableScriptingResourcesFromDSYM(
                        ss, module_fspec, *m_target_sp, dsym_module_fpec);
  EXPECT_EQ(fspecs.size(), 1u);
  EXPECT_EQ(fspecs.begin()->getFirst().GetFilename(), "TestModule_1_1_1.py");
  EXPECT_TRUE(ss.GetString().empty());
}

TEST_F(
    PlatformDarwinLocateTest,
    LocateExecutableScriptingResourcesFromDSYM_ModuleNameIsKeywordAfterReplacement) {
  // Test case where the module name contains "special characters" but after
  // LLDB replaces those the filename is still a keyword. We ensure this by a
  // special case in MockScriptInterpreterPython::IsReservedWord.

  // Create dummy module file at <test-root>/mykeyword-1.1 1.o
  FileSpec module_fspec(CreateFile("mykeyword-1.1 1.o", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  // Create dummy module file at
  // <test-root>/.dSYM/Contents/Resources/DWARF/mykeyword-1.1 1.o
  FileSpec dsym_module_fpec(
      CreateFile("mykeyword-1.1 1.o", m_tmp_dsym_dwarf_dir));
  ASSERT_TRUE(dsym_module_fpec);

  CreateFile("mykeyword-1.1 1.py", m_tmp_dsym_python_dir);

  StreamString ss;
  auto fspecs = std::static_pointer_cast<PlatformDarwin>(m_platform_sp)
                    ->LocateExecutableScriptingResourcesFromDSYM(
                        ss, module_fspec, *m_target_sp, dsym_module_fpec);
  EXPECT_EQ(fspecs.size(), 0u);
  EXPECT_TRUE(
      ss.GetString().contains("conflicts with the keyword 'mykeyword_1_1_1'"));
}

TEST_F(
    PlatformDarwinLocateTest,
    LocateExecutableScriptingResourcesFromDSYM_ModuleNameIsKeywordAfterReplacement_Match_Warning) {
  // Like
  // LocateExecutableScriptingResourcesFromDSYM_ModuleNameIsKeywordAfterReplacement
  // but we place a script with all the replacement characters into the module
  // directory so LLDB loads it. That will still produce a warning.

  // Create dummy module file at <test-root>/mykeyword-1.1 1.o
  FileSpec module_fspec(CreateFile("mykeyword-1.1 1.o", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  // Create dummy module file at
  // <test-root>/.dSYM/Contents/Resources/DWARF/mykeyword-1.1 1.o
  FileSpec dsym_module_fpec(
      CreateFile("mykeyword-1.1 1.o", m_tmp_dsym_dwarf_dir));
  ASSERT_TRUE(dsym_module_fpec);

  CreateFile("mykeyword-1.1 1.py", m_tmp_dsym_python_dir);
  CreateFile("_mykeyword_1_1_1.py", m_tmp_dsym_python_dir);

  StreamString ss;
  auto fspecs = std::static_pointer_cast<PlatformDarwin>(m_platform_sp)
                    ->LocateExecutableScriptingResourcesFromDSYM(
                        ss, module_fspec, *m_target_sp, dsym_module_fpec);
  EXPECT_EQ(fspecs.size(), 1u);
  EXPECT_EQ(fspecs.begin()->getFirst().GetFilename(), "_mykeyword_1_1_1.py");
  EXPECT_TRUE(ss.GetString().contains("Ignoring 'mykeyword-1.1 1.py' and "
                                      "loading '_mykeyword_1_1_1.py' instead"));
}

TEST_F(
    PlatformDarwinLocateTest,
    LocateExecutableScriptingResourcesFromDSYM_ModuleNameIsKeywordAfterReplacement_Match_NoWarning) {
  // Like
  // LocateExecutableScriptingResourcesFromDSYM_ModuleNameIsKeywordAfterReplacement_Match_Warning
  // but we place a script with all the replacement characters into the module
  // directory so LLDB loads it (but no script that matches the original module
  // name, and hence generates no warning).

  // Create dummy module file at <test-root>/mykeyword-1.1 1.o
  FileSpec module_fspec(CreateFile("mykeyword-1.1 1.o", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  // Create dummy module file at
  // <test-root>/.dSYM/Contents/Resources/DWARF/mykeyword-1.1 1.o
  FileSpec dsym_module_fpec(
      CreateFile("mykeyword-1.1 1.o", m_tmp_dsym_dwarf_dir));
  ASSERT_TRUE(dsym_module_fpec);

  CreateFile("_mykeyword_1_1_1.py", m_tmp_dsym_python_dir);

  StreamString ss;
  auto fspecs = std::static_pointer_cast<PlatformDarwin>(m_platform_sp)
                    ->LocateExecutableScriptingResourcesFromDSYM(
                        ss, module_fspec, *m_target_sp, dsym_module_fpec);
  EXPECT_EQ(fspecs.size(), 1u);
  EXPECT_EQ(fspecs.begin()->getFirst().GetFilename(), "_mykeyword_1_1_1.py");
  EXPECT_TRUE(ss.Empty());
}

struct SpecialCharTestCase {
  char special_char;
  char replacement;
};
struct PlatformDarwinLocateWithSpecialCharsTestFixture
    : public testing::WithParamInterface<SpecialCharTestCase>,
      public PlatformDarwinLocateTest {};

TEST_P(PlatformDarwinLocateWithSpecialCharsTestFixture,
       LocateExecutableScriptingResourcesFromDSYM_SpecialCharacters) {
  // Tests the various special characters that `ScriptInterpreterPython`
  // disallows in module names.

  auto [special_char, replacement] = GetParam();

  std::string module_name = llvm::formatv("TestModule{0}.o", special_char);
  std::string script_name = llvm::formatv("TestModule{0}.py", special_char);
  std::string recommended_script_name =
      llvm::formatv("TestModule{0}.py", replacement);

  // Create dummy module file at <test-root>/<module-name>
  FileSpec module_fspec(CreateFile(module_name, m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  // Create dummy module file at
  // <test-root>/.dSYM/Contents/Resources/DWARF/<module-name>
  FileSpec dsym_module_fpec(CreateFile(module_name, m_tmp_dsym_dwarf_dir));
  ASSERT_TRUE(dsym_module_fpec);

  CreateFile(script_name, m_tmp_dsym_python_dir);

  StreamString ss;
  auto fspecs = std::static_pointer_cast<PlatformDarwin>(m_platform_sp)
                    ->LocateExecutableScriptingResourcesFromDSYM(
                        ss, module_fspec, *m_target_sp, dsym_module_fpec);
  EXPECT_EQ(fspecs.size(), 0u);

  std::string expected =
      llvm::formatv("please rename it to '{0}'", recommended_script_name);
  EXPECT_TRUE(ss.GetString().contains(expected));
}

INSTANTIATE_TEST_SUITE_P(PlatformDarwinLocateWithSpecialCharsTest,
                         PlatformDarwinLocateWithSpecialCharsTestFixture,
                         testing::ValuesIn(std::vector<SpecialCharTestCase>{
                             {' ', '_'}, {'.', '_'}, {'-', '_'}, {'+', 'x'}}));
