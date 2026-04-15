//===-- PlatformTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "TestUtils.h"

#include "Plugins/Platform/POSIX/PlatformPOSIX.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

class TestPlatform : public PlatformPOSIX {
public:
  TestPlatform() : PlatformPOSIX(false) {}
};

class PlatformArm : public TestPlatform {
public:
  PlatformArm() = default;

  std::vector<ArchSpec>
  GetSupportedArchitectures(const ArchSpec &process_host_arch) override {
    return {ArchSpec("arm64-apple-ps4")};
  }

  llvm::StringRef GetPluginName() override { return "arm"; }
  llvm::StringRef GetDescription() override { return "arm"; }
};

class PlatformIntel : public TestPlatform {
public:
  PlatformIntel() = default;

  std::vector<ArchSpec>
  GetSupportedArchitectures(const ArchSpec &process_host_arch) override {
    return {ArchSpec("x86_64-apple-ps4")};
  }

  llvm::StringRef GetPluginName() override { return "intel"; }
  llvm::StringRef GetDescription() override { return "intel"; }
};

class PlatformThumb : public TestPlatform {
public:
  static void Initialize() {
    PluginManager::RegisterPlugin("thumb", "thumb",
                                  PlatformThumb::CreateInstance);
  }
  static void Terminate() {
    PluginManager::UnregisterPlugin(PlatformThumb::CreateInstance);
  }

  static PlatformSP CreateInstance(bool force, const ArchSpec *arch) {
    return std::make_shared<PlatformThumb>();
  }

  std::vector<ArchSpec>
  GetSupportedArchitectures(const ArchSpec &process_host_arch) override {
    return {ArchSpec("thumbv7-apple-ps4"), ArchSpec("thumbv7f-apple-ps4")};
  }

  llvm::StringRef GetPluginName() override { return "thumb"; }
  llvm::StringRef GetDescription() override { return "thumb"; }
};

class PlatformTest : public ::testing::Test {
  SubsystemRAII<FileSystem, HostInfo> subsystems;

protected:
  PlatformList list;

  void SetHostPlatform(const PlatformSP &platform_sp) {
    Platform::SetHostPlatform(platform_sp);
    ASSERT_EQ(Platform::GetHostPlatform(), platform_sp);
    list.Append(platform_sp, /*set_selected=*/true);
  }
};

TEST_F(PlatformTest, GetPlatformForArchitecturesHost) {
  SetHostPlatform(std::make_shared<PlatformArm>());

  const std::vector<ArchSpec> archs = {ArchSpec("arm64-apple-ps4"),
                                       ArchSpec("arm64e-apple-ps4")};
  std::vector<PlatformSP> candidates;

  // The host platform matches all architectures.
  PlatformSP platform_sp = list.GetOrCreate(archs, {}, candidates);
  ASSERT_TRUE(platform_sp);
  EXPECT_EQ(platform_sp, Platform::GetHostPlatform());
}

TEST_F(PlatformTest, GetPlatformForArchitecturesSelected) {
  SetHostPlatform(std::make_shared<PlatformIntel>());

  const std::vector<ArchSpec> archs = {ArchSpec("arm64-apple-ps4"),
                                       ArchSpec("arm64e-apple-ps4")};
  std::vector<PlatformSP> candidates;

  // The host platform matches no architectures.
  PlatformSP platform_sp = list.GetOrCreate(archs, {}, candidates);
  ASSERT_FALSE(platform_sp);

  // The selected platform matches all architectures.
  const PlatformSP selected_platform_sp = std::make_shared<PlatformArm>();
  list.Append(selected_platform_sp, /*set_selected=*/true);
  platform_sp = list.GetOrCreate(archs, {}, candidates);
  ASSERT_TRUE(platform_sp);
  EXPECT_EQ(platform_sp, selected_platform_sp);
}

TEST_F(PlatformTest, GetPlatformForArchitecturesSelectedOverHost) {
  SetHostPlatform(std::make_shared<PlatformIntel>());

  const std::vector<ArchSpec> archs = {ArchSpec("arm64-apple-ps4"),
                                       ArchSpec("x86_64-apple-ps4")};
  std::vector<PlatformSP> candidates;

  // The host platform matches one architecture.
  PlatformSP platform_sp = list.GetOrCreate(archs, {}, candidates);
  ASSERT_TRUE(platform_sp);
  EXPECT_EQ(platform_sp, Platform::GetHostPlatform());

  // The selected and host platform each match one architecture.
  // The selected platform is preferred.
  const PlatformSP selected_platform_sp = std::make_shared<PlatformArm>();
  list.Append(selected_platform_sp, /*set_selected=*/true);
  platform_sp = list.GetOrCreate(archs, {}, candidates);
  ASSERT_TRUE(platform_sp);
  EXPECT_EQ(platform_sp, selected_platform_sp);
}

TEST_F(PlatformTest, GetPlatformForArchitecturesCandidates) {
  PlatformThumb::Initialize();

  SetHostPlatform(std::make_shared<PlatformIntel>());

  const PlatformSP selected_platform_sp = std::make_shared<PlatformArm>();
  list.Append(selected_platform_sp, /*set_selected=*/true);

  const std::vector<ArchSpec> archs = {ArchSpec("thumbv7-apple-ps4"),
                                       ArchSpec("thumbv7f-apple-ps4")};
  std::vector<PlatformSP> candidates;

  // The host platform matches one architecture.
  PlatformSP platform_sp = list.GetOrCreate(archs, {}, candidates);
  ASSERT_TRUE(platform_sp);
  EXPECT_EQ(platform_sp->GetName(), "thumb");

  PlatformThumb::Terminate();
}

TEST_F(PlatformTest, CreateUnknown) {
  SetHostPlatform(std::make_shared<PlatformIntel>());
  ASSERT_EQ(list.Create("unknown-platform-name"), nullptr);
  ASSERT_EQ(list.GetOrCreate("dummy"), nullptr);
}

// TestingProperties are only available in asserts builds.
#ifndef NDEBUG
struct PlatformLocateSafePathTest : public PlatformTest {
protected:
  void SetUp() override {
    std::call_once(TestUtilities::g_debugger_initialize_flag,
                   []() { Debugger::Initialize(nullptr); });

    ArchSpec arch("x86_64-apple-macosx-");
    m_platform_sp = std::make_shared<PlatformArm>();
    Platform::SetHostPlatform(m_platform_sp);

    m_debugger_sp = Debugger::CreateInstance();

    m_debugger_sp->GetTargetList().CreateTarget(*m_debugger_sp, "", arch,
                                                lldb_private::eLoadDependentsNo,
                                                m_platform_sp, m_target_sp);

    ASSERT_TRUE(m_target_sp);
    ASSERT_TRUE(m_platform_sp);

    ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory(
        "locate-scripts-from-safe-paths-test", m_tmp_root_dir))
        << "Failed to create test directory.";
  };

  void TearDown() override {
    llvm::sys::fs::remove_directories(m_tmp_root_dir);
    TestingProperties::GetGlobalTestingProperties().SetSafeAutoLoadPaths({});
  }

  DebuggerSP m_debugger_sp;
  PlatformSP m_platform_sp;
  TargetSP m_target_sp;

  /// Root directory for m_tmp_dsym_dwarf_dir and m_tmp_dsym_python_dir
  llvm::SmallString<128> m_tmp_root_dir;

  SubsystemRAII<MockScriptInterpreterPython> subsystems;
};

TEST_F(PlatformLocateSafePathTest,
       LocateScriptingResourcesFromSafePaths_NoSetting) {
  // Tests LocateScriptingResourcesFromSafePaths finds no script if we don't set
  // the safe path setting.

  // Create dummy module file at <test-root>/TestModule.o
  FileSpec module_fspec(CreateFile("TestModule.o", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  llvm::SmallString<128> module_dir(m_tmp_root_dir);
  llvm::sys::path::append(module_dir, "TestModule");
  ASSERT_FALSE(llvm::sys::fs::create_directory(module_dir));

  CreateFile("TestModule.py", module_dir);

  StreamString ss;
  auto file_specs = Platform::LocateExecutableScriptingResourcesFromSafePaths(
      ss, module_fspec, *m_target_sp);

  ASSERT_EQ(file_specs.size(), 0u);
}

TEST_F(PlatformLocateSafePathTest,
       LocateScriptingResourcesFromSafePaths_NoMatch) {
  // Tests LocateScriptingResourcesFromSafePaths finds no directory to load
  // from.

  TestingProperties::GetGlobalTestingProperties().AppendSafeAutoLoadPaths(
      FileSpec(m_tmp_root_dir));

  // Create dummy module file at <test-root>/TestModule.o
  FileSpec module_fspec(CreateFile("TestModule.o", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  // Directory name doesn't match the module name.
  llvm::SmallString<128> module_dir(m_tmp_root_dir);
  llvm::sys::path::append(module_dir, "TestModule1");
  ASSERT_FALSE(llvm::sys::fs::create_directory(module_dir));

  CreateFile("TestModule1.py", module_dir);

  StreamString ss;
  auto file_specs = Platform::LocateExecutableScriptingResourcesFromSafePaths(
      ss, module_fspec, *m_target_sp);

  ASSERT_EQ(file_specs.size(), 0u);
}

TEST_F(PlatformLocateSafePathTest,
       LocateScriptingResourcesFromSafePaths_Match) {
  // Tests LocateScriptingResourcesFromSafePaths locates the
  // <module-name>/<module-name>.py script correctly.

  TestingProperties::GetGlobalTestingProperties().AppendSafeAutoLoadPaths(
      FileSpec(m_tmp_root_dir));

  // Create dummy module file at <test-root>/TestModule.o
  FileSpec module_fspec(CreateFile("TestModule.o", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  llvm::SmallString<128> module_dir(m_tmp_root_dir);
  llvm::sys::path::append(module_dir, "TestModule");
  ASSERT_FALSE(llvm::sys::fs::create_directory(module_dir));

  CreateFile("TestModule.py", module_dir);
  // Other files should be ignored.
  CreateFile("helper.py", module_dir);
  CreateFile("not_a_script.txt", module_dir);

  StreamString ss;
  auto file_specs = Platform::LocateExecutableScriptingResourcesFromSafePaths(
      ss, module_fspec, *m_target_sp);

  EXPECT_EQ(file_specs.size(), 1u);

  auto [fspec, load_style] = *file_specs.begin();
  EXPECT_EQ(fspec.GetFilename(), "TestModule.py");
  EXPECT_EQ(load_style, m_target_sp->GetLoadScriptFromSymbolFile());
}

TEST_F(PlatformLocateSafePathTest,
       LocateScriptingResourcesFromSafePaths_NestedDir) {
  // Tests that a matching Python file nested inside a subdirectory is not
  // picked up. Only <module-name>/<module-name>.py at the top level matters.

  TestingProperties::GetGlobalTestingProperties().AppendSafeAutoLoadPaths(
      FileSpec(m_tmp_root_dir));

  // Create dummy module file at <test-root>/TestModule.o
  FileSpec module_fspec(CreateFile("TestModule.o", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  llvm::SmallString<128> module_dir(m_tmp_root_dir);
  llvm::sys::path::append(module_dir, "TestModule");
  ASSERT_FALSE(llvm::sys::fs::create_directory(module_dir));

  // Create a nested directory that contains the matching Python file.
  llvm::SmallString<128> nested_dir(module_dir);
  llvm::sys::path::append(nested_dir, "nested");
  ASSERT_FALSE(llvm::sys::fs::create_directory(nested_dir));

  CreateFile("TestModule.py", nested_dir);

  StreamString ss;
  auto file_specs = Platform::LocateExecutableScriptingResourcesFromSafePaths(
      ss, module_fspec, *m_target_sp);

  EXPECT_EQ(file_specs.size(), 0u);
}

TEST_F(PlatformLocateSafePathTest,
       LocateScriptingResourcesFromSafePaths_MultiplePaths) {
  // Tests LocateScriptingResourcesFromSafePaths locates the script from the
  // last appended auto-load path.

  // Create dummy module file at <test-root>/TestModule.o
  FileSpec module_fspec(CreateFile("TestModule.o", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  llvm::SmallString<128> path1(m_tmp_root_dir);
  llvm::sys::path::append(path1, "AnotherSafePath");
  ASSERT_FALSE(llvm::sys::fs::create_directory(path1));

  llvm::SmallString<128> path2(m_tmp_root_dir);
  llvm::sys::path::append(path2, "AnotherAnotherSafePath");
  ASSERT_FALSE(llvm::sys::fs::create_directory(path2));

  llvm::SmallString<128> path3(m_tmp_root_dir);
  llvm::sys::path::append(path3, "EmptySafePath");
  ASSERT_FALSE(llvm::sys::fs::create_directory(path3));

  llvm::SmallString<128> module_dir(m_tmp_root_dir);
  llvm::sys::path::append(module_dir, "TestModule");
  ASSERT_FALSE(llvm::sys::fs::create_directory(module_dir));

  llvm::SmallString<128> path1_module_dir(path1);
  llvm::sys::path::append(path1_module_dir, "TestModule");
  ASSERT_FALSE(llvm::sys::fs::create_directory(path1_module_dir));

  llvm::SmallString<128> path2_module_dir(path2);
  llvm::sys::path::append(path2_module_dir, "NotTheTestModule");
  ASSERT_FALSE(llvm::sys::fs::create_directory(path2_module_dir));

  llvm::SmallString<128> path3_module_dir(path3);
  llvm::sys::path::append(path3_module_dir, "TestModule");
  ASSERT_FALSE(llvm::sys::fs::create_directory(path3_module_dir));

  // Place the correctly named script in each module directory.
  CreateFile("TestModule.py", module_dir);
  CreateFile("TestModule.py", path1_module_dir);
  CreateFile("TestModule.py", path2_module_dir);
  // Keep path3 (EmptySafePath) empty.

  TestingProperties::GetGlobalTestingProperties().AppendSafeAutoLoadPaths(
      FileSpec(m_tmp_root_dir));

  TestingProperties::GetGlobalTestingProperties().AppendSafeAutoLoadPaths(
      FileSpec(path1));

  TestingProperties::GetGlobalTestingProperties().AppendSafeAutoLoadPaths(
      FileSpec(path2));

  StreamString ss;
  auto file_specs = Platform::LocateExecutableScriptingResourcesFromSafePaths(
      ss, module_fspec, *m_target_sp);

  // path1 was the last appended path with a matching directory.
  auto [fspec, load_style] = *file_specs.begin();
  EXPECT_TRUE(llvm::StringRef(fspec.GetPath()).contains("AnotherSafePath"));
  EXPECT_EQ(fspec.GetFilename(), "TestModule.py");
  EXPECT_EQ(load_style, m_target_sp->GetLoadScriptFromSymbolFile());

  // Now add another safe path with a valid module directory but no
  // TestModule.py inside. LLDB shouldn't fall back to other matching safe
  // paths.

  TestingProperties::GetGlobalTestingProperties().AppendSafeAutoLoadPaths(
      FileSpec(path3));

  file_specs = Platform::LocateExecutableScriptingResourcesFromSafePaths(
      ss, module_fspec, *m_target_sp);

  EXPECT_EQ(file_specs.size(), 0u);

  // Now place the correctly named script in path3.
  CreateFile("TestModule.py", path3_module_dir);

  file_specs = Platform::LocateExecutableScriptingResourcesFromSafePaths(
      ss, module_fspec, *m_target_sp);

  EXPECT_EQ(file_specs.size(), 1u);

  auto [fspec1, load_style1] = *file_specs.begin();
  EXPECT_TRUE(llvm::StringRef(fspec1.GetPath()).contains("EmptySafePath"));
  EXPECT_EQ(fspec1.GetFilename(), "TestModule.py");
  EXPECT_EQ(load_style1, m_target_sp->GetLoadScriptFromSymbolFile());
}

TEST_F(PlatformLocateSafePathTest,
       LocateScriptingResourcesFromSafePaths_SpecialChars_NoMatch) {
  // Module name has special characters. The directory exists but only contains
  // a script with the original (unsanitized) name. No match.

  TestingProperties::GetGlobalTestingProperties().AppendSafeAutoLoadPaths(
      FileSpec(m_tmp_root_dir));

  // Create dummy module file at <test-root>/TestModule-1.1 1.o
  FileSpec module_fspec(CreateFile("TestModule-1.1 1.o", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  llvm::SmallString<128> module_dir(m_tmp_root_dir);
  llvm::sys::path::append(module_dir, "TestModule-1.1 1");
  ASSERT_FALSE(llvm::sys::fs::create_directory(module_dir));

  // Only the unsanitized name exists.
  FileSpec orig_fspec(CreateFile("TestModule-1.1 1.py", module_dir));
  ASSERT_TRUE(orig_fspec);

  StreamString ss;
  auto file_specs = Platform::LocateExecutableScriptingResourcesFromSafePaths(
      ss, module_fspec, *m_target_sp);

  EXPECT_EQ(file_specs.size(), 0u);

  std::string expected = llvm::formatv(
      "debug script '{0}' cannot be loaded because"
      " 'TestModule-1.1 1.py' contains reserved characters. If you intend to"
      " have this script loaded, please rename it to 'TestModule_1_1_1.py' and "
      "retry.\n",
      orig_fspec.GetPath());
  EXPECT_EQ(ss.GetString(), expected);
}

TEST_F(PlatformLocateSafePathTest,
       LocateScriptingResourcesFromSafePaths_SpecialChars_Match_Warning) {
  // Module name has special characters. Both the original and sanitized scripts
  // exist. LLDB loads the sanitized one and warns.

  TestingProperties::GetGlobalTestingProperties().AppendSafeAutoLoadPaths(
      FileSpec(m_tmp_root_dir));

  // Create dummy module file at <test-root>/TestModule-1.1 1.o
  FileSpec module_fspec(CreateFile("TestModule-1.1 1.o", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  llvm::SmallString<128> module_dir(m_tmp_root_dir);
  llvm::sys::path::append(module_dir, "TestModule-1.1 1");
  ASSERT_FALSE(llvm::sys::fs::create_directory(module_dir));

  FileSpec orig_fspec(CreateFile("TestModule-1.1 1.py", module_dir));
  ASSERT_TRUE(orig_fspec);

  CreateFile("TestModule_1_1_1.py", module_dir);

  StreamString ss;
  auto file_specs = Platform::LocateExecutableScriptingResourcesFromSafePaths(
      ss, module_fspec, *m_target_sp);

  EXPECT_EQ(file_specs.size(), 1u);

  auto [fspec, load_style] = *file_specs.begin();
  EXPECT_EQ(fspec.GetFilename(), "TestModule_1_1_1.py");
  EXPECT_EQ(load_style, m_target_sp->GetLoadScriptFromSymbolFile());

  std::string expected = llvm::formatv(
      "debug script '{0}' cannot be loaded because"
      " 'TestModule-1.1 1.py' contains reserved characters. Ignoring"
      " 'TestModule-1.1 1.py' and loading 'TestModule_1_1_1.py' instead.\n",
      orig_fspec.GetPath());
  EXPECT_EQ(ss.GetString(), expected);
}

TEST_F(PlatformLocateSafePathTest,
       LocateScriptingResourcesFromSafePaths_SpecialChars_Match_NoWarning) {
  // Module name has special characters. Only the sanitized script exists.
  // No warning.

  TestingProperties::GetGlobalTestingProperties().AppendSafeAutoLoadPaths(
      FileSpec(m_tmp_root_dir));

  // Create dummy module file at <test-root>/TestModule-1.1 1.o
  FileSpec module_fspec(CreateFile("TestModule-1.1 1.o", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  llvm::SmallString<128> module_dir(m_tmp_root_dir);
  llvm::sys::path::append(module_dir, "TestModule-1.1 1");
  ASSERT_FALSE(llvm::sys::fs::create_directory(module_dir));

  CreateFile("TestModule_1_1_1.py", module_dir);

  StreamString ss;
  auto file_specs = Platform::LocateExecutableScriptingResourcesFromSafePaths(
      ss, module_fspec, *m_target_sp);

  EXPECT_EQ(file_specs.size(), 1u);

  auto [fspec, load_style] = *file_specs.begin();
  EXPECT_EQ(fspec.GetFilename(), "TestModule_1_1_1.py");
  EXPECT_EQ(load_style, m_target_sp->GetLoadScriptFromSymbolFile());
  EXPECT_TRUE(ss.GetString().empty());
}

TEST_F(PlatformLocateSafePathTest,
       LocateScriptingResourcesFromSafePaths_Keyword_NoMatch) {
  // Module name is a reserved keyword. Only the original script exists.
  // Warns and returns nothing.

  TestingProperties::GetGlobalTestingProperties().AppendSafeAutoLoadPaths(
      FileSpec(m_tmp_root_dir));

  // Create dummy module file at <test-root>/import.o
  FileSpec module_fspec(CreateFile("import.o", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  llvm::SmallString<128> module_dir(m_tmp_root_dir);
  llvm::sys::path::append(module_dir, "import");
  ASSERT_FALSE(llvm::sys::fs::create_directory(module_dir));

  FileSpec orig_fspec(CreateFile("import.py", module_dir));
  ASSERT_TRUE(orig_fspec);

  StreamString ss;
  auto file_specs = Platform::LocateExecutableScriptingResourcesFromSafePaths(
      ss, module_fspec, *m_target_sp);

  EXPECT_EQ(file_specs.size(), 0u);

  std::string expected = llvm::formatv(
      "debug script '{0}' cannot be loaded because 'import.py' "
      "conflicts with the keyword 'import'. If you intend to have this script "
      "loaded, please rename it to '_import.py' and retry.\n",
      orig_fspec.GetPath());
  EXPECT_EQ(ss.GetString(), expected);
}

TEST_F(PlatformLocateSafePathTest,
       LocateScriptingResourcesFromSafePaths_Keyword_Match) {
  // Module name is a reserved keyword. Both original and sanitized scripts
  // exist. Loads the sanitized one and warns.

  TestingProperties::GetGlobalTestingProperties().AppendSafeAutoLoadPaths(
      FileSpec(m_tmp_root_dir));

  // Create dummy module file at <test-root>/import.o
  FileSpec module_fspec(CreateFile("import.o", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  llvm::SmallString<128> module_dir(m_tmp_root_dir);
  llvm::sys::path::append(module_dir, "import");
  ASSERT_FALSE(llvm::sys::fs::create_directory(module_dir));

  FileSpec orig_fspec(CreateFile("import.py", module_dir));
  ASSERT_TRUE(orig_fspec);

  CreateFile("_import.py", module_dir);

  StreamString ss;
  auto file_specs = Platform::LocateExecutableScriptingResourcesFromSafePaths(
      ss, module_fspec, *m_target_sp);

  EXPECT_EQ(file_specs.size(), 1u);

  auto [fspec, load_style] = *file_specs.begin();
  EXPECT_EQ(fspec.GetFilename(), "_import.py");
  EXPECT_EQ(load_style, m_target_sp->GetLoadScriptFromSymbolFile());

  std::string expected =
      llvm::formatv("debug script '{0}' cannot be loaded because 'import.py' "
                    "conflicts with the keyword 'import'. Ignoring 'import.py' "
                    "and loading '_import.py' instead.\n",
                    orig_fspec.GetPath());
  EXPECT_EQ(ss.GetString(), expected);
}

TEST_F(PlatformLocateSafePathTest,
       LocateScriptingResourcesFromSafePaths_Keyword_Match_NoWarning) {
  // Module name is a reserved keyword. Only the sanitized script exists.
  // No warning.

  TestingProperties::GetGlobalTestingProperties().AppendSafeAutoLoadPaths(
      FileSpec(m_tmp_root_dir));

  // Create dummy module file at <test-root>/import.o
  FileSpec module_fspec(CreateFile("import.o", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  llvm::SmallString<128> module_dir(m_tmp_root_dir);
  llvm::sys::path::append(module_dir, "import");
  ASSERT_FALSE(llvm::sys::fs::create_directory(module_dir));

  CreateFile("_import.py", module_dir);

  StreamString ss;
  auto file_specs = Platform::LocateExecutableScriptingResourcesFromSafePaths(
      ss, module_fspec, *m_target_sp);

  EXPECT_EQ(file_specs.size(), 1u);

  auto [fspec, load_style] = *file_specs.begin();
  EXPECT_EQ(fspec.GetFilename(), "_import.py");
  EXPECT_EQ(load_style, m_target_sp->GetLoadScriptFromSymbolFile());
  EXPECT_TRUE(ss.GetString().empty());
}

TEST_F(PlatformLocateSafePathTest,
       LocateScriptingResourcesFromSafePaths_InnerDirectoryHasModuleName) {
  // Test a directory structure like
  // <safe-path>/TestModule/TestModule/TestModule.py. LLDB should not load that
  // inner script.

  TestingProperties::GetGlobalTestingProperties().AppendSafeAutoLoadPaths(
      FileSpec(m_tmp_root_dir));

  // Create dummy module file at <test-root>/TestModule.o
  FileSpec module_fspec(CreateFile("TestModule.o", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  llvm::SmallString<128> inner_dir(m_tmp_root_dir);
  llvm::sys::path::append(inner_dir, "TestModule", "TestModule");
  ASSERT_FALSE(llvm::sys::fs::create_directories(inner_dir));

  CreateFile("TestModule.py", inner_dir);

  StreamString ss;
  auto file_specs = Platform::LocateExecutableScriptingResourcesFromSafePaths(
      ss, module_fspec, *m_target_sp);

  EXPECT_EQ(file_specs.size(), 0u);
  EXPECT_TRUE(ss.GetString().empty());
}

TEST_F(PlatformLocateSafePathTest,
       LocateScriptingResourcesFromSafePaths_RelativePaths) {
  // Make sure we locate scripts correctly if the safe path contains a
  // non-absolute path.

  llvm::SmallString<128> inner_dir(m_tmp_root_dir);
  llvm::sys::path::append(inner_dir, "Inner", "Dir");
  ASSERT_FALSE(llvm::sys::fs::create_directories(inner_dir));

  llvm::SmallString<128> relative_dir(inner_dir);
  llvm::sys::path::append(relative_dir, "..", "..");
  TestingProperties::GetGlobalTestingProperties().AppendSafeAutoLoadPaths(
      FileSpec(relative_dir));

  // Create dummy module file at <test-root>/TestModule.o
  FileSpec module_fspec(CreateFile("TestModule.o", m_tmp_root_dir));
  ASSERT_TRUE(module_fspec);

  llvm::SmallString<128> module_dir(m_tmp_root_dir);
  llvm::sys::path::append(module_dir, "TestModule");
  ASSERT_FALSE(llvm::sys::fs::create_directory(module_dir));

  CreateFile("TestModule.py", module_dir);

  StreamString ss;
  auto file_specs = Platform::LocateExecutableScriptingResourcesFromSafePaths(
      ss, module_fspec, *m_target_sp);

  EXPECT_EQ(file_specs.size(), 1u);
  EXPECT_TRUE(ss.GetString().empty());
}
#endif // NDEBUG
