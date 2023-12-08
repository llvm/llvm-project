#include "gtest/gtest.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/SymbolFile/Symtab/SymbolFileSymtab.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/ModuleCache.h"

using namespace lldb_private;
using namespace lldb;

namespace {

class ModuleCacheTest : public testing::Test {
  SubsystemRAII<FileSystem, HostInfo, ObjectFileELF, SymbolFileSymtab>
      subsystems;

public:
  void SetUp() override;

protected:
  FileSpec s_cache_dir;
  std::string s_test_executable;

  void TryGetAndPut(const FileSpec &cache_dir, const char *hostname,
                    bool expect_download);
};
}

static const char dummy_hostname[] = "dummy_hostname";
static const char dummy_remote_dir[] = "bin";
static const char module_name[] = "TestModule.so";
static const char module_uuid[] =
    "F4E7E991-9B61-6AD4-0073-561AC3D9FA10-C043A476";
static const size_t module_size = 5602;

static FileSpec GetDummyRemotePath() {
  FileSpec fs("/", FileSpec::Style::posix);
  fs.AppendPathComponent(dummy_remote_dir);
  fs.AppendPathComponent(module_name);
  return fs;
}

static FileSpec GetUuidView(FileSpec spec) {
  spec.AppendPathComponent(".cache");
  spec.AppendPathComponent(module_uuid);
  spec.AppendPathComponent(module_name);
  return spec;
}

static FileSpec GetSysrootView(FileSpec spec, const char *hostname) {
  spec.AppendPathComponent(hostname);
  spec.AppendPathComponent(dummy_remote_dir);
  spec.AppendPathComponent(module_name);
  return spec;
}

void ModuleCacheTest::SetUp() {
  s_cache_dir = HostInfo::GetProcessTempDir();
  s_test_executable = GetInputFilePath(module_name);
}

static void VerifyDiskState(const FileSpec &cache_dir, const char *hostname) {
  FileSpec uuid_view = GetUuidView(cache_dir);
  EXPECT_TRUE(FileSystem::Instance().Exists(uuid_view))
      << "uuid_view is: " << uuid_view.GetPath();
  EXPECT_EQ(module_size, FileSystem::Instance().GetByteSize(uuid_view));

  FileSpec sysroot_view = GetSysrootView(cache_dir, hostname);
  EXPECT_TRUE(FileSystem::Instance().Exists(sysroot_view))
      << "sysroot_view is: " << sysroot_view.GetPath();
  EXPECT_EQ(module_size, FileSystem::Instance().GetByteSize(sysroot_view));
}

void ModuleCacheTest::TryGetAndPut(const FileSpec &cache_dir,
                                   const char *hostname, bool expect_download) {
  ModuleCache mc;
  ModuleSpec module_spec;
  module_spec.GetFileSpec() = GetDummyRemotePath();
  module_spec.GetUUID().SetFromStringRef(module_uuid);
  module_spec.SetObjectSize(module_size);
  ModuleSP module_sp;
  bool did_create;
  bool download_called = false;

  Status error = mc.GetAndPut(
      cache_dir, hostname, module_spec,
      [&download_called, this](const ModuleSpec &module_spec,
                               const FileSpec &tmp_download_file_spec) {
        download_called = true;
        EXPECT_STREQ(GetDummyRemotePath().GetPath().c_str(),
                     module_spec.GetFileSpec().GetPath().c_str());
        std::error_code ec = llvm::sys::fs::copy_file(
            s_test_executable, tmp_download_file_spec.GetPath());
        EXPECT_FALSE(ec);
        return Status();
      },
      [](const ModuleSP &module_sp, const FileSpec &tmp_download_file_spec) {
        return Status("Not supported.");
      },
      module_sp, &did_create);
  EXPECT_EQ(expect_download, download_called);

  EXPECT_TRUE(error.Success()) << "Error was: " << error.AsCString();
  EXPECT_TRUE(did_create);
  ASSERT_TRUE(bool(module_sp));

  SymbolContextList sc_list;
  module_sp->FindFunctionSymbols(ConstString("boom"), eFunctionNameTypeFull,
                                 sc_list);
  EXPECT_EQ(1u, sc_list.GetSize());
  EXPECT_STREQ(GetDummyRemotePath().GetPath().c_str(),
               module_sp->GetPlatformFileSpec().GetPath().c_str());
  EXPECT_STREQ(module_uuid, module_sp->GetUUID().GetAsString().c_str());
}

TEST_F(ModuleCacheTest, GetAndPut) {
  FileSpec test_cache_dir = s_cache_dir;
  test_cache_dir.AppendPathComponent("GetAndPut");

  const bool expect_download = true;
  TryGetAndPut(test_cache_dir, dummy_hostname, expect_download);
  VerifyDiskState(test_cache_dir, dummy_hostname);
}

TEST_F(ModuleCacheTest, GetAndPutUuidExists) {
  FileSpec test_cache_dir = s_cache_dir;
  test_cache_dir.AppendPathComponent("GetAndPutUuidExists");

  FileSpec uuid_view = GetUuidView(test_cache_dir);
  std::error_code ec =
      llvm::sys::fs::create_directories(uuid_view.GetDirectory().GetCString());
  ASSERT_FALSE(ec);
  ec = llvm::sys::fs::copy_file(s_test_executable, uuid_view.GetPath().c_str());
  ASSERT_FALSE(ec);

  const bool expect_download = false;
  TryGetAndPut(test_cache_dir, dummy_hostname, expect_download);
  VerifyDiskState(test_cache_dir, dummy_hostname);
}

TEST_F(ModuleCacheTest, GetAndPutStrangeHostname) {
  FileSpec test_cache_dir = s_cache_dir;
  test_cache_dir.AppendPathComponent("GetAndPutStrangeHostname");

  const bool expect_download = true;
  TryGetAndPut(test_cache_dir, "tab\tcolon:asterisk*", expect_download);
  VerifyDiskState(test_cache_dir, "tab_colon_asterisk_");
}
