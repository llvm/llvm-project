//===-- PlatformAndroidTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Platform/Android/PlatformAndroid.h"
#include "Plugins/Platform/Android/PlatformAndroidRemoteGDBServer.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Utility/Connection.h"
#include "gmock/gmock.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_android;
using namespace testing;

namespace {

class MockSyncService : public AdbClient::SyncService {
public:
  MockSyncService() : SyncService(std::unique_ptr<Connection>()) {}

  MOCK_METHOD2(PullFile,
               Status(const FileSpec &remote_file, const FileSpec &local_file));
  MOCK_METHOD4(Stat, Status(const FileSpec &remote_file, uint32_t &mode,
                            uint32_t &size, uint32_t &mtime));
};

typedef std::unique_ptr<AdbClient::SyncService> SyncServiceUP;

class MockAdbClient : public AdbClient {
public:
  explicit MockAdbClient() : AdbClient("mock") {}

  MOCK_METHOD3(ShellToFile,
               Status(const char *command, std::chrono::milliseconds timeout,
                      const FileSpec &output_file_spec));
  MOCK_METHOD1(GetSyncService, SyncServiceUP(Status &error));
};

class PlatformAndroidTest : public PlatformAndroid, public ::testing::Test {
public:
  PlatformAndroidTest() : PlatformAndroid(false) {
    m_remote_platform_sp = PlatformSP(new PlatformAndroidRemoteGDBServer());
  }

  MOCK_METHOD1(GetAdbClient, AdbClientUP(Status &error));
  MOCK_METHOD0(GetPropertyPackageName, llvm::StringRef());
};

} // namespace

TEST_F(PlatformAndroidTest, DownloadModuleSliceWithAdbClientError) {
  EXPECT_CALL(*this, GetAdbClient(_))
      .Times(1)
      .WillOnce(DoAll(WithArg<0>([](auto &arg) {
                        arg = Status("Failed to create AdbClient");
                      }),
                      Return(ByMove(AdbClientUP()))));

  EXPECT_TRUE(
      DownloadModuleSlice(
          FileSpec("/system/app/Test/Test.apk!/lib/arm64-v8a/libtest.so"), 4096,
          3600, FileSpec())
          .Fail());
}

TEST_F(PlatformAndroidTest, DownloadModuleSliceWithNormalFile) {
  auto sync_service = new MockSyncService();
  EXPECT_CALL(*sync_service, Stat(FileSpec("/system/lib64/libc.so"), _, _, _))
      .Times(1)
      .WillOnce(DoAll(SetArgReferee<1>(1), Return(Status())));
  EXPECT_CALL(*sync_service, PullFile(FileSpec("/system/lib64/libc.so"), _))
      .Times(1)
      .WillOnce(Return(Status()));

  auto adb_client = new MockAdbClient();
  EXPECT_CALL(*adb_client, GetSyncService(_))
      .Times(1)
      .WillOnce(Return(ByMove(SyncServiceUP(sync_service))));

  EXPECT_CALL(*this, GetAdbClient(_))
      .Times(1)
      .WillOnce(Return(ByMove(AdbClientUP(adb_client))));

  EXPECT_TRUE(
      DownloadModuleSlice(FileSpec("/system/lib64/libc.so"), 0, 0, FileSpec())
          .Success());
}

TEST_F(PlatformAndroidTest, DownloadModuleSliceWithZipFile) {
  auto adb_client = new MockAdbClient();
  EXPECT_CALL(*adb_client,
              ShellToFile(StrEq("dd if='/system/app/Test/Test.apk' "
                                "iflag=skip_bytes,count_bytes "
                                "skip=4096 count=3600 status=none"),
                          _, _))
      .Times(1)
      .WillOnce(Return(Status()));

  EXPECT_CALL(*this, GetAdbClient(_))
      .Times(1)
      .WillOnce(Return(ByMove(AdbClientUP(adb_client))));

  EXPECT_TRUE(
      DownloadModuleSlice(
          FileSpec("/system/app/Test/Test.apk!/lib/arm64-v8a/libtest.so"), 4096,
          3600, FileSpec())
          .Success());
}

TEST_F(PlatformAndroidTest, DownloadModuleSliceWithZipFileAndRunAs) {
  auto adb_client = new MockAdbClient();
  EXPECT_CALL(*adb_client,
              ShellToFile(StrEq("run-as 'com.example.test' "
                                "dd if='/system/app/Test/Test.apk' "
                                "iflag=skip_bytes,count_bytes "
                                "skip=4096 count=3600 status=none"),
                          _, _))
      .Times(1)
      .WillOnce(Return(Status()));

  EXPECT_CALL(*this, GetPropertyPackageName())
      .Times(1)
      .WillOnce(Return(llvm::StringRef("com.example.test")));

  EXPECT_CALL(*this, GetAdbClient(_))
      .Times(1)
      .WillOnce(Return(ByMove(AdbClientUP(adb_client))));

  EXPECT_TRUE(
      DownloadModuleSlice(
          FileSpec("/system/app/Test/Test.apk!/lib/arm64-v8a/libtest.so"), 4096,
          3600, FileSpec())
          .Success());
}

TEST_F(PlatformAndroidTest, GetFileWithNormalFile) {
  auto sync_service = new MockSyncService();
  EXPECT_CALL(*sync_service, Stat(FileSpec("/data/local/tmp/test"), _, _, _))
      .Times(1)
      .WillOnce(DoAll(SetArgReferee<1>(1), Return(Status())));
  EXPECT_CALL(*sync_service, PullFile(FileSpec("/data/local/tmp/test"), _))
      .Times(1)
      .WillOnce(Return(Status()));

  auto adb_client = new MockAdbClient();
  EXPECT_CALL(*adb_client, GetSyncService(_))
      .Times(1)
      .WillOnce(Return(ByMove(SyncServiceUP(sync_service))));

  EXPECT_CALL(*this, GetAdbClient(_))
      .Times(1)
      .WillOnce(Return(ByMove(AdbClientUP(adb_client))));

  EXPECT_TRUE(GetFile(FileSpec("/data/local/tmp/test"), FileSpec()).Success());
}

TEST_F(PlatformAndroidTest, GetFileWithCatFallback) {
  auto sync_service = new MockSyncService();
  EXPECT_CALL(
      *sync_service,
      Stat(FileSpec("/data/data/com.example.app/lib-main/libtest.so"), _, _, _))
      .Times(1)
      .WillOnce(DoAll(SetArgReferee<1>(0), Return(Status())));

  auto adb_client0 = new MockAdbClient();
  EXPECT_CALL(*adb_client0, GetSyncService(_))
      .Times(1)
      .WillOnce(Return(ByMove(SyncServiceUP(sync_service))));

  auto adb_client1 = new MockAdbClient();
  EXPECT_CALL(
      *adb_client1,
      ShellToFile(StrEq("cat '/data/data/com.example.app/lib-main/libtest.so'"),
                  _, _))
      .Times(1)
      .WillOnce(Return(Status()));

  EXPECT_CALL(*this, GetAdbClient(_))
      .Times(2)
      .WillOnce(Return(ByMove(AdbClientUP(adb_client0))))
      .WillOnce(Return(ByMove(AdbClientUP(adb_client1))));

  EXPECT_TRUE(
      GetFile(FileSpec("/data/data/com.example.app/lib-main/libtest.so"),
              FileSpec())
          .Success());
}

TEST_F(PlatformAndroidTest, GetFileWithCatFallbackAndRunAs) {
  auto sync_service = new MockSyncService();
  EXPECT_CALL(
      *sync_service,
      Stat(FileSpec("/data/data/com.example.app/lib-main/libtest.so"), _, _, _))
      .Times(1)
      .WillOnce(DoAll(SetArgReferee<1>(0), Return(Status())));

  auto adb_client0 = new MockAdbClient();
  EXPECT_CALL(*adb_client0, GetSyncService(_))
      .Times(1)
      .WillOnce(Return(ByMove(SyncServiceUP(sync_service))));

  auto adb_client1 = new MockAdbClient();
  EXPECT_CALL(
      *adb_client1,
      ShellToFile(StrEq("run-as 'com.example.app' "
                        "cat '/data/data/com.example.app/lib-main/libtest.so'"),
                  _, _))
      .Times(1)
      .WillOnce(Return(Status()));

  EXPECT_CALL(*this, GetPropertyPackageName())
      .Times(1)
      .WillOnce(Return(llvm::StringRef("com.example.app")));

  EXPECT_CALL(*this, GetAdbClient(_))
      .Times(2)
      .WillOnce(Return(ByMove(AdbClientUP(adb_client0))))
      .WillOnce(Return(ByMove(AdbClientUP(adb_client1))));

  EXPECT_TRUE(
      GetFile(FileSpec("/data/data/com.example.app/lib-main/libtest.so"),
              FileSpec())
          .Success());
}
