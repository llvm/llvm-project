//===-- AdbClient.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_ANDROID_ADBSYNCSERVICE_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_ANDROID_ADBSYNCSERVICE_H

#include "lldb/Utility/Status.h"
#include <memory>
#include <string>

namespace lldb_private {
class FileSpec;
class Connection;

namespace platform_android {

class AdbSyncService {
public:
  explicit AdbSyncService(const std::string device_id);
  virtual ~AdbSyncService();
  Status SetupSyncConnection();

  virtual Status PullFile(const FileSpec &remote_file, const FileSpec &local_file);
  virtual Status PushFile(const FileSpec &local_file, const FileSpec &remote_file);
  virtual Status Stat(const FileSpec &remote_file, uint32_t &mode, uint32_t &size, uint32_t &mtime);
  virtual bool IsConnected() const;
  
  const std::string &GetDeviceId() const { return m_device_id; }
private:
  Status SendSyncRequest(const char *request_id, const uint32_t data_len, const void *data);
  Status ReadSyncHeader(std::string &response_id, uint32_t &data_len);
  Status PullFileChunk(std::vector<char> &buffer, bool &eof);
  Status internalPullFile(const FileSpec &remote_file, const FileSpec &local_file);
  Status internalPushFile(const FileSpec &local_file, const FileSpec &remote_file);
  Status internalStat(const FileSpec &remote_file, uint32_t &mode, uint32_t &size, uint32_t &mtime);
  Status executeCommand(const std::function<Status()> &cmd);

  std::unique_ptr<Connection> m_conn;
  std::string m_device_id;
};

} // namespace platform_android
} // namespace lldb_private

#endif
