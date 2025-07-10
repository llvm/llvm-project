//===-- AdbClient.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_ANDROID_ADBCLIENT_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_ANDROID_ADBCLIENT_H

#include "lldb/Utility/Status.h"
#include "AdbClientUtils.h"
#include "AdbSyncService.h"
#include <chrono>
#include <functional>
#include <list>
#include <memory>
#include <string>
#include <vector>

namespace lldb_private {

class FileSpec;

namespace platform_android {

class AdbClient {
public:
  enum UnixSocketNamespace {
    UnixSocketNamespaceAbstract,
    UnixSocketNamespaceFileSystem,
  };

  using DeviceIDList = std::list<std::string>;

  static Status CreateByDeviceID(const std::string &device_id, AdbClient &adb);

  AdbClient();
  explicit AdbClient(const std::string &device_id);

  virtual ~AdbClient();

  const std::string &GetDeviceID() const;

  Status GetDevices(DeviceIDList &device_list);

  Status SetPortForwarding(const uint16_t local_port,
                           const uint16_t remote_port);

  Status SetPortForwarding(const uint16_t local_port,
                           llvm::StringRef remote_socket_name,
                           const UnixSocketNamespace socket_namespace);

  Status DeletePortForwarding(const uint16_t local_port);

  Status Shell(const char *command, std::chrono::milliseconds timeout,
               std::string *output);

  virtual Status ShellToFile(const char *command,
                             std::chrono::milliseconds timeout,
                             const FileSpec &output_file_spec);

  Status Connect();

private:
  void SetDeviceID(const std::string &device_id);

  Status SendDeviceMessage(const std::string &packet);

  Status ReadMessageStream(std::vector<char> &message,
                           std::chrono::milliseconds timeout);

  Status internalShell(const char *command, std::chrono::milliseconds timeout,
                       std::vector<char> &output_buf);

  std::string m_device_id;
  std::unique_ptr<Connection> m_conn;
};

} // namespace platform_android
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_ANDROID_ADBCLIENT_H
