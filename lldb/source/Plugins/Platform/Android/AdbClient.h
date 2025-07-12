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

  /// Resolves a device identifier to its canonical form.
  ///
  /// \param device_id the device identifier to resolve (may be empty).
  /// \param [out] resolved_device_id filled with the canonical device ID.
  ///
  /// \returns Status object indicating success or failure. Returns error if
  ///          the device ID cannot be resolved or is ambiguous.
  static Status ResolveDeviceID(const std::string &device_id, std::string &resolved_device_id);

  AdbClient();
  explicit AdbClient(const std::string &device_id);

  virtual ~AdbClient();

  const std::string &GetDeviceID() const;

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
  /// Retrieves a list of all connected Android devices.
  ///
  /// Queries the ADB server for all currently connected devices and populates
  /// the provided list with their device IDs. Note that ADB closes the connection
  /// after this operation, making this AdbClient instance invalid for further use.
  ///
  /// \param [out] device_list filled with device IDs of all connected devices.
  Status GetDevices(DeviceIDList &device_list);

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
