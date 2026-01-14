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
#include "llvm/Support/Error.h"
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
  ///
  /// \returns Expected<std::string> containing the resolved device ID on
  ///          success, or an Error if the device ID cannot be resolved or
  ///          is ambiguous.
  static llvm::Expected<std::string> ResolveDeviceID(llvm::StringRef device_id);

  AdbClient();
  explicit AdbClient(llvm::StringRef device_id);

  virtual ~AdbClient();

  llvm::StringRef GetDeviceID() const;

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
  Status SendDeviceMessage(llvm::StringRef packet);

  Status ReadMessageStream(std::vector<char> &message,
                           std::chrono::milliseconds timeout);

  Status internalShell(const char *command, std::chrono::milliseconds timeout,
                       std::vector<char> &output_buf);

  std::string m_device_id;
  std::unique_ptr<Connection> m_conn;
};

class AdbSyncService {
public:
  explicit AdbSyncService(const std::string device_id);
  virtual ~AdbSyncService();
  Status SetupSyncConnection();

  virtual Status PullFile(const FileSpec &remote_file,
                          const FileSpec &local_file);
  virtual Status PushFile(const FileSpec &local_file,
                          const FileSpec &remote_file);
  virtual Status Stat(const FileSpec &remote_file, uint32_t &mode,
                      uint32_t &size, uint32_t &mtime);
  virtual bool IsConnected() const;

  llvm::StringRef GetDeviceId() const { return m_device_id; }

private:
  Status SendSyncRequest(const char *request_id, const uint32_t data_len,
                         const void *data);
  Status ReadSyncHeader(std::string &response_id, uint32_t &data_len);
  Status PullFileChunk(std::vector<char> &buffer, bool &eof);
  Status PullFileImpl(const FileSpec &remote_file, const FileSpec &local_file);
  Status PushFileImpl(const FileSpec &local_file, const FileSpec &remote_file);
  Status StatImpl(const FileSpec &remote_file, uint32_t &mode, uint32_t &size,
                  uint32_t &mtime);
  Status ExecuteCommand(const std::function<Status()> &cmd);

  std::unique_ptr<Connection> m_conn;
  std::string m_device_id;
};

} // namespace platform_android
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_ANDROID_ADBCLIENT_H
