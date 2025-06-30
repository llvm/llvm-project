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

  class SyncService {
    friend class AdbClient;

  public:
    explicit SyncService(std::unique_ptr<Connection> conn, const std::string &device_id);
    
    virtual ~SyncService();

    virtual Status PullFile(const FileSpec &remote_file,
                            const FileSpec &local_file);

    virtual Status PushFile(const FileSpec &local_file, const FileSpec &remote_file);

    virtual Status Stat(const FileSpec &remote_file, uint32_t &mode,
                        uint32_t &size, uint32_t &mtime);

    virtual bool IsConnected() const;
    
    const std::string &GetDeviceId() const { return m_device_id; }

  protected:
    virtual Status SendSyncRequest(const char *request_id, const uint32_t data_len,
                                   const void *data);
    virtual Status ReadSyncHeader(std::string &response_id, uint32_t &data_len);
    virtual Status ReadAllBytes(void *buffer, size_t size);

  private:

    Status PullFileChunk(std::vector<char> &buffer, bool &eof);

    Status internalPullFile(const FileSpec &remote_file,
                            const FileSpec &local_file);

    Status internalPushFile(const FileSpec &local_file,
                            const FileSpec &remote_file);

    Status internalStat(const FileSpec &remote_file, uint32_t &mode,
                        uint32_t &size, uint32_t &mtime);

    Status executeCommand(const std::function<Status()> &cmd);

    // Internal connection setup methods
    Status SetupSyncConnection(const std::string &device_id);

    std::unique_ptr<Connection> m_conn;
    std::string m_device_id;
  };

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

  Status SelectTargetDevice();

  Status EnterSyncMode();

private:
  void SetDeviceID(const std::string &device_id);

  Status Connect();

  Status SendMessage(const std::string &packet, const bool reconnect = true);

  Status SendDeviceMessage(const std::string &packet);

  Status ReadMessage(std::vector<char> &message);

  Status ReadMessageStream(std::vector<char> &message,
                           std::chrono::milliseconds timeout);

  Status GetResponseError(const char *response_id);

  Status ReadResponseStatus();

  Status internalShell(const char *command, std::chrono::milliseconds timeout,
                       std::vector<char> &output_buf);

  Status ReadAllBytes(void *buffer, size_t size);

  Status ConnectToAdb(Connection &conn);

  std::string m_device_id;
  std::unique_ptr<Connection> m_conn;
};

} // namespace platform_android
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_ANDROID_ADBCLIENT_H
