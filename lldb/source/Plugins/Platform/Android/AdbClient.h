//===-- AdbClient.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_AdbClient_h_
#define liblldb_AdbClient_h_

#include "lldb/Utility/Error.h"
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
    ~SyncService();

    Error PullFile(const FileSpec &remote_file, const FileSpec &local_file);

    Error PushFile(const FileSpec &local_file, const FileSpec &remote_file);

    Error Stat(const FileSpec &remote_file, uint32_t &mode, uint32_t &size,
               uint32_t &mtime);

    bool IsConnected() const;

  private:
    explicit SyncService(std::unique_ptr<Connection> &&conn);

    Error SendSyncRequest(const char *request_id, const uint32_t data_len,
                          const void *data);

    Error ReadSyncHeader(std::string &response_id, uint32_t &data_len);

    Error PullFileChunk(std::vector<char> &buffer, bool &eof);

    Error ReadAllBytes(void *buffer, size_t size);

    Error internalPullFile(const FileSpec &remote_file,
                           const FileSpec &local_file);

    Error internalPushFile(const FileSpec &local_file,
                           const FileSpec &remote_file);

    Error internalStat(const FileSpec &remote_file, uint32_t &mode,
                       uint32_t &size, uint32_t &mtime);

    Error executeCommand(const std::function<Error()> &cmd);

    std::unique_ptr<Connection> m_conn;
  };

  static Error CreateByDeviceID(const std::string &device_id, AdbClient &adb);

  AdbClient();
  explicit AdbClient(const std::string &device_id);

  ~AdbClient();

  const std::string &GetDeviceID() const;

  Error GetDevices(DeviceIDList &device_list);

  Error SetPortForwarding(const uint16_t local_port,
                          const uint16_t remote_port);

  Error SetPortForwarding(const uint16_t local_port,
                          llvm::StringRef remote_socket_name,
                          const UnixSocketNamespace socket_namespace);

  Error DeletePortForwarding(const uint16_t local_port);

  Error Shell(const char *command, std::chrono::milliseconds timeout,
              std::string *output);

  Error ShellToFile(const char *command, std::chrono::milliseconds timeout,
                    const FileSpec &output_file_spec);

  std::unique_ptr<SyncService> GetSyncService(Error &error);

  Error SwitchDeviceTransport();

private:
  Error Connect();

  void SetDeviceID(const std::string &device_id);

  Error SendMessage(const std::string &packet, const bool reconnect = true);

  Error SendDeviceMessage(const std::string &packet);

  Error ReadMessage(std::vector<char> &message);

  Error ReadMessageStream(std::vector<char> &message, std::chrono::milliseconds timeout);

  Error GetResponseError(const char *response_id);

  Error ReadResponseStatus();

  Error Sync();

  Error StartSync();

  Error internalShell(const char *command, std::chrono::milliseconds timeout,
                      std::vector<char> &output_buf);

  Error ReadAllBytes(void *buffer, size_t size);

  std::string m_device_id;
  std::unique_ptr<Connection> m_conn;
};

} // namespace platform_android
} // namespace lldb_private

#endif // liblldb_AdbClient_h_
