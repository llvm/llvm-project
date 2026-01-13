//===-- AdbClient.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AdbClient.h"

#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Utility/Connection.h"
#include "lldb/Utility/DataEncoder.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/Timeout.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileUtilities.h"
#include <chrono>

#include <climits>
#include <cstdlib>
#include <fstream>
#include <sstream>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_android;
using namespace std::chrono;
using namespace llvm;

static const char *kSocketNamespaceAbstract = "localabstract";
static const char *kSocketNamespaceFileSystem = "localfilesystem";
const seconds kReadTimeout(20);
static const char *kOKAY = "OKAY";
static const char *kFAIL = "FAIL";
static const char *kDATA = "DATA";
static const char *kDONE = "DONE";
static const char *kSEND = "SEND";
static const char *kRECV = "RECV";
static const char *kSTAT = "STAT";
static const size_t kSyncPacketLen = 8;
static const size_t kMaxPushData = 2 * 1024;
static const uint32_t kDefaultMode = 0100770;

static Status ReadAllBytes(Connection &conn, void *buffer, size_t size) {
  Status error;
  ConnectionStatus status;
  char *read_buffer = static_cast<char *>(buffer);

  auto now = steady_clock::now();
  const auto deadline = now + kReadTimeout;
  size_t total_read_bytes = 0;
  while (total_read_bytes < size && now < deadline) {
    auto read_bytes =
        conn.Read(read_buffer + total_read_bytes, size - total_read_bytes,
                  duration_cast<microseconds>(deadline - now), status, &error);
    if (error.Fail())
      return error;
    total_read_bytes += read_bytes;
    if (status != eConnectionStatusSuccess)
      break;
    now = steady_clock::now();
  }
  if (total_read_bytes < size)
    error = Status::FromErrorStringWithFormat(
        "Unable to read requested number of bytes. Connection status: %d.",
        status);

  return error;
}

static Status ReadAdbMessage(Connection &conn, std::vector<char> &message) {
  message.clear();

  char buffer[5];
  buffer[4] = 0;

  auto error = ReadAllBytes(conn, buffer, 4);
  if (error.Fail())
    return error;

  unsigned int packet_len = 0;
  sscanf(buffer, "%x", &packet_len);

  message.resize(packet_len, 0);
  error = ReadAllBytes(conn, &message[0], packet_len);
  if (error.Fail())
    message.clear();

  return error;
}

static Status GetResponseError(Connection &conn, const char *response_id) {
  if (strcmp(response_id, kFAIL) != 0)
    return Status::FromErrorStringWithFormat(
        "Got unexpected response id from adb: \"%s\"", response_id);

  std::vector<char> error_message;
  auto error = ReadAdbMessage(conn, error_message);
  if (!error.Success())
    return error;

  std::string error_str(&error_message[0], error_message.size());
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOGF(log, "ADB error: %s", error_str.c_str());
  return Status(error_str);
}

static Status ReadResponseStatus(Connection &conn) {
  char response_id[5];

  const size_t packet_len = 4;
  response_id[packet_len] = 0;

  auto error = ReadAllBytes(conn, response_id, packet_len);
  if (error.Fail())
    return error;

  if (strncmp(response_id, kOKAY, packet_len) != 0)
    return GetResponseError(conn, response_id);

  return error;
}

static Status SendAdbMessage(Connection &conn, llvm::StringRef packet) {
  Status error;

  char length_buffer[5];
  snprintf(length_buffer, sizeof(length_buffer), "%04x",
           static_cast<int>(packet.size()));

  ConnectionStatus status;

  conn.Write(length_buffer, 4, status, &error);
  if (error.Fail())
    return error;

  conn.Write(packet.str().c_str(), packet.size(), status, &error);
  return error;
}

static Status ConnectToAdb(Connection &conn) {
  std::string port = "5037";
  if (const char *env_port = std::getenv("ANDROID_ADB_SERVER_PORT"))
    port = env_port;
  std::string uri = "connect://127.0.0.1:" + port;

  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOGF(log, "Connecting to ADB server at %s", uri.c_str());

  Status error;
  conn.Connect(uri.c_str(), &error);
  return error;
}

static Status EnterSyncMode(Connection &conn) {
  auto error = SendAdbMessage(conn, "sync:");
  if (error.Fail())
    return error;

  return ReadResponseStatus(conn);
}

static Status SelectTargetDevice(Connection &conn, llvm::StringRef device_id) {
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOG(log, "Selecting device: {0}", device_id);

  std::ostringstream msg;
  msg << "host:transport:" << device_id.str();

  auto error = SendAdbMessage(conn, msg.str());
  if (error.Fail())
    return error;

  return ReadResponseStatus(conn);
}

Expected<std::string> AdbClient::ResolveDeviceID(StringRef device_id) {
  StringRef preferred_serial;
  if (!device_id.empty())
    preferred_serial = device_id;
  else if (const char *env_serial = std::getenv("ANDROID_SERIAL"))
    preferred_serial = env_serial;

  if (preferred_serial.empty()) {
    DeviceIDList connected_devices;

    auto GetDevices = [](DeviceIDList &device_list) -> Status {
      device_list.clear();

      // Create temporary ADB client for this operation only
      auto temp_conn = std::make_unique<ConnectionFileDescriptor>();
      auto error = ConnectToAdb(*temp_conn);
      if (error.Fail())
        return error;

      // NOTE: ADB closes the connection after host:devices response.
      // The connection is no longer valid
      error = SendAdbMessage(*temp_conn, "host:devices");
      if (error.Fail())
        return error;

      error = ReadResponseStatus(*temp_conn);
      if (error.Fail())
        return error;

      std::vector<char> in_buffer;
      error = ReadAdbMessage(*temp_conn, in_buffer);

      StringRef response(&in_buffer[0], in_buffer.size());
      SmallVector<StringRef, 4> devices;
      response.split(devices, "\n", -1, false);

      for (const auto &device : devices)
        device_list.push_back(std::string(device.split('\t').first));
      return error;
    };

    Status error = GetDevices(connected_devices);
    if (error.Fail())
      return error.ToError();

    if (connected_devices.size() != 1)
      return createStringError(
          inconvertibleErrorCode(),
          "Expected a single connected device, got instead %zu - try "
          "setting 'ANDROID_SERIAL'",
          connected_devices.size());

    std::string resolved_device_id = std::move(connected_devices.front());
    Log *log = GetLog(LLDBLog::Platform);
    LLDB_LOGF(log, "AdbClient::ResolveDeviceID Resolved device ID: %s",
              resolved_device_id.c_str());
    return resolved_device_id;
  }

  std::string resolved_device_id = preferred_serial.str();
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOGF(log, "AdbClient::ResolveDeviceID Resolved device ID: %s",
            resolved_device_id.c_str());
  return resolved_device_id;
}

AdbClient::AdbClient(llvm::StringRef device_id) : m_device_id(device_id) {
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOGF(log,
            "AdbClient::AdbClient(device_id='%s') - Creating AdbClient with "
            "device ID",
            device_id.str().c_str());
  m_conn = std::make_unique<ConnectionFileDescriptor>();
  Connect();
}

AdbClient::AdbClient() {
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOGF(
      log,
      "AdbClient::AdbClient() - Creating AdbClient with default constructor");
  m_conn = std::make_unique<ConnectionFileDescriptor>();
  Connect();
}

AdbClient::~AdbClient() {
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOGF(log,
            "AdbClient::~AdbClient() - Destroying AdbClient for device: %s",
            m_device_id.c_str());
}

llvm::StringRef AdbClient::GetDeviceID() const { return m_device_id; }

Status AdbClient::Connect() {
  if (m_conn->IsConnected())
    return Status();

  return ConnectToAdb(*m_conn);
}

Status AdbClient::SetPortForwarding(const uint16_t local_port,
                                    const uint16_t remote_port) {
  char message[48];
  snprintf(message, sizeof(message), "forward:tcp:%d;tcp:%d", local_port,
           remote_port);

  Status error = SendDeviceMessage(message);
  if (error.Fail())
    return error;

  return ReadResponseStatus(*m_conn);
}

Status
AdbClient::SetPortForwarding(const uint16_t local_port,
                             llvm::StringRef remote_socket_name,
                             const UnixSocketNamespace socket_namespace) {
  char message[PATH_MAX];
  const char *sock_namespace_str =
      (socket_namespace == UnixSocketNamespaceAbstract)
          ? kSocketNamespaceAbstract
          : kSocketNamespaceFileSystem;
  snprintf(message, sizeof(message), "forward:tcp:%d;%s:%s", local_port,
           sock_namespace_str, remote_socket_name.str().c_str());

  Status error = SendDeviceMessage(message);
  if (error.Fail())
    return error;

  return ReadResponseStatus(*m_conn);
}

Status AdbClient::DeletePortForwarding(const uint16_t local_port) {
  char message[32];
  snprintf(message, sizeof(message), "killforward:tcp:%d", local_port);

  Status error = SendDeviceMessage(message);
  if (error.Fail())
    return error;

  return ReadResponseStatus(*m_conn);
}

Status AdbClient::SendDeviceMessage(llvm::StringRef packet) {
  std::ostringstream msg;
  msg << "host-serial:" << m_device_id << ":" << packet.str();
  return SendAdbMessage(*m_conn, msg.str());
}

Status AdbClient::ReadMessageStream(std::vector<char> &message,
                                    milliseconds timeout) {
  auto start = steady_clock::now();
  message.clear();

  if (!m_conn)
    return Status::FromErrorString("No connection available");

  Status error;
  lldb::ConnectionStatus status = lldb::eConnectionStatusSuccess;
  char buffer[1024];
  while (error.Success() && status == lldb::eConnectionStatusSuccess) {
    auto end = steady_clock::now();
    auto elapsed = end - start;
    if (elapsed >= timeout)
      return Status::FromErrorString("Timed out");

    size_t n = m_conn->Read(buffer, sizeof(buffer),
                            duration_cast<microseconds>(timeout - elapsed),
                            status, &error);
    if (n > 0)
      message.insert(message.end(), &buffer[0], &buffer[n]);
  }
  return error;
}

Status AdbClient::internalShell(const char *command, milliseconds timeout,
                                std::vector<char> &output_buf) {
  output_buf.clear();

  auto error = SelectTargetDevice(*m_conn, m_device_id);
  if (error.Fail())
    return Status::FromErrorStringWithFormat(
        "Failed to select target device: %s", error.AsCString());

  StreamString adb_command;
  adb_command.Printf("shell:%s", command);
  error = SendAdbMessage(*m_conn, std::string(adb_command.GetString()));
  if (error.Fail())
    return error;

  error = ReadResponseStatus(*m_conn);
  if (error.Fail())
    return error;

  error = ReadMessageStream(output_buf, timeout);
  if (error.Fail())
    return error;

  // ADB doesn't propagate return code of shell execution - if
  // output starts with /system/bin/sh: most likely command failed.
  static const char *kShellPrefix = "/system/bin/sh:";
  if (output_buf.size() > strlen(kShellPrefix)) {
    if (!memcmp(&output_buf[0], kShellPrefix, strlen(kShellPrefix)))
      return Status::FromErrorStringWithFormat(
          "Shell command %s failed: %s", command,
          std::string(output_buf.begin(), output_buf.end()).c_str());
  }

  return Status();
}

Status AdbClient::Shell(const char *command, milliseconds timeout,
                        std::string *output) {
  std::vector<char> output_buffer;
  auto error = internalShell(command, timeout, output_buffer);
  if (error.Fail())
    return error;

  if (output)
    output->assign(output_buffer.begin(), output_buffer.end());
  return error;
}

Status AdbClient::ShellToFile(const char *command, milliseconds timeout,
                              const FileSpec &output_file_spec) {
  std::vector<char> output_buffer;
  auto error = internalShell(command, timeout, output_buffer);
  if (error.Fail())
    return error;

  const auto output_filename = output_file_spec.GetPath();
  std::error_code EC;
  llvm::raw_fd_ostream dst(output_filename, EC, llvm::sys::fs::OF_None);
  if (EC)
    return Status::FromErrorStringWithFormat("Unable to open local file %s",
                                             output_filename.c_str());

  dst.write(&output_buffer[0], output_buffer.size());
  dst.close();
  if (dst.has_error())
    return Status::FromErrorStringWithFormat("Failed to write file %s",
                                             output_filename.c_str());
  return Status();
}

Status AdbSyncService::PullFileImpl(const FileSpec &remote_file,
                                    const FileSpec &local_file) {
  const auto local_file_path = local_file.GetPath();
  llvm::FileRemover local_file_remover(local_file_path);

  std::error_code EC;
  llvm::raw_fd_ostream dst(local_file_path, EC, llvm::sys::fs::OF_None);
  if (EC)
    return Status::FromErrorStringWithFormat("Unable to open local file %s",
                                             local_file_path.c_str());

  const auto remote_file_path = remote_file.GetPath(false);
  auto error = SendSyncRequest(kRECV, remote_file_path.length(),
                               remote_file_path.c_str());
  if (error.Fail())
    return error;

  std::vector<char> chunk;
  bool eof = false;
  while (!eof) {
    error = PullFileChunk(chunk, eof);
    if (error.Fail())
      return error;
    if (!eof)
      dst.write(&chunk[0], chunk.size());
  }
  dst.close();
  if (dst.has_error())
    return Status::FromErrorStringWithFormat("Failed to write file %s",
                                             local_file_path.c_str());

  local_file_remover.releaseFile();
  return error;
}

Status AdbSyncService::PushFileImpl(const FileSpec &local_file,
                                    const FileSpec &remote_file) {
  const auto local_file_path(local_file.GetPath());
  std::ifstream src(local_file_path.c_str(), std::ios::in | std::ios::binary);
  if (!src.is_open())
    return Status::FromErrorStringWithFormat("Unable to open local file %s",
                                             local_file_path.c_str());

  std::stringstream file_description;
  file_description << remote_file.GetPath(false).c_str() << "," << kDefaultMode;
  std::string file_description_str = file_description.str();
  auto error = SendSyncRequest(kSEND, file_description_str.length(),
                               file_description_str.c_str());
  if (error.Fail())
    return error;

  char chunk[kMaxPushData];
  while (!src.eof() && !src.read(chunk, kMaxPushData).bad()) {
    size_t chunk_size = src.gcount();
    error = SendSyncRequest(kDATA, chunk_size, chunk);
    if (error.Fail())
      return Status::FromErrorStringWithFormat("Failed to send file chunk: %s",
                                               error.AsCString());
  }
  error = SendSyncRequest(
      kDONE,
      llvm::sys::toTimeT(
          FileSystem::Instance().GetModificationTime(local_file)),
      nullptr);
  if (error.Fail())
    return error;

  std::string response_id;
  uint32_t data_len;
  error = ReadSyncHeader(response_id, data_len);
  if (error.Fail())
    return Status::FromErrorStringWithFormat("Failed to read DONE response: %s",
                                             error.AsCString());
  if (response_id == kFAIL) {
    std::string error_message(data_len, 0);
    error = ReadAllBytes(*m_conn, &error_message[0], data_len);
    if (error.Fail())
      return Status::FromErrorStringWithFormat(
          "Failed to read DONE error message: %s", error.AsCString());
    return Status::FromErrorStringWithFormat("Failed to push file: %s",
                                             error_message.c_str());
  } else if (response_id != kOKAY)
    return Status::FromErrorStringWithFormat("Got unexpected DONE response: %s",
                                             response_id.c_str());

  // If there was an error reading the source file, finish the adb file
  // transfer first so that adb isn't expecting any more data.
  if (src.bad())
    return Status::FromErrorStringWithFormat("Failed read on %s",
                                             local_file_path.c_str());
  return error;
}

Status AdbSyncService::StatImpl(const FileSpec &remote_file, uint32_t &mode,
                                uint32_t &size, uint32_t &mtime) {
  const std::string remote_file_path(remote_file.GetPath(false));
  auto error = SendSyncRequest(kSTAT, remote_file_path.length(),
                               remote_file_path.c_str());
  if (error.Fail())
    return Status::FromErrorStringWithFormat("Failed to send request: %s",
                                             error.AsCString());

  static const size_t stat_len = strlen(kSTAT);
  static const size_t response_len = stat_len + (sizeof(uint32_t) * 3);

  std::vector<char> buffer(response_len);
  error = ReadAllBytes(*m_conn, &buffer[0], buffer.size());
  if (error.Fail())
    return Status::FromErrorStringWithFormat("Failed to read response: %s",
                                             error.AsCString());

  DataExtractor extractor(&buffer[0], buffer.size(), eByteOrderLittle,
                          sizeof(void *));
  offset_t offset = 0;

  const void *command = extractor.GetData(&offset, stat_len);
  if (!command)
    return Status::FromErrorStringWithFormat("Failed to get response command");
  const char *command_str = static_cast<const char *>(command);
  if (strncmp(command_str, kSTAT, stat_len))
    return Status::FromErrorStringWithFormat("Got invalid stat command: %s",
                                             command_str);

  mode = extractor.GetU32(&offset);
  size = extractor.GetU32(&offset);
  mtime = extractor.GetU32(&offset);
  return Status();
}

Status AdbSyncService::PullFile(const FileSpec &remote_file,
                                const FileSpec &local_file) {
  return ExecuteCommand([this, &remote_file, &local_file]() {
    return PullFileImpl(remote_file, local_file);
  });
}

Status AdbSyncService::PushFile(const FileSpec &local_file,
                                const FileSpec &remote_file) {
  return ExecuteCommand([this, &local_file, &remote_file]() {
    return PushFileImpl(local_file, remote_file);
  });
}

Status AdbSyncService::Stat(const FileSpec &remote_file, uint32_t &mode,
                            uint32_t &size, uint32_t &mtime) {
  return ExecuteCommand([this, &remote_file, &mode, &size, &mtime]() {
    return StatImpl(remote_file, mode, size, mtime);
  });
}

bool AdbSyncService::IsConnected() const {
  return m_conn && m_conn->IsConnected();
}

AdbSyncService::AdbSyncService(const std::string device_id)
    : m_device_id(device_id) {
  m_conn = std::make_unique<ConnectionFileDescriptor>();
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOGF(log,
            "AdbSyncService::AdbSyncService() - Creating AdbSyncService for "
            "device: %s",
            m_device_id.c_str());
}

Status AdbSyncService::ExecuteCommand(const std::function<Status()> &cmd) {
  Status error = cmd();
  return error;
}

AdbSyncService::~AdbSyncService() {
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOGF(log,
            "AdbSyncService::~AdbSyncService() - Destroying AdbSyncService for "
            "device: %s",
            m_device_id.c_str());
}

Status AdbSyncService::SendSyncRequest(const char *request_id,
                                       const uint32_t data_len,
                                       const void *data) {
  DataEncoder encoder(eByteOrderLittle, sizeof(void *));
  encoder.AppendData(llvm::StringRef(request_id));
  encoder.AppendU32(data_len);
  llvm::ArrayRef<uint8_t> bytes = encoder.GetData();
  Status error;
  ConnectionStatus status;
  m_conn->Write(bytes.data(), kSyncPacketLen, status, &error);
  if (error.Fail())
    return error;

  if (data)
    m_conn->Write(data, data_len, status, &error);
  return error;
}

Status AdbSyncService::ReadSyncHeader(std::string &response_id,
                                      uint32_t &data_len) {
  char buffer[kSyncPacketLen];

  auto error = ReadAllBytes(*m_conn, buffer, kSyncPacketLen);
  if (error.Success()) {
    response_id.assign(&buffer[0], 4);
    DataExtractor extractor(&buffer[4], 4, eByteOrderLittle, sizeof(void *));
    offset_t offset = 0;
    data_len = extractor.GetU32(&offset);
  }

  return error;
}

Status AdbSyncService::PullFileChunk(std::vector<char> &buffer, bool &eof) {
  buffer.clear();

  std::string response_id;
  uint32_t data_len;
  auto error = ReadSyncHeader(response_id, data_len);
  if (error.Fail())
    return error;

  if (response_id == kDATA) {
    buffer.resize(data_len, 0);
    error = ReadAllBytes(*m_conn, &buffer[0], data_len);
    if (error.Fail())
      buffer.clear();
  } else if (response_id == kDONE) {
    eof = true;
  } else if (response_id == kFAIL) {
    std::string error_message(data_len, 0);
    error = ReadAllBytes(*m_conn, &error_message[0], data_len);
    if (error.Fail())
      return Status::FromErrorStringWithFormat(
          "Failed to read pull error message: %s", error.AsCString());
    return Status::FromErrorStringWithFormat("Failed to pull file: %s",
                                             error_message.c_str());
  } else
    return Status::FromErrorStringWithFormat(
        "Pull failed with unknown response: %s", response_id.c_str());

  return Status();
}

Status AdbSyncService::SetupSyncConnection() {
  Status error = ConnectToAdb(*m_conn);
  if (error.Fail())
    return error;

  error = SelectTargetDevice(*m_conn, m_device_id);
  if (error.Fail())
    return error;

  error = EnterSyncMode(*m_conn);
  return error;
}
