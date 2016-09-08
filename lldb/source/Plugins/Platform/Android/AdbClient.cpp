//===-- AdbClient.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Other libraries and framework includes
#include "lldb/Core/DataBuffer.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataEncoder.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/FileSpec.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileUtilities.h"

// Project includes
#include "AdbClient.h"

#include <limits.h>

#include <algorithm>
#include <fstream>
#include <sstream>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_android;

namespace {

const uint32_t kReadTimeout = 1000000; // 1 second
const char *kOKAY = "OKAY";
const char *kFAIL = "FAIL";
const char *kDATA = "DATA";
const char *kDONE = "DONE";

const char *kSEND = "SEND";
const char *kRECV = "RECV";
const char *kSTAT = "STAT";

const size_t kSyncPacketLen = 8;
// Maximum size of a filesync DATA packet.
const size_t kMaxPushData = 2 * 1024;
// Default mode for pushed files.
const uint32_t kDefaultMode = 0100770; // S_IFREG | S_IRWXU | S_IRWXG

const char *kSocketNamespaceAbstract = "localabstract";
const char *kSocketNamespaceFileSystem = "localfilesystem";

} // namespace

Error AdbClient::CreateByDeviceID(const std::string &device_id,
                                  AdbClient &adb) {
  DeviceIDList connect_devices;
  auto error = adb.GetDevices(connect_devices);
  if (error.Fail())
    return error;

  if (device_id.empty()) {
    if (connect_devices.size() != 1)
      return Error("Expected a single connected device, got instead %" PRIu64,
                   static_cast<uint64_t>(connect_devices.size()));

    adb.SetDeviceID(connect_devices.front());
  } else {
    auto find_it =
        std::find(connect_devices.begin(), connect_devices.end(), device_id);
    if (find_it == connect_devices.end())
      return Error("Device \"%s\" not found", device_id.c_str());

    adb.SetDeviceID(*find_it);
  }
  return error;
}

AdbClient::AdbClient(const std::string &device_id) : m_device_id(device_id) {}

void AdbClient::SetDeviceID(const std::string &device_id) {
  m_device_id = device_id;
}

const std::string &AdbClient::GetDeviceID() const { return m_device_id; }

Error AdbClient::Connect() {
  Error error;
  m_conn.Connect("connect://localhost:5037", &error);

  return error;
}

Error AdbClient::GetDevices(DeviceIDList &device_list) {
  device_list.clear();

  auto error = SendMessage("host:devices");
  if (error.Fail())
    return error;

  error = ReadResponseStatus();
  if (error.Fail())
    return error;

  std::vector<char> in_buffer;
  error = ReadMessage(in_buffer);

  llvm::StringRef response(&in_buffer[0], in_buffer.size());
  llvm::SmallVector<llvm::StringRef, 4> devices;
  response.split(devices, "\n", -1, false);

  for (const auto device : devices)
    device_list.push_back(device.split('\t').first);

  return error;
}

Error AdbClient::SetPortForwarding(const uint16_t local_port,
                                   const uint16_t remote_port) {
  char message[48];
  snprintf(message, sizeof(message), "forward:tcp:%d;tcp:%d", local_port,
           remote_port);

  const auto error = SendDeviceMessage(message);
  if (error.Fail())
    return error;

  return ReadResponseStatus();
}

Error AdbClient::SetPortForwarding(const uint16_t local_port,
                                   const char *remote_socket_name,
                                   const UnixSocketNamespace socket_namespace) {
  char message[PATH_MAX];
  const char *sock_namespace_str =
      (socket_namespace == UnixSocketNamespaceAbstract)
          ? kSocketNamespaceAbstract
          : kSocketNamespaceFileSystem;
  snprintf(message, sizeof(message), "forward:tcp:%d;%s:%s", local_port,
           sock_namespace_str, remote_socket_name);

  const auto error = SendDeviceMessage(message);
  if (error.Fail())
    return error;

  return ReadResponseStatus();
}

Error AdbClient::DeletePortForwarding(const uint16_t local_port) {
  char message[32];
  snprintf(message, sizeof(message), "killforward:tcp:%d", local_port);

  const auto error = SendDeviceMessage(message);
  if (error.Fail())
    return error;

  return ReadResponseStatus();
}

Error AdbClient::SendMessage(const std::string &packet, const bool reconnect) {
  Error error;
  if (reconnect) {
    error = Connect();
    if (error.Fail())
      return error;
  }

  char length_buffer[5];
  snprintf(length_buffer, sizeof(length_buffer), "%04x",
           static_cast<int>(packet.size()));

  ConnectionStatus status;

  m_conn.Write(length_buffer, 4, status, &error);
  if (error.Fail())
    return error;

  m_conn.Write(packet.c_str(), packet.size(), status, &error);
  return error;
}

Error AdbClient::SendDeviceMessage(const std::string &packet) {
  std::ostringstream msg;
  msg << "host-serial:" << m_device_id << ":" << packet;
  return SendMessage(msg.str());
}

Error AdbClient::ReadMessage(std::vector<char> &message) {
  message.clear();

  char buffer[5];
  buffer[4] = 0;

  auto error = ReadAllBytes(buffer, 4);
  if (error.Fail())
    return error;

  unsigned int packet_len = 0;
  sscanf(buffer, "%x", &packet_len);

  message.resize(packet_len, 0);
  error = ReadAllBytes(&message[0], packet_len);
  if (error.Fail())
    message.clear();

  return error;
}

Error AdbClient::ReadMessageStream(std::vector<char> &message,
                                   uint32_t timeout_ms) {
  auto start = std::chrono::steady_clock::now();
  message.clear();

  Error error;
  lldb::ConnectionStatus status = lldb::eConnectionStatusSuccess;
  char buffer[1024];
  while (error.Success() && status == lldb::eConnectionStatusSuccess) {
    auto end = std::chrono::steady_clock::now();
    uint32_t elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    if (elapsed_time >= timeout_ms)
      return Error("Timed out");

    size_t n = m_conn.Read(buffer, sizeof(buffer),
                           1000 * (timeout_ms - elapsed_time), status, &error);
    if (n > 0)
      message.insert(message.end(), &buffer[0], &buffer[n]);
  }
  return error;
}

Error AdbClient::ReadResponseStatus() {
  char response_id[5];

  static const size_t packet_len = 4;
  response_id[packet_len] = 0;

  auto error = ReadAllBytes(response_id, packet_len);
  if (error.Fail())
    return error;

  if (strncmp(response_id, kOKAY, packet_len) != 0)
    return GetResponseError(response_id);

  return error;
}

Error AdbClient::GetResponseError(const char *response_id) {
  if (strcmp(response_id, kFAIL) != 0)
    return Error("Got unexpected response id from adb: \"%s\"", response_id);

  std::vector<char> error_message;
  auto error = ReadMessage(error_message);
  if (error.Success())
    error.SetErrorString(
        std::string(&error_message[0], error_message.size()).c_str());

  return error;
}

Error AdbClient::SwitchDeviceTransport() {
  std::ostringstream msg;
  msg << "host:transport:" << m_device_id;

  auto error = SendMessage(msg.str());
  if (error.Fail())
    return error;

  return ReadResponseStatus();
}

Error AdbClient::PullFile(const FileSpec &remote_file,
                          const FileSpec &local_file) {
  auto error = StartSync();
  if (error.Fail())
    return error;

  const auto local_file_path = local_file.GetPath();
  llvm::FileRemover local_file_remover(local_file_path.c_str());

  std::ofstream dst(local_file_path, std::ios::out | std::ios::binary);
  if (!dst.is_open())
    return Error("Unable to open local file %s", local_file_path.c_str());

  const auto remote_file_path = remote_file.GetPath(false);
  error = SendSyncRequest(kRECV, remote_file_path.length(),
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

  local_file_remover.releaseFile();
  return error;
}

Error AdbClient::PushFile(const FileSpec &local_file,
                          const FileSpec &remote_file) {
  auto error = StartSync();
  if (error.Fail())
    return error;

  const auto local_file_path(local_file.GetPath());
  std::ifstream src(local_file_path.c_str(), std::ios::in | std::ios::binary);
  if (!src.is_open())
    return Error("Unable to open local file %s", local_file_path.c_str());

  std::stringstream file_description;
  file_description << remote_file.GetPath(false).c_str() << "," << kDefaultMode;
  std::string file_description_str = file_description.str();
  error = SendSyncRequest(kSEND, file_description_str.length(),
                          file_description_str.c_str());
  if (error.Fail())
    return error;

  char chunk[kMaxPushData];
  while (!src.eof() && !src.read(chunk, kMaxPushData).bad()) {
    size_t chunk_size = src.gcount();
    error = SendSyncRequest(kDATA, chunk_size, chunk);
    if (error.Fail())
      return Error("Failed to send file chunk: %s", error.AsCString());
  }
  error = SendSyncRequest(kDONE, local_file.GetModificationTime().seconds(),
                          nullptr);
  if (error.Fail())
    return error;

  std::string response_id;
  uint32_t data_len;
  error = ReadSyncHeader(response_id, data_len);
  if (error.Fail())
    return Error("Failed to read DONE response: %s", error.AsCString());
  if (response_id == kFAIL) {
    std::string error_message(data_len, 0);
    error = ReadAllBytes(&error_message[0], data_len);
    if (error.Fail())
      return Error("Failed to read DONE error message: %s", error.AsCString());
    return Error("Failed to push file: %s", error_message.c_str());
  } else if (response_id != kOKAY)
    return Error("Got unexpected DONE response: %s", response_id.c_str());

  // If there was an error reading the source file, finish the adb file
  // transfer first so that adb isn't expecting any more data.
  if (src.bad())
    return Error("Failed read on %s", local_file_path.c_str());
  return error;
}

Error AdbClient::StartSync() {
  auto error = SwitchDeviceTransport();
  if (error.Fail())
    return Error("Failed to switch to device transport: %s", error.AsCString());

  error = Sync();
  if (error.Fail())
    return Error("Sync failed: %s", error.AsCString());

  return error;
}

Error AdbClient::Sync() {
  auto error = SendMessage("sync:", false);
  if (error.Fail())
    return error;

  return ReadResponseStatus();
}

Error AdbClient::PullFileChunk(std::vector<char> &buffer, bool &eof) {
  buffer.clear();

  std::string response_id;
  uint32_t data_len;
  auto error = ReadSyncHeader(response_id, data_len);
  if (error.Fail())
    return error;

  if (response_id == kDATA) {
    buffer.resize(data_len, 0);
    error = ReadAllBytes(&buffer[0], data_len);
    if (error.Fail())
      buffer.clear();
  } else if (response_id == kDONE) {
    eof = true;
  } else if (response_id == kFAIL) {
    std::string error_message(data_len, 0);
    error = ReadAllBytes(&error_message[0], data_len);
    if (error.Fail())
      return Error("Failed to read pull error message: %s", error.AsCString());
    return Error("Failed to pull file: %s", error_message.c_str());
  } else
    return Error("Pull failed with unknown response: %s", response_id.c_str());

  return Error();
}

Error AdbClient::SendSyncRequest(const char *request_id,
                                 const uint32_t data_len, const void *data) {
  const DataBufferSP data_sp(new DataBufferHeap(kSyncPacketLen, 0));
  DataEncoder encoder(data_sp, eByteOrderLittle, sizeof(void *));
  auto offset = encoder.PutData(0, request_id, strlen(request_id));
  encoder.PutU32(offset, data_len);

  Error error;
  ConnectionStatus status;
  m_conn.Write(data_sp->GetBytes(), kSyncPacketLen, status, &error);
  if (error.Fail())
    return error;

  if (data)
    m_conn.Write(data, data_len, status, &error);
  return error;
}

Error AdbClient::ReadSyncHeader(std::string &response_id, uint32_t &data_len) {
  char buffer[kSyncPacketLen];

  auto error = ReadAllBytes(buffer, kSyncPacketLen);
  if (error.Success()) {
    response_id.assign(&buffer[0], 4);
    DataExtractor extractor(&buffer[4], 4, eByteOrderLittle, sizeof(void *));
    offset_t offset = 0;
    data_len = extractor.GetU32(&offset);
  }

  return error;
}

Error AdbClient::ReadAllBytes(void *buffer, size_t size) {
  Error error;
  ConnectionStatus status;
  char *read_buffer = static_cast<char *>(buffer);

  size_t tota_read_bytes = 0;
  while (tota_read_bytes < size) {
    auto read_bytes =
        m_conn.Read(read_buffer + tota_read_bytes, size - tota_read_bytes,
                    kReadTimeout, status, &error);
    if (error.Fail())
      return error;
    tota_read_bytes += read_bytes;
  }
  return error;
}

Error AdbClient::Stat(const FileSpec &remote_file, uint32_t &mode,
                      uint32_t &size, uint32_t &mtime) {
  auto error = StartSync();
  if (error.Fail())
    return error;

  const std::string remote_file_path(remote_file.GetPath(false));
  error = SendSyncRequest(kSTAT, remote_file_path.length(),
                          remote_file_path.c_str());
  if (error.Fail())
    return Error("Failed to send request: %s", error.AsCString());

  static const size_t stat_len = strlen(kSTAT);
  static const size_t response_len = stat_len + (sizeof(uint32_t) * 3);

  std::vector<char> buffer(response_len);
  error = ReadAllBytes(&buffer[0], buffer.size());
  if (error.Fail())
    return Error("Failed to read response: %s", error.AsCString());

  DataExtractor extractor(&buffer[0], buffer.size(), eByteOrderLittle,
                          sizeof(void *));
  offset_t offset = 0;

  const void *command = extractor.GetData(&offset, stat_len);
  if (!command)
    return Error("Failed to get response command");
  const char *command_str = static_cast<const char *>(command);
  if (strncmp(command_str, kSTAT, stat_len))
    return Error("Got invalid stat command: %s", command_str);

  mode = extractor.GetU32(&offset);
  size = extractor.GetU32(&offset);
  mtime = extractor.GetU32(&offset);
  return Error();
}

Error AdbClient::Shell(const char *command, uint32_t timeout_ms,
                       std::string *output) {
  auto error = SwitchDeviceTransport();
  if (error.Fail())
    return Error("Failed to switch to device transport: %s", error.AsCString());

  StreamString adb_command;
  adb_command.Printf("shell:%s", command);
  error = SendMessage(adb_command.GetData(), false);
  if (error.Fail())
    return error;

  error = ReadResponseStatus();
  if (error.Fail())
    return error;

  std::vector<char> in_buffer;
  error = ReadMessageStream(in_buffer, timeout_ms);
  if (error.Fail())
    return error;

  if (output)
    output->assign(in_buffer.begin(), in_buffer.end());
  return error;
}
