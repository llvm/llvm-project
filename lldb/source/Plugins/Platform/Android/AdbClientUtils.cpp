//===-- AdbClientUtils.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "lldb/Utility/Connection.h"
#include "AdbClientUtils.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Timeout.h"
#include <chrono>
#include <cstdlib>
#include <sstream>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_android;
using namespace std::chrono;

namespace lldb_private {
namespace platform_android {
namespace adb_client_utils {

Status ReadAllBytes(Connection &conn, void *buffer, size_t size) {
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
  if (total_read_bytes < size) {
    error = Status::FromErrorStringWithFormat(
        "Unable to read requested number of bytes. Connection status: %d.",
        status);
  }
  return error;
}

Status SendAdbMessage(Connection &conn, const std::string &packet) {
  Status error;

  char length_buffer[5];
  snprintf(length_buffer, sizeof(length_buffer), "%04x",
           static_cast<int>(packet.size()));

  ConnectionStatus status;

  conn.Write(length_buffer, 4, status, &error);
  if (error.Fail())
    return error;

  conn.Write(packet.c_str(), packet.size(), status, &error);
  return error;
}

Status GetResponseError(Connection &conn, const char *response_id) {
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

Status ConnectToAdb(Connection &conn) {
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

Status EnterSyncMode(Connection &conn) {
  auto error = SendAdbMessage(conn, "sync:");
  if (error.Fail())
    return error;

  return ReadResponseStatus(conn);
}

Status SelectTargetDevice(Connection &conn, const std::string &device_id) {
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOGF(log, "Selecting device: %s", device_id.c_str());
    
  std::ostringstream msg;
  msg << "host:transport:" << device_id;

  auto error = SendAdbMessage(conn, msg.str());
  if (error.Fail())
    return error;

  return ReadResponseStatus(conn);
}

Status ReadAdbMessage(Connection &conn, std::vector<char> &message) {
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

Status ReadResponseStatus(Connection &conn) {
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

} // namespace adb_client_utils
} // namespace platform_android
} // namespace lldb_private
