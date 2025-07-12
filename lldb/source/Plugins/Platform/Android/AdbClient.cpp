//===-- AdbClient.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AdbClient.h"

#include "lldb/Host/posix/ConnectionFileDescriptorPosix.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <climits>
#include <cstdlib>
#include <sstream>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_android;
using namespace std::chrono;
using namespace adb_client_utils;

const static char *kSocketNamespaceAbstract = "localabstract";
const static char *kSocketNamespaceFileSystem = "localfilesystem";

Status AdbClient::ResolveDeviceID(const std::string &device_id,
                                  std::string &resolved_device_id) {
  Status error;
  llvm::StringRef preferred_serial;
  if (!device_id.empty()) {
    preferred_serial = device_id;
  } else if (const char *env_serial = std::getenv("ANDROID_SERIAL")) {
    preferred_serial = env_serial;
  }

  if (preferred_serial.empty()) {
    DeviceIDList connected_devices;
    {
      AdbClient temp_adb;
      error = temp_adb.GetDevices(connected_devices);
      // temp_adb's connection is closed after the GetDevices() call.
      // Make it go out of scope to avoid accidental reuse.
    }
    if (error.Fail())
      return error;

    if (connected_devices.size() != 1)
      return Status::FromErrorStringWithFormat(
          "Expected a single connected device, got instead %zu - try "
          "setting 'ANDROID_SERIAL'",
          connected_devices.size());
    
    resolved_device_id = std::move(connected_devices.front());
  } else {
    resolved_device_id = preferred_serial.str();
  }
  
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOGF(log, "AdbClient::ResolveDeviceID Resolved device ID: %s", 
            resolved_device_id.c_str());
  return error;
}


AdbClient::AdbClient(const std::string &device_id) : m_device_id(device_id) {
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOGF(log, "AdbClient::AdbClient(device_id='%s') - Creating AdbClient with device ID", 
              device_id.c_str());
  m_conn = std::make_unique<ConnectionFileDescriptor>();
  Connect();
}

AdbClient::AdbClient() {
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOGF(log, "AdbClient::AdbClient() - Creating AdbClient with default constructor");
  m_conn = std::make_unique<ConnectionFileDescriptor>();
  Connect();
}

AdbClient::~AdbClient() {
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOGF(log, "AdbClient::~AdbClient() - Destroying AdbClient for device: %s", 
              m_device_id.c_str());
}

const std::string &AdbClient::GetDeviceID() const { return m_device_id; }

Status AdbClient::Connect() {
  if (m_conn->IsConnected())
    return Status();

  return ConnectToAdb(*m_conn);
}

Status AdbClient::GetDevices(DeviceIDList &device_list) {
  device_list.clear();

  auto error = SendAdbMessage(*m_conn, "host:devices");
  if (error.Fail())
    return error;
  
  error = ReadResponseStatus(*m_conn);
  if (error.Fail())
  return error;

std::vector<char> in_buffer;
error = ReadAdbMessage(*m_conn, in_buffer);

llvm::StringRef response(&in_buffer[0], in_buffer.size());
llvm::SmallVector<llvm::StringRef, 4> devices;
response.split(devices, "\n", -1, false);

for (const auto &device : devices)
device_list.push_back(std::string(device.split('\t').first));

// WARNING: ADB closes the connection after host:devices response.
// This AdbClient instance is now INVALID and should not be used for any further operations.
// This method should ONLY be called from ResolveDeviceID() which uses a temporary AdbClient.
  return error;
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

Status AdbClient::SendDeviceMessage(const std::string &packet) {
  std::ostringstream msg;
  msg << "host-serial:" << m_device_id << ":" << packet;
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
