//===-- AdbSyncService.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AdbClientUtils.h"
#include "AdbSyncService.h"

#include "lldb/Host/posix/ConnectionFileDescriptorPosix.h"
#include "lldb/Utility/Connection.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileUtilities.h"

#include "lldb/Host/FileSystem.h"
#include "lldb/Utility/DataEncoder.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

#include <climits>

#include <cstdlib>
#include <fstream>
#include <sstream>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_android;
using namespace std::chrono;
using namespace adb_client_utils;


Status AdbSyncService::internalPullFile(const FileSpec &remote_file,
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

Status AdbSyncService::internalPushFile(const FileSpec &local_file,
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

Status AdbSyncService::internalStat(const FileSpec &remote_file,
                                            uint32_t &mode, uint32_t &size,
                                            uint32_t &mtime) {
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
  return executeCommand([this, &remote_file, &local_file]() {
    return internalPullFile(remote_file, local_file);
  });
}

Status AdbSyncService::PushFile(const FileSpec &local_file,
                                        const FileSpec &remote_file) {
  return executeCommand([this, &local_file, &remote_file]() {
    return internalPushFile(local_file, remote_file);
  });
}

Status AdbSyncService::Stat(const FileSpec &remote_file, uint32_t &mode,
                                    uint32_t &size, uint32_t &mtime) {
  return executeCommand([this, &remote_file, &mode, &size, &mtime]() {
    return internalStat(remote_file, mode, size, mtime);
  });
}

bool AdbSyncService::IsConnected() const {
  return m_conn && m_conn->IsConnected();
}

AdbSyncService::AdbSyncService(const std::string device_id)
    : m_device_id(device_id) {
  m_conn = std::make_unique<ConnectionFileDescriptor>();
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOGF(log, "AdbSyncService::AdbSyncService() - Creating AdbSyncService for device: %s", 
            m_device_id.c_str());  
}

Status
AdbSyncService::executeCommand(const std::function<Status()> &cmd) {
  Status error = cmd();
  return error;
}

AdbSyncService::~AdbSyncService() {
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOGF(log, "AdbSyncService::~AdbSyncService() - Destroying AdbSyncService for device: %s", 
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

Status AdbSyncService::PullFileChunk(std::vector<char> &buffer,
                                             bool &eof) {
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
