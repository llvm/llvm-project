//===-- FifoFiles.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FifoFiles.h"
#include "JSONUtils.h"

#if !defined(_WIN32)
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include <chrono>
#include <fstream>
#include <future>
#include <optional>

using namespace llvm;

namespace lldb_dap {

#if defined(_WIN32)
FifoFile::FifoFile(StringRef path, HANDLE handle, bool is_server)
    : m_path(path), m_is_server(is_server), m_pipe_fd(handle) {}
#else
FifoFile::FifoFile(StringRef path, bool is_server)
    : m_path(path), m_is_server(is_server) {}
#endif

FifoFile::~FifoFile() {
#if defined(_WIN32)
  if (m_pipe_fd == INVALID_HANDLE_VALUE)
    return;
  if (m_is_server)
    DisconnectNamedPipe(m_pipe_fd);
  CloseHandle(m_pipe_fd);
#else
  if (m_is_server)
    unlink(m_path.c_str());
#endif
}

Expected<std::shared_ptr<FifoFile>> CreateFifoFile(StringRef path,
                                                   bool is_server) {
#if defined(_WIN32)
  if (!is_server)
    return std::make_shared<FifoFile>(path, INVALID_HANDLE_VALUE, is_server);
  HANDLE handle =
      CreateNamedPipeA(path.data(), PIPE_ACCESS_DUPLEX,
                       PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, 1,
                       1024 * 16, 1024 * 16, 0, NULL);
  if (handle == INVALID_HANDLE_VALUE)
    return createStringError(
        std::error_code(GetLastError(), std::generic_category()),
        "Couldn't create fifo file: %s", path.data());
  return std::make_shared<FifoFile>(path, handle, is_server);
#else
  if (!is_server)
    return std::make_shared<FifoFile>(path, is_server);
  if (int err = mkfifo(path.data(), 0600))
    return createStringError(std::error_code(err, std::generic_category()),
                             "Couldn't create fifo file: %s", path.data());
  return std::make_shared<FifoFile>(path, is_server);
#endif
}

FifoFileIO::FifoFileIO(std::shared_ptr<FifoFile> fifo_file,
                       StringRef other_endpoint_name)
    : m_fifo_file(fifo_file), m_other_endpoint_name(other_endpoint_name) {}

Expected<json::Value> FifoFileIO::ReadJSON(std::chrono::milliseconds timeout) {
  // We use a pointer for this future, because otherwise its normal destructor
  // would wait for the getline to end, rendering the timeout useless.
  std::optional<std::string> line;
  std::future<void> *future =
      new std::future<void>(std::async(std::launch::async, [&]() {
#if defined(_WIN32)
        std::string buffer;
        buffer.reserve(4096);
        char ch;
        DWORD bytes_read = 0;
        while (ReadFile(m_fifo_file->m_pipe_fd, &ch, 1, &bytes_read, NULL) &&
               (bytes_read == 1)) {
          buffer.push_back(ch);
          if (ch == '\n') {
            break;
          }
        }
        if (!buffer.empty())
          line = std::move(buffer);
#else
        std::ifstream reader(m_fifo_file->m_path, std::ifstream::in);
        std::string buffer;
        std::getline(reader, buffer);
        if (!buffer.empty())
          line = buffer;
#endif
      }));
  if (future->wait_for(timeout) == std::future_status::timeout || !line)
    // Indeed this is a leak, but it's intentional. "future" obj destructor
    //  will block on waiting for the worker thread to join. And the worker
    //  thread might be stuck in blocking I/O. Intentionally leaking the  obj
    //  as a hack to avoid blocking main thread, and adding annotation to
    //  supress static code inspection warnings

    // coverity[leaked_storage]
    return createStringError(inconvertibleErrorCode(),
                             "Timed out trying to get messages from the " +
                                 m_other_endpoint_name);
  delete future;
  return json::parse(*line);
}

Error FifoFileIO::SendJSON(const json::Value &json,
                           std::chrono::milliseconds timeout) {
  bool done = false;
  std::future<void> *future =
      new std::future<void>(std::async(std::launch::async, [&]() {
#if defined(_WIN32)
        std::string buffer = JSONToString(json);
        buffer.append("\n");
        DWORD bytes_write = 0;
        WriteFile(m_fifo_file->m_pipe_fd, buffer.c_str(), buffer.size(),
                  &bytes_write, NULL);
        done = bytes_write == buffer.size();
#else
        std::ofstream writer(m_fifo_file->m_path, std::ofstream::out);
        writer << JSONToString(json) << std::endl;
        done = true;
#endif
      }));
  if (future->wait_for(timeout) == std::future_status::timeout || !done) {
    // Indeed this is a leak, but it's intentional. "future" obj destructor will
    // block on waiting for the worker thread to join. And the worker thread
    // might be stuck in blocking I/O. Intentionally leaking the  obj as a hack
    // to avoid blocking main thread, and adding annotation to supress static
    // code inspection warnings"

    // coverity[leaked_storage]
    return createStringError(inconvertibleErrorCode(),
                             "Timed out trying to send messages to the " +
                                 m_other_endpoint_name);
  }
  delete future;
  return Error::success();
}

#if defined(_WIN32)
bool FifoFileIO::Connect() {
  if (m_fifo_file->m_is_server) {
    return ConnectNamedPipe(m_fifo_file->m_pipe_fd, NULL);
  }
  if (!WaitNamedPipeA(m_fifo_file->m_path.c_str(), NMPWAIT_WAIT_FOREVER))
    return false;
  m_fifo_file->m_pipe_fd =
      CreateFileA(m_fifo_file->m_path.c_str(), GENERIC_READ | GENERIC_WRITE, 0,
                  NULL, OPEN_EXISTING, 0, NULL);
  if (m_fifo_file->m_pipe_fd == INVALID_HANDLE_VALUE)
    return false;
  return true;
}
#endif

} // namespace lldb_dap
