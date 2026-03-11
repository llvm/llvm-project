//===-- FifoFiles.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FifoFiles.h"
#include "JSONUtils.h"

#ifdef _WIN32
#include "lldb/Host/windows/PipeWindows.h"
#include "lldb/Host/windows/windows.h"
#include "llvm/Support/Path.h"
#else
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

FifoFile::FifoFile(StringRef path, lldb::pipe_t pipe) : m_path(path) {
#ifdef _WIN32
  if (pipe == INVALID_HANDLE_VALUE)
    pipe = CreateFileA(m_path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, NULL,
                       OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
#endif
  m_pipe = pipe;
}

FifoFile::~FifoFile() {
#ifdef _WIN32
  if (m_pipe != INVALID_HANDLE_VALUE) {
    DisconnectNamedPipe(m_pipe);
    CloseHandle(m_pipe);
  }
#else
  unlink(m_path.c_str());
#endif
}

void FifoFile::WriteLine(std::string line) {
#ifdef _WIN32
  DWORD written;
  line += "\n";
  WriteFile(m_pipe, line.c_str(), static_cast<DWORD>(line.size()), &written,
            NULL);
  FlushFileBuffers(m_pipe);
#else
  std::ofstream writer(m_path, std::ofstream::out);
  writer << line << std::endl;
#endif
}

void FifoFile::Connect() {
#ifdef _WIN32
  ConnectNamedPipe(m_pipe, NULL);
#endif
}

std::string FifoFile::ReadLine() {
#ifdef _WIN32
  std::string buffer;
  char read_buffer[4096];
  DWORD bytes_read;

  if (ReadFile(m_pipe, read_buffer, sizeof(read_buffer) - 1, &bytes_read,
               NULL) &&
      bytes_read > 0) {
    read_buffer[bytes_read] = '\0';
    buffer = read_buffer;
  }

  return buffer;
#else
  std::ifstream reader(m_path, std::ifstream::in);
  std::string buffer;
  std::getline(reader, buffer);
  return buffer;
#endif
}

Expected<std::shared_ptr<FifoFile>> CreateFifoFile(StringRef path) {
#if defined(_WIN32)
  assert(path.starts_with("\\\\.\\pipe\\") &&
         "FifoFile path should start with '\\\\.\\pipe\\'");
  HANDLE pipe_handle =
      CreateNamedPipeA(path.data(), PIPE_ACCESS_DUPLEX,
                       PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                       PIPE_UNLIMITED_INSTANCES, 4096, 4096, 0, NULL);

  if (pipe_handle == INVALID_HANDLE_VALUE) {
    DWORD error = GetLastError();
    return createStringError(std::error_code(error, std::system_category()),
                             "Couldn't create named pipe: %s", path.data());
  }

  return std::make_shared<FifoFile>(path, pipe_handle);
#else
  if (int err = mkfifo(path.data(), 0600))
    return createStringError(std::error_code(err, std::generic_category()),
                             "Couldn't create fifo file: %s", path.data());
  return std::make_shared<FifoFile>(path);
#endif
}

FifoFileIO::FifoFileIO(std::shared_ptr<FifoFile> fifo_file,
                       StringRef other_endpoint_name)
    : m_fifo_file(std::move(fifo_file)),
      m_other_endpoint_name(other_endpoint_name) {}

Expected<json::Value> FifoFileIO::ReadJSON(std::chrono::milliseconds timeout) {
  // We use a pointer for this future, because otherwise its normal destructor
  // would wait for the getline to end, rendering the timeout useless.
  std::optional<std::string> line;
  std::future<void> *future =
      new std::future<void>(std::async(std::launch::async, [&]() {
        std::string buffer = m_fifo_file->ReadLine();
        if (!buffer.empty())
          line = buffer;
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
        m_fifo_file->WriteLine(JSONToString(json));
        done = true;
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

} // namespace lldb_dap
