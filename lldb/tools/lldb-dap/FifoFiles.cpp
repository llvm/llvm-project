//===-- FifoFiles.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FifoFiles.h"
#include "JSONUtils.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/WindowsError.h"

#if defined(_WIN32)
#include <Windows.h>
#include <fcntl.h>
#include <io.h>
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

FifoFile::FifoFile(std::string path, FILE *f) : m_path(path), m_file(f) {}

Expected<FifoFile> FifoFile::create(StringRef path) {
  auto file = fopen(path.data(), "r+");
  if (file == nullptr)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to open fifo file " + path);
  if (setvbuf(file, NULL, _IONBF, 0))
    return createStringError(inconvertibleErrorCode(),
                             "Failed to setvbuf on fifo file " + path);
  return FifoFile(path, file);
}

FifoFile::FifoFile(StringRef path, FILE *f) : m_path(path), m_file(f) {}

FifoFile::FifoFile(FifoFile &&other)
    : m_path(other.m_path), m_file(other.m_file) {
  other.m_path.clear();
  other.m_file = nullptr;
}
FifoFile::~FifoFile() {
  if (m_file)
    fclose(m_file);
#if !defined(_WIN32)
  // Unreferenced named pipes are deleted automatically on Win32
  unlink(m_path.c_str());
#endif
}

// This probably belongs to llvm::sys::fs as another FSEntity type
Error createUniqueNamedPipe(const Twine &prefix, StringRef suffix,
                            int &result_fd,
                            SmallVectorImpl<char> &result_path) {
  std::error_code EC;
#ifdef _WIN32
  const char *middle = suffix.empty() ? "-%%%%%%" : "-%%%%%%.";
  EC = sys::fs::getPotentiallyUniqueFileName(
      "\\\\.\\pipe\\LOCAL\\" + prefix + middle + suffix, result_path);
#else
  EC = sys::fs::getPotentiallyUniqueTempFileName(prefix, suffix, result_path);
#endif
  if (EC)
    return errorCodeToError(EC);
  result_path.push_back(0);
  const char *path = result_path.data();
#ifdef _WIN32
  HANDLE h = ::CreateNamedPipeA(
      path, PIPE_ACCESS_DUPLEX | FILE_FLAG_FIRST_PIPE_INSTANCE,
      PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, 1, 512, 512, 0, NULL);
  if (h == INVALID_HANDLE_VALUE)
    return createStringError(mapLastWindowsError(), "CreateNamedPipe");
  result_fd = _open_osfhandle((intptr_t)h, _O_TEXT | _O_RDWR);
  if (result_fd == -1)
    return createStringError(mapLastWindowsError(), "_open_osfhandle");
#else
  if (mkfifo(path, 0600) == -1)
    return createStringError(std::error_code(errno, std::generic_category()),
                             "mkfifo");
  EC = openFileForWrite(result_path, result_fd, sys::fs::CD_OpenExisting,
                        sys::fs::OF_None, 0600);
  if (EC)
    return errorCodeToError(EC);
#endif
  result_path.pop_back();
  return Error::success();
}

FifoFileIO::FifoFileIO(FifoFile &&fifo_file, StringRef other_endpoint_name)
    : m_fifo_file(std::move(fifo_file)),
      m_other_endpoint_name(other_endpoint_name) {}

Expected<json::Value> FifoFileIO::ReadJSON(std::chrono::milliseconds timeout) {
  // We use a pointer for this future, because otherwise its normal destructor
  // would wait for the getline to end, rendering the timeout useless.
  std::optional<std::string> line;
  std::future<void> *future =
      new std::future<void>(std::async(std::launch::async, [&]() {
        rewind(m_fifo_file.m_file);
        /*
         * The types of messages passing through the fifo are:
         * {"kind": "pid", "pid": 9223372036854775807}
         * {"kind": "didAttach"}
         * {"kind": "error", "error": "error message"}
         *
         * 512 characters should be more than enough.
         */
        constexpr size_t buffer_size = 512;
        char buffer[buffer_size];
        char *ptr = fgets(buffer, buffer_size, m_fifo_file.m_file);
        if (ptr == nullptr || *ptr == 0)
          return;
        size_t len = strlen(buffer);
        if (len <= 1)
          return;
        buffer[len - 1] = '\0'; // remove newline
        line = buffer;
      }));

  if (future->wait_for(timeout) == std::future_status::timeout)
    // Indeed this is a leak, but it's intentional. "future" obj destructor
    //  will block on waiting for the worker thread to join. And the worker
    //  thread might be stuck in blocking I/O. Intentionally leaking the  obj
    //  as a hack to avoid blocking main thread, and adding annotation to
    //  supress static code inspection warnings

    // coverity[leaked_storage]
    return createStringError(inconvertibleErrorCode(),
                             "Timed out trying to get messages from the " +
                                 m_other_endpoint_name);
  if (!line) {
    return createStringError(inconvertibleErrorCode(),
                             "Failed to get messages from the " +
                                 m_other_endpoint_name);
  }
  delete future;
  return json::parse(*line);
}

Error FifoFileIO::SendJSON(const json::Value &json,
                           std::chrono::milliseconds timeout) {
  bool done = false;
  std::future<void> *future =
      new std::future<void>(std::async(std::launch::async, [&]() {
        rewind(m_fifo_file.m_file);
        auto payload = JSONToString(json);
        if (fputs(payload.c_str(), m_fifo_file.m_file) == EOF ||
            fputc('\n', m_fifo_file.m_file) == EOF) {
          return;
        }
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

Error FifoFileIO::WaitForPeer() {
#ifdef _WIN32
  if (!::ConnectNamedPipe((HANDLE)_get_osfhandle(fileno(m_fifo_file.m_file)),
                          NULL) &&
      GetLastError() != ERROR_PIPE_CONNECTED) {
    return createStringError(mapLastWindowsError(),
                             "Failed to connect to the " +
                                 m_other_endpoint_name);
  }
#endif
  return Error::success();
}

} // namespace lldb_dap
