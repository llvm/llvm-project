//===-- FileAction.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <fcntl.h>

#include "lldb/Host/FileAction.h"
#include "lldb/Host/PosixApi.h"
#include "lldb/Utility/Stream.h"

#ifdef _WIN32
#include "lldb/Host/windows/windows.h"
#endif

using namespace lldb_private;

// FileAction member functions

FileAction::FileAction() : m_file_spec() {}

void FileAction::Clear() {
  m_action = eFileActionNone;
  m_file = -1;
  m_arg = -1;
  m_file_spec.Clear();
}

#ifdef _WIN32
HANDLE FileAction::GetHandle() const {
  if (std::holds_alternative<HANDLE>(m_file))
    return std::get<HANDLE>(m_file);
  int file = std::get<int>(m_file);
  switch (file) {
  case STDIN_FILENO:
    return GetStdHandle(STD_INPUT_HANDLE);
  case STDOUT_FILENO:
    return GetStdHandle(STD_OUTPUT_HANDLE);
  case STDERR_FILENO:
    return GetStdHandle(STD_ERROR_HANDLE);
  default:
    return INVALID_HANDLE_VALUE;
  }
}
#endif

const FileSpec &FileAction::GetFileSpec() const { return m_file_spec; }

#ifdef _WIN32
bool FileAction::Open(HANDLE fh, const FileSpec &file_spec, bool read,
                      bool write) {
  if ((read || write) && fh != INVALID_HANDLE_VALUE && file_spec) {
    m_action = eFileActionOpen;
    m_file = fh;
    if (read && write)
      m_arg = O_NOCTTY | O_CREAT | O_RDWR;
    else if (read)
      m_arg = O_NOCTTY | O_RDONLY;
    else
      m_arg = O_NOCTTY | O_CREAT | O_WRONLY | O_TRUNC;
    m_file_spec = file_spec;
    return true;
  } else {
    Clear();
  }
  return false;
}
#endif

bool FileAction::Open(int fd, const FileSpec &file_spec, bool read,
                      bool write) {
  if ((read || write) && fd >= 0 && file_spec) {
    m_action = eFileActionOpen;
    m_file = fd;
    if (read && write)
      m_arg = O_NOCTTY | O_CREAT | O_RDWR;
    else if (read)
      m_arg = O_NOCTTY | O_RDONLY;
    else
      m_arg = O_NOCTTY | O_CREAT | O_WRONLY | O_TRUNC;
    m_file_spec = file_spec;
    return true;
  } else {
    Clear();
  }
  return false;
}

int FileAction::GetFD() const {
#ifdef _WIN32
  if (std::holds_alternative<int>(m_file))
    return std::get<int>(m_file);
  return -1;
#else
  return std::get<int>(m_file);
#endif
}

bool FileAction::Close(int fd) {
  Clear();
  if (fd >= 0) {
    m_action = eFileActionClose;
    m_file = fd;
    return true;
  }
  return false;
}

#ifdef _WIN32
bool FileAction::Duplicate(HANDLE fh, HANDLE dup_fh) {
  Clear();
  if (fh != INVALID_HANDLE_VALUE && dup_fh != INVALID_HANDLE_VALUE) {
    m_action = eFileActionDuplicate;
    m_file = fh;
    m_arg = dup_fh;
    return true;
  }
  return false;
}
#endif

bool FileAction::Duplicate(int fd, int dup_fd) {
  Clear();
  if (fd >= 0 && dup_fd >= 0) {
    m_action = eFileActionDuplicate;
    m_file = fd;
    m_arg = dup_fd;
    return true;
  }
  return false;
}

#ifdef _WIN32
HANDLE FileAction::GetActionArgumentHandle() const {
  if (std::holds_alternative<HANDLE>(m_arg))
    return std::get<HANDLE>(m_arg);
  return INVALID_HANDLE_VALUE;
}
#endif

int FileAction::GetActionArgument() const {
#ifdef _WIN32
  if (std::holds_alternative<int>(m_arg))
    return std::get<int>(m_arg);
  return -1;
#else
  return std::get<int>(m_arg);
#endif
}

void FileAction::Dump(Stream &stream) const {
#ifdef _WIN32
  int file =
      std::holds_alternative<HANDLE>(m_file) ? (int)GetHandle() : GetFD();
  int arg = std::holds_alternative<HANDLE>(m_arg)
                ? (int)GetActionArgumentHandle()
                : GetActionArgument();
#else
  int file = GetFD();
  int arg = GetActionArgument();
#endif
  stream.PutCString("file action: ");
  switch (m_action) {
  case eFileActionClose:
    stream.Printf("close fd %d", file);
    break;
  case eFileActionDuplicate:
    stream.Printf("duplicate fd %d to %d", file, arg);
    break;
  case eFileActionNone:
    stream.PutCString("no action");
    break;
  case eFileActionOpen:
    stream.Printf("open fd %d with '%s', OFLAGS = 0x%x", file,
                  m_file_spec.GetPath().c_str(), arg);
    break;
  }
}
