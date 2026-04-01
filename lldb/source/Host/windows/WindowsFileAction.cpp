//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <fcntl.h>

#include "lldb/Host/PosixApi.h"
#include "lldb/Host/windows/WindowsFileAction.h"
#include "lldb/Host/windows/windows.h"

using namespace lldb_private;

bool WindowsFileAction::Duplicate(HANDLE fh, HANDLE dup_fh) {
  Clear();
  if (fh != INVALID_HANDLE_VALUE && dup_fh != INVALID_HANDLE_VALUE) {
    m_action = eFileActionDuplicate;
    m_handle = fh;
    m_arg_handle = dup_fh;
    return true;
  }
  return false;
}

bool WindowsFileAction::Open(HANDLE fh, const FileSpec &file_spec, bool read,
                             bool write) {
  if ((read || write) && fh != INVALID_HANDLE_VALUE && file_spec) {
    m_action = eFileActionOpen;
    m_handle = fh;
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

HANDLE WindowsFileAction::GetHandle() const {
  if (m_handle != INVALID_HANDLE_VALUE)
    return m_handle;
  switch (m_fd) {
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

HANDLE WindowsFileAction::GetActionArgumentHandle() const {
  return m_arg_handle;
}
