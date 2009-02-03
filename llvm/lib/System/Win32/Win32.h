//===- Win32/Win32.h - Common Win32 Include File ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines things specific to Win32 implementations.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic Win32 code that
//===          is guaranteed to work on *all* Win32 variants.
//===----------------------------------------------------------------------===//

// Require at least Windows 2000 API.
#define _WIN32_WINNT 0x0500

#include "llvm/Config/config.h"     // Get autoconf configuration settings
#include "windows.h"
#include <cassert>
#include <string>

inline bool MakeErrMsg(std::string* ErrMsg, const std::string& prefix) {
  if (!ErrMsg)
    return true;
  char *buffer = NULL;
  FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER|FORMAT_MESSAGE_FROM_SYSTEM,
      NULL, GetLastError(), 0, (LPSTR)&buffer, 1, NULL);
  *ErrMsg = prefix + buffer;
  LocalFree(buffer);
  return true;
}

class AutoHandle {
  HANDLE handle;

public:
  AutoHandle(HANDLE h) : handle(h) {}

  ~AutoHandle() {
    if (handle)
      CloseHandle(handle);
  }

  operator HANDLE() {
    return handle;
  }

  AutoHandle &operator=(HANDLE h) {
    handle = h;
    return *this;
  }
};
