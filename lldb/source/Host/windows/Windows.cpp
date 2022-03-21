//===-- Windows.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file provides Windows support functions

#include "lldb/Host/PosixApi.h"
#include "lldb/Host/windows/windows.h"

#include "llvm/Support/ConvertUTF.h"

#include <cassert>
#include <cctype>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <io.h>

int vasprintf(char **ret, const char *fmt, va_list ap) {
  char *buf;
  int len;
  size_t buflen;
  va_list ap2;

  va_copy(ap2, ap);
  len = vsnprintf(NULL, 0, fmt, ap2);

  if (len >= 0 &&
      (buf = (char *)malloc((buflen = (size_t)(len + 1)))) != NULL) {
    len = vsnprintf(buf, buflen, fmt, ap);
    *ret = buf;
  } else {
    *ret = NULL;
    len = -1;
  }

  va_end(ap2);
  return len;
}

#ifdef _MSC_VER

#if _MSC_VER < 1900
namespace lldb_private {
int vsnprintf(char *buffer, size_t count, const char *format, va_list argptr) {
  int old_errno = errno;
  int r = ::vsnprintf(buffer, count, format, argptr);
  int new_errno = errno;
  buffer[count - 1] = '\0';
  if (r == -1 || r == count) {
    FILE *nul = fopen("nul", "w");
    int bytes_written = ::vfprintf(nul, format, argptr);
    fclose(nul);
    if (bytes_written < count)
      errno = new_errno;
    else {
      errno = old_errno;
      r = bytes_written;
    }
  }
  return r;
}
} // namespace lldb_private
#endif

#endif // _MSC_VER
