//===-- zOSLibFunctions.cpp -----------------------------------------------===//
////
//// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//// See https://llvm.org/LICENSE.txt for license information.
//// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
////
////===--------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// This file defines z/OS implementations for common functions.
//
//===----------------------------------------------------------------------===//

#ifdef __MVS__
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>


// z/OS Unix System Services does not have strsignal() support, so the
// strsignal() function is implemented here.
char *strsignal(int sig) {
  static char msg[256];
  sprintf(msg, "%d", sig);
  return msg;
}

// z/OS Unix System Services does not have strnlen() support, so the strnlen()
// function is implemented here.
size_t strnlen(const char *S, size_t MaxLen) {
  const char *PtrToNullChar =
      static_cast<const char *>(memchr(S, '\0', MaxLen));
  return PtrToNullChar ? PtrToNullChar - S : MaxLen;
}
#endif
