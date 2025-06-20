//===-- Config.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_CONFIG_H
#define LLDB_HOST_CONFIG_H

#define LLDB_EDITLINE_USE_WCHAR 1

#define LLDB_HAVE_EL_RFUNC_T 0

#define HAVE_SYS_EVENT_H 1

#define HAVE_PPOLL 0

#define HAVE_PTSNAME_R 1

#define HAVE_PROCESS_VM_READV 0

#define HAVE_NR_PROCESS_VM_READV 0

#define HAVE_LIBCOMPRESSION 1

#define LLDB_ENABLE_POSIX 1

#define LLDB_ENABLE_TERMIOS 1

#define LLDB_ENABLE_LZMA 0

#define LLVM_ENABLE_CURL 0

#define LLDB_ENABLE_CURSES 1

#define CURSES_HAVE_NCURSES_CURSES_H 0

#define LLDB_ENABLE_LIBEDIT 1

#define LLDB_ENABLE_LIBXML2 1

#define LLDB_ENABLE_LUA 0

#define LLDB_ENABLE_PYTHON 1

#define LLDB_ENABLE_FBSDVMCORE 0

#define LLDB_EMBED_PYTHON_HOME 0

/* #undef LLDB_PYTHON_HOME */

#define LLDB_INSTALL_LIBDIR_BASENAME "lib"

/* #undef LLDB_GLOBAL_INIT_DIRECTORY */

#define LLDB_BUG_REPORT_URL "https://github.com/llvm/llvm-project/issues/"

#endif // #ifndef LLDB_HOST_CONFIG_H
