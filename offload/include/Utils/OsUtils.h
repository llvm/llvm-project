//===-- Utils/OsUtils.h - Target independent OpenMP target RTL -- C++
//------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Useful utilites to interact with the OS environment in a platform independent
// way.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_UTILS_OSUTILS_H
#define OMPTARGET_UTILS_OSUTILS_H

#ifdef _WIN32
#include <windows.h>
#else
#include <limits.h>
#include <unistd.h>
#endif

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace utils {

static inline std::string getExecName() {
#if defined(_WIN32)
  char Buffer[MAX_PATH];
  GetModuleFileNameA(nullptr, Buffer, MAX_PATH);
#else
  char Buffer[PATH_MAX];
  ssize_t Len = readlink("/proc/self/exe", Buffer, sizeof(Buffer) - 1);
  if (Len == -1)
    return "unknown";
  Buffer[Len] = '\0';
#endif
  llvm::StringRef Path(Buffer);

  if (!Path.empty())
    return llvm::sys::path::filename(Path).str();

  return "unknown";
}

} // namespace utils

#endif // OMPTARGET_UTILS_OSUTILS_H