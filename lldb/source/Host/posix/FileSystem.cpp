//===-- FileSystem.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/FileSystem.h"

// C includes
#include <dirent.h>
#include <sys/mount.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef __linux__
#include <linux/magic.h>
#include <sys/mount.h>
#include <sys/statfs.h>
#endif
#if defined(__NetBSD__)
#include <sys/statvfs.h>
#endif

// lldb Includes
#include "lldb/Host/Host.h"
#include "lldb/Utility/Error.h"
#include "lldb/Utility/StreamString.h"

#include "llvm/Support/FileSystem.h"

using namespace lldb;
using namespace lldb_private;

const char *FileSystem::DEV_NULL = "/dev/null";

Error FileSystem::Symlink(const FileSpec &src, const FileSpec &dst) {
  Error error;
  if (::symlink(dst.GetCString(), src.GetCString()) == -1)
    error.SetErrorToErrno();
  return error;
}

Error FileSystem::Readlink(const FileSpec &src, FileSpec &dst) {
  Error error;
  char buf[PATH_MAX];
  ssize_t count = ::readlink(src.GetCString(), buf, sizeof(buf) - 1);
  if (count < 0)
    error.SetErrorToErrno();
  else {
    buf[count] = '\0'; // Success
    dst.SetFile(buf, false);
  }
  return error;
}

Error FileSystem::ResolveSymbolicLink(const FileSpec &src, FileSpec &dst) {
  char resolved_path[PATH_MAX];
  if (!src.GetPath(resolved_path, sizeof(resolved_path))) {
    return Error("Couldn't get the canonical path for %s", src.GetCString());
  }

  char real_path[PATH_MAX + 1];
  if (realpath(resolved_path, real_path) == nullptr) {
    Error err;
    err.SetErrorToErrno();
    return err;
  }

  dst = FileSpec(real_path, false);

  return Error();
}

FILE *FileSystem::Fopen(const char *path, const char *mode) {
  return ::fopen(path, mode);
}
