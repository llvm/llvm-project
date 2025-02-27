//===-- RealpathPrefixes.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/RealpathPrefixes.h"

#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/FileSpecList.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-private-types.h"

using namespace lldb_private;

RealpathPrefixes::RealpathPrefixes(
    const FileSpecList &file_spec_list,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs)
    : m_fs(fs) {
  m_prefixes.reserve(file_spec_list.GetSize());
  for (const FileSpec &file_spec : file_spec_list) {
    m_prefixes.emplace_back(file_spec.GetPath());
  }
}

std::optional<FileSpec>
RealpathPrefixes::ResolveSymlinks(const FileSpec &file_spec) {
  if (m_prefixes.empty())
    return std::nullopt;

  // Test if `b` is a *path* prefix of `a` (not just *string* prefix).
  // E.g. "/foo/bar" is a path prefix of "/foo/bar/baz" but not "/foo/barbaz".
  auto is_path_prefix = [](llvm::StringRef a, llvm::StringRef b,
                           bool case_sensitive,
                           llvm::sys::path::Style style) -> bool {
    if (case_sensitive ? a.consume_front(b) : a.consume_front_insensitive(b))
      // If `b` isn't "/", then it won't end with "/" because it comes from
      // `FileSpec`. After `a` consumes `b`, `a` should either be empty (i.e.
      // `a` == `b`) or end with "/" (the remainder of `a` is a subdirectory).
      return b == "/" || a.empty() ||
             llvm::sys::path::is_separator(a[0], style);
    return false;
  };
  std::string file_spec_path = file_spec.GetPath();
  for (const std::string &prefix : m_prefixes) {
    if (is_path_prefix(file_spec_path, prefix, file_spec.IsCaseSensitive(),
                       file_spec.GetPathStyle())) {
      // Stats and logging.
      IncreaseSourceRealpathAttemptCount();
      Log *log = GetLog(LLDBLog::Source);
      LLDB_LOGF(log, "Realpath'ing support file %s", file_spec_path.c_str());

      // One prefix matched. Try to realpath.
      PathSmallString buff;
      std::error_code ec = m_fs->getRealPath(file_spec_path, buff);
      if (ec)
        return std::nullopt;
      FileSpec realpath(buff, file_spec.GetPathStyle());

      // Only return realpath if it is different from the original file_spec.
      if (realpath != file_spec)
        return realpath;
      return std::nullopt;
    }
  }
  // No prefix matched
  return std::nullopt;
}
