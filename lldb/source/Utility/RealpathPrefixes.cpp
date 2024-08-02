//===-- RealpathPrefixes.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/RealpathPrefixes.h"

#include "lldb/Target/Statistics.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/FileSpecList.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

using namespace lldb_private;

RealpathPrefixes::RealpathPrefixes(const FileSpecList &file_spec_list)
    : m_fs(llvm::vfs::getRealFileSystem()), m_target(nullptr) {
  m_prefixes.reserve(file_spec_list.GetSize());
  for (const FileSpec &file_spec : file_spec_list) {
    m_prefixes.emplace_back(file_spec.GetPath());
  }
}

void RealpathPrefixes::SetFileSystem(
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs) {
  m_fs = fs;
}

std::optional<FileSpec>
RealpathPrefixes::ResolveSymlinks(const FileSpec &file_spec) const {
  if (m_prefixes.empty())
    return std::nullopt;

  auto is_prefix = [](llvm::StringRef a, llvm::StringRef b,
                      bool case_sensitive) -> bool {
    return case_sensitive ? a.consume_front(b) : a.consume_front_insensitive(b);
  };
  std::string file_spec_path = file_spec.GetPath();
  for (const std::string &prefix : m_prefixes) {
    if (is_prefix(file_spec_path, prefix, file_spec.IsCaseSensitive())) {
      // Stats and logging.
      if (m_target)
        m_target->GetStatistics().IncreaseSourceRealpathAttemptCount();
      Log *log = GetLog(LLDBLog::Source);
      LLDB_LOGF(log, "Realpath'ing support file %s", file_spec_path.c_str());

      // One prefix matched. Try to realpath.
      llvm::SmallString<PATH_MAX> buff;
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
