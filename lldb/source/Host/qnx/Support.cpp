//===-- Support.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/qnx/Support.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

#include "llvm/Support/MemoryBuffer.h"

using namespace lldb;

llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
lldb_private::getProcFile(::pid_t pid, const llvm::Twine &file) {
  Log *log = GetLog(LLDBLog::Host);
  std::string path = ("/proc/" + llvm::Twine(pid) + "/" + file).str();
  auto ret = llvm::MemoryBuffer::getFileAsStream(path);
  if (!ret)
    LLDB_LOG(log, "Failed to open {0}: {1}", path, ret.getError().message());
  return ret;
}

llvm::Expected<FileUP> lldb_private::openProcFile(::pid_t pid,
                                                  const llvm::Twine &file,
                                                  File::OpenOptions options,
                                                  uint32_t permissions,
                                                  bool should_close_fd) {
  Log *log = GetLog(LLDBLog::Host);
  std::string path = ("/proc/" + llvm::Twine(pid) + "/" + file).str();
  auto ret = FileSystem::Instance().Open(FileSpec(llvm::StringRef(path)),
                                         options, permissions, should_close_fd);
  if (!ret) {
    LLDB_LOG_ERROR(log, ret.takeError(), "Failed to open {0}: {1}", path);
  }
  return ret;
}
