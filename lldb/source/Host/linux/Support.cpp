//===-- Support.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/linux/Support.h"
#include "lldb/Utility/Log.h"
#include "llvm/Support/MemoryBuffer.h"

llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
lldb_private::getProcFile(::pid_t pid, ::pid_t tid, const llvm::Twine &file) {
  Log *log = GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
  std::string File =
      ("/proc/" + llvm::Twine(pid) + "/task/" + llvm::Twine(tid) + "/" + file)
          .str();
  auto Ret = llvm::MemoryBuffer::getFileAsStream(File);
  if (!Ret)
    LLDB_LOG(log, "Failed to open {0}: {1}", File, Ret.getError().message());
  return Ret;
}

llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
lldb_private::getProcFile(::pid_t pid, const llvm::Twine &file) {
  Log *log = GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
  std::string File = ("/proc/" + llvm::Twine(pid) + "/" + file).str();
  auto Ret = llvm::MemoryBuffer::getFileAsStream(File);
  if (!Ret)
    LLDB_LOG(log, "Failed to open {0}: {1}", File, Ret.getError().message());
  return Ret;
}
