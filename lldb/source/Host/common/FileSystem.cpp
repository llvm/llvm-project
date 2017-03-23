//===-- FileSystem.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/FileSystem.h"

#include "llvm/Support/FileSystem.h"

#include <algorithm>
#include <fstream>
#include <vector>

using namespace lldb;
using namespace lldb_private;

llvm::sys::TimePoint<>
FileSystem::GetModificationTime(const FileSpec &file_spec) {
  llvm::sys::fs::file_status status;
  std::error_code ec = llvm::sys::fs::status(file_spec.GetPath(), status);
  if (ec)
    return llvm::sys::TimePoint<>();
  return status.getLastModificationTime();
}
