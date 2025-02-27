//===-- IOStream.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IOStream.h"
#include "DAPLog.h"
#include "lldb/Utility/IOObject.h"
#include "lldb/Utility/Status.h"
#include <string>

using namespace lldb_dap;
using namespace lldb_private;

bool OutputStream::write_full(llvm::StringRef str) {
  if (!descriptor)
    return false;

  size_t num_bytes = str.size();
  auto status = descriptor->Write(str.data(), num_bytes);
  return status.Success();
}

bool InputStream::read_full(size_t length, std::string &text) {
  if (!descriptor)
    return false;

  std::string data;
  data.resize(length);

  auto status = descriptor->Read(data.data(), length);
  if (status.Fail())
    return false;

  text += data;
  return true;
}

bool InputStream::read_line(std::string &line) {
  line.clear();
  while (true) {
    if (!read_full(1, line))
      return false;

    if (llvm::StringRef(line).ends_with("\r\n"))
      break;
  }
  line.erase(line.size() - 2);
  return true;
}

bool InputStream::read_expected(llvm::StringRef expected) {
  std::string result;
  if (!read_full(expected.size(), result))
    return false;
  if (expected != result) {
    LLDB_LOG(GetLog(DAPLog::Transport), "Warning: Expected '{0}', got '{1}'",
             expected, result);
  }
  return true;
}
