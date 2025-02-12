//===-- IOStream.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IOStream.h"
#include "lldb/Utility/IOObject.h" // IWYU pragma: keep
#include "lldb/Utility/Status.h"
#include <fstream>
#include <string>

using namespace lldb_dap;

bool OutputStream::write_full(llvm::StringRef str) {
  if (!descriptor || !descriptor->IsValid())
    return false;

  size_t bytes = str.size();
  return descriptor->Write(str.data(), bytes).Success();
}

bool InputStream::read_full(std::ofstream *log, size_t length,
                            std::string &text) {
  std::string data;
  data.resize(length);

  char *ptr = &data[0];
  size_t bytes_read = length;
  if (!descriptor->Read(ptr, bytes_read).Success())
    return false;
  text += data;
  return true;
}

bool InputStream::read_line(std::ofstream *log, std::string &line) {
  line.clear();
  while (true) {
    if (!read_full(log, 1, line))
      return false;

    if (llvm::StringRef(line).ends_with("\r\n"))
      break;
  }
  line.erase(line.size() - 2);
  return true;
}

bool InputStream::read_expected(std::ofstream *log, llvm::StringRef expected) {
  std::string result;
  if (!read_full(log, expected.size(), result))
    return false;
  if (expected != result) {
    if (log)
      *log << "Warning: Expected '" << expected.str() << "', got '" << result
           << "\n";
  }
  return true;
}
