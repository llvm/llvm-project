//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAPLog.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace lldb_dap;
using namespace llvm;

static llvm::StringRef last_line(llvm::StringRef str) {
  size_t index = str.find_last_of('\n', str.size() - 1);
  if (index == llvm::StringRef::npos)
    return str;
  return str.substr(index + 1);
}

TEST(DAPLog, Emit) {
  Log::Mutex mux;
  std::string outs;
  raw_string_ostream os(outs);
  Log log(os, mux);
  Log inner_log = log.WithPrefix("my_prefix:");

  // Line includes a timestamp, only check the suffix.
  log.Emit("Hi");
  EXPECT_TRUE(last_line(outs).ends_with(" Hi\n")) << outs;

  inner_log.Emit("foobar");
  EXPECT_TRUE(last_line(outs).ends_with(" my_prefix: foobar\n")) << outs;

  log.Emit("file.cpp", 42, "Hello from a file/line.");
  EXPECT_TRUE(
      last_line(outs).ends_with(" file.cpp:42 Hello from a file/line.\n"))
      << outs;

  inner_log.Emit("file.cpp", 42, "Hello from a file/line.");
  EXPECT_TRUE(last_line(outs).ends_with(
      " file.cpp:42 my_prefix: Hello from a file/line.\n"))
      << outs;
}