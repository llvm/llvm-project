//===-- LogTest.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Core/Log.h"
#include "lldb/Host/Host.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb;
using namespace lldb_private;

static std::string GetLogString(uint32_t log_options, const char *format,
                                int arg) {
  std::string stream_string;
  std::shared_ptr<llvm::raw_string_ostream> stream_sp(
      new llvm::raw_string_ostream(stream_string));
  Log log_(stream_sp);
  log_.GetOptions().Reset(log_options);
  Log *log = &log_;
  LLDB_LOG(log, format, arg);
  return stream_sp->str();
}

TEST(LogTest, LLDB_LOG_nullptr) {
  Log *log = nullptr;
  LLDB_LOG(log, "{0}", 0); // Shouldn't crash
}

TEST(LogTest, log_options) {
  EXPECT_EQ("Hello World 47\n", GetLogString(0, "Hello World {0}", 47));
  EXPECT_EQ("Hello World 47\n",
            GetLogString(LLDB_LOG_OPTION_THREADSAFE, "Hello World {0}", 47));

  {
    std::string msg =
        GetLogString(LLDB_LOG_OPTION_PREPEND_SEQUENCE, "Hello World {0}", 47);
    int seq_no;
    EXPECT_EQ(1, sscanf(msg.c_str(), "%d Hello World 47", &seq_no));
  }

  EXPECT_EQ(
      "LogTest.cpp:GetLogString                                     Hello "
      "World 47\n",
      GetLogString(LLDB_LOG_OPTION_PREPEND_FILE_FUNCTION, "Hello World {0}", 47));

  EXPECT_EQ(llvm::formatv("[{0,0+4}/{1,0+4}] Hello World 47\n",
                          Host::GetCurrentProcessID(),
                          Host::GetCurrentThreadID())
                .str(),
            GetLogString(LLDB_LOG_OPTION_PREPEND_PROC_AND_THREAD,
                         "Hello World {0}", 47));
}
