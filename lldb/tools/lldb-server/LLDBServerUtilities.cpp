//===-- LLDBServerUtilities.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LLDBServerUtilities.h"

#include "lldb/Core/Log.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Interpreter/Args.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace lldb;
using namespace lldb_private::lldb_server;
using namespace llvm;

bool LLDBServerUtilities::SetupLogging(const std::string &log_file,
                                       const StringRef &log_channels,
                                       uint32_t log_options) {
  lldb::StreamSP log_stream_sp;
  if (log_file.empty()) {
    log_stream_sp.reset(new StreamFile(stdout, false));
  } else {
    uint32_t options = File::eOpenOptionWrite | File::eOpenOptionCanCreate |
                       File::eOpenOptionCloseOnExec | File::eOpenOptionAppend;
    if (!(log_options & LLDB_LOG_OPTION_APPEND))
      options |= File::eOpenOptionTruncate;

    log_stream_sp.reset(new StreamFile(log_file.c_str(), options));
  }

  SmallVector<StringRef, 32> channel_array;
  log_channels.split(channel_array, ":", /*MaxSplit*/ -1, /*KeepEmpty*/ false);
  for (auto channel_with_categories : channel_array) {
    StreamString error_stream;
    Args channel_then_categories(channel_with_categories);
    std::string channel(channel_then_categories.GetArgumentAtIndex(0));
    channel_then_categories.Shift(); // Shift off the channel

    bool success = Log::EnableLogChannel(
        log_stream_sp, log_options, channel.c_str(),
        channel_then_categories.GetConstArgumentVector(), error_stream);
    if (!success) {
      fprintf(stderr, "Unable to open log file '%s' for channel \"%s\"\n",
              log_file.c_str(), channel_with_categories.str().c_str());
      return false;
    }
  }
  return true;
}
