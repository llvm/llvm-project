//===-- ThisThread.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/ThisThread.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/Error.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>

using namespace lldb;
using namespace lldb_private;

void ThisThread::SetName(llvm::StringRef name, int max_length) {
  std::string truncated_name(name.data());

  // Thread names are coming in like '<lldb.comm.debugger.edit>' and
  // '<lldb.comm.debugger.editline>'.  So just chopping the end of the string
  // off leads to a lot of similar named threads.  Go through the thread name
  // and search for the last dot and use that.

  if (max_length > 0 &&
      truncated_name.length() > static_cast<size_t>(max_length)) {
    // First see if we can get lucky by removing any initial or final braces.
    std::string::size_type begin = truncated_name.find_first_not_of("(<");
    std::string::size_type end = truncated_name.find_last_not_of(")>.");
    if (end - begin > static_cast<size_t>(max_length)) {
      // We're still too long.  Since this is a dotted component, use everything
      // after the last
      // dot, up to a maximum of |length| characters.
      std::string::size_type last_dot = truncated_name.rfind('.');
      if (last_dot != std::string::npos)
        begin = last_dot + 1;

      end = std::min(end, begin + max_length);
    }

    std::string::size_type count = end - begin + 1;
    truncated_name = truncated_name.substr(begin, count);
  }

  SetName(truncated_name);
}
