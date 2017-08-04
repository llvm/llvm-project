//===-- ProcFileReader.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcFileReader_h_
#define liblldb_ProcFileReader_h_

#include <functional>

#include "lldb/lldb-forward.h"
#include "lldb/lldb-types.h"

namespace lldb_private {
namespace process_linux {

class ProcFileReader {
public:
  static lldb::DataBufferSP ReadIntoDataBuffer(lldb::pid_t pid,
                                               const char *name);

  /// Parse the /proc/{@a pid}/{@a name} file line by line, passing each line to
  /// line_parser, until
  /// either end of file or until line_parser returns false.
  static Error
  ProcessLineByLine(lldb::pid_t pid, const char *name,
                    std::function<bool(const std::string &line)> line_parser);
};

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef liblldb_ProcFileReader_h_
