//===- RemarkUtilHelpers.h ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers for remark utilites
//
//===----------------------------------------------------------------------===//
#include "llvm-c/Remarks.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Remarks/Remark.h"
#include "llvm/Remarks/RemarkFormat.h"
#include "llvm/Remarks/RemarkParser.h"
#include "llvm/Remarks/YAMLRemarkSerializer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/ToolOutputFile.h"

// Keep input + output help + names consistent across the various modes via a
// hideous macro.
#define INPUT_OUTPUT_COMMAND_LINE_OPTIONS(SUBOPT)                              \
  static cl::opt<std::string> InputFileName(cl::Positional, cl::init("-"),     \
                                            cl::desc("<input file>"),          \
                                            cl::sub(SUBOPT));                  \
  static cl::opt<std::string> OutputFileName(                                  \
      "o", cl::init("-"), cl::desc("Output"), cl::value_desc("filename"),      \
      cl::sub(SUBOPT));

// Keep Input format and names consistent accross the modes via a macro.
#define INPUT_FORMAT_COMMAND_LINE_OPTIONS(SUBOPT)                              \
  static cl::opt<Format> InputFormat(                                          \
      "parser", cl::init(Format::Auto),                                        \
      cl::desc("Input remark format to parse"),                                \
      cl::values(                                                              \
          clEnumValN(Format::Auto, "auto", "Automatic detection (default)"),   \
          clEnumValN(Format::YAML, "yaml", "YAML"),                            \
          clEnumValN(Format::Bitstream, "bitstream", "Bitstream")),            \
      cl::sub(SUBOPT));

#define DEBUG_LOC_INFO_COMMAND_LINE_OPTIONS(SUBOPT)                            \
  static cl::opt<bool> UseDebugLoc(                                            \
      "use-debug-loc",                                                         \
      cl::desc(                                                                \
          "Add debug loc information when generating tables for "              \
          "functions. The loc is represented as (path:line number:column "     \
          "number)"),                                                          \
      cl::init(false), cl::sub(SUBOPT));

namespace llvm {
namespace remarks {
Expected<std::unique_ptr<MemoryBuffer>>
getInputMemoryBuffer(StringRef InputFileName);
Expected<std::unique_ptr<ToolOutputFile>>
getOutputFileWithFlags(StringRef OutputFileName, sys::fs::OpenFlags Flags);
Expected<std::unique_ptr<ToolOutputFile>>
getOutputFileForRemarks(StringRef OutputFileName, Format OutputFormat);

/// Filter object which can be either a string or a regex to match with the
/// remark properties.
class FilterMatcher {
  Regex FilterRE;
  std::string FilterStr;
  bool IsRegex;

  FilterMatcher(StringRef Filter, bool IsRegex)
      : FilterRE(Filter), FilterStr(Filter), IsRegex(IsRegex) {}

  static Expected<FilterMatcher> createRE(StringRef Arg, StringRef Value);

public:
  static FilterMatcher createExact(StringRef Filter) { return {Filter, false}; }

  static Expected<FilterMatcher>
  createRE(const llvm::cl::opt<std::string> &Arg);

  static Expected<FilterMatcher> createRE(StringRef Filter,
                                          const cl::list<std::string> &Arg);

  static Expected<std::optional<FilterMatcher>>
  createExactOrRE(const llvm::cl::opt<std::string> &ExactArg,
                  const llvm::cl::opt<std::string> &REArg);

  static FilterMatcher createAny() { return {".*", true}; }

  bool match(StringRef StringToMatch) const {
    if (IsRegex)
      return FilterRE.match(StringToMatch);
    return FilterStr == StringToMatch.trim().str();
  }
};

} // namespace remarks
} // namespace llvm
