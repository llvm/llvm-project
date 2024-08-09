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
      "parser", cl::desc("Input remark format to parse"),                      \
      cl::values(clEnumValN(Format::YAML, "yaml", "YAML"),                     \
                 clEnumValN(Format::Bitstream, "bitstream", "Bitstream")),     \
      cl::sub(SUBOPT));

#define DEBUG_LOC_INFO_COMMAND_LINE_OPTIONS(SUBOPT)                            \
  static cl::opt<bool> UseDebugLoc(                                            \
      "use-debug-loc",                                                         \
      cl::desc(                                                                \
          "Add debug loc information when generating tables for "              \
          "functions. The loc is represented as (path:line number:column "     \
          "number)"),                                                          \
      cl::init(false), cl::sub(SUBOPT));

#define FILTER_COMMAND_LINE_OPTIONS(SUBOPT)                                    \
  static cl::opt<std::string> RemarkNameOpt(                                   \
      "remark-name",                                                           \
      cl::desc("Optional remark name to filter collection by."),               \
      cl::ValueOptional, cl::sub(SUBOPT));                                     \
  static cl::opt<std::string> PassNameOpt(                                     \
      "pass-name", cl::ValueOptional,                                          \
      cl::desc("Optional remark pass name to filter collection by."),          \
      cl::sub(SUBOPT));                                                        \
  static cl::opt<std::string> RemarkFilterArgByOpt(                            \
      "filter-arg-by",                                                         \
      cl::desc("Optional remark arg to filter collection by."),                \
      cl::ValueOptional, cl::sub(SUBOPT));                                     \
  static cl::opt<std::string> RemarkNameOptRE(                                 \
      "rremark-name",                                                          \
      cl::desc("Optional remark name to filter collection by "                 \
               "(accepts regular expressions)."),                              \
      cl::ValueOptional, cl::sub(SUBOPT));                                     \
  static cl::opt<std::string> RemarkArgFilterOptRE(                            \
      "rfilter-arg-by",                                                        \
      cl::desc("Optional remark arg to filter collection by "                  \
               "(accepts regular expressions)."),                              \
      cl::sub(SUBOPT), cl::ValueOptional);                                     \
  static cl::opt<std::string> PassNameOptRE(                                   \
      "rpass-name", cl::ValueOptional,                                         \
      cl::desc("Optional remark pass name to filter collection "               \
               "by (accepts regular expressions)."),                           \
      cl::sub(SUBOPT));                                                        \
  static cl::opt<Type> RemarkTypeOpt(                                          \
      "remark-type",                                                           \
      cl::desc("Optional remark type to filter collection by."),               \
      cl::values(clEnumValN(Type::Unknown, "unknown", "UNKOWN"),               \
                 clEnumValN(Type::Passed, "passed", "PASSED"),                 \
                 clEnumValN(Type::Missed, "missed", "MISSED"),                 \
                 clEnumValN(Type::Analysis, "analysis", "ANALYSIS"),           \
                 clEnumValN(Type::AnalysisFPCommute, "analysis-fp-commute",    \
                            "ANALYSIS_FP_COMMUTE"),                            \
                 clEnumValN(Type::AnalysisAliasing, "analysis-aliasing",       \
                            "ANALYSIS_ALIASING"),                              \
                 clEnumValN(Type::Failure, "failure", "FAILURE")),             \
      cl::init(Type::Failure), cl::sub(SUBOPT));

namespace llvm {
namespace remarks {

/// Filter object which can be either a string or a regex to match with the
/// remark properties.
struct FilterMatcher {
  Regex FilterRE;
  std::string FilterStr;
  bool IsRegex;
  FilterMatcher(std::string Filter, bool IsRegex) : IsRegex(IsRegex) {
    if (IsRegex)
      FilterRE = Regex(Filter);
    else
      FilterStr = Filter;
  }

  bool match(StringRef StringToMatch) const {
    if (IsRegex)
      return FilterRE.match(StringToMatch);
    return FilterStr == StringToMatch.trim().str();
  }
};

/// Filter out remarks based on remark properties based on name, pass name,
/// argument and type.
struct Filters {
  std::optional<FilterMatcher> RemarkNameFilter;
  std::optional<FilterMatcher> PassNameFilter;
  std::optional<FilterMatcher> ArgFilter;
  std::optional<Type> RemarkTypeFilter;
  /// Returns a filter object if all the arguments provided are valid regex
  /// types otherwise return an error.
  static Expected<Filters>
  createRemarkFilter(std::optional<FilterMatcher> RemarkNameFilter,
                     std::optional<FilterMatcher> PassNameFilter,
                     std::optional<FilterMatcher> ArgFilter,
                     std::optional<Type> RemarkTypeFilter) {
    Filters Filter;
    Filter.RemarkNameFilter = std::move(RemarkNameFilter);
    Filter.PassNameFilter = std::move(PassNameFilter);
    Filter.ArgFilter = std::move(ArgFilter);
    Filter.RemarkTypeFilter = std::move(RemarkTypeFilter);
    if (auto E = Filter.regexArgumentsValid())
      return std::move(E);
    return std::move(Filter);
  }
  /// Returns true if \p Remark satisfies all the provided filters.
  bool filterRemark(const Remark &Remark);

private:
  /// Check if arguments can be parsed as valid regex types.
  Error regexArgumentsValid();
};

/// Convert Regex string error to an error object.
inline Error checkRegex(const Regex &Regex) {
  std::string Error;
  if (!Regex.isValid(Error))
    return createStringError(make_error_code(std::errc::invalid_argument),
                             Twine("Regex: ", Error));
  return Error::success();
}

Expected<Filters> getRemarkFilter(cl::opt<std::string> &RemarkNameOpt,
                                  cl::opt<std::string> &RemarkNameOptRE,
                                  cl::opt<std::string> &PassNameOpt,
                                  cl::opt<std::string> &PassNameOptRE,
                                  cl::opt<Type> &RemarkTypeOpt,
                                  cl::opt<std::string> &RemarkFilterArgByOpt,
                                  cl::opt<std::string> &RemarkArgFilterOptRE);
Expected<std::unique_ptr<MemoryBuffer>>
getInputMemoryBuffer(StringRef InputFileName);
Expected<std::unique_ptr<ToolOutputFile>>
getOutputFileWithFlags(StringRef OutputFileName, sys::fs::OpenFlags Flags);
Expected<std::unique_ptr<ToolOutputFile>>
getOutputFileForRemarks(StringRef OutputFileName, Format OutputFormat);
} // namespace remarks
} // namespace llvm
