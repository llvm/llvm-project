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
#include "llvm/ADT/StringRef.h"
#include "llvm/Remarks/Remark.h"
#include "llvm/Remarks/RemarkFormat.h"
#include "llvm/Remarks/RemarkParser.h"
#include "llvm/Remarks/RemarkSerializer.h"
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

#define OUTPUT_FORMAT_COMMAND_LINE_OPTIONS(SUBOPT)                             \
  static cl::opt<Format> OutputFormat(                                         \
      "serializer", cl::init(Format::Auto),                                    \
      cl::desc("Output remark format to serialize"),                           \
      cl::values(clEnumValN(Format::Auto, "auto",                              \
                            "Automatic detection based on output file "        \
                            "extension or parser format (default)"),           \
                 clEnumValN(Format::YAML, "yaml", "YAML"),                     \
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

#define REMARK_FILTER_COMMAND_LINE_OPTIONS(SUBOPT)                             \
  static cl::opt<std::string> FunctionOpt(                                     \
      "function", cl::sub(SUBOPT), cl::ValueOptional,                          \
      cl::desc("Optional function name to filter collection by."));            \
  static cl::opt<std::string> FunctionOptRE(                                   \
      "rfunction", cl::sub(SUBOPT), cl::ValueOptional,                         \
      cl::desc("Optional function name to filter collection by "               \
               "(accepts regular expressions)."));                             \
  static cl::opt<std::string> RemarkNameOpt(                                   \
      "remark-name",                                                           \
      cl::desc("Optional remark name to filter collection by."),               \
      cl::ValueOptional, cl::sub(SUBOPT));                                     \
  static cl::opt<std::string> RemarkNameOptRE(                                 \
      "rremark-name",                                                          \
      cl::desc("Optional remark name to filter collection by "                 \
               "(accepts regular expressions)."),                              \
      cl::ValueOptional, cl::sub(SUBOPT));                                     \
  static cl::opt<std::string> PassNameOpt(                                     \
      "pass-name", cl::ValueOptional,                                          \
      cl::desc("Optional remark pass name to filter collection by."),          \
      cl::sub(SUBOPT));                                                        \
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
      cl::sub(SUBOPT));                                                        \
  static cl::opt<std::string> RemarkFilterArgByOpt(                            \
      "filter-arg-by",                                                         \
      cl::desc("Optional remark arg to filter collection by."),                \
      cl::ValueOptional, cl::sub(SUBOPT));                                     \
  static cl::opt<std::string> RemarkArgFilterOptRE(                            \
      "rfilter-arg-by",                                                        \
      cl::desc("Optional remark arg to filter collection by "                  \
               "(accepts regular expressions)."),                              \
      cl::sub(SUBOPT), cl::ValueOptional);

#define REMARK_FILTER_SETUP_FUNC()                                             \
  static Expected<Filters> getRemarkFilters() {                                \
    auto MaybeFunctionFilter =                                                 \
        FilterMatcher::createExactOrRE(FunctionOpt, FunctionOptRE);            \
    if (!MaybeFunctionFilter)                                                  \
      return MaybeFunctionFilter.takeError();                                  \
                                                                               \
    auto MaybeRemarkNameFilter =                                               \
        FilterMatcher::createExactOrRE(RemarkNameOpt, RemarkNameOptRE);        \
    if (!MaybeRemarkNameFilter)                                                \
      return MaybeRemarkNameFilter.takeError();                                \
                                                                               \
    auto MaybePassNameFilter =                                                 \
        FilterMatcher::createExactOrRE(PassNameOpt, PassNameOptRE);            \
    if (!MaybePassNameFilter)                                                  \
      return MaybePassNameFilter.takeError();                                  \
                                                                               \
    auto MaybeRemarkArgFilter = FilterMatcher::createExactOrRE(                \
        RemarkFilterArgByOpt, RemarkArgFilterOptRE);                           \
    if (!MaybeRemarkArgFilter)                                                 \
      return MaybeRemarkArgFilter.takeError();                                 \
                                                                               \
    std::optional<Type> TypeFilter;                                            \
    if (RemarkTypeOpt.getNumOccurrences())                                     \
      TypeFilter = RemarkTypeOpt.getValue();                                   \
                                                                               \
    return Filters{std::move(*MaybeFunctionFilter),                            \
                   std::move(*MaybeRemarkNameFilter),                          \
                   std::move(*MaybePassNameFilter),                            \
                   std::move(*MaybeRemarkArgFilter), TypeFilter};              \
  }

namespace llvm {
namespace remarks {
Expected<std::unique_ptr<MemoryBuffer>>
getInputMemoryBuffer(StringRef InputFileName);
Expected<std::unique_ptr<ToolOutputFile>>
getOutputFileWithFlags(StringRef OutputFileName, sys::fs::OpenFlags Flags);
Expected<std::unique_ptr<ToolOutputFile>>
getOutputFileForRemarks(StringRef OutputFileName, Format OutputFormat);

/// Choose the serializer format. If \p SelectedFormat is Format::Auto, try to
/// detect the format based on the extension of \p OutputFileName or fall back
/// to \p DefaultFormat.
Format getSerializerFormat(StringRef OutputFileName, Format SelectedFormat,
                           Format DefaultFormat);

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

/// Filter out remarks based on remark properties (function, remark name, pass
/// name, argument values and type).
struct Filters {
  std::optional<FilterMatcher> FunctionFilter;
  std::optional<FilterMatcher> RemarkNameFilter;
  std::optional<FilterMatcher> PassNameFilter;
  std::optional<FilterMatcher> ArgFilter;
  std::optional<Type> RemarkTypeFilter;

  /// Returns true if \p Remark satisfies all the provided filters.
  bool filterRemark(const Remark &Remark);
};

} // namespace remarks
} // namespace llvm
