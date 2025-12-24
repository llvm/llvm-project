//===- RemarkUtilHelpers.cpp ----------------------------------------------===//
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
#include "RemarkUtilHelpers.h"

namespace llvm {
namespace remarks {
/// \returns A MemoryBuffer for the input file on success, and an Error
/// otherwise.
Expected<std::unique_ptr<MemoryBuffer>>
getInputMemoryBuffer(StringRef InputFileName) {
  auto MaybeBuf = MemoryBuffer::getFileOrSTDIN(InputFileName);
  if (auto ErrorCode = MaybeBuf.getError())
    return createStringError(ErrorCode,
                             Twine("Cannot open file '" + InputFileName +
                                   "': " + ErrorCode.message()));
  return std::move(*MaybeBuf);
}

/// \returns A ToolOutputFile which can be used for outputting the results of
/// some tool mode.
/// \p OutputFileName is the desired destination.
/// \p Flags controls whether or not the file is opened for writing in text
/// mode, as a binary, etc. See sys::fs::OpenFlags for more detail.
Expected<std::unique_ptr<ToolOutputFile>>
getOutputFileWithFlags(StringRef OutputFileName, sys::fs::OpenFlags Flags) {
  if (OutputFileName == "")
    OutputFileName = "-";
  std::error_code ErrorCode;
  auto OF = std::make_unique<ToolOutputFile>(OutputFileName, ErrorCode, Flags);
  if (ErrorCode)
    return errorCodeToError(ErrorCode);
  return std::move(OF);
}

/// \returns A ToolOutputFile which can be used for writing remarks on success,
/// and an Error otherwise.
/// \p OutputFileName is the desired destination.
/// \p OutputFormat
Expected<std::unique_ptr<ToolOutputFile>>
getOutputFileForRemarks(StringRef OutputFileName, Format OutputFormat) {
  assert((OutputFormat == Format::YAML || OutputFormat == Format::Bitstream) &&
         "Expected one of YAML or Bitstream!");
  return getOutputFileWithFlags(OutputFileName, OutputFormat == Format::YAML
                                                    ? sys::fs::OF_TextWithCRLF
                                                    : sys::fs::OF_None);
}

Format getSerializerFormat(StringRef OutputFileName, Format SelectedFormat,
                           Format DefaultFormat) {
  if (SelectedFormat != Format::Auto)
    return SelectedFormat;
  SelectedFormat = DefaultFormat;
  if (OutputFileName.empty() || OutputFileName == "-" ||
      OutputFileName.ends_with_insensitive(".yaml") ||
      OutputFileName.ends_with_insensitive(".yml"))
    SelectedFormat = Format::YAML;
  if (OutputFileName.ends_with_insensitive(".bitstream"))
    SelectedFormat = Format::Bitstream;
  return SelectedFormat;
}

Expected<FilterMatcher>
FilterMatcher::createRE(const llvm::cl::opt<std::string> &Arg) {
  return createRE(Arg.ArgStr, Arg);
}

Expected<FilterMatcher>
FilterMatcher::createRE(StringRef Filter, const cl::list<std::string> &Arg) {
  return createRE(Arg.ArgStr, Filter);
}

Expected<FilterMatcher> FilterMatcher::createRE(StringRef Arg,
                                                StringRef Value) {
  FilterMatcher FM(Value, true);
  std::string Error;
  if (!FM.FilterRE.isValid(Error))
    return createStringError(make_error_code(std::errc::invalid_argument),
                             "invalid argument '--" + Arg + "=" + Value +
                                 "': " + Error);
  return std::move(FM);
}

Expected<std::optional<FilterMatcher>>
FilterMatcher::createExactOrRE(const llvm::cl::opt<std::string> &ExactArg,
                               const llvm::cl::opt<std::string> &REArg) {
  if (!ExactArg.empty() && !REArg.empty())
    return createStringError(make_error_code(std::errc::invalid_argument),
                             "conflicting arguments: --" + ExactArg.ArgStr +
                                 " and --" + REArg.ArgStr);

  if (!ExactArg.empty())
    return createExact(ExactArg);

  if (!REArg.empty())
    return createRE(REArg);

  return std::nullopt;
}

bool Filters::filterRemark(const Remark &Remark) {
  if (FunctionFilter && !FunctionFilter->match(Remark.FunctionName))
    return false;
  if (RemarkNameFilter && !RemarkNameFilter->match(Remark.RemarkName))
    return false;
  if (PassNameFilter && !PassNameFilter->match(Remark.PassName))
    return false;
  if (RemarkTypeFilter)
    return *RemarkTypeFilter == Remark.RemarkType;
  if (ArgFilter) {
    if (!any_of(Remark.Args,
                [this](Argument Arg) { return ArgFilter->match(Arg.Val); }))
      return false;
  }
  return true;
}

} // namespace remarks
} // namespace llvm
