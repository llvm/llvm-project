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
/// taking the command line options for filtering the remark based on name, pass
/// name, type and argumetns using string matching or regular expressions and
/// construct a Remark Filter object which can filter the remarks based on the
/// specified properties.
Expected<Filters> getRemarkFilter(cl::opt<std::string> &RemarkNameOpt,
                                  cl::opt<std::string> &RemarkNameOptRE,
                                  cl::opt<std::string> &PassNameOpt,
                                  cl::opt<std::string> &PassNameOptRE,
                                  cl::opt<Type> &RemarkTypeOpt,
                                  cl::opt<std::string> &RemarkFilterArgByOpt,
                                  cl::opt<std::string> &RemarkArgFilterOptRE) {
  // Create Filter properties.
  std::optional<FilterMatcher> RemarkNameFilter;
  std::optional<FilterMatcher> PassNameFilter;
  std::optional<FilterMatcher> RemarkArgFilter;
  std::optional<Type> RemarkType;
  if (!RemarkNameOpt.empty())
    RemarkNameFilter = {RemarkNameOpt, false};
  else if (!RemarkNameOptRE.empty())
    RemarkNameFilter = {RemarkNameOptRE, true};
  if (!PassNameOpt.empty())
    PassNameFilter = {PassNameOpt, false};
  else if (!PassNameOptRE.empty())
    PassNameFilter = {PassNameOptRE, true};
  if (RemarkTypeOpt != Type::Failure)
    RemarkType = RemarkTypeOpt;
  if (!RemarkFilterArgByOpt.empty())
    RemarkArgFilter = {RemarkFilterArgByOpt, false};
  else if (!RemarkArgFilterOptRE.empty())
    RemarkArgFilter = {RemarkArgFilterOptRE, true};
  // Create RemarkFilter.
  return Filters::createRemarkFilter(std::move(RemarkNameFilter),
                                     std::move(PassNameFilter),
                                     std::move(RemarkArgFilter), RemarkType);
}

Error Filters::regexArgumentsValid() {
  if (RemarkNameFilter && RemarkNameFilter->IsRegex)
    if (auto E = checkRegex(RemarkNameFilter->FilterRE))
      return E;
  if (PassNameFilter && PassNameFilter->IsRegex)
    if (auto E = checkRegex(PassNameFilter->FilterRE))
      return E;
  if (ArgFilter && ArgFilter->IsRegex)
    if (auto E = checkRegex(ArgFilter->FilterRE))
      return E;
  return Error::success();
}

bool Filters::filterRemark(const Remark &Remark) {
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
} // namespace remarks
} // namespace llvm
