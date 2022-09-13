//===--------- llvm-remarkutil/RemarkUtil.cpp -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// Utility for remark files.
//===----------------------------------------------------------------------===//

#include "llvm-c/Remarks.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Remarks/Remark.h"
#include "llvm/Remarks/RemarkFormat.h"
#include "llvm/Remarks/RemarkParser.h"
#include "llvm/Remarks/YAMLRemarkSerializer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;
using namespace remarks;

static ExitOnError ExitOnErr;
static cl::OptionCategory RemarkUtilCategory("llvm-remarkutil options");
namespace subopts {
static cl::SubCommand
    YAML2Bitstream("yaml2bitstream",
                   "Convert YAML remarks to bitstream remarks");
static cl::SubCommand
    Bitstream2YAML("bitstream2yaml",
                   "Convert bitstream remarks to YAML remarks");
} // namespace subopts

// Conversions have the same command line options. AFAIK there is no way to
// reuse them, so to avoid duplication, let's just stick this in a hideous
// macro.
#define CONVERSION_COMMAND_LINE_OPTIONS(SUBOPT)                                \
  static cl::opt<std::string> InputFileName(                                   \
      cl::Positional, cl::cat(RemarkUtilCategory), cl::init("-"),              \
      cl::desc("<input file>"), cl::sub(SUBOPT));                              \
  static cl::opt<std::string> OutputFileName(                                  \
      "o", cl::init("-"), cl::cat(RemarkUtilCategory), cl::desc("Output"),     \
      cl::value_desc("filename"), cl::sub(SUBOPT));
namespace yaml2bitstream {
/// Remark format to parse.
static constexpr Format InputFormat = Format::YAML;
/// Remark format to output.
static constexpr Format OutputFormat = Format::Bitstream;
CONVERSION_COMMAND_LINE_OPTIONS(subopts::YAML2Bitstream)
} // namespace yaml2bitstream

namespace bitstream2yaml {
/// Remark format to parse.
static constexpr Format InputFormat = Format::Bitstream;
/// Remark format to output.
static constexpr Format OutputFormat = Format::YAML;
CONVERSION_COMMAND_LINE_OPTIONS(subopts::Bitstream2YAML)
} // namespace bitstream2yaml

/// \returns A MemoryBuffer for the input file on success, and an Error
/// otherwise.
static Expected<std::unique_ptr<MemoryBuffer>>
getInputMemoryBuffer(StringRef InputFileName) {
  auto MaybeBuf = MemoryBuffer::getFileOrSTDIN(InputFileName);
  if (auto ErrorCode = MaybeBuf.getError())
    return createStringError(ErrorCode,
                             Twine("Cannot open file '" + InputFileName +
                                   "': " + ErrorCode.message()));
  return std::move(*MaybeBuf);
}

/// Parses all remarks in the input file.
/// \p [out] ParsedRemarks - Filled with remarks parsed from the input file.
/// \p [out] StrTab - A string table populated for later remark serialization.
/// \returns Error::success() if all remarks were successfully parsed, and an
/// Error otherwise.
static Error tryParseRemarksFromInputFile(
    StringRef InputFileName, Format InputFormat,
    std::vector<std::unique_ptr<Remark>> &ParsedRemarks, StringTable &StrTab) {
  auto MaybeBuf = getInputMemoryBuffer(InputFileName);
  if (!MaybeBuf)
    return MaybeBuf.takeError();
  auto MaybeParser = createRemarkParser(InputFormat, (*MaybeBuf)->getBuffer());
  if (!MaybeParser)
    return MaybeParser.takeError();
  auto &Parser = **MaybeParser;
  auto MaybeRemark = Parser.next();
  // TODO: If we are converting from bitstream to YAML, we don't need to parse
  // early because the string table is not necessary.
  for (; MaybeRemark; MaybeRemark = Parser.next()) {
    StrTab.internalize(**MaybeRemark);
    ParsedRemarks.push_back(std::move(*MaybeRemark));
  }
  auto E = MaybeRemark.takeError();
  if (!E.isA<EndOfFileError>())
    return E;
  consumeError(std::move(E));
  return Error::success();
}

/// \returns A ToolOutputFile which can be used for writing remarks on success,
/// and an Error otherwise.
static Expected<std::unique_ptr<ToolOutputFile>>
getOutputFile(StringRef OutputFileName, Format OutputFormat) {
  if (OutputFileName == "")
    OutputFileName = "-";
  auto Flags = OutputFormat == Format::YAML ? sys::fs::OF_TextWithCRLF
                                            : sys::fs::OF_None;
  std::error_code ErrorCode;
  auto OF = std::make_unique<ToolOutputFile>(OutputFileName, ErrorCode, Flags);
  if (ErrorCode)
    return errorCodeToError(ErrorCode);
  return std::move(OF);
}

/// Reserialize a list of remarks into the desired output format, and output
/// to the user-specified output file.
/// \p ParsedRemarks - A list of remarks.
/// \p StrTab - The string table for the remarks.
/// \returns Error::success() on success.
static Error tryReserializeParsedRemarks(
    StringRef OutputFileName, Format OutputFormat,
    const std::vector<std::unique_ptr<Remark>> &ParsedRemarks,
    StringTable &StrTab) {
  auto MaybeOF = getOutputFile(OutputFileName, OutputFormat);
  if (!MaybeOF)
    return MaybeOF.takeError();
  auto OF = std::move(*MaybeOF);
  auto MaybeSerializer = createRemarkSerializer(
      OutputFormat, SerializerMode::Standalone, OF->os(), std::move(StrTab));
  if (!MaybeSerializer)
    return MaybeSerializer.takeError();
  auto Serializer = std::move(*MaybeSerializer);
  for (const auto &Remark : ParsedRemarks)
    Serializer->emit(*Remark);
  OF->keep();
  return Error::success();
}

/// Parses remarks in the input format, and reserializes them in the desired
/// output format.
/// \returns Error::success() on success, and an Error otherwise.
static Error tryReserialize(StringRef InputFileName, StringRef OutputFileName,
                            Format InputFormat, Format OutputFormat) {
  StringTable StrTab;
  std::vector<std::unique_ptr<Remark>> ParsedRemarks;
  ExitOnErr(tryParseRemarksFromInputFile(InputFileName, InputFormat,
                                         ParsedRemarks, StrTab));
  return tryReserializeParsedRemarks(OutputFileName, OutputFormat,
                                     ParsedRemarks, StrTab);
}

/// Reserialize bitstream remarks as YAML remarks.
/// \returns An Error if reserialization fails, or Error::success() on success.
static Error tryBitstream2YAML() {
  // Use the namespace to get the correct command line globals.
  using namespace bitstream2yaml;
  return tryReserialize(InputFileName, OutputFileName, InputFormat,
                        OutputFormat);
}

/// Reserialize YAML remarks as bitstream remarks.
/// \returns An Error if reserialization fails, or Error::success() on success.
static Error tryYAML2Bitstream() {
  // Use the namespace to get the correct command line globals.
  using namespace yaml2bitstream;
  return tryReserialize(InputFileName, OutputFileName, InputFormat,
                        OutputFormat);
}

/// Handle user-specified suboptions (e.g. yaml2bitstream, bitstream2yaml).
/// \returns An Error if the specified suboption fails or if no suboption was
/// specified. Otherwise, Error::success().
static Error handleSuboptions() {
  if (subopts::Bitstream2YAML)
    return tryBitstream2YAML();
  if (subopts::YAML2Bitstream)
    return tryYAML2Bitstream();
  return make_error<StringError>(
      "Please specify a subcommand. (See -help for options)",
      inconvertibleErrorCode());
}

int main(int argc, const char **argv) {
  InitLLVM X(argc, argv);
  cl::HideUnrelatedOptions(RemarkUtilCategory);
  cl::ParseCommandLineOptions(argc, argv, "Remark file utilities\n");
  ExitOnErr.setBanner(std::string(argv[0]) + ": error: ");
  ExitOnErr(handleSuboptions());
}
