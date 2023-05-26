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
static cl::SubCommand InstructionCount(
    "instruction-count",
    "Function instruction count information (requires asm-printer remarks)");
static cl::SubCommand
    AnnotationCount("annotation-count",
                    "Collect count information from annotation remarks (uses "
                    "AnnotationRemarksPass)");
} // namespace subopts

// Keep input + output help + names consistent across the various modes via a
// hideous macro.
#define INPUT_OUTPUT_COMMAND_LINE_OPTIONS(SUBOPT)                              \
  static cl::opt<std::string> InputFileName(                                   \
      cl::Positional, cl::cat(RemarkUtilCategory), cl::init("-"),              \
      cl::desc("<input file>"), cl::sub(SUBOPT));                              \
  static cl::opt<std::string> OutputFileName(                                  \
      "o", cl::init("-"), cl::cat(RemarkUtilCategory), cl::desc("Output"),     \
      cl::value_desc("filename"), cl::sub(SUBOPT));

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
namespace yaml2bitstream {
/// Remark format to parse.
static constexpr Format InputFormat = Format::YAML;
/// Remark format to output.
static constexpr Format OutputFormat = Format::Bitstream;
INPUT_OUTPUT_COMMAND_LINE_OPTIONS(subopts::YAML2Bitstream)
} // namespace yaml2bitstream

namespace bitstream2yaml {
/// Remark format to parse.
static constexpr Format InputFormat = Format::Bitstream;
/// Remark format to output.
static constexpr Format OutputFormat = Format::YAML;
INPUT_OUTPUT_COMMAND_LINE_OPTIONS(subopts::Bitstream2YAML)
} // namespace bitstream2yaml

namespace instructioncount {
INPUT_FORMAT_COMMAND_LINE_OPTIONS(subopts::InstructionCount)
INPUT_OUTPUT_COMMAND_LINE_OPTIONS(subopts::InstructionCount)
DEBUG_LOC_INFO_COMMAND_LINE_OPTIONS(subopts::InstructionCount)
} // namespace instructioncount

namespace annotationcount {
INPUT_FORMAT_COMMAND_LINE_OPTIONS(subopts::AnnotationCount)
static cl::opt<std::string> AnnotationTypeToCollect(
    "annotation-type", cl::desc("annotation-type remark to collect count for"),
    cl::sub(subopts::AnnotationCount));
INPUT_OUTPUT_COMMAND_LINE_OPTIONS(subopts::AnnotationCount)
DEBUG_LOC_INFO_COMMAND_LINE_OPTIONS(subopts::AnnotationCount)
} // namespace annotationcount

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

/// \returns A ToolOutputFile which can be used for outputting the results of
/// some tool mode.
/// \p OutputFileName is the desired destination.
/// \p Flags controls whether or not the file is opened for writing in text
/// mode, as a binary, etc. See sys::fs::OpenFlags for more detail.
static Expected<std::unique_ptr<ToolOutputFile>>
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
static Expected<std::unique_ptr<ToolOutputFile>>
getOutputFileForRemarks(StringRef OutputFileName, Format OutputFormat) {
  assert((OutputFormat == Format::YAML || OutputFormat == Format::Bitstream) &&
         "Expected one of YAML or Bitstream!");
  return getOutputFileWithFlags(OutputFileName, OutputFormat == Format::YAML
                                                    ? sys::fs::OF_TextWithCRLF
                                                    : sys::fs::OF_None);
}

static bool shouldSkipRemark(bool UseDebugLoc, Remark &Remark) {
  return UseDebugLoc && !Remark.Loc.has_value();
}

namespace yaml2bitstream {
/// Parses all remarks in the input YAML file.
/// \p [out] ParsedRemarks - Filled with remarks parsed from the input file.
/// \p [out] StrTab - A string table populated for later remark serialization.
/// \returns Error::success() if all remarks were successfully parsed, and an
/// Error otherwise.
static Error
tryParseRemarksFromYAMLFile(std::vector<std::unique_ptr<Remark>> &ParsedRemarks,
                            StringTable &StrTab) {
  auto MaybeBuf = getInputMemoryBuffer(InputFileName);
  if (!MaybeBuf)
    return MaybeBuf.takeError();
  auto MaybeParser = createRemarkParser(InputFormat, (*MaybeBuf)->getBuffer());
  if (!MaybeParser)
    return MaybeParser.takeError();
  auto &Parser = **MaybeParser;
  auto MaybeRemark = Parser.next();
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

/// Reserialize a list of parsed YAML remarks into bitstream remarks.
/// \p ParsedRemarks - A list of remarks.
/// \p StrTab - The string table for the remarks.
/// \returns Error::success() on success.
static Error tryReserializeYAML2Bitstream(
    const std::vector<std::unique_ptr<Remark>> &ParsedRemarks,
    StringTable &StrTab) {
  auto MaybeOF = getOutputFileForRemarks(OutputFileName, OutputFormat);
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

/// Parse YAML remarks and reserialize as bitstream remarks.
/// \returns Error::success() on success, and an Error otherwise.
static Error tryYAML2Bitstream() {
  StringTable StrTab;
  std::vector<std::unique_ptr<Remark>> ParsedRemarks;
  ExitOnErr(tryParseRemarksFromYAMLFile(ParsedRemarks, StrTab));
  return tryReserializeYAML2Bitstream(ParsedRemarks, StrTab);
}
} // namespace yaml2bitstream

namespace bitstream2yaml {
/// Parse bitstream remarks and reserialize as YAML remarks.
/// \returns An Error if reserialization fails, or Error::success() on success.
static Error tryBitstream2YAML() {
  // Create the serializer.
  auto MaybeOF = getOutputFileForRemarks(OutputFileName, OutputFormat);
  if (!MaybeOF)
    return MaybeOF.takeError();
  auto OF = std::move(*MaybeOF);
  auto MaybeSerializer = createRemarkSerializer(
      OutputFormat, SerializerMode::Standalone, OF->os());
  if (!MaybeSerializer)
    return MaybeSerializer.takeError();

  // Create the parser.
  auto MaybeBuf = getInputMemoryBuffer(InputFileName);
  if (!MaybeBuf)
    return MaybeBuf.takeError();
  auto Serializer = std::move(*MaybeSerializer);
  auto MaybeParser = createRemarkParser(InputFormat, (*MaybeBuf)->getBuffer());
  if (!MaybeParser)
    return MaybeParser.takeError();
  auto &Parser = **MaybeParser;

  // Parse + reserialize all remarks.
  auto MaybeRemark = Parser.next();
  for (; MaybeRemark; MaybeRemark = Parser.next())
    Serializer->emit(**MaybeRemark);
  auto E = MaybeRemark.takeError();
  if (!E.isA<EndOfFileError>())
    return E;
  consumeError(std::move(E));
  return Error::success();
}
} // namespace bitstream2yaml

namespace instructioncount {
/// Outputs all instruction count remarks in the file as a CSV.
/// \returns Error::success() on success, and an Error otherwise.
static Error tryInstructionCount() {
  // Create the output buffer.
  auto MaybeOF = getOutputFileWithFlags(OutputFileName,
                                        /*Flags = */ sys::fs::OF_TextWithCRLF);
  if (!MaybeOF)
    return MaybeOF.takeError();
  auto OF = std::move(*MaybeOF);
  // Create a parser for the user-specified input format.
  auto MaybeBuf = getInputMemoryBuffer(InputFileName);
  if (!MaybeBuf)
    return MaybeBuf.takeError();
  auto MaybeParser = createRemarkParser(InputFormat, (*MaybeBuf)->getBuffer());
  if (!MaybeParser)
    return MaybeParser.takeError();
  // Emit CSV header.
  if (UseDebugLoc)
    OF->os() << "Source,";
  OF->os() << "Function,InstructionCount\n";
  // Parse all remarks. Whenever we see an instruction count remark, output
  // the file name and the number of instructions.
  auto &Parser = **MaybeParser;
  auto MaybeRemark = Parser.next();
  for (; MaybeRemark; MaybeRemark = Parser.next()) {
    auto &Remark = **MaybeRemark;
    if (Remark.RemarkName != "InstructionCount")
      continue;
    if (shouldSkipRemark(UseDebugLoc, Remark))
      continue;
    auto *InstrCountArg = find_if(Remark.Args, [](const Argument &Arg) {
      return Arg.Key == "NumInstructions";
    });
    assert(InstrCountArg != Remark.Args.end() &&
           "Expected instruction count remarks to have a NumInstructions key?");
    if (UseDebugLoc) {
      std::string Loc = Remark.Loc->SourceFilePath.str() + ":" +
                        std::to_string(Remark.Loc->SourceLine) + +":" +
                        std::to_string(Remark.Loc->SourceColumn);
      OF->os() << Loc << ",";
    }
    OF->os() << Remark.FunctionName << "," << InstrCountArg->Val << "\n";
  }
  auto E = MaybeRemark.takeError();
  if (!E.isA<EndOfFileError>())
    return E;
  consumeError(std::move(E));
  OF->keep();
  return Error::success();
}
} // namespace instructioncount

namespace annotationcount {
static Error tryAnnotationCount() {
  // Create the output buffer.
  auto MaybeOF = getOutputFileWithFlags(OutputFileName,
                                        /*Flags = */ sys::fs::OF_TextWithCRLF);
  if (!MaybeOF)
    return MaybeOF.takeError();
  auto OF = std::move(*MaybeOF);
  // Create a parser for the user-specified input format.
  auto MaybeBuf = getInputMemoryBuffer(InputFileName);
  if (!MaybeBuf)
    return MaybeBuf.takeError();
  auto MaybeParser = createRemarkParser(InputFormat, (*MaybeBuf)->getBuffer());
  if (!MaybeParser)
    return MaybeParser.takeError();
  // Emit CSV header.
  if (UseDebugLoc)
    OF->os() << "Source,";
  OF->os() << "Function,Count\n";
  // Parse all remarks. When we see the specified remark collect the count
  // information.
  auto &Parser = **MaybeParser;
  auto MaybeRemark = Parser.next();
  for (; MaybeRemark; MaybeRemark = Parser.next()) {
    auto &Remark = **MaybeRemark;
    if (Remark.RemarkName != "AnnotationSummary")
      continue;
    if (shouldSkipRemark(UseDebugLoc, Remark))
      continue;
    auto *RemarkNameArg = find_if(Remark.Args, [](const Argument &Arg) {
      return Arg.Key == "type" && Arg.Val == AnnotationTypeToCollect;
    });
    if (RemarkNameArg == Remark.Args.end())
      continue;
    auto *CountArg = find_if(
        Remark.Args, [](const Argument &Arg) { return Arg.Key == "count"; });
    assert(CountArg != Remark.Args.end() &&
           "Expected annotation-type remark to have a count key?");
    if (UseDebugLoc) {
      std::string Loc = Remark.Loc->SourceFilePath.str() + ":" +
                        std::to_string(Remark.Loc->SourceLine) + +":" +
                        std::to_string(Remark.Loc->SourceColumn);
      OF->os() << Loc << ",";
    }
    OF->os() << Remark.FunctionName << "," << CountArg->Val << "\n";
  }
  auto E = MaybeRemark.takeError();
  if (!E.isA<EndOfFileError>())
    return E;
  consumeError(std::move(E));
  OF->keep();
  return Error::success();
}

} // namespace annotationcount
/// Handle user-specified suboptions (e.g. yaml2bitstream, bitstream2yaml).
/// \returns An Error if the specified suboption fails or if no suboption was
/// specified. Otherwise, Error::success().
static Error handleSuboptions() {
  if (subopts::Bitstream2YAML)
    return bitstream2yaml::tryBitstream2YAML();
  if (subopts::YAML2Bitstream)
    return yaml2bitstream::tryYAML2Bitstream();
  if (subopts::InstructionCount)
    return instructioncount::tryInstructionCount();
  if (subopts::AnnotationCount)
    return annotationcount::tryAnnotationCount();

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
