//===- xray-converter.cc - XRay Trace Conversion --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the trace conversion functions.
//
//===----------------------------------------------------------------------===//
#include "xray-converter.h"

#include "xray-extract.h"
#include "xray-registry.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/XRay/Trace.h"
#include "llvm/XRay/YAMLXRayRecord.h"

using namespace llvm;
using namespace xray;

// llvm-xray convert
// ----------------------------------------------------------------------------
static cl::SubCommand Convert("convert", "Trace Format Conversion");
static cl::opt<std::string> ConvertInput(cl::Positional,
                                         cl::desc("<xray log file>"),
                                         cl::Required, cl::sub(Convert));
enum class ConvertFormats { BINARY, YAML };
static cl::opt<ConvertFormats> ConvertOutputFormat(
    "output-format", cl::desc("output format"),
    cl::values(clEnumValN(ConvertFormats::BINARY, "raw", "output in binary"),
               clEnumValN(ConvertFormats::YAML, "yaml", "output in yaml")),
    cl::sub(Convert));
static cl::alias ConvertOutputFormat2("f", cl::aliasopt(ConvertOutputFormat),
                                      cl::desc("Alias for -output-format"),
                                      cl::sub(Convert));
static cl::opt<std::string>
    ConvertOutput("output", cl::value_desc("output file"), cl::init("-"),
                  cl::desc("output file; use '-' for stdout"),
                  cl::sub(Convert));
static cl::alias ConvertOutput2("o", cl::aliasopt(ConvertOutput),
                                cl::desc("Alias for -output"),
                                cl::sub(Convert));

static cl::opt<bool>
    ConvertSymbolize("symbolize",
                     cl::desc("symbolize function ids from the input log"),
                     cl::init(false), cl::sub(Convert));
static cl::alias ConvertSymbolize2("y", cl::aliasopt(ConvertSymbolize),
                                   cl::desc("Alias for -symbolize"),
                                   cl::sub(Convert));

static cl::opt<std::string>
    ConvertInstrMap("instr_map",
                    cl::desc("binary with the instrumentation map, or "
                             "a separate instrumentation map"),
                    cl::value_desc("binary with xray_instr_map"),
                    cl::sub(Convert), cl::init(""));
static cl::alias ConvertInstrMap2("m", cl::aliasopt(ConvertInstrMap),
                                  cl::desc("Alias for -instr_map"),
                                  cl::sub(Convert));
static cl::opt<bool> ConvertSortInput(
    "sort",
    cl::desc("determines whether to sort input log records by timestamp"),
    cl::sub(Convert), cl::init(true));
static cl::alias ConvertSortInput2("s", cl::aliasopt(ConvertSortInput),
                                   cl::desc("Alias for -sort"),
                                   cl::sub(Convert));
static cl::opt<InstrumentationMapExtractor::InputFormats> InstrMapFormat(
    "instr-map-format", cl::desc("format of instrumentation map"),
    cl::values(clEnumValN(InstrumentationMapExtractor::InputFormats::ELF, "elf",
                          "instrumentation map in an ELF header"),
               clEnumValN(InstrumentationMapExtractor::InputFormats::YAML,
                          "yaml", "instrumentation map in YAML")),
    cl::sub(Convert), cl::init(InstrumentationMapExtractor::InputFormats::ELF));
static cl::alias InstrMapFormat2("t", cl::aliasopt(InstrMapFormat),
                                 cl::desc("Alias for -instr-map-format"),
                                 cl::sub(Convert));

using llvm::yaml::IO;
using llvm::yaml::Output;

void TraceConverter::exportAsYAML(const Trace &Records, raw_ostream &OS) {
  YAMLXRayTrace Trace;
  const auto &FH = Records.getFileHeader();
  Trace.Header = {FH.Version, FH.Type, FH.ConstantTSC, FH.NonstopTSC,
                  FH.CycleFrequency};
  Trace.Records.reserve(Records.size());
  for (const auto &R : Records) {
    Trace.Records.push_back({R.RecordType, R.CPU, R.Type, R.FuncId,
                             Symbolize ? FuncIdHelper.SymbolOrNumber(R.FuncId)
                                       : std::to_string(R.FuncId),
                             R.TSC, R.TId});
  }
  Output Out(OS);
  Out << Trace;
}

void TraceConverter::exportAsRAWv1(const Trace &Records, raw_ostream &OS) {
  // First write out the file header, in the correct endian-appropriate format
  // (XRay assumes currently little endian).
  support::endian::Writer<support::endianness::little> Writer(OS);
  const auto &FH = Records.getFileHeader();
  Writer.write(FH.Version);
  Writer.write(FH.Type);
  uint32_t Bitfield{0};
  if (FH.ConstantTSC)
    Bitfield |= 1uL;
  if (FH.NonstopTSC)
    Bitfield |= 1uL << 1;
  Writer.write(Bitfield);
  Writer.write(FH.CycleFrequency);

  // There's 16 bytes of padding at the end of the file header.
  static constexpr uint32_t Padding4B = 0;
  Writer.write(Padding4B);
  Writer.write(Padding4B);
  Writer.write(Padding4B);
  Writer.write(Padding4B);

  // Then write out the rest of the records, still in an endian-appropriate
  // format.
  for (const auto &R : Records) {
    Writer.write(R.RecordType);
    Writer.write(R.CPU);
    switch (R.Type) {
    case RecordTypes::ENTER:
      Writer.write(uint8_t{0});
      break;
    case RecordTypes::EXIT:
      Writer.write(uint8_t{1});
      break;
    }
    Writer.write(R.FuncId);
    Writer.write(R.TSC);
    Writer.write(R.TId);
    Writer.write(Padding4B);
    Writer.write(Padding4B);
    Writer.write(Padding4B);
  }
}

namespace llvm {
namespace xray {

static CommandRegistration Unused(&Convert, []() -> Error {
  // FIXME: Support conversion to BINARY when upgrading XRay trace versions.
  int Fd;
  auto EC = sys::fs::openFileForRead(ConvertInput, Fd);
  if (EC)
    return make_error<StringError>(
        Twine("Cannot open file '") + ConvertInput + "'", EC);

  Error Err = Error::success();
  xray::InstrumentationMapExtractor Extractor(ConvertInstrMap, InstrMapFormat,
                                              Err);
  handleAllErrors(std::move(Err),
                  [&](const ErrorInfoBase &E) { E.log(errs()); });

  const auto &FunctionAddresses = Extractor.getFunctionAddresses();
  symbolize::LLVMSymbolizer::Options Opts(
      symbolize::FunctionNameKind::LinkageName, true, true, false, "");
  symbolize::LLVMSymbolizer Symbolizer(Opts);
  llvm::xray::FuncIdConversionHelper FuncIdHelper(ConvertInstrMap, Symbolizer,
                                                  FunctionAddresses);
  llvm::xray::TraceConverter TC(FuncIdHelper, ConvertSymbolize);
  raw_fd_ostream OS(ConvertOutput, EC,
                    ConvertOutputFormat == ConvertFormats::BINARY
                        ? sys::fs::OpenFlags::F_None
                        : sys::fs::OpenFlags::F_Text);
  if (EC)
    return make_error<StringError>(
        Twine("Cannot open file '") + ConvertOutput + "' for writing.", EC);

  if (auto TraceOrErr = loadTraceFile(ConvertInput, ConvertSortInput)) {
    auto &T = *TraceOrErr;
    switch (ConvertOutputFormat) {
    case ConvertFormats::YAML:
      TC.exportAsYAML(T, OS);
      break;
    case ConvertFormats::BINARY:
      TC.exportAsRAWv1(T, OS);
      break;
    }
  } else {
    return joinErrors(
        make_error<StringError>(
            Twine("Failed loading input file '") + ConvertInput + "'.",
            std::make_error_code(std::errc::executable_format_error)),
        TraceOrErr.takeError());
  }
  return Error::success();
});

} // namespace xray
} // namespace llvm
