//===- llvm/IR/RemarkStreamer.cpp - Remark Streamer -*- C++ -------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the remark outputting as part of
// LLVMContext.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/RemarkStreamer.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Remarks/BitstreamRemarkSerializer.h"
#include "llvm/Remarks/RemarkFormat.h"
#include "llvm/Remarks/RemarkSerializer.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

static cl::opt<cl::boolOrDefault> EnableRemarksSection(
    "remarks-section",
    cl::desc(
        "Emit a section containing remark diagnostics metadata. By default, "
        "this is enabled for the following formats: yaml-strtab, bitstream."),
    cl::init(cl::BOU_UNSET), cl::Hidden);

RemarkStreamer::RemarkStreamer(
    std::unique_ptr<remarks::RemarkSerializer> RemarkSerializer,
    Optional<StringRef> FilenameIn)
    : PassFilter(), RemarkSerializer(std::move(RemarkSerializer)),
      Filename(FilenameIn ? Optional<std::string>(FilenameIn->str()) : None) {}

Error RemarkStreamer::setFilter(StringRef Filter) {
  Regex R = Regex(Filter);
  std::string RegexError;
  if (!R.isValid(RegexError))
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             RegexError.data());
  PassFilter = std::move(R);
  return Error::success();
}

/// DiagnosticKind -> remarks::Type
static remarks::Type toRemarkType(enum DiagnosticKind Kind) {
  switch (Kind) {
  default:
    return remarks::Type::Unknown;
  case DK_OptimizationRemark:
  case DK_MachineOptimizationRemark:
    return remarks::Type::Passed;
  case DK_OptimizationRemarkMissed:
  case DK_MachineOptimizationRemarkMissed:
    return remarks::Type::Missed;
  case DK_OptimizationRemarkAnalysis:
  case DK_MachineOptimizationRemarkAnalysis:
    return remarks::Type::Analysis;
  case DK_OptimizationRemarkAnalysisFPCommute:
    return remarks::Type::AnalysisFPCommute;
  case DK_OptimizationRemarkAnalysisAliasing:
    return remarks::Type::AnalysisAliasing;
  case DK_OptimizationFailure:
    return remarks::Type::Failure;
  }
}

/// DiagnosticLocation -> remarks::RemarkLocation.
static Optional<remarks::RemarkLocation>
toRemarkLocation(const DiagnosticLocation &DL) {
  if (!DL.isValid())
    return None;
  StringRef File = DL.getRelativePath();
  unsigned Line = DL.getLine();
  unsigned Col = DL.getColumn();
  return remarks::RemarkLocation{File, Line, Col};
}

/// LLVM Diagnostic -> Remark
remarks::Remark
RemarkStreamer::toRemark(const DiagnosticInfoOptimizationBase &Diag) {
  remarks::Remark R; // The result.
  R.RemarkType = toRemarkType(static_cast<DiagnosticKind>(Diag.getKind()));
  R.PassName = Diag.getPassName();
  R.RemarkName = Diag.getRemarkName();
  R.FunctionName =
      GlobalValue::dropLLVMManglingEscape(Diag.getFunction().getName());
  R.Loc = toRemarkLocation(Diag.getLocation());
  R.Hotness = Diag.getHotness();

  for (const DiagnosticInfoOptimizationBase::Argument &Arg : Diag.getArgs()) {
    R.Args.emplace_back();
    R.Args.back().Key = Arg.Key;
    R.Args.back().Val = Arg.Val;
    R.Args.back().Loc = toRemarkLocation(Arg.Loc);
  }

  return R;
}

void RemarkStreamer::emit(const DiagnosticInfoOptimizationBase &Diag) {
  if (Optional<Regex> &Filter = PassFilter)
    if (!Filter->match(Diag.getPassName()))
      return;

  // First, convert the diagnostic to a remark.
  remarks::Remark R = toRemark(Diag);
  // Then, emit the remark through the serializer.
  RemarkSerializer->emit(R);
}

bool RemarkStreamer::needsSection() const {
  if (EnableRemarksSection == cl::BOU_TRUE)
    return true;

  if (EnableRemarksSection == cl::BOU_FALSE)
    return false;

  assert(EnableRemarksSection == cl::BOU_UNSET);

  // We only need a section if we're in separate mode.
  if (RemarkSerializer->Mode != remarks::SerializerMode::Separate)
    return false;

  // Only some formats need a section:
  // * bitstream
  // * yaml-strtab
  switch (RemarkSerializer->SerializerFormat) {
  case remarks::Format::YAMLStrTab:
  case remarks::Format::Bitstream:
    return true;
  default:
    return false;
  }
}

char RemarkSetupFileError::ID = 0;
char RemarkSetupPatternError::ID = 0;
char RemarkSetupFormatError::ID = 0;

Expected<std::unique_ptr<ToolOutputFile>>
llvm::setupOptimizationRemarks(LLVMContext &Context, StringRef RemarksFilename,
                               StringRef RemarksPasses, StringRef RemarksFormat,
                               bool RemarksWithHotness,
                               unsigned RemarksHotnessThreshold) {
  if (RemarksWithHotness)
    Context.setDiagnosticsHotnessRequested(true);

  if (RemarksHotnessThreshold)
    Context.setDiagnosticsHotnessThreshold(RemarksHotnessThreshold);

  if (RemarksFilename.empty())
    return nullptr;

  Expected<remarks::Format> Format = remarks::parseFormat(RemarksFormat);
  if (Error E = Format.takeError())
    return make_error<RemarkSetupFormatError>(std::move(E));

  std::error_code EC;
  auto Flags = *Format == remarks::Format::YAML ? sys::fs::OF_Text
                                                : sys::fs::OF_None;
  auto RemarksFile =
      std::make_unique<ToolOutputFile>(RemarksFilename, EC, Flags);
  // We don't use llvm::FileError here because some diagnostics want the file
  // name separately.
  if (EC)
    return make_error<RemarkSetupFileError>(errorCodeToError(EC));

  Expected<std::unique_ptr<remarks::RemarkSerializer>> RemarkSerializer =
      remarks::createRemarkSerializer(
          *Format, remarks::SerializerMode::Separate, RemarksFile->os());
  if (Error E = RemarkSerializer.takeError())
    return make_error<RemarkSetupFormatError>(std::move(E));

  Context.setRemarkStreamer(std::make_unique<RemarkStreamer>(
      std::move(*RemarkSerializer), RemarksFilename));

  if (!RemarksPasses.empty())
    if (Error E = Context.getRemarkStreamer()->setFilter(RemarksPasses))
      return make_error<RemarkSetupPatternError>(std::move(E));

  return std::move(RemarksFile);
}

Error llvm::setupOptimizationRemarks(LLVMContext &Context, raw_ostream &OS,
                                     StringRef RemarksPasses,
                                     StringRef RemarksFormat,
                                     bool RemarksWithHotness,
                                     unsigned RemarksHotnessThreshold) {
  if (RemarksWithHotness)
    Context.setDiagnosticsHotnessRequested(true);

  if (RemarksHotnessThreshold)
    Context.setDiagnosticsHotnessThreshold(RemarksHotnessThreshold);

  Expected<remarks::Format> Format = remarks::parseFormat(RemarksFormat);
  if (Error E = Format.takeError())
    return make_error<RemarkSetupFormatError>(std::move(E));

  Expected<std::unique_ptr<remarks::RemarkSerializer>> RemarkSerializer =
      remarks::createRemarkSerializer(*Format,
                                      remarks::SerializerMode::Separate, OS);
  if (Error E = RemarkSerializer.takeError())
    return make_error<RemarkSetupFormatError>(std::move(E));

  Context.setRemarkStreamer(
      std::make_unique<RemarkStreamer>(std::move(*RemarkSerializer)));

  if (!RemarksPasses.empty())
    if (Error E = Context.getRemarkStreamer()->setFilter(RemarksPasses))
      return make_error<RemarkSetupPatternError>(std::move(E));

  return Error::success();
}
