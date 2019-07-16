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
#include "llvm/Remarks/RemarkFormat.h"

using namespace llvm;

RemarkStreamer::RemarkStreamer(StringRef Filename,
                               std::unique_ptr<remarks::Serializer> Serializer)
    : Filename(Filename), PassFilter(), Serializer(std::move(Serializer)) {
  assert(!Filename.empty() && "This needs to be a real filename.");
}

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
  Serializer->emit(R);
}

char RemarkSetupFileError::ID = 0;
char RemarkSetupPatternError::ID = 0;
char RemarkSetupFormatError::ID = 0;

static std::unique_ptr<remarks::Serializer>
formatToSerializer(remarks::Format RemarksFormat, raw_ostream &OS) {
  switch (RemarksFormat) {
  default:
    llvm_unreachable("Unknown remark serializer format.");
    return nullptr;
  case remarks::Format::YAML:
    return llvm::make_unique<remarks::YAMLSerializer>(OS);
  };
}

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

  std::error_code EC;
  auto RemarksFile =
      llvm::make_unique<ToolOutputFile>(RemarksFilename, EC, sys::fs::F_None);
  // We don't use llvm::FileError here because some diagnostics want the file
  // name separately.
  if (EC)
    return make_error<RemarkSetupFileError>(errorCodeToError(EC));

  Expected<remarks::Format> Format = remarks::parseFormat(RemarksFormat);
  if (Error E = Format.takeError())
    return make_error<RemarkSetupFormatError>(std::move(E));

  Context.setRemarkStreamer(llvm::make_unique<RemarkStreamer>(
      RemarksFilename, formatToSerializer(*Format, RemarksFile->os())));

  if (!RemarksPasses.empty())
    if (Error E = Context.getRemarkStreamer()->setFilter(RemarksPasses))
      return make_error<RemarkSetupPatternError>(std::move(E));

  return std::move(RemarksFile);
}
