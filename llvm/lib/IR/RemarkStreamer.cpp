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
  // Re-use the buffer.
  TmpArgs.clear();

  remarks::Remark R; // The result.
  R.RemarkType = toRemarkType(static_cast<DiagnosticKind>(Diag.getKind()));
  R.PassName = Diag.getPassName();
  R.RemarkName = Diag.getRemarkName();
  R.FunctionName =
      GlobalValue::dropLLVMManglingEscape(Diag.getFunction().getName());
  R.Loc = toRemarkLocation(Diag.getLocation());
  R.Hotness = Diag.getHotness();

  // Use TmpArgs to build the list of arguments and re-use the memory allocated
  // from previous remark conversions.
  for (const DiagnosticInfoOptimizationBase::Argument &Arg : Diag.getArgs()) {
    TmpArgs.emplace_back();
    TmpArgs.back().Key = Arg.Key;
    TmpArgs.back().Val = Arg.Val;
    TmpArgs.back().Loc = toRemarkLocation(Arg.Loc);
  }
  R.Args = TmpArgs; // This is valid until the next call to this function.

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
