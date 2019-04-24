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

using namespace llvm;

RemarkStreamer::RemarkStreamer(StringRef Filename, raw_ostream &OS)
    : Filename(Filename), OS(OS),
      YAMLOutput(OS, reinterpret_cast<void *>(this)), StrTab() {
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

void RemarkStreamer::emit(const DiagnosticInfoOptimizationBase &Diag) {
  if (Optional<Regex> &Filter = PassFilter)
    if (!Filter->match(Diag.getPassName()))
      return;

  DiagnosticInfoOptimizationBase *DiagPtr =
      const_cast<DiagnosticInfoOptimizationBase *>(&Diag);
  YAMLOutput << DiagPtr;
}
