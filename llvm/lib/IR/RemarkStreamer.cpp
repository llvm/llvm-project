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
      YAMLOutput(OS, reinterpret_cast<void *>(this)) {
  assert(!Filename.empty() && "This needs to be a real filename.");
}

void RemarkStreamer::emit(const DiagnosticInfoOptimizationBase &Diag) {
  DiagnosticInfoOptimizationBase *DiagPtr =
      const_cast<DiagnosticInfoOptimizationBase *>(&Diag);
  YAMLOutput << DiagPtr;
}
