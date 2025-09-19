//===- MCCodeEmitter.cpp - Instruction Encoding ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace llvm;

MCCodeEmitter::MCCodeEmitter() = default;

MCCodeEmitter::~MCCodeEmitter() = default;

void MCCodeEmitter::ReportFatalError(const MCInst &Inst,
                                     std::optional<unsigned> OpNum) {
  std::string msg;
  raw_string_ostream Msg(msg);
  Msg << "Unsupported instruction " << Inst;
  if (OpNum)
    Msg << ", OpNum = " << *OpNum;
  report_fatal_error(msg.c_str());
}
