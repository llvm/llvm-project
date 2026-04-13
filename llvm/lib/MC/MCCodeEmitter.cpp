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

void MCCodeEmitter::reportUnsupportedInst(const MCInst &Inst) {
  std::string Msg;
  raw_string_ostream OS(Msg);
  OS << "Unsupported instruction : " << Inst;
  reportFatalInternalError(Msg.c_str());
}

void MCCodeEmitter::reportUnsupportedOperand(const MCInst &Inst,
                                             unsigned OpNum) {
  std::string Msg;
  raw_string_ostream OS(Msg);
  OS << "Unsupported instruction operand : \"" << Inst << "\"[" << OpNum << "]";
  reportFatalInternalError(Msg.c_str());
}
