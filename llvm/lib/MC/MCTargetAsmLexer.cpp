//===-- llvm/MC/MCTargetAsmLexer.cpp - Target Assembly Lexer --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCTargetAsmLexer.h"
using namespace llvm;

MCTargetAsmLexer::MCTargetAsmLexer(const Target &T)
  : TheTarget(T), Lexer(NULL) {
}
MCTargetAsmLexer::~MCTargetAsmLexer() {}
