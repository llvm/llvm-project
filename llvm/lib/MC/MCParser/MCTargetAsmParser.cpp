//===-- MCTargetAsmParser.cpp - Target Assembly Parser ---------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCTargetAsmParser.h"
using namespace llvm;

MCTargetAsmParser::MCTargetAsmParser(MCTargetOptions const &MCOptions)
  : AvailableFeatures(0), ParsingInlineAsm(false), MCOptions(MCOptions)
{
}

MCTargetAsmParser::~MCTargetAsmParser() {
}
