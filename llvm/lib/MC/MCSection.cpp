//===- lib/MC/MCSection.cpp - Machine Code Section Representation ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// MCSection
//===----------------------------------------------------------------------===//

MCSection::~MCSection() {
}

