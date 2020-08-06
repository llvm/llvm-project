//===-- P2TargetObjectFile.cpp - P2 Object Files ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "P2TargetObjectFile.h"

#include "P2TargetMachine.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

void P2TargetObjectFile::Initialize(MCContext &Ctx, const TargetMachine &TM){
    Base::Initialize(Ctx, TM);
    ProgmemDataSection = Ctx.getELFSection(".progmem.data", ELF::SHT_PROGBITS, ELF::SHF_ALLOC);
}