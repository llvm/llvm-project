//===-- P2MachineFunctionInfo.cpp - Private data used for P2 ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "P2MachineFunctionInfo.h"

#include "P2InstrInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

P2FunctionInfo::~P2FunctionInfo() {}

void P2FunctionInfo::anchor() { }