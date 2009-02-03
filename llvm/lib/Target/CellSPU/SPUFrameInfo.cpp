//===-- SPUTargetMachine.cpp - Define TargetMachine for Cell SPU ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the Cell SPU target.
//
//===----------------------------------------------------------------------===//

#include "SPU.h"
#include "SPUFrameInfo.h"
#include "SPURegisterNames.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// SPUFrameInfo:
//===----------------------------------------------------------------------===//

SPUFrameInfo::SPUFrameInfo(const TargetMachine &tm):
  TargetFrameInfo(TargetFrameInfo::StackGrowsDown, 16, 0),
  TM(tm)
{
  LR[0].first = SPU::R0;
  LR[0].second = 16;
}
