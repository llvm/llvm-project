//===-- TargetSubtargetInfo.cpp - General Target Information ---------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the general parts of a Subtarget.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/ADT/SmallVector.h"
using namespace llvm;

//---------------------------------------------------------------------------
// TargetSubtargetInfo Class
//
TargetSubtargetInfo::TargetSubtargetInfo() {}

TargetSubtargetInfo::~TargetSubtargetInfo() {}

bool TargetSubtargetInfo::enablePostRAScheduler(
          CodeGenOpt::Level OptLevel,
          AntiDepBreakMode& Mode,
          RegClassVector& CriticalPathRCs) const {
  Mode = ANTIDEP_NONE;
  CriticalPathRCs.clear();
  return false;
}

