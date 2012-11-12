//===-- MBlazeSelectionDAGInfo.cpp - MBlaze SelectionDAG Info -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MBlazeSelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mblaze-selectiondag-info"
#include "MBlazeTargetMachine.h"
using namespace llvm;

MBlazeSelectionDAGInfo::MBlazeSelectionDAGInfo(const MBlazeTargetMachine &TM)
  : TargetSelectionDAGInfo(TM) {
}

MBlazeSelectionDAGInfo::~MBlazeSelectionDAGInfo() {
}
