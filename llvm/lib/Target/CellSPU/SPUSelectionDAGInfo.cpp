//===-- SPUSelectionDAGInfo.cpp - CellSPU SelectionDAG Info ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SPUSelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "cellspu-selectiondag-info"
#include "SPUTargetMachine.h"
using namespace llvm;

SPUSelectionDAGInfo::SPUSelectionDAGInfo(const SPUTargetMachine &TM)
  : TargetSelectionDAGInfo(TM) {
}

SPUSelectionDAGInfo::~SPUSelectionDAGInfo() {
}
