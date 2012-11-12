//===-- PPCSelectionDAGInfo.cpp - PowerPC SelectionDAG Info ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PPCSelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "powerpc-selectiondag-info"
#include "PPCTargetMachine.h"
using namespace llvm;

PPCSelectionDAGInfo::PPCSelectionDAGInfo(const PPCTargetMachine &TM)
  : TargetSelectionDAGInfo(TM) {
}

PPCSelectionDAGInfo::~PPCSelectionDAGInfo() {
}
