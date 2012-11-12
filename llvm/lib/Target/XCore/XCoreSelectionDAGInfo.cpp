//===-- XCoreSelectionDAGInfo.cpp - XCore SelectionDAG Info ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the XCoreSelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "xcore-selectiondag-info"
#include "XCoreTargetMachine.h"
using namespace llvm;

XCoreSelectionDAGInfo::XCoreSelectionDAGInfo(const XCoreTargetMachine &TM)
  : TargetSelectionDAGInfo(TM) {
}

XCoreSelectionDAGInfo::~XCoreSelectionDAGInfo() {
}
