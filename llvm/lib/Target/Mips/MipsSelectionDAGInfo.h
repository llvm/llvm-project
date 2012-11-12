//===-- MipsSelectionDAGInfo.h - Mips SelectionDAG Info ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Mips subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSSELECTIONDAGINFO_H
#define MIPSSELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class MipsTargetMachine;

class MipsSelectionDAGInfo : public TargetSelectionDAGInfo {
public:
  explicit MipsSelectionDAGInfo(const MipsTargetMachine &TM);
  ~MipsSelectionDAGInfo();
};

}

#endif
