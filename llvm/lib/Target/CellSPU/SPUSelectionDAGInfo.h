//===-- SPUSelectionDAGInfo.h - CellSPU SelectionDAG Info -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the CellSPU subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef CELLSPUSELECTIONDAGINFO_H
#define CELLSPUSELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class SPUTargetMachine;

class SPUSelectionDAGInfo : public TargetSelectionDAGInfo {
public:
  explicit SPUSelectionDAGInfo(const SPUTargetMachine &TM);
  ~SPUSelectionDAGInfo();
};

}

#endif
