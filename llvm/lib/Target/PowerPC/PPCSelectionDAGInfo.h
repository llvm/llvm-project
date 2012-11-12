//===-- PPCSelectionDAGInfo.h - PowerPC SelectionDAG Info -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PowerPC subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef POWERPCCSELECTIONDAGINFO_H
#define POWERPCCSELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class PPCTargetMachine;

class PPCSelectionDAGInfo : public TargetSelectionDAGInfo {
public:
  explicit PPCSelectionDAGInfo(const PPCTargetMachine &TM);
  ~PPCSelectionDAGInfo();
};

}

#endif
