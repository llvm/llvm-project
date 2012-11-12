//===-- SparcSelectionDAGInfo.h - Sparc SelectionDAG Info -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Sparc subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef SPARCSELECTIONDAGINFO_H
#define SPARCSELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class SparcTargetMachine;

class SparcSelectionDAGInfo : public TargetSelectionDAGInfo {
public:
  explicit SparcSelectionDAGInfo(const SparcTargetMachine &TM);
  ~SparcSelectionDAGInfo();
};

}

#endif
