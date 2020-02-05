//===- VEMachineFunctionInfo.h - VE Machine Function Info -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares  VE specific per-machine-function information.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_TARGET_VE_VEMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_VE_VEMACHINEFUNCTIONINFO_H

#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

class VEMachineFunctionInfo : public MachineFunctionInfo {
  virtual void anchor();

private:
  /// IsLeafProc - True if the function is a leaf procedure.
  bool IsLeafProc;

public:
  VEMachineFunctionInfo() : IsLeafProc(false) {}
  explicit VEMachineFunctionInfo(MachineFunction &MF) : IsLeafProc(false) {}

  void setLeafProc(bool rhs) { IsLeafProc = rhs; }
  bool isLeafProc() const { return IsLeafProc; }
};
} // namespace llvm

#endif
