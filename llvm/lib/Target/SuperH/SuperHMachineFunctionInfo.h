//===-- SuperHMachineFunctionInfo.h - SuperH private data -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the SuperH specific subclass of MachineFunctionInfo.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SUPERH_SUPERHMACHINEFUNCTION_H
#define LLVM_LIB_TARGET_SUPERH_SUPERHMACHINEFUNCTION_H

#include "SuperHConstantPoolValue.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGenTypes/MachineValueType.h"

namespace llvm {

class SuperHMachineFunctionInfo : public MachineFunctionInfo {

  /// CPIndexCount - How many constant pool indices are allocated. 
  unsigned CPIndexCount = 0;

public:
  explicit SuperHMachineFunctionInfo(const Function &F,
                                     const TargetSubtargetInfo *STI) {}

  MachineFunctionInfo *
  clone(BumpPtrAllocator &Allocator, MachineFunction &DestMF,
        const DenseMap<MachineBasicBlock *, MachineBasicBlock *> &Src2DstMBB)
      const override;

  // getConstIndexCount - Gets the amount of PIC Labels that have been
  // created thus far.
  unsigned getConstIndexCount() const { return CPIndexCount; }

  // createConstIndex - Creates a new PIC Label UId.
  unsigned createConstIndex() { return CPIndexCount++; }

  // tryGetConstant - SuperH's compressed instruction set means that 
  // immediates and displacements can not be larger than 8 bits. 
  // As such we need to store said immediates and displacements within 
  // constants that are within range of the program counter.
  //
  // As such this function is a helper that:
  //  1. Allocates a constant pool slot for a given node
  //  2. Inserts the target into said slot.
  //  3. Returns the allocated slot, ready to be loaded via
  //     a PC-relative load.
  SuperHConstantPoolConstant *tryGetConstant(GlobalAddressSDNode *N, SelectionDAG &DAG, SHCP::SHCPModifier Modifier);
  SuperHConstantPoolConstant *tryGetConstant(BlockAddressSDNode *N, SelectionDAG &DAG, SHCP::SHCPModifier Modifier);
  SuperHConstantPoolSymbol *tryGetConstant(ExternalSymbolSDNode *N, SelectionDAG &DAG, SHCP::SHCPModifier Modifier);

private:
	virtual void anchor();
};


} // namespace llvm

#endif