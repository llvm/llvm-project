//===-- SPIRVCombinerHelper.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This contains common combine transformations that may be used in a combine
/// pass.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVCOMBINERHELPER_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVCOMBINERHELPER_H

#include "SPIRVSubtarget.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"

namespace llvm {
class SPIRVCombinerHelper : public CombinerHelper {
protected:
  const SPIRVSubtarget &STI;

public:
  using CombinerHelper::CombinerHelper;
  SPIRVCombinerHelper(GISelChangeObserver &Observer, MachineIRBuilder &B,
                      bool IsPreLegalize, GISelValueTracking *VT,
                      MachineDominatorTree *MDT, const LegalizerInfo *LI,
                      const SPIRVSubtarget &STI);

  bool matchLengthToDistance(MachineInstr &MI) const;
  void applySPIRVDistance(MachineInstr &MI) const;
  bool matchSelectToFaceForward(MachineInstr &MI) const;
  void applySPIRVFaceForward(MachineInstr &MI) const;
  bool matchMatrixTranspose(MachineInstr &MI) const;
  void applyMatrixTranspose(MachineInstr &MI) const;
  bool matchMatrixMultiply(MachineInstr &MI) const;
  void applyMatrixMultiply(MachineInstr &MI) const;

private:
  SPIRVTypeInst getDotProductVectorType(Register ResReg, uint32_t K,
                                        SPIRVGlobalRegistry *GR) const;
  SmallVector<Register, 4> extractColumns(Register BReg, uint32_t N,
                                          SPIRVTypeInst SpvVecType,
                                          SPIRVGlobalRegistry *GR) const;
  SmallVector<Register, 4> extractRows(Register AReg, uint32_t NumRows,
                                       uint32_t NumCols,
                                       SPIRVTypeInst SpvRowType,
                                       SPIRVGlobalRegistry *GR) const;
  SmallVector<Register, 16>
  computeDotProducts(const SmallVector<Register, 4> &RowsA,
                     const SmallVector<Register, 4> &ColsB,
                     SPIRVTypeInst SpvVecType, SPIRVGlobalRegistry *GR) const;
  Register computeDotProduct(Register RowA, Register ColB,
                             SPIRVTypeInst SpvVecType,
                             SPIRVGlobalRegistry *GR) const;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRVCOMBINERHELPER_H
