//===-- AMDGPUDwarfDebug.h - AMDGPU DwarfDebug Implementation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// AMDGPU-specific subclass of DwarfDebug. It emits the source locations of
/// instructions that were fused into another instruction (e.g. the Y component
/// of a VOPD) to the line table.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUDWARFDEBUG_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUDWARFDEBUG_H

#include "../../CodeGen/AsmPrinter/DwarfDebug.h"

namespace llvm {

class AMDGPUDwarfDebug : public DwarfDebug {
public:
  explicit AMDGPUDwarfDebug(AsmPrinter *A) : DwarfDebug(A) {}

  void beginInstruction(const MachineInstr *MI) override;

private:
  /// Emit a line-table row for the location of the instruction that was fused
  /// into \p MI, if there is one. It precedes the row of \p MI's own location,
  /// so the latter stays the location of the instruction's address.
  void recordFusedSourceLine(const MachineInstr &MI);
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUDWARFDEBUG_H
