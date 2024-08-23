//=======- GCNDPPCombine.h - optimization for DPP instructions ---==========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The pass combines V_MOV_B32_dpp instruction with its VALU uses as a DPP src0
// operand. If any of the use instruction cannot be combined with the mov the
// whole sequence is reverted.
//
// $old = ...
// $dpp_value = V_MOV_B32_dpp $old, $vgpr_to_be_read_from_other_lane,
//                            dpp_controls..., $row_mask, $bank_mask,
//                            $bound_ctrl
// $res = VALU $dpp_value [, src1]
//
// to
//
// $res = VALU_DPP $combined_old, $vgpr_to_be_read_from_other_lane, [src1,]
//                 dpp_controls..., $row_mask, $bank_mask, $combined_bound_ctrl
//
// Combining rules :
//
// if $row_mask and $bank_mask are fully enabled (0xF) and
//    $bound_ctrl==DPP_BOUND_ZERO or $old==0
// -> $combined_old = undef,
//    $combined_bound_ctrl = DPP_BOUND_ZERO
//
// if the VALU op is binary and
//    $bound_ctrl==DPP_BOUND_OFF and
//    $old==identity value (immediate) for the VALU op
// -> $combined_old = src1,
//    $combined_bound_ctrl = DPP_BOUND_OFF
//
// Otherwise cancel.
//
// The mov_dpp instruction should reside in the same BB as all its uses
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_GCNDPPCOMBINE_H
#define LLVM_LIB_TARGET_AMDGPU_GCNDPPCOMBINE_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {
class GCNDPPCombinePass : public MachinePassInfoMixin<GCNDPPCombinePass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MAM);
};

} // end namespace llvm

#endif