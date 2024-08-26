//===--- SILoadStoreOptimizer.h -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass tries to fuse DS instructions with close by immediate offsets.
// This will fuse operations such as
//  ds_read_b32 v0, v2 offset:16
//  ds_read_b32 v1, v2 offset:32
// ==>
//   ds_read2_b32 v[0:1], v2, offset0:4 offset1:8
//
// The same is done for certain SMEM and VMEM opcodes, e.g.:
//  s_buffer_load_dword s4, s[0:3], 4
//  s_buffer_load_dword s5, s[0:3], 8
// ==>
//  s_buffer_load_dwordx2 s[4:5], s[0:3], 4
//
// This pass also tries to promote constant offset to the immediate by
// adjusting the base. It tries to use a base from the nearby instructions that
// allows it to have a 13bit constant offset and then promotes the 13bit offset
// to the immediate.
// E.g.
//  s_movk_i32 s0, 0x1800
//  v_add_co_u32_e32 v0, vcc, s0, v2
//  v_addc_co_u32_e32 v1, vcc, 0, v6, vcc
//
//  s_movk_i32 s0, 0x1000
//  v_add_co_u32_e32 v5, vcc, s0, v2
//  v_addc_co_u32_e32 v6, vcc, 0, v6, vcc
//  global_load_dwordx2 v[5:6], v[5:6], off
//  global_load_dwordx2 v[0:1], v[0:1], off
// =>
//  s_movk_i32 s0, 0x1000
//  v_add_co_u32_e32 v5, vcc, s0, v2
//  v_addc_co_u32_e32 v6, vcc, 0, v6, vcc
//  global_load_dwordx2 v[5:6], v[5:6], off
//  global_load_dwordx2 v[0:1], v[5:6], off offset:2048
//
// Future improvements:
//
// - This is currently missing stores of constants because loading
//   the constant into the data register is placed between the stores, although
//   this is arguably a scheduling problem.
//
// - Live interval recomputing seems inefficient. This currently only matches
//   one pair, and recomputes live intervals and moves on to the next pair. It
//   would be better to compute a list of all merges that need to occur.
//
// - With a list of instructions to process, we can also merge more. If a
//   cluster of loads have offsets that are too large to fit in the 8-bit
//   offsets, but are close enough to fit in the 8 bits, we can add to the base
//   pointer and use the new reduced offsets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SILOADSTOREOPTIMIZER_H
#define LLVM_LIB_TARGET_AMDGPU_SILOADSTOREOPTIMIZER_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

class SILoadStoreOptimizerPass
    : public PassInfoMixin<SILoadStoreOptimizerPass> {
public:
  SILoadStoreOptimizerPass() = default;
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SILOADSTOREOPTIMIZER_H