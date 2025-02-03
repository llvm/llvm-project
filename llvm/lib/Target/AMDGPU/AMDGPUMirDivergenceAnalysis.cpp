//===- MirDivergenceAnalysis.cpp -- Mir Divergence Analysis Implementation -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is based on Analysis/DivergenceAnalysis.cpp,
// The most important difference is
// introduction of the idea of "Bit-Divergence".
//
// The way booleans are represented in in AMD GPU is a 64-bit uint in a pair of
// scalar registers, where each bit represents a boolean value for one lane. If
// all active lanes have the same bool value (all 1's or all 0's), then we can
// generate a scalar branch, otherwise we must use exec mask to selectively
// execute lanes based on the boolean mask. When all values in a boolean mask
// are the same for all active lanes, we call that mask "bit-uniform",
// otherwise we call it "bit-divergent". This differs from the normal concept
// of "uniform" and "divergent", which represents whether the value may be
// different across the 64 lanes. A "bit-divergent" value is still "uniform" in
// the sense that it is the same 64-bit value from the perspective of all the
// lanes, but when used as branch condition, will cause the branch to be
// divergent, which will cause the uses of any values outside of the control
// flow region to be divergent.
//
// The original DA marks everything including bools as divergent or uniform
// based on the propagation of divergent sources. However, booleans in AMDGPU
// are in fact never "divergent". Comparison operations that receive divergent
// operands instead produce "bit-divergent" or "bit-uniform" 64-bit booleans.
// Between the definition of any boolean mask and its use (particularly in
// branches, cndmasks, or anything that specifially consumes booleans), there
// can be any arbitrary number and types of operations performed on it,
// including combining it with other boolean masks via bit operations.
//
// The XDA algorithm is a modified version of the original DA algorithm to
// simultaneously propagate regular divergence and bit-divergence.
//
// First off, XDA identifies all sources of divergence as well as
// bit-divergence and adds them to the worklist. Then, just like with LLVM DA,
// it pops values off of the worklist to propagate (bit-)divergence to all its
// users, unless the user is always (bit-)uniform when given (bit-)divergent
// operand. It's possible for a value to be marked as both divergent and
// bit-divergent, in which case the regular divergence will trump
// bit-divergence.
//
// The important difference in this propagation step is that there are special
// instructions that when given bit-divergent operands, produce divergent
// values and vice versa.
//
// An example is comparison:
//
// v0 = interp ...               ; divergent
// v1 = interp ...               ; divergent
// s[0:1] = v_cmp v0, v1         ; bit-divergent
//
// v0 and v1 are both divergent, but when propagating them, the v_cmp (and its
// result) is bit-divergent value instead of divergent.
//
//
// An example of the reverse:
//
// v0 = ...                                ; uniform
// s[0:1] = v_cmp v0, v1                   ; bit-divergent
// ...
// branch s[0:1], label                    ; divergent!
// ...
// v1 = ...                                ; uniform
// ...
//
// label:
// v3 = phi v0, v1                         ; divergent! because of divergent branch.
//
// The boolean value is bit-divergent. When passed to the branch as an operand,
// the branch becomes divergent, whose sync dependency will be computed as
// normal to mark the appropriate values divergent (see description in normal
// DA on how this works).
//
// Another difference is in MIR, some branch will be changed into exec update,
// so only propagate control flow divergent on branch inst will not cover exec
// control flow.
// For case like
//  %163:sreg_64_xexec = S_MOV_B64 $exec
//bb.1:
//; predecessors: %bb.1, %bb.0
//  successors: %bb.1(0x40000000), %bb.2(0x40000000); %bb.1(50.00%), %bb.2(50.00%)
//  %162:vreg_512 = PHI %41:vreg_512, %bb.0, %40:vreg_512, %bb.1
//  %167:sgpr_32 = V_READFIRSTLANE_B32 %17:vgpr_32, implicit $exec
//  %168:sreg_64 = V_CMP_EQ_U32_e64 %167:sgpr_32, %17:vgpr_32, implicit $exec
//  %166:sreg_64 = S_AND_SAVEEXEC_B64 %168:sreg_64, implicit-def $exec, implicit-def $scc, implicit $exec
//...
//  $exec = S_XOR_B64_term $exec, %166:sreg_64, implicit-def $scc
//  S_CBRANCH_EXECNZ %bb.1, implicit $exec
// The ... code after SAVEEXEC will be divergent if %168 is divergent.
// The PHI should be divergent when %40 is inside the ...
// To propagate divergent from %168 to the PHI, need to start the propagate from
// SAVEEXEC which is the control flow by update exec.
//
//
// Original:
// This file implements a general divergence analysis for loop vectorization
// and GPU programs. It determines which branches and values in a loop or GPU
// program are divergent. It can help branch optimizations such as jump
// threading and loop unswitching to make better decisions.
//
// GPU programs typically use the SIMD execution model, where multiple threads
// in the same execution group have to execute in lock-step. Therefore, if the
// code contains divergent branches (i.e., threads in a group do not agree on
// which path of the branch to take), the group of threads has to execute all
// the paths from that branch with different subsets of threads enabled until
// they re-converge.
//
// Due to this execution model, some optimizations such as jump
// threading and loop unswitching can interfere with thread re-convergence.
// Therefore, an analysis that computes which branches in a GPU program are
// divergent can help the compiler to selectively run these optimizations.
//
// This implementation is derived from the Vectorization Analysis of the
// Region Vectorizer (RV). That implementation in turn is based on the approach
// described in
//
//   Improving Performance of OpenCL on CPUs
//   Ralf Karrenberg and Sebastian Hack
//   CC '12
//
// This DivergenceAnalysis implementation is generic in the sense that it does
// not itself identify original sources of divergence.
// Instead specialized adapter classes, (LoopDivergenceAnalysis) for loops and
// (GPUDivergenceAnalysis) for GPU programs, identify the sources of divergence
// (e.g., special variables that hold the thread ID or the iteration variable).
//
// The generic implementation propagates divergence to variables that are data
// or sync dependent on a source of divergence.
//
// While data dependency is a well-known concept, the notion of sync dependency
// is worth more explanation. Sync dependence characterizes the control flow
// aspect of the propagation of branch divergence. For example,
//
//   %cond = icmp slt i32 %tid, 10
//   br i1 %cond, label %then, label %else
// then:
//   br label %merge
// else:
//   br label %merge
// merge:
//   %a = phi i32 [ 0, %then ], [ 1, %else ]
//
// Suppose %tid holds the thread ID. Although %a is not data dependent on %tid
// because %tid is not on its use-def chains, %a is sync dependent on %tid
// because the branch "br i1 %cond" depends on %tid and affects which value %a
// is assigned to.
//
// The sync dependence detection (which branch induces divergence in which join
// points) is implemented in the SyncDependenceAnalysis.
//
// The current DivergenceAnalysis implementation has the following limitations:
// 1. intra-procedural. It conservatively considers the arguments of a
//    non-kernel-entry function and the return value of a function call as
//    divergent.
// 2. memory as black box. It conservatively considers values loaded from
//    generic or local address as divergent. This can be improved by leveraging
//    pointer analysis and/or by modelling non-escaping memory objects in SSA
//    as done in RV.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUMirDivergenceAnalysis.h"
#include "GCNSubtarget.h"
#include "AMDGPUSubtarget.h"
#include "Utils/AMDGPUAsmUtils.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "TargetInfo/AMDGPUTargetInfo.h"
#include "SIInstrInfo.h"
//#include "llvm/Analysis/Passes.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/Support/Debug.h"
//#include "newbe/cli/newbe_opts.h"  // AMDGPU change.
#include "llvm/Support/raw_ostream.h"
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "mir-divergence-analysis"

namespace llvm {
bool isAMDGPUOpcodeDivergent(class MachineInstr *MI);
}

//
// TODO: TableGen these
//
bool llvm::isAMDGPUOpcodeDivergent(class MachineInstr *MI) {
  switch (MI->getOpcode()) {
  // case R600::INTERP_LOAD_P0:
  // case R600::INTERP_PAIR_XY:
  // case R600::INTERP_PAIR_ZW:
  // case R600::INTERP_VEC_LOAD:
  // case R600::INTERP_XY:
  // case R600::INTERP_ZW:
  case AMDGPU::V_WRITELANE_B32:

  case AMDGPU::V_INTERP_MOV_F32:
  case AMDGPU::V_INTERP_MOV_F32_e64:
  case AMDGPU::V_INTERP_MOV_F32_e64_vi:
  case AMDGPU::V_INTERP_MOV_F32_si:
  case AMDGPU::V_INTERP_MOV_F32_vi:
  case AMDGPU::V_INTERP_P1LL_F16:
  case AMDGPU::V_INTERP_P1LL_F16_vi:
  case AMDGPU::V_INTERP_P1LV_F16:
  case AMDGPU::V_INTERP_P1LV_F16_vi:
  case AMDGPU::V_INTERP_P1_F32:
  case AMDGPU::V_INTERP_P1_F32_16bank:
  case AMDGPU::V_INTERP_P1_F32_16bank_si:
  case AMDGPU::V_INTERP_P1_F32_16bank_vi:
  case AMDGPU::V_INTERP_P1_F32_e64:
  case AMDGPU::V_INTERP_P1_F32_e64_vi:
  case AMDGPU::V_INTERP_P1_F32_si:
  case AMDGPU::V_INTERP_P1_F32_vi:
  case AMDGPU::V_INTERP_P2_F16:
  case AMDGPU::V_INTERP_P2_F16_vi:
  case AMDGPU::V_INTERP_P2_F32:
  case AMDGPU::V_INTERP_P2_F32_e64:
  case AMDGPU::V_INTERP_P2_F32_e64_vi:
  case AMDGPU::V_INTERP_P2_F32_si:
  case AMDGPU::V_INTERP_P2_F32_vi:

  case AMDGPU::V_MBCNT_HI_U32_B32_e32:
  case AMDGPU::V_MBCNT_HI_U32_B32_e32_gfx6_gfx7:
  case AMDGPU::V_MBCNT_HI_U32_B32_e64:
  case AMDGPU::V_MBCNT_HI_U32_B32_e64_gfx10:
  case AMDGPU::V_MBCNT_HI_U32_B32_e64_gfx6_gfx7:
  case AMDGPU::V_MBCNT_HI_U32_B32_e64_vi:
  case AMDGPU::V_MBCNT_HI_U32_B32_sdwa:
  case AMDGPU::V_MBCNT_LO_U32_B32_e32:
  case AMDGPU::V_MBCNT_LO_U32_B32_e32_gfx6_gfx7:
  case AMDGPU::V_MBCNT_LO_U32_B32_e64:
  case AMDGPU::V_MBCNT_LO_U32_B32_e64_gfx10:
  case AMDGPU::V_MBCNT_LO_U32_B32_e64_gfx6_gfx7:
  case AMDGPU::V_MBCNT_LO_U32_B32_e64_vi:
  case AMDGPU::V_MBCNT_LO_U32_B32_sdwa:

  case AMDGPU::BUFFER_ATOMIC_ADD_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_ADD_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_ADD_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_ADD_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_ADD_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_ADD_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_ADD_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_ADD_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_ADD_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_ADD_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_ADD_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_ADD_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_ADD_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_ADD_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_ADD_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_ADD_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_ADD_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_ADD_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_ADD_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_ADD_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_ADD_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_ADD_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_ADD_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_ADD_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_ADD_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_ADD_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_ADD_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_ADD_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_ADD_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_ADD_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_ADD_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_ADD_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_ADD_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_ADD_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_ADD_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_ADD_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_ADD_X2_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_AND_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_AND_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_AND_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_AND_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_AND_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_AND_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_AND_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_AND_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_AND_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_AND_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_AND_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_AND_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_AND_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_AND_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_AND_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_AND_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_AND_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_AND_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_AND_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_AND_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_AND_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_AND_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_AND_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_AND_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_AND_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_AND_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_AND_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_AND_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_AND_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_AND_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_AND_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_AND_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_AND_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_AND_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_AND_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_AND_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_AND_X2_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_CMPSWAP_X2_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_DEC_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_DEC_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_DEC_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_DEC_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_DEC_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_DEC_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_DEC_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_DEC_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_DEC_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_DEC_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_DEC_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_DEC_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_DEC_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_DEC_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_DEC_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_DEC_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_DEC_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_DEC_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_DEC_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_DEC_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_DEC_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_DEC_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_DEC_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_DEC_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_DEC_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_DEC_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_DEC_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_DEC_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_DEC_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_DEC_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_DEC_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_DEC_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_DEC_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_DEC_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_DEC_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_DEC_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_DEC_X2_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_INC_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_INC_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_INC_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_INC_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_INC_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_INC_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_INC_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_INC_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_INC_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_INC_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_INC_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_INC_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_INC_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_INC_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_INC_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_INC_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_INC_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_INC_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_INC_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_INC_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_INC_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_INC_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_INC_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_INC_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_INC_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_INC_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_INC_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_INC_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_INC_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_INC_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_INC_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_INC_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_INC_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_INC_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_INC_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_INC_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_INC_X2_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_OR_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_OR_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_OR_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_OR_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_OR_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_OR_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_OR_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_OR_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_OR_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_OR_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_OR_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_OR_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_OR_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_OR_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_OR_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_OR_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_OR_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_OR_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_OR_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_OR_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_OR_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_OR_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_OR_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_OR_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_OR_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_OR_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_OR_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_OR_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_OR_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_OR_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_OR_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_OR_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_OR_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_OR_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_OR_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_OR_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_OR_X2_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_SMAX_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_SMAX_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_SMAX_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMAX_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMAX_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_SMAX_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SMAX_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMAX_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMAX_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMAX_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMAX_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMAX_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMAX_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_SMAX_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SMAX_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMAX_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMAX_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMAX_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMAX_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMAX_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMAX_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_SMAX_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SMAX_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMAX_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMAX_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMAX_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMAX_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMAX_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMAX_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_SMAX_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_SMAX_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMAX_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMAX_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMAX_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMAX_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMAX_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMAX_X2_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_SMIN_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_SMIN_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_SMIN_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMIN_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMIN_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_SMIN_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SMIN_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMIN_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMIN_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMIN_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMIN_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMIN_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMIN_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_SMIN_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SMIN_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMIN_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMIN_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMIN_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMIN_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMIN_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMIN_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_SMIN_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SMIN_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMIN_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMIN_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMIN_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMIN_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMIN_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMIN_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_SMIN_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_SMIN_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMIN_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMIN_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMIN_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMIN_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMIN_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SMIN_X2_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_SUB_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_SUB_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_SUB_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SUB_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SUB_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_SUB_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SUB_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SUB_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SUB_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SUB_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SUB_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SUB_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SUB_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_SUB_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SUB_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SUB_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SUB_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SUB_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SUB_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SUB_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SUB_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_SUB_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SUB_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SUB_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SUB_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SUB_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SUB_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SUB_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SUB_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_SUB_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_SUB_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SUB_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SUB_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SUB_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SUB_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SUB_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SUB_X2_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_SWAP_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_SWAP_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_SWAP_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SWAP_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SWAP_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_SWAP_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SWAP_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SWAP_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SWAP_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SWAP_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SWAP_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SWAP_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SWAP_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_SWAP_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SWAP_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SWAP_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SWAP_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SWAP_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SWAP_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SWAP_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SWAP_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_SWAP_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SWAP_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SWAP_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SWAP_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SWAP_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SWAP_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SWAP_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SWAP_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_SWAP_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_SWAP_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SWAP_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SWAP_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SWAP_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SWAP_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SWAP_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_SWAP_X2_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_UMAX_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_UMAX_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_UMAX_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMAX_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMAX_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_UMAX_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_UMAX_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMAX_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMAX_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMAX_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMAX_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMAX_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMAX_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_UMAX_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_UMAX_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMAX_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMAX_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMAX_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMAX_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMAX_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMAX_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_UMAX_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_UMAX_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMAX_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMAX_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMAX_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMAX_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMAX_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMAX_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_UMAX_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_UMAX_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMAX_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMAX_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMAX_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMAX_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMAX_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMAX_X2_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_UMIN_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_UMIN_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_UMIN_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMIN_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMIN_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_UMIN_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_UMIN_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMIN_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMIN_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMIN_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMIN_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMIN_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMIN_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_UMIN_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_UMIN_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMIN_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMIN_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMIN_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMIN_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMIN_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMIN_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_UMIN_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_UMIN_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMIN_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMIN_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMIN_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMIN_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMIN_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMIN_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_UMIN_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_UMIN_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMIN_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMIN_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMIN_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMIN_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMIN_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_UMIN_X2_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_XOR_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_XOR_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_XOR_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_XOR_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_XOR_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_XOR_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_XOR_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_XOR_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_XOR_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_XOR_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_XOR_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_XOR_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_XOR_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_XOR_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_XOR_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_XOR_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_XOR_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_XOR_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_XOR_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_XOR_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_XOR_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_XOR_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_XOR_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_XOR_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_XOR_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_XOR_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_XOR_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_XOR_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_XOR_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_XOR_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_XOR_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_XOR_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_XOR_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_XOR_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_XOR_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_XOR_OFFSET_vi:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_ADDR64:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_ADDR64_RTN:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_ADDR64_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_ADDR64_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_BOTHEN:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_BOTHEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_BOTHEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_BOTHEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_BOTHEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_BOTHEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_BOTHEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_BOTHEN_vi:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_IDXEN:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_IDXEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_IDXEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_IDXEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_IDXEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_IDXEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_IDXEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_IDXEN_vi:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_OFFEN:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_OFFEN_RTN:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_OFFEN_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_OFFEN_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_OFFEN_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_OFFEN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_OFFEN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_OFFEN_vi:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_OFFSET:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_OFFSET_RTN:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_OFFSET_RTN_gfx10:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_OFFSET_RTN_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_OFFSET_RTN_vi:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_OFFSET_gfx10:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_OFFSET_gfx6_gfx7:
  case AMDGPU::BUFFER_ATOMIC_XOR_X2_OFFSET_vi:

  case AMDGPU::IMAGE_ATOMIC_ADD_V1_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_ADD_V1_V1_si:
  case AMDGPU::IMAGE_ATOMIC_ADD_V1_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_ADD_V2_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_ADD_V2_V1_si:
  case AMDGPU::IMAGE_ATOMIC_ADD_V2_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_ADD_V1_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_ADD_V1_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_ADD_V1_V2_si:
  case AMDGPU::IMAGE_ATOMIC_ADD_V1_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_ADD_V2_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_ADD_V2_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_ADD_V2_V2_si:
  case AMDGPU::IMAGE_ATOMIC_ADD_V2_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_ADD_V1_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_ADD_V1_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_ADD_V1_V4_si:
  case AMDGPU::IMAGE_ATOMIC_ADD_V1_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_ADD_V2_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_ADD_V2_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_ADD_V2_V4_si:
  case AMDGPU::IMAGE_ATOMIC_ADD_V2_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_AND_V1_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_AND_V1_V1_si:
  case AMDGPU::IMAGE_ATOMIC_AND_V1_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_AND_V2_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_AND_V2_V1_si:
  case AMDGPU::IMAGE_ATOMIC_AND_V2_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_AND_V1_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_AND_V1_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_AND_V1_V2_si:
  case AMDGPU::IMAGE_ATOMIC_AND_V1_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_AND_V2_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_AND_V2_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_AND_V2_V2_si:
  case AMDGPU::IMAGE_ATOMIC_AND_V2_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_AND_V1_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_AND_V1_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_AND_V1_V4_si:
  case AMDGPU::IMAGE_ATOMIC_AND_V1_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_AND_V2_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_AND_V2_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_AND_V2_V4_si:
  case AMDGPU::IMAGE_ATOMIC_AND_V2_V4_vi:
  //case AMDGPU::IMAGE_ATOMIC_CMPSWAP_V1_V1_gfx10:
  //case AMDGPU::IMAGE_ATOMIC_CMPSWAP_V1_V1_si:
  //case AMDGPU::IMAGE_ATOMIC_CMPSWAP_V1_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_CMPSWAP_V2_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_CMPSWAP_V2_V1_si:
  case AMDGPU::IMAGE_ATOMIC_CMPSWAP_V2_V1_vi:
  //case AMDGPU::IMAGE_ATOMIC_CMPSWAP_V1_V2_gfx10:
  //case AMDGPU::IMAGE_ATOMIC_CMPSWAP_V1_V2_nsa_gfx10:
  //case AMDGPU::IMAGE_ATOMIC_CMPSWAP_V1_V2_si:
  //case AMDGPU::IMAGE_ATOMIC_CMPSWAP_V1_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_CMPSWAP_V2_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_CMPSWAP_V2_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_CMPSWAP_V2_V2_si:
  case AMDGPU::IMAGE_ATOMIC_CMPSWAP_V2_V2_vi:
  //case AMDGPU::IMAGE_ATOMIC_CMPSWAP_V1_V4_gfx10:
  //case AMDGPU::IMAGE_ATOMIC_CMPSWAP_V1_V4_nsa_gfx10:
  //case AMDGPU::IMAGE_ATOMIC_CMPSWAP_V1_V4_si:
  //case AMDGPU::IMAGE_ATOMIC_CMPSWAP_V1_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_CMPSWAP_V2_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_CMPSWAP_V2_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_CMPSWAP_V2_V4_si:
  case AMDGPU::IMAGE_ATOMIC_CMPSWAP_V2_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_DEC_V1_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_DEC_V1_V1_si:
  case AMDGPU::IMAGE_ATOMIC_DEC_V1_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_DEC_V2_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_DEC_V2_V1_si:
  case AMDGPU::IMAGE_ATOMIC_DEC_V2_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_DEC_V1_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_DEC_V1_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_DEC_V1_V2_si:
  case AMDGPU::IMAGE_ATOMIC_DEC_V1_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_DEC_V2_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_DEC_V2_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_DEC_V2_V2_si:
  case AMDGPU::IMAGE_ATOMIC_DEC_V2_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_DEC_V1_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_DEC_V1_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_DEC_V1_V4_si:
  case AMDGPU::IMAGE_ATOMIC_DEC_V1_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_DEC_V2_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_DEC_V2_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_DEC_V2_V4_si:
  case AMDGPU::IMAGE_ATOMIC_DEC_V2_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_INC_V1_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_INC_V1_V1_si:
  case AMDGPU::IMAGE_ATOMIC_INC_V1_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_INC_V2_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_INC_V2_V1_si:
  case AMDGPU::IMAGE_ATOMIC_INC_V2_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_INC_V1_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_INC_V1_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_INC_V1_V2_si:
  case AMDGPU::IMAGE_ATOMIC_INC_V1_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_INC_V2_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_INC_V2_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_INC_V2_V2_si:
  case AMDGPU::IMAGE_ATOMIC_INC_V2_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_INC_V1_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_INC_V1_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_INC_V1_V4_si:
  case AMDGPU::IMAGE_ATOMIC_INC_V1_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_INC_V2_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_INC_V2_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_INC_V2_V4_si:
  case AMDGPU::IMAGE_ATOMIC_INC_V2_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_OR_V1_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_OR_V1_V1_si:
  case AMDGPU::IMAGE_ATOMIC_OR_V1_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_OR_V2_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_OR_V2_V1_si:
  case AMDGPU::IMAGE_ATOMIC_OR_V2_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_OR_V1_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_OR_V1_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_OR_V1_V2_si:
  case AMDGPU::IMAGE_ATOMIC_OR_V1_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_OR_V2_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_OR_V2_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_OR_V2_V2_si:
  case AMDGPU::IMAGE_ATOMIC_OR_V2_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_OR_V1_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_OR_V1_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_OR_V1_V4_si:
  case AMDGPU::IMAGE_ATOMIC_OR_V1_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_OR_V2_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_OR_V2_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_OR_V2_V4_si:
  case AMDGPU::IMAGE_ATOMIC_OR_V2_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_SMAX_V1_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SMAX_V1_V1_si:
  case AMDGPU::IMAGE_ATOMIC_SMAX_V1_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_SMAX_V2_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SMAX_V2_V1_si:
  case AMDGPU::IMAGE_ATOMIC_SMAX_V2_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_SMAX_V1_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SMAX_V1_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SMAX_V1_V2_si:
  case AMDGPU::IMAGE_ATOMIC_SMAX_V1_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_SMAX_V2_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SMAX_V2_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SMAX_V2_V2_si:
  case AMDGPU::IMAGE_ATOMIC_SMAX_V2_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_SMAX_V1_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SMAX_V1_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SMAX_V1_V4_si:
  case AMDGPU::IMAGE_ATOMIC_SMAX_V1_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_SMAX_V2_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SMAX_V2_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SMAX_V2_V4_si:
  case AMDGPU::IMAGE_ATOMIC_SMAX_V2_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_SMIN_V1_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SMIN_V1_V1_si:
  case AMDGPU::IMAGE_ATOMIC_SMIN_V1_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_SMIN_V2_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SMIN_V2_V1_si:
  case AMDGPU::IMAGE_ATOMIC_SMIN_V2_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_SMIN_V1_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SMIN_V1_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SMIN_V1_V2_si:
  case AMDGPU::IMAGE_ATOMIC_SMIN_V1_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_SMIN_V2_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SMIN_V2_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SMIN_V2_V2_si:
  case AMDGPU::IMAGE_ATOMIC_SMIN_V2_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_SMIN_V1_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SMIN_V1_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SMIN_V1_V4_si:
  case AMDGPU::IMAGE_ATOMIC_SMIN_V1_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_SMIN_V2_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SMIN_V2_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SMIN_V2_V4_si:
  case AMDGPU::IMAGE_ATOMIC_SMIN_V2_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_SUB_V1_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SUB_V1_V1_si:
  case AMDGPU::IMAGE_ATOMIC_SUB_V1_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_SUB_V2_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SUB_V2_V1_si:
  case AMDGPU::IMAGE_ATOMIC_SUB_V2_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_SUB_V1_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SUB_V1_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SUB_V1_V2_si:
  case AMDGPU::IMAGE_ATOMIC_SUB_V1_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_SUB_V2_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SUB_V2_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SUB_V2_V2_si:
  case AMDGPU::IMAGE_ATOMIC_SUB_V2_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_SUB_V1_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SUB_V1_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SUB_V1_V4_si:
  case AMDGPU::IMAGE_ATOMIC_SUB_V1_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_SUB_V2_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SUB_V2_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SUB_V2_V4_si:
  case AMDGPU::IMAGE_ATOMIC_SUB_V2_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_SWAP_V1_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SWAP_V1_V1_si:
  case AMDGPU::IMAGE_ATOMIC_SWAP_V1_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_SWAP_V2_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SWAP_V2_V1_si:
  case AMDGPU::IMAGE_ATOMIC_SWAP_V2_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_SWAP_V1_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SWAP_V1_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SWAP_V1_V2_si:
  case AMDGPU::IMAGE_ATOMIC_SWAP_V1_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_SWAP_V2_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SWAP_V2_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SWAP_V2_V2_si:
  case AMDGPU::IMAGE_ATOMIC_SWAP_V2_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_SWAP_V1_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SWAP_V1_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SWAP_V1_V4_si:
  case AMDGPU::IMAGE_ATOMIC_SWAP_V1_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_SWAP_V2_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SWAP_V2_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_SWAP_V2_V4_si:
  case AMDGPU::IMAGE_ATOMIC_SWAP_V2_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_UMAX_V1_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_UMAX_V1_V1_si:
  case AMDGPU::IMAGE_ATOMIC_UMAX_V1_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_UMAX_V2_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_UMAX_V2_V1_si:
  case AMDGPU::IMAGE_ATOMIC_UMAX_V2_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_UMAX_V1_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_UMAX_V1_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_UMAX_V1_V2_si:
  case AMDGPU::IMAGE_ATOMIC_UMAX_V1_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_UMAX_V2_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_UMAX_V2_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_UMAX_V2_V2_si:
  case AMDGPU::IMAGE_ATOMIC_UMAX_V2_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_UMAX_V1_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_UMAX_V1_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_UMAX_V1_V4_si:
  case AMDGPU::IMAGE_ATOMIC_UMAX_V1_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_UMAX_V2_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_UMAX_V2_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_UMAX_V2_V4_si:
  case AMDGPU::IMAGE_ATOMIC_UMAX_V2_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_UMIN_V1_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_UMIN_V1_V1_si:
  case AMDGPU::IMAGE_ATOMIC_UMIN_V1_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_UMIN_V2_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_UMIN_V2_V1_si:
  case AMDGPU::IMAGE_ATOMIC_UMIN_V2_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_UMIN_V1_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_UMIN_V1_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_UMIN_V1_V2_si:
  case AMDGPU::IMAGE_ATOMIC_UMIN_V1_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_UMIN_V2_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_UMIN_V2_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_UMIN_V2_V2_si:
  case AMDGPU::IMAGE_ATOMIC_UMIN_V2_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_UMIN_V1_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_UMIN_V1_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_UMIN_V1_V4_si:
  case AMDGPU::IMAGE_ATOMIC_UMIN_V1_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_UMIN_V2_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_UMIN_V2_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_UMIN_V2_V4_si:
  case AMDGPU::IMAGE_ATOMIC_UMIN_V2_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_XOR_V1_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_XOR_V1_V1_si:
  case AMDGPU::IMAGE_ATOMIC_XOR_V1_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_XOR_V2_V1_gfx10:
  case AMDGPU::IMAGE_ATOMIC_XOR_V2_V1_si:
  case AMDGPU::IMAGE_ATOMIC_XOR_V2_V1_vi:
  case AMDGPU::IMAGE_ATOMIC_XOR_V1_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_XOR_V1_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_XOR_V1_V2_si:
  case AMDGPU::IMAGE_ATOMIC_XOR_V1_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_XOR_V2_V2_gfx10:
  case AMDGPU::IMAGE_ATOMIC_XOR_V2_V2_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_XOR_V2_V2_si:
  case AMDGPU::IMAGE_ATOMIC_XOR_V2_V2_vi:
  case AMDGPU::IMAGE_ATOMIC_XOR_V1_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_XOR_V1_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_XOR_V1_V4_si:
  case AMDGPU::IMAGE_ATOMIC_XOR_V1_V4_vi:
  case AMDGPU::IMAGE_ATOMIC_XOR_V2_V4_gfx10:
  case AMDGPU::IMAGE_ATOMIC_XOR_V2_V4_nsa_gfx10:
  case AMDGPU::IMAGE_ATOMIC_XOR_V2_V4_si:
  case AMDGPU::IMAGE_ATOMIC_XOR_V2_V4_vi:

  case AMDGPU::SI_PS_LIVE:

  case AMDGPU::DS_SWIZZLE_B32:
  case AMDGPU::DS_SWIZZLE_B32_gfx10:
  case AMDGPU::DS_SWIZZLE_B32_gfx6_gfx7:
  case AMDGPU::DS_SWIZZLE_B32_vi:

    return true;

  default:
    break;
  }
  return false;
}

namespace {
bool hasImmOperandWithVal(const MachineInstr *MI, uint16_t srcNameIdx,
                          uint16_t srcModNameIdx, uint64_t Val) {
  unsigned Op = MI->getOpcode();
  unsigned srcIdx = AMDGPU::getNamedOperandIdx(Op, srcNameIdx);
  if (srcIdx == -1)
    return false;
  const MachineOperand &srcMO = MI->getOperand(srcIdx);
  if (srcMO.isImm() && srcMO.getImm() == Val) {

    unsigned modIdx = AMDGPU::getNamedOperandIdx(Op, srcModNameIdx);
    if (modIdx == -1)
      return true;

    const MachineOperand &modMO = MI->getOperand(modIdx);
    if (modMO.getImm() == 0)
      return true;
  }
  return false;
}

bool isConstant(const MachineInstr *MI) {
  unsigned Op = MI->getOpcode();
  switch (Op) {
  default:
    break;
  case AMDGPU::V_OR_B32_e32:
  case AMDGPU::V_OR_B32_e64: {
    // Check special case  or -1, which will get result -1.
    const uint64_t kImm = -1;
    if (hasImmOperandWithVal(MI, AMDGPU::OpName::src0,
                             AMDGPU::OpName::src0_modifiers, kImm))
      return true;
    if (hasImmOperandWithVal(MI, AMDGPU::OpName::src1,
                             AMDGPU::OpName::src1_modifiers, kImm))
      return true;
  } break;
  case AMDGPU::S_OR_B32:
  case AMDGPU::S_OR_B64: {
    // Check special case  or -1, which will get result -1.
    const uint64_t kImm = -1;
    if (hasImmOperandWithVal(MI, AMDGPU::OpName::src0,
                             AMDGPU::OpName::src0_modifiers, kImm))
      return true;
    if (hasImmOperandWithVal(MI, AMDGPU::OpName::src1,
                             AMDGPU::OpName::src1_modifiers, kImm))
      return true;
  } break;
  case AMDGPU::S_AND_B32:
  case AMDGPU::S_AND_B64:
  case AMDGPU::V_AND_B32_e32:
  case AMDGPU::V_AND_B32_e64: {
    // Check special case  and 0, which will get result 0.
    const uint64_t kImm = 0;
    if (hasImmOperandWithVal(MI, AMDGPU::OpName::src0,
                             AMDGPU::OpName::src0_modifiers, kImm))
      return true;
    if (hasImmOperandWithVal(MI, AMDGPU::OpName::src1,
                             AMDGPU::OpName::src1_modifiers, kImm))
      return true;
  } break;
  }
  return false;
}

bool writeBoolDst(const MachineInstr *MI, const SIRegisterInfo *SIRI,
                  const MachineRegisterInfo &MRI) {
  const auto *BoolRC = SIRI->getBoolRC();
  for (const MachineOperand &MO : MI->operands()) {
    if (!MO.isReg())
      continue;
    if (MO.isUse())
      continue;
    unsigned Reg = MO.getReg();
    if (Reg == AMDGPU::EXEC || Reg == AMDGPU::EXEC_LO ||
        Reg == AMDGPU::VCC || Reg == AMDGPU::VCC_LO)
      return true;

    // Check if the written register class overlaps the bool register class.
    //
    // Note that this check is insufficent to catch all of the cases where
    // a "bool" value could be created (for example writing to a register
    // pair s[0:1], then using s0 as a bool value in wave32).
    //
    // The underlying problem is that we have two notions of divergence
    // (bit divergence and wave divergence) but the algorithm only propagates
    // wave divergence. The bit divergence is important for bools because it determines
    // if a branch is uniform or not (and thus catches cases where a uniform value is
    // used outside of a divergent control flow region). For bool values the
    // algorithm will treat normally uniform values (i.e. scalar registers) as divergent
    // in order to try and propagate bit divergence.
    //
    // To fix all the possible bugs here I think we need to actually proagate bit
    // divergence as well as wave divergences. That is a bigger fix and this check should
    // cover most cases of treating a bool value as divergent.
    const TargetRegisterClass *RC = SIRI->getRegClassForReg(MRI, Reg);
    if (SIRI->getCommonSubClass(BoolRC, RC))
      return true;
  }
  return false;
}

bool isAlwaysUniformMI(const MachineInstr *MI, const SIInstrInfo *SIII,
                       const SIRegisterInfo *SIRI,
                       const MachineRegisterInfo &MRI) {
  unsigned Op = MI->getOpcode();
  switch (Op) {
  default:
    // Mark all s_inst always uniform except write to bool dst. This doesn't
    // mean it is bit uniform. When check branch/exec region, will use
    // isBitUniform. A bool might be sreg, but still divergent, since it is just
    // put all lanes in one 64/32 bits sreg.
    if (SIII->isScalarUnit(*MI) && !writeBoolDst(MI, SIRI, MRI) &&
        !MI->isTerminator())
      return true;
    break;
  //case AMDGPU::AMDGPU_MAKE_UNIFORM:
  //case AMDGPU::AMDGPU_WAVE_READ_LANE_FIRST:
  case AMDGPU::V_READFIRSTLANE_B32:
  case AMDGPU::V_READLANE_B32:
  //case AMDGPU::AMDGPU_WAVE_ACTIVE_BALLOT_W32:
  //case AMDGPU::AMDGPU_WAVE_ACTIVE_BALLOT_W64:
    // bool readfirstlane should be 1 bit, which means bit uniform.
    return true;
  case AMDGPU::S_OR_B32:
  case AMDGPU::S_OR_B64: {
    // Check special case  or -1, which will get result -1.
    if (isConstant(MI))
      return true;

    return !writeBoolDst(MI, SIRI, MRI);
  } break;
  case AMDGPU::V_OR_B32_e32:
  case AMDGPU::V_OR_B32_e64: {
    // Check special case  or -1, which will get result -1.
    if (isConstant(MI))
      return true;
  } break;
  case AMDGPU::S_AND_B32:
  case AMDGPU::S_AND_B64: {
    // Check special case  and 0, which will get result 0.
    if (isConstant(MI))
      return true;

    return !writeBoolDst(MI, SIRI, MRI);
  } break;
  case AMDGPU::V_AND_B32_e32:
  case AMDGPU::V_AND_B32_e64: {
    // Check special case  and 0, which will get result 0.
    if (isConstant(MI))
      return true;
  } break;
  }
  return false;
}

bool isPhysicalReg(MachineRegisterInfo &MRI, Register reg) {
  return reg.isPhysical();;
}

bool isRegClass(MachineRegisterInfo &MRI, unsigned reg, unsigned regClassID) {
  return MRI.getRegClass(reg)->getID() == regClassID;
}

// For input reg of MF, vgpr will be divergent.
bool isDivergentInputReg(unsigned Reg, MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI) {
  if (isPhysicalReg(MRI, Reg)) {
    unsigned vir_reg = MRI.getLiveInVirtReg(Reg);
    if (SIRI->isVGPR(MRI, vir_reg))
      return true;
  } else {
   if (SIRI->isVGPR(MRI, Reg))
      return true;
  }
  return false;
}

bool isSourceOfDivergence(MachineInstr *MI, MachineRegisterInfo &MRI,
                          const SIRegisterInfo *SIRI, const SIInstrInfo *SIII) {
  //if (MI->getAMDGPUFlag(MachineInstr::AMDGPUMIFlag::IsDivergent))
  //  return true;
  if (isAMDGPUOpcodeDivergent(MI))
    return true;

  if (isAlwaysUniformMI(MI, SIII, SIRI, MRI))
    return false;

  // If the instruction is neither guaranteed to
  // be uniform or divergent, check whether any
  // of its operands are passed in to the shader as
  // args through vector regs.
  //
  // This makes them divergent.
  for (MachineOperand &op : MI->operands()) {
    if (!op.isReg())
      continue;
    if (op.isDef())
      continue;
    unsigned reg = op.getReg();
    if (MRI.isLiveIn(reg)) {
      if (isDivergentInputReg(reg, MRI, SIRI))
        return true;
    }
  }

  return false;
}

// For VCC, try to find the nearest define inside same BB.
const MachineInstr *findPhysicalDefineInSameMBB(const MachineInstr *MI,
                                                unsigned PhyReg) {
  const MachineBasicBlock *MBB = MI->getParent();
  auto it = MI->getReverseIterator();
  for (it++; it != MBB->rend(); it++) {
    const MachineInstr &TmpMI = *it;
    for (const MachineOperand &DefMO : TmpMI.operands()) {
      if (!DefMO.isReg())
        continue;
      if (DefMO.isUse())
        continue;
      if (DefMO.getReg() == PhyReg)
        return &TmpMI;
    }
  }
  return nullptr;
}

bool isWriteExec(const MachineInstr *MI) {
  for (const MachineOperand &MO : MI->operands()) {
    if (!MO.isReg())
      continue;
    if (MO.isUse())
      continue;
    unsigned Reg = MO.getReg();
    if (Reg == AMDGPU::EXEC ||
        Reg == AMDGPU::EXEC_LO)
      return true;
  }
  return false;
}

bool isVCndMask(unsigned Opcode) {
  switch (Opcode) {
  default:
    return false;
  case AMDGPU::V_CNDMASK_B32_e32:
  case AMDGPU::V_CNDMASK_B32_e64:
  case AMDGPU::V_CNDMASK_B32_dpp:
  case AMDGPU::V_CNDMASK_B32_sdwa:
  case AMDGPU::V_CNDMASK_B64_PSEUDO:
    return true;
  }
}


bool isExecRegionOp(unsigned Op) {
  switch (Op) {
  default:
    return false;
  case AMDGPU::COPY:
  case AMDGPU::S_MOV_B32:
  case AMDGPU::S_MOV_B64:
    return true;
  }
}

bool isRestoreExec(const MachineInstr *MI) {
  unsigned Op = MI->getOpcode();
  if (!isExecRegionOp(Op))
    return false;

  return isWriteExec(MI);
}

const MachineInstr *
findExecRegionBeginFromRegionEnd(const MachineInstr *MI,
                                 const MachineRegisterInfo &MRI) {
  const MachineOperand &MO = MI->getOperand(1);
  if (!MO.isReg())
    return nullptr;
  unsigned Reg = MO.getReg();
  const MachineInstr *Def = MRI.getUniqueVRegDef(Reg);
  if (!Def)
    return nullptr;
  // Make sure the def is S_MOV Reg, Exec.
  if (!isExecRegionOp(Def->getOpcode()))
    return nullptr;
  const MachineOperand &ExecMO = Def->getOperand(1);
  if (!ExecMO.isReg())
    return nullptr;
  unsigned ExecReg = ExecMO.getReg();
  if (ExecReg == AMDGPU::EXEC || ExecReg == AMDGPU::EXEC_LO)
    return Def;
  else
    return nullptr;
}

bool isInsideExecRegion(const MachineInstr &MI, const MachineInstr &RegionBegin,
                        const MachineInstr &RegionEnd,
                        const MachineDominatorTree &DT,
                        const MachinePostDominatorTree &PDT) {
  if (!DT.dominates(&RegionBegin, &MI))
    return false;

  const MachineBasicBlock *MBB = MI.getParent();
  const MachineBasicBlock *RegionEndMBB = RegionEnd.getParent();
  if (MBB != RegionEndMBB) {
    return PDT.dominates(RegionEndMBB, MBB);
  } else {
    // MachineLoop through the basic block until we find A or B.
    MachineBasicBlock::const_iterator I = MBB->begin();
    for (; I != MI.getIterator() && I != RegionEnd.getIterator(); ++I)
      /*empty*/;

    // RegionEnd post-dominates MI if MI is found first in the basic block.
    return I == MI.getIterator();
  }
}

bool isInsideExecRegion(const MachineBasicBlock &MBB,
                        const MachineInstr &RegionBegin,
                        const MachineInstr &RegionEnd,
                        const MachineDominatorTree &DT,
                        const MachinePostDominatorTree &PDT) {
  const MachineBasicBlock *RegionBeginMBB = RegionBegin.getParent();
  const MachineBasicBlock *RegionEndMBB = RegionEnd.getParent();
  if (!DT.dominates(RegionBeginMBB, &MBB))
    return false;
  return PDT.dominates(RegionEndMBB, &MBB);
}

// Map from BB to nearest Exec Region. How to build? Add every MBB unless already has smaller region?
// Then when hit saveExec, propagate leaked users of define inside the exec region.

} // namespace

namespace llvm {
// class DivergenceAnalysis
DivergenceAnalysis::DivergenceAnalysis(
    const MachineFunction &F, const MachineLoop *RegionLoop, const MachineDominatorTree &DT,
    const MachinePostDominatorTree &PDT, const MachineLoopInfo &LI,
    SyncDependenceAnalysis &SDA, bool IsLCSSAForm,
    // AMDGPU change begin.
    DivergentJoinMapTy &JoinMap
    // AMDGPU change end.
    )
    : F(F), MRI(F.getRegInfo()), RegionLoop(RegionLoop), DT(DT), PDT(PDT),
      LI(LI), SDA(SDA), DivergentJoinMap(JoinMap), // AMDGPU change
      IsLCSSAForm(IsLCSSAForm) {
  const GCNSubtarget *ST = &F.getSubtarget<GCNSubtarget>();
  SIRI = ST->getRegisterInfo();
  SIII = ST->getInstrInfo();
}

void DivergenceAnalysis::markDivergent(const ValueTy DivVal) {
  assert(!isAlwaysUniform(DivVal) && "cannot be a divergent");
  // AMDGPU change begin.
  LLVM_DEBUG(const GCNSubtarget *ST = &F.getSubtarget<GCNSubtarget>();
             const SIRegisterInfo *SIRI = ST->getRegisterInfo();
             dbgs() << "\t MarkDivergent :"; printReg(DivVal, SIRI););
  //AMDGPU change end.
  DivergentValues.insert(DivVal);
}

// Mir change.
void DivergenceAnalysis::markDivergent(const MachineInstr &I) {
  for (const MachineOperand &DstMO : I.defs()) {
    unsigned Reg = DstMO.getReg();
    markDivergent(Reg);
  }
  DivergentInsts.insert(&I);
}

void DivergenceAnalysis::addUniformOverride(const ValueTy UniVal) {
  // TODO: support uniform multi-def.
  if (MRI.getUniqueVRegDef(UniVal) == nullptr)
    return;

  UniformOverrides.insert(UniVal);
}

void DivergenceAnalysis::addUniformOverride(const MachineInstr &I) {
  for (const MachineOperand &DstMO : I.defs()) {
    unsigned Reg = DstMO.getReg();
    addUniformOverride(Reg);
  }
  UniformOverridesInsts.insert(&I);
}

bool DivergenceAnalysis::isBitUniform(
    const MachineInstr &I, const llvm::MachineOperand &UseMO,
    llvm::DenseMap<const MachineInstr *, bool> &Processed) const {
  if (UseMO.isImm()) {
    uint64_t val = UseMO.getImm();
    // 0 and -1 are OK since all lanes are still the same.
    if (val == 0 || val == -1)
      return true;
    else
      return false;
  }
  if (!UseMO.isReg())
    return true;
  unsigned Reg = UseMO.getReg();
  // Exec is always bituniform, because all active lanes are 1.
  if (Reg == AMDGPU::EXEC || Reg == AMDGPU::EXEC_LO ||
      // SCC only has 1 bit. Always bituniform.
      Reg == AMDGPU::SCC)
    return true;

  const MachineInstr *UseMI = nullptr;
  if (Reg == AMDGPU::VCC || Reg == AMDGPU::VCC_LO) {
    // Try to find define of this VCC.
    UseMI = findPhysicalDefineInSameMBB(&I, Reg);
  } else {
    UseMI = MRI.getUniqueVRegDef(Reg);
  }
  if (!UseMI) {
    return false;
  }

  bool bResult = isBitUniform(*UseMI, Processed);
  Processed[UseMI] = bResult;
  return bResult;
}

bool DivergenceAnalysis::isBitUniform(
    const MachineInstr &I,
    llvm::DenseMap<const MachineInstr *, bool> &Processed) const {
  auto it = Processed.find(&I);
  if (it != Processed.end())
    return it->second;
  // For branch on MIR, need to make sure all activi lanes are the same.
  // cmp of uniform value will make sure all active lanes are the same.
  // Imm is also the same for all active lanes.
  if (isDivergent(I))
    return false;
  // Uniform cmp is bit uniform.
  if (I.isCompare())
    return true;
  if (isConstant(&I))
    return true;

  // Conservatively consider bituniform to be false.
  Processed[&I] = false;

  // If all operand is bit uniform, then result is bit uniform.
  bool bAllOperandBitUniform = true;
  for (const MachineOperand &UseMO : I.uses()) {
    if (isBitUniform(I, UseMO, Processed))
      continue;
    bAllOperandBitUniform = false;
    break;
  }
  return bAllOperandBitUniform;
}

bool DivergenceAnalysis::updateTerminator(const MachineInstr &Term) const {
  if (Term.getParent()->succ_size() <= 1)
    return false;
  switch (Term.getOpcode()) {
  default: {
    if (updateNormalInstruction(Term))
      return true;
    llvm::DenseMap<const MachineInstr *, bool> Processed;
    // Check bit uniform here if not divergent.
    return !isBitUniform(Term, Processed);
  }
  //case AMDGPU::AMDGPU_CALL_INDIRECT:
  case AMDGPU::SI_CALL:
    return true;
  }
}

bool DivergenceAnalysis::updateNormalInstruction(const MachineInstr &I) const {
  // TODO function calls with side effects, etc
  if (UniformOverridesInsts.find(&I) != UniformOverridesInsts.end())
    return false;
  if (DivergentInsts.find(&I) != DivergentInsts.end())
    return true;
  for (const auto &Op : I.uses()) {
    if (!Op.isReg())
      continue;
    Register Reg = Op.getReg();
    if (Reg.isPhysical()) {
      if (Reg == AMDGPU::EXEC ||
          Reg == AMDGPU::EXEC_LO ||
          Reg == AMDGPU::SCC)
        continue;
      else 
      if (const MachineInstr *DefMI =
              findPhysicalDefineInSameMBB(Op.getParent(), Reg)) {
        if (isDivergent(*DefMI))
          return true;
      } else {
        // If cannot find def in same MBB, just treat it as divergent.
        return true;
      }
    } else {
      if (isDivergent(Op.getReg()))
        return true;
    }
  }
  return false;
}

bool DivergenceAnalysis::isTemporalDivergent(const MachineBasicBlock &ObservingBlock,
                                             const ValueTy Val,
                                             const MachineBasicBlock &IncomingBlock) const { // AMDGPU change
  const MachineBasicBlock *DefBlock = &IncomingBlock; // AMDGPU change: Take def point as incoming block for constants.
  const auto *Inst = MRI.getUniqueVRegDef(Val);
  if (Inst == nullptr)
    return true;
  if (Inst)
      DefBlock = Inst->getParent(); 

  // check whether any divergent loop carrying Val terminates before control
  // proceeds to ObservingBlock
  for (const auto *MachineLoop = LI.getLoopFor(DefBlock); // AMDGPU change
       MachineLoop != RegionLoop && !MachineLoop->contains(&ObservingBlock);
       MachineLoop = MachineLoop->getParentLoop()) {
    if (DivergentLoops.find(MachineLoop) != DivergentLoops.end())
      return true;
  }

  return false;
}

// AMDGPU CHANGE BEGIN
static bool HasIncomingUndefValue(const PHINode_ *Phi) {
  for (unsigned I = 1, E = Phi->getNumOperands(); I != E; I += 2) {
    const MachineOperand &Op = Phi->getOperand(I);
    if (Op.isUndef())
      return true;
  }
  return false;
}

// For case like
//  %163:sreg_64_xexec = S_MOV_B64 $exec
//bb.1:
//; predecessors: %bb.1, %bb.0
//  successors: %bb.1(0x40000000), %bb.2(0x40000000); %bb.1(50.00%), %bb.2(50.00%)
//  %162:vreg_512 = PHI %41:vreg_512, %bb.0, %40:vreg_512, %bb.1
//  %167:sgpr_32 = V_READFIRSTLANE_B32 %17:vgpr_32, implicit $exec
//  %168:sreg_64 = V_CMP_EQ_U32_e64 %167:sgpr_32, %17:vgpr_32, implicit $exec
//  %166:sreg_64 = S_AND_SAVEEXEC_B64 %168:sreg_64, implicit-def $exec, implicit-def $scc, implicit $exec
//...
//  $exec = S_XOR_B64_term $exec, %166:sreg_64, implicit-def $scc
//  S_CBRANCH_EXECNZ %bb.1, implicit $exec
// The ... code after SAVEEXEC will be divergent if %168 is divergent.
// Return the SaveExec which affect MI.
// If not exist, return nullptr.
static const MachineInstr *
findSaveExec(const MachineInstr *MI,
             const SmallVector<const MachineInstr *, 2> &SaveExecs) {
  // No save exec.
  if (SaveExecs.empty())
    return nullptr;
  if (SaveExecs.size() > 1)
    llvm::report_fatal_error(
        "Not support case where, MBB has more than one SaveExec");
  const MachineInstr *SaveExec = SaveExecs.front();
  const MachineBasicBlock *MBB = SaveExec->getParent();
  // Make sure MI is after SaveExec by check it is not before SaveExec.
  // Assume MBB.begin to SaveExec is short here.
  bool bIsAfterSaveExec = true;
  for (auto it = MBB->begin(); it != SaveExec->getIterator(); it++) {
    if (MI == it) {
      bIsAfterSaveExec = false;
      break;
    }
  }
  // Not affect by save exec.
  if (!bIsAfterSaveExec)
    return nullptr;

  return SaveExec;
}

// When a Phi's parent isJoinDivergent,the case make phi divergent is that 2
// incoming values merged from different path of a divergent branch.
// isJoinDivergentOnlyOnSameIncomingValue will check for all
// combinations of incoming values except the BB with same incoming value,
// because if values are same then even divergent branch is not divergent.
// For example phi a:A, b:B, a:C.
// It will check (A,B) (B,C) but not (A, C) Because A
// and C has same value a.
// If only (A,C) is sharing divergent branch,
// then phi a:A, b:B, a:C is still uniform.
// DivergentJoinMap saving MachineBasicBlock pairs which on different path of a
// divergent branch and joined at one block.
// For example,
//    A
//  /   \
// |     \
// |      \
// B       /
// | \    /
// |  \  /
// C   D
// |   /
//  \ /
//   E
// If A is uniform branch, B is divergent branch. Then only (C, D) will be saved
// in DivergentJoinMap.
// DivergentJoinMap is build with updateDisjointMap in
// SyncDependenceAnalysis.cpp when SyncDependenceAnalysis::join_block is called.
// It will only run on divergent branch, so (A, B) is not in
// DivergentDisjointMap when A is uniform.
static bool isJoinDivergentOnlyOnSameIncomingValue(
    const PHINode_ &Phi, const DivergenceAnalysis *pDA, const MachineDominatorTree &DT,
    DivergentJoinMapTy &DivergentJoinMap) {
  // for phi which join divergent, if the incoming values from divergent
  // branch are the same, the phi is still uniform.
  // A
  // | \
  // |  \
  // B   \
  // |\   \
  // | \   |
  // C  D  E
  // |  /  /
  //  \/  /
  //   \ /
  //    F
  // for phi in F like.
  // phi (a:C, a:D, b:E)
  // If A is uniform branch, B is non-uniform branch, phi is uniform.
  SmallDenseSet<unsigned, 8> ValueToBlockMap;
  for (unsigned I = 1, E = Phi.getNumOperands(); I != E; I += 2) {
    const MachineOperand &Op = Phi.getOperand(I);
    if (!Op.isReg())
      continue;
    unsigned Reg = Op.getReg();
    if (pDA->isDivergent(Reg))
      return false;

    ValueToBlockMap.insert(Reg);
  }
  unsigned NumIncoming = (Phi.getNumOperands() - 1) / 2;
  // When there's same incoming value from different incoming block.
  // If divergent select is only on same value, then it is still uniform.
  if (ValueToBlockMap.size() != NumIncoming) {
    // When a phi is on divergent join block, there is incoming block which is
    // comeing from different path of a divergent branch.
    // Check all combination here.
    for (unsigned i = 0; i < NumIncoming; i++) {
      MachineBasicBlock *BB0 = Phi.getOperand(2 + 2 * i).getMBB();
      const MachineOperand &MO0 = Phi.getOperand(1 + 2 * i);
      for (unsigned j = i + 1; j < NumIncoming; j++) {
        MachineBasicBlock *BB1 = Phi.getOperand(2 + 2 * j).getMBB();
        const MachineOperand &MO1 = Phi.getOperand(1 + 2 * j);
        // If value match, no divergent.
        if (MO0.isImm() && MO1.isImm() && MO0.getImm() == MO1.getImm())
          continue;
        if (MO0.isReg() && MO1.isReg() && MO0.getReg() == MO1.getReg() &&
            MO0.getSubReg() == MO1.getSubReg())
          continue;

        // If BB and BB2 is from divergent disjoint, then they will
        // divergent join on phi.
        // This is for case like
        //    A
        //  /   \
        // |     \
        // |      \
        // B       /
        // | \    /
        // |  \  /
        // C   D
        // |   /
        //  \ /
        //   E
        //
        // phi(a:C, b:D)
        // When nearestCommonDominator is A, but B also can be divergent
        // disjoint for C and D.
        if (DivergentJoinMap[BB0].count(BB1))
          return false;
      }
    }
    return true;
  } else {
    return false;
  }
}
// AMDGPU CHANGE END

bool DivergenceAnalysis::updatePHINode(const PHINode_ &Phi) const {
  // AMDGPU CHANGE BEGIN
  // Do not mark phis with undef as incoming values as uniform.
  // When promoting to scalar we will readfirstlane on
  // the phi output. If some of the inputs are undef then
  // this could replace a well defined vector value with an
  // undefined scalar value.
  if (HasIncomingUndefValue(&Phi))
    return true;
  // AMDGPU CHANGE END

  // joining divergent disjoint path in Phi parent block
  if (isJoinDivergent(*Phi.getParent())) {
    // AMDGPU CHANGE BEGIN
    if (true/*TODO: ENABLE_AGGRESSIVE_UNIFORM_ANALYSIS*/) {
      // Continue if the divergent join only on same incoming value.
      if (!isJoinDivergentOnlyOnSameIncomingValue(Phi, this, DT,
                                                  DivergentJoinMap))
        return true;
    } else
    // AMDGPU CHANGE END
    return true;
  }

  // An incoming value could be divergent by itself.
  // Otherwise, an incoming value could be uniform within the loop
  // that carries its definition but it may appear divergent
  // from outside the loop. This happens when divergent loop exits
  // drop definitions of that uniform value in different iterations.
  //
  // for (int i = 0; i < n; ++i) { // 'i' is uniform inside the loop
  //   if (i % thread_id == 0) break;    // divergent loop exit
  // }
  // int divI = i;                 // divI is divergent
  for (unsigned I = 1, E = Phi.getNumOperands(); I != E; I += 2) {
    const MachineOperand &Op = Phi.getOperand(I);
    if (!Op.isReg())
      continue;

    unsigned Reg = Op.getReg();
    const MachineOperand &BB = Phi.getOperand(I + 1);
    if (isDivergent(Reg) ||
        isTemporalDivergent(*Phi.getParent(), Reg, *BB.getMBB()))
      return true;

  }

  return false;
}

bool DivergenceAnalysis::updateVCndMask(const MachineInstr &VCndMask) const {
  // VCndMask require the Cond bituniform to be uniform.
  unsigned Op = VCndMask.getOpcode();
  unsigned src0Idx = AMDGPU::getNamedOperandIdx(Op, AMDGPU::OpName::src0);
  unsigned src1Idx = AMDGPU::getNamedOperandIdx(Op, AMDGPU::OpName::src1);
  unsigned src2Idx = AMDGPU::getNamedOperandIdx(Op, AMDGPU::OpName::src2);

  const MachineOperand &src0 = VCndMask.getOperand(src0Idx);
  const MachineOperand &src1 = VCndMask.getOperand(src1Idx);

  const MachineOperand &cond = VCndMask.getOperand(src2Idx);

  if (isDivergent(src0))
    return true;

  // If src0 == src1, then return src0 divergent.
  if (src0.isReg() && src1.isReg() && src0.getReg() == src1.getReg()) {
    if (src0.getSubReg() == src1.getSubReg() &&
        SIII->hasModifiersSet(VCndMask, AMDGPU::OpName::src0_modifiers) ==
            SIII->hasModifiersSet(VCndMask, AMDGPU::OpName::src1_modifiers))
      return false;
  }

  if (isDivergent(src1))
    return true;

  llvm::DenseMap<const MachineInstr *, bool> Processed;
  return !isBitUniform(VCndMask, cond, Processed);
}

bool DivergenceAnalysis::inRegion(const MachineInstr &I) const {
  return I.getParent() && inRegion(*I.getParent());
}

bool DivergenceAnalysis::inRegion(const MachineBasicBlock &BB) const {
  return (!RegionLoop && BB.getParent() == &F) || RegionLoop->contains(&BB);
}

// marks all users of loop-carried values of the loop headed by LoopHeader as
// divergent
void DivergenceAnalysis::taintLoopLiveOuts(const MachineBasicBlock &LoopHeader) {
  auto *DivLoop = LI.getLoopFor(&LoopHeader);
  assert(DivLoop && "loopHeader is not actually part of a loop");

  SmallVector<MachineBasicBlock *, 8> TaintStack;
  DivLoop->getExitBlocks(TaintStack);

  // Otherwise potential users of loop-carried values could be anywhere in the
  // dominance region of DivLoop (including its fringes for phi nodes)
  DenseSet<const MachineBasicBlock *> Visited;
  for (auto *Block : TaintStack) {
    Visited.insert(Block);
  }
  Visited.insert(&LoopHeader);

  while (!TaintStack.empty()) {
    auto *UserBlock = TaintStack.back();
    TaintStack.pop_back();

    // don't spread divergence beyond the region
    if (!inRegion(*UserBlock))
      continue;

    assert(!DivLoop->contains(UserBlock) &&
           "irreducible control flow detected");

    // phi nodes at the fringes of the dominance region
    if (!DT.dominates(&LoopHeader, UserBlock)) {
      // all PHI nodes of UserBlock become divergent
      pushPHINodes(*UserBlock);
      continue;
    }

    // taint outside users of values carried by DivLoop
    for (auto &I : *UserBlock) {
      if (isAlwaysUniformMI(&I, SIII, SIRI, MRI))
        continue;
      if (isDivergent(I))
        continue;

      for (auto &Op : I.uses()) {
        if (!Op.isReg())
          continue;
        unsigned OpReg = Op.getReg();
        MachineInstr *OpInst = MRI.getUniqueVRegDef(OpReg);
        if (!OpInst)
          continue;
        if (DivLoop->contains(OpInst->getParent())) {
          markDivergent(I);
          pushUsers(I);
          break;
        }
      }
    }

    // visit all blocks in the dominance region
    for (auto *SuccBlock : UserBlock->successors()) {
      if (!Visited.insert(SuccBlock).second) {
        continue;
      }
      TaintStack.push_back(SuccBlock);
    }
  }
}

void DivergenceAnalysis::pushInstruction(const MachineInstr &I) { 
  Worklist.push_back(&I);
}
void DivergenceAnalysis::pushPHINodes(const MachineBasicBlock &Block) {
  for (const auto &Phi : Block.phis()) {
    if (isDivergent(Phi))
      continue;
    pushInstruction(Phi);
  }
}

void DivergenceAnalysis::pushUsers(const ValueTy V) {
  for (const auto &UserInst : MRI.use_nodbg_instructions(V)) {

    if (isDivergent(UserInst))
      continue;

    // only compute divergent inside loop
    if (!inRegion(UserInst))
      continue;

    Worklist.push_back(&UserInst);
  }
}
void DivergenceAnalysis::pushUsers(const MachineInstr &I) {
  for (const auto &DstMO : I.defs()) {
    unsigned Reg = DstMO.getReg();
    pushUsers(Reg);
  }
}

bool DivergenceAnalysis::propagateJoinDivergence(const MachineBasicBlock &JoinBlock,
                                                 const MachineLoop *BranchLoop) {
  LLVM_DEBUG(dbgs() << "\tpropJoinDiv " << JoinBlock.getName() << "\n");

  // ignore divergence outside the region
  if (!inRegion(JoinBlock)) {
    return false;
  }

  // push non-divergent phi nodes in JoinBlock to the worklist
  pushPHINodes(JoinBlock);

  // JoinBlock is a divergent loop exit
  if (BranchLoop && !BranchLoop->contains(&JoinBlock)) {
    return true;
  }

  // disjoint-paths divergent at JoinBlock
  markBlockJoinDivergent(JoinBlock);
  return false;
}

void DivergenceAnalysis::propagateBranchDivergence(const MachineInstr &Term) {
  LLVM_DEBUG(dbgs() << "propBranchDiv " << Term.getParent()->getName() << "\n");

  markDivergent(Term);

  const auto *BranchLoop = LI.getLoopFor(Term.getParent());

  // whether there is a divergent loop exit from BranchLoop (if any)
  bool IsBranchLoopDivergent = false;

  // iterate over all blocks reachable by disjoint from Term within the loop
  // also iterates over loop exits that become divergent due to Term.
  for (const auto *JoinBlock : SDA.join_blocks(Term)) {
    IsBranchLoopDivergent |= propagateJoinDivergence(*JoinBlock, BranchLoop);
  }

  // Branch loop is a divergent loop due to the divergent branch in Term
  if (IsBranchLoopDivergent) {
    assert(BranchLoop);
    if (!DivergentLoops.insert(BranchLoop).second) {
      return;
    }
    propagateLoopDivergence(*BranchLoop);
  }
}

void DivergenceAnalysis::propagateLoopDivergence(const MachineLoop &ExitingLoop) {
  LLVM_DEBUG(dbgs() << "propLoopDiv " << ExitingLoop.getHeader()->getNumber() << "\n");

  // don't propagate beyond region
  if (!inRegion(*ExitingLoop.getHeader()))
    return;

  const auto *BranchLoop = ExitingLoop.getParentLoop();

  // Uses of loop-carried values could occur anywhere
  // within the dominance region of the definition. All loop-carried
  // definitions are dominated by the loop header (reducible control).
  // Thus all users have to be in the dominance region of the loop header,
  // except PHI nodes that can also live at the fringes of the dom region
  // (incoming defining value).
  if (!IsLCSSAForm)
    taintLoopLiveOuts(*ExitingLoop.getHeader());

  // whether there is a divergent loop exit from BranchLoop (if any)
  bool IsBranchLoopDivergent = false;

  // iterate over all blocks reachable by disjoint paths from exits of
  // ExitingLoop also iterates over loop exits (of BranchLoop) that in turn
  // become divergent.
  for (const auto *JoinBlock : SDA.join_blocks(ExitingLoop)) {
    IsBranchLoopDivergent |= propagateJoinDivergence(*JoinBlock, BranchLoop);
  }

  // Branch loop is a divergent due to divergent loop exit in ExitingLoop
  if (IsBranchLoopDivergent) {
    assert(BranchLoop);
    if (!DivergentLoops.insert(BranchLoop).second) {
      return;
    }
    propagateLoopDivergence(*BranchLoop);
  }
}

// For case like
//  %149:sreg_64_xexec = S_MOV_B64 $exec
//
//bb.3:
//; predecessors: %bb.3, %bb.2
//  successors: %bb.3(0x40000000), %bb.4(0x40000000); %bb.3(50.00%), %bb.4(50.00%)
//
//  %148:vreg_512 = PHI %56:vreg_512, %bb.2, %55:vreg_512, %bb.3
//  %153:sgpr_32 = V_READFIRSTLANE_B32 %36:vgpr_32, implicit $exec
//  %154:sreg_64 = V_CMP_EQ_U32_e64 %153:sgpr_32, %36:vgpr_32, implicit $exec
//  %152:sreg_64 = S_AND_SAVEEXEC_B64 %154:sreg_64, implicit-def $exec, implicit-def $scc, implicit $exec
//  $m0 = S_MOV_B32 %153:sgpr_32
//  %55:vreg_512 = V_MOVRELD_B32_V16 %148:vreg_512(tied-def 0), -2, 0, implicit $m0, implicit $exec
//  $exec = S_XOR_B64_term $exec, %152:sreg_64, implicit-def $scc
//  S_CBRANCH_EXECNZ %bb.3, implicit $exec
//
//bb.4:
//; predecessors: %bb.3
//  successors: %bb.5(0x80000000); %bb.5(100.00%)
//
//  $exec = S_MOV_B64 %149:sreg_64_xexec

// bb.3 is inside exec region which exec is saved by %149.
// %152:sreg_64 = S_AND_SAVEEXEC_B64 will update the exec which cause divergence
// when it is not bituniform. Everything inside the exec region need to be
// scaned. Out region or phi use should be marked as divergent and add users to
// worklist.
void DivergenceAnalysis::propagateExecControlFlowDivergence(
    const MachineInstr &SaveExec) {
  const MachineBasicBlock *MBB = SaveExec.getParent();
  auto it = ExecRegionMap.find(MBB);
  if (it == ExecRegionMap.end())
    return;
  ExecRegion &Region = *it->second;
  // One region only need to propagate once.
  if (Region.bPropagated)
    return;
  Region.bPropagated = true;
  // Scan all MIs in the region. Mark out region or phi use as divergent and add
  // their users to worklist.
  auto propagateExecDivergence = [this, Region](const MachineInstr *MI) {
    for (const auto &DstMO : MI->defs()) {
      Register Reg = DstMO.getReg();
      // Only VCC/Exec/m0.
      // Exec always uniform. Assume VCC and m0 not cross region.
      if (Reg.isPhysical())
        continue;
      for (const auto &UserInst : MRI.use_nodbg_instructions(Reg)) {

        if (isDivergent(UserInst))
          continue;

        // only propagate user outside of region or phi which will not be
        // guarded by saveExec.
        if (UserInst.getOpcode() != AMDGPU::PHI &&
            isInsideExecRegion(UserInst, *Region.begin, *Region.end, DT, PDT)) {
          continue;
        }
        // Write exec is not divergent.
        if (isWriteExec(&UserInst))
          continue;

        markDivergent(UserInst);
        pushUsers(UserInst);
      }
    }
  };
  const MachineBasicBlock *RegionBeginMBB = Region.begin->getParent();
  const MachineBasicBlock *RegionEndMBB = Region.end->getParent();
  if (RegionBeginMBB != RegionEndMBB) {
    auto it = Region.begin->getIterator();
    for (it++; it != RegionBeginMBB->end(); it++) {
      const MachineInstr &MI = *it;
      propagateExecDivergence(&MI);
    }

    // All blocks between RegionBeginMBB and RegionEndMBB.
    for (const MachineBasicBlock *MBB : Region.blocks) {
      for (const MachineInstr &MI : *MBB) {
        propagateExecDivergence(&MI);
      }
    }

    for (auto it = RegionEndMBB->begin(); it != Region.end->getIterator();
         it++) {
      const MachineInstr &MI = *it;
      propagateExecDivergence(&MI);
    }

  } else {
    auto it = Region.begin->getIterator();
    for (it++; it != Region.end->getIterator(); it++) {
      const MachineInstr &MI = *it;
      propagateExecDivergence(&MI);
    }
  }
}

void DivergenceAnalysis::compute() {
  SmallVector<ExecRegion, 4> ExecRegions;
  // Build exec regions.
  // Add VCndMask for non-bituniform caused by input sreg.
  for (const MachineBasicBlock &MBB : F) {
    for (const MachineInstr &Term : MBB.terminators()) {
      if (updateTerminator(Term))
        pushInstruction(Term);
    }

    for (const MachineInstr &I : MBB) {
      unsigned Opcode = I.getOpcode();
      if (isVCndMask(Opcode)) {
        // Cond for CndMask needs bit uniform check.
        // Add it to worklist to check bit uniform from input.
        pushInstruction(I);
      } else if (isRestoreExec(&I)) {
        const MachineInstr *RegionBegin =
            findExecRegionBeginFromRegionEnd(&I, MRI);
        if (RegionBegin) {
          ExecRegions.emplace_back(ExecRegion(RegionBegin, &I));
        }
      }
    }
  }

  // Build exec region map.
  for (const MachineBasicBlock &MBB : F) {
    for (ExecRegion &Region : ExecRegions) {
      if (isInsideExecRegion(MBB, *Region.begin, *Region.end, DT, PDT)) {
        // Add block to region.
        if (&MBB != Region.begin->getParent() &&
            &MBB != Region.end->getParent())
          Region.blocks.emplace_back(&MBB);
        // Update ExecRegionMap.
        auto it = ExecRegionMap.find(&MBB);
        if (it == ExecRegionMap.end()) {
          ExecRegionMap[&MBB] = &Region;
        } else {
          // When MBB inside multiple regions, save the smallest one.
          if (isInsideExecRegion(*Region.begin, *it->second->begin,
                                 *it->second->end, DT, PDT)) {
            ExecRegionMap[&MBB] = &Region;
          }
        }
      }
    }
  }

  for (auto DivVal : DivergentValues) {
    LLVM_DEBUG(dbgs() << "\t sourceOfDivergence :"; printReg(DivVal, SIRI);
               dbgs() << "\n";);
    pushUsers(DivVal);
  }

  // propagate divergence
  while (!Worklist.empty()) {
    const MachineInstr *I= Worklist.back();
    Worklist.pop_back();

    // maintain uniformity of overrides
    if (isAlwaysUniformMI(I, SIII, SIRI, MRI)) {
      // If used by terminators, and not bit uniform.
      // Add terminator.
      SmallVector<const MachineInstr *, 2> TermUsers;
      for (const auto &DstMO : I->defs()) {
        unsigned Reg = DstMO.getReg();
        for (const auto &UserInst : MRI.use_nodbg_instructions(Reg)) {

          if (isDivergent(UserInst))
            continue;
          // Only check terminator here.
          if (!UserInst.isTerminator())
            continue;

          // only compute divergent inside loop
          if (!inRegion(UserInst))
            continue;

          TermUsers.emplace_back(&UserInst);
        }
      }

      if (!TermUsers.empty()) {
        llvm::DenseMap<const MachineInstr *, bool> Processed;
        if (!isBitUniform(*I, Processed)) {
          for (const MachineInstr *Term : TermUsers) {
            Worklist.emplace_back(Term);
          }
        }
      }

      continue;
    }

    bool WasDivergent = isDivergent(*I);
    if (WasDivergent)
      continue;

    // propagate divergence caused by terminator
    if (I->isTerminator()) {
      if (updateTerminator(*I)) {
        // propagate control divergence to affected instructions
        propagateBranchDivergence(*I);
        continue;
      }
    }

    // update divergence of I due to divergent operands
    bool DivergentUpd = false;
    unsigned Opcode = I->getOpcode();
    switch (I->getOpcode()) {
    default:
      if (isVCndMask(Opcode)) {
        DivergentUpd = updateVCndMask(*I);
      } else {
        DivergentUpd = updateNormalInstruction(*I);
        llvm::DenseMap<const MachineInstr *, bool> Processed;
        if ((DivergentUpd || !isBitUniform(*I, Processed)) && isWriteExec(I)) {
          // propagate exec control divergence to affected instructions.
          propagateExecControlFlowDivergence(*I);
        }
      }
      break;
    case AMDGPU::PHI:
      DivergentUpd = updatePHINode(*I);
      break;
    }

    // propagate value divergence to users
    if (DivergentUpd) {
      markDivergent(*I);
      pushUsers(*I);
    }
  }
}

bool DivergenceAnalysis::isAlwaysUniform(const ValueTy V) const {
  return UniformOverrides.find(V) != UniformOverrides.end();
}

bool DivergenceAnalysis::isDivergent(const ValueTy V) const {
  return DivergentValues.find(V) != DivergentValues.end();
}

bool DivergenceAnalysis::isDivergent(const MachineOperand &MO) const {
  if (!MO.isReg())
    return false;
  Register Reg = MO.getReg();
  if (Reg.isPhysical()) {
    const MachineInstr *MI = MO.getParent();
    if (MI)
      return isDivergent(!MI);

  } else {
    return isDivergent(Reg);
  }
  return true;
}

bool DivergenceAnalysis::isDivergent(const MachineInstr &I) const {
  if (UniformOverridesInsts.find(&I) != UniformOverridesInsts.end())
    return false;
  if (DivergentInsts.find(&I) != DivergentInsts.end())
    return true;
  for (const MachineOperand &DstMO : I.defs()) {
    unsigned Reg = DstMO.getReg();
    if (isDivergent(Reg))
      return true;
  }
  return false;
}

void DivergenceAnalysis::print(raw_ostream &OS, const Module_ *) const {
  // iterate instructions using instructions() to ensure a deterministic order.
  for (auto &MBB : F)
  for (auto &I : MBB) {
    if (isDivergent(I))
      OS << "DIVERGENT:" << I ;
    // AMDGPU changes begin
    else
      OS << "UNIFORM:" << I ;
    // AMDGPU changes end
  }
}

// class GPUDivergenceAnalysis
MirGPUDivergenceAnalysis::MirGPUDivergenceAnalysis(MachineFunction &F,
                                             const MachineDominatorTree &DT,
                                             const MachinePostDominatorTree &PDT,
                                             const MachineLoopInfo &LI)
    : SDA(DT, PDT, LI, /*AMDGPU change*/DivergentJoinMap),
      DA(F, nullptr, DT, PDT, LI, SDA, false, /*AMDGPU change*/DivergentJoinMap) {
  MachineRegisterInfo &MRI = F.getRegInfo();
  const GCNSubtarget *ST = &F.getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *SIRI = ST->getRegisterInfo();
  const SIInstrInfo *SIII = ST->getInstrInfo();
  for (auto &MBB : F)
    for (auto &I : MBB) {
      if (isSourceOfDivergence(&I, MRI, SIRI, SIII)) {
        DA.markDivergent(I);
      } else if (isAlwaysUniformMI(&I, SIII, SIRI, MRI)) {
        DA.addUniformOverride(I);
      }
    }
  for (auto &ArgIt : F.getRegInfo().liveins()) {
    unsigned Reg = ArgIt.first;
    if (isDivergentInputReg(Reg, MRI, SIRI)) {
      DA.markDivergent(Reg);
    }
  }

  DA.compute();
}

bool MirGPUDivergenceAnalysis::isDivergent(const MachineInstr *I) const {
  return DA.isDivergent(*I);
}

void MirGPUDivergenceAnalysis::print(raw_ostream &OS, const Module_ *mod) const {
  OS << "Divergence of kernel " << DA.getFunction().getName() << " {\n";
  DA.print(OS, mod);
  OS << "}\n";
}

}
