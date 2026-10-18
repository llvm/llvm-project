//===-- AGXSchedule.td - AGX scheduling models -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Latencies are in core cycles, for a single SIMD-group. The G14 numbers are
// measurements on G14C (M2 Max) under Asahi Linux, see AGXUsage.rst for the
// methodology and the source data set. G13 reuses them; the Apple9 model is a
// placeholder with the same values.
//
//===----------------------------------------------------------------------===//

// Simple integer ALU: iadd, bitop, mov.
def WriteALU : SchedWrite;
// 64-bit iadd and 32x32->64 imadd.
def WriteALU64 : SchedWrite;
// imadd, bit field, shift, bit count, convert.
def WriteIMul : SchedWrite;
// fadd, fmul, fmadd (32 and 16 bit), cmpsel.
def WriteFP : SchedWrite;
// floor, ceil, trunc, rint, sin_pt_1.
def WriteFRound : SchedWrite;
// rcp, log2, exp2, sin_pt_2.
def WriteSFU : SchedWrite;
// rsqrt, rsqrt_special.
def WriteSFUSlow : SchedWrite;
// mov_imm.
def WriteMovImm : SchedWrite;
// get_sr.
def WriteSR : SchedWrite;
// simd_shuffle*, ballots.
def WriteShuffle : SchedWrite;
// quad_shuffle*.
def WriteQuadShuffle : SchedWrite;
// simd reductions and scans.
def WriteSIMDReduce : SchedWrite;
// quad reductions and scans.
def WriteQuadReduce : SchedWrite;
// device memory and stack loads.
def WriteDeviceLoad : SchedWrite;
def WriteDeviceStore : SchedWrite;
// threadgroup memory.
def WriteLocalLoad : SchedWrite;
def WriteLocalStore : SchedWrite;
def WriteAtomic : SchedWrite;
// Control flow, barriers and waits.
def WriteControl : SchedWrite;
def WriteBarrier : SchedWrite;

class AGXModel : SchedMachineModel {
  let MicroOpBufferSize = 0;
  let IssueWidth = 1;
  let LoadLatency = 46;
  let MispredictPenalty = 0;
  let CompleteModel = 0;
  let PostRAScheduler = 1;
}

def AGXG14Model : AGXModel;
def AGXG13Model : AGXModel;
def AGXApple9Model : AGXModel;

// Execution resources of one scheduler of a core.
let SchedModel = AGXG14Model in {
def G14ALU : ProcResource<1>;
def G14SFU : ProcResource<1>;
def G14Mem : ProcResource<1>;
}
let SchedModel = AGXG13Model in {
def G13ALU : ProcResource<1>;
def G13SFU : ProcResource<1>;
def G13Mem : ProcResource<1>;
}
let SchedModel = AGXApple9Model in {
def A9ALU : ProcResource<1>;
def A9SFU : ProcResource<1>;
def A9Mem : ProcResource<1>;
}

multiclass AGXWrites<ProcResource ALU, ProcResource SFU, ProcResource Mem> {
  def : WriteRes<WriteALU, [ALU]> { let Latency = 2; }
  def : WriteRes<WriteALU64, [SFU]> { let Latency = 5; let ReleaseAtCycles = [4]; }
  def : WriteRes<WriteIMul, [SFU]> { let Latency = 3; let ReleaseAtCycles = [2]; }
  def : WriteRes<WriteFP, [ALU]> { let Latency = 2; }
  def : WriteRes<WriteFRound, [SFU]> { let Latency = 3; let ReleaseAtCycles = [2]; }
  def : WriteRes<WriteSFU, [SFU]> { let Latency = 5; let ReleaseAtCycles = [2]; }
  def : WriteRes<WriteSFUSlow, [SFU]> { let Latency = 6; let ReleaseAtCycles = [4]; }
  def : WriteRes<WriteMovImm, [ALU]> { let Latency = 1; }
  def : WriteRes<WriteSR, [ALU]> { let Latency = 3; }
  def : WriteRes<WriteShuffle, [SFU]> { let Latency = 5; let ReleaseAtCycles = [2]; }
  def : WriteRes<WriteQuadShuffle, [ALU]> { let Latency = 4; }
  def : WriteRes<WriteSIMDReduce, [SFU]> { let Latency = 24; let ReleaseAtCycles = [24]; }
  def : WriteRes<WriteQuadReduce, [SFU]> { let Latency = 12; let ReleaseAtCycles = [9]; }
  def : WriteRes<WriteDeviceLoad, [Mem]> { let Latency = 46; }
  def : WriteRes<WriteDeviceStore, [Mem]> { let Latency = 3; let ReleaseAtCycles = [3]; }
  def : WriteRes<WriteLocalLoad, [Mem]> { let Latency = 20; let ReleaseAtCycles = [20]; }
  def : WriteRes<WriteLocalStore, [Mem]> { let Latency = 18; let ReleaseAtCycles = [18]; }
  def : WriteRes<WriteAtomic, [Mem]> { let Latency = 60; }
  def : WriteRes<WriteControl, [ALU]> { let Latency = 2; }
  def : WriteRes<WriteBarrier, [Mem]> { let Latency = 14; }
}

let SchedModel = AGXG14Model in defm : AGXWrites<G14ALU, G14SFU, G14Mem>;
let SchedModel = AGXG13Model in defm : AGXWrites<G13ALU, G13SFU, G13Mem>;
let SchedModel = AGXApple9Model in defm : AGXWrites<A9ALU, A9SFU, A9Mem>;
