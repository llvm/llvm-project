//==----- RISCVSchedSiFive8.td - SiFive8 Scheduling Defs -----*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the common bits of scheduling models for the P400, P500,
// and the P600 series cores.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_RISCV_SIFIVE_RISCVSCHEDSIFIVE8_TD
#define LLVM_TARGET_RISCV_SIFIVE_RISCVSCHEDSIFIVE8_TD

multiclass SiFive8ProcResources<int num_iex, int num_fex, int num_vex = 1> {
  // Integer pipes
  foreach i = !range(num_iex) in
  def IEXQ#i  : ProcResource<1>;

  if !gt(num_iex, 1) then
    def IntArith  : ProcResGroup<!foreach(i, !range(num_iex),
                                          !cast<ProcResource>(NAME#"IEXQ"#i))>;
  def Div         : ProcResource<1>;

  // Floating point pipes
  foreach i = !range(num_fex) in
  def FEXQ#i       : ProcResource<1>;

  if !gt(num_fex, 1) then
    def FloatArith  : ProcResGroup<!foreach(i, !range(num_fex),
                                            !cast<ProcResource>(NAME#"FEXQ"#i))>;
  def FloatDiv    : ProcResource<1>;

  // Load/Store units
  def Load        : ProcResource<1>;
  def Store       : ProcResource<1>;

  // Vector pipes
  foreach i = !range(num_vex) in
  def VEXQ#i  : ProcResource<1>;

  if !gt(num_vex, 1) then
    def VectorArith  : ProcResGroup<!foreach(i, !range(num_vex),
                                             !cast<ProcResource>(NAME#"VEXQ"#i))>;

  if !gt(num_vex, 0) then {
    def VLD          : ProcResource<1>;
    def VST          : ProcResource<1>;
    def VDiv         : ProcResource<1>;
    def VFloatDiv    : ProcResource<1>;
  }
}

class SiFive8IntProcResources {
  ProcResourceKind Arith;
  ProcResourceKind Mul;
  ProcResourceKind I2F;
  list<ProcResourceKind> Div;
  ProcResourceKind Branch;
  ProcResourceKind SYS;
  ProcResourceKind VSet;
  ProcResourceKind CMOV;
}

class SiFive8FloatProcResources {
  ProcResourceKind Arith;
  list<ProcResourceKind> Div;
  ProcResourceKind F2I;
}

class SiFive8LSUProcResources {
  ProcResourceKind Load;
  ProcResourceKind Store;
}

class SiFive8VecProcResources {
  ProcResourceKind Arith;
  ProcResourceKind Load;
  ProcResourceKind Store;
  list<ProcResourceKind> Div;
  list<ProcResourceKind> FloatDiv;
}

defvar SiFive8VLEN = 128;

multiclass SiFive8BaseWriteRes<SiFive8IntProcResources IntProcRes,
                               SiFive8FloatProcResources FloatProcRes,
                               SiFive8LSUProcResources LSUProcRes,
                               bit legacy_imul_latency = false> {

  // Integer arithmetic and logic
  def : WriteRes<WriteIALU, [IntProcRes.Arith]>;
  def : WriteRes<WriteIALU32, [IntProcRes.Arith]>;
  def : WriteRes<WriteShiftImm, [IntProcRes.Arith]>;
  def : WriteRes<WriteShiftImm32, [IntProcRes.Arith]>;
  def : WriteRes<WriteShiftReg, [IntProcRes.Arith]>;
  def : WriteRes<WriteShiftReg32, [IntProcRes.Arith]>;
  // Branching
  def : WriteRes<WriteJmp, [IntProcRes.Branch]>;
  def : WriteRes<WriteJal, [IntProcRes.Branch]>;
  def : WriteRes<WriteJalr, [IntProcRes.Branch]>;

  // CMOV
  def WriteCMOV : SchedWriteRes<[IntProcRes.Branch, IntProcRes.CMOV]> {
    let Latency = 2;
    let NumMicroOps = 2;
  }
  def : InstRW<[!cast<SchedWriteRes>(NAME#"WriteCMOV")], (instrs PseudoCCMOVGPRNoX0)>;

  let Latency = !if(legacy_imul_latency, 3, 2) in {
  // Integer multiplication
  def : WriteRes<WriteIMul, [IntProcRes.Mul]>;
  def : WriteRes<WriteIMul32, [IntProcRes.Mul]>;
  // cpop[w] look exactly like multiply.
  def : WriteRes<WriteCPOP, [IntProcRes.Mul]>;
  def : WriteRes<WriteCPOP32, [IntProcRes.Mul]>;
  }

  // Integer division
  def : WriteRes<WriteIDiv, IntProcRes.Div> {
    let Latency = 35;
    let ReleaseAtCycles = [1, 34];
  }
  def : WriteRes<WriteIDiv32,  IntProcRes.Div> {
    let Latency = 20;
    let ReleaseAtCycles = [1, 19];
  }

  // Integer remainder
  def : WriteRes<WriteIRem, IntProcRes.Div> {
    let Latency = 35;
    let ReleaseAtCycles = [1, 34];
  }
  def : WriteRes<WriteIRem32,  IntProcRes.Div> {
    let Latency = 20;
    let ReleaseAtCycles = [1, 19];
  }

  // Bitmanip
  def : WriteRes<WriteRotateImm, [IntProcRes.Arith]>;
  def : WriteRes<WriteRotateImm32, [IntProcRes.Arith]>;
  def : WriteRes<WriteRotateReg, [IntProcRes.Arith]>;
  def : WriteRes<WriteRotateReg32, [IntProcRes.Arith]>;

  def : WriteRes<WriteCLZ, [IntProcRes.Arith]>;
  def : WriteRes<WriteCLZ32, [IntProcRes.Arith]>;
  def : WriteRes<WriteCTZ, [IntProcRes.Arith]>;
  def : WriteRes<WriteCTZ32, [IntProcRes.Arith]>;

  def : WriteRes<WriteORCB, [IntProcRes.Arith]>;
  def : WriteRes<WriteIMinMax, [IntProcRes.Arith]>;

  def : WriteRes<WriteREV8, [IntProcRes.Arith]>;

  def : WriteRes<WriteSHXADD, [IntProcRes.Arith]>;
  def : WriteRes<WriteSHXADD32, [IntProcRes.Arith]>;

  def : WriteRes<WriteSingleBit, [IntProcRes.Arith]>;
  def : WriteRes<WriteSingleBitImm, [IntProcRes.Arith]>;
  def : WriteRes<WriteBEXT, [IntProcRes.Arith]>;
  def : WriteRes<WriteBEXTI, [IntProcRes.Arith]>;

  // Memory
  def : WriteRes<WriteSTB, [LSUProcRes.Store]>;
  def : WriteRes<WriteSTH, [LSUProcRes.Store]>;
  def : WriteRes<WriteSTW, [LSUProcRes.Store]>;
  def : WriteRes<WriteSTD, [LSUProcRes.Store]>;
  def : WriteRes<WriteFST16, [LSUProcRes.Store]>;
  def : WriteRes<WriteFST32, [LSUProcRes.Store]>;
  def : WriteRes<WriteFST64, [LSUProcRes.Store]>;

  let Latency = 4 in {
  def : WriteRes<WriteLDB, [LSUProcRes.Load]>;
  def : WriteRes<WriteLDH, [LSUProcRes.Load]>;
  }
  let Latency = 4 in {
  def : WriteRes<WriteLDW, [LSUProcRes.Load]>;
  def : WriteRes<WriteLDD, [LSUProcRes.Load]>;
  }

  let Latency = 5 in {
  def : WriteRes<WriteFLD16, [LSUProcRes.Load]>;
  def : WriteRes<WriteFLD32, [LSUProcRes.Load]>;
  def : WriteRes<WriteFLD64, [LSUProcRes.Load]>;
  }

  // Atomic memory
  let Latency = 3 in {
  def : WriteRes<WriteAtomicSTW, [LSUProcRes.Store]>;
  def : WriteRes<WriteAtomicSTD, [LSUProcRes.Store]>;
  def : WriteRes<WriteAtomicW,   [LSUProcRes.Load]>;
  def : WriteRes<WriteAtomicD,   [LSUProcRes.Load]>;
  def : WriteRes<WriteAtomicLDW, [LSUProcRes.Load]>;
  def : WriteRes<WriteAtomicLDD, [LSUProcRes.Load]>;
  }

  // Floating point
  let Latency = 4 in {
  def : WriteRes<WriteFMA16, [FloatProcRes.Arith]>;
  def : WriteRes<WriteFMA32, [FloatProcRes.Arith]>;
  def : WriteRes<WriteFMA64, [FloatProcRes.Arith]>;
  }

  let Latency = 2 in {
  def : WriteRes<WriteFSGNJ16, [FloatProcRes.Arith]>;
  def : WriteRes<WriteFSGNJ32, [FloatProcRes.Arith]>;
  def : WriteRes<WriteFSGNJ64, [FloatProcRes.Arith]>;

  def : WriteRes<WriteFMinMax16, [FloatProcRes.Arith]>;
  def : WriteRes<WriteFMinMax32, [FloatProcRes.Arith]>;
  def : WriteRes<WriteFMinMax64, [FloatProcRes.Arith]>;
  }

  // Half precision.
  def : WriteRes<WriteFSqrt16, FloatProcRes.Div> {
    let Latency = 18;
    let ReleaseAtCycles = [1, 17];
  }

  // Single precision.
  def : WriteRes<WriteFSqrt32, FloatProcRes.Div> {
    let Latency = 18;
    let ReleaseAtCycles = [1, 17];
  }

  // Double precision
  def : WriteRes<WriteFSqrt64, FloatProcRes.Div> {
    let Latency = 33;
    let ReleaseAtCycles = [1, 32];
  }

  // Conversions
  let Latency = 2 in {
  def : WriteRes<WriteFCvtI32ToF16, [IntProcRes.I2F]>;
  def : WriteRes<WriteFCvtI32ToF32, [IntProcRes.I2F]>;
  def : WriteRes<WriteFCvtI32ToF64, [IntProcRes.I2F]>;
  def : WriteRes<WriteFCvtI64ToF16, [IntProcRes.I2F]>;
  def : WriteRes<WriteFCvtI64ToF32, [IntProcRes.I2F]>;
  def : WriteRes<WriteFCvtI64ToF64, [IntProcRes.I2F]>;
  def : WriteRes<WriteFCvtF16ToI32, [FloatProcRes.F2I]>;
  def : WriteRes<WriteFCvtF16ToI64, [FloatProcRes.F2I]>;
  def : WriteRes<WriteFCvtF16ToF32, [FloatProcRes.Arith]>;
  def : WriteRes<WriteFCvtF16ToF64, [FloatProcRes.Arith]>;
  def : WriteRes<WriteFCvtF32ToI32, [FloatProcRes.F2I]>;
  def : WriteRes<WriteFCvtF32ToI64, [FloatProcRes.F2I]>;
  def : WriteRes<WriteFCvtF32ToF16, [FloatProcRes.Arith]>;
  def : WriteRes<WriteFCvtF32ToF64, [FloatProcRes.Arith]>;
  def : WriteRes<WriteFCvtF64ToI32, [FloatProcRes.F2I]>;
  def : WriteRes<WriteFCvtF64ToI64, [FloatProcRes.F2I]>;
  def : WriteRes<WriteFCvtF64ToF16, [FloatProcRes.Arith]>;
  def : WriteRes<WriteFCvtF64ToF32, [FloatProcRes.Arith]>;
  // FROUND/FROUNDNX are implemented by `FCVT.f.f`.
  def : WriteRes<WriteFRoundF16, [FloatProcRes.Arith]>;
  def : WriteRes<WriteFRoundF32, [FloatProcRes.Arith]>;
  def : WriteRes<WriteFRoundF64, [FloatProcRes.Arith]>;

  def : WriteRes<WriteFClass16, [FloatProcRes.F2I]>;
  def : WriteRes<WriteFClass32, [FloatProcRes.F2I]>;
  def : WriteRes<WriteFClass64, [FloatProcRes.F2I]>;
  def : WriteRes<WriteFCmp16, [FloatProcRes.F2I]>;
  def : WriteRes<WriteFCmp32, [FloatProcRes.F2I]>;
  def : WriteRes<WriteFCmp64, [FloatProcRes.F2I]>;
  def : WriteRes<WriteFMovI16ToF16, [IntProcRes.I2F]>;
  def : WriteRes<WriteFMovF16ToI16, [FloatProcRes.F2I]>;
  def : WriteRes<WriteFMovI32ToF32, [IntProcRes.I2F]>;
  def : WriteRes<WriteFMovF32ToI32, [FloatProcRes.F2I]>;
  def : WriteRes<WriteFMovI64ToF64, [IntProcRes.I2F]>;
  def : WriteRes<WriteFMovF64ToI64, [FloatProcRes.F2I]>;
  // FLI is implemented by `FMV.f.x`.
  def : WriteRes<WriteFLI16, [IntProcRes.I2F]>;
  def : WriteRes<WriteFLI32, [IntProcRes.I2F]>;
  def : WriteRes<WriteFLI64, [IntProcRes.I2F]>;
  }

  // Others
  def : WriteRes<WriteCSR, [IntProcRes.SYS]>;
  def : WriteRes<WriteNop, []>;

  // FIXME: This could be better modeled by looking at the regclasses of the operands.
  def : InstRW<[WriteIALU, ReadIALU], (instrs COPY)>;
}

/// c is true if mx has the worst case behavior compared to LMULs in MxList.
/// In SiFive8Vec, the worst case LMUL is the Largest LMUL
/// and the worst case sew is the smallest SEW for that LMUL.
class SiFive8VecIsWorstCaseMX<string mx, list<string> MxList> {
  string LLMUL = LargestLMUL<MxList>.r;
  bit c = !eq(mx, LLMUL);
}

class SiFive8VecIsWorstCaseMXSEW<string mx, int sew, list<string> MxList, bit isF = 0> {
  string LLMUL = LargestLMUL<MxList>.r;
  int SSEW = SmallestSEW<mx, isF>.r;
  bit c = !and(!eq(mx, LLMUL), !eq(sew, SSEW));
}

// 1 Micro-Op per cycle.
class SiFive8VecGetLMulCycles<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 1,
    !eq(mx, "M2") : 2,
    !eq(mx, "M4") : 4,
    !eq(mx, "M8") : 8,
    !eq(mx, "MF2") : 1,
    !eq(mx, "MF4") : 1,
    !eq(mx, "MF8") : 1
  );
}

class SiFive8VecGetVLMAX<string mx, int sew> {
  defvar LMUL = SiFive8VecGetLMulCycles<mx>.c;
  int val = !cond(
    !eq(mx, "MF2") : !div(!div(SiFive8VLEN, 2), sew),
    !eq(mx, "MF4") : !div(!div(SiFive8VLEN, 4), sew),
    !eq(mx, "MF8") : !div(!div(SiFive8VLEN, 8), sew),
    true: !div(!mul(SiFive8VLEN, LMUL), sew)
  );
}

class SiFive8VecStridedLdStLatency<string mx, int sew> {
  defvar VL = SiFive8VecGetVLMAX<mx, sew>.val;
  int val = !cond(
    !eq(VL, 2):  13,
    !eq(VL, 4):  18,
    !eq(VL, 8):  22,
    !eq(VL, 16): 30,
    // VL=32,64,128
    true: !sub(VL, 2)
  );
}

// Latency for segmented loads and stores are calculated as vl * nf.
class SiFive8VecSegmentedLdStCycles<string mx, int sew, int nf> {
  int c = !mul(SiFive8VecGetVLMAX<mx, sew>.val, nf);
}

// All the shared WriteRes entries
multiclass SiFive8VecBaseWriteRes<SiFive8IntProcResources IntProcRes,
                                  SiFive8VecProcResources VecProcRes> {
  // Vector Byte Length vlenb
  def : WriteRes<WriteRdVLENB, [IntProcRes.SYS]>;

  // Configuration-Setting Instructions
  def : WriteRes<WriteVSETVLI, [IntProcRes.VSet]>;
  def : WriteRes<WriteVSETIVLI, [IntProcRes.VSet]>;
  def : WriteRes<WriteVSETVL, [IntProcRes.VSet]>;

  // Vector Loads and Stores

  // Note that the latency of vector loads are measured by consuming the loaded
  // value with vmv.x.s before subtracting the latency of vmv.x.s from the number.
  foreach mx = SchedMxList in {
    defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFive8VecIsWorstCaseMX<mx, SchedMxList>.c;
    // The occupancy of unit-strided load / store is equal to LMUL.
    let Latency = 8 in {
      let ReleaseAtCycles = [LMulLat] in {
        defm  : LMULWriteResMX<"WriteVLDE",  [VecProcRes.Load], mx, IsWorstCase>;
        defm  : LMULWriteResMX<"WriteVLDFF", [VecProcRes.Load], mx, IsWorstCase>;

        defm  : LMULWriteResMX<"WriteVSTE",  [VecProcRes.Store], mx, IsWorstCase>;
      }

      // Mask load and store always have EMUL=1.
      let ReleaseAtCycles = [SiFive8VecGetLMulCycles<"M1">.c] in {
        defm  : LMULWriteResMX<"WriteVLDM",  [VecProcRes.Load], mx, IsWorstCase=!eq(mx, "M1")>;
        defm  : LMULWriteResMX<"WriteVSTM",  [VecProcRes.Store], mx, IsWorstCase=!eq(mx, "M1")>;
      }
    }
    foreach eew = [8, 16, 32, 64] in {
      let Latency = SiFive8VecStridedLdStLatency<mx, eew>.val,
          ReleaseAtCycles = [SiFive8VecGetVLMAX<mx, eew>.val] in {
        defm  : LMULWriteResMX<"WriteVLDS"  # eew, [VecProcRes.Load], mx, IsWorstCase>;
        defm  : LMULWriteResMX<"WriteVLDUX" # eew, [VecProcRes.Load], mx, IsWorstCase>;
        defm  : LMULWriteResMX<"WriteVLDOX" # eew, [VecProcRes.Load], mx, IsWorstCase>;

        defm  : LMULWriteResMX<"WriteVSTS"  # eew, [VecProcRes.Store], mx, IsWorstCase>;
        defm  : LMULWriteResMX<"WriteVSTUX" # eew, [VecProcRes.Store], mx, IsWorstCase>;
        defm  : LMULWriteResMX<"WriteVSTOX" # eew, [VecProcRes.Store], mx, IsWorstCase>;
      }
    }
  }

  foreach mx = SchedMxList in {
    foreach nf=2-8 in {
      foreach eew = [8, 16, 32, 64] in {
        defvar LMulLat = SiFive8VecSegmentedLdStCycles<mx, eew, nf>.c;
        defvar IsWorstCase = SiFive8VecIsWorstCaseMX<mx, SchedMxList>.c;
        let Latency = !add(12, LMulLat), ReleaseAtCycles = [!add(12, LMulLat)] in {
          defm  : LMULWriteResMX<"WriteVLSEG" # nf # "e" # eew,   [VecProcRes.Load], mx, IsWorstCase>;
          defm  : LMULWriteResMX<"WriteVLSEGFF" # nf # "e" # eew, [VecProcRes.Load], mx, IsWorstCase>;
          defm  : LMULWriteResMX<"WriteVLSSEG" # nf # "e" # eew,  [VecProcRes.Load], mx, IsWorstCase>;
          defm  : LMULWriteResMX<"WriteVLUXSEG" # nf # "e" # eew, [VecProcRes.Load], mx, IsWorstCase>;
          defm  : LMULWriteResMX<"WriteVLOXSEG" # nf # "e" # eew, [VecProcRes.Load], mx, IsWorstCase>;
        }
        let Latency = !add(1, LMulLat), ReleaseAtCycles = [!add(12, LMulLat)] in {
          defm  : LMULWriteResMX<"WriteVSSEG" # nf # "e" # eew,   [VecProcRes.Store], mx, IsWorstCase>;
          defm  : LMULWriteResMX<"WriteVSSSEG" # nf # "e" # eew,  [VecProcRes.Store], mx, IsWorstCase>;
          defm  : LMULWriteResMX<"WriteVSUXSEG" # nf # "e" # eew, [VecProcRes.Store], mx, IsWorstCase>;
          defm  : LMULWriteResMX<"WriteVSOXSEG" # nf # "e" # eew, [VecProcRes.Store], mx, IsWorstCase>;
        }
      }
    }
  }

  // Whole register move/load/store
  foreach LMul = [1, 2, 4, 8] in {
    let Latency = 8, ReleaseAtCycles = [LMul] in {
      def : WriteRes<!cast<SchedWrite>("WriteVLD" # LMul # "R"), [VecProcRes.Load]>;
      def : WriteRes<!cast<SchedWrite>("WriteVST" # LMul # "R"), [VecProcRes.Store]>;
    }
    let Latency = 2, ReleaseAtCycles = [LMul] in {
      def : WriteRes<!cast<SchedWrite>("WriteVMov" # LMul # "V"), [VecProcRes.Arith]>;
    }
  }

  // Worst case needs 51/45/42/72 * lmul cycles for i8/16/32/64.
  foreach mx = SchedMxList in {
    foreach sew = SchedSEWSet<mx>.val in {
      defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
      defvar IsWorstCase = SiFive8VecIsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
      defvar DivMicroOpLat =
        !cond(!eq(sew, 8): 51, !eq(sew, 16): 45, !eq(sew, 32): 42,
        /* SEW=64 */ true: 72);
      defvar DivLatency = !mul(DivMicroOpLat, LMulLat);
      let Latency = DivLatency, ReleaseAtCycles = [LMulLat, DivLatency] in {
        defm : LMULSEWWriteResMXSEW<"WriteVIDivV", VecProcRes.Div, mx, sew, IsWorstCase>;
        defm : LMULSEWWriteResMXSEW<"WriteVIDivX", VecProcRes.Div, mx, sew, IsWorstCase>;
      }
    }
  }

  // Narrowing Shift and Clips
  foreach mx = SchedMxListW in {
    defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFive8VecIsWorstCaseMX<mx, SchedMxListW>.c;
    let Latency = 2, ReleaseAtCycles = [LMulLat] in {
      defm  : LMULWriteResMX<"WriteVNShiftV", [VecProcRes.Arith], mx, IsWorstCase>;
      defm  : LMULWriteResMX<"WriteVNShiftX", [VecProcRes.Arith], mx, IsWorstCase>;
      defm  : LMULWriteResMX<"WriteVNShiftI", [VecProcRes.Arith], mx, IsWorstCase>;
      defm  : LMULWriteResMX<"WriteVNClipV",  [VecProcRes.Arith], mx, IsWorstCase>;
      defm  : LMULWriteResMX<"WriteVNClipX",  [VecProcRes.Arith], mx, IsWorstCase>;
      defm  : LMULWriteResMX<"WriteVNClipI",  [VecProcRes.Arith], mx, IsWorstCase>;
    }
  }

  // 12. Vector Fixed-Point Arithmetic Instructions
  foreach mx = SchedMxList in {
    defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFive8VecIsWorstCaseMX<mx, SchedMxList>.c;
    let Latency = 6, ReleaseAtCycles = [LMulLat] in {
      defm  : LMULWriteResMX<"WriteVSALUV",   [VecProcRes.Arith], mx, IsWorstCase>;
      defm  : LMULWriteResMX<"WriteVSALUX",   [VecProcRes.Arith], mx, IsWorstCase>;
      defm  : LMULWriteResMX<"WriteVSALUI",   [VecProcRes.Arith], mx, IsWorstCase>;
      defm  : LMULWriteResMX<"WriteVAALUV",   [VecProcRes.Arith], mx, IsWorstCase>;
      defm  : LMULWriteResMX<"WriteVAALUX",   [VecProcRes.Arith], mx, IsWorstCase>;
      defm  : LMULWriteResMX<"WriteVSMulV",   [VecProcRes.Arith], mx, IsWorstCase>;
      defm  : LMULWriteResMX<"WriteVSMulX",   [VecProcRes.Arith], mx, IsWorstCase>;
      defm  : LMULWriteResMX<"WriteVSShiftV", [VecProcRes.Arith], mx, IsWorstCase>;
      defm  : LMULWriteResMX<"WriteVSShiftX", [VecProcRes.Arith], mx, IsWorstCase>;
      defm  : LMULWriteResMX<"WriteVSShiftI", [VecProcRes.Arith], mx, IsWorstCase>;
    }
  }

  // 13. Vector Floating-Point Instructions
  foreach mx = SchedMxListF in {
    foreach sew = SchedSEWSet<mx, isF=1>.val in {
      defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
      defvar IsWorstCase = SiFive8VecIsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;
      let Latency = 6, ReleaseAtCycles = [LMulLat] in {
        defm  : LMULSEWWriteResMXSEW<"WriteVFALUV",  [VecProcRes.Arith], mx, sew, IsWorstCase>;
        defm  : LMULSEWWriteResMXSEW<"WriteVFALUF",  [VecProcRes.Arith], mx, sew, IsWorstCase>;
        defm  : LMULSEWWriteResMXSEW<"WriteVFMulV",  [VecProcRes.Arith], mx, sew, IsWorstCase>;
        defm  : LMULSEWWriteResMXSEW<"WriteVFMulF",  [VecProcRes.Arith], mx, sew, IsWorstCase>;
        defm  : LMULSEWWriteResMXSEW<"WriteVFMulAddV", [VecProcRes.Arith], mx, sew, IsWorstCase>;
        defm  : LMULSEWWriteResMXSEW<"WriteVFMulAddF", [VecProcRes.Arith], mx, sew, IsWorstCase>;
      }
    }
  }

  // Widening
  foreach mx = SchedMxListW in {
    foreach sew = SchedSEWSet<mx, isF=0, isWidening=1>.val in {
      defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
      defvar IsWorstCase = SiFive8VecIsWorstCaseMXSEW<mx, sew, SchedMxListW>.c;
      let Latency = 3, ReleaseAtCycles = [LMulLat] in
      defm  : LMULSEWWriteResMXSEW<"WriteVFWCvtIToFV", [VecProcRes.Arith], mx, sew, IsWorstCase>;
    }
  }
  foreach mx = SchedMxListFW in {
    defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFive8VecIsWorstCaseMX<mx, SchedMxListFW>.c;
    let Latency = 6, ReleaseAtCycles = [LMulLat] in
    defm  : LMULWriteResMX<"WriteVFWCvtFToIV", [VecProcRes.Arith], mx, IsWorstCase>;
  }

  // Narrowing
  foreach mx = SchedMxListW in {
    defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFive8VecIsWorstCaseMX<mx, SchedMxListW>.c;
    let Latency = 3, ReleaseAtCycles = [LMulLat] in {
      defm  : LMULWriteResMX<"WriteVFNCvtFToIV", [VecProcRes.Arith], mx, IsWorstCase>;
    }
  }

  // Worst case needs around 29/25/37 * LMUL cycles for f16/32/64.
  foreach mx = SchedMxListF in {
    foreach sew = SchedSEWSet<mx, 1>.val in {
      defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
      defvar IsWorstCase = SiFive8VecIsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;
      defvar DivMicroOpLat =
        !cond(!eq(sew, 16): 29, !eq(sew, 32): 25, /* SEW=64 */ true: 37);
      defvar DivLatency = !mul(DivMicroOpLat, LMulLat);
      let Latency = DivLatency, ReleaseAtCycles = [LMulLat, DivLatency] in {
        defm  : LMULSEWWriteResMXSEW<"WriteVFDivV",  VecProcRes.FloatDiv, mx, sew, IsWorstCase>;
        defm  : LMULSEWWriteResMXSEW<"WriteVFDivF",  VecProcRes.FloatDiv, mx, sew, IsWorstCase>;
        defm  : LMULSEWWriteResMXSEW<"WriteVFSqrtV", VecProcRes.FloatDiv, mx, sew, IsWorstCase>;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Bypass and advance

multiclass SiFive8BaseReadAdvances {
  def : ReadAdvance<ReadJmp, 0>;
  def : ReadAdvance<ReadJalr, 0>;
  def : ReadAdvance<ReadCSR, 0>;
  def : ReadAdvance<ReadStoreData, 0>;
  def : ReadAdvance<ReadMemBase, 0>;
  def : ReadAdvance<ReadIALU, 0>;
  def : ReadAdvance<ReadIALU32, 0>;
  def : ReadAdvance<ReadShiftImm, 0>;
  def : ReadAdvance<ReadShiftImm32, 0>;
  def : ReadAdvance<ReadShiftReg, 0>;
  def : ReadAdvance<ReadShiftReg32, 0>;
  def : ReadAdvance<ReadIDiv, 0>;
  def : ReadAdvance<ReadIDiv32, 0>;
  def : ReadAdvance<ReadIRem, 0>;
  def : ReadAdvance<ReadIRem32, 0>;
  def : ReadAdvance<ReadIMul, 0>;
  def : ReadAdvance<ReadIMul32, 0>;
  def : ReadAdvance<ReadAtomicWA, 0>;
  def : ReadAdvance<ReadAtomicWD, 0>;
  def : ReadAdvance<ReadAtomicDA, 0>;
  def : ReadAdvance<ReadAtomicDD, 0>;
  def : ReadAdvance<ReadAtomicLDW, 0>;
  def : ReadAdvance<ReadAtomicLDD, 0>;
  def : ReadAdvance<ReadAtomicSTW, 0>;
  def : ReadAdvance<ReadAtomicSTD, 0>;
  def : ReadAdvance<ReadFStoreData, 0>;
  def : ReadAdvance<ReadFMemBase, 0>;
  def : ReadAdvance<ReadFAdd16, 0>;
  def : ReadAdvance<ReadFAdd32, 0>;
  def : ReadAdvance<ReadFAdd64, 0>;
  def : ReadAdvance<ReadFMul16, 0>;
  def : ReadAdvance<ReadFMA16, 0>;
  def : ReadAdvance<ReadFMA16Addend, 0>;
  def : ReadAdvance<ReadFMA32Addend, 0>;
  def : ReadAdvance<ReadFMA64Addend, 0>;
  def : ReadAdvance<ReadFMul32, 0>;
  def : ReadAdvance<ReadFMA32, 0>;
  def : ReadAdvance<ReadFMul64, 0>;
  def : ReadAdvance<ReadFMA64, 0>;
  def : ReadAdvance<ReadFDiv16, 0>;
  def : ReadAdvance<ReadFDiv32, 0>;
  def : ReadAdvance<ReadFDiv64, 0>;
  def : ReadAdvance<ReadFSqrt16, 0>;
  def : ReadAdvance<ReadFSqrt32, 0>;
  def : ReadAdvance<ReadFSqrt64, 0>;
  def : ReadAdvance<ReadFCmp16, 0>;
  def : ReadAdvance<ReadFCmp32, 0>;
  def : ReadAdvance<ReadFCmp64, 0>;
  def : ReadAdvance<ReadFSGNJ16, 0>;
  def : ReadAdvance<ReadFSGNJ32, 0>;
  def : ReadAdvance<ReadFSGNJ64, 0>;
  def : ReadAdvance<ReadFMinMax16, 0>;
  def : ReadAdvance<ReadFMinMax32, 0>;
  def : ReadAdvance<ReadFMinMax64, 0>;
  def : ReadAdvance<ReadFCvtF16ToI32, 0>;
  def : ReadAdvance<ReadFCvtF16ToI64, 0>;
  def : ReadAdvance<ReadFCvtF32ToI32, 0>;
  def : ReadAdvance<ReadFCvtF32ToI64, 0>;
  def : ReadAdvance<ReadFCvtF64ToI32, 0>;
  def : ReadAdvance<ReadFCvtF64ToI64, 0>;
  def : ReadAdvance<ReadFCvtI32ToF16, 0>;
  def : ReadAdvance<ReadFCvtI32ToF32, 0>;
  def : ReadAdvance<ReadFCvtI32ToF64, 0>;
  def : ReadAdvance<ReadFCvtI64ToF16, 0>;
  def : ReadAdvance<ReadFCvtI64ToF32, 0>;
  def : ReadAdvance<ReadFCvtI64ToF64, 0>;
  def : ReadAdvance<ReadFCvtF32ToF64, 0>;
  def : ReadAdvance<ReadFCvtF64ToF32, 0>;
  def : ReadAdvance<ReadFCvtF16ToF32, 0>;
  def : ReadAdvance<ReadFCvtF32ToF16, 0>;
  def : ReadAdvance<ReadFCvtF16ToF64, 0>;
  def : ReadAdvance<ReadFCvtF64ToF16, 0>;
  def : ReadAdvance<ReadFRoundF16, 0>;
  def : ReadAdvance<ReadFRoundF32, 0>;
  def : ReadAdvance<ReadFRoundF64, 0>;
  def : ReadAdvance<ReadFMovF16ToI16, 0>;
  def : ReadAdvance<ReadFMovI16ToF16, 0>;
  def : ReadAdvance<ReadFMovF32ToI32, 0>;
  def : ReadAdvance<ReadFMovI32ToF32, 0>;
  def : ReadAdvance<ReadFMovF64ToI64, 0>;
  def : ReadAdvance<ReadFMovI64ToF64, 0>;
  def : ReadAdvance<ReadFClass16, 0>;
  def : ReadAdvance<ReadFClass32, 0>;
  def : ReadAdvance<ReadFClass64, 0>;

  // Bitmanip
  def : ReadAdvance<ReadRotateImm, 0>;
  def : ReadAdvance<ReadRotateImm32, 0>;
  def : ReadAdvance<ReadRotateReg, 0>;
  def : ReadAdvance<ReadRotateReg32, 0>;
  def : ReadAdvance<ReadCLZ, 0>;
  def : ReadAdvance<ReadCLZ32, 0>;
  def : ReadAdvance<ReadCTZ, 0>;
  def : ReadAdvance<ReadCTZ32, 0>;
  def : ReadAdvance<ReadCPOP, 0>;
  def : ReadAdvance<ReadCPOP32, 0>;
  def : ReadAdvance<ReadORCB, 0>;
  def : ReadAdvance<ReadIMinMax, 0>;
  def : ReadAdvance<ReadREV8, 0>;
  def : ReadAdvance<ReadSHXADD, 0>;
  def : ReadAdvance<ReadSHXADD32, 0>;
  def : ReadAdvance<ReadSingleBit, 0>;
  def : ReadAdvance<ReadSingleBitImm, 0>;
}

multiclass SiFive8VecBaseReadAdvances {
  // 6. Configuration-Setting Instructions
  def : ReadAdvance<ReadVSETVLI, 0>;
  def : ReadAdvance<ReadVSETVL, 0>;

  // 7. Vector Loads and Stores
  def : ReadAdvance<ReadVLDX, 0>;
  def : ReadAdvance<ReadVSTX, 0>;
  defm  : LMULReadAdvance<"ReadVSTEV", 0>;
  defm  : LMULReadAdvance<"ReadVSTM", 0>;
  def : ReadAdvance<ReadVLDSX, 0>;
  def : ReadAdvance<ReadVSTSX, 0>;
  defm  : LMULReadAdvance<"ReadVSTS8V", 0>;
  defm  : LMULReadAdvance<"ReadVSTS16V", 0>;
  defm  : LMULReadAdvance<"ReadVSTS32V", 0>;
  defm  : LMULReadAdvance<"ReadVSTS64V", 0>;
  defm  : LMULReadAdvance<"ReadVLDUXV", 0>;
  defm  : LMULReadAdvance<"ReadVLDOXV", 0>;
  defm  : LMULReadAdvance<"ReadVSTUX8", 0>;
  defm  : LMULReadAdvance<"ReadVSTUX16", 0>;
  defm  : LMULReadAdvance<"ReadVSTUX32", 0>;
  defm  : LMULReadAdvance<"ReadVSTUX64", 0>;
  defm  : LMULReadAdvance<"ReadVSTUXV", 0>;
  defm  : LMULReadAdvance<"ReadVSTUX8V", 0>;
  defm  : LMULReadAdvance<"ReadVSTUX16V", 0>;
  defm  : LMULReadAdvance<"ReadVSTUX32V", 0>;
  defm  : LMULReadAdvance<"ReadVSTUX64V", 0>;
  defm  : LMULReadAdvance<"ReadVSTOX8", 0>;
  defm  : LMULReadAdvance<"ReadVSTOX16", 0>;
  defm  : LMULReadAdvance<"ReadVSTOX32", 0>;
  defm  : LMULReadAdvance<"ReadVSTOX64", 0>;
  defm  : LMULReadAdvance<"ReadVSTOXV", 0>;
  defm  : LMULReadAdvance<"ReadVSTOX8V", 0>;
  defm  : LMULReadAdvance<"ReadVSTOX16V", 0>;
  defm  : LMULReadAdvance<"ReadVSTOX32V", 0>;
  defm  : LMULReadAdvance<"ReadVSTOX64V", 0>;
  // LMUL Aware
  def : ReadAdvance<ReadVST1R, 0>;
  def : ReadAdvance<ReadVST2R, 0>;
  def : ReadAdvance<ReadVST4R, 0>;
  def : ReadAdvance<ReadVST8R, 0>;

  // 12. Vector Integer Arithmetic Instructions
  defm : LMULReadAdvance<"ReadVIALUV", 0>;
  defm : LMULReadAdvance<"ReadVIALUX", 0>;
  defm : LMULReadAdvanceW<"ReadVIWALUV", 0>;
  defm : LMULReadAdvanceW<"ReadVIWALUX", 0>;
  defm : LMULReadAdvance<"ReadVExtV", 0>;
  defm : LMULReadAdvance<"ReadVICALUV", 0>;
  defm : LMULReadAdvance<"ReadVICALUX", 0>;
  defm : LMULReadAdvance<"ReadVShiftV", 0>;
  defm : LMULReadAdvance<"ReadVShiftX", 0>;
  defm : LMULReadAdvanceW<"ReadVNShiftV", 0>;
  defm : LMULReadAdvanceW<"ReadVNShiftX", 0>;
  defm : LMULReadAdvance<"ReadVICmpV", 0>;
  defm : LMULReadAdvance<"ReadVICmpX", 0>;
  defm : LMULReadAdvance<"ReadVIMinMaxV", 0>;
  defm : LMULReadAdvance<"ReadVIMinMaxX", 0>;
  defm : LMULReadAdvance<"ReadVIMulV", 0>;
  defm : LMULReadAdvance<"ReadVIMulX", 0>;
  defm : LMULSEWReadAdvance<"ReadVIDivV", 0>;
  defm : LMULSEWReadAdvance<"ReadVIDivX", 0>;
  defm : LMULReadAdvanceW<"ReadVIWMulV", 0>;
  defm : LMULReadAdvanceW<"ReadVIWMulX", 0>;
  defm : LMULReadAdvance<"ReadVIMulAddV", 0>;
  defm : LMULReadAdvance<"ReadVIMulAddX", 0>;
  defm : LMULReadAdvanceW<"ReadVIWMulAddV", 0>;
  defm : LMULReadAdvanceW<"ReadVIWMulAddX", 0>;
  defm : LMULReadAdvance<"ReadVIMergeV", 0>;
  defm : LMULReadAdvance<"ReadVIMergeX", 0>;
  defm : LMULReadAdvance<"ReadVIMovV", 0>;
  defm : LMULReadAdvance<"ReadVIMovX", 0>;

  // 13. Vector Fixed-Point Arithmetic Instructions
  defm  : LMULReadAdvance<"ReadVSALUV", 0>;
  defm  : LMULReadAdvance<"ReadVSALUX", 0>;
  defm  : LMULReadAdvance<"ReadVAALUV", 0>;
  defm  : LMULReadAdvance<"ReadVAALUX", 0>;
  defm  : LMULReadAdvance<"ReadVSMulV", 0>;
  defm  : LMULReadAdvance<"ReadVSMulX", 0>;
  defm  : LMULReadAdvance<"ReadVSShiftV", 0>;
  defm  : LMULReadAdvance<"ReadVSShiftX", 0>;
  defm  : LMULReadAdvanceW<"ReadVNClipV", 0>;
  defm  : LMULReadAdvanceW<"ReadVNClipX", 0>;

  // 14. Vector Floating-Point Instructions
  defm  : LMULSEWReadAdvanceF<"ReadVFALUV", 0>;
  defm  : LMULSEWReadAdvanceF<"ReadVFALUF", 0>;
  defm  : LMULSEWReadAdvanceFW<"ReadVFWALUV", 0>;
  defm  : LMULSEWReadAdvanceFW<"ReadVFWALUF", 0>;
  defm  : LMULSEWReadAdvanceF<"ReadVFMulV", 0>;
  defm  : LMULSEWReadAdvanceF<"ReadVFMulF", 0>;
  defm  : LMULSEWReadAdvanceF<"ReadVFDivV", 0>;
  defm  : LMULSEWReadAdvanceF<"ReadVFDivF", 0>;
  defm  : LMULSEWReadAdvanceFW<"ReadVFWMulV", 0>;
  defm  : LMULSEWReadAdvanceFW<"ReadVFWMulF", 0>;
  defm  : LMULSEWReadAdvanceF<"ReadVFMulAddV", 0>;
  defm  : LMULSEWReadAdvanceF<"ReadVFMulAddF", 0>;
  defm  : LMULSEWReadAdvanceFW<"ReadVFWMulAddV", 0>;
  defm  : LMULSEWReadAdvanceFW<"ReadVFWMulAddF", 0>;
  defm  : LMULSEWReadAdvanceF<"ReadVFSqrtV", 0>;
  defm  : LMULSEWReadAdvanceF<"ReadVFRecpV", 0>;
  defm  : LMULReadAdvance<"ReadVFCmpV", 0>;
  defm  : LMULReadAdvance<"ReadVFCmpF", 0>;
  defm  : LMULSEWReadAdvanceF<"ReadVFMinMaxV", 0>;
  defm  : LMULSEWReadAdvanceF<"ReadVFMinMaxF", 0>;
  defm  : LMULSEWReadAdvanceF<"ReadVFSgnjV", 0>;
  defm  : LMULSEWReadAdvanceF<"ReadVFSgnjF", 0>;
  defm  : LMULReadAdvance<"ReadVFClassV", 0>;
  defm  : LMULReadAdvance<"ReadVFMergeV", 0>;
  defm  : LMULReadAdvance<"ReadVFMergeF", 0>;
  defm  : LMULReadAdvance<"ReadVFMovF", 0>;
  defm  : LMULSEWReadAdvanceF<"ReadVFCvtIToFV", 0>;
  defm  : LMULReadAdvance<"ReadVFCvtFToIV", 0>;
  defm  : LMULSEWReadAdvanceW<"ReadVFWCvtIToFV", 0>;
  defm  : LMULReadAdvanceFW<"ReadVFWCvtFToIV", 0>;
  defm  : LMULSEWReadAdvanceFW<"ReadVFWCvtFToFV", 0>;
  defm  : LMULSEWReadAdvanceFW<"ReadVFNCvtIToFV", 0>;
  defm  : LMULReadAdvanceW<"ReadVFNCvtFToIV", 0>;
  defm  : LMULSEWReadAdvanceFW<"ReadVFNCvtFToFV", 0>;

  // 15. Vector Reduction Operations
  def : ReadAdvance<ReadVIRedV, 0>;
  def : ReadAdvance<ReadVIRedV0, 0>;
  def : ReadAdvance<ReadVIWRedV, 0>;
  def : ReadAdvance<ReadVIWRedV0, 0>;
  def : ReadAdvance<ReadVFRedV, 0>;
  def : ReadAdvance<ReadVFRedV0, 0>;
  def : ReadAdvance<ReadVFRedOV, 0>;
  def : ReadAdvance<ReadVFRedOV0, 0>;
  def : ReadAdvance<ReadVFWRedV, 0>;
  def : ReadAdvance<ReadVFWRedV0, 0>;
  def : ReadAdvance<ReadVFWRedOV, 0>;
  def : ReadAdvance<ReadVFWRedOV0, 0>;

  // 16. Vector Mask Instructions
  defm  : LMULReadAdvance<"ReadVMALUV", 0>;
  defm  : LMULReadAdvance<"ReadVMPopV", 0>;
  defm  : LMULReadAdvance<"ReadVMFFSV", 0>;
  defm  : LMULReadAdvance<"ReadVMSFSV", 0>;
  defm  : LMULReadAdvance<"ReadVIotaV", 0>;

  // 17. Vector Permutation Instructions
  def : ReadAdvance<ReadVMovXS, 0>;
  def : ReadAdvance<ReadVMovSX_V, 0>;
  def : ReadAdvance<ReadVMovSX_X, 0>;
  def : ReadAdvance<ReadVMovFS, 0>;
  def : ReadAdvance<ReadVMovSF_V, 0>;
  def : ReadAdvance<ReadVMovSF_F, 0>;
  defm  : LMULReadAdvance<"ReadVISlideV", 0>;
  defm  : LMULReadAdvance<"ReadVISlideX", 0>;
  defm  : LMULReadAdvance<"ReadVFSlideV", 0>;
  defm  : LMULReadAdvance<"ReadVFSlideF", 0>;
  defm  : LMULSEWReadAdvance<"ReadVRGatherVV_data", 0>;
  defm  : LMULSEWReadAdvance<"ReadVRGatherVV_index", 0>;
  defm  : LMULSEWReadAdvance<"ReadVRGatherEI16VV_data", 0>;
  defm  : LMULSEWReadAdvance<"ReadVRGatherEI16VV_index", 0>;
  defm  : LMULReadAdvance<"ReadVRGatherVX_data", 0>;
  defm  : LMULReadAdvance<"ReadVRGatherVX_index", 0>;
  defm  : LMULReadAdvance<"ReadVRGatherVI_data", 0>;
  defm  : LMULSEWReadAdvance<"ReadVCompressV", 0>;
  // LMUL Aware
  def : ReadAdvance<ReadVMov1V, 0>;
  def : ReadAdvance<ReadVMov2V, 0>;
  def : ReadAdvance<ReadVMov4V, 0>;
  def : ReadAdvance<ReadVMov8V, 0>;

  // Others
  def : ReadAdvance<ReadVMask, 0>;
  def : ReadAdvance<ReadVPassthru_WorstCase, 0>;
  foreach mx = SchedMxList in {
    def : ReadAdvance<!cast<SchedRead>("ReadVPassthru_" # mx), 0>;
    foreach sew = SchedSEWSet<mx>.val in
      def : ReadAdvance<!cast<SchedRead>("ReadVPassthru_" # mx  # "_E" # sew), 0>;
  }

  // Vector Crypto Extensions
  // Zvbb
  defm  : LMULReadAdvance<"ReadVBREVV", 0>;
  defm  : LMULReadAdvance<"ReadVCLZV", 0>;
  defm  : LMULReadAdvance<"ReadVCPOPV", 0>;
  defm  : LMULReadAdvance<"ReadVCTZV", 0>;
  defm  : LMULReadAdvance<"ReadVWSLLV", 0>;
  defm  : LMULReadAdvance<"ReadVWSLLX", 0>;
  // Zvbc
  defm  : LMULReadAdvance<"ReadVCLMULV", 0>;
  defm  : LMULReadAdvance<"ReadVCLMULX", 0>;
  // Zvkb
  // VANDN uses ReadVIALU[V|X|I]
  defm  : LMULReadAdvance<"ReadVBREV8V", 0>;
  defm  : LMULReadAdvance<"ReadVREV8V", 0>;
  defm  : LMULReadAdvance<"ReadVRotV", 0>;
  defm  : LMULReadAdvance<"ReadVRotX", 0>;
  // Zvkg
  defm  : LMULReadAdvance<"ReadVGHSHV", 0>;
  defm  : LMULReadAdvance<"ReadVGMULV", 0>;
  // Zvknha or Zvknhb
  defm  : LMULReadAdvance<"ReadVSHA2CHV", 0>;
  defm  : LMULReadAdvance<"ReadVSHA2CLV", 0>;
  defm  : LMULSEWReadAdvance<"ReadVSHA2MSV", 0>;
  // Zvkned
  defm  : LMULReadAdvance<"ReadVAESMVV", 0>;
  defm  : LMULReadAdvance<"ReadVAESKF1V", 0>;
  defm  : LMULReadAdvance<"ReadVAESKF2V", 0>;
  defm  : LMULReadAdvance<"ReadVAESZV", 0>;
  // Zvksed
  defm  : LMULReadAdvance<"ReadVSM4KV", 0>;
  defm  : LMULReadAdvance<"ReadVSM4RV", 0>;
  // Zbksh
  defm  : LMULReadAdvance<"ReadVSM3CV", 0>;
  defm  : LMULReadAdvance<"ReadVSM3MEV", 0>;
}
#endif
