//===-- SparcSchedLeon.td - LEON scheduling ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the itineraries of the LEON processors.
//===----------------------------------------------------------------------===//


def LEONIU : FuncUnit; // integer unit
def LEONFPU : FuncUnit; // floating-point unit

// Ref: http://www.atmel.com/Images/doc4226.pdf

def LEON2Itineraries : ProcessorItineraries<
[LEONIU, LEONFPU], [], [
  InstrItinData<IIC_iu_or_fpu_instr, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_iu_instr, [InstrStage<1, [LEONIU]>], [1, 1]>,
  InstrItinData<IIC_ld, [InstrStage<1, [LEONIU]>], [1, 1]>,
  InstrItinData<IIC_fpu_ld, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_fpu_ldd, [InstrStage<1, [LEONIU, LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_normal_instr, [InstrStage<1, [LEONFPU]>], [7, 1]>,
  InstrItinData<IIC_fpu_fast_instr, [InstrStage<1, [LEONFPU]>], [7, 1]>,
  InstrItinData<IIC_jmp, [InstrStage<1, [LEONIU, LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_call, [InstrStage<1, [LEONIU, LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_ldd, [InstrStage<1, [LEONIU, LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_st, [InstrStage<1, [LEONIU, LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_std, [InstrStage<1, [LEONIU, LEONFPU]>], [3, 1]>,
  InstrItinData<IIC_iu_smul, [InstrStage<1, [LEONIU]>], [5, 1]>,
  InstrItinData<IIC_iu_umul, [InstrStage<1, [LEONIU]>], [5, 1]>,
  InstrItinData<IIC_iu_div, [InstrStage<1, [LEONIU]>], [35, 1]>,
  InstrItinData<IIC_ticc, [InstrStage<1, [LEONIU, LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_atomic, [InstrStage<1, [LEONIU, LEONFPU]>], [3, 1]>,
  InstrItinData<IIC_fpu_muls, [InstrStage<1, [LEONFPU]>], [16, 1]>,
  InstrItinData<IIC_fpu_muld, [InstrStage<1, [LEONFPU]>], [21, 1]>,
  InstrItinData<IIC_fpu_divs, [InstrStage<1, [LEONFPU]>], [20, 1]>,
  InstrItinData<IIC_fpu_divd, [InstrStage<1, [LEONFPU]>], [36, 1]>,
  InstrItinData<IIC_fpu_sqrts, [InstrStage<1, [LEONFPU]>], [37, 1]>,
  InstrItinData<IIC_fpu_sqrtd, [InstrStage<1, [LEONFPU]>], [65, 1]>,
  InstrItinData<IIC_fpu_abs, [InstrStage<1, [LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_movs, [InstrStage<1, [LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_negs, [InstrStage<1, [LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_stod, [InstrStage<1, [LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_cmp, [InstrStage<1, [LEONFPU]>], [7, 1]>,
  InstrItinData<IIC_branch, [InstrStage<1, [LEONIU]>], [1, 1]>,
  InstrItinData<IIC_fp_branch, [InstrStage<1, [LEONFPU]>], [7, 1]>,
  InstrItinData<IIC_fpu_st, [InstrStage<1, [LEONIU, LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_std, [InstrStage<1, [LEONIU, LEONFPU]>], [3, 1]>
]>;

def LEON3Itineraries : ProcessorItineraries<
[LEONIU, LEONFPU], [], [
  InstrItinData<IIC_iu_or_fpu_instr, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_iu_instr, [InstrStage<1, [LEONIU]>], [1, 1]>,
  InstrItinData<IIC_ld, [InstrStage<1, [LEONIU]>], [1, 1]>,
  InstrItinData<IIC_fpu_ld, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_fpu_ldd, [InstrStage<1, [LEONIU, LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_normal_instr, [InstrStage<1, [LEONFPU]>], [7, 1]>,
  InstrItinData<IIC_fpu_fast_instr, [InstrStage<1, [LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_jmp, [InstrStage<1, [LEONIU, LEONFPU]>], [3, 1]>,
  InstrItinData<IIC_call, [InstrStage<1, [LEONIU, LEONFPU]>], [3, 1]>,
  InstrItinData<IIC_ldd, [InstrStage<1, [LEONIU, LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_st, [InstrStage<1, [LEONIU, LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_std, [InstrStage<1, [LEONIU, LEONFPU]>], [5, 1]>,
  InstrItinData<IIC_iu_smul, [InstrStage<1, [LEONIU]>], [1, 1]>,
  InstrItinData<IIC_iu_umul, [InstrStage<1, [LEONIU]>], [4, 1]>,
  InstrItinData<IIC_iu_div, [InstrStage<1, [LEONIU]>], [35, 1]>,
  InstrItinData<IIC_smac_umac, [InstrStage<1, [LEONIU]>], [2, 1]>,
  InstrItinData<IIC_ticc, [InstrStage<1, [LEONIU, LEONFPU]>], [5, 1]>,
  InstrItinData<IIC_atomic, [InstrStage<1, [LEONIU, LEONFPU]>], [3, 1]>,
  InstrItinData<IIC_fpu_muls, [InstrStage<1, [LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_fpu_muld, [InstrStage<1, [LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_fpu_divs, [InstrStage<1, [LEONFPU]>], [16, 1]>,
  InstrItinData<IIC_fpu_divd, [InstrStage<1, [LEONFPU]>], [17, 1]>,
  InstrItinData<IIC_fpu_sqrts, [InstrStage<1, [LEONFPU]>], [24, 1]>,
  InstrItinData<IIC_fpu_sqrtd, [InstrStage<1, [LEONFPU]>], [25, 1]>,
  InstrItinData<IIC_fpu_abs, [InstrStage<1, [LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_movs, [InstrStage<1, [LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_negs, [InstrStage<1, [LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_stod, [InstrStage<1, [LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_fpu_cmp, [InstrStage<1, [LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_branch, [InstrStage<1, [LEONIU]>], [1, 1]>,
  InstrItinData<IIC_fp_branch, [InstrStage<1, [LEONFPU]>], [7, 1]>,
  InstrItinData<IIC_fpu_st, [InstrStage<1, [LEONIU, LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_fpu_std, [InstrStage<1, [LEONIU, LEONFPU]>], [5, 1]>
]>;

def LEON4Itineraries : ProcessorItineraries<
[LEONIU, LEONFPU], [], [
  InstrItinData<IIC_iu_or_fpu_instr, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_iu_instr, [InstrStage<1, [LEONIU]>], [1, 1]>,
  InstrItinData<IIC_ld, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_fpu_ld, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_fpu_ldd, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_fpu_normal_instr, [InstrStage<1, [LEONFPU]>], [7, 1]>,
  InstrItinData<IIC_fpu_fast_instr, [InstrStage<1, [LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_jmp, [InstrStage<1, [LEONIU, LEONFPU]>], [3, 1]>,
  InstrItinData<IIC_call, [InstrStage<1, [LEONIU, LEONFPU]>], [3, 1]>,
  InstrItinData<IIC_ldd, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_st, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_std, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_iu_smul, [InstrStage<1, [LEONIU]>], [1, 1]>,
  InstrItinData<IIC_iu_umul, [InstrStage<1, [LEONIU]>], [4, 1]>,
  InstrItinData<IIC_iu_div, [InstrStage<1, [LEONIU]>], [35, 1]>,
  InstrItinData<IIC_smac_umac, [InstrStage<1, [LEONIU]>], [2, 1]>,
  InstrItinData<IIC_ticc, [InstrStage<1, [LEONIU, LEONFPU]>], [5, 1]>,
  InstrItinData<IIC_atomic, [InstrStage<1, [LEONIU, LEONFPU]>], [3, 1]>,
  InstrItinData<IIC_fpu_muls, [InstrStage<1, [LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_fpu_muld, [InstrStage<1, [LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_fpu_divs, [InstrStage<1, [LEONFPU]>], [16, 1]>,
  InstrItinData<IIC_fpu_divd, [InstrStage<1, [LEONFPU]>], [17, 1]>,
  InstrItinData<IIC_fpu_sqrts, [InstrStage<1, [LEONFPU]>], [24, 1]>,
  InstrItinData<IIC_fpu_sqrtd, [InstrStage<1, [LEONFPU]>], [25, 1]>,
  InstrItinData<IIC_fpu_abs, [InstrStage<1, [LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_movs, [InstrStage<1, [LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_negs, [InstrStage<1, [LEONFPU]>], [2, 1]>,
  InstrItinData<IIC_fpu_stod, [InstrStage<1, [LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_fpu_cmp, [InstrStage<1, [LEONFPU]>], [4, 1]>,
  InstrItinData<IIC_branch, [InstrStage<1, [LEONIU]>], [1, 1]>,
  InstrItinData<IIC_fp_branch, [InstrStage<1, [LEONFPU]>], [7, 1]>,
  InstrItinData<IIC_fpu_st, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>,
  InstrItinData<IIC_fpu_std, [InstrStage<1, [LEONIU, LEONFPU]>], [1, 1]>
]>;

// LEON5. Timings are from the GRLIB IP Core User's Manual
// (https://download.gaisler.com/products/GRLIB/doc/grip.pdf) and from
// measurements.
def LEON5Model : SchedMachineModel {
  let IssueWidth = 2;
  let MicroOpBufferSize = 0; // In-order.
  let LoadLatency = 2;
  let MispredictPenalty = 6;
  let CompleteModel = 0;
}

def WriteLeon5ALU       : SchedWrite;
def WriteLeon5ALUcc     : SchedWrite;
def WriteLeon5Load      : SchedWrite;
def WriteLeon5FPLoad    : SchedWrite;
def WriteLeon5Store     : SchedWrite;
def WriteLeon5Mul       : SchedWrite;
def WriteLeon5MulY      : SchedWrite;
def WriteLeon5MulCC     : SchedWrite;
def WriteLeon5MulSCC    : SchedWrite;
def WriteLeon5Div       : SchedWrite;
def WriteLeon5DivExtra  : SchedWrite;
def WriteLeon5RdSpecial : SchedWrite;
def WriteLeon5WrSpecial : SchedWrite;
def WriteLeon5Branch    : SchedWrite;
def WriteLeon5Atomic    : SchedWrite;
def WriteLeon5FP        : SchedWrite;
def WriteLeon5FPCmp     : SchedWrite;
def WriteLeon5FPDivS    : SchedWrite;
def WriteLeon5FPDivD    : SchedWrite;
def WriteLeon5FPSqrtS   : SchedWrite;
def WriteLeon5FPSqrtD   : SchedWrite;
def WriteLeon5FPStore   : SchedWrite;
def ReadLeon5LateALU    : SchedRead;
def ReadLeon5StoreData  : SchedRead;
def ReadLeon5MulOp      : SchedRead;
def ReadLeon5FPOp       : SchedRead;

def Leon5ALU     : ProcResource<2> { let BufferSize = 0; }
def Leon5Mem     : ProcResource<1> { let BufferSize = 0; }
def Leon5Mul     : ProcResource<1> { let BufferSize = 0; }
def Leon5FPALU   : ProcResource<1> { let BufferSize = 0; }
def Leon5FPDS    : ProcResource<1> { let BufferSize = 0; }
def Leon5Lane1   : ProcResource<1> { let BufferSize = 0; }

let SchedModel = LEON5Model in {
  def : WriteRes<WriteLeon5ALU, [Leon5ALU]> { let Latency = 1; }
  def : WriteRes<WriteLeon5ALUcc, [Leon5ALU, Leon5Lane1]> { let Latency = 1; }
  def : WriteRes<WriteLeon5Load, [Leon5Mem]> { let Latency = 2; }
  def : WriteRes<WriteLeon5FPLoad, [Leon5Mem, Leon5FPALU]> {
    let Latency = 6;
  }
  def : WriteRes<WriteLeon5Store, [Leon5Mem]> {
    let Latency = 1;
    let ReleaseAtCycles = [2];
  }
  def : WriteRes<WriteLeon5Mul, [Leon5Mul]> { let Latency = 2; }
  // RD %Y takes the high word directly from the multiplier.
  def : WriteRes<WriteLeon5MulY, []> {
    let Latency = 1;
    let NumMicroOps = 0;
  }
  // UMULCC/SMULCC set the condition codes late and need the second lane.
  def : WriteRes<WriteLeon5MulCC, [Leon5Lane1]> {
    let Latency = 3;
    let NumMicroOps = 0;
  }
  // MULSCC never issues alongside another instruction.
  def : WriteRes<WriteLeon5MulSCC, [Leon5ALU]> {
    let Latency = 1;
    let SingleIssue = true;
  }
  // A divide holds the pipeline until it completes.
  def : WriteRes<WriteLeon5Div, [Leon5Mul]> {
    let Latency = 35;
    let NumMicroOps = 2;
    let ReleaseAtCycles = [35];
  }
  // The Y and ICC results of a divide.
  def : WriteRes<WriteLeon5DivExtra, []> {
    let Latency = 35;
    let NumMicroOps = 0;
  }
  // Reads and writes of special registers issue alone.
  def : WriteRes<WriteLeon5RdSpecial, [Leon5ALU]> {
    let Latency = 1;
    let SingleIssue = true;
  }
  def : WriteRes<WriteLeon5WrSpecial, [Leon5ALU]> {
    let Latency = 3;
    let SingleIssue = true;
  }
  def : WriteRes<WriteLeon5Branch, [Leon5ALU]> { let Latency = 1; }
  def : WriteRes<WriteLeon5Atomic, [Leon5Mem]> {
    let Latency = 5;
    let NumMicroOps = 2;
    let ReleaseAtCycles = [5];
  }

  def : WriteRes<WriteLeon5FP, [Leon5FPALU]> { let Latency = 7; }
  def : WriteRes<WriteLeon5FPCmp, [Leon5FPALU]> { let Latency = 3; }
  // Divide and square root share a unit. The latencies are the worst case.
  def : WriteRes<WriteLeon5FPDivS, [Leon5FPALU, Leon5FPDS]> {
    let Latency = 19;
    let ReleaseAtCycles = [1, 15];
  }
  def : WriteRes<WriteLeon5FPDivD, [Leon5FPALU, Leon5FPDS]> {
    let Latency = 22;
    let ReleaseAtCycles = [1, 18];
  }
  def : WriteRes<WriteLeon5FPSqrtS, [Leon5FPALU, Leon5FPDS]> {
    let Latency = 27;
    let ReleaseAtCycles = [1, 23];
  }
  def : WriteRes<WriteLeon5FPSqrtD, [Leon5FPALU, Leon5FPDS]> {
    let Latency = 30;
    let ReleaseAtCycles = [1, 26];
  }
  def : WriteRes<WriteLeon5FPStore, [Leon5Mem, Leon5FPALU]> {
    let Latency = 1;
    let ReleaseAtCycles = [2, 1];
  }

  // The late ALU and store data take a load or multiply result one cycle after
  // the producer issues.
  def : ReadAdvance<ReadLeon5LateALU, 1, [WriteLeon5Load, WriteLeon5Mul]>;
  def : ReadAdvance<ReadLeon5StoreData, 1, [WriteLeon5Load, WriteLeon5Mul]>;
  // A multiply result is not forwarded to another multiply or a divide.
  def : ReadAdvance<ReadLeon5MulOp, -1, [WriteLeon5Mul]>;
  // FP operations take forwarded results.
  def : ReadAdvance<ReadLeon5FPOp, 3,
                    [WriteLeon5FP, WriteLeon5FPLoad, WriteLeon5FPDivS,
                     WriteLeon5FPDivD, WriteLeon5FPSqrtS, WriteLeon5FPSqrtD],
                    [0, -1]>;

  // Instructions that can execute in the late ALU.
  def : InstRW<[WriteLeon5ALU, ReadLeon5LateALU, ReadLeon5LateALU],
               (instrs ADDrr, SUBrr, ANDrr, ANDNrr, ORrr, ORNrr, XORrr, XNORrr,
                       SLLrr, SRLrr, SRArr)>;
  def : InstRW<[WriteLeon5ALU, ReadLeon5LateALU],
               (instrs ADDri, SUBri, ANDri, ANDNri, ORri, ORNri, XORri, XNORri,
                       SLLri, SRLri, SRAri)>;
  def : InstRW<[WriteLeon5ALUcc, ReadLeon5LateALU, ReadLeon5LateALU],
               (instrs ADDCCrr, SUBCCrr, ANDCCrr, ANDNCCrr, ORCCrr, ORNCCrr,
                       XORCCrr, XNORCCrr)>;
  def : InstRW<[WriteLeon5ALUcc, ReadLeon5LateALU],
               (instrs ADDCCri, SUBCCri, ANDCCri, ANDNCCri, ORCCri, ORNCCri,
                       XORCCri, XNORCCri)>;
  // Other instructions that set the condition codes.
  def : InstRW<[WriteLeon5ALUcc],
               (instrs ADDErr, ADDEri, SUBErr, SUBEri, TADDCCrr, TADDCCri,
                       TSUBCCrr, TSUBCCri, TADDCCTVrr, TADDCCTVri, TSUBCCTVrr,
                       TSUBCCTVri)>;
  def : InstRW<[WriteLeon5ALU], (instrs SETHIi)>;
  def : InstRW<[WriteLeon5MulSCC], (instrs MULSCCrr, MULSCCri)>;

  def : InstRW<[WriteLeon5Mul, WriteLeon5MulY, ReadLeon5MulOp, ReadLeon5MulOp],
               (instrs UMULrr, SMULrr)>;
  def : InstRW<[WriteLeon5Mul, WriteLeon5MulY, ReadLeon5MulOp],
               (instrs UMULri, SMULri)>;
  def : InstRW<[WriteLeon5Mul, WriteLeon5MulY, WriteLeon5MulCC, ReadLeon5MulOp,
                ReadLeon5MulOp], (instrs UMULCCrr, SMULCCrr)>;
  def : InstRW<[WriteLeon5Mul, WriteLeon5MulY, WriteLeon5MulCC, ReadLeon5MulOp],
               (instrs UMULCCri, SMULCCri)>;
  def : InstRW<[WriteLeon5Div, WriteLeon5DivExtra, ReadLeon5MulOp,
                ReadLeon5MulOp], (instrs UDIVrr, SDIVrr)>;
  def : InstRW<[WriteLeon5Div, WriteLeon5DivExtra, ReadLeon5MulOp],
               (instrs UDIVri, SDIVri)>;
  def : InstRW<[WriteLeon5Div, WriteLeon5DivExtra, WriteLeon5DivExtra,
                ReadLeon5MulOp, ReadLeon5MulOp], (instrs UDIVCCrr, SDIVCCrr)>;
  def : InstRW<[WriteLeon5Div, WriteLeon5DivExtra, WriteLeon5DivExtra,
                ReadLeon5MulOp], (instrs UDIVCCri, SDIVCCri)>;

  // Loads with an explicit ASI.
  def : InstRW<[WriteLeon5Load],
               (instrs LDSBArr, LDSHArr, LDUBArr, LDUHArr, LDArr, LDDArr)>;
  def : InstRW<[WriteLeon5FPLoad], (instrs LDFArr, LDDFArr)>;
  def : InstRW<[WriteLeon5RdSpecial], (instrs RDASR, RDPSR, RDWIM, RDTBR)>;
  def : InstRW<[WriteLeon5WrSpecial],
               (instrs WRASRrr, WRASRri, WRPSRrr, WRPSRri, WRWIMrr, WRWIMri,
                       WRTBRrr, WRTBRri)>;

  def : InstRW<[WriteLeon5Store, ReadDefault, ReadDefault, ReadLeon5StoreData],
               (instrs STBrr, STHrr, STrr, STDrr)>;
  def : InstRW<[WriteLeon5Store, ReadDefault, ReadLeon5StoreData],
               (instrs STBri, STHri, STri, STDri)>;

  // FP operations and their FP source operands.
  def : InstRW<[WriteLeon5FP, ReadLeon5FPOp, ReadLeon5FPOp],
               (instrs FADDS, FADDD, FSUBS, FSUBD, FMULS, FMULD, FSMULD)>;
  def : InstRW<[WriteLeon5FP, ReadLeon5FPOp],
               (instrs FMOVS, FNEGS, FABSS, FITOS, FITOD, FSTOI, FDTOI, FSTOD,
                       FDTOS)>;
  def : InstRW<[WriteLeon5FPCmp, ReadLeon5FPOp, ReadLeon5FPOp],
               (instrs FCMPS, FCMPD)>;
  def : InstRW<[WriteLeon5FPDivS, ReadLeon5FPOp, ReadLeon5FPOp],
               (instrs FDIVS)>;
  def : InstRW<[WriteLeon5FPDivD, ReadLeon5FPOp, ReadLeon5FPOp],
               (instrs FDIVD)>;
  def : InstRW<[WriteLeon5FPSqrtS, ReadLeon5FPOp], (instrs FSQRTS)>;
  def : InstRW<[WriteLeon5FPSqrtD, ReadLeon5FPOp], (instrs FSQRTD)>;

  def : ItinRW<[WriteLeon5ALU],
               [IIC_iu_or_fpu_instr, IIC_iu_instr]>;
  def : ItinRW<[WriteLeon5Load], [IIC_ld, IIC_ldd]>;
  def : ItinRW<[WriteLeon5FPLoad], [IIC_fpu_ld, IIC_fpu_ldd]>;
  def : ItinRW<[WriteLeon5Store], [IIC_st, IIC_std]>;
  def : ItinRW<[WriteLeon5Mul], [IIC_iu_smul, IIC_iu_umul]>;
  def : ItinRW<[WriteLeon5Div], [IIC_iu_div]>;
  def : ItinRW<[WriteLeon5Branch], [IIC_branch, IIC_fp_branch]>;
  def : ItinRW<[WriteLeon5Atomic], [IIC_atomic]>;
  def : ItinRW<[WriteLeon5FP],
               [IIC_fpu_normal_instr, IIC_fpu_fast_instr, IIC_fpu_muls,
                IIC_fpu_muld, IIC_fpu_abs, IIC_fpu_movs, IIC_fpu_negs,
                IIC_fpu_stod]>;
  def : ItinRW<[WriteLeon5FPCmp], [IIC_fpu_cmp]>;
  def : ItinRW<[WriteLeon5FPDivS], [IIC_fpu_divs]>;
  def : ItinRW<[WriteLeon5FPDivD], [IIC_fpu_divd]>;
  def : ItinRW<[WriteLeon5FPSqrtS], [IIC_fpu_sqrts]>;
  def : ItinRW<[WriteLeon5FPSqrtD], [IIC_fpu_sqrtd]>;
  def : ItinRW<[WriteLeon5FPStore], [IIC_fpu_st, IIC_fpu_std]>;
}
