//==- MipsScheduleP5600.td - P5600 Scheduling Definitions --*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def MipsP5600Model : SchedMachineModel {
  int IssueWidth = 2; // 2x dispatched per cycle
  int MicroOpBufferSize = 48; // min(48, 48, 64)
  int LoadLatency = 4;
  int MispredictPenalty = 8; // TODO: Estimated

  let CompleteModel = 0;
}

let SchedModel = MipsP5600Model in {

// ALQ Pipelines
// =============

def P5600ALQ : ProcResource<1> { let BufferSize = 16; }
def P5600IssueALU : ProcResource<1> { let Super = P5600ALQ; }

// ALU Pipeline
// ------------

def P5600WriteALU : SchedWriteRes<[P5600IssueALU]>;

// and, lui, nor, or, slti, sltiu, sub, subu, xor
def : ItinRW<[P5600WriteALU],
             [II_AND, II_LUI, II_NOR, II_OR, II_SLTI_SLTIU, II_SUBU, II_XOR]>;

// AGQ Pipelines
// =============

def P5600AGQ : ProcResource<3> { let BufferSize = 16; }
def P5600IssueAL2 : ProcResource<1> { let Super = P5600AGQ; }
def P5600IssueCTISTD : ProcResource<1> { let Super = P5600AGQ; }
def P5600IssueLDST : ProcResource<1> { let Super = P5600AGQ; }

def P5600AL2Div : ProcResource<1>;
// Pseudo-resource used to block CTISTD when handling multi-pipeline splits.
def P5600CTISTD : ProcResource<1>;

// CTISTD Pipeline
// ---------------

def P5600WriteJump : SchedWriteRes<[P5600IssueCTISTD, P5600CTISTD]>;
def P5600WriteJumpAndLink : SchedWriteRes<[P5600IssueCTISTD, P5600CTISTD]> {
  let Latency = 2;
}

// b, beq, beql, bg[et]z, bl[et]z, bne, bnel, j, syscall, jal, bltzal, jalx,
// jalr, jr.hb, jr
def : ItinRW<[P5600WriteJump], [II_B, II_BCC, II_BCCZ, II_BCCZAL, II_J, II_JR]>;
def : ItinRW<[P5600WriteJumpAndLink], [II_JAL, II_JALR]>;

// LDST Pipeline
// -------------

def P5600WriteLoad : SchedWriteRes<[P5600IssueLDST]> {
  let Latency = 4;
}

def P5600WriteLoadShifted : SchedWriteRes<[P5600IssueLDST, P5600CTISTD]> {
  let Latency = 4;
}

def P5600WritePref : SchedWriteRes<[P5600IssueLDST]>;

def P5600WriteStore : SchedWriteRes<[P5600IssueLDST, P5600CTISTD]> {
  // FIXME: This is a bit pessimistic. P5600CTISTD is only used during cycle 2
  //        not during 0, 1, and 2.
  let ResourceCycles = [ 1, 3 ];
}

def P5600WriteGPRFromBypass : SchedWriteRes<[P5600IssueLDST]> {
  let Latency = 2;
}

def P5600WriteStoreFromOtherUnits : SchedWriteRes<[P5600IssueLDST]>;
def P5600WriteLoadToOtherUnits : SchedWriteRes<[P5600IssueLDST]> {
  let Latency = 0;
}

// l[bhw], l[bh]u, ll
def : ItinRW<[P5600WriteLoad], [II_LB, II_LBU, II_LH, II_LHU, II_LW, II_LWU]>;

// lw[lr]
def : ItinRW<[P5600WriteLoadShifted], [II_LWL, II_LWR]>;

// s[bhw], sw[lr]
def : ItinRW<[P5600WriteStore], [II_SB, II_SH, II_SW, II_SWL, II_SWR]>;

// pref
// (this instruction does not exist in the backend yet)
def : ItinRW<[P5600WritePref], []>;

// sc
// (this instruction does not exist in the backend yet)
def : ItinRW<[P5600WriteStore], []>;

// LDST is also used in moves from general purpose registers to floating point
// and MSA.
def P5600WriteMoveGPRToOtherUnits : SchedWriteRes<[P5600IssueLDST]> {
  let Latency = 0;
}

// AL2 Pipeline
// ------------

def P5600WriteAL2 : SchedWriteRes<[P5600IssueAL2]>;
def P5600WriteAL2BitExt : SchedWriteRes<[P5600IssueAL2]> { let Latency = 2; }
def P5600WriteAL2ShadowMov : SchedWriteRes<[P5600IssueAL2]> { let Latency = 2; }
def P5600WriteAL2CondMov : SchedWriteRes<[P5600IssueAL2, P5600CTISTD]> {
  let Latency = 2;
}
def P5600WriteAL2Div : SchedWriteRes<[P5600IssueAL2, P5600AL2Div]> {
  // Estimated worst case
  let Latency = 34;
  let ResourceCycles = [1, 34];
}
def P5600WriteAL2DivU : SchedWriteRes<[P5600IssueAL2, P5600AL2Div]> {
  // Estimated worst case
  let Latency = 34;
  let ResourceCycles = [1, 34];
}
def P5600WriteAL2Mul : SchedWriteRes<[P5600IssueAL2]> { let Latency = 3; }
def P5600WriteAL2Mult: SchedWriteRes<[P5600IssueAL2]> { let Latency = 5; }
def P5600WriteAL2MAdd: SchedWriteRes<[P5600IssueAL2, P5600CTISTD]> {
  let Latency = 5;
}

// clo, clz, di, mfhi, mflo
def : ItinRW<[P5600WriteAL2], [II_CLO, II_CLZ, II_MFHI_MFLO]>;

// ehb, rdhwr, rdpgpr, wrpgpr, wsbh
def : ItinRW<[P5600WriteAL2ShadowMov], [II_RDHWR]>;

// mov[nz]
def : ItinRW<[P5600WriteAL2CondMov], [II_MOVN, II_MOVZ]>;

// divu?
def : ItinRW<[P5600WriteAL2Div], [II_DIV]>;
def : ItinRW<[P5600WriteAL2DivU], [II_DIVU]>;

// mul
def : ItinRW<[P5600WriteAL2Mul], [II_MUL]>;
// multu?, multu?
def : ItinRW<[P5600WriteAL2Mult], [II_MULT, II_MULTU]>;
// maddu?, msubu?, mthi, mtlo
def : ItinRW<[P5600WriteAL2MAdd],
             [II_MADD, II_MADDU, II_MSUB, II_MSUBU, II_MTHI_MTLO]>;

// ext, ins
def : ItinRW<[P5600WriteAL2BitExt],
             [II_EXT, II_INS]>;

// Either ALU or AL2 Pipelines
// ---------------------------
//
// Some instructions can choose between ALU and AL2, but once dispatched to
// ALQ or AGQ respectively they are committed to that path.
// The decision is based on the outcome of the most recent selection when the
// choice was last available. For now, we assume ALU is always chosen.

def P5600WriteEitherALU : SchedWriteVariant<
  // FIXME: Implement selection predicate
  [SchedVar<SchedPredicate<[{1}]>, [P5600WriteALU]>,
   SchedVar<SchedPredicate<[{0}]>, [P5600WriteAL2]>
  ]>;

// add, addi, addiu, addu, andi, ori, rotr, se[bh], sllv?, sr[al]v?, slt, sltu,
// xori
def : ItinRW<[P5600WriteEitherALU],
             [II_ADDI, II_ADDIU, II_ANDI, II_ORI, II_ROTR, II_SEB, II_SEH,
              II_SLT_SLTU, II_SLL, II_SRA, II_SRL, II_XORI, II_ADDU, II_SLLV,
              II_SRAV, II_SRLV]>;

// FPU Pipelines
// =============

def P5600FPQ : ProcResource<3> { let BufferSize = 16; }
def P5600IssueFPUS : ProcResource<1> { let Super = P5600FPQ; }
def P5600IssueFPUL : ProcResource<1> { let Super = P5600FPQ; }
def P5600IssueFPULoad : ProcResource<1> { let Super = P5600FPQ; }

def P5600FPUDivSqrt : ProcResource<2>;

def P5600WriteFPUS : SchedWriteRes<[P5600IssueFPUS]>;
def P5600WriteFPUL : SchedWriteRes<[P5600IssueFPUL]> { let Latency = 4; }
def P5600WriteFPUL_MADDSUB : SchedWriteRes<[P5600IssueFPUL]> { let Latency = 6; }
def P5600WriteFPUDivS : SchedWriteRes<[P5600IssueFPUL, P5600FPUDivSqrt]> {
  // Best/Common/Worst case = 7 / 23 / 27
  let Latency = 23; // Using common case
  let ResourceCycles = [ 1, 23 ];
}
def P5600WriteFPUDivD : SchedWriteRes<[P5600IssueFPUL, P5600FPUDivSqrt]> {
  // Best/Common/Worst case = 7 / 31 / 35
  let Latency = 31; // Using common case
  let ResourceCycles = [ 1, 31 ];
}
def P5600WriteFPURcpS : SchedWriteRes<[P5600IssueFPUL, P5600FPUDivSqrt]> {
  // Best/Common/Worst case = 7 / 19 / 23
  let Latency = 19; // Using common case
  let ResourceCycles = [ 1, 19 ];
}
def P5600WriteFPURcpD : SchedWriteRes<[P5600IssueFPUL, P5600FPUDivSqrt]> {
  // Best/Common/Worst case = 7 / 27 / 31
  let Latency = 27; // Using common case
  let ResourceCycles = [ 1, 27 ];
}
def P5600WriteFPURsqrtS : SchedWriteRes<[P5600IssueFPUL, P5600FPUDivSqrt]> {
  // Best/Common/Worst case = 7 / 27 / 27
  let Latency = 27; // Using common case
  let ResourceCycles = [ 1, 27 ];
}
def P5600WriteFPURsqrtD : SchedWriteRes<[P5600IssueFPUL, P5600FPUDivSqrt]> {
  // Best/Common/Worst case = 7 / 27 / 31
  let Latency = 27; // Using common case
  let ResourceCycles = [ 1, 27 ];
}
def P5600WriteFPUSqrtS : SchedWriteRes<[P5600IssueFPUL, P5600FPUDivSqrt]> {
  // Best/Common/Worst case = 7 / 27 / 31
  let Latency = 27; // Using common case
  let ResourceCycles = [ 1, 27 ];
}
def P5600WriteFPUSqrtD : SchedWriteRes<[P5600IssueFPUL, P5600FPUDivSqrt]> {
  // Best/Common/Worst case = 7 / 35 / 39
  let Latency = 35; // Using common case
  let ResourceCycles = [ 1, 35 ];
}
def P5600WriteMSAShortLogic : SchedWriteRes<[P5600IssueFPUS]>;
def P5600WriteMSAShortInt : SchedWriteRes<[P5600IssueFPUS]> { let Latency = 2; }
def P5600WriteMoveOtherUnitsToFPU : SchedWriteRes<[P5600IssueFPUS]>;

// FPUS is also used in moves from floating point and MSA registers to general
// purpose registers.
def P5600WriteMoveFPUSToOtherUnits : SchedWriteRes<[P5600IssueFPUS]> {
  let Latency = 0;
}

// FPUL is also used in moves from floating point and MSA registers to general
// purpose registers.
def P5600WriteMoveFPULToOtherUnits : SchedWriteRes<[P5600IssueFPUL]>;

// Short Pipe
// ----------
//
// abs.[ds], abs.ps, bc1[tf]l?, mov[tf].[ds], mov[tf], mov.[ds], [cm][ft]c1,
// m[ft]hc1, neg.[ds], neg.ps, nor.v, nori.b, or.v, ori.b, xor.v, xori.b,
// sdxc1, sdc1, st.[bhwd], swc1, swxc1
def : ItinRW<[P5600WriteFPUS], [II_ABS, II_MOVF_D, II_MOVF_S, II_MOVT_D,
                                II_MOVT_S, II_MOV_D, II_MOV_S, II_NEG]>;

// adds_a.[bhwd], adds_[asu].[bhwd], addvi?.[bhwd], asub_[us].[bhwd],
// aver?_[us].[bhwd]
def : InstRW<[P5600WriteMSAShortInt], (instregex "^ADD_A_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortInt], (instregex "^ADDS_[ASU]_[BHWD]$")>;
// TODO: ADDVI_[BHW] might be 1 cycle latency rather than 2. Need to confirm it.
def : InstRW<[P5600WriteMSAShortInt], (instregex "^ADDVI?_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortInt], (instregex "^ASUB_[US].[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortInt], (instregex "^AVER?_[US].[BHWD]$")>;

// and.v, andi.b, move.v, ldi.[bhwd]
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^MOVE_V$")>;
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^LDI_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^(AND|OR|[XN]OR)_V$")>;
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^(AND|OR|[XN]OR)I_B$")>;

// Long Pipe
// ----------
//
// add.[ds], add.ps, cvt.d.[sw], cvt.s.[dw], cvt.w.[sd], cvt.[sw].ps,
// cvt.ps.[sw], c.<cc>.[ds], c.<cc>.ps, mul.[ds], mul.ps, sub.[ds], sub.ps,
// trunc.w.[ds], trunc.w.ps
def : ItinRW<[P5600WriteFPUL],
             [II_ADD_D, II_ADD_S, II_CVT, II_C_CC_D, II_C_CC_S, II_MUL_D,
              II_MUL_S, II_SUB_D, II_SUB_S, II_TRUNC]>;

// div.[ds], div.ps
def : ItinRW<[P5600WriteFPUDivS], [II_DIV_S]>;
def : ItinRW<[P5600WriteFPUDivD], [II_DIV_D]>;

// sqrt.[ds], sqrt.ps
def : ItinRW<[P5600WriteFPUSqrtS], [II_SQRT_S]>;
def : ItinRW<[P5600WriteFPUSqrtD], [II_SQRT_D]>;

// madd.[ds], msub.[ds], nmadd.[ds], nmsub.[ds],
// Operand 0 is read on cycle 5. All other operands are read on operand 0.
def : ItinRW<[SchedReadAdvance<5>, P5600WriteFPUL_MADDSUB],
             [II_MADD_D, II_MADD_S, II_MSUB_D, II_MSUB_S, II_NMADD_D,
              II_NMADD_S, II_NMSUB_D, II_NMSUB_S]>;

// madd.ps, msub.ps, nmadd.ps, nmsub.ps
// Operand 0 and 1 are read on cycle 5. All others are read on operand 0.
// (none of these instructions exist in the backend yet)

// Load Pipe
// ---------
//
// This is typically used in conjunction with the load pipeline under the AGQ
// All the instructions are in the 'Tricky Instructions' section.

def P5600WriteLoadOtherUnitsToFPU : SchedWriteRes<[P5600IssueFPULoad]> {
  let Latency = 4;
}

// Tricky Instructions
// ===================
//
// These instructions are split across multiple uops (in different pipelines)
// that must cooperate to complete the operation

// FIXME: This isn't quite right since the implementation of WriteSequence
//        current aggregates the resources and ignores the exact cycle they are
//        used.
def P5600WriteMoveGPRToFPU : WriteSequence<[P5600WriteMoveGPRToOtherUnits,
                                            P5600WriteMoveOtherUnitsToFPU]>;

// FIXME: This isn't quite right since the implementation of WriteSequence
//        current aggregates the resources and ignores the exact cycle they are
//        used.
def P5600WriteMoveFPUToGPR : WriteSequence<[P5600WriteMoveFPUSToOtherUnits,
                                            P5600WriteGPRFromBypass]>;

// FIXME: This isn't quite right since the implementation of WriteSequence
//        current aggregates the resources and ignores the exact cycle they are
//        used.
def P5600WriteStoreFPUS : WriteSequence<[P5600WriteMoveFPUSToOtherUnits,
                                         P5600WriteStoreFromOtherUnits]>;

// FIXME: This isn't quite right since the implementation of WriteSequence
//        current aggregates the resources and ignores the exact cycle they are
//        used.
def P5600WriteStoreFPUL : WriteSequence<[P5600WriteMoveFPULToOtherUnits,
                                         P5600WriteStoreFromOtherUnits]>;

// FIXME: This isn't quite right since the implementation of WriteSequence
//        current aggregates the resources and ignores the exact cycle they are
//        used.
def P5600WriteLoadFPU : WriteSequence<[P5600WriteLoadToOtherUnits,
                                       P5600WriteLoadOtherUnitsToFPU]>;

// ctc1, mtc1, mthc1
def : ItinRW<[P5600WriteMoveGPRToFPU], [II_CTC1, II_MTC1, II_MTHC1]>;

// bc1[ft], cfc1, mfc1, mfhc1, movf, movt
def : ItinRW<[P5600WriteMoveFPUToGPR],
             [II_BC1F, II_BC1T, II_CFC1, II_MFC1, II_MFHC1, II_MOVF, II_MOVT]>;

// swc1, swxc1, st.[bhwd]
def : ItinRW<[P5600WriteStoreFPUS], [II_SWC1, II_SWXC1]>;
def : InstRW<[P5600WriteStoreFPUS], (instregex "^ST_[BHWD]$")>;

// movn.[ds], movz.[ds]
def : ItinRW<[P5600WriteStoreFPUL], [II_MOVN_D, II_MOVN_S, II_MOVZ_D, II_MOVZ_S]>;

// l[dw]x?c1, ld.[bhwd]
def : ItinRW<[P5600WriteLoadFPU], [II_LDC1, II_LDXC1, II_LWC1, II_LWXC1]>;
def : InstRW<[P5600WriteLoadFPU], (instregex "LD_[BHWD]")>;

// Unsupported Instructions
// ========================
//
// The following instruction classes are never valid on P5600.
//   II_DADDIU, II_DADDU, II_DMFC1, II_DMTC1, II_DMULT, II_DMULTU, II_DROTR,
//   II_DROTR32, II_DROTRV, II_DDIV, II_DSLL, II_DSLL32, II_DSLLV, II_DSRA,
//   II_DSRA32, II_DSRAV, II_DSRL, II_DSRL32, II_DSRLV, II_DSUBU, II_DDIVU,
//   II_JALRC, II_LD, II_LD[LR], II_LUXC1, II_RESTORE, II_SAVE, II_SD, II_SDC1,
//   II_SDL, II_SDR, II_SDXC1
//
// The following instructions are never valid on P5600.
//   addq.ph, rdhwr, repl.ph, repl.qb, subq.ph, subu_s.qb
//
// Guesswork
// =========
//
// This section is largely temporary guesswork.

// ceil.[lw].[ds], floor.[lw].[ds]
// Reason behind guess: trunc.[lw].ds and the various cvt's are in FPUL
def : ItinRW<[P5600WriteFPUL], [II_CEIL, II_FLOOR, II_ROUND]>;

// rotrv
// Reason behind guess: rotr is in the same category and the two register forms
//                      generally follow the immediate forms in this category
def : ItinRW<[P5600WriteEitherALU], [II_ROTRV]>;
}
