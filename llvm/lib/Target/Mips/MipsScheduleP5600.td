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

  list<Predicate> UnsupportedFeatures = [HasMips32r6, HasMips64r6,
                                         HasMips64, HasMips64r2, HasCnMips,
                                         InMicroMips, InMips16Mode,
                                         HasDSP, HasDSPR2, HasMT];
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
             [II_AND, II_LUI, II_NOR, II_OR, II_SLTI_SLTIU, II_SUB, II_SUBU,
              II_XOR]>;

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

// b, beq, beql, bg[et]z, bl[et]z, bne, bnel, j, syscall, jal, bltzal,
// jalr, jr.hb, jr
def : ItinRW<[P5600WriteJump], [II_B, II_BCC, II_BCCZ, II_BCCZAL, II_J, II_JR,
                                II_JR_HB, II_DERET, II_ERET, II_ERETNC, 
                                II_SYSCALL, II_BREAK, II_SDBBP, II_SSNOP,
                                II_TEQ, II_TEQI, II_TGE, II_TGEI, II_TGEIU,
                                II_TGEU, II_TLT, II_TLTI, II_TLTU, II_TNE,
                                II_TNEI, II_TRAP, II_TTLTIU, II_WAIT,
                                II_PAUSE]>;

def : ItinRW<[P5600WriteJumpAndLink], [II_JAL, II_JALR, II_JALR_HB]>;

def P5600COP0 : SchedWriteRes<[P5600IssueCTISTD, P5600CTISTD]>;

def : ItinRW<[P5600COP0], [II_TLBINV, II_TLBINVF, II_TLBP, II_TLBR, II_TLBWI,
                           II_TLBWR, II_MFC0, II_MTC0]>;
// LDST Pipeline
// -------------

def P5600WriteLoad : SchedWriteRes<[P5600IssueLDST]> {
  let Latency = 4;
}

def P5600WriteLoadShifted : SchedWriteRes<[P5600IssueLDST, P5600CTISTD]> {
  let Latency = 4;
}

def P5600WriteCache : SchedWriteRes<[P5600IssueLDST]>;

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
def : ItinRW<[P5600WriteLoad], [II_LB, II_LBE, II_LBU, II_LBUE, II_LH, II_LHE,
                                II_LHU, II_LHUE, II_LW, II_LWE, II_LL, II_LLE,
                                II_LWPC]>;

// lw[lr]
def : ItinRW<[P5600WriteLoadShifted], [II_LWL, II_LWLE, II_LWR, II_LWRE]>;

// s[bhw], sw[lr]
def : ItinRW<[P5600WriteStore], [II_SB, II_SBE, II_SH, II_SHE, II_SW, II_SWE,
                                 II_SWL, II_SWLE, II_SWR, II_SWRE, II_SC,
                                 II_SCE]>;

// pref, cache, sync, synci
def : ItinRW<[P5600WriteCache], [II_PREF, II_PREFE, II_CACHE, II_CACHEE,
                                 II_SYNC, II_SYNCI]>;

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

// clo, clz, di, ei, mfhi, mflo
def : ItinRW<[P5600WriteAL2], [II_CLO, II_CLZ, II_DI, II_EI, II_MFHI_MFLO]>;

// ehb, rdhwr, rdpgpr, wrpgpr, wsbh
def : ItinRW<[P5600WriteAL2ShadowMov], [II_EHB, II_RDHWR, II_WSBH]>;

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
def : ItinRW<[P5600WriteAL2BitExt], [II_EXT, II_INS]>;

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
             [II_ADD, II_ADDI, II_ADDIU, II_ANDI, II_ORI, II_ROTR, II_SEB, II_SEH,
              II_SLT_SLTU, II_SLL, II_SRA, II_SRL, II_XORI, II_ADDU, II_SLLV,
              II_SRAV, II_SRLV, II_LSA]>;
def : InstRW<[], (instrs COPY)>;

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
def P5600WriteFPUDivI : SchedWriteRes<[P5600IssueFPUL, P5600FPUDivSqrt]> {
  // Best/Common/Worst case = 7 / 23 / 27
  let Latency = 23; // Using common case
  let ResourceCycles = [ 1, 23 ];
}
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
def P5600WriteMSAOther3 : SchedWriteRes<[P5600IssueFPUS]> { let Latency = 3; }
def P5600WriteMSALongInt : SchedWriteRes<[P5600IssueFPUS]> { let Latency = 5; }

// vshf.[bhwd], binsl.[bhwd], binsr.[bhwd], insert.[bhwd], sld?.[bhwd],
// bset.[bhwd], bclr.[bhwd], bneg.[bhwd], bsel_v, bseli_b
def : InstRW<[P5600WriteMSAShortInt], (instregex "^VSHF_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortInt], (instregex "^(BINSL|BINSLI)_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortInt], (instregex "^(BINSR|BINSRI)_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortInt], (instregex "^INSERT_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortInt], (instregex "^(SLD|SLDI)_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortInt], (instregex "^(BSET|BSETI)_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortInt], (instregex "^(BCLR|BCLRI)_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortInt], (instregex "^(BNEG|BNEGI)_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortInt], (instregex "^(BSEL_V|BSELI_B)$")>;
def : InstRW<[P5600WriteMSAShortInt], (instregex "^BMN*Z.*$")>;

// pcnt.[bhwd], sat_s.[bhwd], sat_u.bhwd]
def : InstRW<[P5600WriteMSAOther3], (instregex "^PCNT_[BHWD]$")>;
def : InstRW<[P5600WriteMSAOther3], (instregex "^SAT_(S|U)_[BHWD]$")>;

// bnz.[bhwdv], cfcmsa, ctcmsa
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^(BNZ|BZ)_[BHWDV]$")>;
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^C(F|T)CMSA$")>;

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
// aver?_[us].[bhwd], shf.[bhw], fill[bhwd], splat?.[bhwd]
def : InstRW<[P5600WriteMSAShortInt], (instregex "^ADD_A_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortInt], (instregex "^ADDS_[ASU]_[BHWD]$")>;
// TODO: ADDVI_[BHW] might be 1 cycle latency rather than 2. Need to confirm it.
def : InstRW<[P5600WriteMSAShortInt], (instregex "^ADDVI?_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortInt], (instregex "^ASUB_[US].[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortInt], (instregex "^AVER?_[US].[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortInt], (instregex "^SHF_[BHW]$")>;
def : InstRW<[P5600WriteMSAShortInt], (instregex "^FILL_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortInt], (instregex "^(SPLAT|SPLATI)_[BHWD]$")>;

// and.v, andi.b, move.v, ldi.[bhwd]
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^MOVE_V$")>;
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^LDI_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^(AND|OR|[XN]OR)_V$")>;
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^(AND|OR|[XN]OR)I_B$")>;

// fexp2_w, fexp2_d
def : InstRW<[P5600WriteFPUS], (instregex "^FEXP2_(W|D)$")>;

// compare, converts, round to int, floating point truncate.
def : InstRW<[P5600WriteFPUS], (instregex "^(CLT|CLTI)_(S|U)_[BHWD]$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^(CLE|CLEI)_(S|U)_[BHWD]$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^(CEQ|CEQI)_[BHWD]$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^CMP_UN_(S|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^CMP_UEQ_(S|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^CMP_EQ_(S|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^CMP_LT_(S|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^CMP_ULT_(S|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^CMP_LE_(S|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^CMP_ULE_(S|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FS(AF|EQ|LT|LE|NE|OR)_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FSUEQ_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FSULE_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FSULT_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FSUNE_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FSUN_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FCAF_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FCEQ_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FCLE_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FCLT_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FCNE_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FCOR_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FCUEQ_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FCULE_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FCULT_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FCUNE_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FCUN_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FABS_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FFINT_(U|S)_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FFQL_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FFQR_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FTINT_(U|S)_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FRINT_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FTQ_(H|W)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FTRUNC_(U|S)_(W|D)$")>;

// fexdo.[hw], fexupl.[wd], fexupr.[wd]
def : InstRW<[P5600WriteFPUS], (instregex "^FEXDO_(H|W)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FEXUPL_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FEXUPR_(W|D)$")>;

// fclass.[wd], fmax.[wd], fmax_a.[wd], fmin.[wd], fmin_a.[wd], flog2.[wd]
def : InstRW<[P5600WriteFPUS], (instregex "^FCLASS_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FMAX_A_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FMAX_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FMIN_A_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FMIN_(W|D)$")>;
def : InstRW<[P5600WriteFPUS], (instregex "^FLOG2_(W|D)$")>;

// interleave right/left, interleave even/odd, insert
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^(ILVR|ILVL)_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^(ILVEV|ILVOD)_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^INSVE_[BHWD]$")>;

// subs_?.[bhwd], subsus_?.[bhwd], subsuu_?.[bhwd], subvi.[bhwd], subv.[bhwd],
def : InstRW<[P5600WriteMSAShortInt], (instregex "^SUBS_(S|U)_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortInt], (instregex "^SUBSUS_(S|U)_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortInt], (instregex "^SUBSUU_(S|U)_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortInt], (instregex "^SUBVI_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortInt], (instregex "^SUBV_[BHWD]$")>;

// mod_[su].[bhwd], div_[su].[bhwd]
def : InstRW<[P5600WriteFPUDivI], (instregex "^MOD_(S|U)_[BHWD]$")>;
def : InstRW<[P5600WriteFPUDivI], (instregex "^DIV_(S|U)_[BHWD]$")>;

// hadd_[su].[bhwd], hsub_[su].[bhwd], max_[sua].[bhwd], min_[sua].[bhwd],
// maxi_[su].[bhwd], mini_[su].[bhwd], sra?.[bhwd], srar?.[bhwd], srlr.[bhwd],
// sll?.[bhwd], pckev.[bhwd], pckod.[bhwd], nloc.[bhwd], nlzc.[bhwd],
// insve.[bhwd]
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^HADD_(S|U)_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^HSUB_(S|U)_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^(MAX|MIN)_S_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^(MAX|MIN)_U_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^(MAX|MIN)_A_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^(MAXI|MINI)_(S|U)_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^(SRA|SRAI)_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^(SRL|SRLI)_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^(SRAR|SRARI)_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^(SRLR|SRLRI)_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^(SLL|SLLI)_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^(PCKEV|PCKOD)_[BHWD]$")>;
def : InstRW<[P5600WriteMSAShortLogic], (instregex "^(NLOC|NLZC)_[BHWD]$")>;

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

// frcp.[wd], frsqrt.[wd]
def : InstRW<[P5600WriteFPURsqrtD], (instregex "^FRCP_(W|D)$")>;
def : InstRW<[P5600WriteFPURsqrtD], (instregex "^FRSQRT_(W|D)$")>;

def : ItinRW<[P5600WriteFPURsqrtD], [II_RECIP_D, II_RSQRT_D]>;
def : ItinRW<[P5600WriteFPURsqrtS], [II_RECIP_S, II_RSQRT_S]>;

// fmadd.[wd], fmsubb.[wd], fdiv.[wd], fsqrt.[wd], fmul.[wd], fadd.[wd],
// fsub.[wd]
def : InstRW<[P5600WriteFPUL_MADDSUB], (instregex "^FMADD_(W|D)$")>;
def : InstRW<[P5600WriteFPUL_MADDSUB], (instregex "^FMSUB_(W|D)$")>;
def : InstRW<[P5600WriteFPUDivS], (instregex "^FDIV_W$")>;
def : InstRW<[P5600WriteFPUDivD], (instregex "^FDIV_D$")>;
def : InstRW<[P5600WriteFPUSqrtS], (instregex "^FSQRT_W$")>;
def : InstRW<[P5600WriteFPUSqrtD], (instregex "^FSQRT_D$")>;
def : InstRW<[P5600WriteFPUL], (instregex "^FMUL_(W|D)$")>;
def : InstRW<[P5600WriteFPUL], (instregex "^FADD_(W|D)$")>;
def : InstRW<[P5600WriteFPUL], (instregex "^FSUB_(W|D)$")>;

// dpadd_?.[bhwd], dpsub_?.[bhwd], dotp_?.[bhwd], msubv.[bhwd], maddv.[bhwd]
// mulv.[bhwd].
def : InstRW<[P5600WriteMSALongInt], (instregex "^DPADD_(S|U)_[HWD]$")>;
def : InstRW<[P5600WriteMSALongInt], (instregex "^DPSUB_(S|U)_[HWD]$")>;
def : InstRW<[P5600WriteMSALongInt], (instregex "^DOTP_(S|U)_[HWD]$")>;
def : InstRW<[P5600WriteMSALongInt], (instregex "^MSUBV_[BHWD]$")>;
def : InstRW<[P5600WriteMSALongInt], (instregex "^MADDV_[BHWD]$")>;
def : InstRW<[P5600WriteMSALongInt], (instregex "^MULV_[BHWD]$")>;

def : InstRW<[P5600WriteMSALongInt], (instregex "^MADDR_Q_[HW]$")>;
def : InstRW<[P5600WriteMSALongInt], (instregex "^MADD_Q_[HW]$")>;
def : InstRW<[P5600WriteMSALongInt], (instregex "^MSUBR_Q_[HW]$")>;
def : InstRW<[P5600WriteMSALongInt], (instregex "^MSUB_Q_[HW]$")>;
def : InstRW<[P5600WriteMSALongInt], (instregex "^MULR_Q_[HW]$")>;
def : InstRW<[P5600WriteMSALongInt], (instregex "^MUL_Q_[HW]$")>;

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

// copy.[su]_[bhwd]
def : InstRW<[P5600WriteMoveFPUToGPR], (instregex "^COPY_U_[BHW]$")>;
def : InstRW<[P5600WriteMoveFPUToGPR], (instregex "^COPY_S_[BHWD]$")>;

// bc1[ft], cfc1, mfc1, mfhc1, movf, movt
def : ItinRW<[P5600WriteMoveFPUToGPR],
             [II_BC1F, II_BC1FL, II_BC1T, II_BC1TL, II_CFC1, II_MFC1, II_MFHC1, II_MOVF, II_MOVT]>;

// swc1, swxc1, st.[bhwd]
def : ItinRW<[P5600WriteStoreFPUS], [II_SDC1, II_SDXC1, II_SUXC1, II_SWC1,
                                     II_SWXC1]>;
def : InstRW<[P5600WriteStoreFPUS], (instregex "^ST_[BHWD]$")>;

// movn.[ds], movz.[ds]
def : ItinRW<[P5600WriteStoreFPUL], [II_MOVN_D, II_MOVN_S, II_MOVZ_D, II_MOVZ_S]>;

// l[dw]x?c1, ld.[bhwd]
def : ItinRW<[P5600WriteLoadFPU], [II_LDC1, II_LDXC1, II_LWC1, II_LWXC1, II_LUXC1]>;
def : InstRW<[P5600WriteLoadFPU], (instregex "LD_[BHWD]")>;

// Unsupported Instructions
// ========================
//
// The following instruction classes are never valid on P5600.
//   II_DADDIU, II_DADDU, II_DMFC1, II_DMTC1, II_DMULT, II_DMULTU, II_DROTR,
//   II_DROTR32, II_DROTRV, II_DDIV, II_DSLL, II_DSLL32, II_DSLLV, II_DSRA,
//   II_DSRA32, II_DSRAV, II_DSRL, II_DSRL32, II_DSRLV, II_DSUBU, II_DDIVU,
//   II_JALRC, II_LD, II_LD[LR], II_RESTORE, II_SAVE, II_SD, II_SDC1, II_SD[LR]
//
// The following instructions are never valid on P5600.
//   addq.ph, repl.ph, repl.qb, subq.ph, subu_s.qb
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
