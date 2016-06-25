//=- AArch64SchedM1.td - Samsung Exynos-M1 Scheduling Defs ---*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Samsung Exynos-M1 to support
// instruction scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// The Exynos-M1 is a traditional superscalar microprocessor with a
// 4-wide in-order stage for decode and dispatch and a wider issue stage.
// The execution units and loads and stores are out-of-order.

def ExynosM1Model : SchedMachineModel {
  let IssueWidth            =  4; // Up to 4 uops per cycle.
  let MicroOpBufferSize     = 96; // ROB size.
  let LoopMicroOpBufferSize = 24; // Based on the instruction queue size.
  let LoadLatency           =  4; // Optimistic load cases.
  let MispredictPenalty     = 14; // Minimum branch misprediction penalty.
  let CompleteModel         =  0; // Use the default model otherwise.
}

//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available on the Exynos-M1,
// which has 9 pipelines, each with its own queue with out-of-order dispatch.

def M1UnitA  : ProcResource<2>; // Simple integer
def M1UnitC  : ProcResource<1>; // Simple and complex integer
def M1UnitB  : ProcResource<2>; // Branch
def M1UnitL  : ProcResource<1>; // Load
def M1UnitS  : ProcResource<1>; // Store
def M1PipeF0 : ProcResource<1>; // FP #0
def M1PipeF1 : ProcResource<1>; // FP #1

let Super = M1PipeF0 in {
  def M1UnitFMAC   : ProcResource<1>; // FP multiplication
  def M1UnitFCVT   : ProcResource<1>; // FP conversion
  def M1UnitNAL0   : ProcResource<1>; // Simple vector.
  def M1UnitNMISC  : ProcResource<1>; // Miscellanea
  def M1UnitNCRYPT : ProcResource<1>; // Cryptographic
}

let Super = M1PipeF1 in {
  def M1UnitFADD : ProcResource<1>; // Simple FP
  let BufferSize = 1 in
  def M1UnitFVAR : ProcResource<1>; // FP division & square root (serialized)
  def M1UnitNAL1 : ProcResource<1>; // Simple vector.
  def M1UnitFST  : ProcResource<1>; // FP store
}

let SchedModel = ExynosM1Model in {
  def M1UnitALU  : ProcResGroup<[M1UnitA,
                                 M1UnitC]>;    // All simple integer.
  def M1UnitNALU : ProcResGroup<[M1UnitNAL0,
                                 M1UnitNAL1]>; // All simple vector.
}

let SchedModel = ExynosM1Model in {

//===----------------------------------------------------------------------===//
// Coarse scheduling model for the Exynos-M1.

// Branch instructions.
// TODO: Non-conditional direct branches take zero cycles and units.
def : WriteRes<WriteBr,    [M1UnitB]> { let Latency = 1; }
def : WriteRes<WriteBrReg, [M1UnitC]> { let Latency = 1; }
// TODO: Branch and link is much different.

// Arithmetic and logical integer instructions.
def : WriteRes<WriteI,     [M1UnitALU]> { let Latency = 1; }
// TODO: Shift over 3 and some extensions take 2 cycles.
def : WriteRes<WriteISReg, [M1UnitALU]> { let Latency = 1; }
def : WriteRes<WriteIEReg, [M1UnitALU]> { let Latency = 1; }
def : WriteRes<WriteIS,    [M1UnitALU]> { let Latency = 1; }

// Move instructions.
def : WriteRes<WriteImm, [M1UnitALU]> { let Latency = 1; }

// Divide and multiply instructions.
// TODO: Division blocks the divider inside C.
def : WriteRes<WriteID32, [M1UnitC]> { let Latency = 13; }
def : WriteRes<WriteID64, [M1UnitC]> { let Latency = 21; }
// TODO: Long multiplication take 5 cycles and also the ALU.
// TODO: Multiplication with accumulation can be advanced.
def : WriteRes<WriteIM32, [M1UnitC]> { let Latency = 3; }
// TODO: 64-bit multiplication has a throughput of 1/2.
def : WriteRes<WriteIM64, [M1UnitC]> { let Latency = 4; }

// Miscellaneous instructions.
def : WriteRes<WriteExtr, [M1UnitALU,
                           M1UnitALU]> { let Latency = 2; }

// TODO: The latency for the post or pre register is 1 cycle.
def : WriteRes<WriteAdr, []> { let Latency = 0; }

// Load instructions.
def : WriteRes<WriteLD,    [M1UnitL]>   { let Latency = 4; }
// TODO: Extended address requires also the ALU.
def : WriteRes<WriteLDIdx, [M1UnitL]>   { let Latency = 5; }
def : WriteRes<WriteLDHi,  [M1UnitALU]> { let Latency = 4; }

// Store instructions.
def : WriteRes<WriteST,    [M1UnitS]> { let Latency = 1; }
// TODO: Extended address requires also the ALU.
def : WriteRes<WriteSTIdx, [M1UnitS]> { let Latency = 1; }
def : WriteRes<WriteSTP,   [M1UnitS]> { let Latency = 1; }
def : WriteRes<WriteSTX,   [M1UnitS]> { let Latency = 1; }

// FP data instructions.
def : WriteRes<WriteF,    [M1UnitFADD]>  { let Latency = 3; }
// TODO: FCCMP is much different.
def : WriteRes<WriteFCmp, [M1UnitNMISC]> { let Latency = 4; }
// TODO: DP takes longer.
def : WriteRes<WriteFDiv, [M1UnitFVAR]>  { let Latency = 15; }
// TODO: MACC takes longer.
def : WriteRes<WriteFMul, [M1UnitFMAC]>  { let Latency = 4; }

// FP miscellaneous instructions.
// TODO: Conversion between register files is much different.
def : WriteRes<WriteFCvt,  [M1UnitFCVT]> { let Latency = 3; }
def : WriteRes<WriteFImm,  [M1UnitNALU]> { let Latency = 1; }
// TODO: Copy from FPR to GPR is much different.
def : WriteRes<WriteFCopy, [M1UnitS]>    { let Latency = 4; }

// FP load instructions.
// TODO: ASIMD loads are much different.
def : WriteRes<WriteVLD, [M1UnitL]> { let Latency = 5; }

// FP store instructions.
// TODO: ASIMD stores are much different.
def : WriteRes<WriteVST, [M1UnitS, M1UnitFST]> { let Latency = 1; }

// ASIMD FP instructions.
// TODO: Other operations are much different.
def : WriteRes<WriteV, [M1UnitFADD]> { let Latency = 3; }

// Other miscellaneous instructions.
def : WriteRes<WriteAtomic,  []> { let Unsupported = 1; }
def : WriteRes<WriteBarrier, []> { let Latency = 1; }
def : WriteRes<WriteHint,    []> { let Latency = 1; }
def : WriteRes<WriteSys,     []> { let Latency = 1; }

//===----------------------------------------------------------------------===//
// Generic fast forwarding.

// TODO: Add FP register forwarding rules.

def : ReadAdvance<ReadI,       0>;
def : ReadAdvance<ReadISReg,   0>;
def : ReadAdvance<ReadIEReg,   0>;
def : ReadAdvance<ReadIM,      0>;
// Integer multiply-accumulate.
// TODO: The forwarding for WriteIM64 saves actually 3 cycles.
def : ReadAdvance<ReadIMA,     2, [WriteIM32, WriteIM64]>;
def : ReadAdvance<ReadID,      0>;
def : ReadAdvance<ReadExtrHi,  0>;
def : ReadAdvance<ReadAdrBase, 0>;
def : ReadAdvance<ReadVLD,     0>;

//===----------------------------------------------------------------------===//
// Finer scheduling model for the Exynos-M1.

def M1WriteNEONA   : SchedWriteRes<[M1UnitNALU,
                                    M1UnitNALU,
                                    M1UnitFADD]>   { let Latency = 9; }
def M1WriteNEONB   : SchedWriteRes<[M1UnitNALU,
                                    M1UnitFST]>    { let Latency = 5; }
def M1WriteNEONC   : SchedWriteRes<[M1UnitNALU,
                                    M1UnitFST]>    { let Latency = 6; }
def M1WriteNEOND   : SchedWriteRes<[M1UnitNALU,
                                    M1UnitFST,
                                    M1UnitL]>      { let Latency = 10; }
def M1WriteNEONE   : SchedWriteRes<[M1UnitFCVT,
                                    M1UnitFST]>    { let Latency = 8; }
def M1WriteNEONF   : SchedWriteRes<[M1UnitFCVT,
                                    M1UnitFST,
                                    M1UnitL]>      { let Latency = 13; }
def M1WriteNEONG   : SchedWriteRes<[M1UnitNMISC,
                                    M1UnitFST]>    { let Latency = 6; }
def M1WriteNEONH   : SchedWriteRes<[M1UnitNALU,
                                    M1UnitFST]>    { let Latency = 3; }
def M1WriteNEONI   : SchedWriteRes<[M1UnitFST,
                                    M1UnitL]>      { let Latency = 9; }
def M1WriteNEONJ   : SchedWriteRes<[M1UnitNMISC,
                                    M1UnitFMAC]>   { let Latency = 6; }
def M1WriteNEONK   : SchedWriteRes<[M1UnitNMISC,
                                    M1UnitFMAC]>   { let Latency = 7; }
def M1WriteALU1    : SchedWriteRes<[M1UnitALU]>    { let Latency = 1; }
def M1WriteB       : SchedWriteRes<[M1UnitB]>      { let Latency = 1; }
// FIXME: This is the worst case, conditional branch and link.
def M1WriteBL      : SchedWriteRes<[M1UnitB,
                                    M1UnitALU]>    { let Latency = 1; }
// FIXME: This is the worst case, when using LR.
def M1WriteBLR     : SchedWriteRes<[M1UnitB,
                                    M1UnitALU,
                                    M1UnitALU]>    { let Latency = 2; }
def M1WriteC1      : SchedWriteRes<[M1UnitC]>      { let Latency = 1; }
def M1WriteC2      : SchedWriteRes<[M1UnitC]>      { let Latency = 2; }
def M1WriteFADD3   : SchedWriteRes<[M1UnitFADD]>   { let Latency = 3; }
def M1WriteFCVT3   : SchedWriteRes<[M1UnitFCVT]>   { let Latency = 3; }
def M1WriteFCVT4   : SchedWriteRes<[M1UnitFCVT]>   { let Latency = 4; }
def M1WriteFMAC4   : SchedWriteRes<[M1UnitFMAC]>   { let Latency = 4; }
def M1WriteFMAC5   : SchedWriteRes<[M1UnitFMAC]>   { let Latency = 5; }
def M1WriteFVAR15  : SchedWriteRes<[M1UnitFVAR]>   { let Latency = 15; }
def M1WriteFVAR23  : SchedWriteRes<[M1UnitFVAR]>   { let Latency = 23; }
def M1WriteNALU1   : SchedWriteRes<[M1UnitNALU]>   { let Latency = 1; }
def M1WriteNALU2   : SchedWriteRes<[M1UnitNALU]>   { let Latency = 2; }
def M1WriteNAL11   : SchedWriteRes<[M1UnitNAL1]>   { let Latency = 1; }
def M1WriteNAL12   : SchedWriteRes<[M1UnitNAL1]>   { let Latency = 2; }
def M1WriteNAL13   : SchedWriteRes<[M1UnitNAL1]>   { let Latency = 3; }
def M1WriteNCRYPT1 : SchedWriteRes<[M1UnitNCRYPT]> { let Latency = 1; }
def M1WriteNCRYPT5 : SchedWriteRes<[M1UnitNCRYPT]> { let Latency = 5; }
def M1WriteNMISC1  : SchedWriteRes<[M1UnitNMISC]>  { let Latency = 1; }
def M1WriteNMISC2  : SchedWriteRes<[M1UnitNMISC]>  { let Latency = 2; }
def M1WriteNMISC3  : SchedWriteRes<[M1UnitNMISC]>  { let Latency = 3; }
def M1WriteNMISC4  : SchedWriteRes<[M1UnitNMISC]>  { let Latency = 4; }
def M1WriteS4      : SchedWriteRes<[M1UnitS]>      { let Latency = 4; }
def M1WriteTB      : SchedWriteRes<[M1UnitC,
                                    M1UnitALU]>    { let Latency = 2; }

// Branch instructions
def : InstRW<[M1WriteB ],  (instrs Bcc)>;
def : InstRW<[M1WriteBL],  (instrs BL)>;
def : InstRW<[M1WriteBLR], (instrs BLR)>;
def : InstRW<[M1WriteC1],  (instregex "^CBN?Z[WX]")>;
def : InstRW<[M1WriteTB],  (instregex "^TBN?Z[WX]")>;

// Arithmetic and logical integer instructions.
def : InstRW<[M1WriteALU1], (instrs COPY)>;

// Divide and multiply instructions.

// Miscellaneous instructions.

// Load instructions.

// Store instructions.

// FP data instructions.
def : InstRW<[M1WriteNALU1],  (instregex "^F(ABS|NEG)[DS]r")>;
def : InstRW<[M1WriteFADD3],  (instregex "^F(ADD|SUB)[DS]rr")>;
def : InstRW<[M1WriteNEONG],  (instregex "^FCCMPE?[DS]rr")>;
def : InstRW<[M1WriteNMISC4], (instregex "^FCMPE?[DS]r")>;
def : InstRW<[M1WriteFVAR15], (instrs FDIVSrr)>;
def : InstRW<[M1WriteFVAR23], (instrs FDIVDrr)>;
def : InstRW<[M1WriteNMISC2], (instregex "^F(MAX|MIN).+rr")>;
def : InstRW<[M1WriteFMAC4],  (instregex "^FN?MUL[DS]rr")>;
def : InstRW<[M1WriteFMAC5],  (instregex "^FN?M(ADD|SUB)[DS]rrr")>;
def : InstRW<[M1WriteFCVT3],  (instregex "^FRINT.+r")>;
def : InstRW<[M1WriteNEONH],  (instregex "^FCSEL[DS]rrr")>;
def : InstRW<[M1WriteFVAR15], (instrs FSQRTSr)>;
def : InstRW<[M1WriteFVAR23], (instrs FSQRTDr)>;

// FP miscellaneous instructions.
def : InstRW<[M1WriteFCVT3], (instregex "^FCVT[DS][DS]r")>;
def : InstRW<[M1WriteNEONF], (instregex "^[FSU]CVT[AMNPZ][SU](_Int)?[SU]?[XW]?[DS]?[rds]i?")>;
def : InstRW<[M1WriteNEONE], (instregex "^[SU]CVTF[SU]")>;
def : InstRW<[M1WriteNALU1], (instregex "^FMOV[DS][ir]")>;
def : InstRW<[M1WriteS4],    (instregex "^FMOV[WX][DS](High)?r")>;
def : InstRW<[M1WriteNEONI], (instregex "^FMOV[DS][WX](High)?r")>;

// FP load instructions.

// FP store instructions.

// ASIMD instructions.
def : InstRW<[M1WriteNMISC3], (instregex "^[SU]ABAL?v")>;
def : InstRW<[M1WriteNMISC1], (instregex "^[SU]ABDL?v")>;
def : InstRW<[M1WriteNMISC1], (instregex "^(SQ)?ABSv")>;
def : InstRW<[M1WriteNMISC1], (instregex "^SQNEGv")>;
def : InstRW<[M1WriteNALU1],  (instregex "^(ADD|NEG|SUB)v")>;
def : InstRW<[M1WriteNMISC3], (instregex "^[SU]?H(ADD|SUB)v")>;
def : InstRW<[M1WriteNMISC3], (instregex "^[SU]?AD[AD](L|LP|P|W)V?2?v")>;
def : InstRW<[M1WriteNMISC3], (instregex "^[SU]?SUB[LW]2?v")>;
def : InstRW<[M1WriteNMISC3], (instregex "^R?(ADD|SUB)HN?2?v")>;
def : InstRW<[M1WriteNMISC3], (instregex "^[SU]+Q(ADD|SUB)v")>;
def : InstRW<[M1WriteNMISC3], (instregex "^[SU]RHADDv")>;
def : InstRW<[M1WriteNMISC1], (instregex "^CM(EQ|GE|GT|HI|HS|LE|LT)v")>;
def : InstRW<[M1WriteNALU1],  (instregex "^CMTSTv")>;
def : InstRW<[M1WriteNALU1],  (instregex "^(AND|BIC|EOR|MVNI|NOT|ORN|ORR)v")>;
def : InstRW<[M1WriteNMISC1], (instregex "^[SU](MIN|MAX)v")>;
def : InstRW<[M1WriteNMISC2], (instregex "^[SU](MIN|MAX)Pv")>;
def : InstRW<[M1WriteNMISC3], (instregex "^[SU](MIN|MAX)Vv")>;
def : InstRW<[M1WriteNMISC4], (instregex "^(MUL|SQR?DMULH)v")>;
def : InstRW<[M1WriteNMISC4], (instregex "^ML[AS]v")>;
def : InstRW<[M1WriteNMISC4], (instregex "^(S|U|SQD|SQRD)ML[AS][HL]v")>;
def : InstRW<[M1WriteNMISC4], (instregex "^(S|U|SQD)MULLv")>;
def : InstRW<[M1WriteNAL13],  (instregex "^(S|SR|U|UR)SRAv")>;
def : InstRW<[M1WriteNALU1],  (instregex "^[SU]?SH(L|LL|R)2?v")>;
def : InstRW<[M1WriteNALU1],  (instregex "^S[LR]Iv")>;
def : InstRW<[M1WriteNAL13],  (instregex "^[SU]?(Q|QR|R)?SHR(N|U|UN)?2?v")>;
def : InstRW<[M1WriteNAL13],  (instregex "^[SU](Q|QR|R)SHLU?v")>;

// ASIMD FP instructions.
def : InstRW<[M1WriteNALU1],  (instregex "^F(ABS|NEG)v")>;
def : InstRW<[M1WriteNMISC3], (instregex "^F(ABD|ADD|SUB)v")>;
def : InstRW<[M1WriteNEONA],  (instregex "^FADDP")>;
def : InstRW<[M1WriteNMISC1], (instregex "^F(AC|CM)(EQ|GE|GT|LE|LT)v[^1]")>;
def : InstRW<[M1WriteFCVT3],  (instregex "^[FVSU]CVTX?[AFLMNPZ][SU]?(_Int)?v")>;
def : InstRW<[M1WriteFVAR15], (instregex "FDIVv.f32")>;
def : InstRW<[M1WriteFVAR23], (instregex "FDIVv2f64")>;
def : InstRW<[M1WriteFVAR15], (instregex "FSQRTv.f32")>;
def : InstRW<[M1WriteFVAR23], (instregex "FSQRTv2f64")>;
def : InstRW<[M1WriteNMISC1], (instregex "^F(MAX|MIN)(NM)?V?v")>;
def : InstRW<[M1WriteNMISC2], (instregex "^F(MAX|MIN)(NM)?Pv")>;
def : InstRW<[M1WriteNEONJ],  (instregex "^FMULX?v.i")>;
def : InstRW<[M1WriteFMAC4],  (instregex "^FMULX?v.f")>;
def : InstRW<[M1WriteNEONK],  (instregex "^FML[AS]v.i")>;
def : InstRW<[M1WriteFMAC5],  (instregex "^FML[AS]v.f")>;
def : InstRW<[M1WriteFCVT3],  (instregex "^FRINT[AIMNPXZ]v")>;

// ASIMD miscellaneous instructions.
def : InstRW<[M1WriteNALU1],  (instregex "^RBITv")>;
def : InstRW<[M1WriteNAL11],  (instregex "^(BIF|BIT|BSL)v")>;
def : InstRW<[M1WriteNALU1],  (instregex "^CPY")>;
def : InstRW<[M1WriteNEONB],  (instregex "^DUPv.+gpr")>;
def : InstRW<[M1WriteNALU1],  (instregex "^DUPv.+lane")>;
def : InstRW<[M1WriteNAL13],  (instregex "^[SU]?Q?XTU?Nv")>;
def : InstRW<[M1WriteNEONC],  (instregex "^INSv.+gpr")>;
def : InstRW<[M1WriteFCVT4],  (instregex "^[FU](RECP|RSQRT)Ev")>;
def : InstRW<[M1WriteNMISC1], (instregex "^[FU](RECP|RSQRT)Xv")>;
def : InstRW<[M1WriteFMAC5],  (instregex "^F(RECP|RSQRT)Sv")>;
def : InstRW<[M1WriteNALU1],  (instregex "^REV(16|32|64)v")>;
def : InstRW<[M1WriteNAL11],  (instregex "^TB[LX]v8i8One")>;
def : InstRW<[WriteSequence<[M1WriteNAL11], 2>],
                              (instregex "^TB[LX]v8i8Two")>;
def : InstRW<[WriteSequence<[M1WriteNAL11], 3>],
                              (instregex "^TB[LX]v8i8Three")>;
def : InstRW<[WriteSequence<[M1WriteNAL11], 4>],
                              (instregex "^TB[LX]v8i8Four")>;
def : InstRW<[M1WriteNAL12],  (instregex "^TB[LX]v16i8One")>;
def : InstRW<[WriteSequence<[M1WriteNAL12], 2>],
                              (instregex "^TB[LX]v16i8Two")>;
def : InstRW<[WriteSequence<[M1WriteNAL12], 3>],
                              (instregex "^TB[LX]v16i8Three")>;
def : InstRW<[WriteSequence<[M1WriteNAL12], 4>],
                              (instregex "^TB[LX]v16i8Four")>;
def : InstRW<[M1WriteNEOND],  (instregex "^[SU]MOVv")>;
def : InstRW<[M1WriteNALU1],  (instregex "^INSv.+lane")>;
def : InstRW<[M1WriteNALU1],  (instregex "^(TRN|UZP)[12](v8i8|v4i16|v2i32)")>;
def : InstRW<[M1WriteNALU2],  (instregex "^(TRN|UZP)[12](v16i8|v8i16|v4i32|v2i64)")>;
def : InstRW<[M1WriteNALU1],  (instregex "^ZIP[12]v")>;

// ASIMD load instructions.

// ASIMD store instructions.

// Cryptography instructions.
def M1WriteAES : SchedWriteRes<[M1UnitNCRYPT]> { let Latency = 1; }
def M1ReadAES  : SchedReadAdvance<1, [M1WriteAES]>;
def : InstRW<[M1WriteAES, M1ReadAES], (instregex "^AES")>;

def : InstRW<[M1WriteNCRYPT1], (instregex "^PMUL")>;
def : InstRW<[M1WriteNCRYPT1], (instregex "^SHA1(H|SU)")>;
def : InstRW<[M1WriteNCRYPT5], (instregex "^SHA1[CMP]")>;
def : InstRW<[M1WriteNCRYPT1], (instregex "^SHA256SU0")>;
def : InstRW<[M1WriteNCRYPT5], (instregex "^SHA256(H|SU1)")>;

// CRC instructions.
def : InstRW<[M1WriteC2], (instregex "^CRC32")>;

} // SchedModel = ExynosM1Model
