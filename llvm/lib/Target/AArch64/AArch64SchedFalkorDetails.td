//==- AArch64SchedFalkorDetails.td - Falkor Scheduling Defs -*- tablegen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the uop and latency details for the machine model for the
// Qualcomm Falkor subtarget.
//
//===----------------------------------------------------------------------===//

include "AArch64SchedFalkorWriteRes.td"

//===----------------------------------------------------------------------===//
// Specialize the coarse model by associating instruction groups with the
// subtarget-defined types. As the modeled is refined, this will override most
// of the earlier mappings.

// Miscellaneous
// -----------------------------------------------------------------------------

def : InstRW<[WriteI], (instrs COPY)>;

// SIMD Floating-point Instructions
// -----------------------------------------------------------------------------
def : InstRW<[FalkorWr_1VXVY_1cyc],   (instregex "^(FABS|FNEG)(v2f32|v4f16)$")>;

def : InstRW<[FalkorWr_1VXVY_2cyc],   (instregex "^(F(MAX|MIN)(NM)?P?|FAC(GE|GT))(v2f32|v4f16|v2i16p|v2i32p)$")>;
def : InstRW<[FalkorWr_1VXVY_2cyc],   (instregex "^FAC(GE|GT)(16|32|64)$")>;
def : InstRW<[FalkorWr_1VXVY_2cyc],   (instregex "^FCM(EQ|GE|GT)(16|32|64|v2f32|v4f16|v2i32|v4i16)$")>;
def : InstRW<[FalkorWr_1VXVY_2cyc],   (instregex "^FCM(EQ|LE|GE|GT|LT)(v1i16|v1i32|v1i64|v2i32|v4i16)rz$")>;
def : InstRW<[FalkorWr_1VXVY_2cyc],   (instregex "^FRINT(A|I|M|N|P|X|Z)(v2f32|v4f16)$")>;

def : InstRW<[FalkorWr_1VXVY_3cyc],   (instregex "^F(MAX|MIN)(NM)?V(v4i16|v4i32|v8i16)v$")>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instregex "^(FABD|FADD|FSUB)(v2f32|v4f16)$")>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instregex "^FADDP(v2i16p|v2i32p|v2i64p|v2f32|v4f16)$")>;

def : InstRW<[FalkorWr_1VXVY_4cyc],   (instregex "^FCVT(N|M|P|Z|A)(S|U)(v1i32|v1i64|v1f16|v2f32|v4f16)$")>;
def : InstRW<[FalkorWr_1VXVY_4cyc],   (instrs FCVTXNv1i64)>;
def : InstRW<[FalkorWr_1VXVY_4cyc],   (instregex "^FCVTZ(S|U)(v2i32|v4i16)(_shift)?$")>;

def : InstRW<[FalkorWr_1VXVY_5cyc],   (instregex "^(FMUL|FMULX)(v2f32|v4f16|(v1i16_indexed|v4i16_indexed|v1i32_indexed|v2i32_indexed))$")>;
def : InstRW<[FalkorWr_1VXVY_5cyc],   (instrs FMULX16, FMULX32)>;

def : InstRW<[FalkorWr_1VXVY_6cyc],   (instregex "^(FMUL|FMULX)v1i64_indexed$")>;
def : InstRW<[FalkorWr_1VXVY_6cyc],   (instrs FMULX64)>;

def : InstRW<[FalkorWr_2VXVY_1cyc],   (instregex "^(FABS|FNEG)(v2f64|v4f32|v8f16)$")>;

def : InstRW<[FalkorWr_2VXVY_2cyc],   (instregex "^(F(MAX|MIN)(NM)?P?|FAC(GE|GT)|FCM(EQ|GE|GT))(v2f64|v4f32|v8f16|v2i64p)$")>;
def : InstRW<[FalkorWr_2VXVY_2cyc],   (instregex "^FCM(EQ|LE|GE|GT|LT)(v2i64|v4i32|v8i16)rz$")>;
def : InstRW<[FalkorWr_2VXVY_2cyc],   (instregex "^FRINT(A|I|M|N|P|X|Z)(v2f64|v4f32|v8f16)$")>;

def : InstRW<[FalkorWr_1VX_1VY_10cyc],(instregex "^(FDIV|FSQRT)(v2f32|v4f16)$")>;

def : InstRW<[FalkorWr_2VXVY_3cyc],   (instregex "^(FABD|FADD(P)?|FSUB)(v2f64|v4f32|v8f16)$")>;

def : InstRW<[FalkorWr_2VXVY_4cyc],   (instregex "^FCVT(N|M|P|Z|A)(S|U)(v2f64|v4f32|v8f16)$")>;
def : InstRW<[FalkorWr_2VXVY_4cyc],   (instregex "^(FCVTL|FCVTL2)(v2i32|v4i16|v4i32|v8i16)$")>;
def : InstRW<[FalkorWr_2VXVY_4cyc],   (instregex "^FCVTZ(S|U)(v2i64|v4i32|v8i16)(_shift)?$")>;

def : InstRW<[FalkorWr_2VXVY_5cyc],   (instregex "^(FMUL|FMULX)(v2f64|v4f32|v8f16|v8i16_indexed|v4i32_indexed)$")>;

def : InstRW<[FalkorWr_2VXVY_6cyc],   (instregex "^(FMUL|FMULX)v2i64_indexed$")>;

def : InstRW<[FalkorWr_3VXVY_4cyc],   (instregex "^(FCVTX?N|FCVTX?N2)(v1i32|v1i64|v1f16|v2f32|v4f16)$")>;

def : InstRW<[FalkorWr_3VXVY_5cyc],   (instregex "^(FCVTX?N|FCVTX?N2)(v2i32|v4i16|v4i32|v8i16|v4f32)$")>;

def : InstRW<[FalkorWr_2VX_2VY_2cyc], (instregex "^(FDIV|FSQRT)(v2f64|v4f32|v8f16)$")>;

def : InstRW<[FalkorWr_1VXVY_4cyc, FalkorReadVMA],(instregex "^ML(A|S)(v8i8|v4i16|v2i32)(_indexed)?$")>;
def : InstRW<[FalkorWr_2VXVY_4cyc, FalkorReadVMA],(instregex "^ML(A|S)(v16i8|v8i16|v4i32|v2i64)(_indexed)?$")>;

def : InstRW<[FalkorWr_1VXVY_5cyc, FalkorReadFMA],(instregex "^FML(A|S)(v2f32|v4f16|(v1i16_indexed|v4i16_indexed|v1i32_indexed|v2i32_indexed))$")>;
def : InstRW<[FalkorWr_1VXVY_6cyc, FalkorReadFMA],(instregex "^FML(A|S)v1i64_indexed$")>;
def : InstRW<[FalkorWr_2VXVY_5cyc, FalkorReadFMA],(instregex "^FML(A|S)(v2f64|v4f32|v8f16|v8i16_indexed|v4i32_indexed)$")>;
def : InstRW<[FalkorWr_2VXVY_6cyc, FalkorReadFMA],(instregex "^FML(A|S)v2i64_indexed$")>;
// SIMD Integer Instructions
// -----------------------------------------------------------------------------
def : InstRW<[FalkorWr_1VXVY_1cyc],   (instregex "^ADD(v1i64|v2i32|v4i16|v8i8)$")>;
def : InstRW<[FalkorWr_1VXVY_1cyc],   (instrs ADDPv2i64p)>;
def : InstRW<[FalkorWr_1VXVY_1cyc],   (instregex "^(AND|ORR|ORN|BIC|EOR)v8i8$")>;
def : InstRW<[FalkorWr_1VXVY_1cyc],   (instregex "^(BIC|ORR)(v2i32|v4i16)$")>;
def : InstRW<[FalkorWr_1VXVY_1cyc],   (instregex "^NEG(v1i64|v2i32|v4i16|v8i8)$")>;
def : InstRW<[FalkorWr_1VXVY_1cyc],   (instregex "^SUB(v1i64|v2i32|v4i16|v8i8)$")>;

def : InstRW<[FalkorWr_1VXVY_2cyc],   (instregex "^(S|U)(ADDLP|HADD|HSUB|SHL)(v2i32|v4i16|v8i8)(_v.*)?$")>;
def : InstRW<[FalkorWr_1VXVY_2cyc],   (instregex "^(S|U)SHLv1i64$")>;
def : InstRW<[FalkorWr_1VXVY_2cyc],   (instregex "^(S|U)SHR(v2i32|v4i16|v8i8)_shift$")>;
def : InstRW<[FalkorWr_1VXVY_2cyc],   (instregex "^((S|U)?(MAX|MIN)P?|ABS|ADDP|CM(EQ|GE|HS|GT|HI))(v1i64|v2i32|v4i16|v8i8)$")>;
def : InstRW<[FalkorWr_1VXVY_2cyc],   (instregex "^CM(EQ|GE|HS|GT|HI)(v1i64|v2i32|v4i16|v8i8)$")>;
def : InstRW<[FalkorWr_1VXVY_2cyc],   (instregex "^CM(EQ|LE|GE|GT|LT)(v1i64|v2i32|v4i16|v8i8)rz$")>;
def : InstRW<[FalkorWr_1VXVY_2cyc],   (instregex "^CMTST(v1i64|v2i32|v4i16|v8i8)$")>;
def : InstRW<[FalkorWr_1VXVY_2cyc],   (instrs PMULv8i8)>;
def : InstRW<[FalkorWr_1VXVY_2cyc],   (instregex "^SHL(v2i32|v4i16|v8i8)_shift$")>;

def : InstRW<[FalkorWr_1VXVY_3cyc],   (instregex "^SQNEG(v2i32|v4i16|v8i8)$")>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instregex "^(S|U)R?SRA(d|(v2i32|v4i16|v8i8)_shift)$")>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instregex "^(S|U)(ABD|ADALP)(v8i8|v4i16|v2i32)(_v.*)?$")>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instregex "^(S|U)ADDLVv4i16v$")>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instregex "^(S|U)QADD(v1i8|v1i16|v2i16|v1i32|v1i64|v2i32|v4i16|v8i8)$")>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instregex "^(S|U)QSHLU?(d|s|h|b|(v8i8|v4i16|v2i32)_shift)$")>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instregex "^(S|U)(QSHL|RSHL|QRSHL)(v1i8|v1i16|v1i32|v1i64|v2i32|v4i16|v8i8)$")>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instregex "^(SQR?SHRN|UQR?SHRN|SQR?SHRUN)(s|h|b)$")>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instregex "^(S|U)QSUB(v1i8|v1i16|v2i16|v1i32|v1i64|v2i32|v4i16|v8i8)$")>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instregex "^(S|U)RHADD(v2i32|v4i16|v8i8)$")>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instregex "^(S|U)RSHR(v2i32|v4i16|v8i8)_shift$")>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instregex "^(SU|US)QADD(v1i8|v1i16|v2i16|v1i32|v1i64|v2i32|v4i16|v8i8)$")>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instregex "^(S|U)?(MAX|MIN)V(v4i16v|v4i32v)$")>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instrs ADDVv4i16v)>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instregex "^S(L|R)I(d|(v8i8|v4i16|v2i32)_shift)$")>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instregex "^SQABS(v1i8|v1i16|v1i32|v1i64|v2i32|v4i16|v8i8)$")>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instregex "^SQNEG(v1i8|v1i16|v1i32|v1i64)$")>;

def : InstRW<[FalkorWr_1VXVY_4cyc],   (instregex "^(S|U)ADDLVv8i8v$")>;
def : InstRW<[FalkorWr_1VXVY_4cyc],   (instregex "^(S|U)?(MAX|MIN)V(v8i8v|v8i16v)$")>;
def : InstRW<[FalkorWr_1VXVY_4cyc],   (instrs ADDVv8i8v)>;
def : InstRW<[FalkorWr_1VXVY_4cyc],   (instregex "^MUL(v2i32|v4i16|v8i8)(_indexed)?$")>;
def : InstRW<[FalkorWr_1VXVY_4cyc],   (instregex "^SQR?DMULH(v8i8|v4i16|v1i32|v2i32|v1i16)(_indexed)?$")>;
def : InstRW<[FalkorWr_1VXVY_4cyc],   (instregex "^SQDMULL(i16|i32)$")>;
def : InstRW<[FalkorWr_1VXVY_4cyc],   (instregex "^SQRDML(A|S)?H(v8i8|v4i16|v1i32|v2i32|v1i16)(_indexed)?$")>;

def : InstRW<[FalkorWr_1VXVY_5cyc],   (instregex "^(S|U)?(MAX|MIN)Vv16i8v$")>;

def : InstRW<[FalkorWr_2VXVY_3cyc],   (instrs ADDVv4i32v)>;

def : InstRW<[FalkorWr_2VXVY_4cyc],   (instrs ADDVv8i16v)>;
def : InstRW<[FalkorWr_2VXVY_4cyc],   (instregex "^(ADD|SUB)HNv.*$")>;
def : InstRW<[FalkorWr_2VXVY_4cyc],   (instregex "^(S|U)ABA(v2i32|v4i16|v8i8)$")>;

def : InstRW<[FalkorWr_2VXVY_5cyc],   (instrs ADDVv16i8v)>;

def : InstRW<[FalkorWr_2VXVY_6cyc],   (instregex "^(SQR?SHRN|UQR?SHRN|SQR?SHRUN)(v8i8|v16i8|v4i16|v8i16|v2i32|v4i32)_shift?$")>;
def : InstRW<[FalkorWr_2VXVY_6cyc],   (instregex "^R(ADD|SUB)HNv.*$")>;

def : InstRW<[FalkorWr_2VXVY_1cyc],   (instregex "^ADD(v16i8|v8i16|v4i32|v2i64)$")>;
def : InstRW<[FalkorWr_2VXVY_1cyc],   (instrs ADDPv2i64)>; // sz==11
def : InstRW<[FalkorWr_2VXVY_1cyc],   (instregex "^(AND|ORR|ORN|BIC|EOR)v16i8$")>;
def : InstRW<[FalkorWr_2VXVY_1cyc],   (instregex "^(BIC|ORR)(v8i16|v4i32)$")>;
def : InstRW<[FalkorWr_2VXVY_1cyc],   (instregex "^(NEG|SUB)(v16i8|v8i16|v4i32|v2i64)$")>;

def : InstRW<[FalkorWr_2VXVY_2cyc],   (instregex "^(S|U)ADDLv.*$")>;
def : InstRW<[FalkorWr_2VXVY_2cyc],   (instregex "^(S|U)(ADDLP|HADD|HSUB|SHL)(v16i8|v2i64|v4i32|v8i16)(_v.*)?$")>;
def : InstRW<[FalkorWr_2VXVY_2cyc],   (instregex "^(S|U)SHLL(v16i8|v8i16|v4i32|v8i8|v4i16|v2i32)(_shift)?$")>;
def : InstRW<[FalkorWr_2VXVY_2cyc],   (instregex "^(S|U)SHR(v16i8|v8i16|v4i32|v2i64)_shift$")>;
def : InstRW<[FalkorWr_2VXVY_2cyc],   (instregex "^(S|U)SUBLv.*$")>;
def : InstRW<[FalkorWr_2VXVY_2cyc],   (instregex "^((S|U)?(MAX|MIN)P?|ABS)(v16i8|v2i64|v4i32|v8i16)$")>;
def : InstRW<[FalkorWr_2VXVY_2cyc],   (instregex "^ADDP(v4i32|v8i16|v16i8)$")>; // sz!=11
def : InstRW<[FalkorWr_2VXVY_2cyc],   (instregex "^CM(EQ|GE|HS|GT|HI)(v16i8|v2i64|v4i32|v8i16)$")>;
def : InstRW<[FalkorWr_2VXVY_2cyc],   (instregex "^CM(EQ|LE|GE|GT|LT)(v16i8|v2i64|v4i32|v8i16)rz$")>;
def : InstRW<[FalkorWr_2VXVY_2cyc],   (instregex "^(CMTST|PMUL)(v16i8|v2i64|v4i32|v8i16)$")>;
def : InstRW<[FalkorWr_2VXVY_2cyc],   (instregex "^PMULL2?(v8i8|v16i8)$")>;
def : InstRW<[FalkorWr_2VXVY_2cyc],   (instregex "^SHL(v16i8|v8i16|v4i32|v2i64)_shift$")>;
def : InstRW<[FalkorWr_2VXVY_2cyc],   (instregex "^SHLL(v16i8|v8i16|v4i32|v8i8|v4i16|v2i32)(_shift)?$")>;

def : InstRW<[FalkorWr_2VXVY_3cyc],   (instregex "^(S|U)R?SRA(v2i64|v4i32|v8i16|v16i8)_shift$")>;
def : InstRW<[FalkorWr_2VXVY_3cyc],   (instregex "^(S|U)ABD(v16i8|v8i16|v4i32|v2i64)$")>;
def : InstRW<[FalkorWr_2VXVY_3cyc],   (instregex "^(S|U)ABDLv.*$")>;
def : InstRW<[FalkorWr_2VXVY_3cyc],   (instregex "^(S|U)(ADALP|QADD)(v16i8|v8i16|v4i32|v2i64)(_v.*)?$")>;
def : InstRW<[FalkorWr_2VXVY_3cyc],   (instregex "^(S|U)QSHLU?(v2i64|v4i32|v8i16|v16i8)_shift$")>;
def : InstRW<[FalkorWr_2VXVY_3cyc],   (instregex "^(S|U)(QSHL|RSHL|QRSHL|QSUB|RHADD)(v16i8|v8i16|v4i32|v2i64)$")>;
def : InstRW<[FalkorWr_2VXVY_3cyc],   (instregex "^(S|U)RSHR(v2i64|v4i32|v8i16|v16i8)_shift$")>;
def : InstRW<[FalkorWr_2VXVY_3cyc],   (instregex "^(SU|US)QADD(v16i8|v8i16|v4i32|v2i64)$")>;
def : InstRW<[FalkorWr_2VXVY_3cyc],   (instregex "^PMULL2?(v1i64|v2i64)$")>;
def : InstRW<[FalkorWr_2VXVY_3cyc],   (instregex "^S(L|R)I(v16i8|v8i16|v4i32|v2i64)_shift$")>;
def : InstRW<[FalkorWr_2VXVY_3cyc],   (instregex "^SQ(ABS|NEG)(v16i8|v8i16|v4i32|v2i64)$")>;

def : InstRW<[FalkorWr_2VXVY_4cyc],   (instregex "^(MUL|SQR?DMULH)(v16i8|v8i16|v4i32)(_indexed)?$")>;
def : InstRW<[FalkorWr_2VXVY_4cyc],   (instregex "^SQDMULLv.*$")>;
def : InstRW<[FalkorWr_2VXVY_4cyc],   (instregex "^SQRDML(A|S)H(v16i8|v8i16|v4i32)(_indexed)?$")>;

def : InstRW<[FalkorWr_3VXVY_3cyc],   (instregex "^(S|U)ADDLVv4i32v$")>;

def : InstRW<[FalkorWr_3VXVY_5cyc],   (instregex "^(S|U)ADDLVv8i16v$")>;

def : InstRW<[FalkorWr_3VXVY_6cyc],   (instregex "^(S|U)ADDLVv16i8v$")>;

def : InstRW<[FalkorWr_4VXVY_2cyc],   (instregex "^(S|U)(ADD|SUB)Wv.*$")>;

def : InstRW<[FalkorWr_4VXVY_3cyc],   (instregex "^(S|U)ABALv.*$")>;

def : InstRW<[FalkorWr_4VXVY_4cyc],   (instregex "^(S|U)ABA(v16i8|v8i16|v4i32)$")>;

def : InstRW<[FalkorWr_1VXVY_4cyc, FalkorReadVMA],(instregex "^SQD(MLAL|MLSL)(i16|i32)$")>;
def : InstRW<[FalkorWr_2VXVY_4cyc, FalkorReadVMA],(instregex "^SQD(MLAL|MLSL)v.*$")>;
// SIMD Load Instructions
// -----------------------------------------------------------------------------
def : InstRW<[WriteVLD],                               (instregex "^LD1(i64|Onev(8b|4h|2s|1d|16b|8h|4s|2d))$")>;
def : InstRW<[WriteVLD],                               (instregex "LD1Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLD],                               (instrs LD2i64)>;
def : InstRW<[WriteVLD, WriteAdr],                     (instregex "^LD1(i64|Onev(8b|4h|2s|1d|16b|8h|4s|2d))_POST$")>;
def : InstRW<[WriteVLD, WriteAdr],                     (instregex "LD1Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteVLD, WriteAdr],                     (instrs LD2i64_POST)>;

def : InstRW<[FalkorWr_1LD_1VXVY_4cyc],                (instregex "LD1i(8|16|32)$")>;
def : InstRW<[FalkorWr_1LD_1VXVY_4cyc, WriteAdr],      (instregex "LD1i(8|16|32)_POST$")>;

def : InstRW<[FalkorWr_1LD_1none_3cyc],                (instregex "^LD1Twov(8b|4h|2s|1d)$")>;
def : InstRW<[FalkorWr_1LD_1none_3cyc],                (instregex "^LD2Twov(8b|4h|2s|1d)$")>;
def : InstRW<[FalkorWr_1LD_1none_3cyc],                (instregex "^LD2Rv(8b|4h|2s|1d)$")>;
def : InstRW<[FalkorWr_1LD_1none_3cyc, WriteAdr],      (instregex "^LD1Twov(8b|4h|2s|1d)_POST$")>;
def : InstRW<[FalkorWr_1LD_1none_3cyc, WriteAdr],      (instregex "^LD2Twov(8b|4h|2s|1d)_POST$")>;
def : InstRW<[FalkorWr_1LD_1none_3cyc, WriteAdr],      (instregex "^LD2Rv(8b|4h|2s|1d)_POST$")>;

def : InstRW<[FalkorWr_2LD_3cyc],                      (instregex "^LD1Twov(16b|8h|4s|2d)$")>;
def : InstRW<[FalkorWr_2LD_3cyc],                      (instregex "^LD2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[FalkorWr_2LD_3cyc],                      (instregex "^LD2Rv(16b|8h|4s|2d)$")>;
def : InstRW<[FalkorWr_2LD_3cyc],                      (instrs LD3i64)>;
def : InstRW<[FalkorWr_2LD_3cyc],                      (instrs LD4i64)>;
def : InstRW<[FalkorWr_2LD_3cyc, WriteAdr],            (instregex "^LD1Twov(16b|8h|4s|2d)_POST$")>;
def : InstRW<[FalkorWr_2LD_3cyc, WriteAdr],            (instregex "^LD2Twov(16b|8h|4s|2d)_POST$")>;
def : InstRW<[FalkorWr_2LD_3cyc, WriteAdr],            (instregex "^LD2Rv(16b|8h|4s|2d)_POST$")>;
def : InstRW<[FalkorWr_2LD_3cyc, WriteAdr],            (instrs LD3i64_POST)>;
def : InstRW<[FalkorWr_2LD_3cyc, WriteAdr],            (instrs LD4i64_POST)>;

def : InstRW<[FalkorWr_1LD_2VXVY_4cyc],                (instregex "^LD2i(8|16|32)$")>;
def : InstRW<[FalkorWr_1LD_2VXVY_4cyc, WriteAdr],      (instregex "^LD2i(8|16|32)_POST$")>;

def : InstRW<[FalkorWr_2LD_1none_3cyc],                (instregex "^LD1Threev(8b|4h|2s|1d)$")>;
def : InstRW<[FalkorWr_2LD_1none_3cyc],                (instregex "^LD3Rv(8b|4h|2s|1d)$")>;
def : InstRW<[FalkorWr_2LD_1none_3cyc, WriteAdr],      (instregex "^LD1Threev(8b|4h|2s|1d)_POST$")>;
def : InstRW<[FalkorWr_2LD_1none_3cyc, WriteAdr],      (instregex "^LD3Rv(8b|4h|2s|1d)_POST$")>;

def : InstRW<[FalkorWr_3LD_3cyc],                      (instregex "^LD1Threev(16b|8h|4s|2d)$")>;
def : InstRW<[FalkorWr_3LD_3cyc],                      (instrs LD3Threev2d)>;
def : InstRW<[FalkorWr_3LD_3cyc],                      (instregex "^LD3Rv(16b|8h|4s|2d)$")>;
def : InstRW<[FalkorWr_3LD_3cyc, WriteAdr],            (instregex "^LD1Threev(16b|8h|4s|2d)_POST$")>;
def : InstRW<[FalkorWr_3LD_3cyc, WriteAdr],            (instrs LD3Threev2d_POST)>;
def : InstRW<[FalkorWr_3LD_3cyc, WriteAdr],            (instregex "^LD3Rv(16b|8h|4s|2d)_POST$")>;

def : InstRW<[FalkorWr_1LD_3VXVY_4cyc],                (instregex "LD3i(8|16|32)$")>;
def : InstRW<[FalkorWr_1LD_3VXVY_4cyc, WriteAdr],      (instregex "LD3i(8|16|32)_POST$")>;

def : InstRW<[FalkorWr_2LD_2none_3cyc],                (instregex "^LD1Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[FalkorWr_2LD_2none_3cyc],                (instregex "^LD4Rv(8b|4h|2s|1d)$")>;
def : InstRW<[FalkorWr_2LD_2none_3cyc, WriteAdr],      (instregex "^LD1Fourv(8b|4h|2s|1d)_POST$")>;
def : InstRW<[FalkorWr_2LD_2none_3cyc, WriteAdr],      (instregex "^LD4Rv(8b|4h|2s|1d)_POST$")>;

def : InstRW<[FalkorWr_4LD_3cyc],                      (instregex "^LD1Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[FalkorWr_4LD_3cyc],                      (instrs LD4Fourv2d)>;
def : InstRW<[FalkorWr_4LD_3cyc],                      (instregex "^LD4Rv(16b|8h|4s|2d)$")>;
def : InstRW<[FalkorWr_4LD_3cyc, WriteAdr],            (instregex "^LD1Fourv(16b|8h|4s|2d)_POST$")>;
def : InstRW<[FalkorWr_4LD_3cyc, WriteAdr],            (instrs LD4Fourv2d_POST)>;
def : InstRW<[FalkorWr_4LD_3cyc, WriteAdr],            (instregex "^LD4Rv(16b|8h|4s|2d)_POST$")>;

def : InstRW<[FalkorWr_1LD_4VXVY_4cyc],                (instregex "^LD4i(8|16|32)$")>;
def : InstRW<[FalkorWr_1LD_4VXVY_4cyc, WriteAdr],      (instregex "^LD4i(8|16|32)_POST$")>;

def : InstRW<[FalkorWr_2LD_2VXVY_1none_4cyc],          (instregex "LD3Threev(8b|4h|2s|1d)$")>;
def : InstRW<[FalkorWr_2LD_2VXVY_1none_4cyc, WriteAdr],(instregex "LD3Threev(8b|4h|2s|1d)_POST$")>;

def : InstRW<[FalkorWr_2LD_2VXVY_2none_4cyc],          (instregex "^LD4Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[FalkorWr_2LD_2VXVY_2none_4cyc, WriteAdr],(instregex "^LD4Fourv(8b|4h|2s|1d)_POST$")>;

def : InstRW<[FalkorWr_2LD_2VXVY_2LD_2VXVY_4cyc],      (instregex "LD3Threev(16b|8h|4s)$")>;
def : InstRW<[FalkorWr_2LD_2VXVY_2LD_2VXVY_4cyc],      (instregex "^LD4Fourv(16b|8h|4s)$")>;

def : InstRW<[FalkorWr_2LD_2VXVY_1XYZ_2LD_2VXVY_4cyc, WriteAdr],(instregex "LD3Threev(16b|8h|4s)_POST$")>;
def : InstRW<[FalkorWr_2LD_2VXVY_2LD_1XYZ_2VXVY_4cyc, WriteAdr],(instregex "^LD4Fourv(16b|8h|4s)_POST$")>;

// Arithmetic and Logical Instructions
// -----------------------------------------------------------------------------
def : InstRW<[FalkorWr_ADD],          (instregex "^ADD(S)?(W|X)r(s|x)$")>;
def : InstRW<[FalkorWr_1XYZ_1cyc],    (instregex "^AND(S)?(W|X)r(i|r|s)$")>;
def : InstRW<[FalkorWr_1XYZ_1cyc],    (instregex "^BIC(S)?(W|X)r(r|s)$")>;
def : InstRW<[FalkorWr_1XYZ_1cyc],    (instregex "^EON(W|X)r(r|s)$")>;
def : InstRW<[FalkorWr_1XYZ_1cyc],    (instregex "^EOR(W|X)r(i|r|s)$")>;
def : InstRW<[FalkorWr_1XYZ_1cyc],    (instregex "^ORN(W|X)r(r|s)$")>;
def : InstRW<[FalkorWr_1XYZ_1cyc],    (instregex "^ORR(W|X)r(i|r|s)$")>;
def : InstRW<[FalkorWr_2XYZ_2cyc],    (instregex "^SUB(S)?(W|X)r(s|x)$")>;

// SIMD Miscellaneous Instructions
// -----------------------------------------------------------------------------
def : InstRW<[FalkorWr_1GTOV_1cyc],   (instregex "^DUP(v8i8|v4i16|v2i32)(gpr|lane)$")>;
def : InstRW<[FalkorWr_1VXVY_1cyc],   (instregex "^DUP(v16i8|v8i16)(gpr|lane)$")>;
def : InstRW<[FalkorWr_1GTOV_1cyc],   (instregex "^INSv(i8|i16)(gpr|lane)$")>;
def : InstRW<[FalkorWr_1VTOG_1cyc],   (instregex "^(S|U)MOVv.*$")>;
def : InstRW<[FalkorWr_1VXVY_1cyc],   (instregex "^(BIF|BIT|BSL)v8i8$")>;
def : InstRW<[FalkorWr_1VXVY_1cyc],   (instrs EXTv8i8)>;
def : InstRW<[FalkorWr_1VXVY_1cyc],   (instregex "(MOVI|MVNI)(D|v8b_ns|v2i32|v4i16|v2s_msl)$")>;
def : InstRW<[FalkorWr_1VXVY_1cyc],   (instrs TBLv8i8One)>;
def : InstRW<[FalkorWr_1VXVY_1cyc],   (instrs NOTv8i8)>;
def : InstRW<[FalkorWr_1VXVY_1cyc],   (instregex "^REV(16|32|64)v.*$")>;
def : InstRW<[FalkorWr_1VXVY_1cyc],   (instregex "^(TRN1|TRN2|ZIP1|UZP1|UZP2|ZIP2|XTN|XTN2)(v2i32|v2i64|v4i16|v4i32|v8i8|v8i16|v16i8)$")>;

def : InstRW<[FalkorWr_1VXVY_2cyc],   (instregex "^(CLS|CLZ|CNT|RBIT)(v4i32|v8i16|v16i8)$")>;

def : InstRW<[FalkorWr_1VXVY_3cyc],   (instregex "(S|U)QXTU?Nv.*$")>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instrs FRECPEv1i32, FRECPEv1i64, FRSQRTEv1i32, FRSQRTEv1i64, FRECPEv2f32, FRSQRTEv2f32)>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instrs FRECPXv1i32, FRECPXv1i64)>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instrs URECPEv2i32, URSQRTEv2i32)>;

def : InstRW<[FalkorWr_1VXVY_5cyc],   (instrs FRECPS32, FRSQRTS32, FRECPSv2f32, FRSQRTSv2f32)>;

def : InstRW<[FalkorWr_1VXVY_6cyc],   (instrs FRECPS64, FRSQRTS64)>;

def : InstRW<[FalkorWr_1GTOV_1VXVY_2cyc],(instregex "^INSv(i32|i64)(gpr|lane)$")>;
def : InstRW<[FalkorWr_2GTOV_1cyc],   (instregex "^DUP(v4i32|v2i64)(gpr|lane)$")>;
def : InstRW<[FalkorWr_2VXVY_1cyc],   (instrs EXTv16i8)>;
def : InstRW<[FalkorWr_2VXVY_1cyc],   (instregex "(MOVI|MVNI)(v2d_ns|v16b_ns|v4i32|v8i16|v4s_msl)$")>;
def : InstRW<[FalkorWr_2VXVY_1cyc],   (instrs NOTv16i8)>;
def : InstRW<[FalkorWr_2VXVY_1cyc],   (instrs TBLv16i8One)>;

def : InstRW<[FalkorWr_2VXVY_3cyc],   (instrs FRECPEv2f64, FRECPEv4f32, FRSQRTEv2f64, FRSQRTEv4f32)>;
def : InstRW<[FalkorWr_2VXVY_3cyc],   (instrs URECPEv4i32, URSQRTEv4i32)>;

def : InstRW<[FalkorWr_2VXVY_4cyc],   (instrs TBLv8i8Two)>;
def : InstRW<[FalkorWr_2VXVY_4cyc],   (instregex "^TBX(v8|v16)i8One$")>;

def : InstRW<[FalkorWr_2VXVY_5cyc],   (instrs FRECPSv4f32, FRSQRTSv4f32)>;

def : InstRW<[FalkorWr_2VXVY_6cyc],   (instrs FRECPSv2f64, FRSQRTSv2f64)>;

def : InstRW<[FalkorWr_3VXVY_5cyc],   (instregex "^TBL(v8i8Three|v16i8Two)$")>;
def : InstRW<[FalkorWr_3VXVY_5cyc],   (instregex "^TBX(v8i8Two|v16i8Two)$")>;

def : InstRW<[FalkorWr_4VXVY_6cyc],   (instregex "^TBL(v8i8Four|v16i8Three)$")>;
def : InstRW<[FalkorWr_4VXVY_6cyc],   (instregex "^TBX(v8i8Three|v16i8Three)$")>;

def : InstRW<[FalkorWr_5VXVY_7cyc],   (instrs TBLv16i8Four)>;
def : InstRW<[FalkorWr_5VXVY_7cyc],   (instregex "^TBX(v8i8Four|v16i8Four)$")>;

// SIMD Store Instructions
// -----------------------------------------------------------------------------
def : InstRW<[WriteVST],              (instregex "^STP(D|S)(i|post|pre)$")>;

def : InstRW<[WriteVST],                                                        (instregex "^ST1(One(v8b|v4h|v2s|v1d)(_POST)?|(i8|i16|i32|i64)(_POST)?|One(v16b|v8h|v4s|v2d)|Two(v8b|v4h|v2s|v1d))$")>;
def : InstRW<[WriteVST],                                                        (instregex "^ST2(Two(v8b|v4h|v2s|v1d)|(i8|i16|i32|i64))$")>;
def : InstRW<[WriteVST, WriteAdr],                                              (instregex "^ST1(One(v16b|v8h|v4s|v2d)|Two(v8b|v4h|v2s|v1d))_POST$")>;
def : InstRW<[WriteVST, WriteAdr],                                              (instregex "^ST2(Two(v8b|v4h|v2s|v1d)|(i8|i16|i32|i64))_POST$")>;

def : InstRW<[WriteVST, WriteVST],                                              (instregex "^ST1(Two(v16b|v8h|v4s|v2d)|(Three|Four)(v8b|v4h|v2s|v1d))$")>;
def : InstRW<[WriteVST, WriteVST],                                              (instregex "^ST2Two(v16b|v8h|v4s|v2d)$")>;
def : InstRW<[WriteVST, WriteVST],                                              (instregex "^ST3(i8|i16|i32|i64)$")>;
def : InstRW<[WriteVST, WriteVST],                                              (instregex "^ST4(i8|i16|i32|i64)$")>;
def : InstRW<[WriteVST, WriteVST, WriteAdr],                                    (instregex "^ST1(Two(v16b|v8h|v4s|v2d)|(Three|Four)(v8b|v4h|v2s|v1d))_POST$")>;
def : InstRW<[WriteVST, WriteVST, WriteAdr],                                    (instregex "^ST2Two(v16b|v8h|v4s|v2d)_POST$")>;
def : InstRW<[WriteVST, WriteVST, WriteAdr],                                    (instregex "^ST3(i8|i16|i32|i64)_POST$")>;
def : InstRW<[WriteVST, WriteVST, WriteAdr],                                    (instregex "^ST4(i8|i16|i32|i64)_POST$")>;

def : InstRW<[WriteV, WriteVST, WriteVST],                                      (instregex "^ST3Three(v8b|v4h|v2s|v1d)$")>;
def : InstRW<[WriteV, WriteVST, WriteVST, WriteAdr],                            (instregex "^ST3Three(v8b|v4h|v2s|v1d)_POST$")>;

def : InstRW<[WriteVST, WriteVST, WriteVST],                                    (instregex "^ST1Three(v16b|v8h|v4s|v2d)$")>;
def : InstRW<[WriteVST, WriteVST, WriteVST],                                    (instrs ST3Threev2d)>;
def : InstRW<[WriteVST, WriteVST, WriteVST, WriteAdr],                          (instregex "^ST1Three(v16b|v8h|v4s|v2d)_POST$")>;
def : InstRW<[WriteVST, WriteVST, WriteVST, WriteAdr],                          (instrs ST3Threev2d_POST)>;

def : InstRW<[WriteV, WriteV, WriteVST, WriteVST],                              (instregex "^ST4Four(v8b|v4h|v2s|v1d)$")>;
def : InstRW<[WriteV, WriteV, WriteVST, WriteVST, WriteAdr],                    (instregex "^ST4Four(v8b|v4h|v2s|v1d)_POST$")>;

def : InstRW<[WriteVST, WriteVST, WriteVST, WriteVST],                          (instregex "^ST1Four(v16b|v8h|v4s|v2d)$")>;
def : InstRW<[WriteVST, WriteVST, WriteVST, WriteVST],                          (instrs ST4Fourv2d)>;
def : InstRW<[WriteVST, WriteVST, WriteVST, WriteVST, WriteAdr],                (instregex "^ST1Four(v16b|v8h|v4s|v2d)_POST$")>;
def : InstRW<[WriteVST, WriteVST, WriteVST, WriteVST, WriteAdr],                (instrs ST4Fourv2d_POST)>;

def : InstRW<[WriteV, WriteV, WriteVST, WriteVST, WriteVST, WriteVST],          (instregex "^ST3Three(v16b|v8h|v4s)$")>;
def : InstRW<[WriteV, WriteV, WriteVST, WriteVST, WriteVST, WriteVST, WriteAdr],(instregex "^ST3Three(v16b|v8h|v4s)_POST$")>;

def : InstRW<[WriteV, WriteV, WriteV, WriteV, WriteVST, WriteVST, WriteVST, WriteVST],          (instregex "^ST4Four(v16b|v8h|v4s)$")>;
def : InstRW<[WriteV, WriteV, WriteV, WriteV, WriteVST, WriteVST, WriteVST, WriteVST, WriteAdr],(instregex "^ST4Four(v16b|v8h|v4s)_POST$")>;

// Branch Instructions
// -----------------------------------------------------------------------------
def : InstRW<[FalkorWr_1none_0cyc],   (instrs B)>;
def : InstRW<[FalkorWr_1Z_0cyc],      (instregex "^(BR|RET|(CBZ|CBNZ|TBZ|TBNZ)(W|X))$")>;
def : InstRW<[FalkorWr_1ZB_0cyc],     (instrs Bcc)>;
def : InstRW<[FalkorWr_1XYZB_0cyc],   (instrs BL)>;
def : InstRW<[FalkorWr_1Z_1XY_0cyc],  (instrs BLR)>;

// Cryptography Extensions
// -----------------------------------------------------------------------------
def : InstRW<[FalkorWr_1VXVY_1cyc],   (instrs SHA1Hrr)>;
def : InstRW<[FalkorWr_1VXVY_2cyc],   (instrs AESIMCrr, AESMCrr)>;
def : InstRW<[FalkorWr_2VXVY_3cyc],   (instrs AESDrr, AESErr)>;
def : InstRW<[FalkorWr_2VXVY_2cyc],   (instrs SHA1SU0rrr, SHA1SU1rr, SHA256SU0rr)>;
def : InstRW<[FalkorWr_1VX_1VY_4cyc], (instregex "^SHA1(C|M|P)rrr$")>;
def : InstRW<[FalkorWr_1VX_1VY_5cyc], (instrs SHA256H2rrr, SHA256Hrrr)>;
def : InstRW<[FalkorWr_4VXVY_3cyc],   (instrs SHA256SU1rrr)>;

// FP Load Instructions
// -----------------------------------------------------------------------------
def : InstRW<[WriteLD],               (instregex "^LDR((Q|D|S|H|B)ui|(Q|D|S)l)$")>;
def : InstRW<[WriteLD, WriteAdr],     (instregex "^LDR(Q|D|S|H|B)(post|pre)$")>;
def : InstRW<[WriteLD],               (instregex "^LDUR(Q|D|S|H|B)i$")>;
def : InstRW<[FalkorWr_LDR],          (instregex "^LDR(Q|D|H|S|B)ro(W|X)$")>;
def : InstRW<[FalkorWr_2LD_3cyc, WriteLDHi],(instrs LDNPQi)>;
def : InstRW<[FalkorWr_2LD_3cyc, WriteLDHi],(instrs LDPQi)>;
def : InstRW<[FalkorWr_1LD_1none_3cyc, WriteLDHi],(instregex "LDNP(D|S)i$")>;
def : InstRW<[FalkorWr_1LD_1none_3cyc, WriteLDHi],(instregex "LDP(D|S)i$")>;
def : InstRW<[FalkorWr_1LD_1none_3cyc, WriteLDHi, WriteAdr],(instregex "LDP(D|S)(pre|post)$")>;
def : InstRW<[FalkorWr_2LD_3cyc, WriteLDHi, WriteAdr],(instregex "^LDPQ(pre|post)$")>;

// FP Data Processing Instructions
// -----------------------------------------------------------------------------
def : InstRW<[FalkorWr_1VXVY_1cyc],   (instregex "^FCCMP(E)?(H|S|D)rr$")>;
def : InstRW<[FalkorWr_1VXVY_1cyc],   (instregex "^FCMP(E)?(H|S|D)r(r|i)$")>;
def : InstRW<[FalkorWr_1VTOG_1cyc],   (instregex "^FCVT(A|M|N|P)(S|U)U(W|X)(H|S|D)r$")>;
def : InstRW<[FalkorWr_1VXVY_1cyc],   (instregex "^(FABS|FNEG)(H|S|D)r$")>;
def : InstRW<[FalkorWr_1VXVY_1cyc],   (instregex "^FCSEL(H|S|D)rrr$")>;

def : InstRW<[FalkorWr_1VXVY_2cyc],   (instregex "^F(MAX|MIN)(NM)?(H|S|D)rr$")>;
def : InstRW<[FalkorWr_1VXVY_2cyc],   (instregex "^F(MAX|MIN)(NM)?Pv2i(16|32|64)p$")>;
def : InstRW<[FalkorWr_1VXVY_2cyc],   (instrs FCVTHSr, FCVTHDr)>;
def : InstRW<[FalkorWr_1VXVY_2cyc],   (instregex "^FRINT(A|I|M|N|P|X|Z)(H|S|D)r$")>;

def : InstRW<[FalkorWr_1VXVY_3cyc],   (instregex "^FABD(16|32|64)$")>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instregex "^(FADD|FSUB)(H|S|D)rr$")>;
def : InstRW<[FalkorWr_1VXVY_3cyc],   (instrs FCVTSHr, FCVTDHr)>;

def : InstRW<[FalkorWr_1VXVY_4cyc],   (instrs FCVTSDr, FCVTDSr)>;

def : InstRW<[FalkorWr_1VXVY_5cyc],   (instregex "^F(N)?MUL(H|S)rr$")>;

def : InstRW<[FalkorWr_1VXVY_6cyc],   (instregex "^F(N)?MULDrr$")>;

def : InstRW<[FalkorWr_1VX_1VY_10cyc],(instregex "^FDIV(H|S|D)rr$")>;
def : InstRW<[FalkorWr_1VX_1VY_2cyc], (instregex "^FSQRT(H|S|D)r$")>;

def : InstRW<[FalkorWr_1VXVY_5cyc, FalkorReadFMA],(instregex "^F(N)?M(ADD|SUB)(H|S)rrr$")>;
def : InstRW<[FalkorWr_1VXVY_6cyc, FalkorReadFMA],(instregex "^F(N)?M(ADD|SUB)Drrr$")>;

// FP Miscellaneous Instructions
// -----------------------------------------------------------------------------
def : InstRW<[FalkorWr_FMOV],         (instregex "^FMOV(HW|HX|SW|DX|DXHigh)r$")>;
def : InstRW<[FalkorWr_1VTOG_1cyc],   (instregex "^FCVTZ(S|U)(S|U)(W|X)(D|S)ri?$")>;
def : InstRW<[FalkorWr_1VTOG_1cyc],   (instregex "^FMOV(WH|WS|XH|XD|XDHigh)r$")>;
def : InstRW<[FalkorWr_1VXVY_1cyc],   (instregex "^FMOV(Hi|Hr|S0|Si|Sr|D0|Di|Dr|v.*_ns)$")>;

def : InstRW<[FalkorWr_1GTOV_4cyc],   (instregex "^(S|U)CVTF(S|U)(W|X)(D|S)ri$")>;
def : InstRW<[FalkorWr_1VXVY_4cyc],   (instregex "^(S|U)CVTF(v1i16|v1i32|v2i32|v1i64|v4i16|v2f32|v4f16|d|s)(_shift)?")>;

def : InstRW<[FalkorWr_2VXVY_4cyc],   (instregex "^(S|U)CVTF(v2i64|v4i32|v8i16|v2f64|v4f32|v8f16)(_shift)?")>;

// Load Instructions
// -----------------------------------------------------------------------------
def : InstRW<[FalkorWr_1ST_0cyc],     (instrs PRFMui, PRFMl)>;
def : InstRW<[FalkorWr_1ST_0cyc],     (instrs PRFUMi)>;

def : InstRW<[WriteLD, WriteLDHi],    (instregex "^LDNP(W|X)i$")>;
def : InstRW<[WriteLD, WriteLDHi],    (instregex "^LDP(W|X)i$")>;
def : InstRW<[FalkorWr_1LD_3cyc],     (instregex "^LDR(B|H|W|X)ui$")>;
def : InstRW<[WriteLD, WriteAdr],     (instregex "^LDR(B|H|W|X)(post|pre)$")>;
def : InstRW<[FalkorWr_1LD_3cyc],     (instregex "^LDR(W|X)l$")>;
def : InstRW<[FalkorWr_1LD_3cyc],     (instregex "^LDTR(B|H|W|X)i$")>;
def : InstRW<[FalkorWr_1LD_3cyc],     (instregex "^LDUR(B|H|W|X)i$")>;

def : InstRW<[FalkorWr_1LD_4cyc],     (instregex "^LDRS(BW|BX|HW|HX|W)ui$")>;
def : InstRW<[FalkorWr_1LD_4cyc],     (instrs LDRSWl)>;
def : InstRW<[FalkorWr_1LD_4cyc],     (instregex "^LDTRS(BW|BX|HW|HX|W)i$")>;
def : InstRW<[FalkorWr_1LD_4cyc],     (instregex "^LDURS(BW|BX|HW|HX|W)i$")>;

def : InstRW<[FalkorWr_PRFM],         (instregex "^PRFMro(W|X)$")>;
def : InstRW<[FalkorWr_LDR],          (instregex "^LDR(B|H|W|X)ro(W|X)$")>;

def : InstRW<[FalkorWr_LDRS],         (instregex "^LDRS(BW|BX|HW|HX|W)ro(W|X)$")>;

def : InstRW<[FalkorWr_1LD_4cyc, WriteAdr],(instregex "^LDRS(BW|BX|HW|HX|W)(post|pre)$")>;
def : InstRW<[WriteLD, WriteLDHi, WriteAdr],(instregex "^LDP(W|X)(post|pre)$")>;
def : InstRW<[FalkorWr_1LD_4cyc, WriteLDHi],(instrs LDPSWi)>;
def : InstRW<[FalkorWr_1LD_4cyc, WriteLDHi, WriteAdr],(instregex "^LDPSW(post|pre)$")>;

// Miscellaneous Data-Processing Instructions
// -----------------------------------------------------------------------------
def : InstRW<[FalkorWr_1XYZ_1cyc],    (instregex "^(S|U)?BFM(W|X)ri$")>;
def : InstRW<[FalkorWr_1X_2cyc],      (instregex "^CRC32.*$")>;
def : InstRW<[FalkorWr_1XYZ_2cyc],    (instregex "^(CLS|CLZ|RBIT|REV|REV16|REV32)(W|X)r$")>;
def : InstRW<[FalkorWr_2XYZ_2cyc],    (instregex "^EXTR(W|X)rri$")>;

// Divide and Multiply Instructions
// -----------------------------------------------------------------------------
def : InstRW<[FalkorWr_1X_4cyc],      (instregex "^(S|U)M(ADD|SUB)Lrrr$")>;
def : InstRW<[FalkorWr_1X_4cyc],      (instregex "^M(ADD|SUB)Wrrr$")>;

def : InstRW<[FalkorWr_1X_5cyc],      (instregex "^(S|U)MULHrr$")>;
def : InstRW<[FalkorWr_1X_5cyc],      (instregex "^M(ADD|SUB)Xrrr$")>;

def : InstRW<[FalkorWr_1X_1Z_8cyc],   (instregex "^(S|U)DIVWr$")>;
def : InstRW<[FalkorWr_1X_1Z_16cyc],  (instregex "^(S|U)DIVXr$")>;

def : InstRW<[FalkorWr_2VXVY_4cyc],   (instregex "^(S|U)(MLAL|MLSL|MULL)v.*$")>;

// Move and Shift Instructions
// -----------------------------------------------------------------------------
def : InstRW<[FalkorWr_1XYZ_1cyc],    (instregex "^(LSLV|LSRV|ASRV|RORV|MOVK)(W|X).*")>;
def : InstRW<[FalkorWr_1XYZB_1cyc],   (instregex "^ADRP?$")>;
def : InstRW<[FalkorWr_1XYZB_1cyc],   (instregex "^MOVN(W|X)i$")>;
def : InstRW<[FalkorWr_MOVZ],         (instregex "^MOVZ(W|X)i$")>;

// Other Instructions
// -----------------------------------------------------------------------------
def : InstRW<[FalkorWr_1LD_0cyc],     (instrs CLREX, DMB, DSB)>;
def : InstRW<[FalkorWr_1none_0cyc],   (instrs BRK, DCPS1, DCPS2, DCPS3, HINT, HLT, HVC, ISB, SMC, SVC)>;
def : InstRW<[FalkorWr_1ST_0cyc],     (instrs SYSxt, SYSLxt)>;
def : InstRW<[FalkorWr_1Z_0cyc],      (instrs MSRpstateImm1, MSRpstateImm4)>;

def : InstRW<[FalkorWr_1LD_3cyc],     (instregex "^(LDAR(B|H|W|X)|LDAXP(W|X)|LDAXR(B|H|W|X)|LDXP(W|X)|LDXR(B|H|W|X))$")>;
def : InstRW<[FalkorWr_1LD_3cyc],     (instrs MRS)>;

def : InstRW<[FalkorWr_1LD_1Z_3cyc],  (instrs DRPS)>;

def : InstRW<[FalkorWr_1SD_1ST_0cyc], (instrs MSR)>;
def : InstRW<[WriteVST],              (instrs STNPDi, STNPSi)>;
def : InstRW<[WriteSTP],               (instrs STNPWi, STNPXi)>;
def : InstRW<[FalkorWr_2LD_1Z_3cyc],  (instrs ERET)>;

def : InstRW<[WriteST],               (instregex "^LDC.*$")>;
def : InstRW<[WriteST],               (instregex "^STLR(B|H|W|X)$")>;
def : InstRW<[WriteST],               (instregex "^STXP(W|X)$")>;
def : InstRW<[WriteST],               (instregex "^STXR(B|H|W|X)$")>;

def : InstRW<[WriteSTX],              (instregex "^STLXP(W|X)$")>;
def : InstRW<[WriteSTX],              (instregex "^STLXR(B|H|W|X)$")>;
def : InstRW<[WriteVST, WriteVST],    (instrs STNPQi)>;

// Store Instructions
// -----------------------------------------------------------------------------
def : InstRW<[WriteST],               (instregex "^STP(W|X)(i|post|pre)$")>;
def : InstRW<[WriteST],               (instregex "^STR(Q|D|S|BB|HH)ui$")>;
def : InstRW<[WriteST],               (instregex "^STUR(Q|D|S|BB|HH)i$")>;
def : InstRW<[WriteST],               (instregex "^STR(B|H|W|X)(post|pre|ui)$")>;
def : InstRW<[WriteST],               (instregex "^STTR(B|H|W|X)i$")>;
def : InstRW<[WriteST],               (instregex "^STUR(B|H|W|X)i$")>;

def : InstRW<[WriteST, WriteAdr],     (instregex "^STR(B|H|W|X)ro(W|X)$")>;

def : InstRW<[WriteVST, WriteVST],    (instregex "^STPQ(i|post|pre)$")>;
