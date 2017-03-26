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
def : InstRW<[FalkorWr_1VXVY_4cyc],   (instregex "^ML(A|S)(v8i8|v4i16|v2i32)(_indexed)?$")>;

def : InstRW<[FalkorWr_1VXVY_5cyc],   (instregex "^(FML(A|S)|FMUL|FMULX)(v2f32|v4f16|(v1i16_indexed|v4i16_indexed|v1i32_indexed|v2i32_indexed))$")>;
def : InstRW<[FalkorWr_1VXVY_5cyc],   (instrs FMULX16, FMULX32)>;

def : InstRW<[FalkorWr_1VXVY_6cyc],   (instregex "^(FML(A|S)|FMUL|FMULX)v1i64_indexed$")>;
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
def : InstRW<[FalkorWr_2VXVY_4cyc],   (instregex "^ML(A|S)(v16i8|v8i16|v4i32|v2i64)(_indexed)?$")>;

def : InstRW<[FalkorWr_2VXVY_5cyc],   (instregex "^(FML(A|S)|FMUL|FMULX)(v2f64|v4f32|v8f16|v8i16_indexed|v4i32_indexed)$")>;

def : InstRW<[FalkorWr_2VXVY_6cyc],   (instregex "^(FML(A|S)|FMUL|FMULX)v2i64_indexed$")>;

def : InstRW<[FalkorWr_3VXVY_4cyc],   (instregex "^(FCVTX?N|FCVTX?N2)(v1i32|v1i64|v1f16|v2f32|v4f16)$")>;

def : InstRW<[FalkorWr_3VXVY_5cyc],   (instregex "^(FCVTX?N|FCVTX?N2)(v2i32|v4i16|v8i16)$")>;

def : InstRW<[FalkorWr_2VX_2VY_2cyc], (instregex "^(FDIV|FSQRT)(v2f64|v4f32|v8f16)$")>;

// SIMD Load Instructions
// -----------------------------------------------------------------------------
def : InstRW<[WriteVLD],                (instregex "LD1i(8|16|32|64)$")>;
def : InstRW<[WriteVLD],                (instregex "LD1Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[FalkorWr_1LD_1VXVY_4cyc], (instregex "LD1Onev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[FalkorWr_2LD_3cyc],       (instregex "LD1Twov(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[FalkorWr_3LD_3cyc],       (instregex "LD1Threev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[FalkorWr_4LD_3cyc],       (instregex "LD1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLD, WriteAdr],                (instregex "LD1i(8|16|32|64)_POST$")>;
def : InstRW<[WriteVLD, WriteAdr],                (instregex "LD1Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[FalkorWr_1LD_1VXVY_4cyc, WriteAdr], (instregex "LD1Onev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[FalkorWr_2LD_3cyc, WriteAdr],       (instregex "LD1Twov(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[FalkorWr_3LD_3cyc, WriteAdr],       (instregex "LD1Threev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[FalkorWr_4LD_3cyc, WriteAdr],       (instregex "LD1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

def : InstRW<[WriteVLD], (instregex "LD2i(8|16|32|64)$")>;
def : InstRW<[WriteVLD], (instregex "LD2Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLD], (instregex "LD2Twov(8b|4h|2s)$")>;
def : InstRW<[WriteVLD], (instregex "LD2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD2i(8|16|32|64)(_POST)?$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD2Rv(8b|4h|2s|1d|16b|8h|4s|2d)(_POST)?$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD2Twov(8b|4h|2s)(_POST)?$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD2Twov(16b|8h|4s|2d)(_POST)?$")>;

def : InstRW<[WriteVLD], (instregex "LD3i(8|16|32|64)$")>;
def : InstRW<[WriteVLD], (instregex "LD3Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLD], (instregex "LD3Threev(8b|4h|2s|1d|16b|8h|4s)$")>;
def : InstRW<[WriteVLD], (instregex "LD3Threev(2d)$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD3i(8|16|32|64)_POST$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD3Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD3Threev(8b|4h|2s|1d|16b|8h|4s)_POST$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD3Threev(2d)_POST$")>;

def : InstRW<[WriteVLD], (instregex "LD4i(8|16|32|64)$")>;
def : InstRW<[WriteVLD], (instregex "LD4Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLD], (instregex "LD4Fourv(8b|4h|2s|1d|16b|8h|4s)$")>;
def : InstRW<[WriteVLD], (instregex "LD4Fourv(2d)$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD4i(8|16|32|64)_POST$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD4Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD4Fourv(8b|4h|2s|1d|16b|8h|4s)_POST$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD4Fourv(2d)_POST$")>;

// SIMD Store Instructions
// -----------------------------------------------------------------------------
def : InstRW<[WriteVST], (instregex "ST1i(8|16|32|64)$")>;
def : InstRW<[WriteVST], (instregex "ST1Onev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVST], (instregex "ST1Twov(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVST], (instregex "ST1Threev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVST], (instregex "ST1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST1i(8|16|32|64)_POST$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST1Onev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST1Twov(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST1Threev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

def : InstRW<[WriteVST], (instregex "ST2i(8|16|32|64)$")>;
def : InstRW<[WriteVST], (instregex "ST2Twov(8b|4h|2s)$")>;
def : InstRW<[WriteVST], (instregex "ST2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST2i(8|16|32|64)_POST$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST2Twov(8b|4h|2s)_POST$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST2Twov(16b|8h|4s|2d)_POST$")>;

def : InstRW<[WriteVST], (instregex "ST3i(8|16|32|64)$")>;
def : InstRW<[WriteVST], (instregex "ST3Threev(8b|4h|2s|1d|16b|8h|4s)$")>;
def : InstRW<[WriteVST], (instregex "ST3Threev(2d)$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST3i(8|16|32|64)_POST$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST3Threev(8b|4h|2s|1d|16b|8h|4s)_POST$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST3Threev(2d)_POST$")>;

def : InstRW<[WriteVST], (instregex "ST4i(8|16|32|64)$")>;
def : InstRW<[WriteVST], (instregex "ST4Fourv(8b|4h|2s|1d|16b|8h|4s)$")>;
def : InstRW<[WriteVST], (instregex "ST4Fourv(2d)$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST4i(8|16|32|64)_POST$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST4Fourv(8b|4h|2s|1d|16b|8h|4s)_POST$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST4Fourv(2d)_POST$")>;
