//=- HexagonInstrInfoV4.td - Target Desc. for Hexagon Target -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Hexagon V4 instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

let neverHasSideEffects = 1 in
def IMMEXT : Immext<(outs), (ins),
                    "/* immext #... */",
                    []>,
             Requires<[HasV4T]>;

// Hexagon V4 Architecture spec defines 8 instruction classes:
// LD ST ALU32 XTYPE J JR MEMOP NV CR SYSTEM(system is not implemented in the
// compiler)

// LD Instructions:
// ========================================
// Loads (8/16/32/64 bit)
// Deallocframe

// ST Instructions:
// ========================================
// Stores (8/16/32/64 bit)
// Allocframe

// ALU32 Instructions:
// ========================================
// Arithmetic / Logical (32 bit)
// Vector Halfword

// XTYPE Instructions (32/64 bit):
// ========================================
// Arithmetic, Logical, Bit Manipulation
// Multiply (Integer, Fractional, Complex)
// Permute / Vector Permute Operations
// Predicate Operations
// Shift / Shift with Add/Sub/Logical
// Vector Byte ALU
// Vector Halfword (ALU, Shift, Multiply)
// Vector Word (ALU, Shift)

// J Instructions:
// ========================================
// Jump/Call PC-relative

// JR Instructions:
// ========================================
// Jump/Call Register

// MEMOP Instructions:
// ========================================
// Operation on memory (8/16/32 bit)

// NV Instructions:
// ========================================
// New-value Jumps
// New-value Stores

// CR Instructions:
// ========================================
// Control-Register Transfers
// Hardware Loop Setup
// Predicate Logicals & Reductions

// SYSTEM Instructions (not implemented in the compiler):
// ========================================
// Prefetch
// Cache Maintenance
// Bus Operations


//===----------------------------------------------------------------------===//
// ALU32 +
//===----------------------------------------------------------------------===//

// Shift halfword.

let isPredicated = 1 in
def ASLH_cPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1) $dst = aslh($src2)",
            []>,
            Requires<[HasV4T]>;

let isPredicated = 1 in
def ASLH_cNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1) $dst = aslh($src2)",
            []>,
            Requires<[HasV4T]>;

let isPredicated = 1 in
def ASLH_cdnPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1.new) $dst = aslh($src2)",
            []>,
            Requires<[HasV4T]>;

let isPredicated = 1 in
def ASLH_cdnNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1.new) $dst = aslh($src2)",
            []>,
            Requires<[HasV4T]>;

let isPredicated = 1 in
def ASRH_cPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1) $dst = asrh($src2)",
            []>,
            Requires<[HasV4T]>;

let isPredicated = 1 in
def ASRH_cNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1) $dst = asrh($src2)",
            []>,
            Requires<[HasV4T]>;

let isPredicated = 1 in
def ASRH_cdnPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1.new) $dst = asrh($src2)",
            []>,
            Requires<[HasV4T]>;

let isPredicated = 1 in
def ASRH_cdnNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1.new) $dst = asrh($src2)",
            []>,
            Requires<[HasV4T]>;

// Sign extend.

let isPredicated = 1 in
def SXTB_cPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1) $dst = sxtb($src2)",
            []>,
            Requires<[HasV4T]>;

let isPredicated = 1 in
def SXTB_cNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1) $dst = sxtb($src2)",
            []>,
            Requires<[HasV4T]>;

let isPredicated = 1 in
def SXTB_cdnPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1.new) $dst = sxtb($src2)",
            []>,
            Requires<[HasV4T]>;

let isPredicated = 1 in
def SXTB_cdnNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1.new) $dst = sxtb($src2)",
            []>,
            Requires<[HasV4T]>;


let isPredicated = 1 in
def SXTH_cPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1) $dst = sxth($src2)",
            []>,
            Requires<[HasV4T]>;

let isPredicated = 1 in
def SXTH_cNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1) $dst = sxth($src2)",
            []>,
            Requires<[HasV4T]>;

let isPredicated = 1 in
def SXTH_cdnPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1.new) $dst = sxth($src2)",
            []>,
            Requires<[HasV4T]>;

let isPredicated = 1 in
def SXTH_cdnNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1.new) $dst = sxth($src2)",
            []>,
            Requires<[HasV4T]>;

// Zero exten.

let neverHasSideEffects = 1, isPredicated = 1 in
def ZXTB_cPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1) $dst = zxtb($src2)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def ZXTB_cNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1) $dst = zxtb($src2)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def ZXTB_cdnPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1.new) $dst = zxtb($src2)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def ZXTB_cdnNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1.new) $dst = zxtb($src2)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def ZXTH_cPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1) $dst = zxth($src2)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def ZXTH_cNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1) $dst = zxth($src2)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def ZXTH_cdnPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1.new) $dst = zxth($src2)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def ZXTH_cdnNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1.new) $dst = zxth($src2)",
            []>,
            Requires<[HasV4T]>;

// Generate frame index addresses.
let neverHasSideEffects = 1, isReMaterializable = 1 in
def TFR_FI_immext_V4 : ALU32_ri<(outs IntRegs:$dst),
            (ins IntRegs:$src1, s32Imm:$offset),
            "$dst = add($src1, ##$offset)",
            []>,
            Requires<[HasV4T]>;


//===----------------------------------------------------------------------===//
// ALU32 -
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// ALU32/PERM +
//===----------------------------------------------------------------------===//

// Combine
// Rdd=combine(Rs, #s8)
let neverHasSideEffects = 1 in
def COMBINE_ri_V4 : ALU32_ri<(outs DoubleRegs:$dst),
            (ins IntRegs:$src1, s8Imm:$src2),
            "$dst = combine($src1, #$src2)",
            []>,
            Requires<[HasV4T]>;
// Rdd=combine(#s8, Rs)
let neverHasSideEffects = 1 in
def COMBINE_ir_V4 : ALU32_ir<(outs DoubleRegs:$dst),
            (ins s8Imm:$src1, IntRegs:$src2),
            "$dst = combine(#$src1, $src2)",
            []>,
            Requires<[HasV4T]>;
//===----------------------------------------------------------------------===//
// ALU32/PERM +
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// LD +
//===----------------------------------------------------------------------===//
//
// These absolute set addressing mode instructions accept immediate as
// an operand. We have duplicated these patterns to take global address.

let neverHasSideEffects = 1 in
def LDrid_abs_setimm_V4 : LDInst2<(outs DoubleRegs:$dst1, IntRegs:$dst2),
            (ins u6Imm:$addr),
            "$dst1 = memd($dst2=#$addr)",
            []>,
            Requires<[HasV4T]>;

// Rd=memb(Re=#U6)
let neverHasSideEffects = 1 in
def LDrib_abs_setimm_V4 : LDInst2<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins u6Imm:$addr),
            "$dst1 = memb($dst2=#$addr)",
            []>,
            Requires<[HasV4T]>;

// Rd=memh(Re=#U6)
let neverHasSideEffects = 1 in
def LDrih_abs_setimm_V4 : LDInst2<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins u6Imm:$addr),
            "$dst1 = memh($dst2=#$addr)",
            []>,
            Requires<[HasV4T]>;

// Rd=memub(Re=#U6)
let neverHasSideEffects = 1 in
def LDriub_abs_setimm_V4 : LDInst2<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins u6Imm:$addr),
            "$dst1 = memub($dst2=#$addr)",
            []>,
            Requires<[HasV4T]>;

// Rd=memuh(Re=#U6)
let neverHasSideEffects = 1 in
def LDriuh_abs_setimm_V4 : LDInst2<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins u6Imm:$addr),
            "$dst1 = memuh($dst2=#$addr)",
            []>,
            Requires<[HasV4T]>;

// Rd=memw(Re=#U6)
let neverHasSideEffects = 1 in
def LDriw_abs_setimm_V4 : LDInst2<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins u6Imm:$addr),
            "$dst1 = memw($dst2=#$addr)",
            []>,
            Requires<[HasV4T]>;

// Following patterns are defined for absolute set addressing mode
// instruction which take global address as operand.
let neverHasSideEffects = 1 in
def LDrid_abs_set_V4 : LDInst2<(outs DoubleRegs:$dst1, IntRegs:$dst2),
            (ins globaladdress:$addr),
            "$dst1 = memd($dst2=##$addr)",
            []>,
            Requires<[HasV4T]>;

// Rd=memb(Re=#U6)
let neverHasSideEffects = 1 in
def LDrib_abs_set_V4 : LDInst2<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins globaladdress:$addr),
            "$dst1 = memb($dst2=##$addr)",
            []>,
            Requires<[HasV4T]>;

// Rd=memh(Re=#U6)
let neverHasSideEffects = 1 in
def LDrih_abs_set_V4 : LDInst2<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins globaladdress:$addr),
            "$dst1 = memh($dst2=##$addr)",
            []>,
            Requires<[HasV4T]>;

// Rd=memub(Re=#U6)
let neverHasSideEffects = 1 in
def LDriub_abs_set_V4 : LDInst2<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins globaladdress:$addr),
            "$dst1 = memub($dst2=##$addr)",
            []>,
            Requires<[HasV4T]>;

// Rd=memuh(Re=#U6)
let neverHasSideEffects = 1 in
def LDriuh_abs_set_V4 : LDInst2<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins globaladdress:$addr),
            "$dst1 = memuh($dst2=##$addr)",
            []>,
            Requires<[HasV4T]>;

// Rd=memw(Re=#U6)
let neverHasSideEffects = 1 in
def LDriw_abs_set_V4 : LDInst2<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins globaladdress:$addr),
            "$dst1 = memw($dst2=##$addr)",
            []>,
            Requires<[HasV4T]>;

// Load doubleword.
//
// Make sure that in post increment load, the first operand is always the post
// increment operand.
//
// Rdd=memd(Rs+Rt<<#u2)
// Special case pattern for indexed load without offset which is easier to
// match. AddedComplexity of this pattern should be lower than base+offset load
// and lower yet than the more generic version with offset/shift below
// Similar approach is taken for all other base+index loads.
let AddedComplexity = 10, isPredicable = 1 in
def LDrid_indexed_V4 : LDInst<(outs DoubleRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2),
                    "$dst=memd($src1+$src2<<#0)",
                    [(set (i64 DoubleRegs:$dst),
                          (i64 (load (add (i32 IntRegs:$src1),
                                          (i32 IntRegs:$src2)))))]>,
                    Requires<[HasV4T]>;

let AddedComplexity = 40, isPredicable = 1 in
def LDrid_indexed_shl_V4 : LDInst<(outs DoubleRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$offset),
                    "$dst=memd($src1+$src2<<#$offset)",
                    [(set (i64 DoubleRegs:$dst),
                          (i64 (load (add (i32 IntRegs:$src1),
                                          (shl (i32 IntRegs:$src2),
                                               u2ImmPred:$offset)))))]>,
                    Requires<[HasV4T]>;

//// Load doubleword conditionally.
// if ([!]Pv[.new]) Rd=memd(Rs+Rt<<#u2)
// if (Pv) Rd=memd(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDrid_indexed_cPt_V4 : LDInst2<(outs DoubleRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1) $dst=memd($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memd(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDrid_indexed_cdnPt_V4 : LDInst2<(outs DoubleRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1.new) $dst=memd($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memd(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDrid_indexed_cNotPt_V4 : LDInst2<(outs DoubleRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1) $dst=memd($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memd(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDrid_indexed_cdnNotPt_V4 : LDInst2<(outs DoubleRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1.new) $dst=memd($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv) Rd=memd(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDrid_indexed_shl_cPt_V4 : LDInst2<(outs DoubleRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1) $dst=memd($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memd(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDrid_indexed_shl_cdnPt_V4 : LDInst2<(outs DoubleRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1.new) $dst=memd($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memd(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDrid_indexed_shl_cNotPt_V4 : LDInst2<(outs DoubleRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1) $dst=memd($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memd(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDrid_indexed_shl_cdnNotPt_V4 : LDInst2<(outs DoubleRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1.new) $dst=memd($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// Rdd=memd(Rt<<#u2+#U6)

//// Load byte.
// Rd=memb(Rs+Rt<<#u2)
let AddedComplexity = 10, isPredicable = 1 in
def LDrib_indexed_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2),
                    "$dst=memb($src1+$src2<<#0)",
                    [(set (i32 IntRegs:$dst),
                          (i32 (sextloadi8 (add (i32 IntRegs:$src1),
                                                (i32 IntRegs:$src2)))))]>,
                    Requires<[HasV4T]>;

let AddedComplexity = 10, isPredicable = 1 in
def LDriub_indexed_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2),
                    "$dst=memub($src1+$src2<<#0)",
                    [(set (i32 IntRegs:$dst),
                          (i32 (zextloadi8 (add (i32 IntRegs:$src1),
                                                (i32 IntRegs:$src2)))))]>,
                    Requires<[HasV4T]>;

let AddedComplexity = 10, isPredicable = 1 in
def LDriub_ae_indexed_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2),
                    "$dst=memub($src1+$src2<<#0)",
                    [(set (i32 IntRegs:$dst),
                          (i32 (extloadi8 (add (i32 IntRegs:$src1),
                                               (i32 IntRegs:$src2)))))]>,
                    Requires<[HasV4T]>;

let AddedComplexity = 40, isPredicable = 1 in
def LDrib_indexed_shl_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$offset),
                    "$dst=memb($src1+$src2<<#$offset)",
                    [(set (i32 IntRegs:$dst),
                          (i32 (sextloadi8 (add (i32 IntRegs:$src1),
                                                (shl (i32 IntRegs:$src2),
                                                     u2ImmPred:$offset)))))]>,
                    Requires<[HasV4T]>;

let AddedComplexity = 40, isPredicable = 1 in
def LDriub_indexed_shl_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$offset),
                    "$dst=memub($src1+$src2<<#$offset)",
                    [(set (i32 IntRegs:$dst),
                          (i32 (zextloadi8 (add (i32 IntRegs:$src1),
                                                (shl (i32 IntRegs:$src2),
                                                     u2ImmPred:$offset)))))]>,
                    Requires<[HasV4T]>;

let AddedComplexity = 40, isPredicable = 1 in
def LDriub_ae_indexed_shl_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$offset),
                    "$dst=memub($src1+$src2<<#$offset)",
                    [(set (i32 IntRegs:$dst),
                          (i32 (extloadi8 (add (i32 IntRegs:$src1),
                                               (shl (i32 IntRegs:$src2),
                                                    u2ImmPred:$offset)))))]>,
                    Requires<[HasV4T]>;

//// Load byte conditionally.
// if ([!]Pv[.new]) Rd=memb(Rs+Rt<<#u2)
// if (Pv) Rd=memb(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDrib_indexed_cPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1) $dst=memb($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memb(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDrib_indexed_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1.new) $dst=memb($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memb(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDrib_indexed_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1) $dst=memb($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memb(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDrib_indexed_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1.new) $dst=memb($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv) Rd=memb(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDrib_indexed_shl_cPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1) $dst=memb($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memb(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDrib_indexed_shl_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1.new) $dst=memb($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memb(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDrib_indexed_shl_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1) $dst=memb($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memb(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDrib_indexed_shl_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1.new) $dst=memb($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

//// Load unsigned byte conditionally.
// if ([!]Pv[.new]) Rd=memub(Rs+Rt<<#u2)
// if (Pv) Rd=memub(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDriub_indexed_cPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1) $dst=memub($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memub(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDriub_indexed_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1.new) $dst=memub($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memub(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDriub_indexed_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1) $dst=memub($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memub(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDriub_indexed_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1.new) $dst=memub($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv) Rd=memub(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDriub_indexed_shl_cPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1) $dst=memub($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memub(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDriub_indexed_shl_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1.new) $dst=memub($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memub(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDriub_indexed_shl_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1) $dst=memub($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memub(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDriub_indexed_shl_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1.new) $dst=memub($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// Rd=memb(Rt<<#u2+#U6)

//// Load halfword
// Rd=memh(Rs+Rt<<#u2)
let AddedComplexity = 10, isPredicable = 1 in
def LDrih_indexed_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2),
                    "$dst=memh($src1+$src2<<#0)",
                    [(set (i32 IntRegs:$dst),
                          (i32 (sextloadi16 (add (i32 IntRegs:$src1),
                                                 (i32 IntRegs:$src2)))))]>,
                    Requires<[HasV4T]>;

let AddedComplexity = 10, isPredicable = 1 in
def LDriuh_indexed_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2),
                    "$dst=memuh($src1+$src2<<#0)",
                    [(set (i32 IntRegs:$dst),
                          (i32 (zextloadi16 (add (i32 IntRegs:$src1),
                                                 (i32 IntRegs:$src2)))))]>,
                    Requires<[HasV4T]>;

let AddedComplexity = 10, isPredicable = 1 in
def LDriuh_ae_indexed_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2),
                    "$dst=memuh($src1+$src2<<#0)",
                    [(set (i32 IntRegs:$dst),
                          (i32 (extloadi16 (add (i32 IntRegs:$src1),
                                                (i32 IntRegs:$src2)))))]>,
                    Requires<[HasV4T]>;

// Rd=memh(Rs+Rt<<#u2)
let AddedComplexity = 40, isPredicable = 1 in
def LDrih_indexed_shl_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$offset),
                    "$dst=memh($src1+$src2<<#$offset)",
                    [(set (i32 IntRegs:$dst),
                          (i32 (sextloadi16 (add (i32 IntRegs:$src1),
                                                 (shl (i32 IntRegs:$src2),
                                                      u2ImmPred:$offset)))))]>,
                    Requires<[HasV4T]>;

let AddedComplexity = 40, isPredicable = 1 in
def LDriuh_indexed_shl_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$offset),
                    "$dst=memuh($src1+$src2<<#$offset)",
                    [(set (i32 IntRegs:$dst),
                          (i32 (zextloadi16 (add (i32 IntRegs:$src1),
                                                 (shl (i32 IntRegs:$src2),
                                                      u2ImmPred:$offset)))))]>,
                    Requires<[HasV4T]>;

let AddedComplexity = 40, isPredicable = 1 in
def LDriuh_ae_indexed_shl_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$offset),
                    "$dst=memuh($src1+$src2<<#$offset)",
                    [(set (i32 IntRegs:$dst),
                          (i32 (extloadi16 (add (i32 IntRegs:$src1),
                                                (shl (i32 IntRegs:$src2),
                                                     u2ImmPred:$offset)))))]>,
                    Requires<[HasV4T]>;

//// Load halfword conditionally.
// if ([!]Pv[.new]) Rd=memh(Rs+Rt<<#u2)
// if (Pv) Rd=memh(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDrih_indexed_cPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1) $dst=memh($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memh(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDrih_indexed_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1.new) $dst=memh($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memh(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDrih_indexed_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1) $dst=memh($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memh(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDrih_indexed_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1.new) $dst=memh($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv) Rd=memh(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDrih_indexed_shl_cPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1) $dst=memh($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memh(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDrih_indexed_shl_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1.new) $dst=memh($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memh(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDrih_indexed_shl_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1) $dst=memh($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memh(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDrih_indexed_shl_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1.new) $dst=memh($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

//// Load unsigned halfword conditionally.
// if ([!]Pv[.new]) Rd=memuh(Rs+Rt<<#u2)
// if (Pv) Rd=memuh(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDriuh_indexed_cPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1) $dst=memuh($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memuh(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDriuh_indexed_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1.new) $dst=memuh($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memuh(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDriuh_indexed_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1) $dst=memuh($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memuh(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDriuh_indexed_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1.new) $dst=memuh($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv) Rd=memuh(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDriuh_indexed_shl_cPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1) $dst=memuh($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memuh(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDriuh_indexed_shl_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1.new) $dst=memuh($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memuh(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDriuh_indexed_shl_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1) $dst=memuh($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memuh(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDriuh_indexed_shl_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1.new) $dst=memuh($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// Rd=memh(Rt<<#u2+#U6)

//// Load word.
// Load predicate: Fix for bug 5279.
let neverHasSideEffects = 1 in
def LDriw_pred_V4 : LDInst2<(outs PredRegs:$dst),
            (ins MEMri:$addr),
            "Error; should not emit",
            []>,
            Requires<[HasV4T]>;

// Rd=memw(Re=#U6)

// Rd=memw(Rs+Rt<<#u2)
let AddedComplexity = 10, isPredicable = 1 in
def LDriw_indexed_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2),
                    "$dst=memw($src1+$src2<<#0)",
                    [(set (i32 IntRegs:$dst),
                          (i32 (load (add (i32 IntRegs:$src1),
                                          (i32 IntRegs:$src2)))))]>,
                    Requires<[HasV4T]>;

// Rd=memw(Rs+Rt<<#u2)
let AddedComplexity = 40, isPredicable = 1 in
def LDriw_indexed_shl_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$offset),
                    "$dst=memw($src1+$src2<<#$offset)",
                    [(set (i32 IntRegs:$dst),
                          (i32 (load (add (i32 IntRegs:$src1),
                                          (shl (i32 IntRegs:$src2),
                                               u2ImmPred:$offset)))))]>,
                    Requires<[HasV4T]>;

//// Load word conditionally.
// if ([!]Pv[.new]) Rd=memw(Rs+Rt<<#u2)
// if (Pv) Rd=memw(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDriw_indexed_cPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1) $dst=memw($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memh(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDriw_indexed_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1.new) $dst=memw($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memh(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDriw_indexed_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1) $dst=memw($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memh(Rs+Rt<<#u2)
let AddedComplexity = 15, isPredicated = 1 in
def LDriw_indexed_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1.new) $dst=memw($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv) Rd=memh(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDriw_indexed_shl_cPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1) $dst=memw($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memh(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDriw_indexed_shl_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1.new) $dst=memw($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memh(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDriw_indexed_shl_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1) $dst=memw($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memh(Rs+Rt<<#u2)
let AddedComplexity = 45, isPredicated = 1 in
def LDriw_indexed_shl_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1.new) $dst=memw($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// Rd=memw(Rt<<#u2+#U6)


// Post-inc Load, Predicated, Dot new


let hasCtrlDep = 1, neverHasSideEffects = 1, isPredicated = 1 in
def POST_LDrid_cdnPt_V4 : LDInst2PI<(outs DoubleRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_3Imm:$src3),
            "if ($src1.new) $dst1 = memd($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

let hasCtrlDep = 1, neverHasSideEffects = 1, isPredicated = 1 in
def POST_LDrid_cdnNotPt_V4 : LDInst2PI<(outs DoubleRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_3Imm:$src3),
            "if (!$src1.new) $dst1 = memd($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

let hasCtrlDep = 1, neverHasSideEffects = 1, isPredicated = 1 in
def POST_LDrib_cdnPt_V4 : LDInst2PI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_0Imm:$src3),
            "if ($src1.new) $dst1 = memb($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

let hasCtrlDep = 1, neverHasSideEffects = 1, isPredicated = 1 in
def POST_LDrib_cdnNotPt_V4 : LDInst2PI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_0Imm:$src3),
            "if (!$src1.new) $dst1 = memb($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

let hasCtrlDep = 1, neverHasSideEffects = 1, isPredicated = 1 in
def POST_LDrih_cdnPt_V4 : LDInst2PI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_1Imm:$src3),
            "if ($src1.new) $dst1 = memh($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

let hasCtrlDep = 1, neverHasSideEffects = 1, isPredicated = 1 in
def POST_LDrih_cdnNotPt_V4 : LDInst2PI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_1Imm:$src3),
            "if (!$src1.new) $dst1 = memh($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

let hasCtrlDep = 1, neverHasSideEffects = 1, isPredicated = 1 in
def POST_LDriub_cdnPt_V4 : LDInst2PI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_0Imm:$src3),
            "if ($src1.new) $dst1 = memub($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

let hasCtrlDep = 1, neverHasSideEffects = 1, isPredicated = 1 in
def POST_LDriub_cdnNotPt_V4 : LDInst2PI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_0Imm:$src3),
            "if (!$src1.new) $dst1 = memub($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

let hasCtrlDep = 1, neverHasSideEffects = 1, isPredicated = 1 in
def POST_LDriuh_cdnPt_V4 : LDInst2PI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_1Imm:$src3),
            "if ($src1.new) $dst1 = memuh($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

let hasCtrlDep = 1, neverHasSideEffects = 1, isPredicated = 1 in
def POST_LDriuh_cdnNotPt_V4 : LDInst2PI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_1Imm:$src3),
            "if (!$src1.new) $dst1 = memuh($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

let hasCtrlDep = 1, neverHasSideEffects = 1, isPredicated = 1 in
def POST_LDriw_cdnPt_V4 : LDInst2PI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_2Imm:$src3),
            "if ($src1.new) $dst1 = memw($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

let hasCtrlDep = 1, neverHasSideEffects = 1, isPredicated = 1 in
def POST_LDriw_cdnNotPt_V4 : LDInst2PI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_2Imm:$src3),
            "if (!$src1.new) $dst1 = memw($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

/// Load from global offset

let isPredicable = 1, neverHasSideEffects = 1 in
def LDrid_GP_V4 : LDInst2<(outs DoubleRegs:$dst),
            (ins globaladdress:$global, u16Imm:$offset),
            "$dst=memd(#$global+$offset)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def LDrid_GP_cPt_V4 : LDInst2<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if ($src1) $dst=memd(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def LDrid_GP_cNotPt_V4 : LDInst2<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if (!$src1) $dst=memd(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def LDrid_GP_cdnPt_V4 : LDInst2<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if ($src1.new) $dst=memd(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def LDrid_GP_cdnNotPt_V4 : LDInst2<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if (!$src1.new) $dst=memd(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;

let isPredicable = 1, neverHasSideEffects = 1 in
def LDrib_GP_V4 : LDInst2<(outs IntRegs:$dst),
            (ins globaladdress:$global, u16Imm:$offset),
            "$dst=memb(#$global+$offset)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def LDrib_GP_cPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if ($src1) $dst=memb(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def LDrib_GP_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if (!$src1) $dst=memb(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def LDrib_GP_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if ($src1.new) $dst=memb(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def LDrib_GP_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if (!$src1.new) $dst=memb(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;


let isPredicable = 1, neverHasSideEffects = 1 in
def LDriub_GP_V4 : LDInst2<(outs IntRegs:$dst),
            (ins globaladdress:$global, u16Imm:$offset),
            "$dst=memub(#$global+$offset)",
            []>,
            Requires<[HasV4T]>;


let neverHasSideEffects = 1, isPredicated = 1 in
def LDriub_GP_cPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if ($src1) $dst=memub(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def LDriub_GP_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if (!$src1) $dst=memub(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def LDriub_GP_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if ($src1.new) $dst=memub(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def LDriub_GP_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if (!$src1.new) $dst=memub(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;


let isPredicable = 1, neverHasSideEffects = 1 in
def LDrih_GP_V4 : LDInst2<(outs IntRegs:$dst),
            (ins globaladdress:$global, u16Imm:$offset),
            "$dst=memh(#$global+$offset)",
            []>,
            Requires<[HasV4T]>;


let neverHasSideEffects = 1, isPredicated = 1 in
def LDrih_GP_cPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if ($src1) $dst=memh(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def LDrih_GP_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if (!$src1) $dst=memh(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def LDrih_GP_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if ($src1.new) $dst=memh(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def LDrih_GP_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if (!$src1.new) $dst=memh(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;


let isPredicable = 1, neverHasSideEffects = 1 in
def LDriuh_GP_V4 : LDInst2<(outs IntRegs:$dst),
            (ins globaladdress:$global, u16Imm:$offset),
            "$dst=memuh(#$global+$offset)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def LDriuh_GP_cPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if ($src1) $dst=memuh(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def LDriuh_GP_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if (!$src1) $dst=memuh(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def LDriuh_GP_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if ($src1.new) $dst=memuh(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def LDriuh_GP_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if (!$src1.new) $dst=memuh(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;

let isPredicable = 1, neverHasSideEffects = 1 in
def LDriw_GP_V4 : LDInst2<(outs IntRegs:$dst),
            (ins globaladdress:$global, u16Imm:$offset),
            "$dst=memw(#$global+$offset)",
            []>,
            Requires<[HasV4T]>;


let neverHasSideEffects = 1, isPredicated = 1 in
def LDriw_GP_cPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if ($src1) $dst=memw(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def LDriw_GP_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if (!$src1) $dst=memw(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;


let neverHasSideEffects = 1, isPredicated = 1 in
def LDriw_GP_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if ($src1.new) $dst=memw(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def LDriw_GP_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset),
            "if (!$src1.new) $dst=memw(##$global+$offset)",
            []>,
            Requires<[HasV4T]>;


let isPredicable = 1, neverHasSideEffects = 1 in
def LDd_GP_V4 : LDInst2<(outs DoubleRegs:$dst),
            (ins globaladdress:$global),
            "$dst=memd(#$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rtt=memd(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDd_GP_cPt_V4 : LDInst2<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1) $dst=memd(##$global)",
            []>,
            Requires<[HasV4T]>;


// if (!Pv) Rtt=memd(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDd_GP_cNotPt_V4 : LDInst2<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1) $dst=memd(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rtt=memd(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDd_GP_cdnPt_V4 : LDInst2<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1.new) $dst=memd(##$global)",
            []>,
            Requires<[HasV4T]>;


// if (!Pv) Rtt=memd(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDd_GP_cdnNotPt_V4 : LDInst2<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1.new) $dst=memd(##$global)",
            []>,
            Requires<[HasV4T]>;

let isPredicable = 1, neverHasSideEffects = 1 in
def LDb_GP_V4 : LDInst2<(outs IntRegs:$dst),
            (ins globaladdress:$global),
            "$dst=memb(#$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rt=memb(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDb_GP_cPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1) $dst=memb(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) Rt=memb(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDb_GP_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1) $dst=memb(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rt=memb(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDb_GP_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1.new) $dst=memb(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) Rt=memb(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDb_GP_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1.new) $dst=memb(##$global)",
            []>,
            Requires<[HasV4T]>;

let isPredicable = 1, neverHasSideEffects = 1 in
def LDub_GP_V4 : LDInst2<(outs IntRegs:$dst),
            (ins globaladdress:$global),
            "$dst=memub(#$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rt=memub(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDub_GP_cPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1) $dst=memub(##$global)",
            []>,
            Requires<[HasV4T]>;


// if (!Pv) Rt=memub(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDub_GP_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1) $dst=memub(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rt=memub(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDub_GP_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1.new) $dst=memub(##$global)",
            []>,
            Requires<[HasV4T]>;


// if (!Pv) Rt=memub(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDub_GP_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1.new) $dst=memub(##$global)",
            []>,
            Requires<[HasV4T]>;

let isPredicable = 1, neverHasSideEffects = 1 in
def LDh_GP_V4 : LDInst2<(outs IntRegs:$dst),
            (ins globaladdress:$global),
            "$dst=memh(#$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rt=memh(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDh_GP_cPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1) $dst=memh(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) Rt=memh(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDh_GP_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1) $dst=memh(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rt=memh(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDh_GP_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1.new) $dst=memh(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) Rt=memh(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDh_GP_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1.new) $dst=memh(##$global)",
            []>,
            Requires<[HasV4T]>;

let isPredicable = 1, neverHasSideEffects = 1 in
def LDuh_GP_V4 : LDInst2<(outs IntRegs:$dst),
            (ins globaladdress:$global),
            "$dst=memuh(#$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rt=memuh(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDuh_GP_cPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1) $dst=memuh(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) Rt=memuh(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDuh_GP_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1) $dst=memuh(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rt=memuh(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDuh_GP_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1.new) $dst=memuh(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) Rt=memuh(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDuh_GP_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1.new) $dst=memuh(##$global)",
            []>,
            Requires<[HasV4T]>;

let isPredicable = 1, neverHasSideEffects = 1 in
def LDw_GP_V4 : LDInst2<(outs IntRegs:$dst),
            (ins globaladdress:$global),
            "$dst=memw(#$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rt=memw(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDw_GP_cPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1) $dst=memw(##$global)",
            []>,
            Requires<[HasV4T]>;


// if (!Pv) Rt=memw(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDw_GP_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1) $dst=memw(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rt=memw(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDw_GP_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1.new) $dst=memw(##$global)",
            []>,
            Requires<[HasV4T]>;


// if (!Pv) Rt=memw(##global)
let neverHasSideEffects = 1, isPredicated = 1 in
def LDw_GP_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1.new) $dst=memw(##$global)",
            []>,
            Requires<[HasV4T]>;



def : Pat <(atomic_load_64 (HexagonCONST32_GP tglobaladdr:$global)),
           (i64 (LDd_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

def : Pat <(atomic_load_32 (HexagonCONST32_GP tglobaladdr:$global)),
           (i32 (LDw_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

def : Pat <(atomic_load_16 (HexagonCONST32_GP tglobaladdr:$global)),
           (i32 (LDuh_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

def : Pat <(atomic_load_8 (HexagonCONST32_GP tglobaladdr:$global)),
           (i32 (LDub_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

// Map from load(globaladdress) -> memw(#foo + 0)
let AddedComplexity = 100 in
def : Pat <(i64 (load (HexagonCONST32_GP tglobaladdr:$global))),
           (i64 (LDd_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

// Map from Pd = load(globaladdress) -> Rd = memb(globaladdress), Pd = Rd
let AddedComplexity = 100 in
def : Pat <(i1 (load (HexagonCONST32_GP tglobaladdr:$global))),
           (i1 (TFR_PdRs (i32 (LDb_GP_V4 tglobaladdr:$global))))>,
           Requires<[HasV4T]>;

// When the Interprocedural Global Variable optimizer realizes that a certain
// global variable takes only two constant values, it shrinks the global to
// a boolean. Catch those loads here in the following 3 patterns.
let AddedComplexity = 100 in
def : Pat <(i32 (extloadi1 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDb_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

let AddedComplexity = 100 in
def : Pat <(i32 (sextloadi1 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDb_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

// Map from load(globaladdress) -> memb(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (extloadi8 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDb_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

// Map from load(globaladdress) -> memb(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (sextloadi8 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDb_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

let AddedComplexity = 100 in
def : Pat <(i32 (zextloadi1 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDub_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

// Map from load(globaladdress) -> memub(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (zextloadi8 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDub_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

// Map from load(globaladdress) -> memh(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (extloadi16 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDh_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

// Map from load(globaladdress) -> memh(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (sextloadi16 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDh_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

// Map from load(globaladdress) -> memuh(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (zextloadi16 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDuh_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

// Map from load(globaladdress) -> memw(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (load (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDw_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

def : Pat <(atomic_load_64 (add (HexagonCONST32_GP tglobaladdr:$global),
                                u16ImmPred:$offset)),
           (i64 (LDrid_GP_V4 tglobaladdr:$global, u16ImmPred:$offset))>,
           Requires<[HasV4T]>;

def : Pat <(atomic_load_32 (add (HexagonCONST32_GP tglobaladdr:$global),
                                u16ImmPred:$offset)),
           (i32 (LDriw_GP_V4 tglobaladdr:$global, u16ImmPred:$offset))>,
            Requires<[HasV4T]>;

def : Pat <(atomic_load_16 (add (HexagonCONST32_GP tglobaladdr:$global),
                                u16ImmPred:$offset)),
           (i32 (LDriuh_GP_V4 tglobaladdr:$global, u16ImmPred:$offset))>,
            Requires<[HasV4T]>;

def : Pat <(atomic_load_8 (add (HexagonCONST32_GP tglobaladdr:$global),
                               u16ImmPred:$offset)),
           (i32 (LDriub_GP_V4 tglobaladdr:$global, u16ImmPred:$offset))>,
           Requires<[HasV4T]>;

// Map from load(globaladdress + x) -> memd(#foo + x)
let AddedComplexity = 100 in
def : Pat <(i64 (load (add (HexagonCONST32_GP tglobaladdr:$global),
                           u16ImmPred:$offset))),
           (i64 (LDrid_GP_V4 tglobaladdr:$global, u16ImmPred:$offset))>,
           Requires<[HasV4T]>;

// Map from load(globaladdress + x) -> memb(#foo + x)
let AddedComplexity = 100 in
def : Pat <(i32 (extloadi8 (add (HexagonCONST32_GP tglobaladdr:$global),
                           u16ImmPred:$offset))),
           (i32 (LDrib_GP_V4 tglobaladdr:$global, u16ImmPred:$offset))>,
           Requires<[HasV4T]>;

// Map from load(globaladdress + x) -> memb(#foo + x)
let AddedComplexity = 100 in
def : Pat <(i32 (sextloadi8 (add (HexagonCONST32_GP tglobaladdr:$global),
                            u16ImmPred:$offset))),
           (i32 (LDrib_GP_V4 tglobaladdr:$global, u16ImmPred:$offset))>,
           Requires<[HasV4T]>;

// Map from load(globaladdress + x) -> memub(#foo + x)
let AddedComplexity = 100 in
def : Pat <(i32 (zextloadi8 (add (HexagonCONST32_GP tglobaladdr:$global),
                            u16ImmPred:$offset))),
           (i32 (LDriub_GP_V4 tglobaladdr:$global, u16ImmPred:$offset))>,
           Requires<[HasV4T]>;

// Map from load(globaladdress + x) -> memuh(#foo + x)
let AddedComplexity = 100 in
def : Pat <(i32 (extloadi16 (add (HexagonCONST32_GP tglobaladdr:$global),
                            u16ImmPred:$offset))),
           (i32 (LDrih_GP_V4 tglobaladdr:$global, u16ImmPred:$offset))>,
            Requires<[HasV4T]>;

// Map from load(globaladdress + x) -> memh(#foo + x)
let AddedComplexity = 100 in
def : Pat <(i32 (sextloadi16 (add (HexagonCONST32_GP tglobaladdr:$global),
                             u16ImmPred:$offset))),
           (i32 (LDrih_GP_V4 tglobaladdr:$global, u16ImmPred:$offset))>,
           Requires<[HasV4T]>;


// Map from load(globaladdress + x) -> memuh(#foo + x)
let AddedComplexity = 100 in
def : Pat <(i32 (zextloadi16 (add (HexagonCONST32_GP tglobaladdr:$global),
                             u16ImmPred:$offset))),
           (i32 (LDriuh_GP_V4 tglobaladdr:$global, u16ImmPred:$offset))>,
            Requires<[HasV4T]>;

// Map from load(globaladdress + x) -> memw(#foo + x)
let AddedComplexity = 100 in
def : Pat <(i32 (load (add (HexagonCONST32_GP tglobaladdr:$global),
                      u16ImmPred:$offset))),
           (i32 (LDriw_GP_V4 tglobaladdr:$global, u16ImmPred:$offset))>,
            Requires<[HasV4T]>;


//===----------------------------------------------------------------------===//
// LD -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ST +
//===----------------------------------------------------------------------===//
///
/// Assumptions::: ****** DO NOT IGNORE ********
/// 1. Make sure that in post increment store, the zero'th operand is always the
///    post increment operand.
/// 2. Make sure that the store value operand(Rt/Rtt) in a store is always the
///    last operand.
///

// memd(Re=#U6)=Rtt
def STrid_abs_setimm_V4 : STInst2<(outs IntRegs:$dst1),
            (ins DoubleRegs:$src1, u6Imm:$src2),
            "memd($dst1=#$src2) = $src1",
            []>,
            Requires<[HasV4T]>;

// memb(Re=#U6)=Rs
def STrib_abs_setimm_V4 : STInst2<(outs IntRegs:$dst1),
            (ins IntRegs:$src1, u6Imm:$src2),
            "memb($dst1=#$src2) = $src1",
            []>,
            Requires<[HasV4T]>;

// memh(Re=#U6)=Rs
def STrih_abs_setimm_V4 : STInst2<(outs IntRegs:$dst1),
            (ins IntRegs:$src1, u6Imm:$src2),
            "memh($dst1=#$src2) = $src1",
            []>,
            Requires<[HasV4T]>;

// memw(Re=#U6)=Rs
def STriw_abs_setimm_V4 : STInst2<(outs IntRegs:$dst1),
            (ins IntRegs:$src1, u6Imm:$src2),
            "memw($dst1=#$src2) = $src1",
            []>,
            Requires<[HasV4T]>;

// memd(Re=#U6)=Rtt
def STrid_abs_set_V4 : STInst2<(outs IntRegs:$dst1),
            (ins DoubleRegs:$src1, globaladdress:$src2),
            "memd($dst1=##$src2) = $src1",
            []>,
            Requires<[HasV4T]>;

// memb(Re=#U6)=Rs
def STrib_abs_set_V4 : STInst2<(outs IntRegs:$dst1),
            (ins IntRegs:$src1, globaladdress:$src2),
            "memb($dst1=##$src2) = $src1",
            []>,
            Requires<[HasV4T]>;

// memh(Re=#U6)=Rs
def STrih_abs_set_V4 : STInst2<(outs IntRegs:$dst1),
            (ins IntRegs:$src1, globaladdress:$src2),
            "memh($dst1=##$src2) = $src1",
            []>,
            Requires<[HasV4T]>;

// memw(Re=#U6)=Rs
def STriw_abs_set_V4 : STInst2<(outs IntRegs:$dst1),
            (ins IntRegs:$src1, globaladdress:$src2),
            "memw($dst1=##$src2) = $src1",
            []>,
            Requires<[HasV4T]>;

// memd(Rs+Ru<<#u2)=Rtt
let AddedComplexity = 10, isPredicable = 1 in
def STrid_indexed_shl_V4 : STInst<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$src3, DoubleRegs:$src4),
            "memd($src1+$src2<<#$src3) = $src4",
            [(store (i64 DoubleRegs:$src4),
                    (add (i32 IntRegs:$src1),
                         (shl (i32 IntRegs:$src2), u2ImmPred:$src3)))]>,
            Requires<[HasV4T]>;

// memd(Ru<<#u2+#U6)=Rtt
let AddedComplexity = 10 in
def STrid_shl_V4 : STInst<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u6Imm:$src3, DoubleRegs:$src4),
            "memd($src1<<#$src2+#$src3) = $src4",
            [(store (i64 DoubleRegs:$src4),
                    (add (shl (i32 IntRegs:$src1), u2ImmPred:$src2),
                         u6ImmPred:$src3))]>,
            Requires<[HasV4T]>;

// memd(Rx++#s4:3)=Rtt
// memd(Rx++#s4:3:circ(Mu))=Rtt
// memd(Rx++I:circ(Mu))=Rtt
// memd(Rx++Mu)=Rtt
// memd(Rx++Mu:brev)=Rtt
// memd(gp+#u16:3)=Rtt

// Store doubleword conditionally.
// if ([!]Pv[.new]) memd(#u6)=Rtt
// TODO: needs to be implemented.

// if ([!]Pv[.new]) memd(Rs+#u6:3)=Rtt
// if (Pv) memd(Rs+#u6:3)=Rtt
// if (Pv.new) memd(Rs+#u6:3)=Rtt
let AddedComplexity = 10, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrid_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, MEMri:$addr, DoubleRegs:$src2),
            "if ($src1.new) memd($addr) = $src2",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memd(Rs+#u6:3)=Rtt
// if (!Pv.new) memd(Rs+#u6:3)=Rtt
let AddedComplexity = 10, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrid_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, MEMri:$addr, DoubleRegs:$src2),
            "if (!$src1.new) memd($addr) = $src2",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memd(Rs+#u6:3)=Rtt
// if (Pv.new) memd(Rs+#u6:3)=Rtt
let AddedComplexity = 10, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrid_indexed_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_3Imm:$src3,
                 DoubleRegs:$src4),
            "if ($src1.new) memd($src2+#$src3) = $src4",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memd(Rs+#u6:3)=Rtt
// if (!Pv.new) memd(Rs+#u6:3)=Rtt
let AddedComplexity = 10, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrid_indexed_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_3Imm:$src3,
                 DoubleRegs:$src4),
            "if (!$src1.new) memd($src2+#$src3) = $src4",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memd(Rs+Ru<<#u2)=Rtt
// if (Pv) memd(Rs+Ru<<#u2)=Rtt
let AddedComplexity = 10, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrid_indexed_shl_cPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 DoubleRegs:$src5),
            "if ($src1) memd($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memd(Rs+Ru<<#u2)=Rtt
let AddedComplexity = 10, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrid_indexed_shl_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 DoubleRegs:$src5),
            "if ($src1.new) memd($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;
// if (!Pv) memd(Rs+Ru<<#u2)=Rtt
let AddedComplexity = 10, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrid_indexed_shl_cNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 DoubleRegs:$src5),
            "if (!$src1) memd($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;
// if (!Pv.new) memd(Rs+Ru<<#u2)=Rtt
let AddedComplexity = 10, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrid_indexed_shl_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 DoubleRegs:$src5),
            "if (!$src1.new) memd($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memd(Rx++#s4:3)=Rtt
// if (Pv) memd(Rx++#s4:3)=Rtt
// if (Pv.new) memd(Rx++#s4:3)=Rtt
let AddedComplexity = 10, neverHasSideEffects = 1,
    isPredicated = 1 in
def POST_STdri_cdnPt_V4 : STInst2PI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, DoubleRegs:$src2, IntRegs:$src3,
                 s4_3Imm:$offset),
            "if ($src1.new) memd($src3++#$offset) = $src2",
            [],
            "$src3 = $dst">,
            Requires<[HasV4T]>;

// if (!Pv) memd(Rx++#s4:3)=Rtt
// if (!Pv.new) memd(Rx++#s4:3)=Rtt
let AddedComplexity = 10, neverHasSideEffects = 1,
    isPredicated = 1 in
def POST_STdri_cdnNotPt_V4 : STInst2PI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, DoubleRegs:$src2, IntRegs:$src3,
                 s4_3Imm:$offset),
            "if (!$src1.new) memd($src3++#$offset) = $src2",
            [],
            "$src3 = $dst">,
            Requires<[HasV4T]>;


// Store byte.
// memb(Rs+#u6:0)=#S8
let AddedComplexity = 10, isPredicable = 1 in
def STrib_imm_V4 : STInst<(outs),
            (ins IntRegs:$src1, u6_0Imm:$src2, s8Imm:$src3),
            "memb($src1+#$src2) = #$src3",
            [(truncstorei8 s8ImmPred:$src3, (add (i32 IntRegs:$src1),
                                                 u6_0ImmPred:$src2))]>,
            Requires<[HasV4T]>;

// memb(Rs+Ru<<#u2)=Rt
let AddedComplexity = 10, isPredicable = 1 in
def STrib_indexed_shl_V4 : STInst<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$src3, IntRegs:$src4),
            "memb($src1+$src2<<#$src3) = $src4",
            [(truncstorei8 (i32 IntRegs:$src4),
                           (add (i32 IntRegs:$src1),
                                (shl (i32 IntRegs:$src2),
                                          u2ImmPred:$src3)))]>,
            Requires<[HasV4T]>;

// memb(Ru<<#u2+#U6)=Rt
let AddedComplexity = 10 in
def STrib_shl_V4 : STInst<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u6Imm:$src3, IntRegs:$src4),
            "memb($src1<<#$src2+#$src3) = $src4",
            [(truncstorei8 (i32 IntRegs:$src4),
                           (add (shl (i32 IntRegs:$src1), u2ImmPred:$src2),
                                u6ImmPred:$src3))]>,
            Requires<[HasV4T]>;

// memb(Rx++#s4:0:circ(Mu))=Rt
// memb(Rx++I:circ(Mu))=Rt
// memb(Rx++Mu)=Rt
// memb(Rx++Mu:brev)=Rt
// memb(gp+#u16:0)=Rt


// Store byte conditionally.
// if ([!]Pv[.new]) memb(#u6)=Rt
// if ([!]Pv[.new]) memb(Rs+#u6:0)=#S6
// if (Pv) memb(Rs+#u6:0)=#S6
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STrib_imm_cPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3, s6Imm:$src4),
            "if ($src1) memb($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memb(Rs+#u6:0)=#S6
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STrib_imm_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3, s6Imm:$src4),
            "if ($src1.new) memb($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memb(Rs+#u6:0)=#S6
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STrib_imm_cNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3, s6Imm:$src4),
            "if (!$src1) memb($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memb(Rs+#u6:0)=#S6
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STrib_imm_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3, s6Imm:$src4),
            "if (!$src1.new) memb($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memb(Rs+#u6:0)=Rt
// if (Pv) memb(Rs+#u6:0)=Rt
// if (Pv.new) memb(Rs+#u6:0)=Rt
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STrib_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if ($src1.new) memb($addr) = $src2",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memb(Rs+#u6:0)=Rt
// if (!Pv.new) memb(Rs+#u6:0)=Rt
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STrib_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if (!$src1.new) memb($addr) = $src2",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memb(Rs+#u6:0)=Rt
// if (!Pv) memb(Rs+#u6:0)=Rt
// if (Pv.new) memb(Rs+#u6:0)=Rt
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STrib_indexed_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3, IntRegs:$src4),
            "if ($src1.new) memb($src2+#$src3) = $src4",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memb(Rs+#u6:0)=Rt
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STrib_indexed_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3, IntRegs:$src4),
            "if (!$src1.new) memb($src2+#$src3) = $src4",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memb(Rs+Ru<<#u2)=Rt
// if (Pv) memb(Rs+Ru<<#u2)=Rt
let AddedComplexity = 10,
    isPredicated = 1 in
def STrib_indexed_shl_cPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1) memb($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memb(Rs+Ru<<#u2)=Rt
let AddedComplexity = 10,
    isPredicated = 1 in
def STrib_indexed_shl_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1.new) memb($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memb(Rs+Ru<<#u2)=Rt
let AddedComplexity = 10,
    isPredicated = 1 in
def STrib_indexed_shl_cNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1) memb($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memb(Rs+Ru<<#u2)=Rt
let AddedComplexity = 10,
    isPredicated = 1 in
def STrib_indexed_shl_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1.new) memb($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memb(Rx++#s4:0)=Rt
// if (Pv) memb(Rx++#s4:0)=Rt
// if (Pv.new) memb(Rx++#s4:0)=Rt
let hasCtrlDep = 1,
    isPredicated = 1 in
def POST_STbri_cdnPt_V4 : STInst2PI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_0Imm:$offset),
            "if ($src1.new) memb($src3++#$offset) = $src2",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (!Pv) memb(Rx++#s4:0)=Rt
// if (!Pv.new) memb(Rx++#s4:0)=Rt
let hasCtrlDep = 1,
    isPredicated = 1 in
def POST_STbri_cdnNotPt_V4 : STInst2PI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_0Imm:$offset),
            "if (!$src1.new) memb($src3++#$offset) = $src2",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;


// Store halfword.
// TODO: needs to be implemented
// memh(Re=#U6)=Rt.H
// memh(Rs+#s11:1)=Rt.H
// memh(Rs+#u6:1)=#S8
let AddedComplexity = 10, isPredicable = 1 in
def STrih_imm_V4 : STInst<(outs),
            (ins IntRegs:$src1, u6_1Imm:$src2, s8Imm:$src3),
            "memh($src1+#$src2) = #$src3",
            [(truncstorei16 s8ImmPred:$src3, (add (i32 IntRegs:$src1),
                                                  u6_1ImmPred:$src2))]>,
            Requires<[HasV4T]>;

// memh(Rs+Ru<<#u2)=Rt.H
// TODO: needs to be implemented.

// memh(Rs+Ru<<#u2)=Rt
let AddedComplexity = 10, isPredicable = 1 in
def STrih_indexed_shl_V4 : STInst<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$src3, IntRegs:$src4),
            "memh($src1+$src2<<#$src3) = $src4",
            [(truncstorei16 (i32 IntRegs:$src4),
                            (add (i32 IntRegs:$src1),
                                 (shl (i32 IntRegs:$src2),
                                      u2ImmPred:$src3)))]>,
            Requires<[HasV4T]>;

// memh(Ru<<#u2+#U6)=Rt.H
// memh(Ru<<#u2+#U6)=Rt
let AddedComplexity = 10 in
def STrih_shl_V4 : STInst<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u6Imm:$src3, IntRegs:$src4),
            "memh($src1<<#$src2+#$src3) = $src4",
            [(truncstorei16 (i32 IntRegs:$src4),
                            (add (shl (i32 IntRegs:$src1), u2ImmPred:$src2),
                                 u6ImmPred:$src3))]>,
            Requires<[HasV4T]>;

// memh(Rx++#s4:1:circ(Mu))=Rt.H
// memh(Rx++#s4:1:circ(Mu))=Rt
// memh(Rx++I:circ(Mu))=Rt.H
// memh(Rx++I:circ(Mu))=Rt
// memh(Rx++Mu)=Rt.H
// memh(Rx++Mu)=Rt
// memh(Rx++Mu:brev)=Rt.H
// memh(Rx++Mu:brev)=Rt
// memh(gp+#u16:1)=Rt
// if ([!]Pv[.new]) memh(#u6)=Rt.H
// if ([!]Pv[.new]) memh(#u6)=Rt

// if ([!]Pv[.new]) memh(Rs+#u6:1)=#S6
// if (Pv) memh(Rs+#u6:1)=#S6
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STrih_imm_cPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3, s6Imm:$src4),
            "if ($src1) memh($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memh(Rs+#u6:1)=#S6
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STrih_imm_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3, s6Imm:$src4),
            "if ($src1.new) memh($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memh(Rs+#u6:1)=#S6
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STrih_imm_cNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3, s6Imm:$src4),
            "if (!$src1) memh($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memh(Rs+#u6:1)=#S6
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STrih_imm_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3, s6Imm:$src4),
            "if (!$src1.new) memh($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memh(Rs+#u6:1)=Rt.H
// TODO: needs to be implemented.

// if ([!]Pv[.new]) memh(Rs+#u6:1)=Rt
// if (Pv) memh(Rs+#u6:1)=Rt
// if (Pv.new) memh(Rs+#u6:1)=Rt
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STrih_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if ($src1.new) memh($addr) = $src2",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memh(Rs+#u6:1)=Rt
// if (!Pv.new) memh(Rs+#u6:1)=Rt
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STrih_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if (!$src1.new) memh($addr) = $src2",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memh(Rs+#u6:1)=Rt
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STrih_indexed_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3, IntRegs:$src4),
            "if ($src1.new) memh($src2+#$src3) = $src4",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memh(Rs+#u6:1)=Rt
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STrih_indexed_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3, IntRegs:$src4),
            "if (!$src1.new) memh($src2+#$src3) = $src4",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memh(Rs+Ru<<#u2)=Rt.H
// if ([!]Pv[.new]) memh(Rs+Ru<<#u2)=Rt
// if (Pv) memh(Rs+Ru<<#u2)=Rt
let AddedComplexity = 10,
    isPredicated = 1 in
def STrih_indexed_shl_cPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1) memh($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memh(Rs+Ru<<#u2)=Rt
let AddedComplexity = 10,
    isPredicated = 1 in
def STrih_indexed_shl_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1.new) memh($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memh(Rs+Ru<<#u2)=Rt
let AddedComplexity = 10,
    isPredicated = 1 in
def STrih_indexed_shl_cNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1) memh($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memh(Rs+Ru<<#u2)=Rt
let AddedComplexity = 10,
    isPredicated = 1 in
def STrih_indexed_shl_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1.new) memh($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memh(Rx++#s4:1)=Rt.H
// TODO: Needs to be implemented.

// if ([!]Pv[.new]) memh(Rx++#s4:1)=Rt
// if (Pv) memh(Rx++#s4:1)=Rt
// if (Pv.new) memh(Rx++#s4:1)=Rt
let hasCtrlDep = 1,
    isPredicated = 1 in
def POST_SThri_cdnPt_V4 : STInst2PI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_1Imm:$offset),
            "if ($src1.new) memh($src3++#$offset) = $src2",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (!Pv) memh(Rx++#s4:1)=Rt
// if (!Pv.new) memh(Rx++#s4:1)=Rt
let hasCtrlDep = 1,
    isPredicated = 1 in
def POST_SThri_cdnNotPt_V4 : STInst2PI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_1Imm:$offset),
            "if (!$src1.new) memh($src3++#$offset) = $src2",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;


// Store word.
// memw(Re=#U6)=Rt
// TODO: Needs to be implemented.

// Store predicate:
let neverHasSideEffects = 1 in
def STriw_pred_V4 : STInst2<(outs),
            (ins MEMri:$addr, PredRegs:$src1),
            "Error; should not emit",
            []>,
            Requires<[HasV4T]>;


// memw(Rs+#u6:2)=#S8
let AddedComplexity = 10, isPredicable = 1 in
def STriw_imm_V4 : STInst<(outs),
            (ins IntRegs:$src1, u6_2Imm:$src2, s8Imm:$src3),
            "memw($src1+#$src2) = #$src3",
            [(store s8ImmPred:$src3, (add (i32 IntRegs:$src1),
                                          u6_2ImmPred:$src2))]>,
            Requires<[HasV4T]>;

// memw(Rs+Ru<<#u2)=Rt
let AddedComplexity = 10, isPredicable = 1 in
def STriw_indexed_shl_V4 : STInst<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$src3, IntRegs:$src4),
            "memw($src1+$src2<<#$src3) = $src4",
            [(store (i32 IntRegs:$src4), (add (i32 IntRegs:$src1),
                                    (shl (i32 IntRegs:$src2),
                                         u2ImmPred:$src3)))]>,
            Requires<[HasV4T]>;

// memw(Ru<<#u2+#U6)=Rt
let AddedComplexity = 10 in
def STriw_shl_V4 : STInst<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u6Imm:$src3, IntRegs:$src4),
            "memw($src1<<#$src2+#$src3) = $src4",
            [(store (i32 IntRegs:$src4),
                    (add (shl (i32 IntRegs:$src1), u2ImmPred:$src2),
                              u6ImmPred:$src3))]>,
            Requires<[HasV4T]>;

// memw(Rx++#s4:2)=Rt
// memw(Rx++#s4:2:circ(Mu))=Rt
// memw(Rx++I:circ(Mu))=Rt
// memw(Rx++Mu)=Rt
// memw(Rx++Mu:brev)=Rt
// memw(gp+#u16:2)=Rt


// Store word conditionally.

// if ([!]Pv[.new]) memw(Rs+#u6:2)=#S6
// if (Pv) memw(Rs+#u6:2)=#S6
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STriw_imm_cPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3, s6Imm:$src4),
            "if ($src1) memw($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memw(Rs+#u6:2)=#S6
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STriw_imm_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3, s6Imm:$src4),
            "if ($src1.new) memw($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memw(Rs+#u6:2)=#S6
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STriw_imm_cNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3, s6Imm:$src4),
            "if (!$src1) memw($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memw(Rs+#u6:2)=#S6
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STriw_imm_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3, s6Imm:$src4),
            "if (!$src1.new) memw($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memw(Rs+#u6:2)=Rt
// if (Pv) memw(Rs+#u6:2)=Rt
// if (Pv.new) memw(Rs+#u6:2)=Rt
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STriw_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if ($src1.new) memw($addr) = $src2",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memw(Rs+#u6:2)=Rt
// if (!Pv.new) memw(Rs+#u6:2)=Rt
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STriw_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if (!$src1.new) memw($addr) = $src2",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memw(Rs+#u6:2)=Rt
// if (!Pv) memw(Rs+#u6:2)=Rt
// if (Pv.new) memw(Rs+#u6:2)=Rt
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STriw_indexed_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3, IntRegs:$src4),
            "if ($src1.new) memw($src2+#$src3) = $src4",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memw(Rs+#u6:2)=Rt
let neverHasSideEffects = 1,
    isPredicated = 1 in
def STriw_indexed_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3, IntRegs:$src4),
            "if (!$src1.new) memw($src2+#$src3) = $src4",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memw(Rs+Ru<<#u2)=Rt
// if (Pv) memw(Rs+Ru<<#u2)=Rt
let AddedComplexity = 10,
    isPredicated = 1 in
def STriw_indexed_shl_cPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1) memw($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memw(Rs+Ru<<#u2)=Rt
let AddedComplexity = 10,
    isPredicated = 1 in
def STriw_indexed_shl_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1.new) memw($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memw(Rs+Ru<<#u2)=Rt
let AddedComplexity = 10,
    isPredicated = 1 in
def STriw_indexed_shl_cNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1) memw($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memw(Rs+Ru<<#u2)=Rt
let AddedComplexity = 10,
    isPredicated = 1 in
def STriw_indexed_shl_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1.new) memw($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memw(Rx++#s4:2)=Rt
// if (Pv) memw(Rx++#s4:2)=Rt
// if (Pv.new) memw(Rx++#s4:2)=Rt
let hasCtrlDep = 1,
    isPredicated = 1 in
def POST_STwri_cdnPt_V4 : STInst2PI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_2Imm:$offset),
            "if ($src1.new) memw($src3++#$offset) = $src2",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (!Pv) memw(Rx++#s4:2)=Rt
// if (!Pv.new) memw(Rx++#s4:2)=Rt
let hasCtrlDep = 1,
    isPredicated = 1 in
def POST_STwri_cdnNotPt_V4 : STInst2PI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_2Imm:$offset),
            "if (!$src1.new) memw($src3++#$offset) = $src2",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;


/// store to global address

let isPredicable = 1, neverHasSideEffects = 1 in
def STrid_GP_V4 : STInst2<(outs),
            (ins globaladdress:$global, u16Imm:$offset, DoubleRegs:$src),
            "memd(#$global+$offset) = $src",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def STrid_GP_cPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        DoubleRegs:$src2),
            "if ($src1) memd(##$global+$offset) = $src2",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def STrid_GP_cNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        DoubleRegs:$src2),
            "if (!$src1) memd(##$global+$offset) = $src2",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def STrid_GP_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        DoubleRegs:$src2),
            "if ($src1.new) memd(##$global+$offset) = $src2",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def STrid_GP_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        DoubleRegs:$src2),
            "if (!$src1.new) memd(##$global+$offset) = $src2",
            []>,
            Requires<[HasV4T]>;

let isPredicable = 1, neverHasSideEffects = 1 in
def STrib_GP_V4 : STInst2<(outs),
            (ins globaladdress:$global, u16Imm:$offset, IntRegs:$src),
            "memb(#$global+$offset) = $src",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def STrib_GP_cPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if ($src1) memb(##$global+$offset) = $src2",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def STrib_GP_cNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if (!$src1) memb(##$global+$offset) = $src2",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def STrib_GP_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if ($src1.new) memb(##$global+$offset) = $src2",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def STrib_GP_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if (!$src1.new) memb(##$global+$offset) = $src2",
            []>,
            Requires<[HasV4T]>;

let isPredicable = 1, neverHasSideEffects = 1 in
def STrih_GP_V4 : STInst2<(outs),
            (ins globaladdress:$global, u16Imm:$offset, IntRegs:$src),
            "memh(#$global+$offset) = $src",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def STrih_GP_cPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if ($src1) memh(##$global+$offset) = $src2",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def STrih_GP_cNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if (!$src1) memh(##$global+$offset) = $src2",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def STrih_GP_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if ($src1.new) memh(##$global+$offset) = $src2",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def STrih_GP_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if (!$src1.new) memh(##$global+$offset) = $src2",
            []>,
            Requires<[HasV4T]>;

let isPredicable = 1, neverHasSideEffects = 1 in
def STriw_GP_V4 : STInst2<(outs),
            (ins globaladdress:$global, u16Imm:$offset, IntRegs:$src),
            "memw(#$global+$offset) = $src",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def STriw_GP_cPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if ($src1) memw(##$global+$offset) = $src2",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def STriw_GP_cNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if (!$src1) memw(##$global+$offset) = $src2",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def STriw_GP_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if ($src1.new) memw(##$global+$offset) = $src2",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1, isPredicated = 1 in
def STriw_GP_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if (!$src1.new) memw(##$global+$offset) = $src2",
            []>,
            Requires<[HasV4T]>;

// memd(#global)=Rtt
let isPredicable = 1, neverHasSideEffects = 1 in
def STd_GP_V4 : STInst2<(outs),
            (ins globaladdress:$global, DoubleRegs:$src),
            "memd(#$global) = $src",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memd(##global) = Rtt
let neverHasSideEffects = 1, isPredicated = 1 in
def STd_GP_cPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, DoubleRegs:$src2),
            "if ($src1) memd(##$global) = $src2",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memd(##global) = Rtt
let neverHasSideEffects = 1, isPredicated = 1 in
def STd_GP_cNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, DoubleRegs:$src2),
            "if (!$src1) memd(##$global) = $src2",
            []>,
              Requires<[HasV4T]>;

// if (Pv) memd(##global) = Rtt
let neverHasSideEffects = 1, isPredicated = 1 in
def STd_GP_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, DoubleRegs:$src2),
            "if ($src1.new) memd(##$global) = $src2",
            []>,
              Requires<[HasV4T]>;

// if (!Pv) memd(##global) = Rtt
let neverHasSideEffects = 1, isPredicated = 1 in
def STd_GP_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, DoubleRegs:$src2),
            "if (!$src1.new) memd(##$global) = $src2",
            []>,
            Requires<[HasV4T]>;

// memb(#global)=Rt
let isPredicable = 1, neverHasSideEffects = 1 in
def STb_GP_V4 : STInst2<(outs),
            (ins globaladdress:$global, IntRegs:$src),
            "memb(#$global) = $src",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memb(##global) = Rt
let neverHasSideEffects = 1, isPredicated = 1 in
def STb_GP_cPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1) memb(##$global) = $src2",
              []>,
              Requires<[HasV4T]>;

// if (!Pv) memb(##global) = Rt
let neverHasSideEffects = 1, isPredicated = 1 in
def STb_GP_cNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1) memb(##$global) = $src2",
              []>,
              Requires<[HasV4T]>;

// if (Pv) memb(##global) = Rt
let neverHasSideEffects = 1, isPredicated = 1 in
def STb_GP_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1.new) memb(##$global) = $src2",
              []>,
              Requires<[HasV4T]>;

// if (!Pv) memb(##global) = Rt
let neverHasSideEffects = 1, isPredicated = 1 in
def STb_GP_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1.new) memb(##$global) = $src2",
              []>,
              Requires<[HasV4T]>;

// memh(#global)=Rt
let isPredicable = 1, neverHasSideEffects = 1 in
def STh_GP_V4 : STInst2<(outs),
            (ins globaladdress:$global, IntRegs:$src),
            "memh(#$global) = $src",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memh(##global) = Rt
let neverHasSideEffects = 1, isPredicated = 1 in
def STh_GP_cPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1) memh(##$global) = $src2",
              []>,
              Requires<[HasV4T]>;

// if (!Pv) memh(##global) = Rt
let neverHasSideEffects = 1, isPredicated = 1 in
def STh_GP_cNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1) memh(##$global) = $src2",
              []>,
              Requires<[HasV4T]>;

// if (Pv) memh(##global) = Rt
let neverHasSideEffects = 1, isPredicated = 1 in
def STh_GP_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1.new) memh(##$global) = $src2",
              []>,
              Requires<[HasV4T]>;

// if (!Pv) memh(##global) = Rt
let neverHasSideEffects = 1, isPredicated = 1 in
def STh_GP_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1.new) memh(##$global) = $src2",
              []>,
              Requires<[HasV4T]>;

// memw(#global)=Rt
let isPredicable = 1, neverHasSideEffects = 1 in
def STw_GP_V4 : STInst2<(outs),
            (ins globaladdress:$global, IntRegs:$src),
            "memw(#$global) = $src",
              []>,
              Requires<[HasV4T]>;

// if (Pv) memw(##global) = Rt
let neverHasSideEffects = 1, isPredicated = 1 in
def STw_GP_cPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1) memw(##$global) = $src2",
              []>,
              Requires<[HasV4T]>;

// if (!Pv) memw(##global) = Rt
let neverHasSideEffects = 1, isPredicated = 1 in
def STw_GP_cNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1) memw(##$global) = $src2",
              []>,
              Requires<[HasV4T]>;

// if (Pv) memw(##global) = Rt
let neverHasSideEffects = 1, isPredicated = 1 in
def STw_GP_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1.new) memw(##$global) = $src2",
              []>,
              Requires<[HasV4T]>;

// if (!Pv) memw(##global) = Rt
let neverHasSideEffects = 1, isPredicated = 1 in
def STw_GP_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1.new) memw(##$global) = $src2",
            []>,
              Requires<[HasV4T]>;

// 64 bit atomic store
def : Pat <(atomic_store_64 (HexagonCONST32_GP tglobaladdr:$global),
                            (i64 DoubleRegs:$src1)),
           (STd_GP_V4 tglobaladdr:$global, (i64 DoubleRegs:$src1))>,
           Requires<[HasV4T]>;

// Map from store(globaladdress) -> memd(#foo)
let AddedComplexity = 100 in
def : Pat <(store (i64 DoubleRegs:$src1),
                  (HexagonCONST32_GP tglobaladdr:$global)),
           (STd_GP_V4 tglobaladdr:$global, (i64 DoubleRegs:$src1))>,
           Requires<[HasV4T]>;

// 8 bit atomic store
def : Pat < (atomic_store_8 (HexagonCONST32_GP tglobaladdr:$global),
                            (i32 IntRegs:$src1)),
            (STb_GP_V4 tglobaladdr:$global, (i32 IntRegs:$src1))>,
              Requires<[HasV4T]>;

// Map from store(globaladdress) -> memb(#foo)
let AddedComplexity = 100 in
def : Pat<(truncstorei8 (i32 IntRegs:$src1),
          (HexagonCONST32_GP tglobaladdr:$global)),
          (STb_GP_V4 tglobaladdr:$global, (i32 IntRegs:$src1))>,
          Requires<[HasV4T]>;

// Map from "i1 = constant<-1>; memw(CONST32(#foo)) = i1"
//       to "r0 = 1; memw(#foo) = r0"
let AddedComplexity = 100 in
def : Pat<(store (i1 -1), (HexagonCONST32_GP tglobaladdr:$global)),
          (STb_GP_V4 tglobaladdr:$global, (TFRI 1))>,
          Requires<[HasV4T]>;

def : Pat<(atomic_store_16 (HexagonCONST32_GP tglobaladdr:$global),
                           (i32 IntRegs:$src1)),
          (STh_GP_V4 tglobaladdr:$global, (i32 IntRegs:$src1))>,
          Requires<[HasV4T]>;

// Map from store(globaladdress) -> memh(#foo)
let AddedComplexity = 100 in
def : Pat<(truncstorei16 (i32 IntRegs:$src1),
                         (HexagonCONST32_GP tglobaladdr:$global)),
          (STh_GP_V4 tglobaladdr:$global, (i32 IntRegs:$src1))>,
          Requires<[HasV4T]>;

// 32 bit atomic store
def : Pat<(atomic_store_32 (HexagonCONST32_GP tglobaladdr:$global),
                           (i32 IntRegs:$src1)),
          (STw_GP_V4 tglobaladdr:$global, (i32 IntRegs:$src1))>,
          Requires<[HasV4T]>;

// Map from store(globaladdress) -> memw(#foo)
let AddedComplexity = 100 in
def : Pat<(store (i32 IntRegs:$src1), (HexagonCONST32_GP tglobaladdr:$global)),
          (STw_GP_V4 tglobaladdr:$global, (i32 IntRegs:$src1))>,
          Requires<[HasV4T]>;

def : Pat<(atomic_store_64 (add (HexagonCONST32_GP tglobaladdr:$global),
                                u16ImmPred:$offset),
                           (i64 DoubleRegs:$src1)),
          (STrid_GP_V4 tglobaladdr:$global, u16ImmPred:$offset,
                                            (i64 DoubleRegs:$src1))>,
          Requires<[HasV4T]>;

def : Pat<(atomic_store_32 (add (HexagonCONST32_GP tglobaladdr:$global),
                                u16ImmPred:$offset),
                           (i32 IntRegs:$src1)),
          (STriw_GP_V4 tglobaladdr:$global, u16ImmPred:$offset,
                                            (i32 IntRegs:$src1))>,
          Requires<[HasV4T]>;

def : Pat<(atomic_store_16 (add (HexagonCONST32_GP tglobaladdr:$global),
                                u16ImmPred:$offset),
                           (i32 IntRegs:$src1)),
          (STrih_GP_V4 tglobaladdr:$global, u16ImmPred:$offset,
                                            (i32 IntRegs:$src1))>,
          Requires<[HasV4T]>;

def : Pat<(atomic_store_8 (add (HexagonCONST32_GP tglobaladdr:$global),
                               u16ImmPred:$offset),
                          (i32 IntRegs:$src1)),
          (STrib_GP_V4 tglobaladdr:$global, u16ImmPred:$offset,
                                            (i32 IntRegs:$src1))>,
          Requires<[HasV4T]>;

// Map from store(globaladdress + x) -> memd(#foo + x)
let AddedComplexity = 100 in
def : Pat<(store (i64 DoubleRegs:$src1),
                    (add (HexagonCONST32_GP tglobaladdr:$global),
                                        u16ImmPred:$offset)),
          (STrid_GP_V4 tglobaladdr:$global, u16ImmPred:$offset,
                                            (i64 DoubleRegs:$src1))>,
          Requires<[HasV4T]>;

// Map from store(globaladdress + x) -> memb(#foo + x)
let AddedComplexity = 100 in
def : Pat<(truncstorei8 (i32 IntRegs:$src1),
                        (add (HexagonCONST32_GP tglobaladdr:$global),
                             u16ImmPred:$offset)),
          (STrib_GP_V4 tglobaladdr:$global, u16ImmPred:$offset,
                                            (i32 IntRegs:$src1))>,
          Requires<[HasV4T]>;

// Map from store(globaladdress + x) -> memh(#foo + x)
let AddedComplexity = 100 in
def : Pat<(truncstorei16 (i32 IntRegs:$src1),
                         (add (HexagonCONST32_GP tglobaladdr:$global),
                              u16ImmPred:$offset)),
          (STrih_GP_V4 tglobaladdr:$global, u16ImmPred:$offset,
                                            (i32 IntRegs:$src1))>,
          Requires<[HasV4T]>;

// Map from store(globaladdress + x) -> memw(#foo + x)
let AddedComplexity = 100 in
def : Pat<(store (i32 IntRegs:$src1),
                 (add (HexagonCONST32_GP tglobaladdr:$global),
                                u16ImmPred:$offset)),
          (STriw_GP_V4 tglobaladdr:$global, u16ImmPred:$offset,
                                            (i32 IntRegs:$src1))>,
          Requires<[HasV4T]>;



//===----------------------------------------------------------------------===
// ST -
//===----------------------------------------------------------------------===


//===----------------------------------------------------------------------===//
// NV/ST +
//===----------------------------------------------------------------------===//

// Store new-value byte.

// memb(Re=#U6)=Nt.new
// memb(Rs+#s11:0)=Nt.new
let mayStore = 1, isPredicable = 1 in
def STrib_nv_V4 : NVInst_V4<(outs), (ins MEMri:$addr, IntRegs:$src1),
            "memb($addr) = $src1.new",
            []>,
            Requires<[HasV4T]>;

let mayStore = 1, isPredicable = 1 in
def STrib_indexed_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, s11_0Imm:$src2, IntRegs:$src3),
            "memb($src1+#$src2) = $src3.new",
            []>,
            Requires<[HasV4T]>;

// memb(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10, isPredicable = 1 in
def STrib_indexed_shl_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$src3, IntRegs:$src4),
            "memb($src1+$src2<<#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// memb(Ru<<#u2+#U6)=Nt.new
let mayStore = 1, AddedComplexity = 10 in
def STrib_shl_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u6Imm:$src3, IntRegs:$src4),
            "memb($src1<<#$src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// memb(Rx++#s4:0)=Nt.new
let mayStore = 1, hasCtrlDep = 1, isPredicable = 1  in
def POST_STbri_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2, s4_0Imm:$offset),
            "memb($src2++#$offset) = $src1.new",
            [],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

// memb(Rx++#s4:0:circ(Mu))=Nt.new
// memb(Rx++I:circ(Mu))=Nt.new
// memb(Rx++Mu)=Nt.new
// memb(Rx++Mu:brev)=Nt.new

// memb(gp+#u16:0)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_GP_nv_V4 : NVInst_V4<(outs),
            (ins globaladdress:$global, u16Imm:$offset, IntRegs:$src),
            "memb(#$global+$offset) = $src.new",
            []>,
            Requires<[HasV4T]>;

// memb(#global)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STb_GP_nv_V4 : NVInst_V4<(outs),
            (ins globaladdress:$global, IntRegs:$src),
            "memb(#$global) = $src.new",
            []>,
            Requires<[HasV4T]>;

// Store new-value byte conditionally.
// if ([!]Pv[.new]) memb(#u6)=Nt.new
// if (Pv) memb(Rs+#u6:0)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrib_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if ($src1) memb($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memb(Rs+#u6:0)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrib_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if ($src1.new) memb($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memb(Rs+#u6:0)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrib_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if (!$src1) memb($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memb(Rs+#u6:0)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrib_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if (!$src1.new) memb($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memb(Rs+#u6:0)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrib_indexed_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3, IntRegs:$src4),
            "if ($src1) memb($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memb(Rs+#u6:0)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrib_indexed_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3, IntRegs:$src4),
            "if ($src1.new) memb($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memb(Rs+#u6:0)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrib_indexed_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3, IntRegs:$src4),
            "if (!$src1) memb($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memb(Rs+#u6:0)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrib_indexed_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3, IntRegs:$src4),
            "if (!$src1.new) memb($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;


// if ([!]Pv[.new]) memb(Rs+Ru<<#u2)=Nt.new
// if (Pv) memb(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10,
    isPredicated = 1 in
def STrib_indexed_shl_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1) memb($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memb(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10,
    isPredicated = 1 in
def STrib_indexed_shl_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1.new) memb($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memb(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10,
    isPredicated = 1 in
def STrib_indexed_shl_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1) memb($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memb(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10,
    isPredicated = 1 in
def STrib_indexed_shl_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1.new) memb($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memb(Rx++#s4:0)=Nt.new
// if (Pv) memb(Rx++#s4:0)=Nt.new
let mayStore = 1, hasCtrlDep = 1,
    isPredicated = 1 in
def POST_STbri_cPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_0Imm:$offset),
            "if ($src1) memb($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (Pv.new) memb(Rx++#s4:0)=Nt.new
let mayStore = 1, hasCtrlDep = 1,
    isPredicated = 1 in
def POST_STbri_cdnPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_0Imm:$offset),
            "if ($src1.new) memb($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (!Pv) memb(Rx++#s4:0)=Nt.new
let mayStore = 1, hasCtrlDep = 1,
    isPredicated = 1 in
def POST_STbri_cNotPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_0Imm:$offset),
            "if (!$src1) memb($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (!Pv.new) memb(Rx++#s4:0)=Nt.new
let mayStore = 1, hasCtrlDep = 1,
    isPredicated = 1 in
def POST_STbri_cdnNotPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_0Imm:$offset),
            "if (!$src1.new) memb($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;


// Store new-value halfword.
// memh(Re=#U6)=Nt.new
// memh(Rs+#s11:1)=Nt.new
let mayStore = 1, isPredicable = 1 in
def STrih_nv_V4 : NVInst_V4<(outs), (ins MEMri:$addr, IntRegs:$src1),
            "memh($addr) = $src1.new",
            []>,
            Requires<[HasV4T]>;

let mayStore = 1, isPredicable = 1 in
def STrih_indexed_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, s11_1Imm:$src2, IntRegs:$src3),
            "memh($src1+#$src2) = $src3.new",
            []>,
            Requires<[HasV4T]>;

// memh(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10, isPredicable = 1 in
def STrih_indexed_shl_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$src3, IntRegs:$src4),
            "memh($src1+$src2<<#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// memh(Ru<<#u2+#U6)=Nt.new
let mayStore = 1, AddedComplexity = 10 in
def STrih_shl_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u6Imm:$src3, IntRegs:$src4),
            "memh($src1<<#$src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// memh(Rx++#s4:1)=Nt.new
let mayStore = 1, hasCtrlDep = 1, isPredicable = 1  in
def POST_SThri_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2, s4_1Imm:$offset),
            "memh($src2++#$offset) = $src1.new",
            [],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

// memh(Rx++#s4:1:circ(Mu))=Nt.new
// memh(Rx++I:circ(Mu))=Nt.new
// memh(Rx++Mu)=Nt.new
// memh(Rx++Mu:brev)=Nt.new

// memh(gp+#u16:1)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STrih_GP_nv_V4 : NVInst_V4<(outs),
            (ins globaladdress:$global, u16Imm:$offset, IntRegs:$src),
            "memh(#$global+$offset) = $src.new",
            []>,
            Requires<[HasV4T]>;

// memh(#global)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STh_GP_nv_V4 : NVInst_V4<(outs),
            (ins globaladdress:$global, IntRegs:$src),
            "memh(#$global) = $src.new",
            []>,
            Requires<[HasV4T]>;


// Store new-value halfword conditionally.

// if ([!]Pv[.new]) memh(#u6)=Nt.new

// if ([!]Pv[.new]) memh(Rs+#u6:1)=Nt.new
// if (Pv) memh(Rs+#u6:1)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrih_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if ($src1) memh($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memh(Rs+#u6:1)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrih_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if ($src1.new) memh($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memh(Rs+#u6:1)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrih_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if (!$src1) memh($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memh(Rs+#u6:1)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrih_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if (!$src1.new) memh($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memh(Rs+#u6:1)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrih_indexed_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3, IntRegs:$src4),
            "if ($src1) memh($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memh(Rs+#u6:1)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrih_indexed_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3, IntRegs:$src4),
            "if ($src1.new) memh($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memh(Rs+#u6:1)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrih_indexed_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3, IntRegs:$src4),
            "if (!$src1) memh($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memh(Rs+#u6:1)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STrih_indexed_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3, IntRegs:$src4),
            "if (!$src1.new) memh($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memh(Rs+Ru<<#u2)=Nt.new
// if (Pv) memh(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10,
    isPredicated = 1 in
def STrih_indexed_shl_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1) memh($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memh(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10,
    isPredicated = 1 in
def STrih_indexed_shl_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1.new) memh($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memh(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10,
    isPredicated = 1 in
def STrih_indexed_shl_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1) memh($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memh(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10,
    isPredicated = 1 in
def STrih_indexed_shl_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1.new) memh($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[]) memh(Rx++#s4:1)=Nt.new
// if (Pv) memh(Rx++#s4:1)=Nt.new
let mayStore = 1, hasCtrlDep = 1,
    isPredicated = 1 in
def POST_SThri_cPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_1Imm:$offset),
            "if ($src1) memh($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (Pv.new) memh(Rx++#s4:1)=Nt.new
let mayStore = 1, hasCtrlDep = 1,
    isPredicated = 1 in
def POST_SThri_cdnPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_1Imm:$offset),
            "if ($src1.new) memh($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (!Pv) memh(Rx++#s4:1)=Nt.new
let mayStore = 1, hasCtrlDep = 1,
    isPredicated = 1 in
def POST_SThri_cNotPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_1Imm:$offset),
            "if (!$src1) memh($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (!Pv.new) memh(Rx++#s4:1)=Nt.new
let mayStore = 1, hasCtrlDep = 1,
    isPredicated = 1 in
def POST_SThri_cdnNotPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_1Imm:$offset),
            "if (!$src1.new) memh($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;


// Store new-value word.

// memw(Re=#U6)=Nt.new
// memw(Rs+#s11:2)=Nt.new
let mayStore = 1, isPredicable = 1 in
def STriw_nv_V4 : NVInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$src1),
            "memw($addr) = $src1.new",
            []>,
            Requires<[HasV4T]>;

let mayStore = 1, isPredicable = 1 in
def STriw_indexed_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, s11_2Imm:$src2, IntRegs:$src3),
            "memw($src1+#$src2) = $src3.new",
            []>,
            Requires<[HasV4T]>;

// memw(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10, isPredicable = 1 in
def STriw_indexed_shl_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$src3, IntRegs:$src4),
            "memw($src1+$src2<<#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// memw(Ru<<#u2+#U6)=Nt.new
let mayStore = 1, AddedComplexity = 10 in
def STriw_shl_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u6Imm:$src3, IntRegs:$src4),
            "memw($src1<<#$src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// memw(Rx++#s4:2)=Nt.new
let mayStore = 1, hasCtrlDep = 1, isPredicable = 1  in
def POST_STwri_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2, s4_2Imm:$offset),
            "memw($src2++#$offset) = $src1.new",
            [],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

// memw(Rx++#s4:2:circ(Mu))=Nt.new
// memw(Rx++I:circ(Mu))=Nt.new
// memw(Rx++Mu)=Nt.new
// memw(Rx++Mu:brev)=Nt.new
// memw(gp+#u16:2)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STriw_GP_nv_V4 : NVInst_V4<(outs),
            (ins globaladdress:$global, u16Imm:$offset, IntRegs:$src),
            "memw(#$global+$offset) = $src.new",
            []>,
            Requires<[HasV4T]>;

let mayStore = 1, neverHasSideEffects = 1 in
def STw_GP_nv_V4 : NVInst_V4<(outs),
            (ins globaladdress:$global, IntRegs:$src),
            "memw(#$global) = $src.new",
            []>,
            Requires<[HasV4T]>;

// Store new-value word conditionally.

// if ([!]Pv[.new]) memw(#u6)=Nt.new

// if ([!]Pv[.new]) memw(Rs+#u6:2)=Nt.new
// if (Pv) memw(Rs+#u6:2)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STriw_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if ($src1) memw($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memw(Rs+#u6:2)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STriw_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if ($src1.new) memw($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memw(Rs+#u6:2)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STriw_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if (!$src1) memw($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memw(Rs+#u6:2)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STriw_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if (!$src1.new) memw($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memw(Rs+#u6:2)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STriw_indexed_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3, IntRegs:$src4),
            "if ($src1) memw($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memw(Rs+#u6:2)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STriw_indexed_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3, IntRegs:$src4),
            "if ($src1.new) memw($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memw(Rs+#u6:2)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STriw_indexed_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3, IntRegs:$src4),
            "if (!$src1) memw($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memw(Rs+#u6:2)=Nt.new
let mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def STriw_indexed_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3, IntRegs:$src4),
            "if (!$src1.new) memw($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;


// if ([!]Pv[.new]) memw(Rs+Ru<<#u2)=Nt.new
// if (Pv) memw(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10,
    isPredicated = 1 in
def STriw_indexed_shl_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1) memw($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memw(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10,
    isPredicated = 1 in
def STriw_indexed_shl_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1.new) memw($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memw(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10,
    isPredicated = 1 in
def STriw_indexed_shl_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1) memw($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memw(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10,
    isPredicated = 1 in
def STriw_indexed_shl_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1.new) memw($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memw(Rx++#s4:2)=Nt.new
// if (Pv) memw(Rx++#s4:2)=Nt.new
let mayStore = 1, hasCtrlDep = 1,
    isPredicated = 1 in
def POST_STwri_cPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_2Imm:$offset),
            "if ($src1) memw($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (Pv.new) memw(Rx++#s4:2)=Nt.new
let mayStore = 1, hasCtrlDep = 1,
    isPredicated = 1 in
def POST_STwri_cdnPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_2Imm:$offset),
            "if ($src1.new) memw($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (!Pv) memw(Rx++#s4:2)=Nt.new
let mayStore = 1, hasCtrlDep = 1,
    isPredicated = 1 in
def POST_STwri_cNotPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_2Imm:$offset),
            "if (!$src1) memw($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (!Pv.new) memw(Rx++#s4:2)=Nt.new
let mayStore = 1, hasCtrlDep = 1,
    isPredicated = 1 in
def POST_STwri_cdnNotPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_2Imm:$offset),
            "if (!$src1.new) memw($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;



// if (Pv) memb(##global) = Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STb_GP_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1) memb(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memb(##global) = Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STb_GP_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1) memb(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memb(##global) = Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STb_GP_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1.new) memb(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memb(##global) = Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STb_GP_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1.new) memb(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memh(##global) = Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STh_GP_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1) memh(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memh(##global) = Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STh_GP_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1) memh(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memh(##global) = Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STh_GP_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1.new) memh(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memh(##global) = Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STh_GP_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1.new) memh(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memw(##global) = Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STw_GP_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1) memw(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memw(##global) = Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STw_GP_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1) memw(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memw(##global) = Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STw_GP_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1.new) memw(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memw(##global) = Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STw_GP_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1.new) memw(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

let mayStore = 1, neverHasSideEffects = 1 in
def STrib_GP_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if ($src1) memb(##$global+$offset) = $src2.new",
            []>,
            Requires<[HasV4T]>;

let mayStore = 1, neverHasSideEffects = 1 in
def STrib_GP_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if (!$src1) memb(##$global+$offset) = $src2.new",
            []>,
            Requires<[HasV4T]>;

let mayStore = 1, neverHasSideEffects = 1 in
def STrib_GP_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if ($src1.new) memb(##$global+$offset) = $src2.new",
            []>,
            Requires<[HasV4T]>;

let mayStore = 1, neverHasSideEffects = 1 in
def STrib_GP_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if (!$src1.new) memb(##$global+$offset) = $src2.new",
            []>,
            Requires<[HasV4T]>;

let mayStore = 1, neverHasSideEffects = 1 in
def STrih_GP_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if ($src1) memh(##$global+$offset) = $src2.new",
            []>,
            Requires<[HasV4T]>;

let mayStore = 1, neverHasSideEffects = 1 in
def STrih_GP_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if (!$src1) memh(##$global+$offset) = $src2.new",
            []>,
            Requires<[HasV4T]>;

let mayStore = 1, neverHasSideEffects = 1 in
def STrih_GP_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if ($src1.new) memh(##$global+$offset) = $src2.new",
            []>,
            Requires<[HasV4T]>;

let mayStore = 1, neverHasSideEffects = 1 in
def STrih_GP_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if (!$src1.new) memh(##$global+$offset) = $src2.new",
            []>,
            Requires<[HasV4T]>;

let mayStore = 1, neverHasSideEffects = 1 in
def STriw_GP_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if ($src1) memw(##$global+$offset) = $src2.new",
            []>,
            Requires<[HasV4T]>;

let mayStore = 1, neverHasSideEffects = 1 in
def STriw_GP_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if (!$src1) memw(##$global+$offset) = $src2.new",
            []>,
            Requires<[HasV4T]>;

let mayStore = 1, neverHasSideEffects = 1 in
def STriw_GP_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if ($src1.new) memw(##$global+$offset) = $src2.new",
            []>,
            Requires<[HasV4T]>;

let mayStore = 1, neverHasSideEffects = 1 in
def STriw_GP_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, u16Imm:$offset,
                                                        IntRegs:$src2),
            "if (!$src1.new) memw(##$global+$offset) = $src2.new",
            []>,
            Requires<[HasV4T]>;

//===----------------------------------------------------------------------===//
// NV/ST -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// NV/J +
//===----------------------------------------------------------------------===//

multiclass NVJ_type_basic_reg<string NotStr, string OpcStr, string TakenStr> {
  def _ie_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, brtarget:$offset),
            !strconcat("if (", !strconcat(NotStr, !strconcat(OpcStr,
            !strconcat("($src1.new, $src2)) jump:",
            !strconcat(TakenStr, " $offset"))))),
            []>,
            Requires<[HasV4T]>;

  def _nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, brtarget:$offset),
            !strconcat("if (", !strconcat(NotStr, !strconcat(OpcStr,
            !strconcat("($src1.new, $src2)) jump:",
            !strconcat(TakenStr, " $offset"))))),
            []>,
            Requires<[HasV4T]>;
}

multiclass NVJ_type_basic_2ndDotNew<string NotStr, string OpcStr,
                                                   string TakenStr> {
  def _ie_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, brtarget:$offset),
            !strconcat("if (", !strconcat(NotStr, !strconcat(OpcStr,
            !strconcat("($src1, $src2.new)) jump:",
            !strconcat(TakenStr, " $offset"))))),
            []>,
            Requires<[HasV4T]>;

  def _nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, brtarget:$offset),
            !strconcat("if (", !strconcat(NotStr, !strconcat(OpcStr,
            !strconcat("($src1, $src2.new)) jump:",
            !strconcat(TakenStr, " $offset"))))),
            []>,
            Requires<[HasV4T]>;
}

multiclass NVJ_type_basic_imm<string NotStr, string OpcStr, string TakenStr> {
  def _ie_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, u5Imm:$src2, brtarget:$offset),
            !strconcat("if (", !strconcat(NotStr, !strconcat(OpcStr,
            !strconcat("($src1.new, #$src2)) jump:",
            !strconcat(TakenStr, " $offset"))))),
            []>,
            Requires<[HasV4T]>;

  def _nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, u5Imm:$src2, brtarget:$offset),
            !strconcat("if (", !strconcat(NotStr, !strconcat(OpcStr,
            !strconcat("($src1.new, #$src2)) jump:",
            !strconcat(TakenStr, " $offset"))))),
            []>,
            Requires<[HasV4T]>;
}

multiclass NVJ_type_basic_neg<string NotStr, string OpcStr, string TakenStr> {
  def _ie_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, nOneImm:$src2, brtarget:$offset),
            !strconcat("if (", !strconcat(NotStr, !strconcat(OpcStr,
            !strconcat("($src1.new, #$src2)) jump:",
            !strconcat(TakenStr, " $offset"))))),
            []>,
            Requires<[HasV4T]>;

  def _nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, nOneImm:$src2, brtarget:$offset),
            !strconcat("if (", !strconcat(NotStr, !strconcat(OpcStr,
            !strconcat("($src1.new, #$src2)) jump:",
            !strconcat(TakenStr, " $offset"))))),
            []>,
            Requires<[HasV4T]>;
}

multiclass NVJ_type_basic_tstbit<string NotStr, string OpcStr,
                                                string TakenStr> {
  def _ie_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, u1Imm:$src2, brtarget:$offset),
            !strconcat("if (", !strconcat(NotStr, !strconcat(OpcStr,
            !strconcat("($src1.new, #$src2)) jump:",
            !strconcat(TakenStr, " $offset"))))),
            []>,
            Requires<[HasV4T]>;

  def _nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, u1Imm:$src2, brtarget:$offset),
            !strconcat("if (", !strconcat(NotStr, !strconcat(OpcStr,
            !strconcat("($src1.new, #$src2)) jump:",
            !strconcat(TakenStr, " $offset"))))),
            []>,
            Requires<[HasV4T]>;
}

// Multiclass for regular dot new of Ist operand register.
multiclass NVJ_type_br_pred_reg<string NotStr, string OpcStr> {
  defm Pt  : NVJ_type_basic_reg<NotStr, OpcStr, "t">;
  defm Pnt : NVJ_type_basic_reg<NotStr, OpcStr, "nt">;
}

// Multiclass for dot new of 2nd operand register.
multiclass NVJ_type_br_pred_2ndDotNew<string NotStr, string OpcStr> {
  defm Pt  : NVJ_type_basic_2ndDotNew<NotStr, OpcStr, "t">;
  defm Pnt : NVJ_type_basic_2ndDotNew<NotStr, OpcStr, "nt">;
}

// Multiclass for 2nd operand immediate, including -1.
multiclass NVJ_type_br_pred_imm<string NotStr, string OpcStr> {
  defm Pt     : NVJ_type_basic_imm<NotStr, OpcStr, "t">;
  defm Pnt    : NVJ_type_basic_imm<NotStr, OpcStr, "nt">;
  defm Ptneg  : NVJ_type_basic_neg<NotStr, OpcStr, "t">;
  defm Pntneg : NVJ_type_basic_neg<NotStr, OpcStr, "nt">;
}

// Multiclass for 2nd operand immediate, excluding -1.
multiclass NVJ_type_br_pred_imm_only<string NotStr, string OpcStr> {
  defm Pt     : NVJ_type_basic_imm<NotStr, OpcStr, "t">;
  defm Pnt    : NVJ_type_basic_imm<NotStr, OpcStr, "nt">;
}

// Multiclass for tstbit, where 2nd operand is always #0.
multiclass NVJ_type_br_pred_tstbit<string NotStr, string OpcStr> {
  defm Pt     : NVJ_type_basic_tstbit<NotStr, OpcStr, "t">;
  defm Pnt    : NVJ_type_basic_tstbit<NotStr, OpcStr, "nt">;
}

// Multiclass for GT.
multiclass NVJ_type_rr_ri<string OpcStr> {
  defm rrNot   : NVJ_type_br_pred_reg<"!", OpcStr>;
  defm rr      : NVJ_type_br_pred_reg<"",  OpcStr>;
  defm rrdnNot : NVJ_type_br_pred_2ndDotNew<"!", OpcStr>;
  defm rrdn    : NVJ_type_br_pred_2ndDotNew<"",  OpcStr>;
  defm riNot   : NVJ_type_br_pred_imm<"!", OpcStr>;
  defm ri      : NVJ_type_br_pred_imm<"",  OpcStr>;
}

// Multiclass for EQ.
multiclass NVJ_type_rr_ri_no_2ndDotNew<string OpcStr> {
  defm rrNot   : NVJ_type_br_pred_reg<"!", OpcStr>;
  defm rr      : NVJ_type_br_pred_reg<"",  OpcStr>;
  defm riNot   : NVJ_type_br_pred_imm<"!", OpcStr>;
  defm ri      : NVJ_type_br_pred_imm<"",  OpcStr>;
}

// Multiclass for GTU.
multiclass NVJ_type_rr_ri_no_nOne<string OpcStr> {
  defm rrNot   : NVJ_type_br_pred_reg<"!", OpcStr>;
  defm rr      : NVJ_type_br_pred_reg<"",  OpcStr>;
  defm rrdnNot : NVJ_type_br_pred_2ndDotNew<"!", OpcStr>;
  defm rrdn    : NVJ_type_br_pred_2ndDotNew<"",  OpcStr>;
  defm riNot   : NVJ_type_br_pred_imm_only<"!", OpcStr>;
  defm ri      : NVJ_type_br_pred_imm_only<"",  OpcStr>;
}

// Multiclass for tstbit.
multiclass NVJ_type_r0<string OpcStr> {
  defm r0Not : NVJ_type_br_pred_tstbit<"!", OpcStr>;
  defm r0    : NVJ_type_br_pred_tstbit<"",  OpcStr>;
 }

// Base Multiclass for New Value Jump.
multiclass NVJ_type {
  defm GT     : NVJ_type_rr_ri<"cmp.gt">;
  defm EQ     : NVJ_type_rr_ri_no_2ndDotNew<"cmp.eq">;
  defm GTU    : NVJ_type_rr_ri_no_nOne<"cmp.gtu">;
  defm TSTBIT : NVJ_type_r0<"tstbit">;
}

let isBranch = 1, isTerminator=1, neverHasSideEffects = 1, Defs = [PC] in {
  defm JMP_ : NVJ_type;
}

//===----------------------------------------------------------------------===//
// NV/J -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// XTYPE/ALU +
//===----------------------------------------------------------------------===//

//  Add and accumulate.
//  Rd=add(Rs,add(Ru,#s6))
def ADDr_ADDri_V4 : MInst<(outs IntRegs:$dst),
          (ins IntRegs:$src1, IntRegs:$src2, s6Imm:$src3),
          "$dst = add($src1, add($src2, #$src3))",
          [(set (i32 IntRegs:$dst),
           (add (i32 IntRegs:$src1), (add (i32 IntRegs:$src2),
                                          s6ImmPred:$src3)))]>,
          Requires<[HasV4T]>;

//  Rd=add(Rs,sub(#s6,Ru))
def ADDr_SUBri_V4 : MInst<(outs IntRegs:$dst),
          (ins IntRegs:$src1, s6Imm:$src2, IntRegs:$src3),
          "$dst = add($src1, sub(#$src2, $src3))",
          [(set (i32 IntRegs:$dst),
           (add (i32 IntRegs:$src1), (sub s6ImmPred:$src2,
                                          (i32 IntRegs:$src3))))]>,
          Requires<[HasV4T]>;

// Generates the same instruction as ADDr_SUBri_V4 but matches different
// pattern.
//  Rd=add(Rs,sub(#s6,Ru))
def ADDri_SUBr_V4 : MInst<(outs IntRegs:$dst),
          (ins IntRegs:$src1, s6Imm:$src2, IntRegs:$src3),
          "$dst = add($src1, sub(#$src2, $src3))",
          [(set (i32 IntRegs:$dst),
                (sub (add (i32 IntRegs:$src1), s6ImmPred:$src2),
                     (i32 IntRegs:$src3)))]>,
          Requires<[HasV4T]>;


//  Add or subtract doublewords with carry.
//TODO:
//  Rdd=add(Rss,Rtt,Px):carry
//TODO:
//  Rdd=sub(Rss,Rtt,Px):carry


//  Logical doublewords.
//  Rdd=and(Rtt,~Rss)
def ANDd_NOTd_V4 : MInst<(outs DoubleRegs:$dst),
          (ins DoubleRegs:$src1, DoubleRegs:$src2),
          "$dst = and($src1, ~$src2)",
          [(set (i64 DoubleRegs:$dst), (and (i64 DoubleRegs:$src1),
                                      (not (i64 DoubleRegs:$src2))))]>,
          Requires<[HasV4T]>;

//  Rdd=or(Rtt,~Rss)
def ORd_NOTd_V4 : MInst<(outs DoubleRegs:$dst),
          (ins DoubleRegs:$src1, DoubleRegs:$src2),
          "$dst = or($src1, ~$src2)",
          [(set (i64 DoubleRegs:$dst),
           (or (i64 DoubleRegs:$src1), (not (i64 DoubleRegs:$src2))))]>,
          Requires<[HasV4T]>;


//  Logical-logical doublewords.
//  Rxx^=xor(Rss,Rtt)
def XORd_XORdd: MInst_acc<(outs DoubleRegs:$dst),
          (ins DoubleRegs:$src1, DoubleRegs:$src2, DoubleRegs:$src3),
          "$dst ^= xor($src2, $src3)",
          [(set (i64 DoubleRegs:$dst),
           (xor (i64 DoubleRegs:$src1), (xor (i64 DoubleRegs:$src2),
                                             (i64 DoubleRegs:$src3))))],
          "$src1 = $dst">,
          Requires<[HasV4T]>;


// Logical-logical words.
// Rx=or(Ru,and(Rx,#s10))
def ORr_ANDri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, s10Imm:$src3),
            "$dst = or($src1, and($src2, #$src3))",
            [(set (i32 IntRegs:$dst),
                  (or (i32 IntRegs:$src1), (and (i32 IntRegs:$src2),
                                                s10ImmPred:$src3)))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

// Rx[&|^]=and(Rs,Rt)
// Rx&=and(Rs,Rt)
def ANDr_ANDrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst &= and($src2, $src3)",
            [(set (i32 IntRegs:$dst),
                  (and (i32 IntRegs:$src1), (and (i32 IntRegs:$src2),
                                                 (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx|=and(Rs,Rt)
def ORr_ANDrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst |= and($src2, $src3)",
            [(set (i32 IntRegs:$dst),
                  (or (i32 IntRegs:$src1), (and (i32 IntRegs:$src2),
                                                (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx^=and(Rs,Rt)
def XORr_ANDrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst ^= and($src2, $src3)",
            [(set (i32 IntRegs:$dst),
             (xor (i32 IntRegs:$src1), (and (i32 IntRegs:$src2),
                                            (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx[&|^]=and(Rs,~Rt)
// Rx&=and(Rs,~Rt)
def ANDr_ANDr_NOTr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst &= and($src2, ~$src3)",
            [(set (i32 IntRegs:$dst),
                  (and (i32 IntRegs:$src1), (and (i32 IntRegs:$src2),
                                                 (not (i32 IntRegs:$src3)))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx|=and(Rs,~Rt)
def ORr_ANDr_NOTr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst |= and($src2, ~$src3)",
            [(set (i32 IntRegs:$dst),
             (or (i32 IntRegs:$src1), (and (i32 IntRegs:$src2),
                                           (not (i32 IntRegs:$src3)))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx^=and(Rs,~Rt)
def XORr_ANDr_NOTr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst ^= and($src2, ~$src3)",
            [(set (i32 IntRegs:$dst),
             (xor (i32 IntRegs:$src1), (and (i32 IntRegs:$src2),
                                            (not (i32 IntRegs:$src3)))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx[&|^]=or(Rs,Rt)
// Rx&=or(Rs,Rt)
def ANDr_ORrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst &= or($src2, $src3)",
            [(set (i32 IntRegs:$dst),
                  (and (i32 IntRegs:$src1), (or (i32 IntRegs:$src2),
                                                (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx|=or(Rs,Rt)
def ORr_ORrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst |= or($src2, $src3)",
            [(set (i32 IntRegs:$dst),
                  (or (i32 IntRegs:$src1), (or (i32 IntRegs:$src2),
                                               (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx^=or(Rs,Rt)
def XORr_ORrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst ^= or($src2, $src3)",
            [(set (i32 IntRegs:$dst),
             (xor (i32 IntRegs:$src1), (or (i32 IntRegs:$src2),
                                           (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx[&|^]=xor(Rs,Rt)
// Rx&=xor(Rs,Rt)
def ANDr_XORrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst &= xor($src2, $src3)",
            [(set (i32 IntRegs:$dst),
                  (and (i32 IntRegs:$src1), (xor (i32 IntRegs:$src2),
                                                 (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx|=xor(Rs,Rt)
def ORr_XORrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst |= xor($src2, $src3)",
            [(set (i32 IntRegs:$dst),
                  (and (i32 IntRegs:$src1), (xor (i32 IntRegs:$src2),
                                                 (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx^=xor(Rs,Rt)
def XORr_XORrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst ^= xor($src2, $src3)",
            [(set (i32 IntRegs:$dst),
             (and (i32 IntRegs:$src1), (xor (i32 IntRegs:$src2),
                                            (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx|=and(Rs,#s10)
def ORr_ANDri2_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, s10Imm:$src3),
            "$dst |= and($src2, #$src3)",
            [(set (i32 IntRegs:$dst),
                  (or (i32 IntRegs:$src1), (and (i32 IntRegs:$src2),
                                                s10ImmPred:$src3)))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx|=or(Rs,#s10)
def ORr_ORri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, s10Imm:$src3),
            "$dst |= or($src2, #$src3)",
            [(set (i32 IntRegs:$dst),
                  (or (i32 IntRegs:$src1), (and (i32 IntRegs:$src2),
                                                s10ImmPred:$src3)))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;


//    Modulo wrap
//        Rd=modwrap(Rs,Rt)
//    Round
//        Rd=cround(Rs,#u5)
//        Rd=cround(Rs,Rt)
//        Rd=round(Rs,#u5)[:sat]
//        Rd=round(Rs,Rt)[:sat]
//    Vector reduce add unsigned halfwords
//        Rd=vraddh(Rss,Rtt)
//    Vector add bytes
//        Rdd=vaddb(Rss,Rtt)
//    Vector conditional negate
//        Rdd=vcnegh(Rss,Rt)
//        Rxx+=vrcnegh(Rss,Rt)
//    Vector maximum bytes
//        Rdd=vmaxb(Rtt,Rss)
//    Vector reduce maximum halfwords
//        Rxx=vrmaxh(Rss,Ru)
//        Rxx=vrmaxuh(Rss,Ru)
//    Vector reduce maximum words
//        Rxx=vrmaxuw(Rss,Ru)
//        Rxx=vrmaxw(Rss,Ru)
//    Vector minimum bytes
//        Rdd=vminb(Rtt,Rss)
//    Vector reduce minimum halfwords
//        Rxx=vrminh(Rss,Ru)
//        Rxx=vrminuh(Rss,Ru)
//    Vector reduce minimum words
//        Rxx=vrminuw(Rss,Ru)
//        Rxx=vrminw(Rss,Ru)
//    Vector subtract bytes
//        Rdd=vsubb(Rss,Rtt)

//===----------------------------------------------------------------------===//
// XTYPE/ALU -
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// XTYPE/MPY +
//===----------------------------------------------------------------------===//

// Multiply and user lower result.
// Rd=add(#u6,mpyi(Rs,#U6))
def ADDi_MPYri_V4 : MInst<(outs IntRegs:$dst),
            (ins u6Imm:$src1, IntRegs:$src2, u6Imm:$src3),
            "$dst = add(#$src1, mpyi($src2, #$src3))",
            [(set (i32 IntRegs:$dst),
                  (add (mul (i32 IntRegs:$src2), u6ImmPred:$src3),
                       u6ImmPred:$src1))]>,
            Requires<[HasV4T]>;

// Rd=add(#u6,mpyi(Rs,Rt))

def ADDi_MPYrr_V4 : MInst<(outs IntRegs:$dst),
            (ins u6Imm:$src1, IntRegs:$src2, IntRegs:$src3),
            "$dst = add(#$src1, mpyi($src2, $src3))",
            [(set (i32 IntRegs:$dst),
                  (add (mul (i32 IntRegs:$src2), (i32 IntRegs:$src3)),
                       u6ImmPred:$src1))]>,
            Requires<[HasV4T]>;

// Rd=add(Ru,mpyi(#u6:2,Rs))
def ADDr_MPYir_V4 : MInst<(outs IntRegs:$dst),
            (ins IntRegs:$src1, u6Imm:$src2, IntRegs:$src3),
            "$dst = add($src1, mpyi(#$src2, $src3))",
            [(set (i32 IntRegs:$dst),
             (add (i32 IntRegs:$src1), (mul (i32 IntRegs:$src3),
                                            u6_2ImmPred:$src2)))]>,
            Requires<[HasV4T]>;

// Rd=add(Ru,mpyi(Rs,#u6))
def ADDr_MPYri_V4 : MInst<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2, u6Imm:$src3),
            "$dst = add($src1, mpyi($src2, #$src3))",
            [(set (i32 IntRegs:$dst),
                  (add (i32 IntRegs:$src1), (mul (i32 IntRegs:$src2),
                                                 u6ImmPred:$src3)))]>,
            Requires<[HasV4T]>;

// Rx=add(Ru,mpyi(Rx,Rs))
def ADDr_MPYrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "$dst = add($src1, mpyi($src2, $src3))",
            [(set (i32 IntRegs:$dst),
             (add (i32 IntRegs:$src1), (mul (i32 IntRegs:$src2),
                                            (i32 IntRegs:$src3))))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;


// Polynomial multiply words
// Rdd=pmpyw(Rs,Rt)
// Rxx^=pmpyw(Rs,Rt)

// Vector reduce multiply word by signed half (32x16)
// Rdd=vrmpyweh(Rss,Rtt)[:<<1]
// Rdd=vrmpywoh(Rss,Rtt)[:<<1]
// Rxx+=vrmpyweh(Rss,Rtt)[:<<1]
// Rxx+=vrmpywoh(Rss,Rtt)[:<<1]

// Multiply and use upper result
// Rd=mpy(Rs,Rt.H):<<1:sat
// Rd=mpy(Rs,Rt.L):<<1:sat
// Rd=mpy(Rs,Rt):<<1
// Rd=mpy(Rs,Rt):<<1:sat
// Rd=mpysu(Rs,Rt)
// Rx+=mpy(Rs,Rt):<<1:sat
// Rx-=mpy(Rs,Rt):<<1:sat

// Vector multiply bytes
// Rdd=vmpybsu(Rs,Rt)
// Rdd=vmpybu(Rs,Rt)
// Rxx+=vmpybsu(Rs,Rt)
// Rxx+=vmpybu(Rs,Rt)

// Vector polynomial multiply halfwords
// Rdd=vpmpyh(Rs,Rt)
// Rxx^=vpmpyh(Rs,Rt)

//===----------------------------------------------------------------------===//
// XTYPE/MPY -
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// XTYPE/SHIFT +
//===----------------------------------------------------------------------===//

// Shift by immediate and accumulate.
// Rx=add(#u8,asl(Rx,#U5))
def ADDi_ASLri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Imm:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = add(#$src1, asl($src2, #$src3))",
            [(set (i32 IntRegs:$dst),
                  (add (shl (i32 IntRegs:$src2), u5ImmPred:$src3),
                       u8ImmPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

// Rx=add(#u8,lsr(Rx,#U5))
def ADDi_LSRri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Imm:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = add(#$src1, lsr($src2, #$src3))",
            [(set (i32 IntRegs:$dst),
                  (add (srl (i32 IntRegs:$src2), u5ImmPred:$src3),
                       u8ImmPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

// Rx=sub(#u8,asl(Rx,#U5))
def SUBi_ASLri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Imm:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = sub(#$src1, asl($src2, #$src3))",
            [(set (i32 IntRegs:$dst),
                  (sub (shl (i32 IntRegs:$src2), u5ImmPred:$src3),
                       u8ImmPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

// Rx=sub(#u8,lsr(Rx,#U5))
def SUBi_LSRri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Imm:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = sub(#$src1, lsr($src2, #$src3))",
            [(set (i32 IntRegs:$dst),
                  (sub (srl (i32 IntRegs:$src2), u5ImmPred:$src3),
                       u8ImmPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;


//Shift by immediate and logical.
//Rx=and(#u8,asl(Rx,#U5))
def ANDi_ASLri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Imm:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = and(#$src1, asl($src2, #$src3))",
            [(set (i32 IntRegs:$dst),
                  (and (shl (i32 IntRegs:$src2), u5ImmPred:$src3),
                       u8ImmPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

//Rx=and(#u8,lsr(Rx,#U5))
def ANDi_LSRri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Imm:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = and(#$src1, lsr($src2, #$src3))",
            [(set (i32 IntRegs:$dst),
                  (and (srl (i32 IntRegs:$src2), u5ImmPred:$src3),
                       u8ImmPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

//Rx=or(#u8,asl(Rx,#U5))
let AddedComplexity = 30 in
def ORi_ASLri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Imm:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = or(#$src1, asl($src2, #$src3))",
            [(set (i32 IntRegs:$dst),
                  (or (shl (i32 IntRegs:$src2), u5ImmPred:$src3),
                      u8ImmPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

//Rx=or(#u8,lsr(Rx,#U5))
let AddedComplexity = 30 in
def ORi_LSRri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Imm:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = or(#$src1, lsr($src2, #$src3))",
            [(set (i32 IntRegs:$dst),
                  (or (srl (i32 IntRegs:$src2), u5ImmPred:$src3),
                      u8ImmPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;


//Shift by register.
//Rd=lsl(#s6,Rt)
def LSLi_V4 : MInst<(outs IntRegs:$dst), (ins s6Imm:$src1, IntRegs:$src2),
            "$dst = lsl(#$src1, $src2)",
            [(set (i32 IntRegs:$dst), (shl s6ImmPred:$src1,
                                           (i32 IntRegs:$src2)))]>,
            Requires<[HasV4T]>;


//Shift by register and logical.
//Rxx^=asl(Rss,Rt)
def ASLd_rr_xor_V4 : MInst_acc<(outs DoubleRegs:$dst),
            (ins DoubleRegs:$src1, DoubleRegs:$src2, IntRegs:$src3),
            "$dst ^= asl($src2, $src3)",
            [(set (i64 DoubleRegs:$dst),
                  (xor (i64 DoubleRegs:$src1), (shl (i64 DoubleRegs:$src2),
                                                    (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

//Rxx^=asr(Rss,Rt)
def ASRd_rr_xor_V4 : MInst_acc<(outs DoubleRegs:$dst),
            (ins DoubleRegs:$src1, DoubleRegs:$src2, IntRegs:$src3),
            "$dst ^= asr($src2, $src3)",
            [(set (i64 DoubleRegs:$dst),
                  (xor (i64 DoubleRegs:$src1), (sra (i64 DoubleRegs:$src2),
                                                    (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

//Rxx^=lsl(Rss,Rt)
def LSLd_rr_xor_V4 : MInst_acc<(outs DoubleRegs:$dst),
            (ins DoubleRegs:$src1, DoubleRegs:$src2, IntRegs:$src3),
            "$dst ^= lsl($src2, $src3)",
            [(set (i64 DoubleRegs:$dst), (xor (i64 DoubleRegs:$src1),
                                              (shl (i64 DoubleRegs:$src2),
                                                   (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

//Rxx^=lsr(Rss,Rt)
def LSRd_rr_xor_V4 : MInst_acc<(outs DoubleRegs:$dst),
            (ins DoubleRegs:$src1, DoubleRegs:$src2, IntRegs:$src3),
            "$dst ^= lsr($src2, $src3)",
            [(set (i64 DoubleRegs:$dst),
                  (xor (i64 DoubleRegs:$src1), (srl (i64 DoubleRegs:$src2),
                                                    (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;


//===----------------------------------------------------------------------===//
// XTYPE/SHIFT -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MEMOP: Word, Half, Byte
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MEMOP: Word
//
//  Implemented:
//     MEMw_ADDi_indexed_V4  : memw(Rs+#u6:2)+=#U5
//     MEMw_SUBi_indexed_V4  : memw(Rs+#u6:2)-=#U5
//     MEMw_ADDr_indexed_V4  : memw(Rs+#u6:2)+=Rt
//     MEMw_SUBr_indexed_V4  : memw(Rs+#u6:2)-=Rt
//     MEMw_CLRr_indexed_V4  : memw(Rs+#u6:2)&=Rt
//     MEMw_SETr_indexed_V4  : memw(Rs+#u6:2)|=Rt
//     MEMw_ADDi_V4          : memw(Rs+#u6:2)+=#U5
//     MEMw_SUBi_V4          : memw(Rs+#u6:2)-=#U5
//     MEMw_ADDr_V4          : memw(Rs+#u6:2)+=Rt
//     MEMw_SUBr_V4          : memw(Rs+#u6:2)-=Rt
//     MEMw_CLRr_V4          : memw(Rs+#u6:2)&=Rt
//     MEMw_SETr_V4          : memw(Rs+#u6:2)|=Rt
//
//   Not implemented:
//     MEMw_CLRi_indexed_V4  : memw(Rs+#u6:2)=clrbit(#U5)
//     MEMw_SETi_indexed_V4  : memw(Rs+#u6:2)=setbit(#U5)
//     MEMw_CLRi_V4          : memw(Rs+#u6:2)=clrbit(#U5)
//     MEMw_SETi_V4          : memw(Rs+#u6:2)=setbit(#U5)
//===----------------------------------------------------------------------===//


// MEMw_ADDSUBi_indexed_V4:
//   pseudo operation for MEMw_ADDi_indexed_V4 and
//   MEMw_SUBi_indexed_V4 a later pass will change it
//   to the corresponding pattern.
let AddedComplexity = 30 in
def MEMw_ADDSUBi_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_2Imm:$offset, m6Imm:$addend),
            "Error; should not emit",
            [(store (add (load (add (i32 IntRegs:$base), u6_2ImmPred:$offset)),
                         m6ImmPred:$addend),
                    (add (i32 IntRegs:$base), u6_2ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) += #U5
let AddedComplexity = 30 in
def MEMw_ADDi_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_2Imm:$offset, u5Imm:$addend),
            "memw($base+#$offset) += #$addend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) -= #U5
let AddedComplexity = 30 in
def MEMw_SUBi_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_2Imm:$offset, u5Imm:$subend),
            "memw($base+#$offset) -= #$subend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) += Rt
let AddedComplexity = 30 in
def MEMw_ADDr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_2Imm:$offset, IntRegs:$addend),
            "memw($base+#$offset) += $addend",
            [(store (add (load (add (i32 IntRegs:$base), u6_2ImmPred:$offset)),
                         (i32 IntRegs:$addend)),
                    (add (i32 IntRegs:$base), u6_2ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) -= Rt
let AddedComplexity = 30 in
def MEMw_SUBr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_2Imm:$offset, IntRegs:$subend),
            "memw($base+#$offset) -= $subend",
            [(store (sub (load (add (i32 IntRegs:$base), u6_2ImmPred:$offset)),
                         (i32 IntRegs:$subend)),
                    (add (i32 IntRegs:$base), u6_2ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) &= Rt
let AddedComplexity = 30 in
def MEMw_ANDr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_2Imm:$offset, IntRegs:$andend),
            "memw($base+#$offset) &= $andend",
            [(store (and (load (add (i32 IntRegs:$base), u6_2ImmPred:$offset)),
                         (i32 IntRegs:$andend)),
                    (add (i32 IntRegs:$base), u6_2ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) |= Rt
let AddedComplexity = 30 in
def MEMw_ORr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_2Imm:$offset, IntRegs:$orend),
            "memw($base+#$offset) |= $orend",
            [(store (or (load (add (i32 IntRegs:$base), u6_2ImmPred:$offset)),
                        (i32 IntRegs:$orend)),
                    (add (i32 IntRegs:$base), u6_2ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// MEMw_ADDSUBi_V4:
//   Pseudo operation for MEMw_ADDi_V4 and MEMw_SUBi_V4
//   a later pass will change it to the right pattern.
let AddedComplexity = 30 in
def MEMw_ADDSUBi_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, m6Imm:$addend),
            "Error; should not emit",
            [(store (add (load ADDRriU6_2:$addr), m6ImmPred:$addend),
                    ADDRriU6_2:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) += #U5
let AddedComplexity = 30 in
def MEMw_ADDi_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, u5Imm:$addend),
            "memw($addr) += $addend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) -= #U5
let AddedComplexity = 30 in
def MEMw_SUBi_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, u5Imm:$subend),
            "memw($addr) -= $subend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) += Rt
let AddedComplexity = 30 in
def MEMw_ADDr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$addend),
            "memw($addr) += $addend",
            [(store (add (load ADDRriU6_2:$addr), (i32 IntRegs:$addend)),
                    ADDRriU6_2:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) -= Rt
let AddedComplexity = 30 in
def MEMw_SUBr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$subend),
            "memw($addr) -= $subend",
            [(store (sub (load ADDRriU6_2:$addr), (i32 IntRegs:$subend)),
                    ADDRriU6_2:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) &= Rt
let AddedComplexity = 30 in
def MEMw_ANDr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$andend),
            "memw($addr) &= $andend",
            [(store (and (load ADDRriU6_2:$addr), (i32 IntRegs:$andend)),
                    ADDRriU6_2:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) |= Rt
let AddedComplexity = 30 in
def MEMw_ORr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$orend),
            "memw($addr) |= $orend",
            [(store (or (load ADDRriU6_2:$addr), (i32 IntRegs:$orend)),
                    ADDRriU6_2:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

//===----------------------------------------------------------------------===//
// MEMOP: Halfword
//
//  Implemented:
//     MEMh_ADDi_indexed_V4  : memw(Rs+#u6:2)+=#U5
//     MEMh_SUBi_indexed_V4  : memw(Rs+#u6:2)-=#U5
//     MEMh_ADDr_indexed_V4  : memw(Rs+#u6:2)+=Rt
//     MEMh_SUBr_indexed_V4  : memw(Rs+#u6:2)-=Rt
//     MEMh_CLRr_indexed_V4  : memw(Rs+#u6:2)&=Rt
//     MEMh_SETr_indexed_V4  : memw(Rs+#u6:2)|=Rt
//     MEMh_ADDi_V4          : memw(Rs+#u6:2)+=#U5
//     MEMh_SUBi_V4          : memw(Rs+#u6:2)-=#U5
//     MEMh_ADDr_V4          : memw(Rs+#u6:2)+=Rt
//     MEMh_SUBr_V4          : memw(Rs+#u6:2)-=Rt
//     MEMh_CLRr_V4          : memw(Rs+#u6:2)&=Rt
//     MEMh_SETr_V4          : memw(Rs+#u6:2)|=Rt
//
//   Not implemented:
//     MEMh_CLRi_indexed_V4  : memw(Rs+#u6:2)=clrbit(#U5)
//     MEMh_SETi_indexed_V4  : memw(Rs+#u6:2)=setbit(#U5)
//     MEMh_CLRi_V4          : memw(Rs+#u6:2)=clrbit(#U5)
//     MEMh_SETi_V4          : memw(Rs+#u6:2)=setbit(#U5)
//===----------------------------------------------------------------------===//


// MEMh_ADDSUBi_indexed_V4:
//   Pseudo operation for MEMh_ADDi_indexed_V4 and
//   MEMh_SUBi_indexed_V4 a later pass will change it
//   to the corresponding pattern.
let AddedComplexity = 30 in
def MEMh_ADDSUBi_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_1Imm:$offset, m6Imm:$addend),
            "Error; should not emit",
            [(truncstorei16 (add (sextloadi16 (add (i32 IntRegs:$base),
                                                   u6_1ImmPred:$offset)),
                                 m6ImmPred:$addend),
                            (add (i32 IntRegs:$base), u6_1ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) += #U5
let AddedComplexity = 30 in
def MEMh_ADDi_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_1Imm:$offset, u5Imm:$addend),
            "memh($base+#$offset) += $addend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) -= #U5
let AddedComplexity = 30 in
def MEMh_SUBi_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_1Imm:$offset, u5Imm:$subend),
            "memh($base+#$offset) -= $subend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) += Rt
let AddedComplexity = 30 in
def MEMh_ADDr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_1Imm:$offset, IntRegs:$addend),
            "memh($base+#$offset) += $addend",
            [(truncstorei16 (add (sextloadi16 (add (i32 IntRegs:$base),
                                                   u6_1ImmPred:$offset)),
                                 (i32 IntRegs:$addend)),
                            (add (i32 IntRegs:$base), u6_1ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) -= Rt
let AddedComplexity = 30 in
def MEMh_SUBr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_1Imm:$offset, IntRegs:$subend),
            "memh($base+#$offset) -= $subend",
            [(truncstorei16 (sub (sextloadi16 (add (i32 IntRegs:$base),
                                                   u6_1ImmPred:$offset)),
                                 (i32 IntRegs:$subend)),
                            (add (i32 IntRegs:$base), u6_1ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) &= Rt
let AddedComplexity = 30 in
def MEMh_ANDr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_1Imm:$offset, IntRegs:$andend),
            "memh($base+#$offset) += $andend",
            [(truncstorei16 (and (sextloadi16 (add (i32 IntRegs:$base),
                                                   u6_1ImmPred:$offset)),
                                 (i32 IntRegs:$andend)),
                            (add (i32 IntRegs:$base), u6_1ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) |= Rt
let AddedComplexity = 30 in
def MEMh_ORr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_1Imm:$offset, IntRegs:$orend),
            "memh($base+#$offset) |= $orend",
            [(truncstorei16 (or (sextloadi16 (add (i32 IntRegs:$base),
                                              u6_1ImmPred:$offset)),
                             (i32 IntRegs:$orend)),
                            (add (i32 IntRegs:$base), u6_1ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// MEMh_ADDSUBi_V4:
//   Pseudo operation for MEMh_ADDi_V4 and MEMh_SUBi_V4
//   a later pass will change it to the right pattern.
let AddedComplexity = 30 in
def MEMh_ADDSUBi_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, m6Imm:$addend),
            "Error; should not emit",
            [(truncstorei16 (add (sextloadi16 ADDRriU6_1:$addr),
                                 m6ImmPred:$addend), ADDRriU6_1:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) += #U5
let AddedComplexity = 30 in
def MEMh_ADDi_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, u5Imm:$addend),
            "memh($addr) += $addend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) -= #U5
let AddedComplexity = 30 in
def MEMh_SUBi_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, u5Imm:$subend),
            "memh($addr) -= $subend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) += Rt
let AddedComplexity = 30 in
def MEMh_ADDr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$addend),
            "memh($addr) += $addend",
            [(truncstorei16 (add (sextloadi16 ADDRriU6_1:$addr),
                                 (i32 IntRegs:$addend)), ADDRriU6_1:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) -= Rt
let AddedComplexity = 30 in
def MEMh_SUBr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$subend),
            "memh($addr) -= $subend",
            [(truncstorei16 (sub (sextloadi16 ADDRriU6_1:$addr),
                                 (i32 IntRegs:$subend)), ADDRriU6_1:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) &= Rt
let AddedComplexity = 30 in
def MEMh_ANDr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$andend),
            "memh($addr) &= $andend",
            [(truncstorei16 (and (sextloadi16 ADDRriU6_1:$addr),
                                 (i32 IntRegs:$andend)), ADDRriU6_1:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) |= Rt
let AddedComplexity = 30 in
def MEMh_ORr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$orend),
            "memh($addr) |= $orend",
            [(truncstorei16 (or (sextloadi16 ADDRriU6_1:$addr),
                                (i32 IntRegs:$orend)), ADDRriU6_1:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;


//===----------------------------------------------------------------------===//
// MEMOP: Byte
//
//  Implemented:
//     MEMb_ADDi_indexed_V4  : memb(Rs+#u6:0)+=#U5
//     MEMb_SUBi_indexed_V4  : memb(Rs+#u6:0)-=#U5
//     MEMb_ADDr_indexed_V4  : memb(Rs+#u6:0)+=Rt
//     MEMb_SUBr_indexed_V4  : memb(Rs+#u6:0)-=Rt
//     MEMb_CLRr_indexed_V4  : memb(Rs+#u6:0)&=Rt
//     MEMb_SETr_indexed_V4  : memb(Rs+#u6:0)|=Rt
//     MEMb_ADDi_V4          : memb(Rs+#u6:0)+=#U5
//     MEMb_SUBi_V4          : memb(Rs+#u6:0)-=#U5
//     MEMb_ADDr_V4          : memb(Rs+#u6:0)+=Rt
//     MEMb_SUBr_V4          : memb(Rs+#u6:0)-=Rt
//     MEMb_CLRr_V4          : memb(Rs+#u6:0)&=Rt
//     MEMb_SETr_V4          : memb(Rs+#u6:0)|=Rt
//
//   Not implemented:
//     MEMb_CLRi_indexed_V4  : memb(Rs+#u6:0)=clrbit(#U5)
//     MEMb_SETi_indexed_V4  : memb(Rs+#u6:0)=setbit(#U5)
//     MEMb_CLRi_V4          : memb(Rs+#u6:0)=clrbit(#U5)
//     MEMb_SETi_V4          : memb(Rs+#u6:0)=setbit(#U5)
//===----------------------------------------------------------------------===//


// MEMb_ADDSUBi_indexed_V4:
//   Pseudo operation for MEMb_ADDi_indexed_V4 and
//   MEMb_SUBi_indexed_V4 a later pass will change it
//   to the corresponding pattern.
let AddedComplexity = 30 in
def MEMb_ADDSUBi_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_0Imm:$offset, m6Imm:$addend),
            "Error; should not emit",
            [(truncstorei8 (add (sextloadi8 (add (i32 IntRegs:$base),
                                                 u6_0ImmPred:$offset)),
                                m6ImmPred:$addend),
                           (add (i32 IntRegs:$base), u6_0ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) += #U5
let AddedComplexity = 30 in
def MEMb_ADDi_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_0Imm:$offset, u5Imm:$addend),
            "memb($base+#$offset) += $addend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) -= #U5
let AddedComplexity = 30 in
def MEMb_SUBi_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_0Imm:$offset, u5Imm:$subend),
            "memb($base+#$offset) -= $subend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) += Rt
let AddedComplexity = 30 in
def MEMb_ADDr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_0Imm:$offset, IntRegs:$addend),
            "memb($base+#$offset) += $addend",
            [(truncstorei8 (add (sextloadi8 (add (i32 IntRegs:$base),
                                                 u6_0ImmPred:$offset)),
                                (i32 IntRegs:$addend)),
                           (add (i32 IntRegs:$base), u6_0ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) -= Rt
let AddedComplexity = 30 in
def MEMb_SUBr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_0Imm:$offset, IntRegs:$subend),
            "memb($base+#$offset) -= $subend",
            [(truncstorei8 (sub (sextloadi8 (add (i32 IntRegs:$base),
                                                 u6_0ImmPred:$offset)),
                                (i32 IntRegs:$subend)),
                           (add (i32 IntRegs:$base), u6_0ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) &= Rt
let AddedComplexity = 30 in
def MEMb_ANDr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_0Imm:$offset, IntRegs:$andend),
            "memb($base+#$offset) += $andend",
            [(truncstorei8 (and (sextloadi8 (add (i32 IntRegs:$base),
                                                 u6_0ImmPred:$offset)),
                                (i32 IntRegs:$andend)),
                           (add (i32 IntRegs:$base), u6_0ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) |= Rt
let AddedComplexity = 30 in
def MEMb_ORr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_0Imm:$offset, IntRegs:$orend),
            "memb($base+#$offset) |= $orend",
            [(truncstorei8 (or (sextloadi8 (add (i32 IntRegs:$base),
                                                u6_0ImmPred:$offset)),
                               (i32 IntRegs:$orend)),
                           (add (i32 IntRegs:$base), u6_0ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// MEMb_ADDSUBi_V4:
//   Pseudo operation for MEMb_ADDi_V4 and MEMb_SUBi_V4
//   a later pass will change it to the right pattern.
let AddedComplexity = 30 in
def MEMb_ADDSUBi_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, m6Imm:$addend),
            "Error; should not emit",
            [(truncstorei8 (add (sextloadi8 ADDRriU6_0:$addr),
                                m6ImmPred:$addend), ADDRriU6_0:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) += #U5
let AddedComplexity = 30 in
def MEMb_ADDi_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, u5Imm:$addend),
            "memb($addr) += $addend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) -= #U5
let AddedComplexity = 30 in
def MEMb_SUBi_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, u5Imm:$subend),
            "memb($addr) -= $subend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) += Rt
let AddedComplexity = 30 in
def MEMb_ADDr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$addend),
            "memb($addr) += $addend",
            [(truncstorei8 (add (sextloadi8 ADDRriU6_0:$addr),
                                (i32 IntRegs:$addend)), ADDRriU6_0:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) -= Rt
let AddedComplexity = 30 in
def MEMb_SUBr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$subend),
            "memb($addr) -= $subend",
            [(truncstorei8 (sub (sextloadi8 ADDRriU6_0:$addr),
                                (i32 IntRegs:$subend)), ADDRriU6_0:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) &= Rt
let AddedComplexity = 30 in
def MEMb_ANDr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$andend),
            "memb($addr) &= $andend",
            [(truncstorei8 (and (sextloadi8 ADDRriU6_0:$addr),
                                (i32 IntRegs:$andend)), ADDRriU6_0:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) |= Rt
let AddedComplexity = 30 in
def MEMb_ORr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$orend),
            "memb($addr) |= $orend",
            [(truncstorei8 (or (sextloadi8 ADDRriU6_0:$addr),
                               (i32 IntRegs:$orend)), ADDRriU6_0:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;


//===----------------------------------------------------------------------===//
// XTYPE/PRED +
//===----------------------------------------------------------------------===//

// Hexagon V4 only supports these flavors of byte/half compare instructions:
// EQ/GT/GTU. Other flavors like GE/GEU/LT/LTU/LE/LEU are not supported by
// hardware. However, compiler can still implement these patterns through
// appropriate patterns combinations based on current implemented patterns.
// The implemented patterns are: EQ/GT/GTU.
// Missing patterns are: GE/GEU/LT/LTU/LE/LEU.

// Following instruction is not being extended as it results into the
// incorrect code for negative numbers.
// Pd=cmpb.eq(Rs,#u8)

let isCompare = 1 in
def CMPbEQri_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, u8Imm:$src2),
            "$dst = cmpb.eq($src1, #$src2)",
            [(set (i1 PredRegs:$dst),
                  (seteq (and (i32 IntRegs:$src1), 255), u8ImmPred:$src2))]>,
            Requires<[HasV4T]>;

// Pd=cmpb.eq(Rs,Rt)
let isCompare = 1 in
def CMPbEQrr_ubub_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmpb.eq($src1, $src2)",
            [(set (i1 PredRegs:$dst),
                  (seteq (and (xor (i32 IntRegs:$src1),
                                   (i32 IntRegs:$src2)), 255), 0))]>,
            Requires<[HasV4T]>;

// Pd=cmpb.eq(Rs,Rt)
let isCompare = 1 in
def CMPbEQrr_sbsb_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmpb.eq($src1, $src2)",
            [(set (i1 PredRegs:$dst),
                  (seteq (shl (i32 IntRegs:$src1), (i32 24)),
                         (shl (i32 IntRegs:$src2), (i32 24))))]>,
            Requires<[HasV4T]>;

// Pd=cmpb.gt(Rs,Rt)
let isCompare = 1 in
def CMPbGTrr_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmpb.gt($src1, $src2)",
            [(set (i1 PredRegs:$dst),
                  (setgt (shl (i32 IntRegs:$src1), (i32 24)),
                         (shl (i32 IntRegs:$src2), (i32 24))))]>,
            Requires<[HasV4T]>;

// Pd=cmpb.gtu(Rs,#u7)
let isCompare = 1 in
def CMPbGTUri_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, u7Imm:$src2),
            "$dst = cmpb.gtu($src1, #$src2)",
            [(set (i1 PredRegs:$dst), (setugt (and (i32 IntRegs:$src1), 255),
                                              u7ImmPred:$src2))]>,
            Requires<[HasV4T]>;

// Pd=cmpb.gtu(Rs,Rt)
let isCompare = 1 in
def CMPbGTUrr_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmpb.gtu($src1, $src2)",
            [(set (i1 PredRegs:$dst), (setugt (and (i32 IntRegs:$src1), 255),
                                             (and (i32 IntRegs:$src2), 255)))]>,
            Requires<[HasV4T]>;

// Following instruction is not being extended as it results into the incorrect
// code for negative numbers.

// Signed half compare(.eq) ri.
// Pd=cmph.eq(Rs,#s8)
let isCompare = 1 in
def CMPhEQri_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, s8Imm:$src2),
            "$dst = cmph.eq($src1, #$src2)",
            [(set (i1 PredRegs:$dst), (seteq (and (i32 IntRegs:$src1), 65535),
                                             s8ImmPred:$src2))]>,
            Requires<[HasV4T]>;

// Signed half compare(.eq) rr.
// Case 1: xor + and, then compare:
//   r0=xor(r0,r1)
//   r0=and(r0,#0xffff)
//   p0=cmp.eq(r0,#0)
// Pd=cmph.eq(Rs,Rt)
let isCompare = 1 in
def CMPhEQrr_xor_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmph.eq($src1, $src2)",
            [(set (i1 PredRegs:$dst), (seteq (and (xor (i32 IntRegs:$src1),
                                                       (i32 IntRegs:$src2)),
                                                  65535), 0))]>,
            Requires<[HasV4T]>;

// Signed half compare(.eq) rr.
// Case 2: shift left 16 bits then compare:
//   r0=asl(r0,16)
//   r1=asl(r1,16)
//   p0=cmp.eq(r0,r1)
// Pd=cmph.eq(Rs,Rt)
let isCompare = 1 in
def CMPhEQrr_shl_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmph.eq($src1, $src2)",
            [(set (i1 PredRegs:$dst),
                  (seteq (shl (i32 IntRegs:$src1), (i32 16)),
                         (shl (i32 IntRegs:$src2), (i32 16))))]>,
            Requires<[HasV4T]>;

/* Incorrect Pattern -- immediate should be right shifted before being
used in the cmph.gt instruction.
// Signed half compare(.gt) ri.
// Pd=cmph.gt(Rs,#s8)

let isCompare = 1 in
def CMPhGTri_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, s8Imm:$src2),
            "$dst = cmph.gt($src1, #$src2)",
            [(set (i1 PredRegs:$dst),
                  (setgt (shl (i32 IntRegs:$src1), (i32 16)),
                         s8ImmPred:$src2))]>,
            Requires<[HasV4T]>;
*/

// Signed half compare(.gt) rr.
// Pd=cmph.gt(Rs,Rt)
let isCompare = 1 in
def CMPhGTrr_shl_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmph.gt($src1, $src2)",
            [(set (i1 PredRegs:$dst),
                  (setgt (shl (i32 IntRegs:$src1), (i32 16)),
                         (shl (i32 IntRegs:$src2), (i32 16))))]>,
            Requires<[HasV4T]>;

// Unsigned half compare rr (.gtu).
// Pd=cmph.gtu(Rs,Rt)
let isCompare = 1 in
def CMPhGTUrr_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmph.gtu($src1, $src2)",
            [(set (i1 PredRegs:$dst),
                  (setugt (and (i32 IntRegs:$src1), 65535),
                          (and (i32 IntRegs:$src2), 65535)))]>,
            Requires<[HasV4T]>;

// Unsigned half compare ri (.gtu).
// Pd=cmph.gtu(Rs,#u7)
let isCompare = 1 in
def CMPhGTUri_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, u7Imm:$src2),
            "$dst = cmph.gtu($src1, #$src2)",
            [(set (i1 PredRegs:$dst), (setugt (and (i32 IntRegs:$src1), 65535),
                                              u7ImmPred:$src2))]>,
            Requires<[HasV4T]>;

//===----------------------------------------------------------------------===//
// XTYPE/PRED -
//===----------------------------------------------------------------------===//

//Deallocate frame and return.
//    dealloc_return
let isReturn = 1, isTerminator = 1, isBarrier = 1, isPredicable = 1,
  Defs = [R29, R30, R31, PC], Uses = [R29, R31], neverHasSideEffects = 1 in {
  def DEALLOC_RET_V4 : NVInst_V4<(outs), (ins i32imm:$amt1),
            "dealloc_return",
            []>,
            Requires<[HasV4T]>;
}

// Restore registers and dealloc return function call.
let isCall = 1, isBarrier = 1, isReturn = 1, isTerminator = 1,
  Defs = [R29, R30, R31, PC] in {
  def RESTORE_DEALLOC_RET_JMP_V4 : JInst<(outs),
                                   (ins calltarget:$dst),
             "jump $dst // Restore_and_dealloc_return",
             []>,
             Requires<[HasV4T]>;
}

// Restore registers and dealloc frame before a tail call.
let isCall = 1, isBarrier = 1,
  Defs = [R29, R30, R31, PC] in {
  def RESTORE_DEALLOC_BEFORE_TAILCALL_V4 : JInst<(outs),
                                           (ins calltarget:$dst),
             "call $dst // Restore_and_dealloc_before_tailcall",
             []>,
             Requires<[HasV4T]>;
}

// Save registers function call.
let isCall = 1, isBarrier = 1,
  Uses = [R29, R31] in {
  def SAVE_REGISTERS_CALL_V4 : JInst<(outs),
                               (ins calltarget:$dst),
             "call $dst // Save_calle_saved_registers",
             []>,
             Requires<[HasV4T]>;
}

//    if (Ps) dealloc_return
let isReturn = 1, isTerminator = 1,
    Defs = [R29, R30, R31, PC], Uses = [R29, R31], neverHasSideEffects = 1,
    isPredicated = 1 in {
  def DEALLOC_RET_cPt_V4 : NVInst_V4<(outs),
                           (ins PredRegs:$src1, i32imm:$amt1),
            "if ($src1) dealloc_return",
            []>,
            Requires<[HasV4T]>;
}

//    if (!Ps) dealloc_return
let isReturn = 1, isTerminator = 1,
    Defs = [R29, R30, R31, PC], Uses = [R29, R31], neverHasSideEffects = 1,
    isPredicated = 1 in {
  def DEALLOC_RET_cNotPt_V4 : NVInst_V4<(outs), (ins PredRegs:$src1,
                                                     i32imm:$amt1),
            "if (!$src1) dealloc_return",
            []>,
            Requires<[HasV4T]>;
}

//    if (Ps.new) dealloc_return:nt
let isReturn = 1, isTerminator = 1,
    Defs = [R29, R30, R31, PC], Uses = [R29, R31], neverHasSideEffects = 1,
    isPredicated = 1 in {
  def DEALLOC_RET_cdnPnt_V4 : NVInst_V4<(outs), (ins PredRegs:$src1,
                                                     i32imm:$amt1),
            "if ($src1.new) dealloc_return:nt",
            []>,
            Requires<[HasV4T]>;
}

//    if (!Ps.new) dealloc_return:nt
let isReturn = 1, isTerminator = 1,
    Defs = [R29, R30, R31, PC], Uses = [R29, R31], neverHasSideEffects = 1,
    isPredicated = 1 in {
  def DEALLOC_RET_cNotdnPnt_V4 : NVInst_V4<(outs), (ins PredRegs:$src1,
                                                        i32imm:$amt1),
            "if (!$src1.new) dealloc_return:nt",
            []>,
            Requires<[HasV4T]>;
}

//    if (Ps.new) dealloc_return:t
let isReturn = 1, isTerminator = 1,
    Defs = [R29, R30, R31, PC], Uses = [R29, R31], neverHasSideEffects = 1,
    isPredicated = 1 in {
  def DEALLOC_RET_cdnPt_V4 : NVInst_V4<(outs), (ins PredRegs:$src1,
                                                    i32imm:$amt1),
            "if ($src1.new) dealloc_return:t",
            []>,
            Requires<[HasV4T]>;
}

//    if (!Ps.new) dealloc_return:nt
let isReturn = 1, isTerminator = 1,
    Defs = [R29, R30, R31, PC], Uses = [R29, R31], neverHasSideEffects = 1,
    isPredicated = 1 in {
  def DEALLOC_RET_cNotdnPt_V4 : NVInst_V4<(outs), (ins PredRegs:$src1,
                                                       i32imm:$amt1),
            "if (!$src1.new) dealloc_return:t",
            []>,
            Requires<[HasV4T]>;
}


// Load/Store with absolute addressing mode
// memw(#u6)=Rt

multiclass ST_abs<string OpcStr> {
  let isPredicable = 1 in
  def _abs_V4 : STInst2<(outs),
            (ins globaladdress:$absaddr, IntRegs:$src),
            !strconcat(OpcStr, "(##$absaddr) = $src"),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$absaddr, IntRegs:$src2),
            !strconcat("if ($src1)",
            !strconcat(OpcStr, "(##$absaddr) = $src2")),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$absaddr, IntRegs:$src2),
            !strconcat("if (!$src1)",
            !strconcat(OpcStr, "(##$absaddr) = $src2")),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$absaddr, IntRegs:$src2),
            !strconcat("if ($src1.new)",
            !strconcat(OpcStr, "(##$absaddr) = $src2")),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$absaddr, IntRegs:$src2),
            !strconcat("if (!$src1.new)",
            !strconcat(OpcStr, "(##$absaddr) = $src2")),
            []>,
            Requires<[HasV4T]>;

  def _abs_nv_V4 : STInst2<(outs),
            (ins globaladdress:$absaddr, IntRegs:$src),
            !strconcat(OpcStr, "(##$absaddr) = $src.new"),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cPt_nv_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$absaddr, IntRegs:$src2),
            !strconcat("if ($src1)",
            !strconcat(OpcStr, "(##$absaddr) = $src2.new")),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cNotPt_nv_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$absaddr, IntRegs:$src2),
            !strconcat("if (!$src1)",
            !strconcat(OpcStr, "(##$absaddr) = $src2.new")),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cdnPt_nv_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$absaddr, IntRegs:$src2),
            !strconcat("if ($src1.new)",
            !strconcat(OpcStr, "(##$absaddr) = $src2.new")),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cdnNotPt_nv_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$absaddr, IntRegs:$src2),
            !strconcat("if (!$src1.new)",
            !strconcat(OpcStr, "(##$absaddr) = $src2.new")),
            []>,
            Requires<[HasV4T]>;
}

let AddedComplexity = 30, isPredicable = 1 in
def STrid_abs_V4 : STInst<(outs),
          (ins globaladdress:$absaddr, DoubleRegs:$src),
           "memd(##$absaddr) = $src",
          [(store (i64 DoubleRegs:$src),
                  (HexagonCONST32 tglobaladdr:$absaddr))]>,
          Requires<[HasV4T]>;

let AddedComplexity = 30, isPredicated = 1 in
def STrid_abs_cPt_V4 : STInst2<(outs),
          (ins PredRegs:$src1, globaladdress:$absaddr, DoubleRegs:$src2),
          "if ($src1) memd(##$absaddr) = $src2",
          []>,
          Requires<[HasV4T]>;

let AddedComplexity = 30, isPredicated = 1 in
def STrid_abs_cNotPt_V4 : STInst2<(outs),
          (ins PredRegs:$src1, globaladdress:$absaddr, DoubleRegs:$src2),
          "if (!$src1) memd(##$absaddr) = $src2",
          []>,
          Requires<[HasV4T]>;

let AddedComplexity = 30, isPredicated = 1 in
def STrid_abs_cdnPt_V4 : STInst2<(outs),
          (ins PredRegs:$src1, globaladdress:$absaddr, DoubleRegs:$src2),
          "if ($src1.new) memd(##$absaddr) = $src2",
          []>,
          Requires<[HasV4T]>;

let AddedComplexity = 30, isPredicated = 1 in
def STrid_abs_cdnNotPt_V4 : STInst2<(outs),
          (ins PredRegs:$src1, globaladdress:$absaddr, DoubleRegs:$src2),
          "if (!$src1.new) memd(##$absaddr) = $src2",
          []>,
          Requires<[HasV4T]>;

defm STrib : ST_abs<"memb">;
defm STrih : ST_abs<"memh">;
defm STriw : ST_abs<"memw">;

let Predicates = [HasV4T], AddedComplexity  = 30 in
def : Pat<(truncstorei8 (i32 IntRegs:$src1),
                        (HexagonCONST32 tglobaladdr:$absaddr)),
          (STrib_abs_V4 tglobaladdr: $absaddr, IntRegs: $src1)>;

let Predicates = [HasV4T], AddedComplexity  = 30 in
def : Pat<(truncstorei16 (i32 IntRegs:$src1),
                          (HexagonCONST32 tglobaladdr:$absaddr)),
          (STrih_abs_V4 tglobaladdr: $absaddr, IntRegs: $src1)>;

let Predicates = [HasV4T], AddedComplexity  = 30 in
def : Pat<(store (i32 IntRegs:$src1), (HexagonCONST32 tglobaladdr:$absaddr)),
          (STriw_abs_V4 tglobaladdr: $absaddr, IntRegs: $src1)>;


multiclass LD_abs<string OpcStr> {
  let isPredicable = 1 in
  def _abs_V4 : LDInst2<(outs IntRegs:$dst),
            (ins globaladdress:$absaddr),
            !strconcat("$dst = ", !strconcat(OpcStr, "(##$absaddr)")),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$absaddr),
            !strconcat("if ($src1) $dst = ",
            !strconcat(OpcStr, "(##$absaddr)")),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$absaddr),
            !strconcat("if (!$src1) $dst = ",
            !strconcat(OpcStr, "(##$absaddr)")),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$absaddr),
            !strconcat("if ($src1.new) $dst = ",
            !strconcat(OpcStr, "(##$absaddr)")),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$absaddr),
            !strconcat("if (!$src1.new) $dst = ",
            !strconcat(OpcStr, "(##$absaddr)")),
            []>,
            Requires<[HasV4T]>;
}

let AddedComplexity = 30 in
def LDrid_abs_V4 : LDInst<(outs DoubleRegs:$dst),
          (ins globaladdress:$absaddr),
          "$dst = memd(##$absaddr)",
          [(set (i64 DoubleRegs:$dst),
                (load (HexagonCONST32 tglobaladdr:$absaddr)))]>,
          Requires<[HasV4T]>;

let AddedComplexity = 30, isPredicated = 1 in
def LDrid_abs_cPt_V4 : LDInst2<(outs DoubleRegs:$dst),
          (ins PredRegs:$src1, globaladdress:$absaddr),
          "if ($src1) $dst = memd(##$absaddr)",
          []>,
          Requires<[HasV4T]>;

let AddedComplexity = 30, isPredicated = 1 in
def LDrid_abs_cNotPt_V4 : LDInst2<(outs DoubleRegs:$dst),
          (ins PredRegs:$src1, globaladdress:$absaddr),
          "if (!$src1) $dst = memd(##$absaddr)",
          []>,
          Requires<[HasV4T]>;

let AddedComplexity = 30, isPredicated = 1 in
def LDrid_abs_cdnPt_V4 : LDInst2<(outs DoubleRegs:$dst),
          (ins PredRegs:$src1, globaladdress:$absaddr),
          "if ($src1.new) $dst = memd(##$absaddr)",
          []>,
          Requires<[HasV4T]>;

let AddedComplexity = 30, isPredicated = 1 in
def LDrid_abs_cdnNotPt_V4 : LDInst2<(outs DoubleRegs:$dst),
          (ins PredRegs:$src1, globaladdress:$absaddr),
          "if (!$src1.new) $dst = memd(##$absaddr)",
          []>,
          Requires<[HasV4T]>;

defm LDrib : LD_abs<"memb">;
defm LDriub : LD_abs<"memub">;
defm LDrih : LD_abs<"memh">;
defm LDriuh : LD_abs<"memuh">;
defm LDriw : LD_abs<"memw">;


let Predicates = [HasV4T], AddedComplexity  = 30 in
def : Pat<(i32 (load (HexagonCONST32 tglobaladdr:$absaddr))),
          (LDriw_abs_V4 tglobaladdr: $absaddr)>;

let Predicates = [HasV4T], AddedComplexity=30 in
def : Pat<(i32 (sextloadi8 (HexagonCONST32 tglobaladdr:$absaddr))),
          (LDrib_abs_V4 tglobaladdr:$absaddr)>;

let Predicates = [HasV4T], AddedComplexity=30 in
def : Pat<(i32 (zextloadi8 (HexagonCONST32 tglobaladdr:$absaddr))),
          (LDriub_abs_V4 tglobaladdr:$absaddr)>;

let Predicates = [HasV4T], AddedComplexity=30 in
def : Pat<(i32 (sextloadi16 (HexagonCONST32 tglobaladdr:$absaddr))),
          (LDrih_abs_V4 tglobaladdr:$absaddr)>;

let Predicates = [HasV4T], AddedComplexity=30 in
def : Pat<(i32 (zextloadi16 (HexagonCONST32 tglobaladdr:$absaddr))),
          (LDriuh_abs_V4 tglobaladdr:$absaddr)>;

// Transfer global address into a register
let AddedComplexity=50, isMoveImm = 1, isReMaterializable = 1 in
def TFRI_V4 : ALU32_ri<(outs IntRegs:$dst), (ins globaladdress:$src1),
           "$dst = ##$src1",
           [(set IntRegs:$dst, (HexagonCONST32 tglobaladdr:$src1))]>,
           Requires<[HasV4T]>;

let AddedComplexity=50, neverHasSideEffects = 1, isPredicated = 1 in
def TFRI_cPt_V4 : ALU32_ri<(outs IntRegs:$dst),
                           (ins PredRegs:$src1, globaladdress:$src2),
           "if($src1) $dst = ##$src2",
           []>,
           Requires<[HasV4T]>;

let AddedComplexity=50, neverHasSideEffects = 1, isPredicated = 1 in
def TFRI_cNotPt_V4 : ALU32_ri<(outs IntRegs:$dst),
                              (ins PredRegs:$src1, globaladdress:$src2),
           "if(!$src1) $dst = ##$src2",
           []>,
           Requires<[HasV4T]>;

let AddedComplexity=50, neverHasSideEffects = 1, isPredicated = 1 in
def TFRI_cdnPt_V4 : ALU32_ri<(outs IntRegs:$dst),
                             (ins PredRegs:$src1, globaladdress:$src2),
           "if($src1.new) $dst = ##$src2",
           []>,
           Requires<[HasV4T]>;

let AddedComplexity=50, neverHasSideEffects = 1, isPredicated = 1 in
def TFRI_cdnNotPt_V4 : ALU32_ri<(outs IntRegs:$dst),
                                (ins PredRegs:$src1, globaladdress:$src2),
           "if(!$src1.new) $dst = ##$src2",
           []>,
           Requires<[HasV4T]>;

let AddedComplexity = 50, Predicates = [HasV4T] in
def : Pat<(HexagonCONST32_GP tglobaladdr:$src1),
           (TFRI_V4 tglobaladdr:$src1)>;


// Load - Indirect with long offset: These instructions take global address
// as an operand
let AddedComplexity = 10 in
def LDrid_ind_lo_V4 : LDInst<(outs DoubleRegs:$dst),
            (ins IntRegs:$src1, u2Imm:$src2, globaladdress:$offset),
            "$dst=memd($src1<<#$src2+##$offset)",
            [(set (i64 DoubleRegs:$dst),
                  (load (add (shl IntRegs:$src1, u2ImmPred:$src2),
                        (HexagonCONST32 tglobaladdr:$offset))))]>,
            Requires<[HasV4T]>;

let AddedComplexity = 10 in
multiclass LD_indirect_lo<string OpcStr, PatFrag OpNode> {
  def _lo_V4 : LDInst<(outs IntRegs:$dst),
            (ins IntRegs:$src1, u2Imm:$src2, globaladdress:$offset),
            !strconcat("$dst = ",
            !strconcat(OpcStr, "($src1<<#$src2+##$offset)")),
            [(set IntRegs:$dst,
                  (i32 (OpNode (add (shl IntRegs:$src1, u2ImmPred:$src2),
                          (HexagonCONST32 tglobaladdr:$offset)))))]>,
            Requires<[HasV4T]>;
}

defm LDrib_ind : LD_indirect_lo<"memb", sextloadi8>;
defm LDriub_ind : LD_indirect_lo<"memub", zextloadi8>;
defm LDrih_ind : LD_indirect_lo<"memh", sextloadi16>;
defm LDriuh_ind : LD_indirect_lo<"memuh", zextloadi16>;
defm LDriw_ind : LD_indirect_lo<"memw", load>;

// Store - Indirect with long offset: These instructions take global address
// as an operand
let AddedComplexity = 10 in
def STrid_ind_lo_V4 : STInst<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, globaladdress:$src3,
                 DoubleRegs:$src4),
            "memd($src1<<#$src2+#$src3) = $src4",
            [(store (i64 DoubleRegs:$src4),
                 (add (shl IntRegs:$src1, u2ImmPred:$src2),
                      (HexagonCONST32 tglobaladdr:$src3)))]>,
             Requires<[HasV4T]>;

let AddedComplexity = 10 in
multiclass ST_indirect_lo<string OpcStr, PatFrag OpNode> {
  def _lo_V4 : STInst<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, globaladdress:$src3,
                 IntRegs:$src4),
            !strconcat(OpcStr, "($src1<<#$src2+##$src3) = $src4"),
            [(OpNode (i32 IntRegs:$src4),
                 (add (shl IntRegs:$src1, u2ImmPred:$src2),
                      (HexagonCONST32 tglobaladdr:$src3)))]>,
             Requires<[HasV4T]>;
}

defm STrib_ind : ST_indirect_lo<"memb", truncstorei8>;
defm STrih_ind : ST_indirect_lo<"memh", truncstorei16>;
defm STriw_ind : ST_indirect_lo<"memw", store>;

// Store - absolute addressing mode: These instruction take constant
// value as the extended operand
multiclass ST_absimm<string OpcStr> {
  let isPredicable = 1 in
  def _abs_V4 : STInst2<(outs),
            (ins u6Imm:$src1, IntRegs:$src2),
            !strconcat(OpcStr, "(#$src1) = $src2"),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, u6Imm:$src2, IntRegs:$src3),
            !strconcat("if ($src1)", !strconcat(OpcStr, "(#$src2) = $src3")),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, u6Imm:$src2, IntRegs:$src3),
            !strconcat("if (!$src1)", !strconcat(OpcStr, "(#$src2) = $src3")),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, u6Imm:$src2, IntRegs:$src3),
            !strconcat("if ($src1.new)",
            !strconcat(OpcStr, "(#$src2) = $src3")),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, u6Imm:$src2, IntRegs:$src3),
            !strconcat("if (!$src1.new)",
            !strconcat(OpcStr, "(#$src2) = $src3")),
            []>,
            Requires<[HasV4T]>;

  def _abs_nv_V4 : STInst2<(outs),
            (ins u6Imm:$src1, IntRegs:$src2),
            !strconcat(OpcStr, "(#$src1) = $src2.new"),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cPt_nv_V4 : STInst2<(outs),
            (ins PredRegs:$src1, u6Imm:$src2, IntRegs:$src3),
            !strconcat("if ($src1)",
            !strconcat(OpcStr, "(#$src2) = $src3.new")),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cNotPt_nv_V4 : STInst2<(outs),
            (ins PredRegs:$src1, u6Imm:$src2, IntRegs:$src3),
            !strconcat("if (!$src1)",
            !strconcat(OpcStr, "(#$src2) = $src3.new")),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cdnPt_nv_V4 : STInst2<(outs),
            (ins PredRegs:$src1, u6Imm:$src2, IntRegs:$src3),
            !strconcat("if ($src1.new)",
            !strconcat(OpcStr, "(#$src2) = $src3.new")),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cdnNotPt_nv_V4 : STInst2<(outs),
            (ins PredRegs:$src1, u6Imm:$src2, IntRegs:$src3),
            !strconcat("if (!$src1.new)",
            !strconcat(OpcStr, "(#$src2) = $src3.new")),
            []>,
            Requires<[HasV4T]>;
}

defm STrib_imm : ST_absimm<"memb">;
defm STrih_imm : ST_absimm<"memh">;
defm STriw_imm : ST_absimm<"memw">;

let Predicates = [HasV4T], AddedComplexity  = 30 in
def : Pat<(truncstorei8 (i32 IntRegs:$src1), u6ImmPred:$src2),
          (STrib_imm_abs_V4 u6ImmPred:$src2, IntRegs: $src1)>;

let Predicates = [HasV4T], AddedComplexity  = 30 in
def : Pat<(truncstorei16 (i32 IntRegs:$src1), u6ImmPred:$src2),
          (STrih_imm_abs_V4 u6ImmPred:$src2, IntRegs: $src1)>;

let Predicates = [HasV4T], AddedComplexity  = 30 in
def : Pat<(store (i32 IntRegs:$src1), u6ImmPred:$src2),
          (STriw_imm_abs_V4 u6ImmPred:$src2, IntRegs: $src1)>;


// Load - absolute addressing mode: These instruction take constant
// value as the extended operand

multiclass LD_absimm<string OpcStr> {
  let isPredicable = 1 in
  def _abs_V4 : LDInst2<(outs IntRegs:$dst),
            (ins u6Imm:$src),
            !strconcat("$dst = ",
            !strconcat(OpcStr, "(#$src)")),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, u6Imm:$src2),
            !strconcat("if ($src1) $dst = ",
            !strconcat(OpcStr, "(#$src2)")),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, u6Imm:$src2),
            !strconcat("if (!$src1) $dst = ",
            !strconcat(OpcStr, "(#$src2)")),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, u6Imm:$src2),
            !strconcat("if ($src1.new) $dst = ",
            !strconcat(OpcStr, "(#$src2)")),
            []>,
            Requires<[HasV4T]>;

  let isPredicated = 1 in
  def _abs_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, u6Imm:$src2),
            !strconcat("if (!$src1.new) $dst = ",
            !strconcat(OpcStr, "(#$src2)")),
            []>,
            Requires<[HasV4T]>;
}

defm LDrib_imm : LD_absimm<"memb">;
defm LDriub_imm : LD_absimm<"memub">;
defm LDrih_imm : LD_absimm<"memh">;
defm LDriuh_imm : LD_absimm<"memuh">;
defm LDriw_imm : LD_absimm<"memw">;

let Predicates = [HasV4T], AddedComplexity  = 30 in
def : Pat<(i32 (load u6ImmPred:$src)),
          (LDriw_imm_abs_V4 u6ImmPred:$src)>;

let Predicates = [HasV4T], AddedComplexity=30 in
def : Pat<(i32 (sextloadi8 u6ImmPred:$src)),
          (LDrib_imm_abs_V4 u6ImmPred:$src)>;

let Predicates = [HasV4T], AddedComplexity=30 in
def : Pat<(i32 (zextloadi8 u6ImmPred:$src)),
          (LDriub_imm_abs_V4 u6ImmPred:$src)>;

let Predicates = [HasV4T], AddedComplexity=30 in
def : Pat<(i32 (sextloadi16 u6ImmPred:$src)),
          (LDrih_imm_abs_V4 u6ImmPred:$src)>;

let Predicates = [HasV4T], AddedComplexity=30 in
def : Pat<(i32 (zextloadi16 u6ImmPred:$src)),
          (LDriuh_imm_abs_V4 u6ImmPred:$src)>;


// Indexed store double word - global address.
// memw(Rs+#u6:2)=#S8
let AddedComplexity = 10 in
def STriw_offset_ext_V4 : STInst<(outs),
            (ins IntRegs:$src1, u6_2Imm:$src2, globaladdress:$src3),
            "memw($src1+#$src2) = ##$src3",
            [(store (HexagonCONST32 tglobaladdr:$src3),
                    (add IntRegs:$src1, u6_2ImmPred:$src2))]>,
            Requires<[HasV4T]>;


// Indexed store double word - global address.
// memw(Rs+#u6:2)=#S8
let AddedComplexity = 10 in
def STrih_offset_ext_V4 : STInst<(outs),
            (ins IntRegs:$src1, u6_1Imm:$src2, globaladdress:$src3),
            "memh($src1+#$src2) = ##$src3",
            [(truncstorei16 (HexagonCONST32 tglobaladdr:$src3),
                    (add IntRegs:$src1, u6_1ImmPred:$src2))]>,
            Requires<[HasV4T]>;
