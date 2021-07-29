//=- AArch64SMEInstrInfo.td -  AArch64 SME Instructions -*- tablegen -*-----=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AArch64 Scalable Matrix Extension (SME) Instruction definitions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Add vector elements horizontally or vertically to ZA tile.
//===----------------------------------------------------------------------===//

let Predicates = [HasSME] in {
def ADDHA_MPPZ_S : sme_add_vector_to_tile_u32<0b0, "addha">;
def ADDVA_MPPZ_S : sme_add_vector_to_tile_u32<0b1, "addva">;
}

let Predicates = [HasSMEI64] in {
def ADDHA_MPPZ_D : sme_add_vector_to_tile_u64<0b0, "addha">;
def ADDVA_MPPZ_D : sme_add_vector_to_tile_u64<0b1, "addva">;
}

let Predicates = [HasSME] in {
//===----------------------------------------------------------------------===//
// Outer products
//===----------------------------------------------------------------------===//

defm BFMOPA_MPPZZ  : sme_bf16_outer_product<0b0, "bfmopa">;
defm BFMOPS_MPPZZ  : sme_bf16_outer_product<0b1, "bfmops">;

def FMOPA_MPPZZ_S : sme_outer_product_fp32<0b0, "fmopa">;
def FMOPS_MPPZZ_S : sme_outer_product_fp32<0b1, "fmops">;
}

let Predicates = [HasSMEF64] in {
def FMOPA_MPPZZ_D : sme_outer_product_fp64<0b0, "fmopa">;
def FMOPS_MPPZZ_D : sme_outer_product_fp64<0b1, "fmops">;
}

let Predicates = [HasSME] in {
defm FMOPAL_MPPZZ  : sme_f16_outer_product<0b0, "fmopa">;
defm FMOPSL_MPPZZ  : sme_f16_outer_product<0b1, "fmops">;

def SMOPA_MPPZZ_S  : sme_int_outer_product_i32<0b000, "smopa">;
def SMOPS_MPPZZ_S  : sme_int_outer_product_i32<0b001, "smops">;
def UMOPA_MPPZZ_S  : sme_int_outer_product_i32<0b110, "umopa">;
def UMOPS_MPPZZ_S  : sme_int_outer_product_i32<0b111, "umops">;
def SUMOPA_MPPZZ_S : sme_int_outer_product_i32<0b010, "sumopa">;
def SUMOPS_MPPZZ_S : sme_int_outer_product_i32<0b011, "sumops">;
def USMOPA_MPPZZ_S : sme_int_outer_product_i32<0b100, "usmopa">;
def USMOPS_MPPZZ_S : sme_int_outer_product_i32<0b101, "usmops">;
}

let Predicates = [HasSMEI64] in {
def SMOPA_MPPZZ_D  : sme_int_outer_product_i64<0b000, "smopa">;
def SMOPS_MPPZZ_D  : sme_int_outer_product_i64<0b001, "smops">;
def UMOPA_MPPZZ_D  : sme_int_outer_product_i64<0b110, "umopa">;
def UMOPS_MPPZZ_D  : sme_int_outer_product_i64<0b111, "umops">;
def SUMOPA_MPPZZ_D : sme_int_outer_product_i64<0b010, "sumopa">;
def SUMOPS_MPPZZ_D : sme_int_outer_product_i64<0b011, "sumops">;
def USMOPA_MPPZZ_D : sme_int_outer_product_i64<0b100, "usmopa">;
def USMOPS_MPPZZ_D : sme_int_outer_product_i64<0b101, "usmops">;
}

let Predicates = [HasSME] in {
//===----------------------------------------------------------------------===//
// Loads and stores
//===----------------------------------------------------------------------===//

defm LD1_MXIPXX : sme_mem_ld_ss<"ld1">;
defm ST1_MXIPXX : sme_mem_st_ss<"st1">;

//===----------------------------------------------------------------------===//
// Spill + fill
//===----------------------------------------------------------------------===//

defm LDR_ZA : sme_fill<"ldr">;
defm STR_ZA : sme_spill<"str">;

//===----------------------------------------------------------------------===//
// Move instructions
//===----------------------------------------------------------------------===//

defm INSERT_MXIPZ  : sme_vector_to_tile<"mova">;
defm EXTRACT_ZPMXI : sme_tile_to_vector<"mova">;

//===----------------------------------------------------------------------===//
// Zero instruction
//===----------------------------------------------------------------------===//

defm ZERO_M : sme_zero<"zero">;

//===----------------------------------------------------------------------===//
// Mode selection and state access instructions
//===----------------------------------------------------------------------===//

// SME defines three pstate fields to set or clear PSTATE.SM, PSTATE.ZA, or
// both fields:
//
//   MSR SVCRSM, #<imm1>
//   MSR SVCRZA, #<imm1>
//   MSR SVCRSMZA, #<imm1>
//
// It's tricky to using the existing pstate operand defined in
// AArch64SystemOperands.td since it only encodes 5 bits including op1;op2,
// when these fields are also encoded in CRm[3:1].
class MSRpstatesvcrImm0_1
  : PstateWriteSimple<(ins svcr_op:$pstatefield, imm0_1:$imm), "msr",
                      "\t$pstatefield, $imm">,
    Sched<[WriteSys]> {
  bits<3> pstatefield;
  bit imm;
  let Inst{18-16} = 0b011; // op1
  let Inst{11-9} = pstatefield;
  let Inst{8} = imm;
  let Inst{7-5} = 0b011; // op2
}

def MSRpstatesvcrImm1 : MSRpstatesvcrImm0_1;
def : InstAlias<"smstart",    (MSRpstatesvcrImm1 0b011, 0b1)>;
def : InstAlias<"smstart sm", (MSRpstatesvcrImm1 0b001, 0b1)>;
def : InstAlias<"smstart za", (MSRpstatesvcrImm1 0b010, 0b1)>;

def : InstAlias<"smstop",     (MSRpstatesvcrImm1 0b011, 0b0)>;
def : InstAlias<"smstop sm",  (MSRpstatesvcrImm1 0b001, 0b0)>;
def : InstAlias<"smstop za",  (MSRpstatesvcrImm1 0b010, 0b0)>;

//===----------------------------------------------------------------------===//
// SVE2 instructions
//===----------------------------------------------------------------------===//

def REVD_ZPmZ : sve2_int_perm_revd<"revd">;

defm SCLAMP_ZZZ : sve2_clamp<"sclamp", 0b0>;
defm UCLAMP_ZZZ : sve2_clamp<"uclamp", 0b1>;

defm DUP_PPzPRI : sve2_int_perm_dup_p<"dup">;

} // End let Predicates = [HasSME]
