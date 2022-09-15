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

def AArch64_smstart : SDNode<"AArch64ISD::SMSTART", SDTypeProfile<0, 3,
                             [SDTCisInt<0>, SDTCisInt<0>, SDTCisInt<0>]>,
                             [SDNPHasChain, SDNPSideEffect, SDNPVariadic,
                              SDNPOptInGlue, SDNPOutGlue]>;
def AArch64_smstop  : SDNode<"AArch64ISD::SMSTOP", SDTypeProfile<0, 3,
                             [SDTCisInt<0>, SDTCisInt<0>, SDTCisInt<0>]>,
                             [SDNPHasChain, SDNPSideEffect, SDNPVariadic,
                              SDNPOptInGlue, SDNPOutGlue]>;

//===----------------------------------------------------------------------===//
// Add vector elements horizontally or vertically to ZA tile.
//===----------------------------------------------------------------------===//

def SDT_AArch64RDSVL  : SDTypeProfile<1, 1, [SDTCisInt<0>, SDTCisInt<1>]>;
def AArch64rdsvl : SDNode<"AArch64ISD::RDSVL", SDT_AArch64RDSVL>;

let Predicates = [HasSME] in {
def RDSVLI_XI  : sve_int_read_vl_a<0b0, 0b11111, "rdsvl", /*streaming_sve=*/0b1>;
def ADDSPL_XXI : sve_int_arith_vl<0b1, "addspl", /*streaming_sve=*/0b1>;
def ADDSVL_XXI : sve_int_arith_vl<0b0, "addsvl", /*streaming_sve=*/0b1>;

def ADDHA_MPPZ_S : sme_add_vector_to_tile_u32<0b0, "addha">;
def ADDVA_MPPZ_S : sme_add_vector_to_tile_u32<0b1, "addva">;

def : Pat<(AArch64rdsvl (i32 simm6_32b:$imm)), (RDSVLI_XI simm6_32b:$imm)>;
}

let Predicates = [HasSMEI64] in {
def ADDHA_MPPZ_D : sme_add_vector_to_tile_u64<0b0, "addha">;
def ADDVA_MPPZ_D : sme_add_vector_to_tile_u64<0b1, "addva">;
}

let Predicates = [HasSME] in {
//===----------------------------------------------------------------------===//
// Outer products
//===----------------------------------------------------------------------===//

defm BFMOPA_MPPZZ  : sme_bf16_outer_product<0b0, "bfmopa", int_aarch64_sme_mopa_wide>;
defm BFMOPS_MPPZZ  : sme_bf16_outer_product<0b1, "bfmops", int_aarch64_sme_mops_wide>;

defm FMOPA_MPPZZ_S : sme_outer_product_fp32<0b0, "fmopa", int_aarch64_sme_mopa>;
defm FMOPS_MPPZZ_S : sme_outer_product_fp32<0b1, "fmops", int_aarch64_sme_mops>;
}

let Predicates = [HasSMEF64] in {
defm FMOPA_MPPZZ_D : sme_outer_product_fp64<0b0, "fmopa", int_aarch64_sme_mopa>;
defm FMOPS_MPPZZ_D : sme_outer_product_fp64<0b1, "fmops", int_aarch64_sme_mops>;
}

let Predicates = [HasSME] in {
defm FMOPAL_MPPZZ  : sme_f16_outer_product<0b0, "fmopa", int_aarch64_sme_mopa_wide>;
defm FMOPSL_MPPZZ  : sme_f16_outer_product<0b1, "fmops", int_aarch64_sme_mops_wide>;

defm SMOPA_MPPZZ_S  : sme_int_outer_product_i32<0b000, "smopa",  int_aarch64_sme_smopa_wide>;
defm SMOPS_MPPZZ_S  : sme_int_outer_product_i32<0b001, "smops",  int_aarch64_sme_smops_wide>;
defm UMOPA_MPPZZ_S  : sme_int_outer_product_i32<0b110, "umopa",  int_aarch64_sme_umopa_wide>;
defm UMOPS_MPPZZ_S  : sme_int_outer_product_i32<0b111, "umops",  int_aarch64_sme_umops_wide>;
defm SUMOPA_MPPZZ_S : sme_int_outer_product_i32<0b010, "sumopa", int_aarch64_sme_sumopa_wide>;
defm SUMOPS_MPPZZ_S : sme_int_outer_product_i32<0b011, "sumops", int_aarch64_sme_sumops_wide>;
defm USMOPA_MPPZZ_S : sme_int_outer_product_i32<0b100, "usmopa", int_aarch64_sme_usmopa_wide>;
defm USMOPS_MPPZZ_S : sme_int_outer_product_i32<0b101, "usmops", int_aarch64_sme_usmops_wide>;
}

let Predicates = [HasSMEI64] in {
defm SMOPA_MPPZZ_D  : sme_int_outer_product_i64<0b000, "smopa",  int_aarch64_sme_smopa_wide>;
defm SMOPS_MPPZZ_D  : sme_int_outer_product_i64<0b001, "smops",  int_aarch64_sme_smops_wide>;
defm UMOPA_MPPZZ_D  : sme_int_outer_product_i64<0b110, "umopa",  int_aarch64_sme_umopa_wide>;
defm UMOPS_MPPZZ_D  : sme_int_outer_product_i64<0b111, "umops",  int_aarch64_sme_umops_wide>;
defm SUMOPA_MPPZZ_D : sme_int_outer_product_i64<0b010, "sumopa", int_aarch64_sme_sumopa_wide>;
defm SUMOPS_MPPZZ_D : sme_int_outer_product_i64<0b011, "sumops", int_aarch64_sme_sumops_wide>;
defm USMOPA_MPPZZ_D : sme_int_outer_product_i64<0b100, "usmopa", int_aarch64_sme_usmopa_wide>;
defm USMOPS_MPPZZ_D : sme_int_outer_product_i64<0b101, "usmops", int_aarch64_sme_usmops_wide>;
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
def MSRpstatesvcrImm1
  : PstateWriteSimple<(ins svcr_op:$pstatefield, timm0_1:$imm), "msr",
                      "\t$pstatefield, $imm">,
    Sched<[WriteSys]> {
  bits<3> pstatefield;
  bit imm;
  let Inst{18-16} = 0b011; // op1
  let Inst{11-9} = pstatefield;
  let Inst{8} = imm;
  let Inst{7-5} = 0b011; // op2
}

def : InstAlias<"smstart",    (MSRpstatesvcrImm1 0b011, 0b1)>;
def : InstAlias<"smstart sm", (MSRpstatesvcrImm1 0b001, 0b1)>;
def : InstAlias<"smstart za", (MSRpstatesvcrImm1 0b010, 0b1)>;

def : InstAlias<"smstop",     (MSRpstatesvcrImm1 0b011, 0b0)>;
def : InstAlias<"smstop sm",  (MSRpstatesvcrImm1 0b001, 0b0)>;
def : InstAlias<"smstop za",  (MSRpstatesvcrImm1 0b010, 0b0)>;


// Scenario A:
//
//   %pstate.before.call = 1
//   if (%pstate.before.call != 0)
//     smstop (pstate_za|pstate_sm)
//   call fn()
//   if (%pstate.before.call != 0)
//     smstart (pstate_za|pstate_sm)
//
def : Pat<(AArch64_smstop (i32 svcr_op:$pstate), (i64 1), (i64 0)),   // before call
          (MSRpstatesvcrImm1 svcr_op:$pstate, 0b0)>;
def : Pat<(AArch64_smstart (i32 svcr_op:$pstate), (i64 1), (i64 0)),  // after call
          (MSRpstatesvcrImm1 svcr_op:$pstate, 0b1)>;

// Scenario B:
//
//   %pstate.before.call = 0
//   if (%pstate.before.call != 1)
//     smstart (pstate_za|pstate_sm)
//   call fn()
//   if (%pstate.before.call != 1)
//     smstop (pstate_za|pstate_sm)
//
def : Pat<(AArch64_smstart (i32 svcr_op:$pstate), (i64 0), (i64 1)),  // before call
          (MSRpstatesvcrImm1 svcr_op:$pstate, 0b1)>;
def : Pat<(AArch64_smstop (i32 svcr_op:$pstate), (i64 0), (i64 1)),   // after call
          (MSRpstatesvcrImm1 svcr_op:$pstate, 0b0)>;

// Read and write TPIDR2_EL0
def : Pat<(int_aarch64_sme_set_tpidr2 i64:$val),
          (MSR 0xde85, GPR64:$val)>;
def : Pat<(i64 (int_aarch64_sme_get_tpidr2)),
          (MRS 0xde85)>;

//===----------------------------------------------------------------------===//
// SVE2 instructions
//===----------------------------------------------------------------------===//

defm REVD_ZPmZ : sve2_int_perm_revd<"revd", AArch64revd_mt>;

defm SCLAMP_ZZZ : sve2_clamp<"sclamp", 0b0, int_aarch64_sve_sclamp>;
defm UCLAMP_ZZZ : sve2_clamp<"uclamp", 0b1, int_aarch64_sve_uclamp>;

defm PSEL_PPPRI : sve2_int_perm_sel_p<"psel", int_aarch64_sve_psel>;

} // End let Predicates = [HasSME]
