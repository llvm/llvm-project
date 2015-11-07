//===-- SIInstructions.td - SI Instruction Defintions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file was originally auto-generated from a GPU register header file and
// all the instruction definitions were originally commented out.  Instructions
// that are not yet supported remain commented out.
//===----------------------------------------------------------------------===//

class InterpSlots {
int P0 = 2;
int P10 = 0;
int P20 = 1;
}
def INTERP : InterpSlots;

def InterpSlot : Operand<i32> {
  let PrintMethod = "printInterpSlot";
}

def SendMsgImm : Operand<i32> {
  let PrintMethod = "printSendMsg";
}

def isGCN : Predicate<"Subtarget->getGeneration() "
                      ">= AMDGPUSubtarget::SOUTHERN_ISLANDS">,
            AssemblerPredicate<"FeatureGCN">;
def isSI : Predicate<"Subtarget->getGeneration() "
                      "== AMDGPUSubtarget::SOUTHERN_ISLANDS">,
           AssemblerPredicate<"FeatureSouthernIslands">;


def has16BankLDS : Predicate<"Subtarget->getLDSBankCount() == 16">;
def has32BankLDS : Predicate<"Subtarget->getLDSBankCount() == 32">;

def SWaitMatchClass : AsmOperandClass {
  let Name = "SWaitCnt";
  let RenderMethod = "addImmOperands";
  let ParserMethod = "parseSWaitCntOps";
}

def WAIT_FLAG : InstFlag<"printWaitFlag"> {
  let ParserMatchClass = SWaitMatchClass;
}

let SubtargetPredicate = isGCN in {

//===----------------------------------------------------------------------===//
// EXP Instructions
//===----------------------------------------------------------------------===//

defm EXP : EXP_m;

//===----------------------------------------------------------------------===//
// SMRD Instructions
//===----------------------------------------------------------------------===//

let mayLoad = 1 in {

// We are using the SGPR_32 and not the SReg_32 register class for 32-bit
// SMRD instructions, because the SGPR_32 register class does not include M0
// and writing to M0 from an SMRD instruction will hang the GPU.
defm S_LOAD_DWORD : SMRD_Helper <smrd<0x00>, "s_load_dword", SReg_64, SGPR_32>;
defm S_LOAD_DWORDX2 : SMRD_Helper <smrd<0x01>, "s_load_dwordx2", SReg_64, SReg_64>;
defm S_LOAD_DWORDX4 : SMRD_Helper <smrd<0x02>, "s_load_dwordx4", SReg_64, SReg_128>;
defm S_LOAD_DWORDX8 : SMRD_Helper <smrd<0x03>, "s_load_dwordx8", SReg_64, SReg_256>;
defm S_LOAD_DWORDX16 : SMRD_Helper <smrd<0x04>, "s_load_dwordx16", SReg_64, SReg_512>;

defm S_BUFFER_LOAD_DWORD : SMRD_Helper <
  smrd<0x08>, "s_buffer_load_dword", SReg_128, SGPR_32
>;

defm S_BUFFER_LOAD_DWORDX2 : SMRD_Helper <
  smrd<0x09>, "s_buffer_load_dwordx2", SReg_128, SReg_64
>;

defm S_BUFFER_LOAD_DWORDX4 : SMRD_Helper <
  smrd<0x0a>, "s_buffer_load_dwordx4", SReg_128, SReg_128
>;

defm S_BUFFER_LOAD_DWORDX8 : SMRD_Helper <
  smrd<0x0b>, "s_buffer_load_dwordx8", SReg_128, SReg_256
>;

defm S_BUFFER_LOAD_DWORDX16 : SMRD_Helper <
  smrd<0x0c>, "s_buffer_load_dwordx16", SReg_128, SReg_512
>;

} // mayLoad = 1

//def S_MEMTIME : SMRD_ <0x0000001e, "s_memtime", []>;

defm S_DCACHE_INV : SMRD_Inval <smrd<0x1f, 0x20>, "s_dcache_inv",
  int_amdgcn_s_dcache_inv>;

//===----------------------------------------------------------------------===//
// SOP1 Instructions
//===----------------------------------------------------------------------===//

let isMoveImm = 1 in {
  let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
    defm S_MOV_B32 : SOP1_32 <sop1<0x03, 0x00>, "s_mov_b32", []>;
    defm S_MOV_B64 : SOP1_64 <sop1<0x04, 0x01>, "s_mov_b64", []>;
  } // let isRematerializeable = 1

  let Uses = [SCC] in {
    defm S_CMOV_B32 : SOP1_32 <sop1<0x05, 0x02>, "s_cmov_b32", []>;
    defm S_CMOV_B64 : SOP1_64 <sop1<0x06, 0x03>, "s_cmov_b64", []>;
  } // End Uses = [SCC]
} // End isMoveImm = 1

let Defs = [SCC] in {
  defm S_NOT_B32 : SOP1_32 <sop1<0x07, 0x04>, "s_not_b32",
    [(set i32:$dst, (not i32:$src0))]
  >;

  defm S_NOT_B64 : SOP1_64 <sop1<0x08, 0x05>, "s_not_b64",
    [(set i64:$dst, (not i64:$src0))]
  >;
  defm S_WQM_B32 : SOP1_32 <sop1<0x09, 0x06>, "s_wqm_b32", []>;
  defm S_WQM_B64 : SOP1_64 <sop1<0x0a, 0x07>, "s_wqm_b64", []>;
} // End Defs = [SCC]


defm S_BREV_B32 : SOP1_32 <sop1<0x0b, 0x08>, "s_brev_b32",
  [(set i32:$dst, (AMDGPUbrev i32:$src0))]
>;
defm S_BREV_B64 : SOP1_64 <sop1<0x0c, 0x09>, "s_brev_b64", []>;

let Defs = [SCC] in {
  defm S_BCNT0_I32_B32 : SOP1_32 <sop1<0x0d, 0x0a>, "s_bcnt0_i32_b32", []>;
  defm S_BCNT0_I32_B64 : SOP1_32_64 <sop1<0x0e, 0x0b>, "s_bcnt0_i32_b64", []>;
  defm S_BCNT1_I32_B32 : SOP1_32 <sop1<0x0f, 0x0c>, "s_bcnt1_i32_b32",
    [(set i32:$dst, (ctpop i32:$src0))]
  >;
  defm S_BCNT1_I32_B64 : SOP1_32_64 <sop1<0x10, 0x0d>, "s_bcnt1_i32_b64", []>;
} // End Defs = [SCC]

defm S_FF0_I32_B32 : SOP1_32 <sop1<0x11, 0x0e>, "s_ff0_i32_b32", []>;
defm S_FF0_I32_B64 : SOP1_32_64 <sop1<0x12, 0x0f>, "s_ff0_i32_b64", []>;
defm S_FF1_I32_B32 : SOP1_32 <sop1<0x13, 0x10>, "s_ff1_i32_b32",
  [(set i32:$dst, (cttz_zero_undef i32:$src0))]
>;
defm S_FF1_I32_B64 : SOP1_32_64 <sop1<0x14, 0x11>, "s_ff1_i32_b64", []>;

defm S_FLBIT_I32_B32 : SOP1_32 <sop1<0x15, 0x12>, "s_flbit_i32_b32",
  [(set i32:$dst, (ctlz_zero_undef i32:$src0))]
>;

defm S_FLBIT_I32_B64 : SOP1_32_64 <sop1<0x16, 0x13>, "s_flbit_i32_b64", []>;
defm S_FLBIT_I32 : SOP1_32 <sop1<0x17, 0x14>, "s_flbit_i32",
  [(set i32:$dst, (int_AMDGPU_flbit_i32 i32:$src0))]
>;
defm S_FLBIT_I32_I64 : SOP1_32_64 <sop1<0x18, 0x15>, "s_flbit_i32_i64", []>;
defm S_SEXT_I32_I8 : SOP1_32 <sop1<0x19, 0x16>, "s_sext_i32_i8",
  [(set i32:$dst, (sext_inreg i32:$src0, i8))]
>;
defm S_SEXT_I32_I16 : SOP1_32 <sop1<0x1a, 0x17>, "s_sext_i32_i16",
  [(set i32:$dst, (sext_inreg i32:$src0, i16))]
>;

defm S_BITSET0_B32 : SOP1_32 <sop1<0x1b, 0x18>, "s_bitset0_b32", []>;
defm S_BITSET0_B64 : SOP1_64 <sop1<0x1c, 0x19>, "s_bitset0_b64", []>;
defm S_BITSET1_B32 : SOP1_32 <sop1<0x1d, 0x1a>, "s_bitset1_b32", []>;
defm S_BITSET1_B64 : SOP1_64 <sop1<0x1e, 0x1b>, "s_bitset1_b64", []>;
defm S_GETPC_B64 : SOP1_64_0 <sop1<0x1f, 0x1c>, "s_getpc_b64", []>;
defm S_SETPC_B64 : SOP1_64 <sop1<0x20, 0x1d>, "s_setpc_b64", []>;
defm S_SWAPPC_B64 : SOP1_64 <sop1<0x21, 0x1e>, "s_swappc_b64", []>;
defm S_RFE_B64 : SOP1_64 <sop1<0x22, 0x1f>, "s_rfe_b64", []>;

let hasSideEffects = 1, Uses = [EXEC], Defs = [EXEC, SCC] in {

defm S_AND_SAVEEXEC_B64 : SOP1_64 <sop1<0x24, 0x20>, "s_and_saveexec_b64", []>;
defm S_OR_SAVEEXEC_B64 : SOP1_64 <sop1<0x25, 0x21>, "s_or_saveexec_b64", []>;
defm S_XOR_SAVEEXEC_B64 : SOP1_64 <sop1<0x26, 0x22>, "s_xor_saveexec_b64", []>;
defm S_ANDN2_SAVEEXEC_B64 : SOP1_64 <sop1<0x27, 0x23>, "s_andn2_saveexec_b64", []>;
defm S_ORN2_SAVEEXEC_B64 : SOP1_64 <sop1<0x28, 0x24>, "s_orn2_saveexec_b64", []>;
defm S_NAND_SAVEEXEC_B64 : SOP1_64 <sop1<0x29, 0x25>, "s_nand_saveexec_b64", []>;
defm S_NOR_SAVEEXEC_B64 : SOP1_64 <sop1<0x2a, 0x26>, "s_nor_saveexec_b64", []>;
defm S_XNOR_SAVEEXEC_B64 : SOP1_64 <sop1<0x2b, 0x27>, "s_xnor_saveexec_b64", []>;

} // End hasSideEffects = 1, Uses = [EXEC], Defs = [EXEC, SCC]

defm S_QUADMASK_B32 : SOP1_32 <sop1<0x2c, 0x28>, "s_quadmask_b32", []>;
defm S_QUADMASK_B64 : SOP1_64 <sop1<0x2d, 0x29>, "s_quadmask_b64", []>;

let Uses = [M0] in {
defm S_MOVRELS_B32 : SOP1_32 <sop1<0x2e, 0x2a>, "s_movrels_b32", []>;
defm S_MOVRELS_B64 : SOP1_64 <sop1<0x2f, 0x2b>, "s_movrels_b64", []>;
defm S_MOVRELD_B32 : SOP1_32 <sop1<0x30, 0x2c>, "s_movreld_b32", []>;
defm S_MOVRELD_B64 : SOP1_64 <sop1<0x31, 0x2d>, "s_movreld_b64", []>;
} // End Uses = [M0]

defm S_CBRANCH_JOIN : SOP1_1 <sop1<0x32, 0x2e>, "s_cbranch_join", []>;
defm S_MOV_REGRD_B32 : SOP1_32 <sop1<0x33, 0x2f>, "s_mov_regrd_b32", []>;
let Defs = [SCC] in {
  defm S_ABS_I32 : SOP1_32 <sop1<0x34, 0x30>, "s_abs_i32", []>;
} // End Defs = [SCC]
defm S_MOV_FED_B32 : SOP1_32 <sop1<0x35, 0x31>, "s_mov_fed_b32", []>;

//===----------------------------------------------------------------------===//
// SOP2 Instructions
//===----------------------------------------------------------------------===//

let Defs = [SCC] in { // Carry out goes to SCC
let isCommutable = 1 in {
defm S_ADD_U32 : SOP2_32 <sop2<0x00>, "s_add_u32", []>;
defm S_ADD_I32 : SOP2_32 <sop2<0x02>, "s_add_i32",
  [(set i32:$dst, (add SSrc_32:$src0, SSrc_32:$src1))]
>;
} // End isCommutable = 1

defm S_SUB_U32 : SOP2_32 <sop2<0x01>, "s_sub_u32", []>;
defm S_SUB_I32 : SOP2_32 <sop2<0x03>, "s_sub_i32",
  [(set i32:$dst, (sub SSrc_32:$src0, SSrc_32:$src1))]
>;

let Uses = [SCC] in { // Carry in comes from SCC
let isCommutable = 1 in {
defm S_ADDC_U32 : SOP2_32 <sop2<0x04>, "s_addc_u32",
  [(set i32:$dst, (adde (i32 SSrc_32:$src0), (i32 SSrc_32:$src1)))]>;
} // End isCommutable = 1

defm S_SUBB_U32 : SOP2_32 <sop2<0x05>, "s_subb_u32",
  [(set i32:$dst, (sube (i32 SSrc_32:$src0), (i32 SSrc_32:$src1)))]>;
} // End Uses = [SCC]

defm S_MIN_I32 : SOP2_32 <sop2<0x06>, "s_min_i32",
  [(set i32:$dst, (smin i32:$src0, i32:$src1))]
>;
defm S_MIN_U32 : SOP2_32 <sop2<0x07>, "s_min_u32",
  [(set i32:$dst, (umin i32:$src0, i32:$src1))]
>;
defm S_MAX_I32 : SOP2_32 <sop2<0x08>, "s_max_i32",
  [(set i32:$dst, (smax i32:$src0, i32:$src1))]
>;
defm S_MAX_U32 : SOP2_32 <sop2<0x09>, "s_max_u32",
  [(set i32:$dst, (umax i32:$src0, i32:$src1))]
>;
} // End Defs = [SCC]


let Uses = [SCC] in {
  defm S_CSELECT_B32 : SOP2_32 <sop2<0x0a>, "s_cselect_b32", []>;
  defm S_CSELECT_B64 : SOP2_64 <sop2<0x0b>, "s_cselect_b64", []>;
} // End Uses = [SCC]

let Defs = [SCC] in {
defm S_AND_B32 : SOP2_32 <sop2<0x0e, 0x0c>, "s_and_b32",
  [(set i32:$dst, (and i32:$src0, i32:$src1))]
>;

defm S_AND_B64 : SOP2_64 <sop2<0x0f, 0x0d>, "s_and_b64",
  [(set i64:$dst, (and i64:$src0, i64:$src1))]
>;

defm S_OR_B32 : SOP2_32 <sop2<0x10, 0x0e>, "s_or_b32",
  [(set i32:$dst, (or i32:$src0, i32:$src1))]
>;

defm S_OR_B64 : SOP2_64 <sop2<0x11, 0x0f>, "s_or_b64",
  [(set i64:$dst, (or i64:$src0, i64:$src1))]
>;

defm S_XOR_B32 : SOP2_32 <sop2<0x12, 0x10>, "s_xor_b32",
  [(set i32:$dst, (xor i32:$src0, i32:$src1))]
>;

defm S_XOR_B64 : SOP2_64 <sop2<0x13, 0x11>, "s_xor_b64",
  [(set i64:$dst, (xor i64:$src0, i64:$src1))]
>;
defm S_ANDN2_B32 : SOP2_32 <sop2<0x14, 0x12>, "s_andn2_b32", []>;
defm S_ANDN2_B64 : SOP2_64 <sop2<0x15, 0x13>, "s_andn2_b64", []>;
defm S_ORN2_B32 : SOP2_32 <sop2<0x16, 0x14>, "s_orn2_b32", []>;
defm S_ORN2_B64 : SOP2_64 <sop2<0x17, 0x15>, "s_orn2_b64", []>;
defm S_NAND_B32 : SOP2_32 <sop2<0x18, 0x16>, "s_nand_b32", []>;
defm S_NAND_B64 : SOP2_64 <sop2<0x19, 0x17>, "s_nand_b64", []>;
defm S_NOR_B32 : SOP2_32 <sop2<0x1a, 0x18>, "s_nor_b32", []>;
defm S_NOR_B64 : SOP2_64 <sop2<0x1b, 0x19>, "s_nor_b64", []>;
defm S_XNOR_B32 : SOP2_32 <sop2<0x1c, 0x1a>, "s_xnor_b32", []>;
defm S_XNOR_B64 : SOP2_64 <sop2<0x1d, 0x1b>, "s_xnor_b64", []>;
} // End Defs = [SCC]

// Use added complexity so these patterns are preferred to the VALU patterns.
let AddedComplexity = 1 in {
let Defs = [SCC] in {

defm S_LSHL_B32 : SOP2_32 <sop2<0x1e, 0x1c>, "s_lshl_b32",
  [(set i32:$dst, (shl i32:$src0, i32:$src1))]
>;
defm S_LSHL_B64 : SOP2_64_32 <sop2<0x1f, 0x1d>, "s_lshl_b64",
  [(set i64:$dst, (shl i64:$src0, i32:$src1))]
>;
defm S_LSHR_B32 : SOP2_32 <sop2<0x20, 0x1e>, "s_lshr_b32",
  [(set i32:$dst, (srl i32:$src0, i32:$src1))]
>;
defm S_LSHR_B64 : SOP2_64_32 <sop2<0x21, 0x1f>, "s_lshr_b64",
  [(set i64:$dst, (srl i64:$src0, i32:$src1))]
>;
defm S_ASHR_I32 : SOP2_32 <sop2<0x22, 0x20>, "s_ashr_i32",
  [(set i32:$dst, (sra i32:$src0, i32:$src1))]
>;
defm S_ASHR_I64 : SOP2_64_32 <sop2<0x23, 0x21>, "s_ashr_i64",
  [(set i64:$dst, (sra i64:$src0, i32:$src1))]
>;
} // End Defs = [SCC]

defm S_BFM_B32 : SOP2_32 <sop2<0x24, 0x22>, "s_bfm_b32",
  [(set i32:$dst, (AMDGPUbfm i32:$src0, i32:$src1))]>;
defm S_BFM_B64 : SOP2_64 <sop2<0x25, 0x23>, "s_bfm_b64", []>;
defm S_MUL_I32 : SOP2_32 <sop2<0x26, 0x24>, "s_mul_i32",
  [(set i32:$dst, (mul i32:$src0, i32:$src1))]
>;

} // End AddedComplexity = 1

let Defs = [SCC] in {
defm S_BFE_U32 : SOP2_32 <sop2<0x27, 0x25>, "s_bfe_u32", []>;
defm S_BFE_I32 : SOP2_32 <sop2<0x28, 0x26>, "s_bfe_i32", []>;
defm S_BFE_U64 : SOP2_64 <sop2<0x29, 0x27>, "s_bfe_u64", []>;
defm S_BFE_I64 : SOP2_64_32 <sop2<0x2a, 0x28>, "s_bfe_i64", []>;
} // End Defs = [SCC]

let sdst = 0 in {
defm S_CBRANCH_G_FORK : SOP2_m <
  sop2<0x2b, 0x29>, "s_cbranch_g_fork", (outs),
  (ins SReg_64:$src0, SReg_64:$src1), "s_cbranch_g_fork $src0, $src1", []
>;
}

let Defs = [SCC] in {
defm S_ABSDIFF_I32 : SOP2_32 <sop2<0x2c, 0x2a>, "s_absdiff_i32", []>;
} // End Defs = [SCC]

//===----------------------------------------------------------------------===//
// SOPC Instructions
//===----------------------------------------------------------------------===//

def S_CMP_EQ_I32 : SOPC_32 <0x00000000, "s_cmp_eq_i32">;
def S_CMP_LG_I32 : SOPC_32 <0x00000001, "s_cmp_lg_i32">;
def S_CMP_GT_I32 : SOPC_32 <0x00000002, "s_cmp_gt_i32">;
def S_CMP_GE_I32 : SOPC_32 <0x00000003, "s_cmp_ge_i32">;
def S_CMP_LT_I32 : SOPC_32 <0x00000004, "s_cmp_lt_i32">;
def S_CMP_LE_I32 : SOPC_32 <0x00000005, "s_cmp_le_i32">;
def S_CMP_EQ_U32 : SOPC_32 <0x00000006, "s_cmp_eq_u32">;
def S_CMP_LG_U32 : SOPC_32 <0x00000007, "s_cmp_lg_u32">;
def S_CMP_GT_U32 : SOPC_32 <0x00000008, "s_cmp_gt_u32">;
def S_CMP_GE_U32 : SOPC_32 <0x00000009, "s_cmp_ge_u32">;
def S_CMP_LT_U32 : SOPC_32 <0x0000000a, "s_cmp_lt_u32">;
def S_CMP_LE_U32 : SOPC_32 <0x0000000b, "s_cmp_le_u32">;
////def S_BITCMP0_B32 : SOPC_BITCMP0 <0x0000000c, "s_bitcmp0_b32", []>;
////def S_BITCMP1_B32 : SOPC_BITCMP1 <0x0000000d, "s_bitcmp1_b32", []>;
////def S_BITCMP0_B64 : SOPC_BITCMP0 <0x0000000e, "s_bitcmp0_b64", []>;
////def S_BITCMP1_B64 : SOPC_BITCMP1 <0x0000000f, "s_bitcmp1_b64", []>;
//def S_SETVSKIP : SOPC_ <0x00000010, "s_setvskip", []>;

//===----------------------------------------------------------------------===//
// SOPK Instructions
//===----------------------------------------------------------------------===//

let isReMaterializable = 1, isMoveImm = 1 in {
defm S_MOVK_I32 : SOPK_32 <sopk<0x00>, "s_movk_i32", []>;
} // End isReMaterializable = 1
let Uses = [SCC] in {
  defm S_CMOVK_I32 : SOPK_32 <sopk<0x02, 0x01>, "s_cmovk_i32", []>;
}

let isCompare = 1 in {

/*
This instruction is disabled for now until we can figure out how to teach
the instruction selector to correctly use the  S_CMP* vs V_CMP*
instructions.

When this instruction is enabled the code generator sometimes produces this
invalid sequence:

SCC = S_CMPK_EQ_I32 SGPR0, imm
VCC = COPY SCC
VGPR0 = V_CNDMASK VCC, VGPR0, VGPR1

defm S_CMPK_EQ_I32 : SOPK_SCC <sopk<0x03, 0x02>, "s_cmpk_eq_i32",
  [(set i1:$dst, (setcc i32:$src0, imm:$src1, SETEQ))]
>;
*/

defm S_CMPK_EQ_I32 : SOPK_SCC <sopk<0x03, 0x02>, "s_cmpk_eq_i32", []>;
defm S_CMPK_LG_I32 : SOPK_SCC <sopk<0x04, 0x03>, "s_cmpk_lg_i32", []>;
defm S_CMPK_GT_I32 : SOPK_SCC <sopk<0x05, 0x04>, "s_cmpk_gt_i32", []>;
defm S_CMPK_GE_I32 : SOPK_SCC <sopk<0x06, 0x05>, "s_cmpk_ge_i32", []>;
defm S_CMPK_LT_I32 : SOPK_SCC <sopk<0x07, 0x06>, "s_cmpk_lt_i32", []>;
defm S_CMPK_LE_I32 : SOPK_SCC <sopk<0x08, 0x07>, "s_cmpk_le_i32", []>;
defm S_CMPK_EQ_U32 : SOPK_SCC <sopk<0x09, 0x08>, "s_cmpk_eq_u32", []>;
defm S_CMPK_LG_U32 : SOPK_SCC <sopk<0x0a, 0x09>, "s_cmpk_lg_u32", []>;
defm S_CMPK_GT_U32 : SOPK_SCC <sopk<0x0b, 0x0a>, "s_cmpk_gt_u32", []>;
defm S_CMPK_GE_U32 : SOPK_SCC <sopk<0x0c, 0x0b>, "s_cmpk_ge_u32", []>;
defm S_CMPK_LT_U32 : SOPK_SCC <sopk<0x0d, 0x0c>, "s_cmpk_lt_u32", []>;
defm S_CMPK_LE_U32 : SOPK_SCC <sopk<0x0e, 0x0d>, "s_cmpk_le_u32", []>;
} // End isCompare = 1

let Defs = [SCC], isCommutable = 1, DisableEncoding = "$src0",
    Constraints = "$sdst = $src0" in {
  defm S_ADDK_I32 : SOPK_32TIE <sopk<0x0f, 0x0e>, "s_addk_i32", []>;
  defm S_MULK_I32 : SOPK_32TIE <sopk<0x10, 0x0f>, "s_mulk_i32", []>;
}

defm S_CBRANCH_I_FORK : SOPK_m <
  sopk<0x11, 0x10>, "s_cbranch_i_fork", (outs),
  (ins SReg_64:$sdst, u16imm:$simm16), " $sdst, $simm16"
>;
defm S_GETREG_B32 : SOPK_32 <sopk<0x12, 0x11>, "s_getreg_b32", []>;
defm S_SETREG_B32 : SOPK_m <
  sopk<0x13, 0x12>, "s_setreg_b32", (outs),
  (ins SReg_32:$sdst, u16imm:$simm16), " $sdst, $simm16"
>;
// FIXME: Not on SI?
//defm S_GETREG_REGRD_B32 : SOPK_32 <sopk<0x14, 0x13>, "s_getreg_regrd_b32", []>;
defm S_SETREG_IMM32_B32 : SOPK_IMM32 <
  sopk<0x15, 0x14>, "s_setreg_imm32_b32", (outs),
  (ins i32imm:$imm, u16imm:$simm16), " $imm, $simm16"
>;

//===----------------------------------------------------------------------===//
// SOPP Instructions
//===----------------------------------------------------------------------===//

def S_NOP : SOPP <0x00000000, (ins i16imm:$simm16), "s_nop $simm16">;

let isTerminator = 1 in {

def S_ENDPGM : SOPP <0x00000001, (ins), "s_endpgm",
  [(IL_retflag)]> {
  let simm16 = 0;
  let isBarrier = 1;
  let hasCtrlDep = 1;
}

let isBranch = 1 in {
def S_BRANCH : SOPP <
  0x00000002, (ins sopp_brtarget:$simm16), "s_branch $simm16",
  [(br bb:$simm16)]> {
  let isBarrier = 1;
}

let Uses = [SCC] in {
def S_CBRANCH_SCC0 : SOPP <
  0x00000004, (ins sopp_brtarget:$simm16),
  "s_cbranch_scc0 $simm16"
>;
def S_CBRANCH_SCC1 : SOPP <
  0x00000005, (ins sopp_brtarget:$simm16),
  "s_cbranch_scc1 $simm16"
>;
} // End Uses = [SCC]

let Uses = [VCC] in {
def S_CBRANCH_VCCZ : SOPP <
  0x00000006, (ins sopp_brtarget:$simm16),
  "s_cbranch_vccz $simm16"
>;
def S_CBRANCH_VCCNZ : SOPP <
  0x00000007, (ins sopp_brtarget:$simm16),
  "s_cbranch_vccnz $simm16"
>;
} // End Uses = [VCC]

let Uses = [EXEC] in {
def S_CBRANCH_EXECZ : SOPP <
  0x00000008, (ins sopp_brtarget:$simm16),
  "s_cbranch_execz $simm16"
>;
def S_CBRANCH_EXECNZ : SOPP <
  0x00000009, (ins sopp_brtarget:$simm16),
  "s_cbranch_execnz $simm16"
>;
} // End Uses = [EXEC]


} // End isBranch = 1
} // End isTerminator = 1

let hasSideEffects = 1 in {
def S_BARRIER : SOPP <0x0000000a, (ins), "s_barrier",
  [(int_AMDGPU_barrier_local)]
> {
  let SchedRW = [WriteBarrier];
  let simm16 = 0;
  let mayLoad = 1;
  let mayStore = 1;
  let isConvergent = 1;
}

def S_WAITCNT : SOPP <0x0000000c, (ins WAIT_FLAG:$simm16), "s_waitcnt $simm16">;
def S_SETHALT : SOPP <0x0000000d, (ins i16imm:$simm16), "s_sethalt $simm16">;
def S_SLEEP : SOPP <0x0000000e, (ins i16imm:$simm16), "s_sleep $simm16">;
def S_SETPRIO : SOPP <0x0000000f, (ins i16imm:$sim16), "s_setprio $sim16">;

let Uses = [EXEC, M0] in {
  def S_SENDMSG : SOPP <0x00000010, (ins SendMsgImm:$simm16), "s_sendmsg $simm16",
      [(AMDGPUsendmsg (i32 imm:$simm16))]
  >;
} // End Uses = [EXEC, M0]

def S_SENDMSGHALT : SOPP <0x00000011, (ins i16imm:$simm16), "s_sendmsghalt $simm16">;
def S_TRAP : SOPP <0x00000012, (ins i16imm:$simm16), "s_trap $simm16">;
def S_ICACHE_INV : SOPP <0x00000013, (ins), "s_icache_inv"> {
	let simm16 = 0;
}
def S_INCPERFLEVEL : SOPP <0x00000014, (ins i16imm:$simm16), "s_incperflevel $simm16">;
def S_DECPERFLEVEL : SOPP <0x00000015, (ins i16imm:$simm16), "s_decperflevel $simm16">;
def S_TTRACEDATA : SOPP <0x00000016, (ins), "s_ttracedata"> {
  let simm16 = 0;
}
} // End hasSideEffects

//===----------------------------------------------------------------------===//
// VOPC Instructions
//===----------------------------------------------------------------------===//

let isCompare = 1, isCommutable = 1 in {

defm V_CMP_F_F32 : VOPC_F32 <vopc<0x0, 0x40>, "v_cmp_f_f32">;
defm V_CMP_LT_F32 : VOPC_F32 <vopc<0x1, 0x41>, "v_cmp_lt_f32", COND_OLT, "v_cmp_gt_f32">;
defm V_CMP_EQ_F32 : VOPC_F32 <vopc<0x2, 0x42>, "v_cmp_eq_f32", COND_OEQ>;
defm V_CMP_LE_F32 : VOPC_F32 <vopc<0x3, 0x43>, "v_cmp_le_f32", COND_OLE, "v_cmp_ge_f32">;
defm V_CMP_GT_F32 : VOPC_F32 <vopc<0x4, 0x44>, "v_cmp_gt_f32", COND_OGT>;
defm V_CMP_LG_F32 : VOPC_F32 <vopc<0x5, 0x45>, "v_cmp_lg_f32", COND_ONE>;
defm V_CMP_GE_F32 : VOPC_F32 <vopc<0x6, 0x46>, "v_cmp_ge_f32", COND_OGE>;
defm V_CMP_O_F32 : VOPC_F32 <vopc<0x7, 0x47>, "v_cmp_o_f32", COND_O>;
defm V_CMP_U_F32 : VOPC_F32 <vopc<0x8, 0x48>, "v_cmp_u_f32", COND_UO>;
defm V_CMP_NGE_F32 : VOPC_F32 <vopc<0x9, 0x49>, "v_cmp_nge_f32",  COND_ULT, "v_cmp_nle_f32">;
defm V_CMP_NLG_F32 : VOPC_F32 <vopc<0xa, 0x4a>, "v_cmp_nlg_f32", COND_UEQ>;
defm V_CMP_NGT_F32 : VOPC_F32 <vopc<0xb, 0x4b>, "v_cmp_ngt_f32", COND_ULE, "v_cmp_nlt_f32">;
defm V_CMP_NLE_F32 : VOPC_F32 <vopc<0xc, 0x4c>, "v_cmp_nle_f32", COND_UGT>;
defm V_CMP_NEQ_F32 : VOPC_F32 <vopc<0xd, 0x4d>, "v_cmp_neq_f32", COND_UNE>;
defm V_CMP_NLT_F32 : VOPC_F32 <vopc<0xe, 0x4e>, "v_cmp_nlt_f32", COND_UGE>;
defm V_CMP_TRU_F32 : VOPC_F32 <vopc<0xf, 0x4f>, "v_cmp_tru_f32">;


defm V_CMPX_F_F32 : VOPCX_F32 <vopc<0x10, 0x50>, "v_cmpx_f_f32">;
defm V_CMPX_LT_F32 : VOPCX_F32 <vopc<0x11, 0x51>, "v_cmpx_lt_f32", "v_cmpx_gt_f32">;
defm V_CMPX_EQ_F32 : VOPCX_F32 <vopc<0x12, 0x52>, "v_cmpx_eq_f32">;
defm V_CMPX_LE_F32 : VOPCX_F32 <vopc<0x13, 0x53>, "v_cmpx_le_f32", "v_cmpx_ge_f32">;
defm V_CMPX_GT_F32 : VOPCX_F32 <vopc<0x14, 0x54>, "v_cmpx_gt_f32">;
defm V_CMPX_LG_F32 : VOPCX_F32 <vopc<0x15, 0x55>, "v_cmpx_lg_f32">;
defm V_CMPX_GE_F32 : VOPCX_F32 <vopc<0x16, 0x56>, "v_cmpx_ge_f32">;
defm V_CMPX_O_F32 : VOPCX_F32 <vopc<0x17, 0x57>, "v_cmpx_o_f32">;
defm V_CMPX_U_F32 : VOPCX_F32 <vopc<0x18, 0x58>, "v_cmpx_u_f32">;
defm V_CMPX_NGE_F32 : VOPCX_F32 <vopc<0x19, 0x59>, "v_cmpx_nge_f32">;
defm V_CMPX_NLG_F32 : VOPCX_F32 <vopc<0x1a, 0x5a>, "v_cmpx_nlg_f32">;
defm V_CMPX_NGT_F32 : VOPCX_F32 <vopc<0x1b, 0x5b>, "v_cmpx_ngt_f32">;
defm V_CMPX_NLE_F32 : VOPCX_F32 <vopc<0x1c, 0x5c>, "v_cmpx_nle_f32">;
defm V_CMPX_NEQ_F32 : VOPCX_F32 <vopc<0x1d, 0x5d>, "v_cmpx_neq_f32">;
defm V_CMPX_NLT_F32 : VOPCX_F32 <vopc<0x1e, 0x5e>, "v_cmpx_nlt_f32">;
defm V_CMPX_TRU_F32 : VOPCX_F32 <vopc<0x1f, 0x5f>, "v_cmpx_tru_f32">;


defm V_CMP_F_F64 : VOPC_F64 <vopc<0x20, 0x60>, "v_cmp_f_f64">;
defm V_CMP_LT_F64 : VOPC_F64 <vopc<0x21, 0x61>, "v_cmp_lt_f64", COND_OLT, "v_cmp_gt_f64">;
defm V_CMP_EQ_F64 : VOPC_F64 <vopc<0x22, 0x62>, "v_cmp_eq_f64", COND_OEQ>;
defm V_CMP_LE_F64 : VOPC_F64 <vopc<0x23, 0x63>, "v_cmp_le_f64", COND_OLE, "v_cmp_ge_f64">;
defm V_CMP_GT_F64 : VOPC_F64 <vopc<0x24, 0x64>, "v_cmp_gt_f64", COND_OGT>;
defm V_CMP_LG_F64 : VOPC_F64 <vopc<0x25, 0x65>, "v_cmp_lg_f64", COND_ONE>;
defm V_CMP_GE_F64 : VOPC_F64 <vopc<0x26, 0x66>, "v_cmp_ge_f64", COND_OGE>;
defm V_CMP_O_F64 : VOPC_F64 <vopc<0x27, 0x67>, "v_cmp_o_f64", COND_O>;
defm V_CMP_U_F64 : VOPC_F64 <vopc<0x28, 0x68>, "v_cmp_u_f64", COND_UO>;
defm V_CMP_NGE_F64 : VOPC_F64 <vopc<0x29, 0x69>, "v_cmp_nge_f64", COND_ULT, "v_cmp_nle_f64">;
defm V_CMP_NLG_F64 : VOPC_F64 <vopc<0x2a, 0x6a>, "v_cmp_nlg_f64", COND_UEQ>;
defm V_CMP_NGT_F64 : VOPC_F64 <vopc<0x2b, 0x6b>, "v_cmp_ngt_f64", COND_ULE, "v_cmp_nlt_f64">;
defm V_CMP_NLE_F64 : VOPC_F64 <vopc<0x2c, 0x6c>, "v_cmp_nle_f64", COND_UGT>;
defm V_CMP_NEQ_F64 : VOPC_F64 <vopc<0x2d, 0x6d>, "v_cmp_neq_f64", COND_UNE>;
defm V_CMP_NLT_F64 : VOPC_F64 <vopc<0x2e, 0x6e>, "v_cmp_nlt_f64", COND_UGE>;
defm V_CMP_TRU_F64 : VOPC_F64 <vopc<0x2f, 0x6f>, "v_cmp_tru_f64">;


defm V_CMPX_F_F64 : VOPCX_F64 <vopc<0x30, 0x70>, "v_cmpx_f_f64">;
defm V_CMPX_LT_F64 : VOPCX_F64 <vopc<0x31, 0x71>, "v_cmpx_lt_f64", "v_cmpx_gt_f64">;
defm V_CMPX_EQ_F64 : VOPCX_F64 <vopc<0x32, 0x72>, "v_cmpx_eq_f64">;
defm V_CMPX_LE_F64 : VOPCX_F64 <vopc<0x33, 0x73>, "v_cmpx_le_f64", "v_cmpx_ge_f64">;
defm V_CMPX_GT_F64 : VOPCX_F64 <vopc<0x34, 0x74>, "v_cmpx_gt_f64">;
defm V_CMPX_LG_F64 : VOPCX_F64 <vopc<0x35, 0x75>, "v_cmpx_lg_f64">;
defm V_CMPX_GE_F64 : VOPCX_F64 <vopc<0x36, 0x76>, "v_cmpx_ge_f64">;
defm V_CMPX_O_F64 : VOPCX_F64 <vopc<0x37, 0x77>, "v_cmpx_o_f64">;
defm V_CMPX_U_F64 : VOPCX_F64 <vopc<0x38, 0x78>, "v_cmpx_u_f64">;
defm V_CMPX_NGE_F64 : VOPCX_F64 <vopc<0x39, 0x79>, "v_cmpx_nge_f64", "v_cmpx_nle_f64">;
defm V_CMPX_NLG_F64 : VOPCX_F64 <vopc<0x3a, 0x7a>, "v_cmpx_nlg_f64">;
defm V_CMPX_NGT_F64 : VOPCX_F64 <vopc<0x3b, 0x7b>, "v_cmpx_ngt_f64", "v_cmpx_nlt_f64">;
defm V_CMPX_NLE_F64 : VOPCX_F64 <vopc<0x3c, 0x7c>, "v_cmpx_nle_f64">;
defm V_CMPX_NEQ_F64 : VOPCX_F64 <vopc<0x3d, 0x7d>, "v_cmpx_neq_f64">;
defm V_CMPX_NLT_F64 : VOPCX_F64 <vopc<0x3e, 0x7e>, "v_cmpx_nlt_f64">;
defm V_CMPX_TRU_F64 : VOPCX_F64 <vopc<0x3f, 0x7f>, "v_cmpx_tru_f64">;


let SubtargetPredicate = isSICI in {

defm V_CMPS_F_F32 : VOPC_F32 <vopc<0x40>, "v_cmps_f_f32">;
defm V_CMPS_LT_F32 : VOPC_F32 <vopc<0x41>, "v_cmps_lt_f32", COND_NULL, "v_cmps_gt_f32">;
defm V_CMPS_EQ_F32 : VOPC_F32 <vopc<0x42>, "v_cmps_eq_f32">;
defm V_CMPS_LE_F32 : VOPC_F32 <vopc<0x43>, "v_cmps_le_f32", COND_NULL, "v_cmps_ge_f32">;
defm V_CMPS_GT_F32 : VOPC_F32 <vopc<0x44>, "v_cmps_gt_f32">;
defm V_CMPS_LG_F32 : VOPC_F32 <vopc<0x45>, "v_cmps_lg_f32">;
defm V_CMPS_GE_F32 : VOPC_F32 <vopc<0x46>, "v_cmps_ge_f32">;
defm V_CMPS_O_F32 : VOPC_F32 <vopc<0x47>, "v_cmps_o_f32">;
defm V_CMPS_U_F32 : VOPC_F32 <vopc<0x48>, "v_cmps_u_f32">;
defm V_CMPS_NGE_F32 : VOPC_F32 <vopc<0x49>, "v_cmps_nge_f32", COND_NULL, "v_cmps_nle_f32">;
defm V_CMPS_NLG_F32 : VOPC_F32 <vopc<0x4a>, "v_cmps_nlg_f32">;
defm V_CMPS_NGT_F32 : VOPC_F32 <vopc<0x4b>, "v_cmps_ngt_f32", COND_NULL, "v_cmps_nlt_f32">;
defm V_CMPS_NLE_F32 : VOPC_F32 <vopc<0x4c>, "v_cmps_nle_f32">;
defm V_CMPS_NEQ_F32 : VOPC_F32 <vopc<0x4d>, "v_cmps_neq_f32">;
defm V_CMPS_NLT_F32 : VOPC_F32 <vopc<0x4e>, "v_cmps_nlt_f32">;
defm V_CMPS_TRU_F32 : VOPC_F32 <vopc<0x4f>, "v_cmps_tru_f32">;


defm V_CMPSX_F_F32 : VOPCX_F32 <vopc<0x50>, "v_cmpsx_f_f32">;
defm V_CMPSX_LT_F32 : VOPCX_F32 <vopc<0x51>, "v_cmpsx_lt_f32", "v_cmpsx_gt_f32">;
defm V_CMPSX_EQ_F32 : VOPCX_F32 <vopc<0x52>, "v_cmpsx_eq_f32">;
defm V_CMPSX_LE_F32 : VOPCX_F32 <vopc<0x53>, "v_cmpsx_le_f32", "v_cmpsx_ge_f32">;
defm V_CMPSX_GT_F32 : VOPCX_F32 <vopc<0x54>, "v_cmpsx_gt_f32">;
defm V_CMPSX_LG_F32 : VOPCX_F32 <vopc<0x55>, "v_cmpsx_lg_f32">;
defm V_CMPSX_GE_F32 : VOPCX_F32 <vopc<0x56>, "v_cmpsx_ge_f32">;
defm V_CMPSX_O_F32 : VOPCX_F32 <vopc<0x57>, "v_cmpsx_o_f32">;
defm V_CMPSX_U_F32 : VOPCX_F32 <vopc<0x58>, "v_cmpsx_u_f32">;
defm V_CMPSX_NGE_F32 : VOPCX_F32 <vopc<0x59>, "v_cmpsx_nge_f32", "v_cmpsx_nle_f32">;
defm V_CMPSX_NLG_F32 : VOPCX_F32 <vopc<0x5a>, "v_cmpsx_nlg_f32">;
defm V_CMPSX_NGT_F32 : VOPCX_F32 <vopc<0x5b>, "v_cmpsx_ngt_f32", "v_cmpsx_nlt_f32">;
defm V_CMPSX_NLE_F32 : VOPCX_F32 <vopc<0x5c>, "v_cmpsx_nle_f32">;
defm V_CMPSX_NEQ_F32 : VOPCX_F32 <vopc<0x5d>, "v_cmpsx_neq_f32">;
defm V_CMPSX_NLT_F32 : VOPCX_F32 <vopc<0x5e>, "v_cmpsx_nlt_f32">;
defm V_CMPSX_TRU_F32 : VOPCX_F32 <vopc<0x5f>, "v_cmpsx_tru_f32">;


defm V_CMPS_F_F64 : VOPC_F64 <vopc<0x60>, "v_cmps_f_f64">;
defm V_CMPS_LT_F64 : VOPC_F64 <vopc<0x61>, "v_cmps_lt_f64", COND_NULL, "v_cmps_gt_f64">;
defm V_CMPS_EQ_F64 : VOPC_F64 <vopc<0x62>, "v_cmps_eq_f64">;
defm V_CMPS_LE_F64 : VOPC_F64 <vopc<0x63>, "v_cmps_le_f64", COND_NULL, "v_cmps_ge_f64">;
defm V_CMPS_GT_F64 : VOPC_F64 <vopc<0x64>, "v_cmps_gt_f64">;
defm V_CMPS_LG_F64 : VOPC_F64 <vopc<0x65>, "v_cmps_lg_f64">;
defm V_CMPS_GE_F64 : VOPC_F64 <vopc<0x66>, "v_cmps_ge_f64">;
defm V_CMPS_O_F64 : VOPC_F64 <vopc<0x67>, "v_cmps_o_f64">;
defm V_CMPS_U_F64 : VOPC_F64 <vopc<0x68>, "v_cmps_u_f64">;
defm V_CMPS_NGE_F64 : VOPC_F64 <vopc<0x69>, "v_cmps_nge_f64", COND_NULL, "v_cmps_nle_f64">;
defm V_CMPS_NLG_F64 : VOPC_F64 <vopc<0x6a>, "v_cmps_nlg_f64">;
defm V_CMPS_NGT_F64 : VOPC_F64 <vopc<0x6b>, "v_cmps_ngt_f64", COND_NULL, "v_cmps_nlt_f64">;
defm V_CMPS_NLE_F64 : VOPC_F64 <vopc<0x6c>, "v_cmps_nle_f64">;
defm V_CMPS_NEQ_F64 : VOPC_F64 <vopc<0x6d>, "v_cmps_neq_f64">;
defm V_CMPS_NLT_F64 : VOPC_F64 <vopc<0x6e>, "v_cmps_nlt_f64">;
defm V_CMPS_TRU_F64 : VOPC_F64 <vopc<0x6f>, "v_cmps_tru_f64">;


defm V_CMPSX_F_F64 : VOPCX_F64 <vopc<0x70>, "v_cmpsx_f_f64">;
defm V_CMPSX_LT_F64 : VOPCX_F64 <vopc<0x71>, "v_cmpsx_lt_f64", "v_cmpsx_gt_f64">;
defm V_CMPSX_EQ_F64 : VOPCX_F64 <vopc<0x72>, "v_cmpsx_eq_f64">;
defm V_CMPSX_LE_F64 : VOPCX_F64 <vopc<0x73>, "v_cmpsx_le_f64", "v_cmpsx_ge_f64">;
defm V_CMPSX_GT_F64 : VOPCX_F64 <vopc<0x74>, "v_cmpsx_gt_f64">;
defm V_CMPSX_LG_F64 : VOPCX_F64 <vopc<0x75>, "v_cmpsx_lg_f64">;
defm V_CMPSX_GE_F64 : VOPCX_F64 <vopc<0x76>, "v_cmpsx_ge_f64">;
defm V_CMPSX_O_F64 : VOPCX_F64 <vopc<0x77>, "v_cmpsx_o_f64">;
defm V_CMPSX_U_F64 : VOPCX_F64 <vopc<0x78>, "v_cmpsx_u_f64">;
defm V_CMPSX_NGE_F64 : VOPCX_F64 <vopc<0x79>, "v_cmpsx_nge_f64", "v_cmpsx_nle_f64">;
defm V_CMPSX_NLG_F64 : VOPCX_F64 <vopc<0x7a>, "v_cmpsx_nlg_f64">;
defm V_CMPSX_NGT_F64 : VOPCX_F64 <vopc<0x7b>, "v_cmpsx_ngt_f64", "v_cmpsx_nlt_f64">;
defm V_CMPSX_NLE_F64 : VOPCX_F64 <vopc<0x7c>, "v_cmpsx_nle_f64">;
defm V_CMPSX_NEQ_F64 : VOPCX_F64 <vopc<0x7d>, "v_cmpsx_neq_f64">;
defm V_CMPSX_NLT_F64 : VOPCX_F64 <vopc<0x7e>, "v_cmpsx_nlt_f64">;
defm V_CMPSX_TRU_F64 : VOPCX_F64 <vopc<0x7f>, "v_cmpsx_tru_f64">;

} // End SubtargetPredicate = isSICI

defm V_CMP_F_I32 : VOPC_I32 <vopc<0x80, 0xc0>, "v_cmp_f_i32">;
defm V_CMP_LT_I32 : VOPC_I32 <vopc<0x81, 0xc1>, "v_cmp_lt_i32", COND_SLT, "v_cmp_gt_i32">;
defm V_CMP_EQ_I32 : VOPC_I32 <vopc<0x82, 0xc2>, "v_cmp_eq_i32", COND_EQ>;
defm V_CMP_LE_I32 : VOPC_I32 <vopc<0x83, 0xc3>, "v_cmp_le_i32", COND_SLE, "v_cmp_ge_i32">;
defm V_CMP_GT_I32 : VOPC_I32 <vopc<0x84, 0xc4>, "v_cmp_gt_i32", COND_SGT>;
defm V_CMP_NE_I32 : VOPC_I32 <vopc<0x85, 0xc5>, "v_cmp_ne_i32", COND_NE>;
defm V_CMP_GE_I32 : VOPC_I32 <vopc<0x86, 0xc6>, "v_cmp_ge_i32", COND_SGE>;
defm V_CMP_T_I32 : VOPC_I32 <vopc<0x87, 0xc7>, "v_cmp_t_i32">;


defm V_CMPX_F_I32 : VOPCX_I32 <vopc<0x90, 0xd0>, "v_cmpx_f_i32">;
defm V_CMPX_LT_I32 : VOPCX_I32 <vopc<0x91, 0xd1>, "v_cmpx_lt_i32", "v_cmpx_gt_i32">;
defm V_CMPX_EQ_I32 : VOPCX_I32 <vopc<0x92, 0xd2>, "v_cmpx_eq_i32">;
defm V_CMPX_LE_I32 : VOPCX_I32 <vopc<0x93, 0xd3>, "v_cmpx_le_i32", "v_cmpx_ge_i32">;
defm V_CMPX_GT_I32 : VOPCX_I32 <vopc<0x94, 0xd4>, "v_cmpx_gt_i32">;
defm V_CMPX_NE_I32 : VOPCX_I32 <vopc<0x95, 0xd5>, "v_cmpx_ne_i32">;
defm V_CMPX_GE_I32 : VOPCX_I32 <vopc<0x96, 0xd6>, "v_cmpx_ge_i32">;
defm V_CMPX_T_I32 : VOPCX_I32 <vopc<0x97, 0xd7>, "v_cmpx_t_i32">;


defm V_CMP_F_I64 : VOPC_I64 <vopc<0xa0, 0xe0>, "v_cmp_f_i64">;
defm V_CMP_LT_I64 : VOPC_I64 <vopc<0xa1, 0xe1>, "v_cmp_lt_i64", COND_SLT, "v_cmp_gt_i64">;
defm V_CMP_EQ_I64 : VOPC_I64 <vopc<0xa2, 0xe2>, "v_cmp_eq_i64", COND_EQ>;
defm V_CMP_LE_I64 : VOPC_I64 <vopc<0xa3, 0xe3>, "v_cmp_le_i64", COND_SLE, "v_cmp_ge_i64">;
defm V_CMP_GT_I64 : VOPC_I64 <vopc<0xa4, 0xe4>, "v_cmp_gt_i64", COND_SGT>;
defm V_CMP_NE_I64 : VOPC_I64 <vopc<0xa5, 0xe5>, "v_cmp_ne_i64", COND_NE>;
defm V_CMP_GE_I64 : VOPC_I64 <vopc<0xa6, 0xe6>, "v_cmp_ge_i64", COND_SGE>;
defm V_CMP_T_I64 : VOPC_I64 <vopc<0xa7, 0xe7>, "v_cmp_t_i64">;


defm V_CMPX_F_I64 : VOPCX_I64 <vopc<0xb0, 0xf0>, "v_cmpx_f_i64">;
defm V_CMPX_LT_I64 : VOPCX_I64 <vopc<0xb1, 0xf1>, "v_cmpx_lt_i64", "v_cmpx_gt_i64">;
defm V_CMPX_EQ_I64 : VOPCX_I64 <vopc<0xb2, 0xf2>, "v_cmpx_eq_i64">;
defm V_CMPX_LE_I64 : VOPCX_I64 <vopc<0xb3, 0xf3>, "v_cmpx_le_i64", "v_cmpx_ge_i64">;
defm V_CMPX_GT_I64 : VOPCX_I64 <vopc<0xb4, 0xf4>, "v_cmpx_gt_i64">;
defm V_CMPX_NE_I64 : VOPCX_I64 <vopc<0xb5, 0xf5>, "v_cmpx_ne_i64">;
defm V_CMPX_GE_I64 : VOPCX_I64 <vopc<0xb6, 0xf6>, "v_cmpx_ge_i64">;
defm V_CMPX_T_I64 : VOPCX_I64 <vopc<0xb7, 0xf7>, "v_cmpx_t_i64">;


defm V_CMP_F_U32 : VOPC_I32 <vopc<0xc0, 0xc8>, "v_cmp_f_u32">;
defm V_CMP_LT_U32 : VOPC_I32 <vopc<0xc1, 0xc9>, "v_cmp_lt_u32", COND_ULT, "v_cmp_gt_u32">;
defm V_CMP_EQ_U32 : VOPC_I32 <vopc<0xc2, 0xca>, "v_cmp_eq_u32", COND_EQ>;
defm V_CMP_LE_U32 : VOPC_I32 <vopc<0xc3, 0xcb>, "v_cmp_le_u32", COND_ULE, "v_cmp_ge_u32">;
defm V_CMP_GT_U32 : VOPC_I32 <vopc<0xc4, 0xcc>, "v_cmp_gt_u32", COND_UGT>;
defm V_CMP_NE_U32 : VOPC_I32 <vopc<0xc5, 0xcd>, "v_cmp_ne_u32", COND_NE>;
defm V_CMP_GE_U32 : VOPC_I32 <vopc<0xc6, 0xce>, "v_cmp_ge_u32", COND_UGE>;
defm V_CMP_T_U32 : VOPC_I32 <vopc<0xc7, 0xcf>, "v_cmp_t_u32">;


defm V_CMPX_F_U32 : VOPCX_I32 <vopc<0xd0, 0xd8>, "v_cmpx_f_u32">;
defm V_CMPX_LT_U32 : VOPCX_I32 <vopc<0xd1, 0xd9>, "v_cmpx_lt_u32", "v_cmpx_gt_u32">;
defm V_CMPX_EQ_U32 : VOPCX_I32 <vopc<0xd2, 0xda>, "v_cmpx_eq_u32">;
defm V_CMPX_LE_U32 : VOPCX_I32 <vopc<0xd3, 0xdb>, "v_cmpx_le_u32", "v_cmpx_le_u32">;
defm V_CMPX_GT_U32 : VOPCX_I32 <vopc<0xd4, 0xdc>, "v_cmpx_gt_u32">;
defm V_CMPX_NE_U32 : VOPCX_I32 <vopc<0xd5, 0xdd>, "v_cmpx_ne_u32">;
defm V_CMPX_GE_U32 : VOPCX_I32 <vopc<0xd6, 0xde>, "v_cmpx_ge_u32">;
defm V_CMPX_T_U32 : VOPCX_I32 <vopc<0xd7, 0xdf>, "v_cmpx_t_u32">;


defm V_CMP_F_U64 : VOPC_I64 <vopc<0xe0, 0xe8>, "v_cmp_f_u64">;
defm V_CMP_LT_U64 : VOPC_I64 <vopc<0xe1, 0xe9>, "v_cmp_lt_u64", COND_ULT, "v_cmp_gt_u64">;
defm V_CMP_EQ_U64 : VOPC_I64 <vopc<0xe2, 0xea>, "v_cmp_eq_u64", COND_EQ>;
defm V_CMP_LE_U64 : VOPC_I64 <vopc<0xe3, 0xeb>, "v_cmp_le_u64", COND_ULE, "v_cmp_ge_u64">;
defm V_CMP_GT_U64 : VOPC_I64 <vopc<0xe4, 0xec>, "v_cmp_gt_u64", COND_UGT>;
defm V_CMP_NE_U64 : VOPC_I64 <vopc<0xe5, 0xed>, "v_cmp_ne_u64", COND_NE>;
defm V_CMP_GE_U64 : VOPC_I64 <vopc<0xe6, 0xee>, "v_cmp_ge_u64", COND_UGE>;
defm V_CMP_T_U64 : VOPC_I64 <vopc<0xe7, 0xef>, "v_cmp_t_u64">;

defm V_CMPX_F_U64 : VOPCX_I64 <vopc<0xf0, 0xf8>, "v_cmpx_f_u64">;
defm V_CMPX_LT_U64 : VOPCX_I64 <vopc<0xf1, 0xf9>, "v_cmpx_lt_u64", "v_cmpx_gt_u64">;
defm V_CMPX_EQ_U64 : VOPCX_I64 <vopc<0xf2, 0xfa>, "v_cmpx_eq_u64">;
defm V_CMPX_LE_U64 : VOPCX_I64 <vopc<0xf3, 0xfb>, "v_cmpx_le_u64", "v_cmpx_ge_u64">;
defm V_CMPX_GT_U64 : VOPCX_I64 <vopc<0xf4, 0xfc>, "v_cmpx_gt_u64">;
defm V_CMPX_NE_U64 : VOPCX_I64 <vopc<0xf5, 0xfd>, "v_cmpx_ne_u64">;
defm V_CMPX_GE_U64 : VOPCX_I64 <vopc<0xf6, 0xfe>, "v_cmpx_ge_u64">;
defm V_CMPX_T_U64 : VOPCX_I64 <vopc<0xf7, 0xff>, "v_cmpx_t_u64">;

} // End isCompare = 1, isCommutable = 1

defm V_CMP_CLASS_F32 : VOPC_CLASS_F32 <vopc<0x88, 0x10>, "v_cmp_class_f32">;
defm V_CMPX_CLASS_F32 : VOPCX_CLASS_F32 <vopc<0x98, 0x11>, "v_cmpx_class_f32">;
defm V_CMP_CLASS_F64 : VOPC_CLASS_F64 <vopc<0xa8, 0x12>, "v_cmp_class_f64">;
defm V_CMPX_CLASS_F64 : VOPCX_CLASS_F64 <vopc<0xb8, 0x13>, "v_cmpx_class_f64">;

//===----------------------------------------------------------------------===//
// DS Instructions
//===----------------------------------------------------------------------===//

defm DS_ADD_U32 : DS_1A1D_NORET <0x0, "ds_add_u32", VGPR_32>;
defm DS_SUB_U32 : DS_1A1D_NORET <0x1, "ds_sub_u32", VGPR_32>;
defm DS_RSUB_U32 : DS_1A1D_NORET <0x2, "ds_rsub_u32", VGPR_32>;
defm DS_INC_U32 : DS_1A1D_NORET <0x3, "ds_inc_u32", VGPR_32>;
defm DS_DEC_U32 : DS_1A1D_NORET <0x4, "ds_dec_u32", VGPR_32>;
defm DS_MIN_I32 : DS_1A1D_NORET <0x5, "ds_min_i32", VGPR_32>;
defm DS_MAX_I32 : DS_1A1D_NORET <0x6, "ds_max_i32", VGPR_32>;
defm DS_MIN_U32 : DS_1A1D_NORET <0x7, "ds_min_u32", VGPR_32>;
defm DS_MAX_U32 : DS_1A1D_NORET <0x8, "ds_max_u32", VGPR_32>;
defm DS_AND_B32 : DS_1A1D_NORET <0x9, "ds_and_b32", VGPR_32>;
defm DS_OR_B32 : DS_1A1D_NORET <0xa, "ds_or_b32", VGPR_32>;
defm DS_XOR_B32 : DS_1A1D_NORET <0xb, "ds_xor_b32", VGPR_32>;
defm DS_MSKOR_B32 : DS_1A2D_NORET <0xc, "ds_mskor_b32", VGPR_32>;
let mayLoad = 0 in {
defm DS_WRITE_B32 : DS_1A1D_NORET <0xd, "ds_write_b32", VGPR_32>;
defm DS_WRITE2_B32 : DS_1A1D_Off8_NORET <0xe, "ds_write2_b32", VGPR_32>;
defm DS_WRITE2ST64_B32 : DS_1A1D_Off8_NORET <0xf, "ds_write2st64_b32", VGPR_32>;
}
defm DS_CMPST_B32 : DS_1A2D_NORET <0x10, "ds_cmpst_b32", VGPR_32>;
defm DS_CMPST_F32 : DS_1A2D_NORET <0x11, "ds_cmpst_f32", VGPR_32>;
defm DS_MIN_F32 : DS_1A2D_NORET <0x12, "ds_min_f32", VGPR_32>;
defm DS_MAX_F32 : DS_1A2D_NORET <0x13, "ds_max_f32", VGPR_32>;

defm DS_GWS_INIT : DS_1A_GDS <0x19, "ds_gws_init">;
defm DS_GWS_SEMA_V : DS_1A_GDS <0x1a, "ds_gws_sema_v">;
defm DS_GWS_SEMA_BR : DS_1A_GDS <0x1b, "ds_gws_sema_br">;
defm DS_GWS_SEMA_P : DS_1A_GDS <0x1c, "ds_gws_sema_p">;
defm DS_GWS_BARRIER : DS_1A_GDS <0x1d, "ds_gws_barrier">;
let mayLoad = 0 in {
defm DS_WRITE_B8 : DS_1A1D_NORET <0x1e, "ds_write_b8", VGPR_32>;
defm DS_WRITE_B16 : DS_1A1D_NORET <0x1f, "ds_write_b16", VGPR_32>;
}
defm DS_ADD_RTN_U32 : DS_1A1D_RET <0x20, "ds_add_rtn_u32", VGPR_32, "ds_add_u32">;
defm DS_SUB_RTN_U32 : DS_1A1D_RET <0x21, "ds_sub_rtn_u32", VGPR_32, "ds_sub_u32">;
defm DS_RSUB_RTN_U32 : DS_1A1D_RET <0x22, "ds_rsub_rtn_u32", VGPR_32, "ds_rsub_u32">;
defm DS_INC_RTN_U32 : DS_1A1D_RET <0x23, "ds_inc_rtn_u32", VGPR_32, "ds_inc_u32">;
defm DS_DEC_RTN_U32 : DS_1A1D_RET <0x24, "ds_dec_rtn_u32", VGPR_32, "ds_dec_u32">;
defm DS_MIN_RTN_I32 : DS_1A1D_RET <0x25, "ds_min_rtn_i32", VGPR_32, "ds_min_i32">;
defm DS_MAX_RTN_I32 : DS_1A1D_RET <0x26, "ds_max_rtn_i32", VGPR_32, "ds_max_i32">;
defm DS_MIN_RTN_U32 : DS_1A1D_RET <0x27, "ds_min_rtn_u32", VGPR_32, "ds_min_u32">;
defm DS_MAX_RTN_U32 : DS_1A1D_RET <0x28, "ds_max_rtn_u32", VGPR_32, "ds_max_u32">;
defm DS_AND_RTN_B32 : DS_1A1D_RET <0x29, "ds_and_rtn_b32", VGPR_32, "ds_and_b32">;
defm DS_OR_RTN_B32 : DS_1A1D_RET <0x2a, "ds_or_rtn_b32", VGPR_32, "ds_or_b32">;
defm DS_XOR_RTN_B32 : DS_1A1D_RET <0x2b, "ds_xor_rtn_b32", VGPR_32, "ds_xor_b32">;
defm DS_MSKOR_RTN_B32 : DS_1A2D_RET <0x2c, "ds_mskor_rtn_b32", VGPR_32, "ds_mskor_b32">;
defm DS_WRXCHG_RTN_B32 : DS_1A1D_RET <0x2d, "ds_wrxchg_rtn_b32", VGPR_32>;
defm DS_WRXCHG2_RTN_B32 : DS_1A2D_RET <
  0x2e, "ds_wrxchg2_rtn_b32", VReg_64, "", VGPR_32
>;
defm DS_WRXCHG2ST64_RTN_B32 : DS_1A2D_RET <
  0x2f, "ds_wrxchg2st64_rtn_b32", VReg_64, "", VGPR_32
>;
defm DS_CMPST_RTN_B32 : DS_1A2D_RET <0x30, "ds_cmpst_rtn_b32", VGPR_32, "ds_cmpst_b32">;
defm DS_CMPST_RTN_F32 : DS_1A2D_RET <0x31, "ds_cmpst_rtn_f32", VGPR_32, "ds_cmpst_f32">;
defm DS_MIN_RTN_F32 : DS_1A2D_RET <0x32, "ds_min_rtn_f32", VGPR_32, "ds_min_f32">;
defm DS_MAX_RTN_F32 : DS_1A2D_RET <0x33, "ds_max_rtn_f32", VGPR_32, "ds_max_f32">;
defm DS_SWIZZLE_B32 : DS_1A_RET <0x35, "ds_swizzle_b32", VGPR_32>;
let mayStore = 0 in {
defm DS_READ_B32 : DS_1A_RET <0x36, "ds_read_b32", VGPR_32>;
defm DS_READ2_B32 : DS_1A_Off8_RET <0x37, "ds_read2_b32", VReg_64>;
defm DS_READ2ST64_B32 : DS_1A_Off8_RET <0x38, "ds_read2st64_b32", VReg_64>;
defm DS_READ_I8 : DS_1A_RET <0x39, "ds_read_i8", VGPR_32>;
defm DS_READ_U8 : DS_1A_RET <0x3a, "ds_read_u8", VGPR_32>;
defm DS_READ_I16 : DS_1A_RET <0x3b, "ds_read_i16", VGPR_32>;
defm DS_READ_U16 : DS_1A_RET <0x3c, "ds_read_u16", VGPR_32>;
}
defm DS_CONSUME : DS_0A_RET <0x3d, "ds_consume">;
defm DS_APPEND : DS_0A_RET <0x3e, "ds_append">;
defm DS_ORDERED_COUNT : DS_1A_RET_GDS <0x3f, "ds_ordered_count">;
defm DS_ADD_U64 : DS_1A1D_NORET <0x40, "ds_add_u64", VReg_64>;
defm DS_SUB_U64 : DS_1A1D_NORET <0x41, "ds_sub_u64", VReg_64>;
defm DS_RSUB_U64 : DS_1A1D_NORET <0x42, "ds_rsub_u64", VReg_64>;
defm DS_INC_U64 : DS_1A1D_NORET <0x43, "ds_inc_u64", VReg_64>;
defm DS_DEC_U64 : DS_1A1D_NORET <0x44, "ds_dec_u64", VReg_64>;
defm DS_MIN_I64 : DS_1A1D_NORET <0x45, "ds_min_i64", VReg_64>;
defm DS_MAX_I64 : DS_1A1D_NORET <0x46, "ds_max_i64", VReg_64>;
defm DS_MIN_U64 : DS_1A1D_NORET <0x47, "ds_min_u64", VReg_64>;
defm DS_MAX_U64 : DS_1A1D_NORET <0x48, "ds_max_u64", VReg_64>;
defm DS_AND_B64 : DS_1A1D_NORET <0x49, "ds_and_b64", VReg_64>;
defm DS_OR_B64 : DS_1A1D_NORET <0x4a, "ds_or_b64", VReg_64>;
defm DS_XOR_B64 : DS_1A1D_NORET <0x4b, "ds_xor_b64", VReg_64>;
defm DS_MSKOR_B64 : DS_1A2D_NORET <0x4c, "ds_mskor_b64", VReg_64>;
let mayLoad = 0 in {
defm DS_WRITE_B64 : DS_1A1D_NORET <0x4d, "ds_write_b64", VReg_64>;
defm DS_WRITE2_B64 : DS_1A1D_Off8_NORET <0x4E, "ds_write2_b64", VReg_64>;
defm DS_WRITE2ST64_B64 : DS_1A1D_Off8_NORET <0x4f, "ds_write2st64_b64", VReg_64>;
}
defm DS_CMPST_B64 : DS_1A2D_NORET <0x50, "ds_cmpst_b64", VReg_64>;
defm DS_CMPST_F64 : DS_1A2D_NORET <0x51, "ds_cmpst_f64", VReg_64>;
defm DS_MIN_F64 : DS_1A1D_NORET <0x52, "ds_min_f64", VReg_64>;
defm DS_MAX_F64 : DS_1A1D_NORET <0x53, "ds_max_f64", VReg_64>;

defm DS_ADD_RTN_U64 : DS_1A1D_RET <0x60, "ds_add_rtn_u64", VReg_64, "ds_add_u64">;
defm DS_SUB_RTN_U64 : DS_1A1D_RET <0x61, "ds_sub_rtn_u64", VReg_64, "ds_sub_u64">;
defm DS_RSUB_RTN_U64 : DS_1A1D_RET <0x62, "ds_rsub_rtn_u64", VReg_64, "ds_rsub_u64">;
defm DS_INC_RTN_U64 : DS_1A1D_RET <0x63, "ds_inc_rtn_u64", VReg_64, "ds_inc_u64">;
defm DS_DEC_RTN_U64 : DS_1A1D_RET <0x64, "ds_dec_rtn_u64", VReg_64, "ds_dec_u64">;
defm DS_MIN_RTN_I64 : DS_1A1D_RET <0x65, "ds_min_rtn_i64", VReg_64, "ds_min_i64">;
defm DS_MAX_RTN_I64 : DS_1A1D_RET <0x66, "ds_max_rtn_i64", VReg_64, "ds_max_i64">;
defm DS_MIN_RTN_U64 : DS_1A1D_RET <0x67, "ds_min_rtn_u64", VReg_64, "ds_min_u64">;
defm DS_MAX_RTN_U64 : DS_1A1D_RET <0x68, "ds_max_rtn_u64", VReg_64, "ds_max_u64">;
defm DS_AND_RTN_B64 : DS_1A1D_RET <0x69, "ds_and_rtn_b64", VReg_64, "ds_and_b64">;
defm DS_OR_RTN_B64 : DS_1A1D_RET <0x6a, "ds_or_rtn_b64", VReg_64, "ds_or_b64">;
defm DS_XOR_RTN_B64 : DS_1A1D_RET <0x6b, "ds_xor_rtn_b64", VReg_64, "ds_xor_b64">;
defm DS_MSKOR_RTN_B64 : DS_1A2D_RET <0x6c, "ds_mskor_rtn_b64", VReg_64, "ds_mskor_b64">;
defm DS_WRXCHG_RTN_B64 : DS_1A1D_RET <0x6d, "ds_wrxchg_rtn_b64", VReg_64, "ds_wrxchg_b64">;
defm DS_WRXCHG2_RTN_B64 : DS_1A2D_RET <0x6e, "ds_wrxchg2_rtn_b64", VReg_128, "ds_wrxchg2_b64", VReg_64>;
defm DS_WRXCHG2ST64_RTN_B64 : DS_1A2D_RET <0x6f, "ds_wrxchg2st64_rtn_b64", VReg_128, "ds_wrxchg2st64_b64", VReg_64>;
defm DS_CMPST_RTN_B64 : DS_1A2D_RET <0x70, "ds_cmpst_rtn_b64", VReg_64, "ds_cmpst_b64">;
defm DS_CMPST_RTN_F64 : DS_1A2D_RET <0x71, "ds_cmpst_rtn_f64", VReg_64, "ds_cmpst_f64">;
defm DS_MIN_RTN_F64 : DS_1A1D_RET <0x72, "ds_min_rtn_f64", VReg_64, "ds_min_f64">;
defm DS_MAX_RTN_F64 : DS_1A1D_RET <0x73, "ds_max_rtn_f64", VReg_64, "ds_max_f64">;

let mayStore = 0 in {
defm DS_READ_B64 : DS_1A_RET <0x76, "ds_read_b64", VReg_64>;
defm DS_READ2_B64 : DS_1A_Off8_RET <0x77, "ds_read2_b64", VReg_128>;
defm DS_READ2ST64_B64 : DS_1A_Off8_RET <0x78, "ds_read2st64_b64", VReg_128>;
}

defm DS_ADD_SRC2_U32 : DS_1A <0x80, "ds_add_src2_u32">;
defm DS_SUB_SRC2_U32 : DS_1A <0x81, "ds_sub_src2_u32">;
defm DS_RSUB_SRC2_U32 : DS_1A <0x82, "ds_rsub_src2_u32">;
defm DS_INC_SRC2_U32 : DS_1A <0x83, "ds_inc_src2_u32">;
defm DS_DEC_SRC2_U32 : DS_1A <0x84, "ds_dec_src2_u32">;
defm DS_MIN_SRC2_I32 : DS_1A <0x85, "ds_min_src2_i32">;
defm DS_MAX_SRC2_I32 : DS_1A <0x86, "ds_max_src2_i32">;
defm DS_MIN_SRC2_U32 : DS_1A <0x87, "ds_min_src2_u32">;
defm DS_MAX_SRC2_U32 : DS_1A <0x88, "ds_max_src2_u32">;
defm DS_AND_SRC2_B32 : DS_1A <0x89, "ds_and_src_b32">;
defm DS_OR_SRC2_B32 : DS_1A <0x8a, "ds_or_src2_b32">;
defm DS_XOR_SRC2_B32 : DS_1A <0x8b, "ds_xor_src2_b32">;
defm DS_WRITE_SRC2_B32 : DS_1A <0x8c, "ds_write_src2_b32">;

defm DS_MIN_SRC2_F32 : DS_1A <0x92, "ds_min_src2_f32">;
defm DS_MAX_SRC2_F32 : DS_1A <0x93, "ds_max_src2_f32">;

defm DS_ADD_SRC2_U64 : DS_1A <0xc0, "ds_add_src2_u64">;
defm DS_SUB_SRC2_U64 : DS_1A <0xc1, "ds_sub_src2_u64">;
defm DS_RSUB_SRC2_U64 : DS_1A <0xc2, "ds_rsub_src2_u64">;
defm DS_INC_SRC2_U64 : DS_1A <0xc3, "ds_inc_src2_u64">;
defm DS_DEC_SRC2_U64 : DS_1A <0xc4, "ds_dec_src2_u64">;
defm DS_MIN_SRC2_I64 : DS_1A <0xc5, "ds_min_src2_i64">;
defm DS_MAX_SRC2_I64 : DS_1A <0xc6, "ds_max_src2_i64">;
defm DS_MIN_SRC2_U64 : DS_1A <0xc7, "ds_min_src2_u64">;
defm DS_MAX_SRC2_U64 : DS_1A <0xc8, "ds_max_src2_u64">;
defm DS_AND_SRC2_B64 : DS_1A <0xc9, "ds_and_src2_b64">;
defm DS_OR_SRC2_B64 : DS_1A <0xca, "ds_or_src2_b64">;
defm DS_XOR_SRC2_B64 : DS_1A <0xcb, "ds_xor_src2_b64">;
defm DS_WRITE_SRC2_B64 : DS_1A <0xcc, "ds_write_src2_b64">;

defm DS_MIN_SRC2_F64 : DS_1A <0xd2, "ds_min_src2_f64">;
defm DS_MAX_SRC2_F64 : DS_1A <0xd3, "ds_max_src2_f64">;

//===----------------------------------------------------------------------===//
// MUBUF Instructions
//===----------------------------------------------------------------------===//

defm BUFFER_LOAD_FORMAT_X : MUBUF_Load_Helper <
  mubuf<0x00>, "buffer_load_format_x", VGPR_32
>;
defm BUFFER_LOAD_FORMAT_XY : MUBUF_Load_Helper <
  mubuf<0x01>, "buffer_load_format_xy", VReg_64
>;
defm BUFFER_LOAD_FORMAT_XYZ : MUBUF_Load_Helper <
  mubuf<0x02>, "buffer_load_format_xyz", VReg_96
>;
defm BUFFER_LOAD_FORMAT_XYZW : MUBUF_Load_Helper <
  mubuf<0x03>, "buffer_load_format_xyzw", VReg_128
>;
defm BUFFER_STORE_FORMAT_X : MUBUF_Store_Helper <
  mubuf<0x04>, "buffer_store_format_x", VGPR_32
>;
defm BUFFER_STORE_FORMAT_XY : MUBUF_Store_Helper <
  mubuf<0x05>, "buffer_store_format_xy", VReg_64
>;
defm BUFFER_STORE_FORMAT_XYZ : MUBUF_Store_Helper <
  mubuf<0x06>, "buffer_store_format_xyz", VReg_96
>;
defm BUFFER_STORE_FORMAT_XYZW : MUBUF_Store_Helper <
  mubuf<0x07>, "buffer_store_format_xyzw", VReg_128
>;
defm BUFFER_LOAD_UBYTE : MUBUF_Load_Helper <
  mubuf<0x08, 0x10>, "buffer_load_ubyte", VGPR_32, i32, az_extloadi8_global
>;
defm BUFFER_LOAD_SBYTE : MUBUF_Load_Helper <
  mubuf<0x09, 0x11>, "buffer_load_sbyte", VGPR_32, i32, sextloadi8_global
>;
defm BUFFER_LOAD_USHORT : MUBUF_Load_Helper <
  mubuf<0x0a, 0x12>, "buffer_load_ushort", VGPR_32, i32, az_extloadi16_global
>;
defm BUFFER_LOAD_SSHORT : MUBUF_Load_Helper <
  mubuf<0x0b, 0x13>, "buffer_load_sshort", VGPR_32, i32, sextloadi16_global
>;
defm BUFFER_LOAD_DWORD : MUBUF_Load_Helper <
  mubuf<0x0c, 0x14>, "buffer_load_dword", VGPR_32, i32, global_load
>;
defm BUFFER_LOAD_DWORDX2 : MUBUF_Load_Helper <
  mubuf<0x0d, 0x15>, "buffer_load_dwordx2", VReg_64, v2i32, global_load
>;
defm BUFFER_LOAD_DWORDX4 : MUBUF_Load_Helper <
  mubuf<0x0e, 0x17>, "buffer_load_dwordx4", VReg_128, v4i32, global_load
>;

defm BUFFER_STORE_BYTE : MUBUF_Store_Helper <
  mubuf<0x18>, "buffer_store_byte", VGPR_32, i32, truncstorei8_global
>;

defm BUFFER_STORE_SHORT : MUBUF_Store_Helper <
  mubuf<0x1a>, "buffer_store_short", VGPR_32, i32, truncstorei16_global
>;

defm BUFFER_STORE_DWORD : MUBUF_Store_Helper <
  mubuf<0x1c>, "buffer_store_dword", VGPR_32, i32, global_store
>;

defm BUFFER_STORE_DWORDX2 : MUBUF_Store_Helper <
  mubuf<0x1d>, "buffer_store_dwordx2", VReg_64, v2i32, global_store
>;

defm BUFFER_STORE_DWORDX4 : MUBUF_Store_Helper <
  mubuf<0x1e, 0x1f>, "buffer_store_dwordx4", VReg_128, v4i32, global_store
>;

defm BUFFER_ATOMIC_SWAP : MUBUF_Atomic <
  mubuf<0x30, 0x40>, "buffer_atomic_swap", VGPR_32, i32, atomic_swap_global
>;
//def BUFFER_ATOMIC_CMPSWAP : MUBUF_ <mubuf<0x31, 0x41>, "buffer_atomic_cmpswap", []>;
defm BUFFER_ATOMIC_ADD : MUBUF_Atomic <
  mubuf<0x32, 0x42>, "buffer_atomic_add", VGPR_32, i32, atomic_add_global
>;
defm BUFFER_ATOMIC_SUB : MUBUF_Atomic <
  mubuf<0x33, 0x43>, "buffer_atomic_sub", VGPR_32, i32, atomic_sub_global
>;
//def BUFFER_ATOMIC_RSUB : MUBUF_ <mubuf<0x34>, "buffer_atomic_rsub", []>; // isn't on CI & VI
defm BUFFER_ATOMIC_SMIN : MUBUF_Atomic <
  mubuf<0x35, 0x44>, "buffer_atomic_smin", VGPR_32, i32, atomic_min_global
>;
defm BUFFER_ATOMIC_UMIN : MUBUF_Atomic <
  mubuf<0x36, 0x45>, "buffer_atomic_umin", VGPR_32, i32, atomic_umin_global
>;
defm BUFFER_ATOMIC_SMAX : MUBUF_Atomic <
  mubuf<0x37, 0x46>, "buffer_atomic_smax", VGPR_32, i32, atomic_max_global
>;
defm BUFFER_ATOMIC_UMAX : MUBUF_Atomic <
  mubuf<0x38, 0x47>, "buffer_atomic_umax", VGPR_32, i32, atomic_umax_global
>;
defm BUFFER_ATOMIC_AND : MUBUF_Atomic <
  mubuf<0x39, 0x48>, "buffer_atomic_and", VGPR_32, i32, atomic_and_global
>;
defm BUFFER_ATOMIC_OR : MUBUF_Atomic <
  mubuf<0x3a, 0x49>, "buffer_atomic_or", VGPR_32, i32, atomic_or_global
>;
defm BUFFER_ATOMIC_XOR : MUBUF_Atomic <
  mubuf<0x3b, 0x4a>, "buffer_atomic_xor", VGPR_32, i32, atomic_xor_global
>;
//def BUFFER_ATOMIC_INC : MUBUF_ <mubuf<0x3c, 0x4b>, "buffer_atomic_inc", []>;
//def BUFFER_ATOMIC_DEC : MUBUF_ <mubuf<0x3d, 0x4c>, "buffer_atomic_dec", []>;
//def BUFFER_ATOMIC_FCMPSWAP : MUBUF_ <mubuf<0x3e>, "buffer_atomic_fcmpswap", []>; // isn't on VI
//def BUFFER_ATOMIC_FMIN : MUBUF_ <mubuf<0x3f>, "buffer_atomic_fmin", []>; // isn't on VI
//def BUFFER_ATOMIC_FMAX : MUBUF_ <mubuf<0x40>, "buffer_atomic_fmax", []>; // isn't on VI
//def BUFFER_ATOMIC_SWAP_X2 : MUBUF_X2 <mubuf<0x50, 0x60>, "buffer_atomic_swap_x2", []>;
//def BUFFER_ATOMIC_CMPSWAP_X2 : MUBUF_X2 <mubuf<0x51, 0x61>, "buffer_atomic_cmpswap_x2", []>;
//def BUFFER_ATOMIC_ADD_X2 : MUBUF_X2 <mubuf<0x52, 0x62>, "buffer_atomic_add_x2", []>;
//def BUFFER_ATOMIC_SUB_X2 : MUBUF_X2 <mubuf<0x53, 0x63>, "buffer_atomic_sub_x2", []>;
//def BUFFER_ATOMIC_RSUB_X2 : MUBUF_X2 <mubuf<0x54>, "buffer_atomic_rsub_x2", []>; // isn't on CI & VI
//def BUFFER_ATOMIC_SMIN_X2 : MUBUF_X2 <mubuf<0x55, 0x64>, "buffer_atomic_smin_x2", []>;
//def BUFFER_ATOMIC_UMIN_X2 : MUBUF_X2 <mubuf<0x56, 0x65>, "buffer_atomic_umin_x2", []>;
//def BUFFER_ATOMIC_SMAX_X2 : MUBUF_X2 <mubuf<0x57, 0x66>, "buffer_atomic_smax_x2", []>;
//def BUFFER_ATOMIC_UMAX_X2 : MUBUF_X2 <mubuf<0x58, 0x67>, "buffer_atomic_umax_x2", []>;
//def BUFFER_ATOMIC_AND_X2 : MUBUF_X2 <mubuf<0x59, 0x68>, "buffer_atomic_and_x2", []>;
//def BUFFER_ATOMIC_OR_X2 : MUBUF_X2 <mubuf<0x5a, 0x69>, "buffer_atomic_or_x2", []>;
//def BUFFER_ATOMIC_XOR_X2 : MUBUF_X2 <mubuf<0x5b, 0x6a>, "buffer_atomic_xor_x2", []>;
//def BUFFER_ATOMIC_INC_X2 : MUBUF_X2 <mubuf<0x5c, 0x6b>, "buffer_atomic_inc_x2", []>;
//def BUFFER_ATOMIC_DEC_X2 : MUBUF_X2 <mubuf<0x5d, 0x6c>, "buffer_atomic_dec_x2", []>;
//def BUFFER_ATOMIC_FCMPSWAP_X2 : MUBUF_X2 <mubuf<0x5e>, "buffer_atomic_fcmpswap_x2", []>; // isn't on VI
//def BUFFER_ATOMIC_FMIN_X2 : MUBUF_X2 <mubuf<0x5f>, "buffer_atomic_fmin_x2", []>; // isn't on VI
//def BUFFER_ATOMIC_FMAX_X2 : MUBUF_X2 <mubuf<0x60>, "buffer_atomic_fmax_x2", []>; // isn't on VI

let SubtargetPredicate = isSI in {
defm BUFFER_WBINVL1_SC : MUBUF_Invalidate <mubuf<0x70>, "buffer_wbinvl1_sc", int_amdgcn_buffer_wbinvl1_sc>; // isn't on CI & VI
}

defm BUFFER_WBINVL1 : MUBUF_Invalidate <mubuf<0x71, 0x3e>, "buffer_wbinvl1", int_amdgcn_buffer_wbinvl1>;

//===----------------------------------------------------------------------===//
// MTBUF Instructions
//===----------------------------------------------------------------------===//

//def TBUFFER_LOAD_FORMAT_X : MTBUF_ <0x00000000, "tbuffer_load_format_x", []>;
//def TBUFFER_LOAD_FORMAT_XY : MTBUF_ <0x00000001, "tbuffer_load_format_xy", []>;
//def TBUFFER_LOAD_FORMAT_XYZ : MTBUF_ <0x00000002, "tbuffer_load_format_xyz", []>;
defm TBUFFER_LOAD_FORMAT_XYZW : MTBUF_Load_Helper <0x00000003, "tbuffer_load_format_xyzw", VReg_128>;
defm TBUFFER_STORE_FORMAT_X : MTBUF_Store_Helper <0x00000004, "tbuffer_store_format_x", VGPR_32>;
defm TBUFFER_STORE_FORMAT_XY : MTBUF_Store_Helper <0x00000005, "tbuffer_store_format_xy", VReg_64>;
defm TBUFFER_STORE_FORMAT_XYZ : MTBUF_Store_Helper <0x00000006, "tbuffer_store_format_xyz", VReg_128>;
defm TBUFFER_STORE_FORMAT_XYZW : MTBUF_Store_Helper <0x00000007, "tbuffer_store_format_xyzw", VReg_128>;

//===----------------------------------------------------------------------===//
// MIMG Instructions
//===----------------------------------------------------------------------===//

defm IMAGE_LOAD : MIMG_NoSampler <0x00000000, "image_load">;
defm IMAGE_LOAD_MIP : MIMG_NoSampler <0x00000001, "image_load_mip">;
//def IMAGE_LOAD_PCK : MIMG_NoPattern_ <"image_load_pck", 0x00000002>;
//def IMAGE_LOAD_PCK_SGN : MIMG_NoPattern_ <"image_load_pck_sgn", 0x00000003>;
//def IMAGE_LOAD_MIP_PCK : MIMG_NoPattern_ <"image_load_mip_pck", 0x00000004>;
//def IMAGE_LOAD_MIP_PCK_SGN : MIMG_NoPattern_ <"image_load_mip_pck_sgn", 0x00000005>;
//def IMAGE_STORE : MIMG_NoPattern_ <"image_store", 0x00000008>;
//def IMAGE_STORE_MIP : MIMG_NoPattern_ <"image_store_mip", 0x00000009>;
//def IMAGE_STORE_PCK : MIMG_NoPattern_ <"image_store_pck", 0x0000000a>;
//def IMAGE_STORE_MIP_PCK : MIMG_NoPattern_ <"image_store_mip_pck", 0x0000000b>;
defm IMAGE_GET_RESINFO : MIMG_NoSampler <0x0000000e, "image_get_resinfo">;
//def IMAGE_ATOMIC_SWAP : MIMG_NoPattern_ <"image_atomic_swap", 0x0000000f>;
//def IMAGE_ATOMIC_CMPSWAP : MIMG_NoPattern_ <"image_atomic_cmpswap", 0x00000010>;
//def IMAGE_ATOMIC_ADD : MIMG_NoPattern_ <"image_atomic_add", 0x00000011>;
//def IMAGE_ATOMIC_SUB : MIMG_NoPattern_ <"image_atomic_sub", 0x00000012>;
//def IMAGE_ATOMIC_RSUB : MIMG_NoPattern_ <"image_atomic_rsub", 0x00000013>;
//def IMAGE_ATOMIC_SMIN : MIMG_NoPattern_ <"image_atomic_smin", 0x00000014>;
//def IMAGE_ATOMIC_UMIN : MIMG_NoPattern_ <"image_atomic_umin", 0x00000015>;
//def IMAGE_ATOMIC_SMAX : MIMG_NoPattern_ <"image_atomic_smax", 0x00000016>;
//def IMAGE_ATOMIC_UMAX : MIMG_NoPattern_ <"image_atomic_umax", 0x00000017>;
//def IMAGE_ATOMIC_AND : MIMG_NoPattern_ <"image_atomic_and", 0x00000018>;
//def IMAGE_ATOMIC_OR : MIMG_NoPattern_ <"image_atomic_or", 0x00000019>;
//def IMAGE_ATOMIC_XOR : MIMG_NoPattern_ <"image_atomic_xor", 0x0000001a>;
//def IMAGE_ATOMIC_INC : MIMG_NoPattern_ <"image_atomic_inc", 0x0000001b>;
//def IMAGE_ATOMIC_DEC : MIMG_NoPattern_ <"image_atomic_dec", 0x0000001c>;
//def IMAGE_ATOMIC_FCMPSWAP : MIMG_NoPattern_ <"image_atomic_fcmpswap", 0x0000001d>;
//def IMAGE_ATOMIC_FMIN : MIMG_NoPattern_ <"image_atomic_fmin", 0x0000001e>;
//def IMAGE_ATOMIC_FMAX : MIMG_NoPattern_ <"image_atomic_fmax", 0x0000001f>;
defm IMAGE_SAMPLE           : MIMG_Sampler_WQM <0x00000020, "image_sample">;
defm IMAGE_SAMPLE_CL        : MIMG_Sampler_WQM <0x00000021, "image_sample_cl">;
defm IMAGE_SAMPLE_D         : MIMG_Sampler <0x00000022, "image_sample_d">;
defm IMAGE_SAMPLE_D_CL      : MIMG_Sampler <0x00000023, "image_sample_d_cl">;
defm IMAGE_SAMPLE_L         : MIMG_Sampler <0x00000024, "image_sample_l">;
defm IMAGE_SAMPLE_B         : MIMG_Sampler_WQM <0x00000025, "image_sample_b">;
defm IMAGE_SAMPLE_B_CL      : MIMG_Sampler_WQM <0x00000026, "image_sample_b_cl">;
defm IMAGE_SAMPLE_LZ        : MIMG_Sampler <0x00000027, "image_sample_lz">;
defm IMAGE_SAMPLE_C         : MIMG_Sampler_WQM <0x00000028, "image_sample_c">;
defm IMAGE_SAMPLE_C_CL      : MIMG_Sampler_WQM <0x00000029, "image_sample_c_cl">;
defm IMAGE_SAMPLE_C_D       : MIMG_Sampler <0x0000002a, "image_sample_c_d">;
defm IMAGE_SAMPLE_C_D_CL    : MIMG_Sampler <0x0000002b, "image_sample_c_d_cl">;
defm IMAGE_SAMPLE_C_L       : MIMG_Sampler <0x0000002c, "image_sample_c_l">;
defm IMAGE_SAMPLE_C_B       : MIMG_Sampler_WQM <0x0000002d, "image_sample_c_b">;
defm IMAGE_SAMPLE_C_B_CL    : MIMG_Sampler_WQM <0x0000002e, "image_sample_c_b_cl">;
defm IMAGE_SAMPLE_C_LZ      : MIMG_Sampler <0x0000002f, "image_sample_c_lz">;
defm IMAGE_SAMPLE_O         : MIMG_Sampler_WQM <0x00000030, "image_sample_o">;
defm IMAGE_SAMPLE_CL_O      : MIMG_Sampler_WQM <0x00000031, "image_sample_cl_o">;
defm IMAGE_SAMPLE_D_O       : MIMG_Sampler <0x00000032, "image_sample_d_o">;
defm IMAGE_SAMPLE_D_CL_O    : MIMG_Sampler <0x00000033, "image_sample_d_cl_o">;
defm IMAGE_SAMPLE_L_O       : MIMG_Sampler <0x00000034, "image_sample_l_o">;
defm IMAGE_SAMPLE_B_O       : MIMG_Sampler_WQM <0x00000035, "image_sample_b_o">;
defm IMAGE_SAMPLE_B_CL_O    : MIMG_Sampler_WQM <0x00000036, "image_sample_b_cl_o">;
defm IMAGE_SAMPLE_LZ_O      : MIMG_Sampler <0x00000037, "image_sample_lz_o">;
defm IMAGE_SAMPLE_C_O       : MIMG_Sampler_WQM <0x00000038, "image_sample_c_o">;
defm IMAGE_SAMPLE_C_CL_O    : MIMG_Sampler_WQM <0x00000039, "image_sample_c_cl_o">;
defm IMAGE_SAMPLE_C_D_O     : MIMG_Sampler <0x0000003a, "image_sample_c_d_o">;
defm IMAGE_SAMPLE_C_D_CL_O  : MIMG_Sampler <0x0000003b, "image_sample_c_d_cl_o">;
defm IMAGE_SAMPLE_C_L_O     : MIMG_Sampler <0x0000003c, "image_sample_c_l_o">;
defm IMAGE_SAMPLE_C_B_O     : MIMG_Sampler_WQM <0x0000003d, "image_sample_c_b_o">;
defm IMAGE_SAMPLE_C_B_CL_O  : MIMG_Sampler_WQM <0x0000003e, "image_sample_c_b_cl_o">;
defm IMAGE_SAMPLE_C_LZ_O    : MIMG_Sampler <0x0000003f, "image_sample_c_lz_o">;
defm IMAGE_GATHER4          : MIMG_Gather_WQM <0x00000040, "image_gather4">;
defm IMAGE_GATHER4_CL       : MIMG_Gather_WQM <0x00000041, "image_gather4_cl">;
defm IMAGE_GATHER4_L        : MIMG_Gather <0x00000044, "image_gather4_l">;
defm IMAGE_GATHER4_B        : MIMG_Gather_WQM <0x00000045, "image_gather4_b">;
defm IMAGE_GATHER4_B_CL     : MIMG_Gather_WQM <0x00000046, "image_gather4_b_cl">;
defm IMAGE_GATHER4_LZ       : MIMG_Gather <0x00000047, "image_gather4_lz">;
defm IMAGE_GATHER4_C        : MIMG_Gather_WQM <0x00000048, "image_gather4_c">;
defm IMAGE_GATHER4_C_CL     : MIMG_Gather_WQM <0x00000049, "image_gather4_c_cl">;
defm IMAGE_GATHER4_C_L      : MIMG_Gather <0x0000004c, "image_gather4_c_l">;
defm IMAGE_GATHER4_C_B      : MIMG_Gather_WQM <0x0000004d, "image_gather4_c_b">;
defm IMAGE_GATHER4_C_B_CL   : MIMG_Gather_WQM <0x0000004e, "image_gather4_c_b_cl">;
defm IMAGE_GATHER4_C_LZ     : MIMG_Gather <0x0000004f, "image_gather4_c_lz">;
defm IMAGE_GATHER4_O        : MIMG_Gather_WQM <0x00000050, "image_gather4_o">;
defm IMAGE_GATHER4_CL_O     : MIMG_Gather_WQM <0x00000051, "image_gather4_cl_o">;
defm IMAGE_GATHER4_L_O      : MIMG_Gather <0x00000054, "image_gather4_l_o">;
defm IMAGE_GATHER4_B_O      : MIMG_Gather_WQM <0x00000055, "image_gather4_b_o">;
defm IMAGE_GATHER4_B_CL_O   : MIMG_Gather <0x00000056, "image_gather4_b_cl_o">;
defm IMAGE_GATHER4_LZ_O     : MIMG_Gather <0x00000057, "image_gather4_lz_o">;
defm IMAGE_GATHER4_C_O      : MIMG_Gather_WQM <0x00000058, "image_gather4_c_o">;
defm IMAGE_GATHER4_C_CL_O   : MIMG_Gather_WQM <0x00000059, "image_gather4_c_cl_o">;
defm IMAGE_GATHER4_C_L_O    : MIMG_Gather <0x0000005c, "image_gather4_c_l_o">;
defm IMAGE_GATHER4_C_B_O    : MIMG_Gather_WQM <0x0000005d, "image_gather4_c_b_o">;
defm IMAGE_GATHER4_C_B_CL_O : MIMG_Gather_WQM <0x0000005e, "image_gather4_c_b_cl_o">;
defm IMAGE_GATHER4_C_LZ_O   : MIMG_Gather <0x0000005f, "image_gather4_c_lz_o">;
defm IMAGE_GET_LOD          : MIMG_Sampler_WQM <0x00000060, "image_get_lod">;
defm IMAGE_SAMPLE_CD        : MIMG_Sampler <0x00000068, "image_sample_cd">;
defm IMAGE_SAMPLE_CD_CL     : MIMG_Sampler <0x00000069, "image_sample_cd_cl">;
defm IMAGE_SAMPLE_C_CD      : MIMG_Sampler <0x0000006a, "image_sample_c_cd">;
defm IMAGE_SAMPLE_C_CD_CL   : MIMG_Sampler <0x0000006b, "image_sample_c_cd_cl">;
defm IMAGE_SAMPLE_CD_O      : MIMG_Sampler <0x0000006c, "image_sample_cd_o">;
defm IMAGE_SAMPLE_CD_CL_O   : MIMG_Sampler <0x0000006d, "image_sample_cd_cl_o">;
defm IMAGE_SAMPLE_C_CD_O    : MIMG_Sampler <0x0000006e, "image_sample_c_cd_o">;
defm IMAGE_SAMPLE_C_CD_CL_O : MIMG_Sampler <0x0000006f, "image_sample_c_cd_cl_o">;
//def IMAGE_RSRC256 : MIMG_NoPattern_RSRC256 <"image_rsrc256", 0x0000007e>;
//def IMAGE_SAMPLER : MIMG_NoPattern_ <"image_sampler", 0x0000007f>;

//===----------------------------------------------------------------------===//
// VOP1 Instructions
//===----------------------------------------------------------------------===//

let vdst = 0, src0 = 0, VOPAsmPrefer32Bit = 1 in {
defm V_NOP : VOP1Inst <vop1<0x0>, "v_nop", VOP_NONE>;
}

let isMoveImm = 1, isReMaterializable = 1, isAsCheapAsAMove = 1 in {
defm V_MOV_B32 : VOP1Inst <vop1<0x1>, "v_mov_b32", VOP_I32_I32>;
} // End isMoveImm = 1

let Uses = [EXEC] in {

// FIXME: Specify SchedRW for READFIRSTLANE_B32

def V_READFIRSTLANE_B32 : VOP1 <
  0x00000002,
  (outs SReg_32:$vdst),
  (ins VGPR_32:$src0),
  "v_readfirstlane_b32 $vdst, $src0",
  []
>;

}

let SchedRW = [WriteQuarterRate32] in {

defm V_CVT_I32_F64 : VOP1Inst <vop1<0x3>, "v_cvt_i32_f64",
  VOP_I32_F64, fp_to_sint
>;
defm V_CVT_F64_I32 : VOP1Inst <vop1<0x4>, "v_cvt_f64_i32",
  VOP_F64_I32, sint_to_fp
>;
defm V_CVT_F32_I32 : VOP1Inst <vop1<0x5>, "v_cvt_f32_i32",
  VOP_F32_I32, sint_to_fp
>;
defm V_CVT_F32_U32 : VOP1Inst <vop1<0x6>, "v_cvt_f32_u32",
  VOP_F32_I32, uint_to_fp
>;
defm V_CVT_U32_F32 : VOP1Inst <vop1<0x7>, "v_cvt_u32_f32",
  VOP_I32_F32, fp_to_uint
>;
defm V_CVT_I32_F32 : VOP1Inst <vop1<0x8>, "v_cvt_i32_f32",
  VOP_I32_F32, fp_to_sint
>;
defm V_CVT_F16_F32 : VOP1Inst <vop1<0xa>, "v_cvt_f16_f32",
  VOP_I32_F32, fp_to_f16
>;
defm V_CVT_F32_F16 : VOP1Inst <vop1<0xb>, "v_cvt_f32_f16",
  VOP_F32_I32, f16_to_fp
>;
defm V_CVT_RPI_I32_F32 : VOP1Inst <vop1<0xc>, "v_cvt_rpi_i32_f32",
  VOP_I32_F32, cvt_rpi_i32_f32>;
defm V_CVT_FLR_I32_F32 : VOP1Inst <vop1<0xd>, "v_cvt_flr_i32_f32",
  VOP_I32_F32, cvt_flr_i32_f32>;
defm V_CVT_OFF_F32_I4 : VOP1Inst  <vop1<0x0e>, "v_cvt_off_f32_i4", VOP_F32_I32>;
defm V_CVT_F32_F64 : VOP1Inst <vop1<0xf>, "v_cvt_f32_f64",
  VOP_F32_F64, fround
>;
defm V_CVT_F64_F32 : VOP1Inst <vop1<0x10>, "v_cvt_f64_f32",
  VOP_F64_F32, fextend
>;
defm V_CVT_F32_UBYTE0 : VOP1Inst <vop1<0x11>, "v_cvt_f32_ubyte0",
  VOP_F32_I32, AMDGPUcvt_f32_ubyte0
>;
defm V_CVT_F32_UBYTE1 : VOP1Inst <vop1<0x12>, "v_cvt_f32_ubyte1",
  VOP_F32_I32, AMDGPUcvt_f32_ubyte1
>;
defm V_CVT_F32_UBYTE2 : VOP1Inst <vop1<0x13>, "v_cvt_f32_ubyte2",
  VOP_F32_I32, AMDGPUcvt_f32_ubyte2
>;
defm V_CVT_F32_UBYTE3 : VOP1Inst <vop1<0x14>, "v_cvt_f32_ubyte3",
  VOP_F32_I32, AMDGPUcvt_f32_ubyte3
>;
defm V_CVT_U32_F64 : VOP1Inst <vop1<0x15>, "v_cvt_u32_f64",
  VOP_I32_F64, fp_to_uint
>;
defm V_CVT_F64_U32 : VOP1Inst <vop1<0x16>, "v_cvt_f64_u32",
  VOP_F64_I32, uint_to_fp
>;

} // let SchedRW = [WriteQuarterRate32]

defm V_FRACT_F32 : VOP1Inst <vop1<0x20, 0x1b>, "v_fract_f32",
  VOP_F32_F32, AMDGPUfract
>;
defm V_TRUNC_F32 : VOP1Inst <vop1<0x21, 0x1c>, "v_trunc_f32",
  VOP_F32_F32, ftrunc
>;
defm V_CEIL_F32 : VOP1Inst <vop1<0x22, 0x1d>, "v_ceil_f32",
  VOP_F32_F32, fceil
>;
defm V_RNDNE_F32 : VOP1Inst <vop1<0x23, 0x1e>, "v_rndne_f32",
  VOP_F32_F32, frint
>;
defm V_FLOOR_F32 : VOP1Inst <vop1<0x24, 0x1f>, "v_floor_f32",
  VOP_F32_F32, ffloor
>;
defm V_EXP_F32 : VOP1Inst <vop1<0x25, 0x20>, "v_exp_f32",
  VOP_F32_F32, fexp2
>;

let SchedRW = [WriteQuarterRate32] in {

defm V_LOG_F32 : VOP1Inst <vop1<0x27, 0x21>, "v_log_f32",
  VOP_F32_F32, flog2
>;
defm V_RCP_F32 : VOP1Inst <vop1<0x2a, 0x22>, "v_rcp_f32",
  VOP_F32_F32, AMDGPUrcp
>;
defm V_RCP_IFLAG_F32 : VOP1Inst <vop1<0x2b, 0x23>, "v_rcp_iflag_f32",
  VOP_F32_F32
>;
defm V_RSQ_F32 : VOP1Inst <vop1<0x2e, 0x24>, "v_rsq_f32",
  VOP_F32_F32, AMDGPUrsq
>;

} //let SchedRW = [WriteQuarterRate32]

let SchedRW = [WriteDouble] in {

defm V_RCP_F64 : VOP1Inst <vop1<0x2f, 0x25>, "v_rcp_f64",
  VOP_F64_F64, AMDGPUrcp
>;
defm V_RSQ_F64 : VOP1Inst <vop1<0x31, 0x26>, "v_rsq_f64",
  VOP_F64_F64, AMDGPUrsq
>;

} // let SchedRW = [WriteDouble];

defm V_SQRT_F32 : VOP1Inst <vop1<0x33, 0x27>, "v_sqrt_f32",
  VOP_F32_F32, fsqrt
>;

let SchedRW = [WriteDouble] in {

defm V_SQRT_F64 : VOP1Inst <vop1<0x34, 0x28>, "v_sqrt_f64",
  VOP_F64_F64, fsqrt
>;

} // End SchedRW = [WriteDouble]

let SchedRW = [WriteQuarterRate32] in {

defm V_SIN_F32 : VOP1Inst <vop1<0x35, 0x29>, "v_sin_f32",
  VOP_F32_F32, AMDGPUsin
>;
defm V_COS_F32 : VOP1Inst <vop1<0x36, 0x2a>, "v_cos_f32",
  VOP_F32_F32, AMDGPUcos
>;

} // End SchedRW = [WriteQuarterRate32]

defm V_NOT_B32 : VOP1Inst <vop1<0x37, 0x2b>, "v_not_b32", VOP_I32_I32>;
defm V_BFREV_B32 : VOP1Inst <vop1<0x38, 0x2c>, "v_bfrev_b32", VOP_I32_I32>;
defm V_FFBH_U32 : VOP1Inst <vop1<0x39, 0x2d>, "v_ffbh_u32", VOP_I32_I32>;
defm V_FFBL_B32 : VOP1Inst <vop1<0x3a, 0x2e>, "v_ffbl_b32", VOP_I32_I32>;
defm V_FFBH_I32 : VOP1Inst <vop1<0x3b, 0x2f>, "v_ffbh_i32", VOP_I32_I32>;
defm V_FREXP_EXP_I32_F64 : VOP1Inst <vop1<0x3c,0x30>, "v_frexp_exp_i32_f64",
  VOP_I32_F64
>;

let SchedRW = [WriteDoubleAdd] in {
defm V_FREXP_MANT_F64 : VOP1Inst <vop1<0x3d, 0x31>, "v_frexp_mant_f64",
  VOP_F64_F64
>;

defm V_FRACT_F64 : VOP1Inst <vop1<0x3e, 0x32>, "v_fract_f64",
  VOP_F64_F64
>;
} // End SchedRW = [WriteDoubleAdd]


defm V_FREXP_EXP_I32_F32 : VOP1Inst <vop1<0x3f, 0x33>, "v_frexp_exp_i32_f32",
  VOP_I32_F32
>;
defm V_FREXP_MANT_F32 : VOP1Inst <vop1<0x40, 0x34>, "v_frexp_mant_f32",
  VOP_F32_F32
>;
let vdst = 0, src0 = 0, VOPAsmPrefer32Bit = 1 in {
defm V_CLREXCP : VOP1Inst <vop1<0x41,0x35>, "v_clrexcp", VOP_NONE>;
}

let Uses = [M0, EXEC] in {
defm V_MOVRELD_B32 : VOP1Inst <vop1<0x42, 0x36>, "v_movreld_b32", VOP_I32_I32>;
defm V_MOVRELS_B32 : VOP1Inst <vop1<0x43, 0x37>, "v_movrels_b32", VOP_I32_I32>;
defm V_MOVRELSD_B32 : VOP1Inst <vop1<0x44, 0x38>, "v_movrelsd_b32", VOP_I32_I32>;
} // End Uses = [M0, EXEC]

// These instruction only exist on SI and CI
let SubtargetPredicate = isSICI in {

let SchedRW = [WriteQuarterRate32] in {

defm V_MOV_FED_B32 : VOP1InstSI <vop1<0x9>, "v_mov_fed_b32", VOP_I32_I32>;
defm V_LOG_CLAMP_F32 : VOP1InstSI <vop1<0x26>, "v_log_clamp_f32", VOP_F32_F32>;
defm V_RCP_CLAMP_F32 : VOP1InstSI <vop1<0x28>, "v_rcp_clamp_f32", VOP_F32_F32>;
defm V_RCP_LEGACY_F32 : VOP1InstSI <vop1<0x29>, "v_rcp_legacy_f32", VOP_F32_F32>;
defm V_RSQ_CLAMP_F32 : VOP1InstSI <vop1<0x2c>, "v_rsq_clamp_f32",
  VOP_F32_F32, AMDGPUrsq_clamped
>;
defm V_RSQ_LEGACY_F32 : VOP1InstSI <vop1<0x2d>, "v_rsq_legacy_f32",
  VOP_F32_F32, AMDGPUrsq_legacy
>;

} // End SchedRW = [WriteQuarterRate32]

let SchedRW = [WriteDouble] in {

defm V_RCP_CLAMP_F64 : VOP1InstSI <vop1<0x30>, "v_rcp_clamp_f64", VOP_F64_F64>;
defm V_RSQ_CLAMP_F64 : VOP1InstSI <vop1<0x32>, "v_rsq_clamp_f64",
  VOP_F64_F64, AMDGPUrsq_clamped
>;

} // End SchedRW = [WriteDouble]

} // End SubtargetPredicate = isSICI

//===----------------------------------------------------------------------===//
// VINTRP Instructions
//===----------------------------------------------------------------------===//

let Uses = [M0, EXEC] in {

// FIXME: Specify SchedRW for VINTRP insturctions.

multiclass V_INTERP_P1_F32_m : VINTRP_m <
  0x00000000,
  (outs VGPR_32:$dst),
  (ins VGPR_32:$i, i32imm:$attr_chan, i32imm:$attr),
  "v_interp_p1_f32 $dst, $i, $attr_chan, $attr, [m0]",
  [(set f32:$dst, (AMDGPUinterp_p1 i32:$i, (i32 imm:$attr_chan),
                                           (i32 imm:$attr)))]
>;

let OtherPredicates = [has32BankLDS] in {

defm V_INTERP_P1_F32 : V_INTERP_P1_F32_m;

} // End OtherPredicates = [has32BankLDS]

let OtherPredicates = [has16BankLDS], Constraints = "@earlyclobber $dst" in {

defm V_INTERP_P1_F32_16bank : V_INTERP_P1_F32_m;

} // End OtherPredicates = [has32BankLDS], Constraints = "@earlyclobber $dst"

let DisableEncoding = "$src0", Constraints = "$src0 = $dst" in {

defm V_INTERP_P2_F32 : VINTRP_m <
  0x00000001,
  (outs VGPR_32:$dst),
  (ins VGPR_32:$src0, VGPR_32:$j, i32imm:$attr_chan, i32imm:$attr),
  "v_interp_p2_f32 $dst, [$src0], $j, $attr_chan, $attr, [m0]",
  [(set f32:$dst, (AMDGPUinterp_p2 f32:$src0, i32:$j, (i32 imm:$attr_chan),
                                                     (i32 imm:$attr)))]>;

} // End DisableEncoding = "$src0", Constraints = "$src0 = $dst"

defm V_INTERP_MOV_F32 : VINTRP_m <
  0x00000002,
  (outs VGPR_32:$dst),
  (ins InterpSlot:$src0, i32imm:$attr_chan, i32imm:$attr),
  "v_interp_mov_f32 $dst, $src0, $attr_chan, $attr, [m0]",
  [(set f32:$dst, (AMDGPUinterp_mov (i32 imm:$src0), (i32 imm:$attr_chan),
                                    (i32 imm:$attr)))]>;

} // End Uses = [M0, EXEC]

//===----------------------------------------------------------------------===//
// VOP2 Instructions
//===----------------------------------------------------------------------===//

multiclass V_CNDMASK <vop2 op, string name> {
  defm _e32 : VOP2_m <op, name, VOP_CNDMASK, [], name>;

  defm _e64  : VOP3_m <
      op, VOP_CNDMASK.Outs, VOP_CNDMASK.Ins64,
      name#!cast<string>(VOP_CNDMASK.Asm64), [], name, 3>;
}

defm V_CNDMASK_B32 : V_CNDMASK<vop2<0x0>, "v_cndmask_b32">;

let isCommutable = 1 in {
defm V_ADD_F32 : VOP2Inst <vop2<0x3, 0x1>, "v_add_f32",
  VOP_F32_F32_F32, fadd
>;

defm V_SUB_F32 : VOP2Inst <vop2<0x4, 0x2>, "v_sub_f32", VOP_F32_F32_F32, fsub>;
defm V_SUBREV_F32 : VOP2Inst <vop2<0x5, 0x3>, "v_subrev_f32",
  VOP_F32_F32_F32, null_frag, "v_sub_f32"
>;
} // End isCommutable = 1

let isCommutable = 1 in {

defm V_MUL_LEGACY_F32 : VOP2Inst <vop2<0x7, 0x4>, "v_mul_legacy_f32",
  VOP_F32_F32_F32, int_AMDGPU_mul
>;

defm V_MUL_F32 : VOP2Inst <vop2<0x8, 0x5>, "v_mul_f32",
  VOP_F32_F32_F32, fmul
>;

defm V_MUL_I32_I24 : VOP2Inst <vop2<0x9, 0x6>, "v_mul_i32_i24",
  VOP_I32_I32_I32, AMDGPUmul_i24
>;

defm V_MUL_HI_I32_I24 : VOP2Inst <vop2<0xa,0x7>, "v_mul_hi_i32_i24",
  VOP_I32_I32_I32
>;

defm V_MUL_U32_U24 : VOP2Inst <vop2<0xb, 0x8>, "v_mul_u32_u24",
  VOP_I32_I32_I32, AMDGPUmul_u24
>;

defm V_MUL_HI_U32_U24 : VOP2Inst <vop2<0xc,0x9>, "v_mul_hi_u32_u24",
 VOP_I32_I32_I32
>;

defm V_MIN_F32 : VOP2Inst <vop2<0xf, 0xa>, "v_min_f32", VOP_F32_F32_F32,
  fminnum>;
defm V_MAX_F32 : VOP2Inst <vop2<0x10, 0xb>, "v_max_f32", VOP_F32_F32_F32,
  fmaxnum>;
defm V_MIN_I32 : VOP2Inst <vop2<0x11, 0xc>, "v_min_i32", VOP_I32_I32_I32>;
defm V_MAX_I32 : VOP2Inst <vop2<0x12, 0xd>, "v_max_i32", VOP_I32_I32_I32>;
defm V_MIN_U32 : VOP2Inst <vop2<0x13, 0xe>, "v_min_u32", VOP_I32_I32_I32>;
defm V_MAX_U32 : VOP2Inst <vop2<0x14, 0xf>, "v_max_u32", VOP_I32_I32_I32>;

defm V_LSHRREV_B32 : VOP2Inst <
  vop2<0x16, 0x10>, "v_lshrrev_b32", VOP_I32_I32_I32, null_frag,
    "v_lshr_b32"
>;

defm V_ASHRREV_I32 : VOP2Inst <
  vop2<0x18, 0x11>, "v_ashrrev_i32", VOP_I32_I32_I32, null_frag,
    "v_ashr_i32"
>;

defm V_LSHLREV_B32 : VOP2Inst <
  vop2<0x1a, 0x12>, "v_lshlrev_b32", VOP_I32_I32_I32, null_frag,
    "v_lshl_b32"
>;

defm V_AND_B32 : VOP2Inst <vop2<0x1b, 0x13>, "v_and_b32", VOP_I32_I32_I32>;
defm V_OR_B32 : VOP2Inst <vop2<0x1c, 0x14>, "v_or_b32", VOP_I32_I32_I32>;
defm V_XOR_B32 : VOP2Inst <vop2<0x1d, 0x15>, "v_xor_b32", VOP_I32_I32_I32>;

let Constraints = "$dst = $src2", DisableEncoding="$src2",
    isConvertibleToThreeAddress = 1 in {
defm V_MAC_F32 : VOP2Inst <vop2<0x1f, 0x16>, "v_mac_f32", VOP_MAC>;
}
} // End isCommutable = 1

defm V_MADMK_F32 : VOP2MADK <vop2<0x20, 0x17>, "v_madmk_f32">;

let isCommutable = 1 in {
defm V_MADAK_F32 : VOP2MADK <vop2<0x21, 0x18>, "v_madak_f32">;
} // End isCommutable = 1

let isCommutable = 1 in {
// No patterns so that the scalar instructions are always selected.
// The scalar versions will be replaced with vector when needed later.

// V_ADD_I32, V_SUB_I32, and V_SUBREV_I32 where renamed to *_U32 in VI,
// but the VI instructions behave the same as the SI versions.
defm V_ADD_I32 : VOP2bInst <vop2<0x25, 0x19>, "v_add_i32",
  VOP2b_I32_I1_I32_I32
>;
defm V_SUB_I32 : VOP2bInst <vop2<0x26, 0x1a>, "v_sub_i32", VOP2b_I32_I1_I32_I32>;

defm V_SUBREV_I32 : VOP2bInst <vop2<0x27, 0x1b>, "v_subrev_i32",
  VOP2b_I32_I1_I32_I32, null_frag, "v_sub_i32"
>;

defm V_ADDC_U32 : VOP2bInst <vop2<0x28, 0x1c>, "v_addc_u32",
  VOP2b_I32_I1_I32_I32_I1
>;
defm V_SUBB_U32 : VOP2bInst <vop2<0x29, 0x1d>, "v_subb_u32",
  VOP2b_I32_I1_I32_I32_I1
>;
defm V_SUBBREV_U32 : VOP2bInst <vop2<0x2a, 0x1e>, "v_subbrev_u32",
  VOP2b_I32_I1_I32_I32_I1, null_frag, "v_subb_u32"
>;

} // End isCommutable = 1

defm V_READLANE_B32 : VOP2SI_3VI_m <
  vop3 <0x001, 0x289>,
  "v_readlane_b32",
  (outs SReg_32:$vdst),
  (ins VGPR_32:$src0, SCSrc_32:$src1),
  "v_readlane_b32 $vdst, $src0, $src1"
>;

defm V_WRITELANE_B32 : VOP2SI_3VI_m <
  vop3 <0x002, 0x28a>,
  "v_writelane_b32",
  (outs VGPR_32:$vdst),
  (ins SReg_32:$src0, SCSrc_32:$src1),
  "v_writelane_b32 $vdst, $src0, $src1"
>;

// These instructions only exist on SI and CI
let SubtargetPredicate = isSICI in {

let isCommutable = 1 in {
defm V_MAC_LEGACY_F32 : VOP2InstSI <vop2<0x6>, "v_mac_legacy_f32",
  VOP_F32_F32_F32
>;
} // End isCommutable = 1

defm V_MIN_LEGACY_F32 : VOP2InstSI <vop2<0xd>, "v_min_legacy_f32",
  VOP_F32_F32_F32, AMDGPUfmin_legacy
>;
defm V_MAX_LEGACY_F32 : VOP2InstSI <vop2<0xe>, "v_max_legacy_f32",
  VOP_F32_F32_F32, AMDGPUfmax_legacy
>;

let isCommutable = 1 in {
defm V_LSHR_B32 : VOP2InstSI <vop2<0x15>, "v_lshr_b32", VOP_I32_I32_I32>;
defm V_ASHR_I32 : VOP2InstSI <vop2<0x17>, "v_ashr_i32", VOP_I32_I32_I32>;
defm V_LSHL_B32 : VOP2InstSI <vop2<0x19>, "v_lshl_b32", VOP_I32_I32_I32>;
} // End isCommutable = 1
} // End let SubtargetPredicate = SICI

defm V_BFM_B32 : VOP2_VI3_Inst <vop23<0x1e, 0x293>, "v_bfm_b32",
  VOP_I32_I32_I32
>;
defm V_BCNT_U32_B32 : VOP2_VI3_Inst <vop23<0x22, 0x28b>, "v_bcnt_u32_b32",
  VOP_I32_I32_I32
>;
defm V_MBCNT_LO_U32_B32 : VOP2_VI3_Inst <vop23<0x23, 0x28c>, "v_mbcnt_lo_u32_b32",
  VOP_I32_I32_I32
>;
defm V_MBCNT_HI_U32_B32 : VOP2_VI3_Inst <vop23<0x24, 0x28d>, "v_mbcnt_hi_u32_b32",
  VOP_I32_I32_I32
>;
defm V_LDEXP_F32 : VOP2_VI3_Inst <vop23<0x2b, 0x288>, "v_ldexp_f32",
  VOP_F32_F32_I32, AMDGPUldexp
>;

defm V_CVT_PKACCUM_U8_F32 : VOP2_VI3_Inst <vop23<0x2c, 0x1f0>, "v_cvt_pkaccum_u8_f32",
  VOP_I32_F32_I32>; // TODO: set "Uses = dst"

defm V_CVT_PKNORM_I16_F32 : VOP2_VI3_Inst <vop23<0x2d, 0x294>, "v_cvt_pknorm_i16_f32",
  VOP_I32_F32_F32
>;
defm V_CVT_PKNORM_U16_F32 : VOP2_VI3_Inst <vop23<0x2e, 0x295>, "v_cvt_pknorm_u16_f32",
  VOP_I32_F32_F32
>;
defm V_CVT_PKRTZ_F16_F32 : VOP2_VI3_Inst <vop23<0x2f, 0x296>, "v_cvt_pkrtz_f16_f32",
  VOP_I32_F32_F32, int_SI_packf16
>;
defm V_CVT_PK_U16_U32 : VOP2_VI3_Inst <vop23<0x30, 0x297>, "v_cvt_pk_u16_u32",
  VOP_I32_I32_I32
>;
defm V_CVT_PK_I16_I32 : VOP2_VI3_Inst <vop23<0x31, 0x298>, "v_cvt_pk_i16_i32",
  VOP_I32_I32_I32
>;

//===----------------------------------------------------------------------===//
// VOP3 Instructions
//===----------------------------------------------------------------------===//

let isCommutable = 1 in {
defm V_MAD_LEGACY_F32 : VOP3Inst <vop3<0x140, 0x1c0>, "v_mad_legacy_f32",
  VOP_F32_F32_F32_F32
>;

defm V_MAD_F32 : VOP3Inst <vop3<0x141, 0x1c1>, "v_mad_f32",
  VOP_F32_F32_F32_F32, fmad
>;

defm V_MAD_I32_I24 : VOP3Inst <vop3<0x142, 0x1c2>, "v_mad_i32_i24",
  VOP_I32_I32_I32_I32, AMDGPUmad_i24
>;
defm V_MAD_U32_U24 : VOP3Inst <vop3<0x143, 0x1c3>, "v_mad_u32_u24",
  VOP_I32_I32_I32_I32, AMDGPUmad_u24
>;
} // End isCommutable = 1

defm V_CUBEID_F32 : VOP3Inst <vop3<0x144, 0x1c4>, "v_cubeid_f32",
  VOP_F32_F32_F32_F32
>;
defm V_CUBESC_F32 : VOP3Inst <vop3<0x145, 0x1c5>, "v_cubesc_f32",
  VOP_F32_F32_F32_F32
>;
defm V_CUBETC_F32 : VOP3Inst <vop3<0x146, 0x1c6>, "v_cubetc_f32",
  VOP_F32_F32_F32_F32
>;
defm V_CUBEMA_F32 : VOP3Inst <vop3<0x147, 0x1c7>, "v_cubema_f32",
  VOP_F32_F32_F32_F32
>;

defm V_BFE_U32 : VOP3Inst <vop3<0x148, 0x1c8>, "v_bfe_u32",
  VOP_I32_I32_I32_I32, AMDGPUbfe_u32
>;
defm V_BFE_I32 : VOP3Inst <vop3<0x149, 0x1c9>, "v_bfe_i32",
  VOP_I32_I32_I32_I32, AMDGPUbfe_i32
>;

defm V_BFI_B32 : VOP3Inst <vop3<0x14a, 0x1ca>, "v_bfi_b32",
  VOP_I32_I32_I32_I32, AMDGPUbfi
>;

let isCommutable = 1 in {
defm V_FMA_F32 : VOP3Inst <vop3<0x14b, 0x1cb>, "v_fma_f32",
  VOP_F32_F32_F32_F32, fma
>;
defm V_FMA_F64 : VOP3Inst <vop3<0x14c, 0x1cc>, "v_fma_f64",
  VOP_F64_F64_F64_F64, fma
>;
} // End isCommutable = 1

//def V_LERP_U8 : VOP3_U8 <0x0000014d, "v_lerp_u8", []>;
defm V_ALIGNBIT_B32 : VOP3Inst <vop3<0x14e, 0x1ce>, "v_alignbit_b32",
  VOP_I32_I32_I32_I32
>;
defm V_ALIGNBYTE_B32 : VOP3Inst <vop3<0x14f, 0x1cf>, "v_alignbyte_b32",
  VOP_I32_I32_I32_I32
>;

defm V_MIN3_F32 : VOP3Inst <vop3<0x151, 0x1d0>, "v_min3_f32",
  VOP_F32_F32_F32_F32, AMDGPUfmin3>;

defm V_MIN3_I32 : VOP3Inst <vop3<0x152, 0x1d1>, "v_min3_i32",
  VOP_I32_I32_I32_I32, AMDGPUsmin3
>;
defm V_MIN3_U32 : VOP3Inst <vop3<0x153, 0x1d2>, "v_min3_u32",
  VOP_I32_I32_I32_I32, AMDGPUumin3
>;
defm V_MAX3_F32 : VOP3Inst <vop3<0x154, 0x1d3>, "v_max3_f32",
  VOP_F32_F32_F32_F32, AMDGPUfmax3
>;
defm V_MAX3_I32 : VOP3Inst <vop3<0x155, 0x1d4>, "v_max3_i32",
  VOP_I32_I32_I32_I32, AMDGPUsmax3
>;
defm V_MAX3_U32 : VOP3Inst <vop3<0x156, 0x1d5>, "v_max3_u32",
  VOP_I32_I32_I32_I32, AMDGPUumax3
>;
defm V_MED3_F32 : VOP3Inst <vop3<0x157, 0x1d6>, "v_med3_f32",
  VOP_F32_F32_F32_F32
>;
defm V_MED3_I32 : VOP3Inst <vop3<0x158, 0x1d7>, "v_med3_i32",
  VOP_I32_I32_I32_I32
>;
defm V_MED3_U32 : VOP3Inst <vop3<0x159, 0x1d8>, "v_med3_u32",
  VOP_I32_I32_I32_I32
>;

//def V_SAD_U8 : VOP3_U8 <0x0000015a, "v_sad_u8", []>;
//def V_SAD_HI_U8 : VOP3_U8 <0x0000015b, "v_sad_hi_u8", []>;
//def V_SAD_U16 : VOP3_U16 <0x0000015c, "v_sad_u16", []>;
defm V_SAD_U32 : VOP3Inst <vop3<0x15d, 0x1dc>, "v_sad_u32",
  VOP_I32_I32_I32_I32
>;
////def V_CVT_PK_U8_F32 : VOP3_U8 <0x0000015e, "v_cvt_pk_u8_f32", []>;
defm V_DIV_FIXUP_F32 : VOP3Inst <
  vop3<0x15f, 0x1de>, "v_div_fixup_f32", VOP_F32_F32_F32_F32, AMDGPUdiv_fixup
>;

let SchedRW = [WriteDoubleAdd] in {

defm V_DIV_FIXUP_F64 : VOP3Inst <
  vop3<0x160, 0x1df>, "v_div_fixup_f64", VOP_F64_F64_F64_F64, AMDGPUdiv_fixup
>;

} // End SchedRW = [WriteDouble]

let SchedRW = [WriteDoubleAdd] in {
let isCommutable = 1 in {

defm V_ADD_F64 : VOP3Inst <vop3<0x164, 0x280>, "v_add_f64",
  VOP_F64_F64_F64, fadd
>;
defm V_MUL_F64 : VOP3Inst <vop3<0x165, 0x281>, "v_mul_f64",
  VOP_F64_F64_F64, fmul
>;

defm V_MIN_F64 : VOP3Inst <vop3<0x166, 0x282>, "v_min_f64",
  VOP_F64_F64_F64, fminnum
>;
defm V_MAX_F64 : VOP3Inst <vop3<0x167, 0x283>, "v_max_f64",
  VOP_F64_F64_F64, fmaxnum
>;

} // isCommutable = 1

defm V_LDEXP_F64 : VOP3Inst <vop3<0x168, 0x284>, "v_ldexp_f64",
  VOP_F64_F64_I32, AMDGPUldexp
>;

} // let SchedRW = [WriteDoubleAdd]

let isCommutable = 1, SchedRW = [WriteQuarterRate32] in {

defm V_MUL_LO_U32 : VOP3Inst <vop3<0x169, 0x285>, "v_mul_lo_u32",
  VOP_I32_I32_I32
>;
defm V_MUL_HI_U32 : VOP3Inst <vop3<0x16a, 0x286>, "v_mul_hi_u32",
  VOP_I32_I32_I32
>;

defm V_MUL_LO_I32 : VOP3Inst <vop3<0x16b, 0x285>, "v_mul_lo_i32",
  VOP_I32_I32_I32
>;
defm V_MUL_HI_I32 : VOP3Inst <vop3<0x16c, 0x287>, "v_mul_hi_i32",
  VOP_I32_I32_I32
>;

} // isCommutable = 1, SchedRW = [WriteQuarterRate32]

let SchedRW = [WriteFloatFMA, WriteSALU] in {
defm V_DIV_SCALE_F32 : VOP3bInst <vop3<0x16d, 0x1e0>, "v_div_scale_f32",
  VOP3b_F32_I1_F32_F32_F32
>;
}

let SchedRW = [WriteDouble, WriteSALU] in {
// Double precision division pre-scale.
defm V_DIV_SCALE_F64 : VOP3bInst <vop3<0x16e, 0x1e1>, "v_div_scale_f64",
  VOP3b_F64_I1_F64_F64_F64
>;
} // let SchedRW = [WriteDouble]

let isCommutable = 1, Uses = [VCC, EXEC] in {

let SchedRW = [WriteFloatFMA] in {
// v_div_fmas_f32:
//   result = src0 * src1 + src2
//   if (vcc)
//     result *= 2^32
//
defm V_DIV_FMAS_F32 : VOP3_VCC_Inst <vop3<0x16f, 0x1e2>, "v_div_fmas_f32",
  VOP_F32_F32_F32_F32, AMDGPUdiv_fmas
>;
}

let SchedRW = [WriteDouble] in {
// v_div_fmas_f64:
//   result = src0 * src1 + src2
//   if (vcc)
//     result *= 2^64
//
defm V_DIV_FMAS_F64 : VOP3_VCC_Inst <vop3<0x170, 0x1e3>, "v_div_fmas_f64",
  VOP_F64_F64_F64_F64, AMDGPUdiv_fmas
>;

} // End SchedRW = [WriteDouble]
} // End isCommutable = 1, Uses = [VCC, EXEC]

//def V_MSAD_U8 : VOP3_U8 <0x00000171, "v_msad_u8", []>;
//def V_QSAD_U8 : VOP3_U8 <0x00000172, "v_qsad_u8", []>;
//def V_MQSAD_U8 : VOP3_U8 <0x00000173, "v_mqsad_u8", []>;

let SchedRW = [WriteDouble] in {
defm V_TRIG_PREOP_F64 : VOP3Inst <
  vop3<0x174, 0x292>, "v_trig_preop_f64", VOP_F64_F64_I32, AMDGPUtrig_preop
>;

} // let SchedRW = [WriteDouble]

// These instructions only exist on SI and CI
let SubtargetPredicate = isSICI in {

defm V_LSHL_B64 : VOP3Inst <vop3<0x161>, "v_lshl_b64", VOP_I64_I64_I32>;
defm V_LSHR_B64 : VOP3Inst <vop3<0x162>, "v_lshr_b64", VOP_I64_I64_I32>;
defm V_ASHR_I64 : VOP3Inst <vop3<0x163>, "v_ashr_i64", VOP_I64_I64_I32>;

defm V_MULLIT_F32 : VOP3Inst <vop3<0x150>, "v_mullit_f32",
  VOP_F32_F32_F32_F32>;

} // End SubtargetPredicate = isSICI

let SubtargetPredicate = isVI in {

defm V_LSHLREV_B64 : VOP3Inst <vop3<0, 0x28f>, "v_lshlrev_b64",
  VOP_I64_I32_I64
>;
defm V_LSHRREV_B64 : VOP3Inst <vop3<0, 0x290>, "v_lshrrev_b64",
  VOP_I64_I32_I64
>;
defm V_ASHRREV_I64 : VOP3Inst <vop3<0, 0x291>, "v_ashrrev_i64",
  VOP_I64_I32_I64
>;

} // End SubtargetPredicate = isVI

//===----------------------------------------------------------------------===//
// Pseudo Instructions
//===----------------------------------------------------------------------===//
let isCodeGenOnly = 1, isPseudo = 1 in {

// For use in patterns
def V_CNDMASK_B64_PSEUDO : VOP3Common <(outs VReg_64:$dst),
  (ins VSrc_64:$src0, VSrc_64:$src1, SSrc_64:$src2), "", []
>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, Uses = [EXEC] in {
// 64-bit vector move instruction.  This is mainly used by the SIFoldOperands
// pass to enable folding of inline immediates.
def V_MOV_B64_PSEUDO : InstSI <(outs VReg_64:$dst), (ins VSrc_64:$src0), "", []>;
} // end let hasSideEffects = 0, mayLoad = 0, mayStore = 0

let hasSideEffects = 1, SALU = 1 in {
def SGPR_USE : InstSI <(outs),(ins), "", []>;
}

// SI pseudo instructions. These are used by the CFG structurizer pass
// and should be lowered to ISA instructions prior to codegen.

let mayLoad = 1, mayStore = 1, hasSideEffects = 1 in {
let Uses = [EXEC], Defs = [EXEC] in {

let isBranch = 1, isTerminator = 1 in {

def SI_IF: InstSI <
  (outs SReg_64:$dst),
  (ins SReg_64:$vcc, brtarget:$target),
  "",
  [(set i64:$dst, (int_SI_if i1:$vcc, bb:$target))]
>;

def SI_ELSE : InstSI <
  (outs SReg_64:$dst),
  (ins SReg_64:$src, brtarget:$target),
  "",
  [(set i64:$dst, (int_SI_else i64:$src, bb:$target))]
> {
  let Constraints = "$src = $dst";
}

def SI_LOOP : InstSI <
  (outs),
  (ins SReg_64:$saved, brtarget:$target),
  "si_loop $saved, $target",
  [(int_SI_loop i64:$saved, bb:$target)]
>;

} // end isBranch = 1, isTerminator = 1

def SI_BREAK : InstSI <
  (outs SReg_64:$dst),
  (ins SReg_64:$src),
  "si_else $dst, $src",
  [(set i64:$dst, (int_SI_break i64:$src))]
>;

def SI_IF_BREAK : InstSI <
  (outs SReg_64:$dst),
  (ins SReg_64:$vcc, SReg_64:$src),
  "si_if_break $dst, $vcc, $src",
  [(set i64:$dst, (int_SI_if_break i1:$vcc, i64:$src))]
>;

def SI_ELSE_BREAK : InstSI <
  (outs SReg_64:$dst),
  (ins SReg_64:$src0, SReg_64:$src1),
  "si_else_break $dst, $src0, $src1",
  [(set i64:$dst, (int_SI_else_break i64:$src0, i64:$src1))]
>;

def SI_END_CF : InstSI <
  (outs),
  (ins SReg_64:$saved),
  "si_end_cf $saved",
  [(int_SI_end_cf i64:$saved)]
>;

} // End Uses = [EXEC], Defs = [EXEC]

let Uses = [EXEC], Defs = [EXEC,VCC] in {
def SI_KILL : InstSI <
  (outs),
  (ins VSrc_32:$src),
  "si_kill $src",
  [(int_AMDGPU_kill f32:$src)]
>;
} // End Uses = [EXEC], Defs = [EXEC,VCC]

} // end mayLoad = 1, mayStore = 1, hasSideEffects = 1

let Uses = [EXEC], Defs = [EXEC,VCC,M0] in {

//defm SI_ : RegisterLoadStore <VGPR_32, FRAMEri, ADDRIndirect>;

let UseNamedOperandTable = 1 in {

def SI_RegisterLoad : InstSI <
  (outs VGPR_32:$dst, SReg_64:$temp),
  (ins FRAMEri32:$addr, i32imm:$chan),
  "", []
> {
  let isRegisterLoad = 1;
  let mayLoad = 1;
}

class SIRegStore<dag outs> : InstSI <
  outs,
  (ins VGPR_32:$val, FRAMEri32:$addr, i32imm:$chan),
  "", []
> {
  let isRegisterStore = 1;
  let mayStore = 1;
}

let usesCustomInserter = 1 in {
def SI_RegisterStorePseudo : SIRegStore<(outs)>;
} // End usesCustomInserter = 1
def SI_RegisterStore : SIRegStore<(outs SReg_64:$temp)>;


} // End UseNamedOperandTable = 1

class SI_INDIRECT_SRC<RegisterClass rc> : InstSI <
  (outs VGPR_32:$dst, SReg_64:$temp),
  (ins rc:$src, VSrc_32:$idx, i32imm:$off),
  "si_indirect_src $dst, $temp, $src, $idx, $off",
  []
>;

class SI_INDIRECT_DST<RegisterClass rc> : InstSI <
  (outs rc:$dst, SReg_64:$temp),
  (ins unknown:$src, VSrc_32:$idx, i32imm:$off, VGPR_32:$val),
  "si_indirect_dst $dst, $temp, $src, $idx, $off, $val",
  []
> {
  let Constraints = "$src = $dst";
}

// TODO: We can support indirect SGPR access.
def SI_INDIRECT_SRC_V1 : SI_INDIRECT_SRC<VGPR_32>;
def SI_INDIRECT_SRC_V2 : SI_INDIRECT_SRC<VReg_64>;
def SI_INDIRECT_SRC_V4 : SI_INDIRECT_SRC<VReg_128>;
def SI_INDIRECT_SRC_V8 : SI_INDIRECT_SRC<VReg_256>;
def SI_INDIRECT_SRC_V16 : SI_INDIRECT_SRC<VReg_512>;

def SI_INDIRECT_DST_V1 : SI_INDIRECT_DST<VGPR_32>;
def SI_INDIRECT_DST_V2 : SI_INDIRECT_DST<VReg_64>;
def SI_INDIRECT_DST_V4 : SI_INDIRECT_DST<VReg_128>;
def SI_INDIRECT_DST_V8 : SI_INDIRECT_DST<VReg_256>;
def SI_INDIRECT_DST_V16 : SI_INDIRECT_DST<VReg_512>;

} // Uses = [EXEC,VCC,M0], Defs = [EXEC,VCC,M0]

multiclass SI_SPILL_SGPR <RegisterClass sgpr_class> {

  let UseNamedOperandTable = 1, Uses = [EXEC] in {
    def _SAVE : InstSI <
      (outs),
      (ins sgpr_class:$src, i32imm:$frame_idx),
      "", []
    > {
      let mayStore = 1;
      let mayLoad = 0;
    }

    def _RESTORE : InstSI <
      (outs sgpr_class:$dst),
      (ins i32imm:$frame_idx),
      "", []
    > {
      let mayStore = 0;
      let mayLoad = 1;
    }
  } // End UseNamedOperandTable = 1
}

// It's unclear whether you can use M0 as the output of v_readlane_b32
// instructions, so use SGPR_32 register class for spills to prevent
// this from happening.
defm SI_SPILL_S32  : SI_SPILL_SGPR <SGPR_32>;
defm SI_SPILL_S64  : SI_SPILL_SGPR <SReg_64>;
defm SI_SPILL_S128 : SI_SPILL_SGPR <SReg_128>;
defm SI_SPILL_S256 : SI_SPILL_SGPR <SReg_256>;
defm SI_SPILL_S512 : SI_SPILL_SGPR <SReg_512>;

multiclass SI_SPILL_VGPR <RegisterClass vgpr_class> {
  let UseNamedOperandTable = 1, VGPRSpill = 1, Uses = [EXEC] in {
    def _SAVE : InstSI <
      (outs),
      (ins vgpr_class:$src, i32imm:$frame_idx, SReg_128:$scratch_rsrc,
           SReg_32:$scratch_offset),
      "", []
    > {
      let mayStore = 1;
      let mayLoad = 0;
    }

    def _RESTORE : InstSI <
      (outs vgpr_class:$dst),
      (ins i32imm:$frame_idx, SReg_128:$scratch_rsrc, SReg_32:$scratch_offset),
      "", []
    > {
      let mayStore = 0;
      let mayLoad = 1;
    }
  } // End UseNamedOperandTable = 1, VGPRSpill = 1
}

defm SI_SPILL_V32  : SI_SPILL_VGPR <VGPR_32>;
defm SI_SPILL_V64  : SI_SPILL_VGPR <VReg_64>;
defm SI_SPILL_V96  : SI_SPILL_VGPR <VReg_96>;
defm SI_SPILL_V128 : SI_SPILL_VGPR <VReg_128>;
defm SI_SPILL_V256 : SI_SPILL_VGPR <VReg_256>;
defm SI_SPILL_V512 : SI_SPILL_VGPR <VReg_512>;

let Defs = [SCC] in {

def SI_CONSTDATA_PTR : InstSI <
  (outs SReg_64:$dst),
  (ins),
  "", [(set SReg_64:$dst, (i64 SIconstdata_ptr))]
> {
  let SALU = 1;
}

} // End Defs = [SCC]

} // end IsCodeGenOnly, isPseudo

} // end SubtargetPredicate = isGCN

let Predicates = [isGCN] in {

def : Pat<
  (int_AMDGPU_cndlt f32:$src0, f32:$src1, f32:$src2),
  (V_CNDMASK_B32_e64 $src2, $src1,
                     (V_CMP_GT_F32_e64 SRCMODS.NONE, 0, SRCMODS.NONE, $src0,
                                       DSTCLAMP.NONE, DSTOMOD.NONE))
>;

def : Pat <
  (int_AMDGPU_kilp),
  (SI_KILL 0xbf800000)
>;

/* int_SI_vs_load_input */
def : Pat<
  (SIload_input v4i32:$tlst, imm:$attr_offset, i32:$buf_idx_vgpr),
  (BUFFER_LOAD_FORMAT_XYZW_IDXEN $buf_idx_vgpr, $tlst, 0, imm:$attr_offset, 0, 0, 0)
>;

/* int_SI_export */
def : Pat <
  (int_SI_export imm:$en, imm:$vm, imm:$done, imm:$tgt, imm:$compr,
                 f32:$src0, f32:$src1, f32:$src2, f32:$src3),
  (EXP imm:$en, imm:$tgt, imm:$compr, imm:$done, imm:$vm,
       $src0, $src1, $src2, $src3)
>;

//===----------------------------------------------------------------------===//
// SMRD Patterns
//===----------------------------------------------------------------------===//

multiclass SMRD_Pattern <string Instr, ValueType vt> {

  // 1. IMM offset
  def : Pat <
    (constant_load (SMRDImm i64:$sbase, i32:$offset)),
    (vt (!cast<SMRD>(Instr#"_IMM") $sbase, $offset))
  >;

  // 2. SGPR offset
  def : Pat <
    (constant_load (SMRDSgpr i64:$sbase, i32:$offset)),
    (vt (!cast<SMRD>(Instr#"_SGPR") $sbase, $offset))
  >;

  def : Pat <
    (constant_load (SMRDImm32 i64:$sbase, i32:$offset)),
    (vt (!cast<SMRD>(Instr#"_IMM_ci") $sbase, $offset))
  > {
    let Predicates = [isCIOnly];
  }
}

defm : SMRD_Pattern <"S_LOAD_DWORD", i32>;
defm : SMRD_Pattern <"S_LOAD_DWORDX2", v2i32>;
defm : SMRD_Pattern <"S_LOAD_DWORDX4", v4i32>;
defm : SMRD_Pattern <"S_LOAD_DWORDX8", v32i8>;
defm : SMRD_Pattern <"S_LOAD_DWORDX8", v8i32>;
defm : SMRD_Pattern <"S_LOAD_DWORDX16", v16i32>;

// 1. Offset as an immediate
def : Pat <
  (SIload_constant v4i32:$sbase, (SMRDBufferImm i32:$offset)),
  (S_BUFFER_LOAD_DWORD_IMM $sbase, $offset)
>;

// 2. Offset loaded in an 32bit SGPR
def : Pat <
  (SIload_constant v4i32:$sbase, (SMRDBufferSgpr i32:$offset)),
  (S_BUFFER_LOAD_DWORD_SGPR $sbase, $offset)
>;

let Predicates = [isCI] in {

def : Pat <
  (SIload_constant v4i32:$sbase, (SMRDBufferImm32 i32:$offset)),
  (S_BUFFER_LOAD_DWORD_IMM_ci $sbase, $offset)
>;

} // End Predicates = [isCI]

//===----------------------------------------------------------------------===//
// SOP1 Patterns
//===----------------------------------------------------------------------===//

def : Pat <
  (i64 (ctpop i64:$src)),
    (i64 (REG_SEQUENCE SReg_64,
     (S_BCNT1_I32_B64 $src), sub0,
     (S_MOV_B32 0), sub1))
>;

//===----------------------------------------------------------------------===//
// SOP2 Patterns
//===----------------------------------------------------------------------===//

// V_ADD_I32_e32/S_ADD_U32 produces carry in VCC/SCC. For the vector
// case, the sgpr-copies pass will fix this to use the vector version.
def : Pat <
  (i32 (addc i32:$src0, i32:$src1)),
  (S_ADD_U32 $src0, $src1)
>;

//===----------------------------------------------------------------------===//
// SOPP Patterns
//===----------------------------------------------------------------------===//

def : Pat <
  (int_AMDGPU_barrier_global),
  (S_BARRIER)
>;

//===----------------------------------------------------------------------===//
// VOP1 Patterns
//===----------------------------------------------------------------------===//

let Predicates = [UnsafeFPMath] in {

//def : RcpPat<V_RCP_F64_e32, f64>;
//defm : RsqPat<V_RSQ_F64_e32, f64>;
//defm : RsqPat<V_RSQ_F32_e32, f32>;

def : RsqPat<V_RSQ_F32_e32, f32>;
def : RsqPat<V_RSQ_F64_e32, f64>;
}

//===----------------------------------------------------------------------===//
// VOP2 Patterns
//===----------------------------------------------------------------------===//

def : Pat <
  (i32 (add (i32 (ctpop i32:$popcnt)), i32:$val)),
  (V_BCNT_U32_B32_e64 $popcnt, $val)
>;

def : Pat <
  (i32 (select i1:$src0, i32:$src1, i32:$src2)),
  (V_CNDMASK_B32_e64 $src2, $src1, $src0)
>;

// Pattern for V_MAC_F32
def : Pat <
  (fmad  (VOP3NoMods0 f32:$src0, i32:$src0_modifiers, i1:$clamp, i32:$omod),
         (VOP3NoMods f32:$src1, i32:$src1_modifiers),
         (VOP3NoMods f32:$src2, i32:$src2_modifiers)),
  (V_MAC_F32_e64 $src0_modifiers, $src0, $src1_modifiers, $src1,
                 $src2_modifiers, $src2, $clamp, $omod)
>;

/********** ======================= **********/
/********** Image sampling patterns **********/
/********** ======================= **********/

// Image + sampler
class SampleRawPattern<SDPatternOperator name, MIMG opcode, ValueType vt> : Pat <
  (name vt:$addr, v8i32:$rsrc, v4i32:$sampler, i32:$dmask, i32:$unorm,
        i32:$r128, i32:$da, i32:$glc, i32:$slc, i32:$tfe, i32:$lwe),
  (opcode (as_i32imm $dmask), (as_i1imm $unorm), (as_i1imm $glc), (as_i1imm $da),
          (as_i1imm $r128), (as_i1imm $tfe), (as_i1imm $lwe), (as_i1imm $slc),
          $addr, $rsrc, $sampler)
>;

multiclass SampleRawPatterns<SDPatternOperator name, string opcode> {
  def : SampleRawPattern<name, !cast<MIMG>(opcode # _V4_V1), i32>;
  def : SampleRawPattern<name, !cast<MIMG>(opcode # _V4_V2), v2i32>;
  def : SampleRawPattern<name, !cast<MIMG>(opcode # _V4_V4), v4i32>;
  def : SampleRawPattern<name, !cast<MIMG>(opcode # _V4_V8), v8i32>;
  def : SampleRawPattern<name, !cast<MIMG>(opcode # _V4_V16), v16i32>;
}

// Image only
class ImagePattern<SDPatternOperator name, MIMG opcode, ValueType vt> : Pat <
  (name vt:$addr, v8i32:$rsrc, i32:$dmask, i32:$unorm,
        i32:$r128, i32:$da, i32:$glc, i32:$slc, i32:$tfe, i32:$lwe),
  (opcode (as_i32imm $dmask), (as_i1imm $unorm), (as_i1imm $glc), (as_i1imm $da),
          (as_i1imm $r128), (as_i1imm $tfe), (as_i1imm $lwe), (as_i1imm $slc),
          $addr, $rsrc)
>;

multiclass ImagePatterns<SDPatternOperator name, string opcode> {
  def : ImagePattern<name, !cast<MIMG>(opcode # _V4_V1), i32>;
  def : ImagePattern<name, !cast<MIMG>(opcode # _V4_V2), v2i32>;
  def : ImagePattern<name, !cast<MIMG>(opcode # _V4_V4), v4i32>;
}

// Basic sample
defm : SampleRawPatterns<int_SI_image_sample,           "IMAGE_SAMPLE">;
defm : SampleRawPatterns<int_SI_image_sample_cl,        "IMAGE_SAMPLE_CL">;
defm : SampleRawPatterns<int_SI_image_sample_d,         "IMAGE_SAMPLE_D">;
defm : SampleRawPatterns<int_SI_image_sample_d_cl,      "IMAGE_SAMPLE_D_CL">;
defm : SampleRawPatterns<int_SI_image_sample_l,         "IMAGE_SAMPLE_L">;
defm : SampleRawPatterns<int_SI_image_sample_b,         "IMAGE_SAMPLE_B">;
defm : SampleRawPatterns<int_SI_image_sample_b_cl,      "IMAGE_SAMPLE_B_CL">;
defm : SampleRawPatterns<int_SI_image_sample_lz,        "IMAGE_SAMPLE_LZ">;
defm : SampleRawPatterns<int_SI_image_sample_cd,        "IMAGE_SAMPLE_CD">;
defm : SampleRawPatterns<int_SI_image_sample_cd_cl,     "IMAGE_SAMPLE_CD_CL">;

// Sample with comparison
defm : SampleRawPatterns<int_SI_image_sample_c,         "IMAGE_SAMPLE_C">;
defm : SampleRawPatterns<int_SI_image_sample_c_cl,      "IMAGE_SAMPLE_C_CL">;
defm : SampleRawPatterns<int_SI_image_sample_c_d,       "IMAGE_SAMPLE_C_D">;
defm : SampleRawPatterns<int_SI_image_sample_c_d_cl,    "IMAGE_SAMPLE_C_D_CL">;
defm : SampleRawPatterns<int_SI_image_sample_c_l,       "IMAGE_SAMPLE_C_L">;
defm : SampleRawPatterns<int_SI_image_sample_c_b,       "IMAGE_SAMPLE_C_B">;
defm : SampleRawPatterns<int_SI_image_sample_c_b_cl,    "IMAGE_SAMPLE_C_B_CL">;
defm : SampleRawPatterns<int_SI_image_sample_c_lz,      "IMAGE_SAMPLE_C_LZ">;
defm : SampleRawPatterns<int_SI_image_sample_c_cd,      "IMAGE_SAMPLE_C_CD">;
defm : SampleRawPatterns<int_SI_image_sample_c_cd_cl,   "IMAGE_SAMPLE_C_CD_CL">;

// Sample with offsets
defm : SampleRawPatterns<int_SI_image_sample_o,         "IMAGE_SAMPLE_O">;
defm : SampleRawPatterns<int_SI_image_sample_cl_o,      "IMAGE_SAMPLE_CL_O">;
defm : SampleRawPatterns<int_SI_image_sample_d_o,       "IMAGE_SAMPLE_D_O">;
defm : SampleRawPatterns<int_SI_image_sample_d_cl_o,    "IMAGE_SAMPLE_D_CL_O">;
defm : SampleRawPatterns<int_SI_image_sample_l_o,       "IMAGE_SAMPLE_L_O">;
defm : SampleRawPatterns<int_SI_image_sample_b_o,       "IMAGE_SAMPLE_B_O">;
defm : SampleRawPatterns<int_SI_image_sample_b_cl_o,    "IMAGE_SAMPLE_B_CL_O">;
defm : SampleRawPatterns<int_SI_image_sample_lz_o,      "IMAGE_SAMPLE_LZ_O">;
defm : SampleRawPatterns<int_SI_image_sample_cd_o,      "IMAGE_SAMPLE_CD_O">;
defm : SampleRawPatterns<int_SI_image_sample_cd_cl_o,   "IMAGE_SAMPLE_CD_CL_O">;

// Sample with comparison and offsets
defm : SampleRawPatterns<int_SI_image_sample_c_o,       "IMAGE_SAMPLE_C_O">;
defm : SampleRawPatterns<int_SI_image_sample_c_cl_o,    "IMAGE_SAMPLE_C_CL_O">;
defm : SampleRawPatterns<int_SI_image_sample_c_d_o,     "IMAGE_SAMPLE_C_D_O">;
defm : SampleRawPatterns<int_SI_image_sample_c_d_cl_o,  "IMAGE_SAMPLE_C_D_CL_O">;
defm : SampleRawPatterns<int_SI_image_sample_c_l_o,     "IMAGE_SAMPLE_C_L_O">;
defm : SampleRawPatterns<int_SI_image_sample_c_b_o,     "IMAGE_SAMPLE_C_B_O">;
defm : SampleRawPatterns<int_SI_image_sample_c_b_cl_o,  "IMAGE_SAMPLE_C_B_CL_O">;
defm : SampleRawPatterns<int_SI_image_sample_c_lz_o,    "IMAGE_SAMPLE_C_LZ_O">;
defm : SampleRawPatterns<int_SI_image_sample_c_cd_o,    "IMAGE_SAMPLE_C_CD_O">;
defm : SampleRawPatterns<int_SI_image_sample_c_cd_cl_o, "IMAGE_SAMPLE_C_CD_CL_O">;

// Gather opcodes
// Only the variants which make sense are defined.
def : SampleRawPattern<int_SI_gather4,           IMAGE_GATHER4_V4_V2,        v2i32>;
def : SampleRawPattern<int_SI_gather4,           IMAGE_GATHER4_V4_V4,        v4i32>;
def : SampleRawPattern<int_SI_gather4_cl,        IMAGE_GATHER4_CL_V4_V4,     v4i32>;
def : SampleRawPattern<int_SI_gather4_l,         IMAGE_GATHER4_L_V4_V4,      v4i32>;
def : SampleRawPattern<int_SI_gather4_b,         IMAGE_GATHER4_B_V4_V4,      v4i32>;
def : SampleRawPattern<int_SI_gather4_b_cl,      IMAGE_GATHER4_B_CL_V4_V4,   v4i32>;
def : SampleRawPattern<int_SI_gather4_b_cl,      IMAGE_GATHER4_B_CL_V4_V8,   v8i32>;
def : SampleRawPattern<int_SI_gather4_lz,        IMAGE_GATHER4_LZ_V4_V2,     v2i32>;
def : SampleRawPattern<int_SI_gather4_lz,        IMAGE_GATHER4_LZ_V4_V4,     v4i32>;

def : SampleRawPattern<int_SI_gather4_c,         IMAGE_GATHER4_C_V4_V4,      v4i32>;
def : SampleRawPattern<int_SI_gather4_c_cl,      IMAGE_GATHER4_C_CL_V4_V4,   v4i32>;
def : SampleRawPattern<int_SI_gather4_c_cl,      IMAGE_GATHER4_C_CL_V4_V8,   v8i32>;
def : SampleRawPattern<int_SI_gather4_c_l,       IMAGE_GATHER4_C_L_V4_V4,    v4i32>;
def : SampleRawPattern<int_SI_gather4_c_l,       IMAGE_GATHER4_C_L_V4_V8,    v8i32>;
def : SampleRawPattern<int_SI_gather4_c_b,       IMAGE_GATHER4_C_B_V4_V4,    v4i32>;
def : SampleRawPattern<int_SI_gather4_c_b,       IMAGE_GATHER4_C_B_V4_V8,    v8i32>;
def : SampleRawPattern<int_SI_gather4_c_b_cl,    IMAGE_GATHER4_C_B_CL_V4_V8, v8i32>;
def : SampleRawPattern<int_SI_gather4_c_lz,      IMAGE_GATHER4_C_LZ_V4_V4,   v4i32>;

def : SampleRawPattern<int_SI_gather4_o,         IMAGE_GATHER4_O_V4_V4,      v4i32>;
def : SampleRawPattern<int_SI_gather4_cl_o,      IMAGE_GATHER4_CL_O_V4_V4,   v4i32>;
def : SampleRawPattern<int_SI_gather4_cl_o,      IMAGE_GATHER4_CL_O_V4_V8,   v8i32>;
def : SampleRawPattern<int_SI_gather4_l_o,       IMAGE_GATHER4_L_O_V4_V4,    v4i32>;
def : SampleRawPattern<int_SI_gather4_l_o,       IMAGE_GATHER4_L_O_V4_V8,    v8i32>;
def : SampleRawPattern<int_SI_gather4_b_o,       IMAGE_GATHER4_B_O_V4_V4,    v4i32>;
def : SampleRawPattern<int_SI_gather4_b_o,       IMAGE_GATHER4_B_O_V4_V8,    v8i32>;
def : SampleRawPattern<int_SI_gather4_b_cl_o,    IMAGE_GATHER4_B_CL_O_V4_V8, v8i32>;
def : SampleRawPattern<int_SI_gather4_lz_o,      IMAGE_GATHER4_LZ_O_V4_V4,   v4i32>;

def : SampleRawPattern<int_SI_gather4_c_o,       IMAGE_GATHER4_C_O_V4_V4,    v4i32>;
def : SampleRawPattern<int_SI_gather4_c_o,       IMAGE_GATHER4_C_O_V4_V8,    v8i32>;
def : SampleRawPattern<int_SI_gather4_c_cl_o,    IMAGE_GATHER4_C_CL_O_V4_V8, v8i32>;
def : SampleRawPattern<int_SI_gather4_c_l_o,     IMAGE_GATHER4_C_L_O_V4_V8,  v8i32>;
def : SampleRawPattern<int_SI_gather4_c_b_o,     IMAGE_GATHER4_C_B_O_V4_V8,  v8i32>;
def : SampleRawPattern<int_SI_gather4_c_b_cl_o,  IMAGE_GATHER4_C_B_CL_O_V4_V8, v8i32>;
def : SampleRawPattern<int_SI_gather4_c_lz_o,    IMAGE_GATHER4_C_LZ_O_V4_V4, v4i32>;
def : SampleRawPattern<int_SI_gather4_c_lz_o,    IMAGE_GATHER4_C_LZ_O_V4_V8, v8i32>;

def : SampleRawPattern<int_SI_getlod, IMAGE_GET_LOD_V4_V1, i32>;
def : SampleRawPattern<int_SI_getlod, IMAGE_GET_LOD_V4_V2, v2i32>;
def : SampleRawPattern<int_SI_getlod, IMAGE_GET_LOD_V4_V4, v4i32>;

def : ImagePattern<int_SI_getresinfo, IMAGE_GET_RESINFO_V4_V1, i32>;
defm : ImagePatterns<int_SI_image_load, "IMAGE_LOAD">;
defm : ImagePatterns<int_SI_image_load_mip, "IMAGE_LOAD_MIP">;

/* SIsample for simple 1D texture lookup */
def : Pat <
  (SIsample i32:$addr, v32i8:$rsrc, v4i32:$sampler, imm),
  (IMAGE_SAMPLE_V4_V1 0xf, 0, 0, 0, 0, 0, 0, 0, $addr, $rsrc, $sampler)
>;

class SamplePattern<SDNode name, MIMG opcode, ValueType vt> : Pat <
    (name vt:$addr, v32i8:$rsrc, v4i32:$sampler, imm),
    (opcode 0xf, 0, 0, 0, 0, 0, 0, 0, $addr, $rsrc, $sampler)
>;

class SampleRectPattern<SDNode name, MIMG opcode, ValueType vt> : Pat <
    (name vt:$addr, v32i8:$rsrc, v4i32:$sampler, TEX_RECT),
    (opcode 0xf, 1, 0, 0, 0, 0, 0, 0, $addr, $rsrc, $sampler)
>;

class SampleArrayPattern<SDNode name, MIMG opcode, ValueType vt> : Pat <
    (name vt:$addr, v32i8:$rsrc, v4i32:$sampler, TEX_ARRAY),
    (opcode 0xf, 0, 0, 1, 0, 0, 0, 0, $addr, $rsrc, $sampler)
>;

class SampleShadowPattern<SDNode name, MIMG opcode,
                          ValueType vt> : Pat <
    (name vt:$addr, v32i8:$rsrc, v4i32:$sampler, TEX_SHADOW),
    (opcode 0xf, 0, 0, 0, 0, 0, 0, 0, $addr, $rsrc, $sampler)
>;

class SampleShadowArrayPattern<SDNode name, MIMG opcode,
                               ValueType vt> : Pat <
    (name vt:$addr, v32i8:$rsrc, v4i32:$sampler, TEX_SHADOW_ARRAY),
    (opcode 0xf, 0, 0, 1, 0, 0, 0, 0, $addr, $rsrc, $sampler)
>;

/* SIsample* for texture lookups consuming more address parameters */
multiclass SamplePatterns<MIMG sample, MIMG sample_c, MIMG sample_l,
                          MIMG sample_c_l, MIMG sample_b, MIMG sample_c_b,
MIMG sample_d, MIMG sample_c_d, ValueType addr_type> {
  def : SamplePattern <SIsample, sample, addr_type>;
  def : SampleRectPattern <SIsample, sample, addr_type>;
  def : SampleArrayPattern <SIsample, sample, addr_type>;
  def : SampleShadowPattern <SIsample, sample_c, addr_type>;
  def : SampleShadowArrayPattern <SIsample, sample_c, addr_type>;

  def : SamplePattern <SIsamplel, sample_l, addr_type>;
  def : SampleArrayPattern <SIsamplel, sample_l, addr_type>;
  def : SampleShadowPattern <SIsamplel, sample_c_l, addr_type>;
  def : SampleShadowArrayPattern <SIsamplel, sample_c_l, addr_type>;

  def : SamplePattern <SIsampleb, sample_b, addr_type>;
  def : SampleArrayPattern <SIsampleb, sample_b, addr_type>;
  def : SampleShadowPattern <SIsampleb, sample_c_b, addr_type>;
  def : SampleShadowArrayPattern <SIsampleb, sample_c_b, addr_type>;

  def : SamplePattern <SIsampled, sample_d, addr_type>;
  def : SampleArrayPattern <SIsampled, sample_d, addr_type>;
  def : SampleShadowPattern <SIsampled, sample_c_d, addr_type>;
  def : SampleShadowArrayPattern <SIsampled, sample_c_d, addr_type>;
}

defm : SamplePatterns<IMAGE_SAMPLE_V4_V2, IMAGE_SAMPLE_C_V4_V2,
                      IMAGE_SAMPLE_L_V4_V2, IMAGE_SAMPLE_C_L_V4_V2,
                      IMAGE_SAMPLE_B_V4_V2, IMAGE_SAMPLE_C_B_V4_V2,
                      IMAGE_SAMPLE_D_V4_V2, IMAGE_SAMPLE_C_D_V4_V2,
                      v2i32>;
defm : SamplePatterns<IMAGE_SAMPLE_V4_V4, IMAGE_SAMPLE_C_V4_V4,
                      IMAGE_SAMPLE_L_V4_V4, IMAGE_SAMPLE_C_L_V4_V4,
                      IMAGE_SAMPLE_B_V4_V4, IMAGE_SAMPLE_C_B_V4_V4,
                      IMAGE_SAMPLE_D_V4_V4, IMAGE_SAMPLE_C_D_V4_V4,
                      v4i32>;
defm : SamplePatterns<IMAGE_SAMPLE_V4_V8, IMAGE_SAMPLE_C_V4_V8,
                      IMAGE_SAMPLE_L_V4_V8, IMAGE_SAMPLE_C_L_V4_V8,
                      IMAGE_SAMPLE_B_V4_V8, IMAGE_SAMPLE_C_B_V4_V8,
                      IMAGE_SAMPLE_D_V4_V8, IMAGE_SAMPLE_C_D_V4_V8,
                      v8i32>;
defm : SamplePatterns<IMAGE_SAMPLE_V4_V16, IMAGE_SAMPLE_C_V4_V16,
                      IMAGE_SAMPLE_L_V4_V16, IMAGE_SAMPLE_C_L_V4_V16,
                      IMAGE_SAMPLE_B_V4_V16, IMAGE_SAMPLE_C_B_V4_V16,
                      IMAGE_SAMPLE_D_V4_V16, IMAGE_SAMPLE_C_D_V4_V16,
                      v16i32>;

/* int_SI_imageload for texture fetches consuming varying address parameters */
class ImageLoadPattern<Intrinsic name, MIMG opcode, ValueType addr_type> : Pat <
    (name addr_type:$addr, v32i8:$rsrc, imm),
    (opcode 0xf, 0, 0, 0, 0, 0, 0, 0, $addr, $rsrc)
>;

class ImageLoadArrayPattern<Intrinsic name, MIMG opcode, ValueType addr_type> : Pat <
    (name addr_type:$addr, v32i8:$rsrc, TEX_ARRAY),
    (opcode 0xf, 0, 0, 1, 0, 0, 0, 0, $addr, $rsrc)
>;

class ImageLoadMSAAPattern<Intrinsic name, MIMG opcode, ValueType addr_type> : Pat <
    (name addr_type:$addr, v32i8:$rsrc, TEX_MSAA),
    (opcode 0xf, 0, 0, 0, 0, 0, 0, 0, $addr, $rsrc)
>;

class ImageLoadArrayMSAAPattern<Intrinsic name, MIMG opcode, ValueType addr_type> : Pat <
    (name addr_type:$addr, v32i8:$rsrc, TEX_ARRAY_MSAA),
    (opcode 0xf, 0, 0, 1, 0, 0, 0, 0, $addr, $rsrc)
>;

multiclass ImageLoadPatterns<MIMG opcode, ValueType addr_type> {
  def : ImageLoadPattern <int_SI_imageload, opcode, addr_type>;
  def : ImageLoadArrayPattern <int_SI_imageload, opcode, addr_type>;
}

multiclass ImageLoadMSAAPatterns<MIMG opcode, ValueType addr_type> {
  def : ImageLoadMSAAPattern <int_SI_imageload, opcode, addr_type>;
  def : ImageLoadArrayMSAAPattern <int_SI_imageload, opcode, addr_type>;
}

defm : ImageLoadPatterns<IMAGE_LOAD_MIP_V4_V2, v2i32>;
defm : ImageLoadPatterns<IMAGE_LOAD_MIP_V4_V4, v4i32>;

defm : ImageLoadMSAAPatterns<IMAGE_LOAD_V4_V2, v2i32>;
defm : ImageLoadMSAAPatterns<IMAGE_LOAD_V4_V4, v4i32>;

/* Image resource information */
def : Pat <
  (int_SI_resinfo i32:$mipid, v32i8:$rsrc, imm),
  (IMAGE_GET_RESINFO_V4_V1 0xf, 0, 0, 0, 0, 0, 0, 0, (V_MOV_B32_e32 $mipid), $rsrc)
>;

def : Pat <
  (int_SI_resinfo i32:$mipid, v32i8:$rsrc, TEX_ARRAY),
  (IMAGE_GET_RESINFO_V4_V1 0xf, 0, 0, 1, 0, 0, 0, 0, (V_MOV_B32_e32 $mipid), $rsrc)
>;

def : Pat <
  (int_SI_resinfo i32:$mipid, v32i8:$rsrc, TEX_ARRAY_MSAA),
  (IMAGE_GET_RESINFO_V4_V1 0xf, 0, 0, 1, 0, 0, 0, 0, (V_MOV_B32_e32 $mipid), $rsrc)
>;

/********** ============================================ **********/
/********** Extraction, Insertion, Building and Casting  **********/
/********** ============================================ **********/

foreach Index = 0-2 in {
  def Extract_Element_v2i32_#Index : Extract_Element <
    i32, v2i32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v2i32_#Index : Insert_Element <
    i32, v2i32, Index, !cast<SubRegIndex>(sub#Index)
  >;

  def Extract_Element_v2f32_#Index : Extract_Element <
    f32, v2f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v2f32_#Index : Insert_Element <
    f32, v2f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
}

foreach Index = 0-3 in {
  def Extract_Element_v4i32_#Index : Extract_Element <
    i32, v4i32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v4i32_#Index : Insert_Element <
    i32, v4i32, Index, !cast<SubRegIndex>(sub#Index)
  >;

  def Extract_Element_v4f32_#Index : Extract_Element <
    f32, v4f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v4f32_#Index : Insert_Element <
    f32, v4f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
}

foreach Index = 0-7 in {
  def Extract_Element_v8i32_#Index : Extract_Element <
    i32, v8i32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v8i32_#Index : Insert_Element <
    i32, v8i32, Index, !cast<SubRegIndex>(sub#Index)
  >;

  def Extract_Element_v8f32_#Index : Extract_Element <
    f32, v8f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v8f32_#Index : Insert_Element <
    f32, v8f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
}

foreach Index = 0-15 in {
  def Extract_Element_v16i32_#Index : Extract_Element <
    i32, v16i32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v16i32_#Index : Insert_Element <
    i32, v16i32, Index, !cast<SubRegIndex>(sub#Index)
  >;

  def Extract_Element_v16f32_#Index : Extract_Element <
    f32, v16f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v16f32_#Index : Insert_Element <
    f32, v16f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
}

def : BitConvert <i32, f32, SReg_32>;
def : BitConvert <i32, f32, VGPR_32>;

def : BitConvert <f32, i32, SReg_32>;
def : BitConvert <f32, i32, VGPR_32>;

def : BitConvert <i64, f64, VReg_64>;

def : BitConvert <f64, i64, VReg_64>;

def : BitConvert <v2f32, v2i32, VReg_64>;
def : BitConvert <v2i32, v2f32, VReg_64>;
def : BitConvert <v2i32, i64, VReg_64>;
def : BitConvert <i64, v2i32, VReg_64>;
def : BitConvert <v2f32, i64, VReg_64>;
def : BitConvert <i64, v2f32, VReg_64>;
def : BitConvert <v2i32, f64, VReg_64>;
def : BitConvert <f64, v2i32, VReg_64>;
def : BitConvert <v4f32, v4i32, VReg_128>;
def : BitConvert <v4i32, v4f32, VReg_128>;

def : BitConvert <v8f32, v8i32, SReg_256>;
def : BitConvert <v8i32, v8f32, SReg_256>;
def : BitConvert <v8i32, v32i8, SReg_256>;
def : BitConvert <v32i8, v8i32, SReg_256>;
def : BitConvert <v8i32, v32i8, VReg_256>;
def : BitConvert <v8i32, v8f32, VReg_256>;
def : BitConvert <v8f32, v8i32, VReg_256>;
def : BitConvert <v32i8, v8i32, VReg_256>;

def : BitConvert <v16i32, v16f32, VReg_512>;
def : BitConvert <v16f32, v16i32, VReg_512>;

/********** =================== **********/
/********** Src & Dst modifiers **********/
/********** =================== **********/

def : Pat <
  (AMDGPUclamp (VOP3Mods0Clamp f32:$src0, i32:$src0_modifiers, i32:$omod),
               (f32 FP_ZERO), (f32 FP_ONE)),
  (V_ADD_F32_e64 $src0_modifiers, $src0, 0, 0, 1, $omod)
>;

/********** ================================ **********/
/********** Floating point absolute/negative **********/
/********** ================================ **********/

// Prevent expanding both fneg and fabs.

def : Pat <
  (fneg (fabs f32:$src)),
  (S_OR_B32 $src, 0x80000000) /* Set sign bit */
>;

// FIXME: Should use S_OR_B32
def : Pat <
  (fneg (fabs f64:$src)),
  (REG_SEQUENCE VReg_64,
    (i32 (EXTRACT_SUBREG f64:$src, sub0)),
    sub0,
    (V_OR_B32_e32 (EXTRACT_SUBREG f64:$src, sub1),
                  (V_MOV_B32_e32 0x80000000)), // Set sign bit.
    sub1)
>;

def : Pat <
  (fabs f32:$src),
  (V_AND_B32_e32 $src, (V_MOV_B32_e32 0x7fffffff))
>;

def : Pat <
  (fneg f32:$src),
  (V_XOR_B32_e32 $src, (V_MOV_B32_e32 0x80000000))
>;

def : Pat <
  (fabs f64:$src),
  (REG_SEQUENCE VReg_64,
    (i32 (EXTRACT_SUBREG f64:$src, sub0)),
    sub0,
    (V_AND_B32_e32 (EXTRACT_SUBREG f64:$src, sub1),
                   (V_MOV_B32_e32 0x7fffffff)), // Set sign bit.
     sub1)
>;

def : Pat <
  (fneg f64:$src),
  (REG_SEQUENCE VReg_64,
    (i32 (EXTRACT_SUBREG f64:$src, sub0)),
    sub0,
    (V_XOR_B32_e32 (EXTRACT_SUBREG f64:$src, sub1),
                   (V_MOV_B32_e32 0x80000000)),
    sub1)
>;

/********** ================== **********/
/********** Immediate Patterns **********/
/********** ================== **********/

def : Pat <
  (SGPRImm<(i32 imm)>:$imm),
  (S_MOV_B32 imm:$imm)
>;

def : Pat <
  (SGPRImm<(f32 fpimm)>:$imm),
  (S_MOV_B32 (f32 (bitcast_fpimm_to_i32 $imm)))
>;

def : Pat <
  (i32 imm:$imm),
  (V_MOV_B32_e32 imm:$imm)
>;

def : Pat <
  (f32 fpimm:$imm),
  (V_MOV_B32_e32 (f32 (bitcast_fpimm_to_i32 $imm)))
>;

def : Pat <
  (i64 InlineImm<i64>:$imm),
  (S_MOV_B64 InlineImm<i64>:$imm)
>;

// XXX - Should this use a s_cmp to set SCC?

// Set to sign-extended 64-bit value (true = -1, false = 0)
def : Pat <
  (i1 imm:$imm),
  (S_MOV_B64 (i64 (as_i64imm $imm)))
>;

def : Pat <
  (f64 InlineFPImm<f64>:$imm),
  (S_MOV_B64 (f64 (bitcast_fpimm_to_i64 InlineFPImm<f64>:$imm)))
>;

/********** ================== **********/
/********** Intrinsic Patterns **********/
/********** ================== **********/

/* llvm.AMDGPU.pow */
def : POW_Common <V_LOG_F32_e32, V_EXP_F32_e32, V_MUL_LEGACY_F32_e32>;

def : Pat <
  (int_AMDGPU_div f32:$src0, f32:$src1),
  (V_MUL_LEGACY_F32_e32 $src0, (V_RCP_LEGACY_F32_e32 $src1))
>;

def : Pat <
  (int_AMDGPU_cube v4f32:$src),
  (REG_SEQUENCE VReg_128,
    (V_CUBETC_F32 0 /* src0_modifiers */, (EXTRACT_SUBREG $src, sub0),
                  0 /* src1_modifiers */, (EXTRACT_SUBREG $src, sub1),
                  0 /* src2_modifiers */, (EXTRACT_SUBREG $src, sub2),
                  0 /* clamp */, 0 /* omod */), sub0,
    (V_CUBESC_F32 0 /* src0_modifiers */, (EXTRACT_SUBREG $src, sub0),
                  0 /* src1_modifiers */,(EXTRACT_SUBREG $src, sub1),
                  0 /* src2_modifiers */,(EXTRACT_SUBREG $src, sub2),
                  0 /* clamp */, 0 /* omod */), sub1,
    (V_CUBEMA_F32 0 /* src1_modifiers */,(EXTRACT_SUBREG $src, sub0),
                  0 /* src1_modifiers */,(EXTRACT_SUBREG $src, sub1),
                  0 /* src1_modifiers */,(EXTRACT_SUBREG $src, sub2),
                  0 /* clamp */, 0 /* omod */), sub2,
    (V_CUBEID_F32 0 /* src1_modifiers */,(EXTRACT_SUBREG $src, sub0),
                  0 /* src1_modifiers */,(EXTRACT_SUBREG $src, sub1),
                  0 /* src1_modifiers */,(EXTRACT_SUBREG $src, sub2),
                  0 /* clamp */, 0 /* omod */), sub3)
>;

def : Pat <
  (i32 (sext i1:$src0)),
  (V_CNDMASK_B32_e64 (i32 0), (i32 -1), $src0)
>;

class Ext32Pat <SDNode ext> : Pat <
  (i32 (ext i1:$src0)),
  (V_CNDMASK_B32_e64 (i32 0), (i32 1), $src0)
>;

def : Ext32Pat <zext>;
def : Ext32Pat <anyext>;

// Offset in an 32Bit VGPR
def : Pat <
  (SIload_constant v4i32:$sbase, i32:$voff),
  (BUFFER_LOAD_DWORD_OFFEN $voff, $sbase, 0, 0, 0, 0, 0)
>;

// The multiplication scales from [0,1] to the unsigned integer range
def : Pat <
  (AMDGPUurecip i32:$src0),
  (V_CVT_U32_F32_e32
    (V_MUL_F32_e32 CONST.FP_UINT_MAX_PLUS_1,
                   (V_RCP_IFLAG_F32_e32 (V_CVT_F32_U32_e32 $src0))))
>;

def : Pat <
  (int_SI_tid),
  (V_MBCNT_HI_U32_B32_e64 0xffffffff,
                          (V_MBCNT_LO_U32_B32_e64 0xffffffff, 0))
>;

//===----------------------------------------------------------------------===//
// VOP3 Patterns
//===----------------------------------------------------------------------===//

def : IMad24Pat<V_MAD_I32_I24>;
def : UMad24Pat<V_MAD_U32_U24>;

def : Pat <
  (mulhu i32:$src0, i32:$src1),
  (V_MUL_HI_U32 $src0, $src1)
>;

def : Pat <
  (mulhs i32:$src0, i32:$src1),
  (V_MUL_HI_I32 $src0, $src1)
>;

defm : BFIPatterns <V_BFI_B32, S_MOV_B32, SReg_64>;
def : ROTRPattern <V_ALIGNBIT_B32>;

/********** ======================= **********/
/**********   Load/Store Patterns   **********/
/********** ======================= **********/

class DSReadPat <DS inst, ValueType vt, PatFrag frag> : Pat <
  (vt (frag (DS1Addr1Offset i32:$ptr, i32:$offset))),
  (inst $ptr, (as_i16imm $offset), (i1 0))
>;

def : DSReadPat <DS_READ_I8,  i32, si_sextload_local_i8>;
def : DSReadPat <DS_READ_U8,  i32, si_az_extload_local_i8>;
def : DSReadPat <DS_READ_I16, i32, si_sextload_local_i16>;
def : DSReadPat <DS_READ_U16, i32, si_az_extload_local_i16>;
def : DSReadPat <DS_READ_B32, i32, si_load_local>;

let AddedComplexity = 100 in {

def : DSReadPat <DS_READ_B64, v2i32, si_load_local_align8>;

} // End AddedComplexity = 100

def : Pat <
  (v2i32 (si_load_local (DS64Bit4ByteAligned i32:$ptr, i8:$offset0,
                                                    i8:$offset1))),
  (DS_READ2_B32 $ptr, $offset0, $offset1, (i1 0))
>;

class DSWritePat <DS inst, ValueType vt, PatFrag frag> : Pat <
  (frag vt:$value, (DS1Addr1Offset i32:$ptr, i32:$offset)),
  (inst $ptr, $value, (as_i16imm $offset), (i1 0))
>;

def : DSWritePat <DS_WRITE_B8, i32, si_truncstore_local_i8>;
def : DSWritePat <DS_WRITE_B16, i32, si_truncstore_local_i16>;
def : DSWritePat <DS_WRITE_B32, i32, si_store_local>;

let AddedComplexity = 100 in {

def : DSWritePat <DS_WRITE_B64, v2i32, si_store_local_align8>;
} // End AddedComplexity = 100

def : Pat <
  (si_store_local v2i32:$value, (DS64Bit4ByteAligned i32:$ptr, i8:$offset0,
                                                               i8:$offset1)),
  (DS_WRITE2_B32 $ptr, (EXTRACT_SUBREG $value, sub0),
                       (EXTRACT_SUBREG $value, sub1), $offset0, $offset1,
                       (i1 0))
>;

class DSAtomicRetPat<DS inst, ValueType vt, PatFrag frag> : Pat <
  (frag (DS1Addr1Offset i32:$ptr, i32:$offset), vt:$value),
  (inst $ptr, $value, (as_i16imm $offset), (i1 0))
>;

// Special case of DSAtomicRetPat for add / sub 1 -> inc / dec
//
// We need to use something for the data0, so we set a register to
// -1. For the non-rtn variants, the manual says it does
// DS[A] = (DS[A] >= D0) ? 0 : DS[A] + 1, and setting D0 to uint_max
// will always do the increment so I'm assuming it's the same.
class DSAtomicIncRetPat<DS inst, ValueType vt,
                        Instruction LoadImm, PatFrag frag> : Pat <
  (frag (DS1Addr1Offset i32:$ptr, i32:$offset), (vt 1)),
  (inst $ptr, (LoadImm (vt -1)), (as_i16imm $offset), (i1 0))
>;


class DSAtomicCmpXChg <DS inst, ValueType vt, PatFrag frag> : Pat <
  (frag (DS1Addr1Offset i32:$ptr, i32:$offset), vt:$cmp, vt:$swap),
  (inst $ptr, $cmp, $swap, (as_i16imm $offset), (i1 0))
>;


// 32-bit atomics.
def : DSAtomicIncRetPat<DS_INC_RTN_U32, i32,
                        V_MOV_B32_e32, si_atomic_load_add_local>;
def : DSAtomicIncRetPat<DS_DEC_RTN_U32, i32,
                        V_MOV_B32_e32, si_atomic_load_sub_local>;

def : DSAtomicRetPat<DS_WRXCHG_RTN_B32, i32, si_atomic_swap_local>;
def : DSAtomicRetPat<DS_ADD_RTN_U32, i32, si_atomic_load_add_local>;
def : DSAtomicRetPat<DS_SUB_RTN_U32, i32, si_atomic_load_sub_local>;
def : DSAtomicRetPat<DS_AND_RTN_B32, i32, si_atomic_load_and_local>;
def : DSAtomicRetPat<DS_OR_RTN_B32, i32, si_atomic_load_or_local>;
def : DSAtomicRetPat<DS_XOR_RTN_B32, i32, si_atomic_load_xor_local>;
def : DSAtomicRetPat<DS_MIN_RTN_I32, i32, si_atomic_load_min_local>;
def : DSAtomicRetPat<DS_MAX_RTN_I32, i32, si_atomic_load_max_local>;
def : DSAtomicRetPat<DS_MIN_RTN_U32, i32, si_atomic_load_umin_local>;
def : DSAtomicRetPat<DS_MAX_RTN_U32, i32, si_atomic_load_umax_local>;

def : DSAtomicCmpXChg<DS_CMPST_RTN_B32, i32, si_atomic_cmp_swap_32_local>;

// 64-bit atomics.
def : DSAtomicIncRetPat<DS_INC_RTN_U64, i64,
                        V_MOV_B64_PSEUDO, si_atomic_load_add_local>;
def : DSAtomicIncRetPat<DS_DEC_RTN_U64, i64,
                        V_MOV_B64_PSEUDO, si_atomic_load_sub_local>;

def : DSAtomicRetPat<DS_WRXCHG_RTN_B64, i64, si_atomic_swap_local>;
def : DSAtomicRetPat<DS_ADD_RTN_U64, i64, si_atomic_load_add_local>;
def : DSAtomicRetPat<DS_SUB_RTN_U64, i64, si_atomic_load_sub_local>;
def : DSAtomicRetPat<DS_AND_RTN_B64, i64, si_atomic_load_and_local>;
def : DSAtomicRetPat<DS_OR_RTN_B64, i64, si_atomic_load_or_local>;
def : DSAtomicRetPat<DS_XOR_RTN_B64, i64, si_atomic_load_xor_local>;
def : DSAtomicRetPat<DS_MIN_RTN_I64, i64, si_atomic_load_min_local>;
def : DSAtomicRetPat<DS_MAX_RTN_I64, i64, si_atomic_load_max_local>;
def : DSAtomicRetPat<DS_MIN_RTN_U64, i64, si_atomic_load_umin_local>;
def : DSAtomicRetPat<DS_MAX_RTN_U64, i64, si_atomic_load_umax_local>;

def : DSAtomicCmpXChg<DS_CMPST_RTN_B64, i64, si_atomic_cmp_swap_64_local>;


//===----------------------------------------------------------------------===//
// MUBUF Patterns
//===----------------------------------------------------------------------===//

multiclass MUBUFLoad_Pattern <MUBUF Instr_ADDR64, ValueType vt,
                              PatFrag constant_ld> {
  def : Pat <
     (vt (constant_ld (MUBUFAddr64 v4i32:$srsrc, i64:$vaddr, i32:$soffset,
                                   i16:$offset, i1:$glc, i1:$slc, i1:$tfe))),
     (Instr_ADDR64 $vaddr, $srsrc, $soffset, $offset, $glc, $slc, $tfe)
  >;
}

let Predicates = [isSICI] in {
defm : MUBUFLoad_Pattern <BUFFER_LOAD_SBYTE_ADDR64, i32, sextloadi8_constant>;
defm : MUBUFLoad_Pattern <BUFFER_LOAD_UBYTE_ADDR64, i32, az_extloadi8_constant>;
defm : MUBUFLoad_Pattern <BUFFER_LOAD_SSHORT_ADDR64, i32, sextloadi16_constant>;
defm : MUBUFLoad_Pattern <BUFFER_LOAD_USHORT_ADDR64, i32, az_extloadi16_constant>;
} // End Predicates = [isSICI]

class MUBUFScratchLoadPat <MUBUF Instr, ValueType vt, PatFrag ld> : Pat <
  (vt (ld (MUBUFScratch v4i32:$srsrc, i32:$vaddr,
                        i32:$soffset, u16imm:$offset))),
  (Instr $vaddr, $srsrc, $soffset, $offset, 0, 0, 0)
>;

def : MUBUFScratchLoadPat <BUFFER_LOAD_SBYTE_OFFEN, i32, sextloadi8_private>;
def : MUBUFScratchLoadPat <BUFFER_LOAD_UBYTE_OFFEN, i32, extloadi8_private>;
def : MUBUFScratchLoadPat <BUFFER_LOAD_SSHORT_OFFEN, i32, sextloadi16_private>;
def : MUBUFScratchLoadPat <BUFFER_LOAD_USHORT_OFFEN, i32, extloadi16_private>;
def : MUBUFScratchLoadPat <BUFFER_LOAD_DWORD_OFFEN, i32, load_private>;
def : MUBUFScratchLoadPat <BUFFER_LOAD_DWORDX2_OFFEN, v2i32, load_private>;
def : MUBUFScratchLoadPat <BUFFER_LOAD_DWORDX4_OFFEN, v4i32, load_private>;

// BUFFER_LOAD_DWORD*, addr64=0
multiclass MUBUF_Load_Dword <ValueType vt, MUBUF offset, MUBUF offen, MUBUF idxen,
                             MUBUF bothen> {

  def : Pat <
    (vt (int_SI_buffer_load_dword v4i32:$rsrc, (i32 imm), i32:$soffset,
                                  imm:$offset, 0, 0, imm:$glc, imm:$slc,
                                  imm:$tfe)),
    (offset $rsrc, $soffset, (as_i16imm $offset), (as_i1imm $glc),
            (as_i1imm $slc), (as_i1imm $tfe))
  >;

  def : Pat <
    (vt (int_SI_buffer_load_dword v4i32:$rsrc, i32:$vaddr, i32:$soffset,
                                  imm:$offset, 1, 0, imm:$glc, imm:$slc,
                                  imm:$tfe)),
    (offen $vaddr, $rsrc, $soffset, (as_i16imm $offset), (as_i1imm $glc), (as_i1imm $slc),
           (as_i1imm $tfe))
  >;

  def : Pat <
    (vt (int_SI_buffer_load_dword v4i32:$rsrc, i32:$vaddr, i32:$soffset,
                                  imm:$offset, 0, 1, imm:$glc, imm:$slc,
                                  imm:$tfe)),
    (idxen $vaddr, $rsrc, $soffset, (as_i16imm $offset), (as_i1imm $glc),
           (as_i1imm $slc), (as_i1imm $tfe))
  >;

  def : Pat <
    (vt (int_SI_buffer_load_dword v4i32:$rsrc, v2i32:$vaddr, i32:$soffset,
                                  imm:$offset, 1, 1, imm:$glc, imm:$slc,
                                  imm:$tfe)),
    (bothen $vaddr, $rsrc, $soffset, (as_i16imm $offset), (as_i1imm $glc), (as_i1imm $slc),
            (as_i1imm $tfe))
  >;
}

defm : MUBUF_Load_Dword <i32, BUFFER_LOAD_DWORD_OFFSET, BUFFER_LOAD_DWORD_OFFEN,
                         BUFFER_LOAD_DWORD_IDXEN, BUFFER_LOAD_DWORD_BOTHEN>;
defm : MUBUF_Load_Dword <v2i32, BUFFER_LOAD_DWORDX2_OFFSET, BUFFER_LOAD_DWORDX2_OFFEN,
                         BUFFER_LOAD_DWORDX2_IDXEN, BUFFER_LOAD_DWORDX2_BOTHEN>;
defm : MUBUF_Load_Dword <v4i32, BUFFER_LOAD_DWORDX4_OFFSET, BUFFER_LOAD_DWORDX4_OFFEN,
                         BUFFER_LOAD_DWORDX4_IDXEN, BUFFER_LOAD_DWORDX4_BOTHEN>;

class MUBUFScratchStorePat <MUBUF Instr, ValueType vt, PatFrag st> : Pat <
  (st vt:$value, (MUBUFScratch v4i32:$srsrc, i32:$vaddr, i32:$soffset,
                               u16imm:$offset)),
  (Instr $value, $vaddr, $srsrc, $soffset, $offset, 0, 0, 0)
>;

def : MUBUFScratchStorePat <BUFFER_STORE_BYTE_OFFEN, i32, truncstorei8_private>;
def : MUBUFScratchStorePat <BUFFER_STORE_SHORT_OFFEN, i32, truncstorei16_private>;
def : MUBUFScratchStorePat <BUFFER_STORE_DWORD_OFFEN, i32, store_private>;
def : MUBUFScratchStorePat <BUFFER_STORE_DWORDX2_OFFEN, v2i32, store_private>;
def : MUBUFScratchStorePat <BUFFER_STORE_DWORDX4_OFFEN, v4i32, store_private>;

/*
class MUBUFStore_Pattern <MUBUF Instr, ValueType vt, PatFrag st> : Pat <
  (st vt:$value, (MUBUFScratch v4i32:$srsrc, i64:$vaddr, u16imm:$offset)),
  (Instr $value, $srsrc, $vaddr, $offset)
>;

let Predicates = [isSICI] in {
def : MUBUFStore_Pattern <BUFFER_STORE_BYTE_ADDR64, i32, truncstorei8_private>;
def : MUBUFStore_Pattern <BUFFER_STORE_SHORT_ADDR64, i32, truncstorei16_private>;
def : MUBUFStore_Pattern <BUFFER_STORE_DWORD_ADDR64, i32, store_private>;
def : MUBUFStore_Pattern <BUFFER_STORE_DWORDX2_ADDR64, v2i32, store_private>;
def : MUBUFStore_Pattern <BUFFER_STORE_DWORDX4_ADDR64, v4i32, store_private>;
} // End Predicates = [isSICI]

*/

//===----------------------------------------------------------------------===//
// MTBUF Patterns
//===----------------------------------------------------------------------===//

// TBUFFER_STORE_FORMAT_*, addr64=0
class MTBUF_StoreResource <ValueType vt, int num_channels, MTBUF opcode> : Pat<
  (SItbuffer_store v4i32:$rsrc, vt:$vdata, num_channels, i32:$vaddr,
                   i32:$soffset, imm:$inst_offset, imm:$dfmt,
                   imm:$nfmt, imm:$offen, imm:$idxen,
                   imm:$glc, imm:$slc, imm:$tfe),
  (opcode
    $vdata, (as_i16imm $inst_offset), (as_i1imm $offen), (as_i1imm $idxen),
    (as_i1imm $glc), 0, (as_i8imm $dfmt), (as_i8imm $nfmt), $vaddr, $rsrc,
    (as_i1imm $slc), (as_i1imm $tfe), $soffset)
>;

def : MTBUF_StoreResource <i32, 1, TBUFFER_STORE_FORMAT_X>;
def : MTBUF_StoreResource <v2i32, 2, TBUFFER_STORE_FORMAT_XY>;
def : MTBUF_StoreResource <v4i32, 3, TBUFFER_STORE_FORMAT_XYZ>;
def : MTBUF_StoreResource <v4i32, 4, TBUFFER_STORE_FORMAT_XYZW>;

/********** ====================== **********/
/**********   Indirect adressing   **********/
/********** ====================== **********/

multiclass SI_INDIRECT_Pattern <ValueType vt, ValueType eltvt, string VecSize> {

  // 1. Extract with offset
  def : Pat<
    (eltvt (vector_extract vt:$vec, (add i32:$idx, imm:$off))),
    (!cast<Instruction>("SI_INDIRECT_SRC_"#VecSize) $vec, $idx, imm:$off)
  >;

  // 2. Extract without offset
  def : Pat<
    (eltvt (vector_extract vt:$vec, i32:$idx)),
    (!cast<Instruction>("SI_INDIRECT_SRC_"#VecSize) $vec, $idx, 0)
  >;

  // 3. Insert with offset
  def : Pat<
    (vector_insert vt:$vec, eltvt:$val, (add i32:$idx, imm:$off)),
    (!cast<Instruction>("SI_INDIRECT_DST_"#VecSize) $vec, $idx, imm:$off, $val)
  >;

  // 4. Insert without offset
  def : Pat<
    (vector_insert vt:$vec, eltvt:$val, i32:$idx),
    (!cast<Instruction>("SI_INDIRECT_DST_"#VecSize) $vec, $idx, 0, $val)
  >;
}

defm : SI_INDIRECT_Pattern <v2f32, f32, "V2">;
defm : SI_INDIRECT_Pattern <v4f32, f32, "V4">;
defm : SI_INDIRECT_Pattern <v8f32, f32, "V8">;
defm : SI_INDIRECT_Pattern <v16f32, f32, "V16">;

defm : SI_INDIRECT_Pattern <v2i32, i32, "V2">;
defm : SI_INDIRECT_Pattern <v4i32, i32, "V4">;
defm : SI_INDIRECT_Pattern <v8i32, i32, "V8">;
defm : SI_INDIRECT_Pattern <v16i32, i32, "V16">;

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

def : Pat<(i32 (sext_inreg i32:$src, i1)),
  (S_BFE_I32 i32:$src, 65536)>; // 0 | 1 << 16

// Handle sext_inreg in i64
def : Pat <
  (i64 (sext_inreg i64:$src, i1)),
  (S_BFE_I64 i64:$src, 0x10000) // 0 | 1 << 16
>;

def : Pat <
  (i64 (sext_inreg i64:$src, i8)),
  (S_BFE_I64 i64:$src, 0x80000) // 0 | 8 << 16
>;

def : Pat <
  (i64 (sext_inreg i64:$src, i16)),
  (S_BFE_I64 i64:$src, 0x100000) // 0 | 16 << 16
>;

def : Pat <
  (i64 (sext_inreg i64:$src, i32)),
  (S_BFE_I64 i64:$src, 0x200000) // 0 | 32 << 16
>;

class ZExt_i64_i32_Pat <SDNode ext> : Pat <
  (i64 (ext i32:$src)),
  (REG_SEQUENCE SReg_64, $src, sub0, (S_MOV_B32 0), sub1)
>;

class ZExt_i64_i1_Pat <SDNode ext> : Pat <
  (i64 (ext i1:$src)),
    (REG_SEQUENCE VReg_64,
      (V_CNDMASK_B32_e64 (i32 0), (i32 1), $src), sub0,
      (S_MOV_B32 0), sub1)
>;


def : ZExt_i64_i32_Pat<zext>;
def : ZExt_i64_i32_Pat<anyext>;
def : ZExt_i64_i1_Pat<zext>;
def : ZExt_i64_i1_Pat<anyext>;

def : Pat <
  (i64 (sext i32:$src)),
    (REG_SEQUENCE SReg_64, $src, sub0,
    (S_ASHR_I32 $src, 31), sub1)
>;

def : Pat <
  (i64 (sext i1:$src)),
  (REG_SEQUENCE VReg_64,
    (V_CNDMASK_B32_e64 0, -1, $src), sub0,
    (V_CNDMASK_B32_e64 0, -1, $src), sub1)
>;

// If we need to perform a logical operation on i1 values, we need to
// use vector comparisons since there is only one SCC register. Vector
// comparisions still write to a pair of SGPRs, so treat these as
// 64-bit comparisons. When legalizing SGPR copies, instructions
// resulting in the copies from SCC to these instructions will be
// moved to the VALU.
def : Pat <
  (i1 (and i1:$src0, i1:$src1)),
  (S_AND_B64 $src0, $src1)
>;

def : Pat <
  (i1 (or i1:$src0, i1:$src1)),
  (S_OR_B64 $src0, $src1)
>;

def : Pat <
  (i1 (xor i1:$src0, i1:$src1)),
  (S_XOR_B64 $src0, $src1)
>;

def : Pat <
  (f32 (sint_to_fp i1:$src)),
  (V_CNDMASK_B32_e64 (i32 0), CONST.FP32_NEG_ONE, $src)
>;

def : Pat <
  (f32 (uint_to_fp i1:$src)),
  (V_CNDMASK_B32_e64 (i32 0), CONST.FP32_ONE, $src)
>;

def : Pat <
  (f64 (sint_to_fp i1:$src)),
  (V_CVT_F64_I32_e32 (V_CNDMASK_B32_e64 (i32 0), (i32 -1), $src))
>;

def : Pat <
  (f64 (uint_to_fp i1:$src)),
  (V_CVT_F64_U32_e32 (V_CNDMASK_B32_e64 (i32 0), (i32 1), $src))
>;

//===----------------------------------------------------------------------===//
// Miscellaneous Patterns
//===----------------------------------------------------------------------===//

def : Pat <
  (i32 (trunc i64:$a)),
  (EXTRACT_SUBREG $a, sub0)
>;

def : Pat <
  (i1 (trunc i32:$a)),
  (V_CMP_EQ_I32_e64 (S_AND_B32 (i32 1), $a), 1)
>;

def : Pat <
  (i1 (trunc i64:$a)),
  (V_CMP_EQ_I32_e64 (S_AND_B32 (i32 1),
                    (EXTRACT_SUBREG $a, sub0)), 1)
>;

def : Pat <
  (i32 (bswap i32:$a)),
  (V_BFI_B32 (S_MOV_B32 0x00ff00ff),
             (V_ALIGNBIT_B32 $a, $a, 24),
             (V_ALIGNBIT_B32 $a, $a, 8))
>;

def : Pat <
  (f32 (select i1:$src2, f32:$src1, f32:$src0)),
  (V_CNDMASK_B32_e64 $src0, $src1, $src2)
>;

multiclass BFMPatterns <ValueType vt, InstSI BFM, InstSI MOV> {
  def : Pat <
    (vt (shl (vt (add (vt (shl 1, vt:$a)), -1)), vt:$b)),
    (BFM $a, $b)
  >;

  def : Pat <
    (vt (add (vt (shl 1, vt:$a)), -1)),
    (BFM $a, (MOV 0))
  >;
}

defm : BFMPatterns <i32, S_BFM_B32, S_MOV_B32>;
// FIXME: defm : BFMPatterns <i64, S_BFM_B64, S_MOV_B64>;

def : BFEPattern <V_BFE_U32, S_MOV_B32>;

//===----------------------------------------------------------------------===//
// Fract Patterns
//===----------------------------------------------------------------------===//

let Predicates = [isSI] in {

// V_FRACT is buggy on SI, so the F32 version is never used and (x-floor(x)) is
// used instead. However, SI doesn't have V_FLOOR_F64, so the most efficient
// way to implement it is using V_FRACT_F64.
// The workaround for the V_FRACT bug is:
//    fract(x) = isnan(x) ? x : min(V_FRACT(x), 0.99999999999999999)

// Convert (x + (-floor(x)) to fract(x)
def : Pat <
  (f64 (fadd (f64 (VOP3Mods f64:$x, i32:$mods)),
             (f64 (fneg (f64 (ffloor (f64 (VOP3Mods f64:$x, i32:$mods)))))))),
  (V_CNDMASK_B64_PSEUDO
      (V_MIN_F64
          SRCMODS.NONE,
          (V_FRACT_F64_e64 $mods, $x, DSTCLAMP.NONE, DSTOMOD.NONE),
          SRCMODS.NONE,
          (V_MOV_B64_PSEUDO 0x3fefffffffffffff),
          DSTCLAMP.NONE, DSTOMOD.NONE),
      $x,
      (V_CMP_CLASS_F64_e64 SRCMODS.NONE, $x, 3/*NaN*/))
>;

// Convert floor(x) to (x - fract(x))
def : Pat <
  (f64 (ffloor (f64 (VOP3Mods f64:$x, i32:$mods)))),
  (V_ADD_F64
      $mods,
      $x,
      SRCMODS.NEG,
      (V_CNDMASK_B64_PSEUDO
         (V_MIN_F64
             SRCMODS.NONE,
             (V_FRACT_F64_e64 $mods, $x, DSTCLAMP.NONE, DSTOMOD.NONE),
             SRCMODS.NONE,
             (V_MOV_B64_PSEUDO 0x3fefffffffffffff),
             DSTCLAMP.NONE, DSTOMOD.NONE),
         $x,
         (V_CMP_CLASS_F64_e64 SRCMODS.NONE, $x, 3/*NaN*/)),
      DSTCLAMP.NONE, DSTOMOD.NONE)
>;

} // End Predicates = [isSI]

//============================================================================//
// Miscellaneous Optimization Patterns
//============================================================================//

def : SHA256MaPattern <V_BFI_B32, V_XOR_B32_e64>;

//============================================================================//
// Assembler aliases
//============================================================================//

def : MnemonicAlias<"v_add_u32", "v_add_i32">;
def : MnemonicAlias<"v_sub_u32", "v_sub_i32">;
def : MnemonicAlias<"v_subrev_u32", "v_subrev_i32">;

} // End isGCN predicate
