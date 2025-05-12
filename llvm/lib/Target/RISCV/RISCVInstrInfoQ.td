//===-- RISCVInstrInfoQ.td - RISC-V 'Q' instructions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Q',
// Quad-Precision Floating-Point instruction set extension.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

def QExt : ExtInfo<"", "", [HasStdExtQ], f128, FPR128, FPR32, FPR64, ?>;

defvar QExts = [QExt];
defvar QExtsRV64 = [QExt];

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtQ] in {
  let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
  def FLQ : RVInstI<0b100, OPC_LOAD_FP, (outs FPR128:$rd),
                    (ins GPRMem:$rs1, simm12:$imm12), "flq",
                    "$rd, ${imm12}(${rs1})">;
  // Operands for stores are in the order srcreg, base, offset rather than
  // reflecting the order these fields are specified in the instruction
  // encoding.
  let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
  def FSQ : RVInstS<0b100, OPC_STORE_FP, (outs),
                    (ins FPR128:$rs2, GPRMem:$rs1, simm12:$imm12), "fsq",
                    "$rs2, ${imm12}(${rs1})">;
} // Predicates = [HasStdExtQ]

foreach Ext = QExts in {
  defm FMADD_Q : FPFMA_rrr_frm_m<OPC_MADD, 0b11, "fmadd.q", Ext>;
  defm FMSUB_Q : FPFMA_rrr_frm_m<OPC_MSUB, 0b11, "fmsub.q", Ext>;
  defm FNMSUB_Q : FPFMA_rrr_frm_m<OPC_NMSUB, 0b11, "fnmsub.q", Ext>;
  defm FNMADD_Q : FPFMA_rrr_frm_m<OPC_NMADD, 0b11, "fnmadd.q", Ext>;

  defm FADD_Q : FPALU_rr_frm_m<0b0000011, "fadd.q", Ext>;
  defm FSUB_Q : FPALU_rr_frm_m<0b0000111, "fsub.q", Ext>;

  defm FMUL_Q : FPALU_rr_frm_m<0b0001011, "fmul.q", Ext>;

  defm FDIV_Q : FPALU_rr_frm_m<0b0001111, "fdiv.q", Ext>;

  defm FSQRT_Q : FPUnaryOp_r_frm_m<0b0101111, 0b00000, Ext, Ext.PrimaryTy,
                                   Ext.PrimaryTy, "fsqrt.q">;

  let mayRaiseFPException = 0 in {
    defm FSGNJ_Q : FPALU_rr_m<0b0010011, 0b000, "fsgnj.q", Ext>;
    defm FSGNJN_Q : FPALU_rr_m<0b0010011, 0b001, "fsgnjn.q", Ext>;
    defm FSGNJX_Q : FPALU_rr_m<0b0010011, 0b010, "fsgnjx.q", Ext>;
  }

  defm FMIN_Q : FPALU_rr_m<0b0010111, 0b000, "fmin.q", Ext, Commutable = 1>;
  defm FMAX_Q : FPALU_rr_m<0b0010111, 0b001, "fmax.q", Ext, Commutable = 1>;

  defm FCVT_S_Q : FPUnaryOp_r_frm_m<0b0100000, 0b00011, Ext, Ext.F32Ty,
                                    Ext.PrimaryTy, "fcvt.s.q">;

  defm FCVT_Q_S : FPUnaryOp_r_frmlegacy_m<0b0100011, 0b00000, Ext,
                                          Ext.PrimaryTy, Ext.F32Ty, "fcvt.q.s">;

  defm FCVT_D_Q : FPUnaryOp_r_frm_m<0b0100001, 0b00011, Ext, Ext.F64Ty,
                                    Ext.PrimaryTy, "fcvt.d.q">;

  defm FCVT_Q_D : FPUnaryOp_r_frmlegacy_m<0b0100011, 0b00001, Ext,
                                          Ext.PrimaryTy, Ext.F64Ty, "fcvt.q.d">;

  defm FEQ_Q : FPCmp_rr_m<0b1010011, 0b010, "feq.q", Ext, Commutable = 1>;
  defm FLT_Q : FPCmp_rr_m<0b1010011, 0b001, "flt.q", Ext>;
  defm FLE_Q : FPCmp_rr_m<0b1010011, 0b000, "fle.q", Ext>;

  let mayRaiseFPException = 0 in 
  defm FCLASS_Q : FPUnaryOp_r_m<0b1110011, 0b00000, 0b001, Ext, GPR,
                                Ext.PrimaryTy, "fclass.q">;

  let IsSignExtendingOpW = 1 in 
  defm FCVT_W_Q  : FPUnaryOp_r_frm_m<0b1100011, 0b00000, Ext, GPR,
                                     Ext.PrimaryTy, "fcvt.w.q">;

  let IsSignExtendingOpW = 1 in 
  defm FCVT_WU_Q : FPUnaryOp_r_frm_m<0b1100011, 0b00001, Ext, GPR,
                                     Ext.PrimaryTy, "fcvt.wu.q">;

  let mayRaiseFPException = 0 in 
  defm FCVT_Q_W  : FPUnaryOp_r_frmlegacy_m<0b1101011, 0b00000, Ext,
                                           Ext.PrimaryTy, GPR, "fcvt.q.w">;

  let mayRaiseFPException = 0 in 
  defm FCVT_Q_WU : FPUnaryOp_r_frmlegacy_m<0b1101011, 0b00001, Ext,
                                           Ext.PrimaryTy, GPR, "fcvt.q.wu">;
} // foreach Ext = QExts

foreach Ext = QExtsRV64 in {
  defm FCVT_L_Q  : FPUnaryOp_r_frm_m<0b1100011, 0b00010, Ext, GPR, 
                                     Ext.PrimaryTy, "fcvt.l.q", [IsRV64]>;

  defm FCVT_LU_Q : FPUnaryOp_r_frm_m<0b1100011, 0b00011, Ext, GPR,
                                     Ext.PrimaryTy, "fcvt.lu.q", [IsRV64]>;

  let mayRaiseFPException = 0 in 
  defm FCVT_Q_L : FPUnaryOp_r_frmlegacy_m<0b1101011, 0b00010, Ext,
                                          Ext.PrimaryTy, GPR, "fcvt.q.l",
                                          [IsRV64]>;

  let mayRaiseFPException = 0 in 
  defm FCVT_Q_LU : FPUnaryOp_r_frmlegacy_m<0b1101011, 0b00011, Ext,
                                           Ext.PrimaryTy, GPR, "fcvt.q.lu",
                                           [IsRV64]>;
} // foreach Ext = QExtsRV64

//===----------------------------------------------------------------------===//
// Assembler Pseudo Instructions (User-Level ISA, Version 2.2, Chapter 20)
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtQ] in {
  def : InstAlias<"flq $rd, (${rs1})",  (FLQ FPR128:$rd,  GPR:$rs1, 0), 0>;
  def : InstAlias<"fsq $rs2, (${rs1})", (FSQ FPR128:$rs2, GPR:$rs1, 0), 0>;

  def : InstAlias<"fmv.q $rd, $rs",  (FSGNJ_Q FPR128:$rd, FPR128:$rs,
                                      FPR128:$rs)>;
  def : InstAlias<"fabs.q $rd, $rs", (FSGNJX_Q FPR128:$rd, FPR128:$rs,
                                      FPR128:$rs)>;
  def : InstAlias<"fneg.q $rd, $rs", (FSGNJN_Q FPR128:$rd, FPR128:$rs,
                                      FPR128:$rs)>;

  // fgt.q/fge.q are recognised by the GNU assembler but the canonical
  // flt.q/fle.q forms will always be printed. Therefore, set a zero weight.
  def : InstAlias<"fgt.q $rd, $rs, $rt",
                  (FLT_Q GPR:$rd, FPR128:$rt, FPR128:$rs), 0>;
  def : InstAlias<"fge.q $rd, $rs, $rt",
                  (FLE_Q GPR:$rd, FPR128:$rt, FPR128:$rs), 0>;

  def PseudoFLQ : PseudoFloatLoad<"flq", FPR128>;
  def PseudoFSQ : PseudoStore<"fsq", FPR128>;
} // Predicates = [HasStdExtQ]
