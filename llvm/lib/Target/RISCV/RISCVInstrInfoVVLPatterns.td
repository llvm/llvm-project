//===- RISCVInstrInfoVVLPatterns.td - RVV VL patterns ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file contains the required infrastructure and VL patterns to
/// support code generation for the standard 'V' (Vector) extension, version
/// 0.10.  This version is still experimental as the 'V' extension hasn't been
/// ratified yet.
///
/// This file is included from and depends upon RISCVInstrInfoVPseudos.td
///
/// Note: the patterns for RVV intrinsics are found in
/// RISCVInstrInfoVPseudos.td.
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Helpers to define the VL patterns.
//===----------------------------------------------------------------------===//

def SDT_RISCVVLE_VL : SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisPtrTy<1>,
                                           SDTCisVT<2, XLenVT>]>;
def SDT_RISCVVSE_VL : SDTypeProfile<0, 3, [SDTCisVec<0>, SDTCisPtrTy<1>,
                                           SDTCisVT<2, XLenVT>]>;

def SDT_RISCVIntBinOp_VL : SDTypeProfile<1, 4, [SDTCisSameAs<0, 1>,
                                                SDTCisSameAs<0, 2>,
                                                SDTCisVec<0>, SDTCisInt<0>,
                                                SDTCVecEltisVT<3, i1>,
                                                SDTCisSameNumEltsAs<0, 3>,
                                                SDTCisVT<4, XLenVT>]>;

def SDT_RISCVFPUnOp_VL : SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>,
                                              SDTCisVec<0>, SDTCisFP<0>,
                                              SDTCVecEltisVT<2, i1>,
                                              SDTCisSameNumEltsAs<0, 2>,
                                              SDTCisVT<3, XLenVT>]>;
def SDT_RISCVFPBinOp_VL : SDTypeProfile<1, 4, [SDTCisSameAs<0, 1>,
                                               SDTCisSameAs<0, 2>,
                                               SDTCisVec<0>, SDTCisFP<0>,
                                               SDTCVecEltisVT<3, i1>,
                                               SDTCisSameNumEltsAs<0, 3>,
                                               SDTCisVT<4, XLenVT>]>;

def riscv_vle_vl : SDNode<"RISCVISD::VLE_VL", SDT_RISCVVLE_VL,
                          [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_vse_vl : SDNode<"RISCVISD::VSE_VL", SDT_RISCVVSE_VL,
                          [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

def riscv_add_vl  : SDNode<"RISCVISD::ADD_VL",  SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
def riscv_sub_vl  : SDNode<"RISCVISD::SUB_VL",  SDT_RISCVIntBinOp_VL>;
def riscv_mul_vl  : SDNode<"RISCVISD::MUL_VL",  SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
def riscv_and_vl  : SDNode<"RISCVISD::AND_VL",  SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
def riscv_or_vl   : SDNode<"RISCVISD::OR_VL",   SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
def riscv_xor_vl  : SDNode<"RISCVISD::XOR_VL",  SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
def riscv_sdiv_vl : SDNode<"RISCVISD::SDIV_VL", SDT_RISCVIntBinOp_VL>;
def riscv_srem_vl : SDNode<"RISCVISD::SREM_VL", SDT_RISCVIntBinOp_VL>;
def riscv_udiv_vl : SDNode<"RISCVISD::UDIV_VL", SDT_RISCVIntBinOp_VL>;
def riscv_urem_vl : SDNode<"RISCVISD::UREM_VL", SDT_RISCVIntBinOp_VL>;
def riscv_shl_vl  : SDNode<"RISCVISD::SHL_VL",  SDT_RISCVIntBinOp_VL>;
def riscv_sra_vl  : SDNode<"RISCVISD::SRA_VL",  SDT_RISCVIntBinOp_VL>;
def riscv_srl_vl  : SDNode<"RISCVISD::SRL_VL",  SDT_RISCVIntBinOp_VL>;
def riscv_fadd_vl : SDNode<"RISCVISD::FADD_VL", SDT_RISCVFPBinOp_VL, [SDNPCommutative]>;
def riscv_fsub_vl : SDNode<"RISCVISD::FSUB_VL", SDT_RISCVFPBinOp_VL>;
def riscv_fmul_vl : SDNode<"RISCVISD::FMUL_VL", SDT_RISCVFPBinOp_VL, [SDNPCommutative]>;
def riscv_fdiv_vl : SDNode<"RISCVISD::FDIV_VL", SDT_RISCVFPBinOp_VL>;
def riscv_fneg_vl : SDNode<"RISCVISD::FNEG_VL", SDT_RISCVFPUnOp_VL>;

def SDT_RISCVVecFMA_VL : SDTypeProfile<1, 5, [SDTCisSameAs<0, 1>,
                                              SDTCisSameAs<0, 2>,
                                              SDTCisSameAs<0, 3>,
                                              SDTCisVec<0>, SDTCisFP<0>,
                                              SDTCVecEltisVT<4, i1>,
                                              SDTCisSameNumEltsAs<0, 4>,
                                              SDTCisVT<5, XLenVT>]>;
def riscv_fma_vl : SDNode<"RISCVISD::FMA_VL", SDT_RISCVVecFMA_VL>;

def SDT_RISCVVMSETCLR_VL : SDTypeProfile<1, 1, [SDTCisVec<0>,
                                                SDTCVecEltisVT<0, i1>,
                                                SDTCisVT<1, XLenVT>]>;
def riscv_vmclr_vl : SDNode<"RISCVISD::VMCLR_VL", SDT_RISCVVMSETCLR_VL>;
def riscv_vmset_vl : SDNode<"RISCVISD::VMSET_VL", SDT_RISCVVMSETCLR_VL>;

def true_mask           : PatLeaf<(riscv_vmset_vl (XLenVT srcvalue))>;

class VPatBinaryVL_VV<SDNode vop,
                      string instruction_name,
                      ValueType result_type,
                      ValueType op_type,
                      ValueType mask_type,
                      int sew,
                      LMULInfo vlmul,
                      VReg RetClass,
                      VReg op_reg_class> :
    Pat<(result_type (vop
                     (op_type op_reg_class:$rs1),
                     (op_type op_reg_class:$rs2),
                     (mask_type true_mask),
                     (XLenVT (VLOp GPR:$vl)))),
        (!cast<Instruction>(instruction_name#"_VV_"# vlmul.MX)
                     op_reg_class:$rs1,
                     op_reg_class:$rs2,
                     GPR:$vl, sew)>;

multiclass VPatBinaryVL_VV_VX<SDNode vop, string instruction_name> {
  foreach vti = AllIntegerVectors in {
    def : VPatBinaryVL_VV<vop, instruction_name,
                          vti.Vector, vti.Vector, vti.Mask, vti.SEW,
                          vti.LMul, vti.RegClass, vti.RegClass>;
    // FIXME: Support splats.
  }
}

multiclass VPatBinaryVL_VV_VX_VI<SDNode vop, string instruction_name,
                                 Operand ImmType = simm5> {
  foreach vti = AllIntegerVectors in {
    def : VPatBinaryVL_VV<vop, instruction_name,
                          vti.Vector, vti.Vector, vti.Mask, vti.SEW,
                          vti.LMul, vti.RegClass, vti.RegClass>;
    // FIXME: Support splats.
  }
}

multiclass VPatBinaryFPVL_VV_VF<SDNode vop, string instruction_name> {
  foreach vti = AllFloatVectors in {
    def : VPatBinaryVL_VV<vop, instruction_name,
                          vti.Vector, vti.Vector, vti.Mask, vti.SEW,
                          vti.LMul, vti.RegClass, vti.RegClass>;
    // FIXME: Support splats.
  }
}

//===----------------------------------------------------------------------===//
// Patterns.
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtV] in {

// 7.4. Vector Unit-Stride Instructions
foreach vti = AllVectors in {
  defvar load_instr = !cast<Instruction>("PseudoVLE"#vti.SEW#"_V_"#vti.LMul.MX);
  defvar store_instr = !cast<Instruction>("PseudoVSE"#vti.SEW#"_V_"#vti.LMul.MX);
  // Load
  def : Pat<(vti.Vector (riscv_vle_vl RVVBaseAddr:$rs1, (XLenVT (VLOp GPR:$vl)))),
            (load_instr RVVBaseAddr:$rs1, GPR:$vl, vti.SEW)>;
  // Store
  def : Pat<(riscv_vse_vl (vti.Vector vti.RegClass:$rs2), RVVBaseAddr:$rs1, (XLenVT (VLOp GPR:$vl))),
            (store_instr vti.RegClass:$rs2, RVVBaseAddr:$rs1, GPR:$vl, vti.SEW)>;
}

// 12.1. Vector Single-Width Integer Add and Subtract
defm "" : VPatBinaryVL_VV_VX_VI<riscv_add_vl, "PseudoVADD">;
defm "" : VPatBinaryVL_VV_VX<riscv_sub_vl, "PseudoVSUB">;

// 12.5. Vector Bitwise Logical Instructions
defm "" : VPatBinaryVL_VV_VX_VI<riscv_and_vl, "PseudoVAND">;
defm "" : VPatBinaryVL_VV_VX_VI<riscv_or_vl,  "PseudoVOR">;
defm "" : VPatBinaryVL_VV_VX_VI<riscv_xor_vl, "PseudoVXOR">;

// 12.6. Vector Single-Width Bit Shift Instructions
defm "" : VPatBinaryVL_VV_VX_VI<riscv_shl_vl, "PseudoVSLL", uimm5>;
defm "" : VPatBinaryVL_VV_VX_VI<riscv_srl_vl, "PseudoVSRL", uimm5>;
defm "" : VPatBinaryVL_VV_VX_VI<riscv_sra_vl, "PseudoVSRA", uimm5>;

// 12.10. Vector Single-Width Integer Multiply Instructions
defm "" : VPatBinaryVL_VV_VX<riscv_mul_vl, "PseudoVMUL">;

// 12.11. Vector Integer Divide Instructions
defm "" : VPatBinaryVL_VV_VX<riscv_udiv_vl, "PseudoVDIVU">;
defm "" : VPatBinaryVL_VV_VX<riscv_sdiv_vl, "PseudoVDIV">;
defm "" : VPatBinaryVL_VV_VX<riscv_urem_vl, "PseudoVREMU">;
defm "" : VPatBinaryVL_VV_VX<riscv_srem_vl, "PseudoVREM">;

} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV, HasStdExtF] in {

// 14.2. Vector Single-Width Floating-Point Add/Subtract Instructions
defm "" : VPatBinaryFPVL_VV_VF<riscv_fadd_vl, "PseudoVFADD">;
defm "" : VPatBinaryFPVL_VV_VF<riscv_fsub_vl, "PseudoVFSUB">;

// 14.4. Vector Single-Width Floating-Point Multiply/Divide Instructions
defm "" : VPatBinaryFPVL_VV_VF<riscv_fmul_vl, "PseudoVFMUL">;
defm "" : VPatBinaryFPVL_VV_VF<riscv_fdiv_vl, "PseudoVFDIV">;

// 14.6 Vector Single-Width Floating-Point Fused Multiply-Add Instructions.
foreach vti = AllFloatVectors in {
  // NOTE: We choose VFMADD because it has the most commuting freedom. So it
  // works best with how TwoAddressInstructionPass tries commuting.
  def : Pat<(vti.Vector (riscv_fma_vl vti.RegClass:$rd, vti.RegClass:$rs1,
                                      vti.RegClass:$rs2, (vti.Mask true_mask),
                                      (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVFMADD_VV_"# vti.LMul.MX)
                 vti.RegClass:$rd, vti.RegClass:$rs1, vti.RegClass:$rs2,
                 GPR:$vl, vti.SEW)>;
}

// 14.12. Vector Floating-Point Sign-Injection Instructions
// Handle fneg with VFSGNJN using the same input for both operands.
foreach vti = AllFloatVectors in {
  def : Pat<(riscv_fneg_vl (vti.Vector vti.RegClass:$rs), (vti.Mask true_mask),
                           (XLenVT (VLOp GPR:$vl))),
            (!cast<Instruction>("PseudoVFSGNJN_VV_"# vti.LMul.MX)
                 vti.RegClass:$rs, vti.RegClass:$rs, GPR:$vl, vti.SEW)>;
}

} // Predicates = [HasStdExtV, HasStdExtF]
