//===- RISCVInstrInfoVSDPatterns.td - RVV SDNode patterns --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file contains the required infrastructure and SDNode patterns to
/// support code generation for the standard 'V' (Vector) extension, version
/// 0.9.  This version is still experimental as the 'V' extension hasn't been
/// ratified yet.
///
/// This file is included from and depends upon RISCVInstrInfoVPseudos.td
///
/// Note: the patterns for RVV intrinsics are found in
/// RISCVInstrInfoVPseudos.td.
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Helpers to define the SDNode patterns.
//===----------------------------------------------------------------------===//

def SDTSplatI64 : SDTypeProfile<1, 1, [
  SDTCVecEltisVT<0, i64>, SDTCisVT<1, i32>
]>;

def rv32_splat_i64 : SDNode<"RISCVISD::SPLAT_VECTOR_I64", SDTSplatI64>;

// Penalize the generic form with Complexity=1 to give the simm5/uimm5 variants
// precedence
def SplatPat       : ComplexPattern<vAny, 1, "selectVSplat", [], [], 1>;

def SplatPat_simm5 : ComplexPattern<vAny, 1, "selectVSplatSimm5", []>;
def SplatPat_uimm5 : ComplexPattern<vAny, 1, "selectVSplatUimm5", []>;

multiclass VPatUSLoadStoreSDNode<LLVMType type,
                                 LLVMType mask_type,
                                 int sew,
                                 LMULInfo vlmul,
                                 RegisterClass reg_rs1,
                                 VReg reg_class>
{
  defvar load_instr = !cast<Instruction>("PseudoVLE"#sew#"_V_"#vlmul.MX);
  defvar store_instr = !cast<Instruction>("PseudoVSE"#sew#"_V_"#vlmul.MX);
  // Load
  def : Pat<(type (load reg_rs1:$rs1)),
            (load_instr reg_rs1:$rs1, VLMax, sew)>;
  // Store
  def : Pat<(store type:$rs2, reg_rs1:$rs1),
            (store_instr reg_class:$rs2, reg_rs1:$rs1, VLMax, sew)>;
}

multiclass VPatUSLoadStoreSDNodes<RegisterClass reg_rs1> {
  foreach vti = AllVectors in
    defm "" : VPatUSLoadStoreSDNode<vti.Vector, vti.Mask, vti.SEW, vti.LMul,
                                    reg_rs1, vti.RegClass>;
}

class VPatBinarySDNode_VV<SDNode vop,
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
                     (op_type op_reg_class:$rs2))),
        (!cast<Instruction>(instruction_name#"_VV_"# vlmul.MX)
                     op_reg_class:$rs1,
                     op_reg_class:$rs2,
                     VLMax, sew)>;

class VPatBinarySDNode_XI<SDNode vop,
                          string instruction_name,
                          string suffix,
                          ValueType result_type,
                          ValueType vop_type,
                          ValueType xop_type,
                          ValueType mask_type,
                          int sew,
                          LMULInfo vlmul,
                          VReg RetClass,
                          VReg vop_reg_class,
                          ComplexPattern SplatPatKind,
                          DAGOperand xop_kind> :
    Pat<(result_type (vop
                     (vop_type vop_reg_class:$rs1),
                     (vop_type (SplatPatKind xop_kind:$rs2)))),
        (!cast<Instruction>(instruction_name#_#suffix#_# vlmul.MX)
                     vop_reg_class:$rs1,
                     xop_kind:$rs2,
                     VLMax, sew)>;

multiclass VPatBinarySDNode_VV_VX_VI<SDNode vop, string instruction_name,
                                     Operand ImmType = simm5>
{
  foreach vti = AllIntegerVectors in {
    def : VPatBinarySDNode_VV<vop, instruction_name,
                              vti.Vector, vti.Vector, vti.Mask, vti.SEW,
                              vti.LMul, vti.RegClass, vti.RegClass>;
    def : VPatBinarySDNode_XI<vop, instruction_name, "VX",
                              vti.Vector, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                              vti.LMul, vti.RegClass, vti.RegClass,
                              SplatPat, GPR>;
    def : VPatBinarySDNode_XI<vop, instruction_name, "VI",
                              vti.Vector, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                              vti.LMul, vti.RegClass, vti.RegClass,
                              !cast<ComplexPattern>(SplatPat#_#ImmType),
                              ImmType>;
  }
}

//===----------------------------------------------------------------------===//
// Patterns.
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtV] in {

// 7.4. Vector Unit-Stride Instructions
defm "" : VPatUSLoadStoreSDNodes<GPR>;
defm "" : VPatUSLoadStoreSDNodes<AddrFI>;

// 12.1. Vector Single-Width Integer Add and Subtract
defm "" : VPatBinarySDNode_VV_VX_VI<add, "PseudoVADD">;

// 12.5. Vector Bitwise Logical Instructions
defm "" : VPatBinarySDNode_VV_VX_VI<or, "PseudoVOR">;

// 12.6. Vector Single-Width Bit Shift Instructions
defm "" : VPatBinarySDNode_VV_VX_VI<shl, "PseudoVSLL", uimm5>;
defm "" : VPatBinarySDNode_VV_VX_VI<srl, "PseudoVSRL", uimm5>;

} // Predicates = [HasStdExtV]

//===----------------------------------------------------------------------===//
// Vector Splats
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtV] in {
foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (splat_vector GPR:$rs1)),
            (!cast<Instruction>("PseudoVMV_V_X_" # vti.LMul.MX)
              GPR:$rs1, VLMax, vti.SEW)>;
  def : Pat<(vti.Vector (splat_vector simm5:$rs1)),
            (!cast<Instruction>("PseudoVMV_V_I_" # vti.LMul.MX)
              simm5:$rs1, VLMax, vti.SEW)>;
}
} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV, IsRV32] in {
foreach vti = AllIntegerVectors in {
  if !eq(vti.SEW, 64) then {
    def : Pat<(vti.Vector (rv32_splat_i64 GPR:$rs1)),
              (!cast<Instruction>("PseudoVMV_V_X_" # vti.LMul.MX)
                GPR:$rs1, VLMax, vti.SEW)>;
    def : Pat<(vti.Vector (rv32_splat_i64 simm5:$rs1)),
              (!cast<Instruction>("PseudoVMV_V_I_" # vti.LMul.MX)
                simm5:$rs1, VLMax, vti.SEW)>;
  }
}
} // Predicates = [HasStdExtV, IsRV32]
