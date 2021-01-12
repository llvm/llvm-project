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

class SwapHelper<dag Prefix, dag A, dag B, dag Suffix, bit swap> {
   dag Value = !con(Prefix, !if(swap, B, A), !if(swap, A, B), Suffix);
}

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

multiclass VPatBinarySDNode_VV_VX<SDNode vop, string instruction_name>
{
  foreach vti = AllIntegerVectors in {
    def : VPatBinarySDNode_VV<vop, instruction_name,
                              vti.Vector, vti.Vector, vti.Mask, vti.SEW,
                              vti.LMul, vti.RegClass, vti.RegClass>;
    def : VPatBinarySDNode_XI<vop, instruction_name, "VX",
                              vti.Vector, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                              vti.LMul, vti.RegClass, vti.RegClass,
                              SplatPat, GPR>;
  }
}

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

class VPatBinarySDNode_VF<SDNode vop,
                          string instruction_name,
                          ValueType result_type,
                          ValueType vop_type,
                          ValueType xop_type,
                          ValueType mask_type,
                          int sew,
                          LMULInfo vlmul,
                          VReg RetClass,
                          VReg vop_reg_class,
                          DAGOperand xop_kind> :
    Pat<(result_type (vop (vop_type vop_reg_class:$rs1),
                          (vop_type (splat_vector xop_kind:$rs2)))),
        (!cast<Instruction>(instruction_name#"_VF_"#vlmul.MX)
                     vop_reg_class:$rs1,
                     ToFPR32<xop_type, xop_kind, "rs2">.ret,
                     VLMax, sew)>;

multiclass VPatBinaryFPSDNode_VV_VF<SDNode vop, string instruction_name> {
  foreach vti = AllFloatVectors in {
    def : VPatBinarySDNode_VV<vop, instruction_name,
                              vti.Vector, vti.Vector, vti.Mask, vti.SEW,
                              vti.LMul, vti.RegClass, vti.RegClass>;
    def : VPatBinarySDNode_VF<vop, instruction_name,
                              vti.Vector, vti.Vector, vti.Scalar, vti.Mask,
                              vti.SEW, vti.LMul, vti.RegClass, vti.RegClass,
                              vti.ScalarRegClass>;
  }
}

multiclass VPatBinaryFPSDNode_R_VF<SDNode vop, string instruction_name> {
  foreach fvti = AllFloatVectors in
    def : Pat<(fvti.Vector (vop (fvti.Vector (splat_vector fvti.Scalar:$rs2)),
                                (fvti.Vector fvti.RegClass:$rs1))),
              (!cast<Instruction>(instruction_name#"_VF_"#fvti.LMul.MX)
                           fvti.RegClass:$rs1,
                           ToFPR32<fvti.Scalar, fvti.ScalarRegClass, "rs2">.ret,
                           VLMax, fvti.SEW)>;
}

multiclass VPatIntegerSetCCSDNode_VV<CondCode cc,
                                     string instruction_name,
                                     bit swap = 0> {
  foreach vti = AllIntegerVectors in {
    defvar instruction = !cast<Instruction>(instruction_name#"_VV_"#vti.LMul.MX);
    def : Pat<(vti.Mask (setcc (vti.Vector vti.RegClass:$rs1),
                               (vti.Vector vti.RegClass:$rs2), cc)),
              SwapHelper<(instruction),
                         (instruction vti.RegClass:$rs1),
                         (instruction vti.RegClass:$rs2),
                         (instruction VLMax, vti.SEW),
                         swap>.Value>;
  }
}

multiclass VPatIntegerSetCCSDNode_XI<CondCode cc,
                                     string instruction_name,
                                     string kind,
                                     ComplexPattern SplatPatKind,
                                     DAGOperand xop_kind,
                                     bit swap = 0> {
  foreach vti = AllIntegerVectors in {
    defvar instruction = !cast<Instruction>(instruction_name#_#kind#_#vti.LMul.MX);
    def : Pat<(vti.Mask (setcc (vti.Vector vti.RegClass:$rs1),
                               (vti.Vector (SplatPatKind xop_kind:$rs2)), cc)),
              SwapHelper<(instruction),
                         (instruction vti.RegClass:$rs1),
                         (instruction xop_kind:$rs2),
                         (instruction VLMax, vti.SEW),
                         swap>.Value>;
  }
}

multiclass VPatIntegerSetCCSDNode_VV_VX_VI<CondCode cc,
                                           string instruction_name,
                                           bit swap = 0> {
  defm : VPatIntegerSetCCSDNode_VV<cc, instruction_name, swap>;
  defm : VPatIntegerSetCCSDNode_XI<cc, instruction_name, "VX",
                                   SplatPat, GPR, swap>;
  defm : VPatIntegerSetCCSDNode_XI<cc, instruction_name, "VI",
                                   SplatPat_simm5, simm5, swap>;
}

multiclass VPatIntegerSetCCSDNode_VV_VX<CondCode cc,
                                        string instruction_name,
                                        bit swap = 0> {
  defm : VPatIntegerSetCCSDNode_VV<cc, instruction_name, swap>;
  defm : VPatIntegerSetCCSDNode_XI<cc, instruction_name, "VX",
                                   SplatPat, GPR, swap>;
}

multiclass VPatIntegerSetCCSDNode_VX_VI<CondCode cc,
                                        string instruction_name,
                                        bit swap = 0> {
  defm : VPatIntegerSetCCSDNode_XI<cc, instruction_name, "VX",
                                   SplatPat, GPR, swap>;
  defm : VPatIntegerSetCCSDNode_XI<cc, instruction_name, "VI",
                                   SplatPat_simm5, simm5, swap>;
}

multiclass VPatFPSetCCSDNode_VV<CondCode cc, string instruction_name> {
  foreach fvti = AllFloatVectors in
    def : Pat<(fvti.Mask (setcc (fvti.Vector fvti.RegClass:$rs1),
                                (fvti.Vector fvti.RegClass:$rs2),
                                cc)),
              (!cast<Instruction>(instruction_name#"_VV_"#fvti.LMul.MX)
                  fvti.RegClass:$rs1, fvti.RegClass:$rs2, VLMax, fvti.SEW)>;
}

multiclass VPatFPSetCCSDNode_VF<CondCode cc, string instruction_name> {
  foreach fvti = AllFloatVectors in
    def : Pat<(fvti.Mask (setcc (fvti.Vector fvti.RegClass:$rs1),
                                (fvti.Vector (splat_vector fvti.ScalarRegClass:$rs2)),
                                cc)),
              (!cast<Instruction>(instruction_name#"_VF_"#fvti.LMul.MX)
                  fvti.RegClass:$rs1,
                  ToFPR32<fvti.Scalar, fvti.ScalarRegClass, "rs2">.ret,
                  VLMax, fvti.SEW)>;
}

multiclass VPatFPSetCCSDNode_FV<CondCode cc, string swapped_op_instruction_name> {
  foreach fvti = AllFloatVectors in
    def : Pat<(fvti.Mask (setcc (fvti.Vector (splat_vector fvti.ScalarRegClass:$rs2)),
                                (fvti.Vector fvti.RegClass:$rs1),
                                cc)),
              (!cast<Instruction>(swapped_op_instruction_name#"_VF_"#fvti.LMul.MX)
                  fvti.RegClass:$rs1,
                  ToFPR32<fvti.Scalar, fvti.ScalarRegClass, "rs2">.ret,
                  VLMax, fvti.SEW)>;
}

multiclass VPatFPSetCCSDNode_VV_VF_FV<CondCode cc,
                                      string inst_name,
                                      string swapped_op_inst_name> {
  defm : VPatFPSetCCSDNode_VV<cc, inst_name>;
  defm : VPatFPSetCCSDNode_VF<cc, inst_name>;
  defm : VPatFPSetCCSDNode_FV<cc, swapped_op_inst_name>;
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
defm "" : VPatBinarySDNode_VV_VX<sub, "PseudoVSUB">;
// Handle VRSUB specially since it's the only integer binary op with reversed
// pattern operands
foreach vti = AllIntegerVectors in {
  def : Pat<(sub (vti.Vector (SplatPat XLenVT:$rs2)),
                 (vti.Vector vti.RegClass:$rs1)),
            (!cast<Instruction>("PseudoVRSUB_VX_"# vti.LMul.MX)
                 vti.RegClass:$rs1, GPR:$rs2, VLMax, vti.SEW)>;
  def : Pat<(sub (vti.Vector (SplatPat_simm5 XLenVT:$rs2)),
                 (vti.Vector vti.RegClass:$rs1)),
            (!cast<Instruction>("PseudoVRSUB_VI_"# vti.LMul.MX)
                 vti.RegClass:$rs1, simm5:$rs2, VLMax, vti.SEW)>;
}

// 12.5. Vector Bitwise Logical Instructions
defm "" : VPatBinarySDNode_VV_VX_VI<and, "PseudoVAND">;
defm "" : VPatBinarySDNode_VV_VX_VI<or, "PseudoVOR">;
defm "" : VPatBinarySDNode_VV_VX_VI<xor, "PseudoVXOR">;

// 12.6. Vector Single-Width Bit Shift Instructions
defm "" : VPatBinarySDNode_VV_VX_VI<shl, "PseudoVSLL", uimm5>;
defm "" : VPatBinarySDNode_VV_VX_VI<srl, "PseudoVSRL", uimm5>;
defm "" : VPatBinarySDNode_VV_VX_VI<sra, "PseudoVSRA", uimm5>;

// 12.8. Vector Integer Comparison Instructions
defm "" : VPatIntegerSetCCSDNode_VV_VX_VI<SETEQ,  "PseudoVMSEQ">;
defm "" : VPatIntegerSetCCSDNode_VV_VX_VI<SETNE,  "PseudoVMSNE">;

// FIXME: Support immediate forms of these by choosing SLE decrementing the
// immediate
defm "" : VPatIntegerSetCCSDNode_VV_VX<SETLT,  "PseudoVMSLT">;
defm "" : VPatIntegerSetCCSDNode_VV_VX<SETULT, "PseudoVMSLTU">;

defm "" : VPatIntegerSetCCSDNode_VV<SETGT,  "PseudoVMSLT", /*swap*/1>;
defm "" : VPatIntegerSetCCSDNode_VV<SETUGT, "PseudoVMSLTU", /*swap*/1>;
defm "" : VPatIntegerSetCCSDNode_VX_VI<SETGT,  "PseudoVMSGT">;
defm "" : VPatIntegerSetCCSDNode_VX_VI<SETUGT, "PseudoVMSGTU">;

defm "" : VPatIntegerSetCCSDNode_VV_VX_VI<SETLE,  "PseudoVMSLE">;
defm "" : VPatIntegerSetCCSDNode_VV_VX_VI<SETULE, "PseudoVMSLEU">;

// FIXME: Support immediate forms of these by choosing SGT and decrementing the
// immediate
defm "" : VPatIntegerSetCCSDNode_VV<SETGE,  "PseudoVMSLE", /*swap*/1>;
defm "" : VPatIntegerSetCCSDNode_VV<SETUGE, "PseudoVMSLEU", /*swap*/1>;

// 12.9. Vector Integer Min/Max Instructions
defm "" : VPatBinarySDNode_VV_VX<umin, "PseudoVMINU">;
defm "" : VPatBinarySDNode_VV_VX<smin, "PseudoVMIN">;
defm "" : VPatBinarySDNode_VV_VX<umax, "PseudoVMAXU">;
defm "" : VPatBinarySDNode_VV_VX<smax, "PseudoVMAX">;

// 12.10. Vector Single-Width Integer Multiply Instructions
defm "" : VPatBinarySDNode_VV_VX<mul, "PseudoVMUL">;
defm "" : VPatBinarySDNode_VV_VX<mulhs, "PseudoVMULH">;
defm "" : VPatBinarySDNode_VV_VX<mulhu, "PseudoVMULHU">;

// 12.11. Vector Integer Divide Instructions
defm "" : VPatBinarySDNode_VV_VX<sdiv, "PseudoVDIVU">;
defm "" : VPatBinarySDNode_VV_VX<udiv, "PseudoVDIV">;
defm "" : VPatBinarySDNode_VV_VX<urem, "PseudoVREMU">;
defm "" : VPatBinarySDNode_VV_VX<srem, "PseudoVREM">;

// 12.16. Vector Integer Merge Instructions
foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (vselect (vti.Mask VMV0:$vm), vti.RegClass:$rs1,
                                                      vti.RegClass:$rs2)),
            (!cast<Instruction>("PseudoVMERGE_VVM_"#vti.LMul.MX)
                 vti.RegClass:$rs2, vti.RegClass:$rs1, VMV0:$vm, VLMax, vti.SEW)>;

  def : Pat<(vti.Vector (vselect (vti.Mask VMV0:$vm), (SplatPat XLenVT:$rs1),
                                                      vti.RegClass:$rs2)),
            (!cast<Instruction>("PseudoVMERGE_VXM_"#vti.LMul.MX)
                 vti.RegClass:$rs2, GPR:$rs1, VMV0:$vm, VLMax, vti.SEW)>;

  def : Pat<(vti.Vector (vselect (vti.Mask VMV0:$vm), (SplatPat_simm5 simm5:$rs1),
                                                      vti.RegClass:$rs2)),
            (!cast<Instruction>("PseudoVMERGE_VIM_"#vti.LMul.MX)
                 vti.RegClass:$rs2, simm5:$rs1, VMV0:$vm, VLMax, vti.SEW)>;
}

// 16.1. Vector Mask-Register Logical Instructions
foreach mti = AllMasks in {
  def : Pat<(mti.Mask (and VR:$rs1, VR:$rs2)),
            (!cast<Instruction>("PseudoVMAND_MM_"#mti.LMul.MX)
                 VR:$rs1, VR:$rs2, VLMax, mti.SEW)>;
  def : Pat<(mti.Mask (or VR:$rs1, VR:$rs2)),
            (!cast<Instruction>("PseudoVMOR_MM_"#mti.LMul.MX)
                 VR:$rs1, VR:$rs2, VLMax, mti.SEW)>;
  def : Pat<(mti.Mask (xor VR:$rs1, VR:$rs2)),
            (!cast<Instruction>("PseudoVMXOR_MM_"#mti.LMul.MX)
                 VR:$rs1, VR:$rs2, VLMax, mti.SEW)>;

  def : Pat<(mti.Mask (vnot (and VR:$rs1, VR:$rs2))),
            (!cast<Instruction>("PseudoVMNAND_MM_"#mti.LMul.MX)
                 VR:$rs1, VR:$rs2, VLMax, mti.SEW)>;
  def : Pat<(mti.Mask (vnot (or VR:$rs1, VR:$rs2))),
            (!cast<Instruction>("PseudoVMNOR_MM_"#mti.LMul.MX)
                 VR:$rs1, VR:$rs2, VLMax, mti.SEW)>;
  def : Pat<(mti.Mask (vnot (xor VR:$rs1, VR:$rs2))),
            (!cast<Instruction>("PseudoVMXNOR_MM_"#mti.LMul.MX)
                 VR:$rs1, VR:$rs2, VLMax, mti.SEW)>;

  def : Pat<(mti.Mask (and VR:$rs1, (vnot VR:$rs2))),
            (!cast<Instruction>("PseudoVMANDNOT_MM_"#mti.LMul.MX)
                 VR:$rs1, VR:$rs2, VLMax, mti.SEW)>;
  def : Pat<(mti.Mask (or VR:$rs1, (vnot VR:$rs2))),
            (!cast<Instruction>("PseudoVMORNOT_MM_"#mti.LMul.MX)
                 VR:$rs1, VR:$rs2, VLMax, mti.SEW)>;
}

} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV, HasStdExtF] in {

// 14.2. Vector Single-Width Floating-Point Add/Subtract Instructions
defm "" : VPatBinaryFPSDNode_VV_VF<fadd, "PseudoVFADD">;
defm "" : VPatBinaryFPSDNode_VV_VF<fsub, "PseudoVFSUB">;
defm "" : VPatBinaryFPSDNode_R_VF<fsub, "PseudoVFRSUB">;

// 14.4. Vector Single-Width Floating-Point Multiply/Divide Instructions
defm "" : VPatBinaryFPSDNode_VV_VF<fmul, "PseudoVFMUL">;
defm "" : VPatBinaryFPSDNode_VV_VF<fdiv, "PseudoVFDIV">;
defm "" : VPatBinaryFPSDNode_R_VF<fdiv, "PseudoVFRDIV">;

// 14.11. Vector Floating-Point Compare Instructions
defm "" : VPatFPSetCCSDNode_VV_VF_FV<SETEQ,  "PseudoVMFEQ", "PseudoVMFEQ">;
defm "" : VPatFPSetCCSDNode_VV_VF_FV<SETOEQ, "PseudoVMFEQ", "PseudoVMFEQ">;

defm "" : VPatFPSetCCSDNode_VV_VF_FV<SETNE,  "PseudoVMFNE", "PseudoVMFNE">;
defm "" : VPatFPSetCCSDNode_VV_VF_FV<SETUNE, "PseudoVMFNE", "PseudoVMFNE">;

defm "" : VPatFPSetCCSDNode_VV_VF_FV<SETLT,  "PseudoVMFLT", "PseudoVMFGT">;
defm "" : VPatFPSetCCSDNode_VV_VF_FV<SETOLT, "PseudoVMFLT", "PseudoVMFGT">;

defm "" : VPatFPSetCCSDNode_VV_VF_FV<SETLE,  "PseudoVMFLE", "PseudoVMFGE">;
defm "" : VPatFPSetCCSDNode_VV_VF_FV<SETOLE, "PseudoVMFLE", "PseudoVMFGE">;

// Floating-point vselects:
// 12.16. Vector Integer Merge Instructions
// 14.13. Vector Floating-Point Merge Instruction
foreach fvti = AllFloatVectors in {
  def : Pat<(fvti.Vector (vselect (fvti.Mask VMV0:$vm), fvti.RegClass:$rs1,
                                                        fvti.RegClass:$rs2)),
            (!cast<Instruction>("PseudoVMERGE_VVM_"#fvti.LMul.MX)
                 fvti.RegClass:$rs2, fvti.RegClass:$rs1, VMV0:$vm,
                 VLMax, fvti.SEW)>;

  def : Pat<(fvti.Vector (vselect (fvti.Mask VMV0:$vm),
                                  (splat_vector fvti.ScalarRegClass:$rs1),
                                  fvti.RegClass:$rs2)),
            (!cast<Instruction>("PseudoVFMERGE_VFM_"#fvti.LMul.MX)
                 fvti.RegClass:$rs2,
                 ToFPR32<fvti.Scalar, fvti.ScalarRegClass, "rs1">.ret,
                 VMV0:$vm, VLMax, fvti.SEW)>;

  def : Pat<(fvti.Vector (vselect (fvti.Mask VMV0:$vm),
                                  (splat_vector (fvti.Scalar fpimm0)),
                                  fvti.RegClass:$rs2)),
            (!cast<Instruction>("PseudoVMERGE_VIM_"#fvti.LMul.MX)
                 fvti.RegClass:$rs2, 0, VMV0:$vm, VLMax, fvti.SEW)>;
}
} // Predicates = [HasStdExtV, HasStdExtF]

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

foreach mti = AllMasks in {
  def : Pat<(mti.Mask immAllOnesV),
            (!cast<Instruction>("PseudoVMSET_M_"#mti.BX) VLMax, mti.SEW)>;
  def : Pat<(mti.Mask immAllZerosV),
            (!cast<Instruction>("PseudoVMCLR_M_"#mti.BX) VLMax, mti.SEW)>;
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

let Predicates = [HasStdExtV, HasStdExtF] in {
foreach fvti = AllFloatVectors in {
  def : Pat<(fvti.Vector (splat_vector fvti.ScalarRegClass:$rs1)),
            (!cast<Instruction>("PseudoVFMV_V_F_"#fvti.LMul.MX)
              ToFPR32<fvti.Scalar, fvti.ScalarRegClass, "rs1">.ret,
              VLMax, fvti.SEW)>;

  def : Pat<(fvti.Vector (splat_vector (fvti.Scalar fpimm0))),
            (!cast<Instruction>("PseudoVMV_V_I_"#fvti.LMul.MX)
              0, VLMax, fvti.SEW)>;
}
} // Predicates = [HasStdExtV, HasStdExtF]
