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
// Helpers to define the SDNode patterns.
//===----------------------------------------------------------------------===//

def SDTSplatI64 : SDTypeProfile<1, 1, [
  SDTCVecEltisVT<0, i64>, SDTCisVT<1, i32>
]>;

def rv32_splat_i64 : SDNode<"RISCVISD::SPLAT_VECTOR_I64", SDTSplatI64>;

def riscv_trunc_vector : SDNode<"RISCVISD::TRUNCATE_VECTOR",
                                SDTypeProfile<1, 1,
                                 [SDTCisVec<0>, SDTCisVec<1>]>>;

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
                                 OutPatFrag avl,
                                 RegisterClass reg_rs1,
                                 VReg reg_class>
{
  defvar load_instr = !cast<Instruction>("PseudoVLE"#sew#"_V_"#vlmul.MX);
  defvar store_instr = !cast<Instruction>("PseudoVSE"#sew#"_V_"#vlmul.MX);
  // Load
  def : Pat<(type (load reg_rs1:$rs1)),
            (load_instr reg_rs1:$rs1, avl, sew)>;
  // Store
  def : Pat<(store type:$rs2, reg_rs1:$rs1),
            (store_instr reg_class:$rs2, reg_rs1:$rs1, avl, sew)>;
}

multiclass VPatUSLoadStoreSDNodes<RegisterClass reg_rs1> {
  foreach vti = AllVectors in
    defm "" : VPatUSLoadStoreSDNode<vti.Vector, vti.Mask, vti.SEW, vti.LMul,
                                    vti.AVL, reg_rs1, vti.RegClass>;
}

class VPatBinarySDNode_VV<SDNode vop,
                          string instruction_name,
                          ValueType result_type,
                          ValueType op_type,
                          ValueType mask_type,
                          int sew,
                          LMULInfo vlmul,
                          OutPatFrag avl,
                          VReg RetClass,
                          VReg op_reg_class> :
    Pat<(result_type (vop
                     (op_type op_reg_class:$rs1),
                     (op_type op_reg_class:$rs2))),
        (!cast<Instruction>(instruction_name#"_VV_"# vlmul.MX)
                     op_reg_class:$rs1,
                     op_reg_class:$rs2,
                     avl, sew)>;

class VPatBinarySDNode_XI<SDNode vop,
                          string instruction_name,
                          string suffix,
                          ValueType result_type,
                          ValueType vop_type,
                          ValueType xop_type,
                          ValueType mask_type,
                          int sew,
                          LMULInfo vlmul,
                          OutPatFrag avl,
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
                     avl, sew)>;

multiclass VPatBinarySDNode_VV_VX<SDNode vop, string instruction_name>
{
  foreach vti = AllIntegerVectors in {
    def : VPatBinarySDNode_VV<vop, instruction_name,
                              vti.Vector, vti.Vector, vti.Mask, vti.SEW,
                              vti.LMul, vti.AVL, vti.RegClass, vti.RegClass>;
    def : VPatBinarySDNode_XI<vop, instruction_name, "VX",
                              vti.Vector, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                              vti.LMul, vti.AVL, vti.RegClass, vti.RegClass,
                              SplatPat, GPR>;
  }
}

multiclass VPatBinarySDNode_VV_VX_VI<SDNode vop, string instruction_name,
                                     Operand ImmType = simm5>
{
  foreach vti = AllIntegerVectors in {
    def : VPatBinarySDNode_VV<vop, instruction_name,
                              vti.Vector, vti.Vector, vti.Mask, vti.SEW,
                              vti.LMul, vti.AVL, vti.RegClass, vti.RegClass>;
    def : VPatBinarySDNode_XI<vop, instruction_name, "VX",
                              vti.Vector, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                              vti.LMul, vti.AVL, vti.RegClass, vti.RegClass,
                              SplatPat, GPR>;
    def : VPatBinarySDNode_XI<vop, instruction_name, "VI",
                              vti.Vector, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                              vti.LMul, vti.AVL, vti.RegClass, vti.RegClass,
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
                          OutPatFrag avl,
                          VReg RetClass,
                          VReg vop_reg_class,
                          DAGOperand xop_kind> :
    Pat<(result_type (vop (vop_type vop_reg_class:$rs1),
                          (vop_type (splat_vector xop_kind:$rs2)))),
        (!cast<Instruction>(instruction_name#"_"#vlmul.MX)
                     vop_reg_class:$rs1,
                     (xop_type xop_kind:$rs2),
                     avl, sew)>;

multiclass VPatBinaryFPSDNode_VV_VF<SDNode vop, string instruction_name> {
  foreach vti = AllFloatVectors in {
    def : VPatBinarySDNode_VV<vop, instruction_name,
                              vti.Vector, vti.Vector, vti.Mask, vti.SEW,
                              vti.LMul, vti.AVL, vti.RegClass, vti.RegClass>;
    def : VPatBinarySDNode_VF<vop, instruction_name#"_V"#vti.ScalarSuffix,
                              vti.Vector, vti.Vector, vti.Scalar, vti.Mask,
                              vti.SEW, vti.LMul, vti.AVL, vti.RegClass, vti.RegClass,
                              vti.ScalarRegClass>;
  }
}

multiclass VPatBinaryFPSDNode_R_VF<SDNode vop, string instruction_name> {
  foreach fvti = AllFloatVectors in
    def : Pat<(fvti.Vector (vop (fvti.Vector (splat_vector fvti.Scalar:$rs2)),
                                (fvti.Vector fvti.RegClass:$rs1))),
              (!cast<Instruction>(instruction_name#"_V"#fvti.ScalarSuffix#"_"#fvti.LMul.MX)
                           fvti.RegClass:$rs1,
                           (fvti.Scalar fvti.ScalarRegClass:$rs2),
                           fvti.AVL, fvti.SEW)>;
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
                         (instruction vti.AVL, vti.SEW),
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
                         (instruction vti.AVL, vti.SEW),
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
                  fvti.RegClass:$rs1, fvti.RegClass:$rs2, fvti.AVL, fvti.SEW)>;
}

multiclass VPatFPSetCCSDNode_VF<CondCode cc, string instruction_name> {
  foreach fvti = AllFloatVectors in
    def : Pat<(fvti.Mask (setcc (fvti.Vector fvti.RegClass:$rs1),
                                (fvti.Vector (splat_vector fvti.ScalarRegClass:$rs2)),
                                cc)),
              (!cast<Instruction>(instruction_name#"_V"#fvti.ScalarSuffix#"_"#fvti.LMul.MX)
                  fvti.RegClass:$rs1,
                  (fvti.Scalar fvti.ScalarRegClass:$rs2),
                  fvti.AVL, fvti.SEW)>;
}

multiclass VPatFPSetCCSDNode_FV<CondCode cc, string swapped_op_instruction_name> {
  foreach fvti = AllFloatVectors in
    def : Pat<(fvti.Mask (setcc (fvti.Vector (splat_vector fvti.ScalarRegClass:$rs2)),
                                (fvti.Vector fvti.RegClass:$rs1),
                                cc)),
              (!cast<Instruction>(swapped_op_instruction_name#"_V"#fvti.ScalarSuffix#"_"#fvti.LMul.MX)
                  fvti.RegClass:$rs1,
                  (fvti.Scalar fvti.ScalarRegClass:$rs2),
                  fvti.AVL, fvti.SEW)>;
}

multiclass VPatFPSetCCSDNode_VV_VF_FV<CondCode cc,
                                      string inst_name,
                                      string swapped_op_inst_name> {
  defm : VPatFPSetCCSDNode_VV<cc, inst_name>;
  defm : VPatFPSetCCSDNode_VF<cc, inst_name>;
  defm : VPatFPSetCCSDNode_FV<cc, swapped_op_inst_name>;
}

multiclass VPatExtendSDNode_V<list<SDNode> ops, string inst_name, string suffix,
                              list <VTypeInfoToFraction> fraction_list> {
  foreach vtiTofti = fraction_list in {
    defvar vti = vtiTofti.Vti;
    defvar fti = vtiTofti.Fti;
    foreach op = ops in
      def : Pat<(vti.Vector (op (fti.Vector fti.RegClass:$rs2))),
                (!cast<Instruction>(inst_name#"_"#suffix#"_"#vti.LMul.MX)
                    fti.RegClass:$rs2, fti.AVL, vti.SEW)>;
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
defm "" : VPatBinarySDNode_VV_VX<sub, "PseudoVSUB">;
// Handle VRSUB specially since it's the only integer binary op with reversed
// pattern operands
foreach vti = AllIntegerVectors in {
  def : Pat<(sub (vti.Vector (SplatPat XLenVT:$rs2)),
                 (vti.Vector vti.RegClass:$rs1)),
            (!cast<Instruction>("PseudoVRSUB_VX_"# vti.LMul.MX)
                 vti.RegClass:$rs1, GPR:$rs2, vti.AVL, vti.SEW)>;
  def : Pat<(sub (vti.Vector (SplatPat_simm5 XLenVT:$rs2)),
                 (vti.Vector vti.RegClass:$rs1)),
            (!cast<Instruction>("PseudoVRSUB_VI_"# vti.LMul.MX)
                 vti.RegClass:$rs1, simm5:$rs2, vti.AVL, vti.SEW)>;
}

// 12.3. Vector Integer Extension
defm "" : VPatExtendSDNode_V<[zext, anyext], "PseudoVZEXT", "VF2",
                             AllFractionableVF2IntVectors>;
defm "" : VPatExtendSDNode_V<[sext],         "PseudoVSEXT", "VF2",
                             AllFractionableVF2IntVectors>;
defm "" : VPatExtendSDNode_V<[zext, anyext], "PseudoVZEXT", "VF4",
                             AllFractionableVF4IntVectors>;
defm "" : VPatExtendSDNode_V<[sext],         "PseudoVSEXT", "VF4",
                             AllFractionableVF4IntVectors>;
defm "" : VPatExtendSDNode_V<[zext, anyext], "PseudoVZEXT", "VF8",
                             AllFractionableVF8IntVectors>;
defm "" : VPatExtendSDNode_V<[sext],         "PseudoVSEXT", "VF8",
                             AllFractionableVF8IntVectors>;

// 12.5. Vector Bitwise Logical Instructions
defm "" : VPatBinarySDNode_VV_VX_VI<and, "PseudoVAND">;
defm "" : VPatBinarySDNode_VV_VX_VI<or, "PseudoVOR">;
defm "" : VPatBinarySDNode_VV_VX_VI<xor, "PseudoVXOR">;

// 12.6. Vector Single-Width Bit Shift Instructions
defm "" : VPatBinarySDNode_VV_VX_VI<shl, "PseudoVSLL", uimm5>;
defm "" : VPatBinarySDNode_VV_VX_VI<srl, "PseudoVSRL", uimm5>;
defm "" : VPatBinarySDNode_VV_VX_VI<sra, "PseudoVSRA", uimm5>;

// 12.7. Vector Narrowing Integer Right Shift Instructions
foreach vtiTofti = AllFractionableVF2IntVectors in {
  defvar vti = vtiTofti.Vti;
  defvar fti = vtiTofti.Fti;
  def : Pat<(fti.Vector (riscv_trunc_vector (vti.Vector vti.RegClass:$rs1))),
            (!cast<Instruction>("PseudoVNSRL_WI_"#fti.LMul.MX)
                vti.RegClass:$rs1, 0, fti.AVL, fti.SEW)>;
}

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
defm "" : VPatBinarySDNode_VV_VX<udiv, "PseudoVDIVU">;
defm "" : VPatBinarySDNode_VV_VX<sdiv, "PseudoVDIV">;
defm "" : VPatBinarySDNode_VV_VX<urem, "PseudoVREMU">;
defm "" : VPatBinarySDNode_VV_VX<srem, "PseudoVREM">;

// 12.16. Vector Integer Merge Instructions
foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (vselect (vti.Mask VMV0:$vm), vti.RegClass:$rs1,
                                                      vti.RegClass:$rs2)),
            (!cast<Instruction>("PseudoVMERGE_VVM_"#vti.LMul.MX)
                 vti.RegClass:$rs2, vti.RegClass:$rs1, VMV0:$vm,
                 vti.AVL, vti.SEW)>;

  def : Pat<(vti.Vector (vselect (vti.Mask VMV0:$vm), (SplatPat XLenVT:$rs1),
                                                      vti.RegClass:$rs2)),
            (!cast<Instruction>("PseudoVMERGE_VXM_"#vti.LMul.MX)
                 vti.RegClass:$rs2, GPR:$rs1, VMV0:$vm, vti.AVL, vti.SEW)>;

  def : Pat<(vti.Vector (vselect (vti.Mask VMV0:$vm), (SplatPat_simm5 simm5:$rs1),
                                                      vti.RegClass:$rs2)),
            (!cast<Instruction>("PseudoVMERGE_VIM_"#vti.LMul.MX)
                 vti.RegClass:$rs2, simm5:$rs1, VMV0:$vm, vti.AVL, vti.SEW)>;
}

// 16.1. Vector Mask-Register Logical Instructions
foreach mti = AllMasks in {
  def : Pat<(mti.Mask (and VR:$rs1, VR:$rs2)),
            (!cast<Instruction>("PseudoVMAND_MM_"#mti.LMul.MX)
                 VR:$rs1, VR:$rs2, mti.AVL, mti.SEW)>;
  def : Pat<(mti.Mask (or VR:$rs1, VR:$rs2)),
            (!cast<Instruction>("PseudoVMOR_MM_"#mti.LMul.MX)
                 VR:$rs1, VR:$rs2, mti.AVL, mti.SEW)>;
  def : Pat<(mti.Mask (xor VR:$rs1, VR:$rs2)),
            (!cast<Instruction>("PseudoVMXOR_MM_"#mti.LMul.MX)
                 VR:$rs1, VR:$rs2, mti.AVL, mti.SEW)>;

  def : Pat<(mti.Mask (vnot (and VR:$rs1, VR:$rs2))),
            (!cast<Instruction>("PseudoVMNAND_MM_"#mti.LMul.MX)
                 VR:$rs1, VR:$rs2, mti.AVL, mti.SEW)>;
  def : Pat<(mti.Mask (vnot (or VR:$rs1, VR:$rs2))),
            (!cast<Instruction>("PseudoVMNOR_MM_"#mti.LMul.MX)
                 VR:$rs1, VR:$rs2, mti.AVL, mti.SEW)>;
  def : Pat<(mti.Mask (vnot (xor VR:$rs1, VR:$rs2))),
            (!cast<Instruction>("PseudoVMXNOR_MM_"#mti.LMul.MX)
                 VR:$rs1, VR:$rs2, mti.AVL, mti.SEW)>;

  def : Pat<(mti.Mask (and VR:$rs1, (vnot VR:$rs2))),
            (!cast<Instruction>("PseudoVMANDNOT_MM_"#mti.LMul.MX)
                 VR:$rs1, VR:$rs2, mti.AVL, mti.SEW)>;
  def : Pat<(mti.Mask (or VR:$rs1, (vnot VR:$rs2))),
            (!cast<Instruction>("PseudoVMORNOT_MM_"#mti.LMul.MX)
                 VR:$rs1, VR:$rs2, mti.AVL, mti.SEW)>;
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
                 fvti.AVL, fvti.SEW)>;

  def : Pat<(fvti.Vector (vselect (fvti.Mask VMV0:$vm),
                                  (splat_vector fvti.ScalarRegClass:$rs1),
                                  fvti.RegClass:$rs2)),
            (!cast<Instruction>("PseudoVFMERGE_V"#fvti.ScalarSuffix#"M_"#fvti.LMul.MX)
                 fvti.RegClass:$rs2,
                 (fvti.Scalar fvti.ScalarRegClass:$rs1),
                 VMV0:$vm, fvti.AVL, fvti.SEW)>;

  def : Pat<(fvti.Vector (vselect (fvti.Mask VMV0:$vm),
                                  (splat_vector (fvti.Scalar fpimm0)),
                                  fvti.RegClass:$rs2)),
            (!cast<Instruction>("PseudoVMERGE_VIM_"#fvti.LMul.MX)
                 fvti.RegClass:$rs2, 0, VMV0:$vm, fvti.AVL, fvti.SEW)>;
}
} // Predicates = [HasStdExtV, HasStdExtF]

//===----------------------------------------------------------------------===//
// Vector Splats
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtV] in {
foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (splat_vector GPR:$rs1)),
            (!cast<Instruction>("PseudoVMV_V_X_" # vti.LMul.MX)
              GPR:$rs1, vti.AVL, vti.SEW)>;
  def : Pat<(vti.Vector (splat_vector simm5:$rs1)),
            (!cast<Instruction>("PseudoVMV_V_I_" # vti.LMul.MX)
              simm5:$rs1, vti.AVL, vti.SEW)>;
}

foreach mti = AllMasks in {
  def : Pat<(mti.Mask immAllOnesV),
            (!cast<Instruction>("PseudoVMSET_M_"#mti.BX) mti.AVL, mti.SEW)>;
  def : Pat<(mti.Mask immAllZerosV),
            (!cast<Instruction>("PseudoVMCLR_M_"#mti.BX) mti.AVL, mti.SEW)>;
}
} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV, IsRV32] in {
foreach vti = AllIntegerVectors in {
  if !eq(vti.SEW, 64) then {
    def : Pat<(vti.Vector (rv32_splat_i64 GPR:$rs1)),
              (!cast<Instruction>("PseudoVMV_V_X_" # vti.LMul.MX)
                GPR:$rs1, vti.AVL, vti.SEW)>;
    def : Pat<(vti.Vector (rv32_splat_i64 simm5:$rs1)),
              (!cast<Instruction>("PseudoVMV_V_I_" # vti.LMul.MX)
                simm5:$rs1, vti.AVL, vti.SEW)>;
  }
}
} // Predicates = [HasStdExtV, IsRV32]

let Predicates = [HasStdExtV, HasStdExtF] in {
foreach fvti = AllFloatVectors in {
  def : Pat<(fvti.Vector (splat_vector fvti.ScalarRegClass:$rs1)),
            (!cast<Instruction>("PseudoVFMV_V_"#fvti.ScalarSuffix#"_"#fvti.LMul.MX)
              (fvti.Scalar fvti.ScalarRegClass:$rs1),
              fvti.AVL, fvti.SEW)>;

  def : Pat<(fvti.Vector (splat_vector (fvti.Scalar fpimm0))),
            (!cast<Instruction>("PseudoVMV_V_I_"#fvti.LMul.MX)
              0, fvti.AVL, fvti.SEW)>;
}
} // Predicates = [HasStdExtV, HasStdExtF]

//===----------------------------------------------------------------------===//
// Vector Element Inserts/Extracts
//===----------------------------------------------------------------------===//

// The built-in TableGen 'extractelt' and 'insertelt' nodes must return the
// same type as the vector element type. On RISC-V, XLenVT is the only legal
// integer type, so for integer inserts/extracts we use a custom node which
// returns XLenVT.
def riscv_insert_vector_elt
    : SDNode<"ISD::INSERT_VECTOR_ELT",
             SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>, SDTCisVT<2, XLenVT>,
                                  SDTCisPtrTy<3>]>, []>;
def riscv_extract_vector_elt
    : SDNode<"ISD::EXTRACT_VECTOR_ELT",
             SDTypeProfile<1, 2, [SDTCisVT<0, XLenVT>, SDTCisPtrTy<2>]>, []>;

multiclass VPatInsertExtractElt_XI_Idx<bit IsFloat> {
  defvar vtilist = !if(IsFloat, AllFloatVectors, AllIntegerVectors);
  defvar insertelt_node = !if(IsFloat, insertelt, riscv_insert_vector_elt);
  defvar extractelt_node = !if(IsFloat, extractelt, riscv_extract_vector_elt);
  foreach vti = vtilist in {
    defvar MX = vti.LMul.MX;
    defvar vmv_xf_s_inst = !cast<Instruction>(!strconcat("PseudoV",
                                                         !if(IsFloat, "F", ""),
                                                         "MV_",
                                                         vti.ScalarSuffix,
                                                         "_S_", MX));
    defvar vmv_s_xf_inst = !cast<Instruction>(!strconcat("PseudoV",
                                                         !if(IsFloat, "F", ""),
                                                         "MV_S_",
                                                         vti.ScalarSuffix,
                                                         "_", MX));
    // Only pattern-match insert/extract-element operations where the index is
    // 0. Any other index will have been custom-lowered to slide the vector
    // correctly into place (and, in the case of insert, slide it back again
    // afterwards).
    def : Pat<(vti.Scalar (extractelt_node (vti.Vector vti.RegClass:$rs2), 0)),
              (vmv_xf_s_inst vti.RegClass:$rs2, vti.SEW)>;

    def : Pat<(vti.Vector (insertelt_node (vti.Vector vti.RegClass:$merge),
                                          vti.ScalarRegClass:$rs1, 0)),
              (vmv_s_xf_inst vti.RegClass:$merge,
                      (vti.Scalar vti.ScalarRegClass:$rs1),
                      vti.AVL, vti.SEW)>;
  }
}

let Predicates = [HasStdExtV] in
defm "" : VPatInsertExtractElt_XI_Idx</*IsFloat*/0>;
let Predicates = [HasStdExtV, HasStdExtF] in
defm "" : VPatInsertExtractElt_XI_Idx</*IsFloat*/1>;

//===----------------------------------------------------------------------===//
// Miscellaneous RISCVISD SDNodes
//===----------------------------------------------------------------------===//

def riscv_vid
    : SDNode<"RISCVISD::VID", SDTypeProfile<1, 0, [SDTCisVec<0>]>, []>;

def SDTRVVSlide : SDTypeProfile<1, 3, [
  SDTCisVec<0>, SDTCisSameAs<1, 0>, SDTCisSameAs<2, 0>, SDTCisVT<3, XLenVT>
]>;

def riscv_slideup : SDNode<"RISCVISD::VSLIDEUP", SDTRVVSlide, []>;
def riscv_slidedown : SDNode<"RISCVISD::VSLIDEDOWN", SDTRVVSlide, []>;

let Predicates = [HasStdExtV] in {

foreach vti = AllIntegerVectors in
  def : Pat<(vti.Vector riscv_vid),
            (!cast<Instruction>("PseudoVID_V_"#vti.LMul.MX) vti.AVL, vti.SEW)>;

foreach vti = !listconcat(AllIntegerVectors, AllFloatVectors) in {
    def : Pat<(vti.Vector (riscv_slideup (vti.Vector vti.RegClass:$rs3),
                                         (vti.Vector vti.RegClass:$rs1),
                                         uimm5:$rs2)),
              (!cast<Instruction>("PseudoVSLIDEUP_VI_"#vti.LMul.MX)
                  vti.RegClass:$rs3, vti.RegClass:$rs1, uimm5:$rs2,
                  vti.AVL, vti.SEW)>;

    def : Pat<(vti.Vector (riscv_slideup (vti.Vector vti.RegClass:$rs3),
                                         (vti.Vector vti.RegClass:$rs1),
                                         GPR:$rs2)),
              (!cast<Instruction>("PseudoVSLIDEUP_VX_"#vti.LMul.MX)
                  vti.RegClass:$rs3, vti.RegClass:$rs1, GPR:$rs2,
                  vti.AVL, vti.SEW)>;

    def : Pat<(vti.Vector (riscv_slidedown (vti.Vector vti.RegClass:$rs3),
                                           (vti.Vector vti.RegClass:$rs1),
                                           uimm5:$rs2)),
              (!cast<Instruction>("PseudoVSLIDEDOWN_VI_"#vti.LMul.MX)
                  vti.RegClass:$rs3, vti.RegClass:$rs1, uimm5:$rs2,
                  vti.AVL, vti.SEW)>;

    def : Pat<(vti.Vector (riscv_slidedown (vti.Vector vti.RegClass:$rs3),
                                           (vti.Vector vti.RegClass:$rs1),
                                           GPR:$rs2)),
              (!cast<Instruction>("PseudoVSLIDEDOWN_VX_"#vti.LMul.MX)
                  vti.RegClass:$rs3, vti.RegClass:$rs1, GPR:$rs2,
                  vti.AVL, vti.SEW)>;
}
} // Predicates = [HasStdExtV]
