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

// Give explicit Complexity to prefer simm5/uimm5.
def SplatPat       : ComplexPattern<vAny, 1, "selectVSplat",      [splat_vector, rv32_splat_i64], [], 1>;
def SplatPat_simm5 : ComplexPattern<vAny, 1, "selectVSplatSimm5", [splat_vector, rv32_splat_i64], [], 2>;
def SplatPat_uimm5 : ComplexPattern<vAny, 1, "selectVSplatUimm5", [splat_vector, rv32_splat_i64], [], 2>;

class SwapHelper<dag Prefix, dag A, dag B, dag Suffix, bit swap> {
   dag Value = !con(Prefix, !if(swap, B, A), !if(swap, A, B), Suffix);
}

multiclass VPatUSLoadStoreSDNode<ValueType type,
                                 int sew,
                                 LMULInfo vlmul,
                                 OutPatFrag avl,
                                 VReg reg_class>
{
  defvar load_instr = !cast<Instruction>("PseudoVLE"#sew#"_V_"#vlmul.MX);
  defvar store_instr = !cast<Instruction>("PseudoVSE"#sew#"_V_"#vlmul.MX);
  // Load
  def : Pat<(type (load BaseAddr:$rs1)),
            (load_instr BaseAddr:$rs1, avl, sew)>;
  // Store
  def : Pat<(store type:$rs2, BaseAddr:$rs1),
            (store_instr reg_class:$rs2, BaseAddr:$rs1, avl, sew)>;
}

multiclass VPatUSLoadStoreWholeVRSDNode<ValueType type,
                                        int sew,
                                        LMULInfo vlmul,
                                        VReg reg_class>
{
  defvar load_instr =
    !cast<Instruction>("VL"#!substr(vlmul.MX, 1)#"RE"#sew#"_V");
  defvar store_instr =
    !cast<Instruction>("VS"#!substr(vlmul.MX, 1)#"R_V");

  // Load
  def : Pat<(type (load BaseAddr:$rs1)),
            (load_instr BaseAddr:$rs1)>;
  // Store
  def : Pat<(store type:$rs2, BaseAddr:$rs1),
            (store_instr reg_class:$rs2, BaseAddr:$rs1)>;
}

multiclass VPatUSLoadStoreMaskSDNode<MTypeInfo m>
{
  defvar load_instr = !cast<Instruction>("PseudoVLE1_V_"#m.BX);
  defvar store_instr = !cast<Instruction>("PseudoVSE1_V_"#m.BX);
  // Load
  def : Pat<(m.Mask (load BaseAddr:$rs1)),
            (load_instr BaseAddr:$rs1, m.AVL, m.SEW)>;
  // Store
  def : Pat<(store m.Mask:$rs2, BaseAddr:$rs1),
            (store_instr VR:$rs2, BaseAddr:$rs1, m.AVL, m.SEW)>;
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
                              vti.Vector, vti.Vector, vti.Mask, vti.SEW,
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
                              vti.Vector, vti.Vector, vti.Mask, vti.SEW,
                              vti.LMul, vti.AVL, vti.RegClass, vti.RegClass,
                              SplatPat, GPR>;
    def : VPatBinarySDNode_XI<vop, instruction_name, "VI",
                              vti.Vector, vti.Vector, vti.Mask, vti.SEW,
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

multiclass VPatFPSetCCSDNode_VV_VF_FV<CondCode cc,
                                      string inst_name,
                                      string swapped_op_inst_name> {
  foreach fvti = AllFloatVectors in {
    def : Pat<(fvti.Mask (setcc (fvti.Vector fvti.RegClass:$rs1),
                                (fvti.Vector fvti.RegClass:$rs2),
                                cc)),
              (!cast<Instruction>(inst_name#"_VV_"#fvti.LMul.MX)
                  fvti.RegClass:$rs1, fvti.RegClass:$rs2, fvti.AVL, fvti.SEW)>;
    def : Pat<(fvti.Mask (setcc (fvti.Vector fvti.RegClass:$rs1),
                                (splat_vector fvti.ScalarRegClass:$rs2),
                                cc)),
              (!cast<Instruction>(inst_name#"_V"#fvti.ScalarSuffix#"_"#fvti.LMul.MX)
                  fvti.RegClass:$rs1, fvti.ScalarRegClass:$rs2,
                  fvti.AVL, fvti.SEW)>;
    def : Pat<(fvti.Mask (setcc (splat_vector fvti.ScalarRegClass:$rs2),
                                (fvti.Vector fvti.RegClass:$rs1),
                                cc)),
              (!cast<Instruction>(swapped_op_inst_name#"_V"#fvti.ScalarSuffix#"_"#fvti.LMul.MX)
                  fvti.RegClass:$rs1, fvti.ScalarRegClass:$rs2,
                  fvti.AVL, fvti.SEW)>;
  }
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

multiclass VPatConvertI2FPSDNode_V<SDNode vop, string instruction_name> {
  foreach fvti = AllFloatVectors in {
    defvar ivti = GetIntVTypeInfo<fvti>.Vti;
    def : Pat<(fvti.Vector (vop (ivti.Vector ivti.RegClass:$rs1))),
              (!cast<Instruction>(instruction_name#"_"#fvti.LMul.MX)
                  ivti.RegClass:$rs1, fvti.AVL, fvti.SEW)>;
  }
}

multiclass VPatConvertFP2ISDNode_V<SDNode vop, string instruction_name> {
  foreach fvti = AllFloatVectors in {
    defvar ivti = GetIntVTypeInfo<fvti>.Vti;
    def : Pat<(ivti.Vector (vop (fvti.Vector fvti.RegClass:$rs1))),
              (!cast<Instruction>(instruction_name#"_"#ivti.LMul.MX)
                  fvti.RegClass:$rs1, ivti.AVL, ivti.SEW)>;
  }
}

multiclass VPatWConvertI2FPSDNode_V<SDNode vop, string instruction_name> {
  foreach vtiToWti = AllWidenableIntToFloatVectors in {
    defvar ivti = vtiToWti.Vti;
    defvar fwti = vtiToWti.Wti;
    def : Pat<(fwti.Vector (vop (ivti.Vector ivti.RegClass:$rs1))),
              (!cast<Instruction>(instruction_name#"_"#ivti.LMul.MX)
                  ivti.RegClass:$rs1, ivti.AVL, ivti.SEW)>;
  }
}

multiclass VPatWConvertFP2ISDNode_V<SDNode vop, string instruction_name> {
  foreach fvtiToFWti = AllWidenableFloatVectors in {
    defvar fvti = fvtiToFWti.Vti;
    defvar iwti = GetIntVTypeInfo<fvtiToFWti.Wti>.Vti;
    def : Pat<(iwti.Vector (vop (fvti.Vector fvti.RegClass:$rs1))),
              (!cast<Instruction>(instruction_name#"_"#fvti.LMul.MX)
                  fvti.RegClass:$rs1, fvti.AVL, fvti.SEW)>;
  }
}

multiclass VPatNConvertI2FPSDNode_V<SDNode vop, string instruction_name> {
  foreach fvtiToFWti = AllWidenableFloatVectors in {
    defvar fvti = fvtiToFWti.Vti;
    defvar iwti = GetIntVTypeInfo<fvtiToFWti.Wti>.Vti;
    def : Pat<(fvti.Vector (vop (iwti.Vector iwti.RegClass:$rs1))),
              (!cast<Instruction>(instruction_name#"_"#fvti.LMul.MX)
                  iwti.RegClass:$rs1, fvti.AVL, fvti.SEW)>;
  }
}

multiclass VPatNConvertFP2ISDNode_V<SDNode vop, string instruction_name> {
  foreach vtiToWti = AllWidenableIntToFloatVectors in {
    defvar vti = vtiToWti.Vti;
    defvar fwti = vtiToWti.Wti;
    def : Pat<(vti.Vector (vop (fwti.Vector fwti.RegClass:$rs1))),
              (!cast<Instruction>(instruction_name#"_"#vti.LMul.MX)
                  fwti.RegClass:$rs1, vti.AVL, vti.SEW)>;
  }
}

//===----------------------------------------------------------------------===//
// Patterns.
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtV] in {

// 7.4. Vector Unit-Stride Instructions
foreach vti = !listconcat(FractionalGroupIntegerVectors,
                          FractionalGroupFloatVectors) in
  defm "" : VPatUSLoadStoreSDNode<vti.Vector, vti.SEW, vti.LMul,
                                  vti.AVL, vti.RegClass>;
foreach vti = [VI8M1, VI16M1, VI32M1, VI64M1, VF16M1, VF32M1, VF64M1] in
  defm "" : VPatUSLoadStoreWholeVRSDNode<vti.Vector, vti.SEW, vti.LMul,
                                         vti.RegClass>;
foreach vti = !listconcat(GroupIntegerVectors, GroupFloatVectors) in
  defm "" : VPatUSLoadStoreWholeVRSDNode<vti.Vector, vti.SEW, vti.LMul,
                                         vti.RegClass>;
foreach mti = AllMasks in
  defm "" : VPatUSLoadStoreMaskSDNode<mti>;

// 12.1. Vector Single-Width Integer Add and Subtract
defm "" : VPatBinarySDNode_VV_VX_VI<add, "PseudoVADD">;
defm "" : VPatBinarySDNode_VV_VX<sub, "PseudoVSUB">;
// Handle VRSUB specially since it's the only integer binary op with reversed
// pattern operands
foreach vti = AllIntegerVectors in {
  def : Pat<(sub (vti.Vector (SplatPat GPR:$rs2)),
                 (vti.Vector vti.RegClass:$rs1)),
            (!cast<Instruction>("PseudoVRSUB_VX_"# vti.LMul.MX)
                 vti.RegClass:$rs1, GPR:$rs2, vti.AVL, vti.SEW)>;
  def : Pat<(sub (vti.Vector (SplatPat_simm5 simm5:$rs2)),
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

// 12.15. Vector Integer Merge Instructions
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

  // Handle vnot the same as the vnot.mm pseudoinstruction.
  def : Pat<(mti.Mask (vnot VR:$rs)),
            (!cast<Instruction>("PseudoVMNAND_MM_"#mti.LMul.MX)
                 VR:$rs, VR:$rs, mti.AVL, mti.SEW)>;
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

// 14.6 Vector Single-Width Floating-Point Fused Multiply-Add Instructions.
foreach fvti = AllFloatVectors in {
  // NOTE: We choose VFMADD because it has the most commuting freedom. So it
  // works best with how TwoAddressInstructionPass tries commuting.
  defvar suffix = fvti.LMul.MX # "_COMMUTABLE";
  def : Pat<(fvti.Vector (fma fvti.RegClass:$rs1, fvti.RegClass:$rd,
                              fvti.RegClass:$rs2)),
            (!cast<Instruction>("PseudoVFMADD_VV_"# suffix)
                 fvti.RegClass:$rd, fvti.RegClass:$rs1, fvti.RegClass:$rs2,
                 fvti.AVL, fvti.SEW)>;
  def : Pat<(fvti.Vector (fma fvti.RegClass:$rs1, fvti.RegClass:$rd,
                              (fneg fvti.RegClass:$rs2))),
            (!cast<Instruction>("PseudoVFMSUB_VV_"# suffix)
                 fvti.RegClass:$rd, fvti.RegClass:$rs1, fvti.RegClass:$rs2,
                 fvti.AVL, fvti.SEW)>;
  def : Pat<(fvti.Vector (fma (fneg fvti.RegClass:$rs1), fvti.RegClass:$rd,
                              (fneg fvti.RegClass:$rs2))),
            (!cast<Instruction>("PseudoVFNMADD_VV_"# suffix)
                 fvti.RegClass:$rd, fvti.RegClass:$rs1, fvti.RegClass:$rs2,
                 fvti.AVL, fvti.SEW)>;
  def : Pat<(fvti.Vector (fma (fneg fvti.RegClass:$rs1), fvti.RegClass:$rd,
                              fvti.RegClass:$rs2)),
            (!cast<Instruction>("PseudoVFNMSUB_VV_"# suffix)
                 fvti.RegClass:$rd, fvti.RegClass:$rs1, fvti.RegClass:$rs2,
                 fvti.AVL, fvti.SEW)>;

  // The choice of VFMADD here is arbitrary, vfmadd.vf and vfmacc.vf are equally
  // commutable.
  def : Pat<(fvti.Vector (fma (splat_vector fvti.ScalarRegClass:$rs1),
                              fvti.RegClass:$rd, fvti.RegClass:$rs2)),
            (!cast<Instruction>("PseudoVFMADD_V" # fvti.ScalarSuffix # "_" # suffix)
                 fvti.RegClass:$rd, fvti.ScalarRegClass:$rs1, fvti.RegClass:$rs2,
                 fvti.AVL, fvti.SEW)>;
  def : Pat<(fvti.Vector (fma (splat_vector fvti.ScalarRegClass:$rs1),
                              fvti.RegClass:$rd, (fneg fvti.RegClass:$rs2))),
            (!cast<Instruction>("PseudoVFMSUB_V" # fvti.ScalarSuffix # "_" # suffix)
                 fvti.RegClass:$rd, fvti.ScalarRegClass:$rs1, fvti.RegClass:$rs2,
                 fvti.AVL, fvti.SEW)>;

  def : Pat<(fvti.Vector (fma (splat_vector fvti.ScalarRegClass:$rs1),
                              (fneg fvti.RegClass:$rd), (fneg fvti.RegClass:$rs2))),
            (!cast<Instruction>("PseudoVFNMADD_V" # fvti.ScalarSuffix # "_" # suffix)
                 fvti.RegClass:$rd, fvti.ScalarRegClass:$rs1, fvti.RegClass:$rs2,
                 fvti.AVL, fvti.SEW)>;
  def : Pat<(fvti.Vector (fma (splat_vector fvti.ScalarRegClass:$rs1),
                              (fneg fvti.RegClass:$rd), fvti.RegClass:$rs2)),
            (!cast<Instruction>("PseudoVFNMSUB_V" # fvti.ScalarSuffix # "_" # suffix)
                 fvti.RegClass:$rd, fvti.ScalarRegClass:$rs1, fvti.RegClass:$rs2,
                 fvti.AVL, fvti.SEW)>;

  // The splat might be negated.
  def : Pat<(fvti.Vector (fma (fneg (splat_vector fvti.ScalarRegClass:$rs1)),
                              fvti.RegClass:$rd, (fneg fvti.RegClass:$rs2))),
            (!cast<Instruction>("PseudoVFNMADD_V" # fvti.ScalarSuffix # "_" # suffix)
                 fvti.RegClass:$rd, fvti.ScalarRegClass:$rs1, fvti.RegClass:$rs2,
                 fvti.AVL, fvti.SEW)>;
  def : Pat<(fvti.Vector (fma (fneg (splat_vector fvti.ScalarRegClass:$rs1)),
                              fvti.RegClass:$rd, fvti.RegClass:$rs2)),
            (!cast<Instruction>("PseudoVFNMSUB_V" # fvti.ScalarSuffix # "_" # suffix)
                 fvti.RegClass:$rd, fvti.ScalarRegClass:$rs1, fvti.RegClass:$rs2,
                 fvti.AVL, fvti.SEW)>;
}

foreach vti = AllFloatVectors in {
  // 14.8. Vector Floating-Point Square-Root Instruction
  def : Pat<(fsqrt (vti.Vector vti.RegClass:$rs2)),
            (!cast<Instruction>("PseudoVFSQRT_V_"# vti.LMul.MX)
                 vti.RegClass:$rs2, vti.AVL, vti.SEW)>;

  // 14.12. Vector Floating-Point Sign-Injection Instructions
  def : Pat<(fabs (vti.Vector vti.RegClass:$rs)),
            (!cast<Instruction>("PseudoVFSGNJX_VV_"# vti.LMul.MX)
                 vti.RegClass:$rs, vti.RegClass:$rs, vti.AVL, vti.SEW)>;
  // Handle fneg with VFSGNJN using the same input for both operands.
  def : Pat<(fneg (vti.Vector vti.RegClass:$rs)),
            (!cast<Instruction>("PseudoVFSGNJN_VV_"# vti.LMul.MX)
                 vti.RegClass:$rs, vti.RegClass:$rs, vti.AVL, vti.SEW)>;

  def : Pat<(vti.Vector (fcopysign (vti.Vector vti.RegClass:$rs1),
                                   (vti.Vector vti.RegClass:$rs2))),
            (!cast<Instruction>("PseudoVFSGNJ_VV_"# vti.LMul.MX)
                 vti.RegClass:$rs1, vti.RegClass:$rs2, vti.AVL, vti.SEW)>;
  def : Pat<(vti.Vector (fcopysign (vti.Vector vti.RegClass:$rs1),
                                   (vti.Vector (splat_vector vti.ScalarRegClass:$rs2)))),
            (!cast<Instruction>("PseudoVFSGNJ_V"#vti.ScalarSuffix#"_"#vti.LMul.MX)
                 vti.RegClass:$rs1, vti.ScalarRegClass:$rs2, vti.AVL, vti.SEW)>;

  def : Pat<(vti.Vector (fcopysign (vti.Vector vti.RegClass:$rs1),
                                   (vti.Vector (fneg vti.RegClass:$rs2)))),
            (!cast<Instruction>("PseudoVFSGNJN_VV_"# vti.LMul.MX)
                 vti.RegClass:$rs1, vti.RegClass:$rs2, vti.AVL, vti.SEW)>;
  def : Pat<(vti.Vector (fcopysign (vti.Vector vti.RegClass:$rs1),
                                   (vti.Vector (fneg (splat_vector vti.ScalarRegClass:$rs2))))),
            (!cast<Instruction>("PseudoVFSGNJN_V"#vti.ScalarSuffix#"_"#vti.LMul.MX)
                 vti.RegClass:$rs1, vti.ScalarRegClass:$rs2, vti.AVL, vti.SEW)>;
}

// 14.13. Vector Floating-Point Compare Instructions
defm "" : VPatFPSetCCSDNode_VV_VF_FV<SETEQ,  "PseudoVMFEQ", "PseudoVMFEQ">;
defm "" : VPatFPSetCCSDNode_VV_VF_FV<SETOEQ, "PseudoVMFEQ", "PseudoVMFEQ">;

defm "" : VPatFPSetCCSDNode_VV_VF_FV<SETNE,  "PseudoVMFNE", "PseudoVMFNE">;
defm "" : VPatFPSetCCSDNode_VV_VF_FV<SETUNE, "PseudoVMFNE", "PseudoVMFNE">;

defm "" : VPatFPSetCCSDNode_VV_VF_FV<SETLT,  "PseudoVMFLT", "PseudoVMFGT">;
defm "" : VPatFPSetCCSDNode_VV_VF_FV<SETOLT, "PseudoVMFLT", "PseudoVMFGT">;

defm "" : VPatFPSetCCSDNode_VV_VF_FV<SETLE,  "PseudoVMFLE", "PseudoVMFGE">;
defm "" : VPatFPSetCCSDNode_VV_VF_FV<SETOLE, "PseudoVMFLE", "PseudoVMFGE">;

// Floating-point vselects:
// 12.15. Vector Integer Merge Instructions
// 14.15. Vector Floating-Point Merge Instruction
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

// 14.17. Vector Single-Width Floating-Point/Integer Type-Convert Instructions
defm "" : VPatConvertFP2ISDNode_V<fp_to_sint, "PseudoVFCVT_RTZ_X_F_V">;
defm "" : VPatConvertFP2ISDNode_V<fp_to_uint, "PseudoVFCVT_RTZ_XU_F_V">;
defm "" : VPatConvertI2FPSDNode_V<sint_to_fp, "PseudoVFCVT_F_X_V">;
defm "" : VPatConvertI2FPSDNode_V<uint_to_fp, "PseudoVFCVT_F_XU_V">;

// 14.18. Widening Floating-Point/Integer Type-Convert Instructions
defm "" : VPatWConvertFP2ISDNode_V<fp_to_sint, "PseudoVFWCVT_RTZ_X_F_V">;
defm "" : VPatWConvertFP2ISDNode_V<fp_to_uint, "PseudoVFWCVT_RTZ_XU_F_V">;
defm "" : VPatWConvertI2FPSDNode_V<sint_to_fp, "PseudoVFWCVT_F_X_V">;
defm "" : VPatWConvertI2FPSDNode_V<uint_to_fp, "PseudoVFWCVT_F_XU_V">;
foreach fvtiToFWti = AllWidenableFloatVectors in {
  defvar fvti = fvtiToFWti.Vti;
  defvar fwti = fvtiToFWti.Wti;
  def : Pat<(fwti.Vector (fpextend (fvti.Vector fvti.RegClass:$rs1))),
            (!cast<Instruction>("PseudoVFWCVT_F_F_V_"#fvti.LMul.MX)
                fvti.RegClass:$rs1, fvti.AVL, fvti.SEW)>;
}

// 14.19. Narrowing Floating-Point/Integer Type-Convert Instructions
defm "" : VPatNConvertFP2ISDNode_V<fp_to_sint, "PseudoVFNCVT_RTZ_X_F_W">;
defm "" : VPatNConvertFP2ISDNode_V<fp_to_uint, "PseudoVFNCVT_RTZ_XU_F_W">;
defm "" : VPatNConvertI2FPSDNode_V<sint_to_fp, "PseudoVFNCVT_F_X_W">;
defm "" : VPatNConvertI2FPSDNode_V<uint_to_fp, "PseudoVFNCVT_F_XU_W">;
foreach fvtiToFWti = AllWidenableFloatVectors in {
  defvar fvti = fvtiToFWti.Vti;
  defvar fwti = fvtiToFWti.Wti;
  def : Pat<(fvti.Vector (fpround (fwti.Vector fwti.RegClass:$rs1))),
            (!cast<Instruction>("PseudoVFNCVT_F_F_W_"#fvti.LMul.MX)
                fwti.RegClass:$rs1, fvti.AVL, fvti.SEW)>;
}
} // Predicates = [HasStdExtV, HasStdExtF]

//===----------------------------------------------------------------------===//
// Vector Splats
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtV] in {
foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (SplatPat GPR:$rs1)),
            (!cast<Instruction>("PseudoVMV_V_X_" # vti.LMul.MX)
              GPR:$rs1, vti.AVL, vti.SEW)>;
  def : Pat<(vti.Vector (SplatPat_simm5 simm5:$rs1)),
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
// Vector Element Extracts
//===----------------------------------------------------------------------===//
let Predicates = [HasStdExtV, HasStdExtF] in
foreach vti = AllFloatVectors in {
  defvar vmv_f_s_inst = !cast<Instruction>(!strconcat("PseudoVFMV_",
                                                       vti.ScalarSuffix,
                                                       "_S_", vti.LMul.MX));
  // Only pattern-match extract-element operations where the index is 0. Any
  // other index will have been custom-lowered to slide the vector correctly
  // into place.
  def : Pat<(vti.Scalar (extractelt (vti.Vector vti.RegClass:$rs2), 0)),
            (vmv_f_s_inst vti.RegClass:$rs2, vti.SEW)>;
}
