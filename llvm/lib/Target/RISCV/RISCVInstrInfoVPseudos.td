//===-- RISCVInstrInfoVPseudos.td - RISC-V 'V' Pseudos -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file contains the required infrastructure to support code generation
/// for the standard 'V' (Vector) extension, version 0.9.  This version is still
/// experimental as the 'V' extension hasn't been ratified yet.
///
/// This file is included from RISCVInstrInfoV.td
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Utilities.
//===----------------------------------------------------------------------===//

// This class describes information associated to the LMUL.
class LMULInfo<int lmul, VReg regclass, string mx> {
  bits<3> value = lmul; // This is encoded as the vlmul field of vtype.
  VReg vrclass = regclass;
  string MX = mx;
}

// Associate LMUL with tablegen records of register classes.
def V_M1  : LMULInfo<0b000,   VR,  "M1">;
def V_M2  : LMULInfo<0b001, VRM2,  "M2">;
def V_M4  : LMULInfo<0b010, VRM4,  "M4">;
def V_M8  : LMULInfo<0b011, VRM8,  "M8">;

def V_MF8 : LMULInfo<0b101,   VR, "MF8">;
def V_MF4 : LMULInfo<0b110,   VR, "MF4">;
def V_MF2 : LMULInfo<0b111,   VR, "MF2">;

// Used to iterate over all possible LMULs.
def MxList {
  list<LMULInfo> m = [V_MF8, V_MF4, V_MF2, V_M1, V_M2, V_M4, V_M8];
}

class shift_amount<int num> {
  int val = !if(!eq(num, 1), 0, !add(1, shift_amount<!srl(num, 1)>.val));
}

// Output pattern for X0 used to represent VLMAX in the pseudo instructions.
def VLMax : OutPatFrag<(ops), (XLenVT X0)>;

// List of EEW.
defvar EEWList = [8, 16, 32, 64];

//===----------------------------------------------------------------------===//
// Vector register and vector group type information.
//===----------------------------------------------------------------------===//

class VTypeInfo<ValueType Vec, ValueType Mas, int Sew, VReg Reg, LMULInfo M>
{
  ValueType Vector = Vec;
  ValueType Mask = Mas;
  int SEW = Sew;
  VReg RegClass = Reg;
  LMULInfo LMul = M;
}

class GroupVTypeInfo<ValueType Vec, ValueType VecM1, ValueType Mas,
                          int Sew, VReg Reg, LMULInfo M>
    : VTypeInfo<Vec, Mas, Sew, Reg, M>
{
  ValueType VectorM1 = VecM1;
}

defset list<VTypeInfo> AllVectors = {
  defset list<VTypeInfo> AllIntegerVectors = {
    def : VTypeInfo<vint8mf8_t,  vbool64_t,  8, VR, V_MF8>;
    def : VTypeInfo<vint8mf4_t,  vbool32_t,  8, VR, V_MF4>;
    def : VTypeInfo<vint8mf2_t,  vbool16_t,  8, VR, V_MF2>;
    def : VTypeInfo<vint8m1_t,   vbool8_t,   8, VR, V_M1>;
    def : VTypeInfo<vint16mf4_t, vbool64_t, 16, VR, V_MF4>;
    def : VTypeInfo<vint16mf2_t, vbool32_t, 16, VR, V_MF2>;
    def : VTypeInfo<vint16m1_t,  vbool16_t, 16, VR, V_M1>;
    def : VTypeInfo<vint32mf2_t, vbool64_t, 32, VR, V_MF2>;
    def : VTypeInfo<vint32m1_t,  vbool32_t, 32, VR, V_M1>;
    def : VTypeInfo<vint64m1_t,  vbool64_t, 64, VR, V_M1>;

    def : GroupVTypeInfo<vint8m2_t, vint8m1_t, vbool4_t, 8, VRM2, V_M2>;
    def : GroupVTypeInfo<vint8m4_t, vint8m1_t, vbool2_t, 8, VRM4, V_M4>;
    def : GroupVTypeInfo<vint8m8_t, vint8m1_t, vbool1_t, 8, VRM8, V_M8>;

    def : GroupVTypeInfo<vint16m2_t,vint16m1_t,vbool8_t, 16,VRM2, V_M2>;
    def : GroupVTypeInfo<vint16m4_t,vint16m1_t,vbool4_t, 16,VRM4, V_M4>;
    def : GroupVTypeInfo<vint16m8_t,vint16m1_t,vbool2_t, 16,VRM8, V_M8>;

    def : GroupVTypeInfo<vint32m2_t,vint32m1_t,vbool16_t,32,VRM2, V_M2>;
    def : GroupVTypeInfo<vint32m4_t,vint32m1_t,vbool8_t, 32,VRM4, V_M4>;
    def : GroupVTypeInfo<vint32m8_t,vint32m1_t,vbool4_t, 32,VRM8, V_M8>;

    def : GroupVTypeInfo<vint64m2_t,vint64m1_t,vbool32_t,64,VRM2, V_M2>;
    def : GroupVTypeInfo<vint64m4_t,vint64m1_t,vbool16_t,64,VRM4, V_M4>;
    def : GroupVTypeInfo<vint64m8_t,vint64m1_t,vbool8_t, 64,VRM8, V_M8>;
  }
}

// This class holds the record of the RISCVVPseudoTable below.
// This represents the information we need in codegen for each pseudo.
class RISCVVPseudo {
  Pseudo Pseudo = !cast<Pseudo>(NAME); // Used as a key.
  Instruction BaseInstr;
  bits<8> VLIndex;
  bits<8> SEWIndex;
  bits<8> MergeOpIndex;
  bits<3> VLMul;
}

// The actual table.
def RISCVVPseudosTable : GenericTable {
  let FilterClass = "RISCVVPseudo";
  let CppTypeName = "PseudoInfo";
  let Fields = [ "Pseudo", "BaseInstr", "VLIndex", "SEWIndex", "MergeOpIndex",
                 "VLMul" ];
  let PrimaryKey = [ "Pseudo" ];
  let PrimaryKeyName = "getPseudoInfo";
}

//===----------------------------------------------------------------------===//
// Helpers to define the different pseudo instructions.
//===----------------------------------------------------------------------===//

multiclass pseudo_binary<VReg result_reg_class,
                         VReg op1_reg_class,
                         DAGOperand op2_kind,
                         LMULInfo vlmul > {
  let Constraints = "$rd = $merge",
      Uses = [VL, VTYPE], VLIndex = 5, SEWIndex = 6, MergeOpIndex = 1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)) in
    def "_"# vlmul.MX : Pseudo<(outs result_reg_class:$rd),
                            (ins result_reg_class:$merge,
                                 op1_reg_class:$rs2, op2_kind:$rs1,
                                 VMaskOp:$vm, GPR:$vl, ixlenimm:$sew),
                            []>,
                     RISCVVPseudo;
}

multiclass pseudo_binary_v_vv_vx_vi<Operand imm_type = simm5,
                                    bit force_earlyclobber = 0> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach m = MxList.m in
  {
    let VLMul = m.value in
    {
      defvar evr = m.vrclass;
      defm _VV : pseudo_binary<evr, evr, evr, m>;
      defm _VX : pseudo_binary<evr, evr, GPR, m>;
      defm _VI : pseudo_binary<evr, evr, imm_type, m>;
    }
  }
}

//===----------------------------------------------------------------------===//
// Helpers to define the different patterns.
//===----------------------------------------------------------------------===//

multiclass pat_vop_binary<SDNode vop,
                          string instruction_name,
                          ValueType result_type,
                          ValueType op_type,
                          ValueType mask_type,
                          int sew,
                          LMULInfo vlmul,
                          VReg result_reg_class,
                          VReg op_reg_class,
                          bit swap = 0>
{
  defvar instruction = !cast<Instruction>(instruction_name#"_VV_"# vlmul.MX);
  def : Pat<(result_type (vop
                          (op_type op_reg_class:$rs1),
                          (op_type op_reg_class:$rs2))),
            (instruction (result_type (IMPLICIT_DEF)),
                         op_reg_class:$rs1,
                         op_reg_class:$rs2,
                         (mask_type zero_reg),
                         VLMax, sew)>;
}

multiclass pat_vop_binary_common<SDNode vop,
                                 string instruction_name,
                                 list<VTypeInfo> vtilist>
{
  foreach vti = vtilist in
  defm : pat_vop_binary<vop, instruction_name,
                        vti.Vector, vti.Vector, vti.Mask, vti.SEW,
                        vti.LMul, vti.RegClass, vti.RegClass>;
}

//===----------------------------------------------------------------------===//
// Pseudo instructions and patterns.
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtV] in {

//===----------------------------------------------------------------------===//
// 6. Configuration-Setting Instructions
//===----------------------------------------------------------------------===//

// Pseudos.
let hasSideEffects = 1, mayLoad = 0, mayStore = 0, Defs = [VL, VTYPE] in {
def PseudoVSETVLI : Pseudo<(outs GPR:$rd), (ins GPR:$rs1, VTypeIOp:$vtypei), []>;

}

//===----------------------------------------------------------------------===//
// 7. Vector Loads and Stores
//===----------------------------------------------------------------------===//

// Pseudos.
foreach eew = EEWList in {
  foreach lmul = MxList.m in {
    defvar LInfo = lmul.MX;
    defvar vreg = lmul.vrclass;
    defvar vlmul = lmul.value;
    defvar constraint = "$rd = $merge";

    let mayLoad = 1, mayStore = 0, hasSideEffects = 0,
        usesCustomInserter = 1,
        VLMul = vlmul in
    {
      let Uses = [VL, VTYPE], VLIndex = 4, SEWIndex = 5, MergeOpIndex = 1,
          Constraints = constraint,
          BaseInstr = !cast<Instruction>("VLE" # eew # "_V") in
          def "PseudoVLE" # eew # "_V_" # LInfo
            : Pseudo<(outs vreg:$rd),
                     (ins vreg:$merge, GPR:$rs1, VMaskOp:$mask, GPR:$vl,
                      ixlenimm:$sew),
                     []>,
              RISCVVPseudo;
    }

    let mayLoad = 0, mayStore = 1, hasSideEffects = 0,
        usesCustomInserter = 1,
        VLMul = vlmul in
    {
      // Masked stores do not have a merge operand as merge is done in memory
      let Uses = [VL, VTYPE],
          VLIndex = 3, SEWIndex = 4, MergeOpIndex = -1,
          BaseInstr = !cast<Instruction>("VSE" # eew # "_V") in
        def "PseudoVSE" # eew # "_V_" # LInfo
            : Pseudo<(outs),
                     (ins vreg:$rd, GPR:$rs1, VMaskOp:$mask, GPR:$vl,
                          ixlenimm:$sew),
                     []>,
              RISCVVPseudo;
    }
  }
}

// Patterns.
multiclass pat_load_store<LLVMType type,
                          LLVMType mask_type,
                          int sew,
                          LMULInfo vlmul,
                          VReg reg_class>
{
  defvar load_instr = !cast<Instruction>("PseudoVLE" # sew # "_V_"# vlmul.MX);
  defvar store_instr = !cast<Instruction>("PseudoVSE" # sew # "_V_"# vlmul.MX);
  // Load
  def : Pat<(type (load GPR:$rs1)),
            (load_instr (type (IMPLICIT_DEF)),
             GPR:$rs1,
             (mask_type zero_reg),
             VLMax, sew)>;
  def : Pat<(type (load AddrFI:$rs1)),
             (load_instr (type (IMPLICIT_DEF)),
             AddrFI:$rs1,
             (mask_type zero_reg),
             VLMax, sew)>;

  // Store
  def : Pat<(store type:$rs2, GPR:$rs1),
            (store_instr reg_class:$rs2, GPR:$rs1,
             (mask_type zero_reg),
              VLMax, sew)>;
  def : Pat<(store type:$rs2, AddrFI:$rs1),
            (store_instr reg_class:$rs2, AddrFI:$rs1,
             (mask_type zero_reg),
             VLMax, sew)>;
}

foreach vti = AllVectors in
{
  defm : pat_load_store<vti.Vector, vti.Mask,
                        vti.SEW, vti.LMul, vti.RegClass>;
}

//===----------------------------------------------------------------------===//
// 12. Vector Integer Arithmetic Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 12.1. Vector Single-Width Integer Add and Subtract
//===----------------------------------------------------------------------===//

// Pseudo instructions.
defm PseudoVADD        : pseudo_binary_v_vv_vx_vi;

// Whole-register vector patterns.
defm "" : pat_vop_binary_common<add, "PseudoVADD", AllIntegerVectors>;

} // Predicates = [HasStdExtV]
