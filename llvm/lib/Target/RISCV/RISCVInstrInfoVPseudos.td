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

// X0 has special meaning for vsetvl/vsetvli.
//  rd | rs1 |   AVL value | Effect on vl
//--------------------------------------------------------------
// !X0 |  X0 |       VLMAX | Set vl to VLMAX
//  X0 |  X0 | Value in vl | Keep current vl, just change vtype.
def NoX0 : SDNodeXForm<undef,
[{
  auto *C = dyn_cast<ConstantSDNode>(N);
  if (C && C->isNullValue()) {
    SDLoc DL(N);
    return SDValue(CurDAG->getMachineNode(RISCV::ADDI, DL, Subtarget->getXLenVT(),
                   CurDAG->getRegister(RISCV::X0, Subtarget->getXLenVT()),
                   CurDAG->getTargetConstant(0, DL, Subtarget->getXLenVT())), 0);
  }
  return SDValue(N, 0);
}]>;

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

// We only model FPR32 for V instructions in RISCVInstrInfoV.td.
// FP16/FP32/FP64 registers are alias each other. Convert FPR16 and FPR64
// to FPR32 for V instructions is enough.
class ToFPR32<ValueType type, DAGOperand operand, string name> {
  dag ret = !cond(!eq(!cast<string>(operand), !cast<string>(FPR64)):
                      (EXTRACT_SUBREG !dag(type, [FPR64], [name]), sub_32),
                  !eq(!cast<string>(operand), !cast<string>(FPR16)):
                      (SUBREG_TO_REG (i16 -1), !dag(type, [FPR16], [name]), sub_16),
                  !eq(1, 1):
                      !dag(type, [operand], [name]));
}

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
// The definition should be consistent with `struct PseudoInfo` in
// RISCVBaseInfo.h.
class CONST8b<bits<8> val> {
  bits<8> V = val;
}
def InvalidIndex : CONST8b<0x80>;
class RISCVVPseudo {
  Pseudo Pseudo = !cast<Pseudo>(NAME); // Used as a key.
  Instruction BaseInstr;
  bits<8> VLIndex = InvalidIndex.V;
  bits<8> SEWIndex = InvalidIndex.V;
  bits<8> MergeOpIndex = InvalidIndex.V;
  bits<3> VLMul;
  bit HasDummyMask = 0;
}

// The actual table.
def RISCVVPseudosTable : GenericTable {
  let FilterClass = "RISCVVPseudo";
  let CppTypeName = "PseudoInfo";
  let Fields = [ "Pseudo", "BaseInstr", "VLIndex", "SEWIndex", "MergeOpIndex",
                 "VLMul", "HasDummyMask" ];
  let PrimaryKey = [ "Pseudo" ];
  let PrimaryKeyName = "getPseudoInfo";
}

def RISCVVIntrinsicsTable : GenericTable {
  let FilterClass = "RISCVVIntrinsic";
  let CppTypeName = "RISCVVIntrinsicInfo";
  let Fields = ["IntrinsicID", "ExtendOperand"];
  let PrimaryKey = ["IntrinsicID"];
  let PrimaryKeyName = "getRISCVVIntrinsicInfo";
}

//===----------------------------------------------------------------------===//
// Helpers to define the different pseudo instructions.
//===----------------------------------------------------------------------===//

class PseudoToVInst<string PseudoInst> {
  string VInst = !subst("_M8", "",
                 !subst("_M4", "",
                 !subst("_M2", "",
                 !subst("_M1", "",
                 !subst("_MF2", "",
                 !subst("_MF4", "",
                 !subst("_MF8", "",
                 !subst("_MASK", "",
                 !subst("Pseudo", "", PseudoInst)))))))));
}

// The destination vector register group for a masked vector instruction cannot
// overlap the source mask register (v0), unless the destination vector register
// is being written with a mask value (e.g., comparisons) or the scalar result
// of a reduction.
class GetVRegNoV0<VReg VRegClass> {
  VReg R = !cond(!eq(VRegClass, VR) : VRNoV0,
                 !eq(VRegClass, VRM2) : VRM2NoV0,
                 !eq(VRegClass, VRM4) : VRM4NoV0,
                 !eq(VRegClass, VRM8) : VRM8NoV0,
                 !eq(1, 1) : VRegClass);
}

class VPseudo<Instruction instr, LMULInfo m, dag outs, dag ins> :
      Pseudo<outs, ins, []>, RISCVVPseudo {
  let BaseInstr = instr;
  let VLMul = m.value;
}

class VPseudoBinaryNoMask<VReg RetClass,
                          VReg Op1Class,
                          DAGOperand Op2Class> :
        Pseudo<(outs RetClass:$rd),
               (ins Op1Class:$rs2, Op2Class:$rs1, GPR:$vl, ixlenimm:$sew), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Uses = [VL, VTYPE];
  let VLIndex = 3;
  let SEWIndex = 4;
  let HasDummyMask = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoBinaryMask<VReg RetClass,
                        VReg Op1Class,
                        DAGOperand Op2Class> :
        Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
                (ins GetVRegNoV0<RetClass>.R:$merge,
                     Op1Class:$rs2, Op2Class:$rs1,
                     VMaskOp:$vm, GPR:$vl, ixlenimm:$sew), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Constraints = "$rd = $merge";
  let Uses = [VL, VTYPE];
  let VLIndex = 5;
  let SEWIndex = 6;
  let MergeOpIndex = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

multiclass VPseudoBinary<VReg RetClass,
                         VReg Op1Class,
                         DAGOperand Op2Class,
                         LMULInfo MInfo> {
  let VLMul = MInfo.value in {
    def "_" # MInfo.MX : VPseudoBinaryNoMask<RetClass, Op1Class, Op2Class>;
    def "_" # MInfo.MX # "_MASK" : VPseudoBinaryMask<RetClass, Op1Class, Op2Class>;
  }
}

multiclass VPseudoBinaryV_VV {
  foreach m = MxList.m in
    defm _VV : VPseudoBinary<m.vrclass, m.vrclass, m.vrclass, m>;
}

multiclass VPseudoBinaryV_VX {
  foreach m = MxList.m in
    defm _VX : VPseudoBinary<m.vrclass, m.vrclass, GPR, m>;
}

multiclass VPseudoBinaryV_VI<Operand ImmType = simm5> {
  foreach m = MxList.m in
    defm _VI : VPseudoBinary<m.vrclass, m.vrclass, ImmType, m>;
}

multiclass VPseudoBinaryV_VV_VX_VI<Operand ImmType = simm5> {
  defm "" : VPseudoBinaryV_VV;
  defm "" : VPseudoBinaryV_VX;
  defm "" : VPseudoBinaryV_VI<ImmType>;
}

multiclass VPseudoBinaryV_VV_VX {
  defm "" : VPseudoBinaryV_VV;
  defm "" : VPseudoBinaryV_VX;
}

multiclass VPseudoBinaryV_VX_VI<Operand ImmType = simm5> {
  defm "" : VPseudoBinaryV_VX;
  defm "" : VPseudoBinaryV_VI<ImmType>;
}

//===----------------------------------------------------------------------===//
// Helpers to define the different patterns.
//===----------------------------------------------------------------------===//
class VPatBinarySDNode<SDNode vop,
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

multiclass VPatBinarySDNode<SDNode vop, string instruction_name>
{
  foreach vti = AllIntegerVectors in
    def : VPatBinarySDNode<vop, instruction_name,
                           vti.Vector, vti.Vector, vti.Mask, vti.SEW,
                           vti.LMul, vti.RegClass, vti.RegClass>;
}

class VPatBinaryNoMask<string intrinsic_name,
                       string inst,
                       string kind,
                       ValueType result_type,
                       ValueType op1_type,
                       ValueType op2_type,
                       int sew,
                       LMULInfo vlmul,
                       VReg op1_reg_class,
                       DAGOperand op2_kind> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                   (op1_type op1_reg_class:$rs1),
                   (op2_type op2_kind:$rs2),
                   (XLenVT GPR:$vl))),
                   (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX)
                   (op1_type op1_reg_class:$rs1),
                   ToFPR32<op2_type, op2_kind, "rs2">.ret,
                   (NoX0 GPR:$vl), sew)>;

class VPatBinaryMask<string intrinsic_name,
                     string inst,
                     string kind,
                     ValueType result_type,
                     ValueType op1_type,
                     ValueType op2_type,
                     ValueType mask_type,
                     int sew,
                     LMULInfo vlmul,
                     VReg result_reg_class,
                     VReg op1_reg_class,
                     DAGOperand op2_kind> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic_name#"_mask")
                   (result_type result_reg_class:$merge),
                   (op1_type op1_reg_class:$rs1),
                   (op2_type op2_kind:$rs2),
                   (mask_type V0),
                   (XLenVT GPR:$vl))),
                   (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX#"_MASK")
                   (result_type result_reg_class:$merge),
                   (op1_type op1_reg_class:$rs1),
                   ToFPR32<op2_type, op2_kind, "rs2">.ret,
                   (mask_type V0), (NoX0 GPR:$vl), sew)>;

multiclass VPatBinary<string intrinsic,
                      string inst,
                      string kind,
                      ValueType result_type,
                      ValueType op1_type,
                      ValueType op2_type,
                      ValueType mask_type,
                      int sew,
                      LMULInfo vlmul,
                      VReg result_reg_class,
                      VReg op1_reg_class,
                      DAGOperand op2_kind>
{
  def : VPatBinaryNoMask<intrinsic, inst, kind, result_type, op1_type, op2_type,
                         sew, vlmul, op1_reg_class, op2_kind>;
  def : VPatBinaryMask<intrinsic, inst, kind, result_type, op1_type, op2_type,
                       mask_type, sew, vlmul, result_reg_class, op1_reg_class,
                       op2_kind>;
}

multiclass VPatBinaryV_VV<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    defm : VPatBinary<intrinsic, instruction, "VV",
                      vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                      vti.SEW, vti.LMul, vti.RegClass,
                      vti.RegClass, vti.RegClass>;
}

multiclass VPatBinaryV_VX<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    defm : VPatBinary<intrinsic, instruction, "VX",
                      vti.Vector, vti.Vector, XLenVT, vti.Mask,
                      vti.SEW, vti.LMul, vti.RegClass,
                      vti.RegClass, GPR>;
}

multiclass VPatBinaryV_VI<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist, Operand imm_type> {
  foreach vti = vtilist in
    defm : VPatBinary<intrinsic, instruction, "VI",
                      vti.Vector, vti.Vector, XLenVT, vti.Mask,
                      vti.SEW, vti.LMul, vti.RegClass,
                      vti.RegClass, imm_type>;
}

multiclass VPatBinaryV_VV_VX_VI<string intrinsic, string instruction,
                                list<VTypeInfo> vtilist>
{
  defm "" : VPatBinaryV_VV<intrinsic, instruction, vtilist>;
  defm "" : VPatBinaryV_VX<intrinsic, instruction, vtilist>;
  defm "" : VPatBinaryV_VI<intrinsic, instruction, vtilist, simm5>;
}

multiclass VPatBinaryV_VV_VX<string intrinsic, string instruction,
                             list<VTypeInfo> vtilist>
{
  defm "" : VPatBinaryV_VV<intrinsic, instruction, vtilist>;
  defm "" : VPatBinaryV_VX<intrinsic, instruction, vtilist>;
}

multiclass VPatBinaryV_VX_VI<string intrinsic, string instruction,
                             list<VTypeInfo> vtilist>
{
  defm "" : VPatBinaryV_VX<intrinsic, instruction, vtilist>;
  defm "" : VPatBinaryV_VI<intrinsic, instruction, vtilist, simm5>;
}

//===----------------------------------------------------------------------===//
// Pseudo instructions and patterns.
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtV] in {

//===----------------------------------------------------------------------===//
// Pseudo Instructions for CodeGen
//===----------------------------------------------------------------------===//
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  def PseudoVMV1R_V : VPseudo<VMV1R_V, V_M1, (outs VR:$vd), (ins VR:$vs2)>;
  def PseudoVMV2R_V : VPseudo<VMV2R_V, V_M2, (outs VRM2:$vd), (ins VRM2:$vs2)>;
  def PseudoVMV4R_V : VPseudo<VMV4R_V, V_M4, (outs VRM4:$vd), (ins VRM4:$vs2)>;
  def PseudoVMV8R_V : VPseudo<VMV8R_V, V_M8, (outs VRM8:$vd), (ins VRM8:$vs2)>;
}

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
// Pseudo Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 12. Vector Integer Arithmetic Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 12.1. Vector Single-Width Integer Add and Subtract
//===----------------------------------------------------------------------===//
defm PseudoVADD        : VPseudoBinaryV_VV_VX_VI;
defm PseudoVSUB        : VPseudoBinaryV_VV_VX;
defm PseudoVRSUB       : VPseudoBinaryV_VX_VI;

//===----------------------------------------------------------------------===//
// Patterns.
//===----------------------------------------------------------------------===//

// Whole-register vector patterns.
defm "" : VPatBinarySDNode<add, "PseudoVADD">;

//===----------------------------------------------------------------------===//
// 12. Vector Integer Arithmetic Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 12.1. Vector Single-Width Integer Add and Subtract
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryV_VV_VX_VI<"int_riscv_vadd", "PseudoVADD", AllIntegerVectors>;
defm "" : VPatBinaryV_VV_VX<"int_riscv_vsub", "PseudoVSUB", AllIntegerVectors>;
defm "" : VPatBinaryV_VX_VI<"int_riscv_vrsub", "PseudoVRSUB", AllIntegerVectors>;

} // Predicates = [HasStdExtV]
