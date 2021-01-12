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

def riscv_vmv_x_s : SDNode<"RISCVISD::VMV_X_S",
                           SDTypeProfile<1, 1, [SDTCisInt<0>, SDTCisVec<1>,
                                                SDTCisInt<1>]>>;

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

def DecImm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue() - 1, SDLoc(N),
                                   N->getValueType(0));
}]>;

//===----------------------------------------------------------------------===//
// Utilities.
//===----------------------------------------------------------------------===//

// This class describes information associated to the LMUL.
class LMULInfo<int lmul, VReg regclass, VReg wregclass,
               VReg f2regclass, VReg f4regclass, VReg f8regclass, string mx> {
  bits<3> value = lmul; // This is encoded as the vlmul field of vtype.
  VReg vrclass = regclass;
  VReg wvrclass = wregclass;
  VReg f8vrclass = f8regclass;
  VReg f4vrclass = f4regclass;
  VReg f2vrclass = f2regclass;
  string MX = mx;
}

// Associate LMUL with tablegen records of register classes.
def V_M1  : LMULInfo<0b000,   VR,        VRM2,   VR,   VR, VR, "M1">;
def V_M2  : LMULInfo<0b001, VRM2,        VRM4,   VR,   VR, VR, "M2">;
def V_M4  : LMULInfo<0b010, VRM4,        VRM8, VRM2,   VR, VR, "M4">;
def V_M8  : LMULInfo<0b011, VRM8,/*NoVReg*/VR, VRM4, VRM2, VR, "M8">;

def V_MF8 : LMULInfo<0b101, VR, VR,/*NoVReg*/VR,/*NoVReg*/VR,/*NoVReg*/VR, "MF8">;
def V_MF4 : LMULInfo<0b110, VR, VR,          VR,/*NoVReg*/VR,/*NoVReg*/VR, "MF4">;
def V_MF2 : LMULInfo<0b111, VR, VR,          VR,          VR,/*NoVReg*/VR, "MF2">;

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

class VTypeInfo<ValueType Vec, ValueType Mas, int Sew, VReg Reg, LMULInfo M,
                ValueType Scal = XLenVT, RegisterClass ScalarReg = GPR>
{
  ValueType Vector = Vec;
  ValueType Mask = Mas;
  int SEW = Sew;
  VReg RegClass = Reg;
  LMULInfo LMul = M;
  ValueType Scalar = Scal;
  RegisterClass ScalarRegClass = ScalarReg;
}

class GroupVTypeInfo<ValueType Vec, ValueType VecM1, ValueType Mas, int Sew,
                     VReg Reg, LMULInfo M, ValueType Scal = XLenVT,
                     RegisterClass ScalarReg = GPR>
    : VTypeInfo<Vec, Mas, Sew, Reg, M, Scal, ScalarReg>
{
  ValueType VectorM1 = VecM1;
}

defset list<VTypeInfo> AllVectors = {
  defset list<VTypeInfo> AllIntegerVectors = {
    defset list<VTypeInfo> NoGroupIntegerVectors = {
      def VI8MF8: VTypeInfo<vint8mf8_t,  vbool64_t,  8, VR, V_MF8>;
      def VI8MF4: VTypeInfo<vint8mf4_t,  vbool32_t,  8, VR, V_MF4>;
      def VI8MF2: VTypeInfo<vint8mf2_t,  vbool16_t,  8, VR, V_MF2>;
      def VI8M1: VTypeInfo<vint8m1_t,   vbool8_t,   8, VR, V_M1>;
      def VI16MF4: VTypeInfo<vint16mf4_t, vbool64_t, 16, VR, V_MF4>;
      def VI16MF2: VTypeInfo<vint16mf2_t, vbool32_t, 16, VR, V_MF2>;
      def VI16M1: VTypeInfo<vint16m1_t,  vbool16_t, 16, VR, V_M1>;
      def VI32MF2: VTypeInfo<vint32mf2_t, vbool64_t, 32, VR, V_MF2>;
      def VI32M1: VTypeInfo<vint32m1_t,  vbool32_t, 32, VR, V_M1>;
      def VI64M1: VTypeInfo<vint64m1_t,  vbool64_t, 64, VR, V_M1>;
    }
    defset list<GroupVTypeInfo> GroupIntegerVectors = {
      def VI8M2: GroupVTypeInfo<vint8m2_t, vint8m1_t, vbool4_t, 8, VRM2, V_M2>;
      def VI8M4: GroupVTypeInfo<vint8m4_t, vint8m1_t, vbool2_t, 8, VRM4, V_M4>;
      def VI8M8: GroupVTypeInfo<vint8m8_t, vint8m1_t, vbool1_t, 8, VRM8, V_M8>;

      def VI16M2: GroupVTypeInfo<vint16m2_t,vint16m1_t,vbool8_t, 16,VRM2, V_M2>;
      def VI16M4: GroupVTypeInfo<vint16m4_t,vint16m1_t,vbool4_t, 16,VRM4, V_M4>;
      def VI16M8: GroupVTypeInfo<vint16m8_t,vint16m1_t,vbool2_t, 16,VRM8, V_M8>;

      def VI32M2: GroupVTypeInfo<vint32m2_t,vint32m1_t,vbool16_t,32,VRM2, V_M2>;
      def VI32M4: GroupVTypeInfo<vint32m4_t,vint32m1_t,vbool8_t, 32,VRM4, V_M4>;
      def VI32M8: GroupVTypeInfo<vint32m8_t,vint32m1_t,vbool4_t, 32,VRM8, V_M8>;

      def VI64M2: GroupVTypeInfo<vint64m2_t,vint64m1_t,vbool32_t,64,VRM2, V_M2>;
      def VI64M4: GroupVTypeInfo<vint64m4_t,vint64m1_t,vbool16_t,64,VRM4, V_M4>;
      def VI64M8: GroupVTypeInfo<vint64m8_t,vint64m1_t,vbool8_t, 64,VRM8, V_M8>;
    }
  }

  defset list<VTypeInfo> AllFloatVectors = {
    defset list<VTypeInfo> NoGroupFloatVectors = {
      def VF16MF4: VTypeInfo<vfloat16mf4_t, vbool64_t, 16, VR, V_MF4, f16, FPR16>;
      def VF16MF2: VTypeInfo<vfloat16mf2_t, vbool32_t, 16, VR, V_MF2, f16, FPR16>;
      def VF16M1:  VTypeInfo<vfloat16m1_t,  vbool16_t, 16, VR, V_M1,  f16, FPR16>;

      def VF32MF2: VTypeInfo<vfloat32mf2_t,vbool64_t, 32, VR, V_MF2, f32, FPR32>;
      def VF32M1:  VTypeInfo<vfloat32m1_t, vbool32_t, 32, VR, V_M1,  f32, FPR32>;

      def VF64M1: VTypeInfo<vfloat64m1_t, vbool64_t, 64, VR, V_M1, f64, FPR64>;
    }

    defset list<GroupVTypeInfo> GroupFloatVectors = {
      def VF16M2: GroupVTypeInfo<vfloat16m2_t, vfloat16m1_t, vbool8_t, 16,
                                  VRM2, V_M2, f16, FPR16>;
      def VF16M4: GroupVTypeInfo<vfloat16m4_t, vfloat16m1_t, vbool4_t, 16,
                                  VRM4, V_M4, f16, FPR16>;
      def VF16M8: GroupVTypeInfo<vfloat16m8_t, vfloat16m1_t, vbool2_t, 16,
                                  VRM8, V_M8, f16, FPR16>;

      def VF32M2: GroupVTypeInfo<vfloat32m2_t, vfloat32m1_t, vbool16_t, 32,
                                  VRM2, V_M2, f32, FPR32>;
      def VF32M4: GroupVTypeInfo<vfloat32m4_t, vfloat32m1_t, vbool8_t,  32,
                                  VRM4, V_M4, f32, FPR32>;
      def VF32M8: GroupVTypeInfo<vfloat32m8_t, vfloat32m1_t, vbool4_t,  32,
                                  VRM8, V_M8, f32, FPR32>;

      def VF64M2: GroupVTypeInfo<vfloat64m2_t, vfloat64m1_t, vbool32_t, 64,
                                  VRM2, V_M2, f64, FPR64>;
      def VF64M4: GroupVTypeInfo<vfloat64m4_t, vfloat64m1_t, vbool16_t, 64,
                                  VRM4, V_M4, f64, FPR64>;
      def VF64M8: GroupVTypeInfo<vfloat64m8_t, vfloat64m1_t, vbool8_t,  64,
                                  VRM8, V_M8, f64, FPR64>;
    }
  }
}

// This functor is used to obtain the int vector type that has the same SEW and
// multiplier as the input parameter type
class GetIntVTypeInfo<VTypeInfo vti>
{
  // Equivalent integer vector type. Eg.
  //   VI8M1 → VI8M1 (identity)
  //   VF64M4 → VI64M4
  VTypeInfo Vti = !cast<VTypeInfo>(!subst("VF", "VI", !cast<string>(vti)));
}

class MTypeInfo<ValueType Mas, LMULInfo M, string Bx> {
  ValueType Mask = Mas;
  // {SEW, VLMul} values set a valid VType to deal with this mask type.
  // we assume SEW=8 and set corresponding LMUL.
  int SEW = 8;
  LMULInfo LMul = M;
  string BX = Bx; // Appendix of mask operations.
}

defset list<MTypeInfo> AllMasks = {
  // vbool<n>_t, <n> = SEW/LMUL, we assume SEW=8 and corresponding LMUL.
  def : MTypeInfo<vbool64_t, V_MF8, "B1">;
  def : MTypeInfo<vbool32_t, V_MF4, "B2">;
  def : MTypeInfo<vbool16_t, V_MF2, "B4">;
  def : MTypeInfo<vbool8_t, V_M1, "B8">;
  def : MTypeInfo<vbool4_t, V_M2, "B16">;
  def : MTypeInfo<vbool2_t, V_M4, "B32">;
  def : MTypeInfo<vbool1_t, V_M8, "B64">;
}

class VTypeInfoToWide<VTypeInfo vti, VTypeInfo wti>
{
  VTypeInfo Vti = vti;
  VTypeInfo Wti = wti;
}

class VTypeInfoToFraction<VTypeInfo vti, VTypeInfo fti>
{
  VTypeInfo Vti = vti;
  VTypeInfo Fti = fti;
}

defset list<VTypeInfoToWide> AllWidenableIntVectors = {
  def : VTypeInfoToWide<VI8MF8,  VI16MF4>;
  def : VTypeInfoToWide<VI8MF4,  VI16MF2>;
  def : VTypeInfoToWide<VI8MF2,  VI16M1>;
  def : VTypeInfoToWide<VI8M1,   VI16M2>;
  def : VTypeInfoToWide<VI8M2,   VI16M4>;
  def : VTypeInfoToWide<VI8M4,   VI16M8>;

  def : VTypeInfoToWide<VI16MF4, VI32MF2>;
  def : VTypeInfoToWide<VI16MF2, VI32M1>;
  def : VTypeInfoToWide<VI16M1,  VI32M2>;
  def : VTypeInfoToWide<VI16M2,  VI32M4>;
  def : VTypeInfoToWide<VI16M4,  VI32M8>;

  def : VTypeInfoToWide<VI32MF2, VI64M1>;
  def : VTypeInfoToWide<VI32M1,  VI64M2>;
  def : VTypeInfoToWide<VI32M2,  VI64M4>;
  def : VTypeInfoToWide<VI32M4,  VI64M8>;
}

defset list<VTypeInfoToWide> AllWidenableFloatVectors = {
  def : VTypeInfoToWide<VF16MF4, VF32MF2>;
  def : VTypeInfoToWide<VF16MF2, VF32M1>;
  def : VTypeInfoToWide<VF16M1, VF32M2>;
  def : VTypeInfoToWide<VF16M2, VF32M4>;
  def : VTypeInfoToWide<VF16M4, VF32M8>;

  def : VTypeInfoToWide<VF32MF2, VF64M1>;
  def : VTypeInfoToWide<VF32M1, VF64M2>;
  def : VTypeInfoToWide<VF32M2, VF64M4>;
  def : VTypeInfoToWide<VF32M4, VF64M8>;
}

defset list<VTypeInfoToFraction> AllFractionableVF2IntVectors = {
  def : VTypeInfoToFraction<VI16MF4, VI8MF8>;
  def : VTypeInfoToFraction<VI16MF2, VI8MF4>;
  def : VTypeInfoToFraction<VI16M1, VI8MF2>;
  def : VTypeInfoToFraction<VI16M2, VI8M1>;
  def : VTypeInfoToFraction<VI16M4, VI8M2>;
  def : VTypeInfoToFraction<VI16M8, VI8M4>;
  def : VTypeInfoToFraction<VI32MF2, VI16MF4>;
  def : VTypeInfoToFraction<VI32M1, VI16MF2>;
  def : VTypeInfoToFraction<VI32M2, VI16M1>;
  def : VTypeInfoToFraction<VI32M4, VI16M2>;
  def : VTypeInfoToFraction<VI32M8, VI16M4>;
  def : VTypeInfoToFraction<VI64M1, VI32MF2>;
  def : VTypeInfoToFraction<VI64M2, VI32M1>;
  def : VTypeInfoToFraction<VI64M4, VI32M2>;
  def : VTypeInfoToFraction<VI64M8, VI32M4>;
}

defset list<VTypeInfoToFraction> AllFractionableVF4IntVectors = {
  def : VTypeInfoToFraction<VI32MF2, VI8MF8>;
  def : VTypeInfoToFraction<VI32M1, VI8MF4>;
  def : VTypeInfoToFraction<VI32M2, VI8MF2>;
  def : VTypeInfoToFraction<VI32M4, VI8M1>;
  def : VTypeInfoToFraction<VI32M8, VI8M2>;
  def : VTypeInfoToFraction<VI64M1, VI16MF4>;
  def : VTypeInfoToFraction<VI64M2, VI16MF2>;
  def : VTypeInfoToFraction<VI64M4, VI16M1>;
  def : VTypeInfoToFraction<VI64M8, VI16M2>;
}

defset list<VTypeInfoToFraction> AllFractionableVF8IntVectors = {
  def : VTypeInfoToFraction<VI64M1, VI8MF8>;
  def : VTypeInfoToFraction<VI64M2, VI8MF4>;
  def : VTypeInfoToFraction<VI64M4, VI8MF2>;
  def : VTypeInfoToFraction<VI64M8, VI8M1>;
}

defset list<VTypeInfoToWide> AllWidenableIntToFloatVectors = {
  def : VTypeInfoToWide<VI8MF8, VF16MF4>;
  def : VTypeInfoToWide<VI8MF4, VF16MF2>;
  def : VTypeInfoToWide<VI8MF2, VF16M1>;
  def : VTypeInfoToWide<VI8M1, VF16M2>;
  def : VTypeInfoToWide<VI8M2, VF16M4>;
  def : VTypeInfoToWide<VI8M4, VF16M8>;

  def : VTypeInfoToWide<VI16MF4, VF32MF2>;
  def : VTypeInfoToWide<VI16MF2, VF32M1>;
  def : VTypeInfoToWide<VI16M1, VF32M2>;
  def : VTypeInfoToWide<VI16M2, VF32M4>;
  def : VTypeInfoToWide<VI16M4, VF32M8>;

  def : VTypeInfoToWide<VI32MF2, VF64M1>;
  def : VTypeInfoToWide<VI32M1, VF64M2>;
  def : VTypeInfoToWide<VI32M2, VF64M4>;
  def : VTypeInfoToWide<VI32M4, VF64M8>;
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
}

// The actual table.
def RISCVVPseudosTable : GenericTable {
  let FilterClass = "RISCVVPseudo";
  let CppTypeName = "PseudoInfo";
  let Fields = [ "Pseudo", "BaseInstr" ];
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
                 !subst("_B1", "",
                 !subst("_B2", "",
                 !subst("_B4", "",
                 !subst("_B8", "",
                 !subst("_B16", "",
                 !subst("_B32", "",
                 !subst("_B64", "",
                 !subst("_MASK", "",
                 !subst("Pseudo", "", PseudoInst))))))))))))))));
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

// Join strings in list using separator and ignoring empty elements
class Join<list<string> strings, string separator> {
  string ret = !foldl(!head(strings), !tail(strings), a, b,
                      !cond(
                        !and(!empty(a), !empty(b)) : "",
                        !empty(a) : b,
                        !empty(b) : a,
                        1 : a#separator#b));
}

class VPseudo<Instruction instr, LMULInfo m, dag outs, dag ins> :
      Pseudo<outs, ins, []>, RISCVVPseudo {
  let BaseInstr = instr;
  let VLMul = m.value;
}

class VPseudoUSLoadNoMask<VReg RetClass>:
      Pseudo<(outs RetClass:$rd),
             (ins GPR:$rs1, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoUSLoadMask<VReg RetClass>:
      Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
              (ins GetVRegNoV0<RetClass>.R:$merge,
                   GPR:$rs1,
                   VMaskOp:$vm, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Constraints = "$rd = $merge";
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoSLoadNoMask<VReg RetClass>:
      Pseudo<(outs RetClass:$rd),
             (ins GPR:$rs1, GPR:$rs2, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoSLoadMask<VReg RetClass>:
      Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
              (ins GetVRegNoV0<RetClass>.R:$merge,
                   GPR:$rs1, GPR:$rs2,
                   VMaskOp:$vm, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Constraints = "$rd = $merge";
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoILoadNoMask<VReg RetClass, VReg IdxClass>:
      Pseudo<(outs RetClass:$rd),
             (ins GPR:$rs1, IdxClass:$rs2, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoILoadMask<VReg RetClass, VReg IdxClass>:
      Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
              (ins GetVRegNoV0<RetClass>.R:$merge,
                   GPR:$rs1, IdxClass:$rs2,
                   VMaskOp:$vm, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Constraints = "$rd = $merge";
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoUSStoreNoMask<VReg StClass>:
      Pseudo<(outs),
              (ins StClass:$rd, GPR:$rs1, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoUSStoreMask<VReg StClass>:
      Pseudo<(outs),
              (ins StClass:$rd, GPR:$rs1, VMaskOp:$vm, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoSStoreNoMask<VReg StClass>:
      Pseudo<(outs),
              (ins StClass:$rd, GPR:$rs1, GPR:$rs2, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoSStoreMask<VReg StClass>:
      Pseudo<(outs),
              (ins StClass:$rd, GPR:$rs1, GPR:$rs2, VMaskOp:$vm, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

// Unary instruction that is never masked so HasDummyMask=0.
class VPseudoUnaryNoDummyMask<VReg RetClass,
                              DAGOperand Op2Class> :
        Pseudo<(outs RetClass:$rd),
               (ins Op2Class:$rs1, GPR:$vl, ixlenimm:$sew), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoNullaryNoMask<VReg RegClass>:
      Pseudo<(outs RegClass:$rd),
             (ins GPR:$vl, ixlenimm:$sew),
             []>, RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoNullaryMask<VReg RegClass>:
      Pseudo<(outs GetVRegNoV0<RegClass>.R:$rd),
             (ins GetVRegNoV0<RegClass>.R:$merge, VMaskOp:$vm, GPR:$vl,
              ixlenimm:$sew), []>, RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Constraints ="$rd = $merge";
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

// Nullary for pseudo instructions. They are expanded in
// RISCVExpandPseudoInsts pass.
class VPseudoNullaryPseudoM<string BaseInst>
       : Pseudo<(outs VR:$rd), (ins GPR:$vl, ixlenimm:$sew), []>,
       RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  // BaseInstr is not used in RISCVExpandPseudoInsts pass.
  // Just fill a corresponding real v-inst to pass tablegen check.
  let BaseInstr = !cast<Instruction>(BaseInst);
}

// RetClass could be GPR or VReg.
class VPseudoUnaryNoMask<DAGOperand RetClass, VReg OpClass, string Constraint = ""> :
        Pseudo<(outs RetClass:$rd),
               (ins OpClass:$rs2, GPR:$vl, ixlenimm:$sew), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Constraints = Constraint;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoUnaryMask<VReg RetClass, VReg OpClass, string Constraint = ""> :
        Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
               (ins GetVRegNoV0<RetClass>.R:$merge, OpClass:$rs2,
                    VMaskOp:$vm, GPR:$vl, ixlenimm:$sew), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Constraints = Join<[Constraint, "$rd = $merge"], ",">.ret;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

// mask unary operation without maskedoff
class VPseudoMaskUnarySOutMask:
        Pseudo<(outs GPR:$rd),
               (ins VR:$rs1, VMaskOp:$vm, GPR:$vl, ixlenimm:$sew), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

// Masked mask operation have no $rd=$merge constraints
class VPseudoUnaryMOutMask:
        Pseudo<(outs VR:$rd),
               (ins VR:$merge, VR:$rs1, VMaskOp:$vm, GPR:$vl, ixlenimm:$sew), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Constraints = "$rd = $merge";
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

// Mask can be V0~V31
class VPseudoUnaryAnyMask<VReg RetClass,
                          VReg Op1Class> :
      Pseudo<(outs RetClass:$rd),
             (ins RetClass:$merge,
                  Op1Class:$rs2,
                  VR:$vm, GPR:$vl, ixlenimm:$sew),
             []>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Constraints = "@earlyclobber $rd, $rd = $merge";
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoBinaryNoMask<VReg RetClass,
                          VReg Op1Class,
                          DAGOperand Op2Class,
                          string Constraint> :
        Pseudo<(outs RetClass:$rd),
               (ins Op1Class:$rs2, Op2Class:$rs1, GPR:$vl, ixlenimm:$sew), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Constraints = Constraint;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoIStoreNoMask<VReg StClass, VReg IdxClass>:
      Pseudo<(outs),
              (ins StClass:$rd, GPR:$rs1, IdxClass:$rs2, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoIStoreMask<VReg StClass, VReg IdxClass>:
      Pseudo<(outs),
              (ins StClass:$rd, GPR:$rs1, IdxClass:$rs2, VMaskOp:$vm, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoBinaryMask<VReg RetClass,
                        VReg Op1Class,
                        DAGOperand Op2Class,
                        string Constraint> :
        Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
                (ins GetVRegNoV0<RetClass>.R:$merge,
                     Op1Class:$rs2, Op2Class:$rs1,
                     VMaskOp:$vm, GPR:$vl, ixlenimm:$sew), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Constraints = Join<[Constraint, "$rd = $merge"], ",">.ret;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoBinaryCarryIn<VReg RetClass,
                           VReg Op1Class,
                           DAGOperand Op2Class,
                           LMULInfo MInfo,
                           bit CarryIn,
                           string Constraint> :
        Pseudo<(outs RetClass:$rd),
               !if(CarryIn,
                  (ins Op1Class:$rs2, Op2Class:$rs1, VMV0:$carry, GPR:$vl,
                       ixlenimm:$sew),
                  (ins Op1Class:$rs2, Op2Class:$rs1, GPR:$vl, ixlenimm:$sew)), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Constraints = Constraint;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 0;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
  let VLMul = MInfo.value;
}

class VPseudoTernaryNoMask<VReg RetClass,
                           VReg Op1Class,
                           DAGOperand Op2Class,
                           string Constraint> :
        Pseudo<(outs RetClass:$rd),
               (ins RetClass:$rs3, Op1Class:$rs1, Op2Class:$rs2,
                    GPR:$vl, ixlenimm:$sew),
               []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Constraints = Join<[Constraint, "$rd = $rs3"], ",">.ret;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let HasDummyMask = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

multiclass VPseudoUSLoad {
  foreach lmul = MxList.m in {
    defvar LInfo = lmul.MX;
    defvar vreg = lmul.vrclass;
    let VLMul = lmul.value in {
      def "_V_" # LInfo : VPseudoUSLoadNoMask<vreg>;
      def "_V_" # LInfo # "_MASK" : VPseudoUSLoadMask<vreg>;
    }
  }
}

multiclass VPseudoSLoad {
  foreach lmul = MxList.m in {
    defvar LInfo = lmul.MX;
    defvar vreg = lmul.vrclass;
    let VLMul = lmul.value in {
      def "_V_" # LInfo : VPseudoSLoadNoMask<vreg>;
      def "_V_" # LInfo # "_MASK" : VPseudoSLoadMask<vreg>;
    }
  }
}

multiclass VPseudoILoad {
  foreach lmul = MxList.m in
  foreach idx_lmul = MxList.m in {
    defvar LInfo = lmul.MX;
    defvar Vreg = lmul.vrclass;
    defvar IdxLInfo = idx_lmul.MX;
    defvar IdxVreg = idx_lmul.vrclass;
    let VLMul = lmul.value in {
      def "_V_" # IdxLInfo # "_" # LInfo : VPseudoILoadNoMask<Vreg, IdxVreg>;
      def "_V_" # IdxLInfo # "_" # LInfo # "_MASK" : VPseudoILoadMask<Vreg, IdxVreg>;
    }
  }
}

multiclass VPseudoUSStore {
  foreach lmul = MxList.m in {
    defvar LInfo = lmul.MX;
    defvar vreg = lmul.vrclass;
    let VLMul = lmul.value in {
      def "_V_" # LInfo : VPseudoUSStoreNoMask<vreg>;
      def "_V_" # LInfo # "_MASK" : VPseudoUSStoreMask<vreg>;
    }
  }
}

multiclass VPseudoSStore {
  foreach lmul = MxList.m in {
    defvar LInfo = lmul.MX;
    defvar vreg = lmul.vrclass;
    let VLMul = lmul.value in {
      def "_V_" # LInfo : VPseudoSStoreNoMask<vreg>;
      def "_V_" # LInfo # "_MASK" : VPseudoSStoreMask<vreg>;
    }
  }
}

multiclass VPseudoIStore {
  foreach lmul = MxList.m in
  foreach idx_lmul = MxList.m in {
    defvar LInfo = lmul.MX;
    defvar Vreg = lmul.vrclass;
    defvar IdxLInfo = idx_lmul.MX;
    defvar IdxVreg = idx_lmul.vrclass;
    let VLMul = lmul.value in {
      def "_V_" # IdxLInfo # "_" # LInfo : VPseudoIStoreNoMask<Vreg, IdxVreg>;
      def "_V_" # IdxLInfo # "_" # LInfo # "_MASK" : VPseudoIStoreMask<Vreg, IdxVreg>;
    }
  }
}

multiclass VPseudoUnaryS_M {
  foreach mti = AllMasks in
  {
    let VLMul = mti.LMul.value in {
      def "_M_" # mti.BX : VPseudoUnaryNoMask<GPR, VR>;
      def "_M_" # mti.BX # "_MASK" : VPseudoMaskUnarySOutMask;
    }
  }
}

multiclass VPseudoUnaryM_M {
  defvar constraint = "@earlyclobber $rd";
  foreach mti = AllMasks in
  {
    let VLMul = mti.LMul.value in {
      def "_M_" # mti.BX : VPseudoUnaryNoMask<VR, VR, constraint>;
      def "_M_" # mti.BX # "_MASK" : VPseudoUnaryMask<VR, VR, constraint>;
    }
  }
}

multiclass VPseudoMaskNullaryV {
  foreach m = MxList.m in {
    let VLMul = m.value in {
      def "_V_" # m.MX : VPseudoNullaryNoMask<m.vrclass>;
      def "_V_" # m.MX # "_MASK" : VPseudoNullaryMask<m.vrclass>;
    }
  }
}

multiclass VPseudoNullaryPseudoM <string BaseInst> {
  foreach mti = AllMasks in {
    let VLMul = mti.LMul.value in {
      def "_M_" # mti.BX : VPseudoNullaryPseudoM<BaseInst # "_MM">;
    }
  }
}

multiclass VPseudoUnaryV_M {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxList.m in {
    let VLMul = m.value in {
      def "_" # m.MX : VPseudoUnaryNoMask<m.vrclass, VR, constraint>;
      def "_" # m.MX # "_MASK" : VPseudoUnaryMask<m.vrclass, VR, constraint>;
    }
  }
}

multiclass VPseudoUnaryV_V_AnyMask {
  foreach m = MxList.m in {
    let VLMul = m.value in
      def _VM # "_" # m.MX # "_MASK" : VPseudoUnaryAnyMask<m.vrclass, m.vrclass>;
  }
}

multiclass VPseudoBinary<VReg RetClass,
                         VReg Op1Class,
                         DAGOperand Op2Class,
                         LMULInfo MInfo,
                         string Constraint = ""> {
  let VLMul = MInfo.value in {
    def "_" # MInfo.MX : VPseudoBinaryNoMask<RetClass, Op1Class, Op2Class,
                                             Constraint>;
    def "_" # MInfo.MX # "_MASK" : VPseudoBinaryMask<RetClass, Op1Class, Op2Class,
                                                     Constraint>;
  }
}

multiclass VPseudoBinaryV_VV<string Constraint = ""> {
  foreach m = MxList.m in
    defm _VV : VPseudoBinary<m.vrclass, m.vrclass, m.vrclass, m, Constraint>;
}

multiclass VPseudoBinaryV_VX<bit IsFloat, string Constraint = ""> {
  foreach m = MxList.m in
    defm !if(IsFloat, "_VF", "_VX") : VPseudoBinary<m.vrclass, m.vrclass,
                             !if(IsFloat, FPR32, GPR), m, Constraint>;
}

multiclass VPseudoBinaryV_VI<Operand ImmType = simm5, string Constraint = ""> {
  foreach m = MxList.m in
    defm _VI : VPseudoBinary<m.vrclass, m.vrclass, ImmType, m, Constraint>;
}

multiclass VPseudoBinaryM_MM {
  foreach m = MxList.m in
    let VLMul = m.value in {
      def "_MM_" # m.MX : VPseudoBinaryNoMask<VR, VR, VR, "">;
    }
}

// We use earlyclobber here due to
// * The destination EEW is smaller than the source EEW and the overlap is
//   in the lowest-numbered part of the source register group is legal.
//   Otherwise, it is illegal.
// * The destination EEW is greater than the source EEW, the source EMUL is
//   at least 1, and the overlap is in the highest-numbered part of the
//   destination register group is legal. Otherwise, it is illegal.
multiclass VPseudoBinaryW_VV {
  foreach m = MxList.m[0-5] in
    defm _VV : VPseudoBinary<m.wvrclass, m.vrclass, m.vrclass, m,
                             "@earlyclobber $rd">;
}

multiclass VPseudoBinaryW_VX<bit IsFloat> {
  foreach m = MxList.m[0-5] in
    defm !if(IsFloat, "_VF", "_VX") : VPseudoBinary<m.wvrclass, m.vrclass,
                             !if(IsFloat, FPR32, GPR), m,
                             "@earlyclobber $rd">;
}

multiclass VPseudoBinaryW_WV {
  foreach m = MxList.m[0-5] in
    defm _WV : VPseudoBinary<m.wvrclass, m.wvrclass, m.vrclass, m,
                             "@earlyclobber $rd">;
}

multiclass VPseudoBinaryW_WX<bit IsFloat> {
  foreach m = MxList.m[0-5] in
    defm !if(IsFloat, "_WF", "_WX") : VPseudoBinary<m.wvrclass, m.wvrclass,
                             !if(IsFloat, FPR32, GPR), m,
                             "@earlyclobber $rd">;
}

multiclass VPseudoBinaryV_WV {
  foreach m = MxList.m[0-5] in
    defm _WV : VPseudoBinary<m.vrclass, m.wvrclass, m.vrclass, m,
                             "@earlyclobber $rd">;
}

multiclass VPseudoBinaryV_WX {
  foreach m = MxList.m[0-5] in
    defm _WX : VPseudoBinary<m.vrclass, m.wvrclass, GPR, m,
                             "@earlyclobber $rd">;
}

multiclass VPseudoBinaryV_WI {
  foreach m = MxList.m[0-5] in
    defm _WI : VPseudoBinary<m.vrclass, m.wvrclass, uimm5, m,
                             "@earlyclobber $rd">;
}

// For vadc and vsbc, the instruction encoding is reserved if the destination
// vector register is v0.
// For vadc and vsbc, CarryIn == 1 and CarryOut == 0
multiclass VPseudoBinaryV_VM<bit CarryOut = 0, bit CarryIn = 1,
                             string Constraint = ""> {
  foreach m = MxList.m in
    def "_VV" # !if(CarryIn, "M", "") # "_" # m.MX :
      VPseudoBinaryCarryIn<!if(CarryOut, VR,
                           !if(!and(CarryIn, !not(CarryOut)),
                               GetVRegNoV0<m.vrclass>.R, m.vrclass)),
                           m.vrclass, m.vrclass, m, CarryIn, Constraint>;
}

multiclass VPseudoBinaryV_XM<bit CarryOut = 0, bit CarryIn = 1,
                             string Constraint = "", bit IsFloat = 0> {
  foreach m = MxList.m in
    def !if(IsFloat, "_VF", "_VX") # !if(CarryIn, "M", "") # "_" # m.MX :
      VPseudoBinaryCarryIn<!if(CarryOut, VR,
                           !if(!and(CarryIn, !not(CarryOut)),
                               GetVRegNoV0<m.vrclass>.R, m.vrclass)),
                           m.vrclass, !if(IsFloat, FPR32, GPR),
                           m, CarryIn, Constraint>;
}

multiclass VPseudoBinaryV_IM<bit CarryOut = 0, bit CarryIn = 1,
                             string Constraint = ""> {
  foreach m = MxList.m in
    def "_VI" # !if(CarryIn, "M", "") # "_" # m.MX :
      VPseudoBinaryCarryIn<!if(CarryOut, VR,
                           !if(!and(CarryIn, !not(CarryOut)),
                               GetVRegNoV0<m.vrclass>.R, m.vrclass)),
                           m.vrclass, simm5, m, CarryIn, Constraint>;
}

multiclass VPseudoUnaryV_V_X_I_NoDummyMask {
  foreach m = MxList.m in {
    let VLMul = m.value in {
      def "_V_" # m.MX : VPseudoUnaryNoDummyMask<m.vrclass, m.vrclass>;
      def "_X_" # m.MX : VPseudoUnaryNoDummyMask<m.vrclass, GPR>;
      def "_I_" # m.MX : VPseudoUnaryNoDummyMask<m.vrclass, simm5>;
    }
  }
}

multiclass VPseudoUnaryV_F_NoDummyMask {
  foreach m = MxList.m in {
    let VLMul = m.value in {
      def "_F_" # m.MX : VPseudoUnaryNoDummyMask<m.vrclass, FPR32>;
    }
  }
}

multiclass VPseudoUnaryV_V {
  foreach m = MxList.m in {
    let VLMul = m.value in {
      def "_V_" # m.MX : VPseudoUnaryNoMask<m.vrclass, m.vrclass>;
      def "_V_" # m.MX # "_MASK" : VPseudoUnaryMask<m.vrclass, m.vrclass>;
    }
  }
}

multiclass PseudoUnaryV_VF2 {
  defvar constraints = "@earlyclobber $rd";
  foreach m = MxList.m[1-6] in
  {
    let VLMul = m.value in {
      def "_" # m.MX : VPseudoUnaryNoMask<m.vrclass, m.f2vrclass, constraints>;
      def "_" # m.MX # "_MASK" : VPseudoUnaryMask<m.vrclass, m.f2vrclass,
                                                  constraints>;
    }
  }
}

multiclass PseudoUnaryV_VF4 {
  defvar constraints = "@earlyclobber $rd";
  foreach m = MxList.m[2-6] in
  {
    let VLMul = m.value in {
      def "_" # m.MX : VPseudoUnaryNoMask<m.vrclass, m.f4vrclass, constraints>;
      def "_" # m.MX # "_MASK" : VPseudoUnaryMask<m.vrclass, m.f4vrclass,
                                                  constraints>;
    }
  }
}

multiclass PseudoUnaryV_VF8 {
  defvar constraints = "@earlyclobber $rd";
  foreach m = MxList.m[3-6] in
  {
    let VLMul = m.value in {
      def "_" # m.MX : VPseudoUnaryNoMask<m.vrclass, m.f8vrclass, constraints>;
      def "_" # m.MX # "_MASK" : VPseudoUnaryMask<m.vrclass, m.f8vrclass,
                                                  constraints>;
    }
  }
}

// The destination EEW is 1.
// The source EEW is 8, 16, 32, or 64.
// When the destination EEW is different from source EEW, we need to use
// @earlyclobber to avoid the overlap between destination and source registers.
multiclass VPseudoBinaryM_VV {
  foreach m = MxList.m in
    defm _VV : VPseudoBinary<VR, m.vrclass, m.vrclass, m, "@earlyclobber $rd">;
}

multiclass VPseudoBinaryM_VX<bit IsFloat> {
  foreach m = MxList.m in
    defm !if(IsFloat, "_VF", "_VX") :
      VPseudoBinary<VR, m.vrclass, !if(IsFloat, FPR32, GPR), m,
                    "@earlyclobber $rd">;
}

multiclass VPseudoBinaryM_VI {
  foreach m = MxList.m in
    defm _VI : VPseudoBinary<VR, m.vrclass, simm5, m, "@earlyclobber $rd">;
}

multiclass VPseudoBinaryV_VV_VX_VI<Operand ImmType = simm5, string Constraint = ""> {
  defm "" : VPseudoBinaryV_VV<Constraint>;
  defm "" : VPseudoBinaryV_VX</*IsFloat=*/false, Constraint>;
  defm "" : VPseudoBinaryV_VI<ImmType, Constraint>;
}

multiclass VPseudoBinaryV_VV_VX<bit IsFloat = 0> {
  defm "" : VPseudoBinaryV_VV;
  defm "" : VPseudoBinaryV_VX<IsFloat>;
}

multiclass VPseudoBinaryV_VX_VI<Operand ImmType = simm5> {
  defm "" : VPseudoBinaryV_VX</*IsFloat=*/0>;
  defm "" : VPseudoBinaryV_VI<ImmType>;
}

multiclass VPseudoBinaryW_VV_VX<bit IsFloat = 0> {
  defm "" : VPseudoBinaryW_VV;
  defm "" : VPseudoBinaryW_VX<IsFloat>;
}

multiclass VPseudoBinaryW_WV_WX<bit IsFloat = 0> {
  defm "" : VPseudoBinaryW_WV;
  defm "" : VPseudoBinaryW_WX<IsFloat>;
}

multiclass VPseudoBinaryV_VM_XM_IM {
  defm "" : VPseudoBinaryV_VM;
  defm "" : VPseudoBinaryV_XM;
  defm "" : VPseudoBinaryV_IM;
}

multiclass VPseudoBinaryV_VM_XM {
  defm "" : VPseudoBinaryV_VM;
  defm "" : VPseudoBinaryV_XM;
}

multiclass VPseudoBinaryM_VM_XM_IM<string Constraint> {
  defm "" : VPseudoBinaryV_VM</*CarryOut=*/1, /*CarryIn=*/1, Constraint>;
  defm "" : VPseudoBinaryV_XM</*CarryOut=*/1, /*CarryIn=*/1, Constraint>;
  defm "" : VPseudoBinaryV_IM</*CarryOut=*/1, /*CarryIn=*/1, Constraint>;
}

multiclass VPseudoBinaryM_VM_XM<string Constraint> {
  defm "" : VPseudoBinaryV_VM</*CarryOut=*/1, /*CarryIn=*/1, Constraint>;
  defm "" : VPseudoBinaryV_XM</*CarryOut=*/1, /*CarryIn=*/1, Constraint>;
}

multiclass VPseudoBinaryM_V_X_I<string Constraint> {
  defm "" : VPseudoBinaryV_VM</*CarryOut=*/1, /*CarryIn=*/0, Constraint>;
  defm "" : VPseudoBinaryV_XM</*CarryOut=*/1, /*CarryIn=*/0, Constraint>;
  defm "" : VPseudoBinaryV_IM</*CarryOut=*/1, /*CarryIn=*/0, Constraint>;
}

multiclass VPseudoBinaryM_V_X<string Constraint> {
  defm "" : VPseudoBinaryV_VM</*CarryOut=*/1, /*CarryIn=*/0, Constraint>;
  defm "" : VPseudoBinaryV_XM</*CarryOut=*/1, /*CarryIn=*/0, Constraint>;
}

multiclass VPseudoBinaryV_WV_WX_WI {
  defm "" : VPseudoBinaryV_WV;
  defm "" : VPseudoBinaryV_WX;
  defm "" : VPseudoBinaryV_WI;
}

multiclass VPseudoTernary<VReg RetClass,
                          VReg Op1Class,
                          RegisterClass Op2Class,
                          LMULInfo MInfo,
                          string Constraint = ""> {
  let VLMul = MInfo.value in {
    def "_" # MInfo.MX : VPseudoTernaryNoMask<RetClass, Op1Class, Op2Class, Constraint>;
    def "_" # MInfo.MX # "_MASK" : VPseudoBinaryMask<RetClass, Op1Class, Op2Class, Constraint>;
  }
}

multiclass VPseudoTernaryV_VV<string Constraint = ""> {
  foreach m = MxList.m in
    defm _VV : VPseudoTernary<m.vrclass, m.vrclass, m.vrclass, m, Constraint>;
}

multiclass VPseudoTernaryV_VX<string Constraint = ""> {
  foreach m = MxList.m in
    defm _VX : VPseudoTernary<m.vrclass, m.vrclass, GPR, m, Constraint>;
}

multiclass VPseudoTernaryV_VX_AAXA<bit IsFloat, string Constraint = ""> {
  foreach m = MxList.m in
    defm !if(IsFloat, "_VF", "_VX") : VPseudoTernary<m.vrclass,
                             !if(IsFloat, FPR32, GPR), m.vrclass, m, Constraint>;
}

multiclass VPseudoTernaryW_VV {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxList.m in
    defm _VV : VPseudoTernary<m.wvrclass, m.vrclass, m.vrclass, m, constraint>;
}

multiclass VPseudoTernaryW_VX<bit IsFloat> {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxList.m in
    defm !if(IsFloat, "_VF", "_VX")  : VPseudoTernary<m.wvrclass,
                             !if(IsFloat, FPR32, GPR), m.vrclass, m, constraint>;
}

multiclass VPseudoTernaryV_VI<Operand ImmType = simm5, string Constraint = ""> {
  foreach m = MxList.m in
    defm _VI : VPseudoTernary<m.vrclass, m.vrclass, ImmType, m, Constraint>;
}

multiclass VPseudoTernaryV_VV_VX_AAXA<bit IsFloat = 0, string Constraint = ""> {
  defm "" : VPseudoTernaryV_VV<Constraint>;
  defm "" : VPseudoTernaryV_VX_AAXA<IsFloat, Constraint>;
}

multiclass VPseudoTernaryV_VX_VI<Operand ImmType = simm5, string Constraint = ""> {
  defm "" : VPseudoTernaryV_VX<Constraint>;
  defm "" : VPseudoTernaryV_VI<ImmType, Constraint>;
}

multiclass VPseudoTernaryW_VV_VX<bit IsFloat = false> {
  defm "" : VPseudoTernaryW_VV;
  defm "" : VPseudoTernaryW_VX<IsFloat>;
}

multiclass VPseudoBinaryM_VV_VX_VI {
  defm "" : VPseudoBinaryM_VV;
  defm "" : VPseudoBinaryM_VX</*IsFloat=*/0>;
  defm "" : VPseudoBinaryM_VI;
}

multiclass VPseudoBinaryM_VV_VX<bit IsFloat = 0> {
  defm "" : VPseudoBinaryM_VV;
  defm "" : VPseudoBinaryM_VX<IsFloat>;
}

multiclass VPseudoBinaryM_VX_VI {
  defm "" : VPseudoBinaryM_VX</*IsFloat=*/0>;
  defm "" : VPseudoBinaryM_VI;
}

multiclass VPseudoReductionV_VS {
  foreach m = MxList.m in {
    let WritesElement0 = 1 in
    defm _VS : VPseudoTernary<V_M1.vrclass, m.vrclass, V_M1.vrclass, m>;
  }
}

multiclass VPseudoConversion<VReg RetClass,
                             VReg Op1Class,
                             LMULInfo MInfo,
                             string Constraint = ""> {
  let VLMul = MInfo.value in {
    def "_" # MInfo.MX : VPseudoUnaryNoMask<RetClass, Op1Class, Constraint>;
    def "_" # MInfo.MX # "_MASK" : VPseudoUnaryMask<RetClass, Op1Class,
                                                    Constraint>;
  }
}

multiclass VPseudoConversionV_V {
  foreach m = MxList.m in
    defm _V : VPseudoConversion<m.vrclass, m.vrclass, m>;
}

multiclass VPseudoConversionW_V {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxList.m[0-5] in
    defm _V : VPseudoConversion<m.wvrclass, m.vrclass, m, constraint>;
}

multiclass VPseudoConversionV_W {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxList.m[0-5] in
    defm _W : VPseudoConversion<m.vrclass, m.wvrclass, m, constraint>;
}

//===----------------------------------------------------------------------===//
// Helpers to define the intrinsic patterns.
//===----------------------------------------------------------------------===//

class VPatUnaryNoMask<string intrinsic_name,
                      string inst,
                      string kind,
                      ValueType result_type,
                      ValueType op2_type,
                      int sew,
                      LMULInfo vlmul,
                      VReg op2_reg_class> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                   (op2_type op2_reg_class:$rs2),
                   (XLenVT GPR:$vl))),
                   (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX)
                   (op2_type op2_reg_class:$rs2),
                   (NoX0 GPR:$vl), sew)>;

class VPatUnaryMask<string intrinsic_name,
                    string inst,
                    string kind,
                    ValueType result_type,
                    ValueType op2_type,
                    ValueType mask_type,
                    int sew,
                    LMULInfo vlmul,
                    VReg result_reg_class,
                    VReg op2_reg_class> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic_name#"_mask")
                   (result_type result_reg_class:$merge),
                   (op2_type op2_reg_class:$rs2),
                   (mask_type V0),
                   (XLenVT GPR:$vl))),
                   (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX#"_MASK")
                   (result_type result_reg_class:$merge),
                   (op2_type op2_reg_class:$rs2),
                   (mask_type V0), (NoX0 GPR:$vl), sew)>;

class VPatMaskUnaryNoMask<string intrinsic_name,
                          string inst,
                          MTypeInfo mti> :
  Pat<(mti.Mask (!cast<Intrinsic>(intrinsic_name)
                (mti.Mask VR:$rs2),
                (XLenVT GPR:$vl))),
                (!cast<Instruction>(inst#"_M_"#mti.BX)
                (mti.Mask VR:$rs2),
                (NoX0 GPR:$vl), mti.SEW)>;

class VPatMaskUnaryMask<string intrinsic_name,
                        string inst,
                        MTypeInfo mti> :
  Pat<(mti.Mask (!cast<Intrinsic>(intrinsic_name#"_mask")
                (mti.Mask VR:$merge),
                (mti.Mask VR:$rs2),
                (mti.Mask V0),
                (XLenVT GPR:$vl))),
                (!cast<Instruction>(inst#"_M_"#mti.BX#"_MASK")
                (mti.Mask VR:$merge),
                (mti.Mask VR:$rs2),
                (mti.Mask V0), (NoX0 GPR:$vl), mti.SEW)>;

class VPatUnaryAnyMask<string intrinsic,
                       string inst,
                       string kind,
                       ValueType result_type,
                       ValueType op1_type,
                       ValueType mask_type,
                       int sew,
                       LMULInfo vlmul,
                       VReg result_reg_class,
                       VReg op1_reg_class> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic#"_mask")
                   (result_type result_reg_class:$merge),
                   (op1_type op1_reg_class:$rs1),
                   (mask_type VR:$rs2),
                   (XLenVT GPR:$vl))),
                   (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX#"_MASK")
                   (result_type result_reg_class:$merge),
                   (op1_type op1_reg_class:$rs1),
                   (mask_type VR:$rs2),
                   (NoX0 GPR:$vl), sew)>;

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

class VPatTernaryNoMask<string intrinsic,
                        string inst,
                        string kind,
                        ValueType result_type,
                        ValueType op1_type,
                        ValueType op2_type,
                        ValueType mask_type,
                        int sew,
                        LMULInfo vlmul,
                        VReg result_reg_class,
                        RegisterClass op1_reg_class,
                        DAGOperand op2_kind> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic)
                    (result_type result_reg_class:$rs3),
                    (op1_type op1_reg_class:$rs1),
                    (op2_type op2_kind:$rs2),
                    (XLenVT GPR:$vl))),
                   (!cast<Instruction>(inst#_#kind#"_"# vlmul.MX)
                    result_reg_class:$rs3,
                    ToFPR32<op1_type, op1_reg_class, "rs1">.ret,
                    op2_kind:$rs2,
                    (NoX0 GPR:$vl), sew)>;

class VPatTernaryMask<string intrinsic,
                      string inst,
                      string kind,
                      ValueType result_type,
                      ValueType op1_type,
                      ValueType op2_type,
                      ValueType mask_type,
                      int sew,
                      LMULInfo vlmul,
                      VReg result_reg_class,
                      RegisterClass op1_reg_class,
                      DAGOperand op2_kind> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic#"_mask")
                    (result_type result_reg_class:$rs3),
                    (op1_type op1_reg_class:$rs1),
                    (op2_type op2_kind:$rs2),
                    (mask_type V0),
                    (XLenVT GPR:$vl))),
                   (!cast<Instruction>(inst#_#kind#"_"# vlmul.MX # "_MASK")
                    result_reg_class:$rs3,
                    ToFPR32<op1_type, op1_reg_class, "rs1">.ret,
                    op2_kind:$rs2,
                    (mask_type V0),
                    (NoX0 GPR:$vl), sew)>;

multiclass VPatUSLoad<string intrinsic,
                      string inst,
                      LLVMType type,
                      LLVMType mask_type,
                      int sew,
                      LMULInfo vlmul,
                      VReg reg_class>
{
    defvar Intr = !cast<Intrinsic>(intrinsic);
    defvar Pseudo = !cast<Instruction>(inst#"_V_"#vlmul.MX);
    def : Pat<(type (Intr GPR:$rs1, GPR:$vl)),
                    (Pseudo $rs1, (NoX0 GPR:$vl), sew)>;
    defvar IntrMask = !cast<Intrinsic>(intrinsic # "_mask");
    defvar PseudoMask = !cast<Instruction>(inst#"_V_"#vlmul.MX#"_MASK");
    def : Pat<(type (IntrMask (type GetVRegNoV0<reg_class>.R:$merge),
                               GPR:$rs1, (mask_type V0), GPR:$vl)),
                    (PseudoMask $merge,
                                $rs1, (mask_type V0), (NoX0 GPR:$vl), sew)>;
}

multiclass VPatSLoad<string intrinsic,
                     string inst,
                     LLVMType type,
                     LLVMType mask_type,
                     int sew,
                     LMULInfo vlmul,
                     VReg reg_class>
{
    defvar Intr = !cast<Intrinsic>(intrinsic);
    defvar Pseudo = !cast<Instruction>(inst#"_V_"#vlmul.MX);
    def : Pat<(type (Intr GPR:$rs1, GPR:$rs2, GPR:$vl)),
                    (Pseudo $rs1, $rs2, (NoX0 GPR:$vl), sew)>;
    defvar IntrMask = !cast<Intrinsic>(intrinsic # "_mask");
    defvar PseudoMask = !cast<Instruction>(inst#"_V_"#vlmul.MX#"_MASK");
    def : Pat<(type (IntrMask (type GetVRegNoV0<reg_class>.R:$merge),
                               GPR:$rs1, GPR:$rs2, (mask_type V0), GPR:$vl)),
                    (PseudoMask $merge,
                                $rs1, $rs2, (mask_type V0), (NoX0 GPR:$vl), sew)>;
}

multiclass VPatILoad<string intrinsic,
                     string inst,
                     LLVMType type,
                     LLVMType idx_type,
                     LLVMType mask_type,
                     int sew,
                     LMULInfo vlmul,
                     LMULInfo idx_vlmul,
                     VReg reg_class,
                     VReg idx_reg_class>
{
    defvar Intr = !cast<Intrinsic>(intrinsic);
    defvar Pseudo = !cast<Instruction>(inst#"_V_"#idx_vlmul.MX#"_"#vlmul.MX);
    def : Pat<(type (Intr GPR:$rs1, (idx_type idx_reg_class:$rs2), GPR:$vl)),
                    (Pseudo $rs1, $rs2, (NoX0 GPR:$vl), sew)>;

    defvar IntrMask = !cast<Intrinsic>(intrinsic # "_mask");
    defvar PseudoMask = !cast<Instruction>(inst#"_V_"#idx_vlmul.MX#"_"#vlmul.MX#"_MASK");
    def : Pat<(type (IntrMask (type GetVRegNoV0<reg_class>.R:$merge),
                               GPR:$rs1, (idx_type idx_reg_class:$rs2),
                               (mask_type V0), GPR:$vl)),
                    (PseudoMask $merge,
                                $rs1, $rs2, (mask_type V0), (NoX0 GPR:$vl), sew)>;
}

multiclass VPatUSStore<string intrinsic,
                       string inst,
                       LLVMType type,
                       LLVMType mask_type,
                       int sew,
                       LMULInfo vlmul,
                       VReg reg_class>
{
    defvar Intr = !cast<Intrinsic>(intrinsic);
    defvar Pseudo = !cast<Instruction>(inst#"_V_"#vlmul.MX);
    def : Pat<(Intr (type reg_class:$rs3), GPR:$rs1, GPR:$vl),
                    (Pseudo $rs3, $rs1, (NoX0 GPR:$vl), sew)>;
    defvar IntrMask = !cast<Intrinsic>(intrinsic # "_mask");
    defvar PseudoMask = !cast<Instruction>(inst#"_V_"#vlmul.MX#"_MASK");
    def : Pat<(IntrMask (type reg_class:$rs3), GPR:$rs1, (mask_type V0), GPR:$vl),
              (PseudoMask $rs3, $rs1, (mask_type V0), (NoX0 GPR:$vl), sew)>;
}

multiclass VPatSStore<string intrinsic,
                      string inst,
                      LLVMType type,
                      LLVMType mask_type,
                      int sew,
                      LMULInfo vlmul,
                      VReg reg_class>
{
    defvar Intr = !cast<Intrinsic>(intrinsic);
    defvar Pseudo = !cast<Instruction>(inst#"_V_"#vlmul.MX);
    def : Pat<(Intr (type reg_class:$rs3), GPR:$rs1, GPR:$rs2, GPR:$vl),
                    (Pseudo $rs3, $rs1, $rs2, (NoX0 GPR:$vl), sew)>;
    defvar IntrMask = !cast<Intrinsic>(intrinsic # "_mask");
    defvar PseudoMask = !cast<Instruction>(inst#"_V_"#vlmul.MX#"_MASK");
    def : Pat<(IntrMask (type reg_class:$rs3), GPR:$rs1, GPR:$rs2, (mask_type V0), GPR:$vl),
              (PseudoMask $rs3, $rs1, $rs2, (mask_type V0), (NoX0 GPR:$vl), sew)>;
}

multiclass VPatIStore<string intrinsic,
                      string inst,
                      LLVMType type,
                      LLVMType idx_type,
                      LLVMType mask_type,
                      int sew,
                      LMULInfo vlmul,
                      LMULInfo idx_vlmul,
                      VReg reg_class,
                      VReg idx_reg_class>
{
    defvar Intr = !cast<Intrinsic>(intrinsic);
    defvar Pseudo = !cast<Instruction>(inst#"_V_"#idx_vlmul.MX#"_"#vlmul.MX);
    def : Pat<(Intr (type reg_class:$rs3), GPR:$rs1,
                    (idx_type idx_reg_class:$rs2), GPR:$vl),
              (Pseudo $rs3, $rs1, $rs2, (NoX0 GPR:$vl), sew)>;
    defvar IntrMask = !cast<Intrinsic>(intrinsic # "_mask");
    defvar PseudoMask = !cast<Instruction>(inst#"_V_"#idx_vlmul.MX#"_"#vlmul.MX#"_MASK");
    def : Pat<(IntrMask (type reg_class:$rs3), GPR:$rs1,
                        (idx_type idx_reg_class:$rs2), (mask_type V0), GPR:$vl),
              (PseudoMask $rs3, $rs1, $rs2, (mask_type V0), (NoX0 GPR:$vl), sew)>;
}

multiclass VPatUnaryS_M<string intrinsic_name,
                             string inst>
{
  foreach mti = AllMasks in {
    def : Pat<(XLenVT (!cast<Intrinsic>(intrinsic_name)
                      (mti.Mask VR:$rs1), GPR:$vl)),
                      (!cast<Instruction>(inst#"_M_"#mti.BX) $rs1,
                      (NoX0 GPR:$vl), mti.SEW)>;
    def : Pat<(XLenVT (!cast<Intrinsic>(intrinsic_name # "_mask")
                      (mti.Mask VR:$rs1), (mti.Mask V0), GPR:$vl)),
                      (!cast<Instruction>(inst#"_M_"#mti.BX#"_MASK") $rs1,
                      (mti.Mask V0), (NoX0 GPR:$vl), mti.SEW)>;
  }
}

multiclass VPatUnaryV_V_AnyMask<string intrinsic, string instruction,
                                list<VTypeInfo> vtilist> {
  foreach vti = vtilist in {
    def : VPatUnaryAnyMask<intrinsic, instruction, "VM",
                           vti.Vector, vti.Vector, vti.Mask,
                           vti.SEW, vti.LMul, vti.RegClass,
                           vti.RegClass>;
  }
}

multiclass VPatUnaryM_M<string intrinsic,
                         string inst>
{
  foreach mti = AllMasks in {
    def : VPatMaskUnaryNoMask<intrinsic, inst, mti>;
    def : VPatMaskUnaryMask<intrinsic, inst, mti>;
  }
}

multiclass VPatUnaryV_M<string intrinsic, string instruction>
{
  foreach vti = AllIntegerVectors in {
    def : VPatUnaryNoMask<intrinsic, instruction, "M", vti.Vector, vti.Mask,
                          vti.SEW, vti.LMul, VR>;
    def : VPatUnaryMask<intrinsic, instruction, "M", vti.Vector, vti.Mask,
                        vti.Mask, vti.SEW, vti.LMul, vti.RegClass, VR>;
  }
}

multiclass VPatUnaryV_VF<string intrinsic, string instruction, string suffix,
                         list<VTypeInfoToFraction> fractionList>
{
  foreach vtiTofti = fractionList in
  {
      defvar vti = vtiTofti.Vti;
      defvar fti = vtiTofti.Fti;
      def : VPatUnaryNoMask<intrinsic, instruction, suffix,
                            vti.Vector, fti.Vector,
                            vti.SEW, vti.LMul, fti.RegClass>;
      def : VPatUnaryMask<intrinsic, instruction, suffix,
                          vti.Vector, fti.Vector, vti.Mask,
                          vti.SEW, vti.LMul, vti.RegClass, fti.RegClass>;
   }
}

multiclass VPatUnaryV_V<string intrinsic, string instruction,
                        list<VTypeInfo> vtilist> {
  foreach vti = vtilist in {
    def : VPatUnaryNoMask<intrinsic, instruction, "V",
                          vti.Vector, vti.Vector,
                          vti.SEW, vti.LMul, vti.RegClass>;
    def : VPatUnaryMask<intrinsic, instruction, "V",
                        vti.Vector, vti.Vector, vti.Mask,
                        vti.SEW, vti.LMul, vti.RegClass, vti.RegClass>;
  }
}

multiclass VPatNullaryV<string intrinsic, string instruction>
{
  foreach vti = AllIntegerVectors in {
    def : Pat<(vti.Vector (!cast<Intrinsic>(intrinsic)
                          (XLenVT GPR:$vl))),
                          (!cast<Instruction>(instruction#"_V_" # vti.LMul.MX)
                          (NoX0 GPR:$vl), vti.SEW)>;
    def : Pat<(vti.Vector (!cast<Intrinsic>(intrinsic # "_mask")
                          (vti.Vector vti.RegClass:$merge),
                          (vti.Mask V0), (XLenVT GPR:$vl))),
                          (!cast<Instruction>(instruction#"_V_" # vti.LMul.MX # "_MASK")
                          vti.RegClass:$merge, (vti.Mask V0),
                          (NoX0 GPR:$vl), vti.SEW)>;
  }
}

multiclass VPatNullaryM<string intrinsic, string inst> {
  foreach mti = AllMasks in
    def : Pat<(mti.Mask (!cast<Intrinsic>(intrinsic)
                        (XLenVT GPR:$vl))),
                        (!cast<Instruction>(inst#"_M_"#mti.BX)
                        (NoX0 GPR:$vl), mti.SEW)>;
}

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

multiclass VPatBinaryCarryIn<string intrinsic,
                             string inst,
                             string kind,
                             ValueType result_type,
                             ValueType op1_type,
                             ValueType op2_type,
                             ValueType mask_type,
                             int sew,
                             LMULInfo vlmul,
                             VReg op1_reg_class,
                             DAGOperand op2_kind>
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic)
                         (op1_type op1_reg_class:$rs1),
                         (op2_type op2_kind:$rs2),
                         (mask_type V0),
                         (XLenVT GPR:$vl))),
                         (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX)
                         (op1_type op1_reg_class:$rs1),
                         ToFPR32<op2_type, op2_kind, "rs2">.ret,
                         (mask_type V0), (NoX0 GPR:$vl), sew)>;
}

multiclass VPatBinaryMaskOut<string intrinsic,
                             string inst,
                             string kind,
                             ValueType result_type,
                             ValueType op1_type,
                             ValueType op2_type,
                             int sew,
                             LMULInfo vlmul,
                             VReg op1_reg_class,
                             DAGOperand op2_kind>
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic)
                         (op1_type op1_reg_class:$rs1),
                         (op2_type op2_kind:$rs2),
                         (XLenVT GPR:$vl))),
                         (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX)
                         (op1_type op1_reg_class:$rs1),
                         ToFPR32<op2_type, op2_kind, "rs2">.ret,
                         (NoX0 GPR:$vl), sew)>;
}

multiclass VPatConversion<string intrinsic,
                          string inst,
                          string kind,
                          ValueType result_type,
                          ValueType op1_type,
                          ValueType mask_type,
                          int sew,
                          LMULInfo vlmul,
                          VReg result_reg_class,
                          VReg op1_reg_class>
{
  def : VPatUnaryNoMask<intrinsic, inst, kind, result_type, op1_type,
                        sew, vlmul, op1_reg_class>;
  def : VPatUnaryMask<intrinsic, inst, kind, result_type, op1_type,
                      mask_type, sew, vlmul, result_reg_class, op1_reg_class>;
}

multiclass VPatBinaryV_VV<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    defm : VPatBinary<intrinsic, instruction, "VV",
                      vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                      vti.SEW, vti.LMul, vti.RegClass,
                      vti.RegClass, vti.RegClass>;
}

multiclass VPatBinaryV_VV_INT<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist> {
  foreach vti = vtilist in {
    defvar ivti = GetIntVTypeInfo<vti>.Vti;
    defm : VPatBinary<intrinsic, instruction, "VV",
                      vti.Vector, vti.Vector, ivti.Vector, vti.Mask,
                      vti.SEW, vti.LMul, vti.RegClass,
                      vti.RegClass, vti.RegClass>;
  }
}

multiclass VPatBinaryV_VX<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    defm : VPatBinary<intrinsic, instruction,
                      !if(!eq(vti.Scalar, XLenVT), "VX", "VF"),
                      vti.Vector, vti.Vector, vti.Scalar, vti.Mask,
                      vti.SEW, vti.LMul, vti.RegClass,
                      vti.RegClass, vti.ScalarRegClass>;
}

multiclass VPatBinaryV_VX_INT<string intrinsic, string instruction,
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

multiclass VPatBinaryM_MM<string intrinsic, string instruction> {
  foreach mti = AllMasks in
    def : VPatBinaryNoMask<intrinsic, instruction, "MM",
                           mti.Mask, mti.Mask, mti.Mask,
                           mti.SEW, mti.LMul, VR, VR>;
}

multiclass VPatBinaryW_VV<string intrinsic, string instruction,
                          list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    defm : VPatBinary<intrinsic, instruction, "VV",
                      Wti.Vector, Vti.Vector, Vti.Vector, Vti.Mask,
                      Vti.SEW, Vti.LMul, Wti.RegClass,
                      Vti.RegClass, Vti.RegClass>;
  }
}

multiclass VPatBinaryW_VX<string intrinsic, string instruction,
                          list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    defm : VPatBinary<intrinsic, instruction,
                      !if(!eq(Vti.Scalar, XLenVT), "VX", "VF"),
                      Wti.Vector, Vti.Vector, Vti.Scalar, Vti.Mask,
                      Vti.SEW, Vti.LMul, Wti.RegClass,
                      Vti.RegClass, Vti.ScalarRegClass>;
  }
}

multiclass VPatBinaryW_WV<string intrinsic, string instruction,
                          list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    defm : VPatBinary<intrinsic, instruction, "WV",
                      Wti.Vector, Wti.Vector, Vti.Vector, Vti.Mask,
                      Vti.SEW, Vti.LMul, Wti.RegClass,
                      Wti.RegClass, Vti.RegClass>;
  }
}

multiclass VPatBinaryW_WX<string intrinsic, string instruction,
                          list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    defm : VPatBinary<intrinsic, instruction,
                      !if(!eq(Vti.Scalar, XLenVT), "WX", "WF"),
                      Wti.Vector, Wti.Vector, Vti.Scalar, Vti.Mask,
                      Vti.SEW, Vti.LMul, Wti.RegClass,
                      Wti.RegClass, Vti.ScalarRegClass>;
  }
}

multiclass VPatBinaryV_WV<string intrinsic, string instruction,
                          list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    defm : VPatBinary<intrinsic, instruction, "WV",
                      Vti.Vector, Wti.Vector, Vti.Vector, Vti.Mask,
                      Vti.SEW, Vti.LMul, Vti.RegClass,
                      Wti.RegClass, Vti.RegClass>;
  }
}

multiclass VPatBinaryV_WX<string intrinsic, string instruction,
                          list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    defm : VPatBinary<intrinsic, instruction,
                      !if(!eq(Vti.Scalar, XLenVT), "WX", "WF"),
                      Vti.Vector, Wti.Vector, Vti.Scalar, Vti.Mask,
                      Vti.SEW, Vti.LMul, Vti.RegClass,
                      Wti.RegClass, Vti.ScalarRegClass>;
  }
}

multiclass VPatBinaryV_WI<string intrinsic, string instruction,
                          list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    defm : VPatBinary<intrinsic, instruction, "WI",
                      Vti.Vector, Wti.Vector, XLenVT, Vti.Mask,
                      Vti.SEW, Vti.LMul, Vti.RegClass,
                      Wti.RegClass, uimm5>;
  }
}

multiclass VPatBinaryV_VM<string intrinsic, string instruction,
                          bit CarryOut = 0,
                          list<VTypeInfo> vtilist = AllIntegerVectors> {
  foreach vti = vtilist in
    defm : VPatBinaryCarryIn<intrinsic, instruction, "VVM",
                             !if(CarryOut, vti.Mask, vti.Vector),
                             vti.Vector, vti.Vector, vti.Mask,
                             vti.SEW, vti.LMul,
                             vti.RegClass, vti.RegClass>;
}

multiclass VPatBinaryV_XM<string intrinsic, string instruction,
                          bit CarryOut = 0,
                          list<VTypeInfo> vtilist = AllIntegerVectors> {
  foreach vti = vtilist in
    defm : VPatBinaryCarryIn<intrinsic, instruction,
                             !if(!eq(vti.Scalar, XLenVT), "VXM", "VFM"),
                             !if(CarryOut, vti.Mask, vti.Vector),
                             vti.Vector, vti.Scalar, vti.Mask,
                             vti.SEW, vti.LMul,
                             vti.RegClass, vti.ScalarRegClass>;
}

multiclass VPatBinaryV_IM<string intrinsic, string instruction,
                          bit CarryOut = 0> {
  foreach vti = AllIntegerVectors in
    defm : VPatBinaryCarryIn<intrinsic, instruction, "VIM",
                             !if(CarryOut, vti.Mask, vti.Vector),
                             vti.Vector, XLenVT, vti.Mask,
                             vti.SEW, vti.LMul,
                             vti.RegClass, simm5>;
}

multiclass VPatBinaryV_V<string intrinsic, string instruction> {
  foreach vti = AllIntegerVectors in
    defm : VPatBinaryMaskOut<intrinsic, instruction, "VV",
                             vti.Mask, vti.Vector, vti.Vector,
                             vti.SEW, vti.LMul,
                             vti.RegClass, vti.RegClass>;
}

multiclass VPatBinaryV_X<string intrinsic, string instruction> {
  foreach vti = AllIntegerVectors in
    defm : VPatBinaryMaskOut<intrinsic, instruction, "VX",
                             vti.Mask, vti.Vector, XLenVT,
                             vti.SEW, vti.LMul,
                             vti.RegClass, GPR>;
}

multiclass VPatBinaryV_I<string intrinsic, string instruction> {
  foreach vti = AllIntegerVectors in
    defm : VPatBinaryMaskOut<intrinsic, instruction, "VI",
                             vti.Mask, vti.Vector, XLenVT,
                             vti.SEW, vti.LMul,
                             vti.RegClass, simm5>;
}

multiclass VPatBinaryM_VV<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    defm : VPatBinary<intrinsic, instruction, "VV",
                      vti.Mask, vti.Vector, vti.Vector, vti.Mask,
                      vti.SEW, vti.LMul, VR,
                      vti.RegClass, vti.RegClass>;
}

multiclass VPatBinaryM_VX<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    defm : VPatBinary<intrinsic, instruction,
                      !if(!eq(vti.Scalar, XLenVT), "VX", "VF"),
                      vti.Mask, vti.Vector, vti.Scalar, vti.Mask,
                      vti.SEW, vti.LMul, VR,
                      vti.RegClass, vti.ScalarRegClass>;
}

multiclass VPatBinaryM_VI<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    defm : VPatBinary<intrinsic, instruction, "VI",
                      vti.Mask, vti.Vector, XLenVT, vti.Mask,
                      vti.SEW, vti.LMul, VR,
                      vti.RegClass, simm5>;
}

multiclass VPatBinaryV_VV_VX_VI<string intrinsic, string instruction,
                                list<VTypeInfo> vtilist, Operand ImmType = simm5>
{
  defm "" : VPatBinaryV_VV<intrinsic, instruction, vtilist>;
  defm "" : VPatBinaryV_VX<intrinsic, instruction, vtilist>;
  defm "" : VPatBinaryV_VI<intrinsic, instruction, vtilist, ImmType>;
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

multiclass VPatBinaryW_VV_VX<string intrinsic, string instruction,
                             list<VTypeInfoToWide> vtilist>
{
  defm "" : VPatBinaryW_VV<intrinsic, instruction, vtilist>;
  defm "" : VPatBinaryW_VX<intrinsic, instruction, vtilist>;
}

multiclass VPatBinaryW_WV_WX<string intrinsic, string instruction,
                             list<VTypeInfoToWide> vtilist>
{
  defm "" : VPatBinaryW_WV<intrinsic, instruction, vtilist>;
  defm "" : VPatBinaryW_WX<intrinsic, instruction, vtilist>;
}

multiclass VPatBinaryV_WV_WX_WI<string intrinsic, string instruction,
                                list<VTypeInfoToWide> vtilist>
{
  defm "" : VPatBinaryV_WV<intrinsic, instruction, vtilist>;
  defm "" : VPatBinaryV_WX<intrinsic, instruction, vtilist>;
  defm "" : VPatBinaryV_WI<intrinsic, instruction, vtilist>;
}

multiclass VPatBinaryV_VM_XM_IM<string intrinsic, string instruction>
{
  defm "" : VPatBinaryV_VM<intrinsic, instruction>;
  defm "" : VPatBinaryV_XM<intrinsic, instruction>;
  defm "" : VPatBinaryV_IM<intrinsic, instruction>;
}

multiclass VPatBinaryM_VM_XM_IM<string intrinsic, string instruction>
{
  defm "" : VPatBinaryV_VM<intrinsic, instruction, /*CarryOut=*/1>;
  defm "" : VPatBinaryV_XM<intrinsic, instruction, /*CarryOut=*/1>;
  defm "" : VPatBinaryV_IM<intrinsic, instruction, /*CarryOut=*/1>;
}

multiclass VPatBinaryM_V_X_I<string intrinsic, string instruction>
{
  defm "" : VPatBinaryV_V<intrinsic, instruction>;
  defm "" : VPatBinaryV_X<intrinsic, instruction>;
  defm "" : VPatBinaryV_I<intrinsic, instruction>;
}

multiclass VPatBinaryV_VM_XM<string intrinsic, string instruction>
{
  defm "" : VPatBinaryV_VM<intrinsic, instruction>;
  defm "" : VPatBinaryV_XM<intrinsic, instruction>;
}

multiclass VPatBinaryM_VM_XM<string intrinsic, string instruction>
{
  defm "" : VPatBinaryV_VM<intrinsic, instruction, /*CarryOut=*/1>;
  defm "" : VPatBinaryV_XM<intrinsic, instruction, /*CarryOut=*/1>;
}

multiclass VPatBinaryM_V_X<string intrinsic, string instruction>
{
  defm "" : VPatBinaryV_V<intrinsic, instruction>;
  defm "" : VPatBinaryV_X<intrinsic, instruction>;
}

multiclass VPatTernary<string intrinsic,
                       string inst,
                       string kind,
                       ValueType result_type,
                       ValueType op1_type,
                       ValueType op2_type,
                       ValueType mask_type,
                       int sew,
                       LMULInfo vlmul,
                       VReg result_reg_class,
                       RegisterClass op1_reg_class,
                       DAGOperand op2_kind> {
  def : VPatTernaryNoMask<intrinsic, inst, kind, result_type, op1_type, op2_type,
                    mask_type, sew, vlmul, result_reg_class, op1_reg_class,
                    op2_kind>;
  def : VPatTernaryMask<intrinsic, inst, kind, result_type, op1_type, op2_type,
                        mask_type, sew, vlmul, result_reg_class, op1_reg_class,
                        op2_kind>;
}

multiclass VPatTernaryV_VV<string intrinsic, string instruction,
                           list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    defm : VPatTernary<intrinsic, instruction, "VV",
                       vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                       vti.SEW, vti.LMul, vti.RegClass,
                       vti.RegClass, vti.RegClass>;
}

multiclass VPatTernaryV_VX<string intrinsic, string instruction,
                           list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    defm : VPatTernary<intrinsic, instruction, "VX",
                       vti.Vector, vti.Vector, XLenVT, vti.Mask,
                       vti.SEW, vti.LMul, vti.RegClass,
                       vti.RegClass, GPR>;
}

multiclass VPatTernaryV_VX_AAXA<string intrinsic, string instruction,
                           list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    defm : VPatTernary<intrinsic, instruction,
                       !if(!eq(vti.Scalar, XLenVT), "VX", "VF"),
                       vti.Vector, vti.Scalar, vti.Vector, vti.Mask,
                       vti.SEW, vti.LMul, vti.RegClass,
                       vti.ScalarRegClass, vti.RegClass>;
}

multiclass VPatTernaryV_VI<string intrinsic, string instruction,
                           list<VTypeInfo> vtilist, Operand Imm_type> {
  foreach vti = vtilist in
    defm : VPatTernary<intrinsic, instruction, "VI",
                      vti.Vector, vti.Vector, XLenVT, vti.Mask,
                      vti.SEW, vti.LMul, vti.RegClass,
                      vti.RegClass, Imm_type>;
}

multiclass VPatTernaryW_VV<string intrinsic, string instruction,
                           list<VTypeInfoToWide> vtilist> {
  foreach vtiToWti = vtilist in {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;
    defm : VPatTernary<intrinsic, instruction, "VV",
                      wti.Vector, vti.Vector, vti.Vector,
                      vti.Mask, vti.SEW, vti.LMul,
                      wti.RegClass, vti.RegClass, vti.RegClass>;
  }
}

multiclass VPatTernaryW_VX<string intrinsic, string instruction,
                           list<VTypeInfoToWide> vtilist> {
  foreach vtiToWti = vtilist in {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;
    defm : VPatTernary<intrinsic, instruction,
                       !if(!eq(vti.Scalar, XLenVT), "VX", "VF"),
                       wti.Vector, vti.Scalar, vti.Vector,
                       vti.Mask, vti.SEW, vti.LMul,
                       wti.RegClass, vti.ScalarRegClass, vti.RegClass>;
  }
}

multiclass VPatTernaryV_VV_VX_AAXA<string intrinsic, string instruction,
                              list<VTypeInfo> vtilist> {
  defm "" : VPatTernaryV_VV<intrinsic, instruction, vtilist>;
  defm "" : VPatTernaryV_VX_AAXA<intrinsic, instruction, vtilist>;
}

multiclass VPatTernaryV_VX_VI<string intrinsic, string instruction,
                              list<VTypeInfo> vtilist, Operand Imm_type = simm5> {
  defm "" : VPatTernaryV_VX<intrinsic, instruction, vtilist>;
  defm "" : VPatTernaryV_VI<intrinsic, instruction, vtilist, Imm_type>;
}

multiclass VPatBinaryM_VV_VX_VI<string intrinsic, string instruction,
                                list<VTypeInfo> vtilist>
{
  defm "" : VPatBinaryM_VV<intrinsic, instruction, vtilist>;
  defm "" : VPatBinaryM_VX<intrinsic, instruction, vtilist>;
  defm "" : VPatBinaryM_VI<intrinsic, instruction, vtilist>;
}

multiclass VPatTernaryW_VV_VX<string intrinsic, string instruction,
                              list<VTypeInfoToWide> vtilist> {
  defm "" : VPatTernaryW_VV<intrinsic, instruction, vtilist>;
  defm "" : VPatTernaryW_VX<intrinsic, instruction, vtilist>;
}

multiclass VPatBinaryM_VV_VX<string intrinsic, string instruction,
                             list<VTypeInfo> vtilist>
{
  defm "" : VPatBinaryM_VV<intrinsic, instruction, vtilist>;
  defm "" : VPatBinaryM_VX<intrinsic, instruction, vtilist>;
}

multiclass VPatBinaryM_VX_VI<string intrinsic, string instruction,
                             list<VTypeInfo> vtilist>
{
  defm "" : VPatBinaryM_VX<intrinsic, instruction, vtilist>;
  defm "" : VPatBinaryM_VI<intrinsic, instruction, vtilist>;
}

multiclass VPatBinaryV_VV_VX_VI_INT<string intrinsic, string instruction,
                                    list<VTypeInfo> vtilist, Operand ImmType = simm5>
{
  defm "" : VPatBinaryV_VV_INT<intrinsic, instruction, vtilist>;
  defm "" : VPatBinaryV_VX_INT<intrinsic, instruction, vtilist>;
  defm "" : VPatBinaryV_VI<intrinsic, instruction, vtilist, ImmType>;
}

multiclass VPatReductionV_VS<string intrinsic, string instruction, bit IsFloat = 0> {
  foreach vti = !if(IsFloat, NoGroupFloatVectors, NoGroupIntegerVectors) in
  {
    defvar vectorM1 = !cast<VTypeInfo>(!if(IsFloat, "VF", "VI") # vti.SEW # "M1");
    defm : VPatTernary<intrinsic, instruction, "VS",
                       vectorM1.Vector, vti.Vector,
                       vectorM1.Vector, vti.Mask,
                       vti.SEW, vti.LMul,
                       VR, vti.RegClass, VR>;
  }
  foreach gvti = !if(IsFloat, GroupFloatVectors, GroupIntegerVectors) in
  {
    defm : VPatTernary<intrinsic, instruction, "VS",
                       gvti.VectorM1, gvti.Vector,
                       gvti.VectorM1, gvti.Mask,
                       gvti.SEW, gvti.LMul,
                       VR, gvti.RegClass, VR>;
  }
}

multiclass VPatReductionW_VS<string intrinsic, string instruction, bit IsFloat = 0> {
  foreach vti = !if(IsFloat, AllFloatVectors, AllIntegerVectors) in
  {
    defvar wtiSEW = !mul(vti.SEW, 2);
    if !le(wtiSEW, 64) then {
      defvar wtiM1 = !cast<VTypeInfo>(!if(IsFloat, "VF", "VI") # wtiSEW # "M1");
      defm : VPatTernary<intrinsic, instruction, "VS",
                         wtiM1.Vector, vti.Vector,
                         wtiM1.Vector, vti.Mask,
                         vti.SEW, vti.LMul,
                         wtiM1.RegClass, vti.RegClass,
                         wtiM1.RegClass>;
    }
  }
}

multiclass VPatConversionVI_VF<string intrinsic,
                               string instruction>
{
  foreach fvti = AllFloatVectors in
  {
    defvar ivti = GetIntVTypeInfo<fvti>.Vti;

    defm : VPatConversion<intrinsic, instruction, "V",
                          ivti.Vector, fvti.Vector, ivti.Mask, fvti.SEW,
                          fvti.LMul, ivti.RegClass, fvti.RegClass>;
  }
}

multiclass VPatConversionVF_VI<string intrinsic,
                               string instruction>
{
  foreach fvti = AllFloatVectors in
  {
    defvar ivti = GetIntVTypeInfo<fvti>.Vti;

    defm : VPatConversion<intrinsic, instruction, "V",
                          fvti.Vector, ivti.Vector, fvti.Mask, ivti.SEW,
                          ivti.LMul, fvti.RegClass, ivti.RegClass>;
  }
}

multiclass VPatConversionWI_VF<string intrinsic, string instruction> {
  foreach fvtiToFWti = AllWidenableFloatVectors in
  {
    defvar fvti = fvtiToFWti.Vti;
    defvar iwti = GetIntVTypeInfo<fvtiToFWti.Wti>.Vti;

    defm : VPatConversion<intrinsic, instruction, "V",
                          iwti.Vector, fvti.Vector, iwti.Mask, fvti.SEW,
                          fvti.LMul, iwti.RegClass, fvti.RegClass>;
  }
}

multiclass VPatConversionWF_VI<string intrinsic, string instruction> {
  foreach vtiToWti = AllWidenableIntToFloatVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar fwti = vtiToWti.Wti;

    defm : VPatConversion<intrinsic, instruction, "V",
                          fwti.Vector, vti.Vector, fwti.Mask, vti.SEW,
                          vti.LMul, fwti.RegClass, vti.RegClass>;
  }
}

multiclass VPatConversionWF_VF <string intrinsic, string instruction> {
  foreach fvtiToFWti = AllWidenableFloatVectors in
  {
    defvar fvti = fvtiToFWti.Vti;
    defvar fwti = fvtiToFWti.Wti;

    defm : VPatConversion<intrinsic, instruction, "V",
                          fwti.Vector, fvti.Vector, fwti.Mask, fvti.SEW,
                          fvti.LMul, fwti.RegClass, fvti.RegClass>;
  }
}

multiclass VPatConversionVI_WF <string intrinsic, string instruction> {
  foreach vtiToWti = AllWidenableIntToFloatVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar fwti = vtiToWti.Wti;

    defm : VPatConversion<intrinsic, instruction, "W",
                          vti.Vector, fwti.Vector, vti.Mask, vti.SEW,
                          vti.LMul, vti.RegClass, fwti.RegClass>;
  }
}

multiclass VPatConversionVF_WI <string intrinsic, string instruction> {
  foreach fvtiToFWti = AllWidenableFloatVectors in
  {
    defvar fvti = fvtiToFWti.Vti;
    defvar iwti = GetIntVTypeInfo<fvtiToFWti.Wti>.Vti;

    defm : VPatConversion<intrinsic, instruction, "W",
                          fvti.Vector, iwti.Vector, fvti.Mask, fvti.SEW,
                          fvti.LMul, fvti.RegClass, iwti.RegClass>;
  }
}

multiclass VPatConversionVF_WF <string intrinsic, string instruction> {
  foreach fvtiToFWti = AllWidenableFloatVectors in
  {
    defvar fvti = fvtiToFWti.Vti;
    defvar fwti = fvtiToFWti.Wti;

    defm : VPatConversion<intrinsic, instruction, "W",
                          fvti.Vector, fwti.Vector, fvti.Mask, fvti.SEW,
                          fvti.LMul, fvti.RegClass, fwti.RegClass>;
  }
}

//===----------------------------------------------------------------------===//
// Pseudo instructions
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

//===----------------------------------------------------------------------===//
// 7.4 Vector Unit-Stride Instructions
//===----------------------------------------------------------------------===//

// Pseudos Unit-Stride Loads and Stores
foreach eew = EEWList in {
  defm PseudoVLE # eew : VPseudoUSLoad;
  defm PseudoVSE # eew : VPseudoUSStore;
}

//===----------------------------------------------------------------------===//
// 7.5 Vector Strided Instructions
//===----------------------------------------------------------------------===//

// Vector Strided Loads and Stores
foreach eew = EEWList in {
  defm PseudoVLSE # eew : VPseudoSLoad;
  defm PseudoVSSE # eew : VPseudoSStore;
}

//===----------------------------------------------------------------------===//
// 7.6 Vector Indexed Instructions
//===----------------------------------------------------------------------===//

// Vector Indexed Loads and Stores
foreach eew = EEWList in {
  defm PseudoVLXEI # eew : VPseudoILoad;
  defm PseudoVSXEI # eew : VPseudoIStore;
  defm PseudoVSUXEI # eew : VPseudoIStore;
}

//===----------------------------------------------------------------------===//
// 7.7. Unit-stride Fault-Only-First Loads
//===----------------------------------------------------------------------===//

// vleff may update VL register
let hasSideEffects = 1, Defs = [VL] in
foreach eew = EEWList in {
  defm PseudoVLE # eew # FF : VPseudoUSLoad;
}

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
// 12.2. Vector Widening Integer Add/Subtract
//===----------------------------------------------------------------------===//
defm PseudoVWADDU    : VPseudoBinaryW_VV_VX;
defm PseudoVWSUBU    : VPseudoBinaryW_VV_VX;
defm PseudoVWADD     : VPseudoBinaryW_VV_VX;
defm PseudoVWSUB     : VPseudoBinaryW_VV_VX;
defm PseudoVWADDU    : VPseudoBinaryW_WV_WX;
defm PseudoVWSUBU    : VPseudoBinaryW_WV_WX;
defm PseudoVWADD     : VPseudoBinaryW_WV_WX;
defm PseudoVWSUB     : VPseudoBinaryW_WV_WX;

//===----------------------------------------------------------------------===//
// 12.3. Vector Integer Extension
//===----------------------------------------------------------------------===//
defm PseudoVZEXT_VF2 : PseudoUnaryV_VF2;
defm PseudoVZEXT_VF4 : PseudoUnaryV_VF4;
defm PseudoVZEXT_VF8 : PseudoUnaryV_VF8;
defm PseudoVSEXT_VF2 : PseudoUnaryV_VF2;
defm PseudoVSEXT_VF4 : PseudoUnaryV_VF4;
defm PseudoVSEXT_VF8 : PseudoUnaryV_VF8;

//===----------------------------------------------------------------------===//
// 12.4. Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
//===----------------------------------------------------------------------===//
defm PseudoVADC        : VPseudoBinaryV_VM_XM_IM;
defm PseudoVMADC       : VPseudoBinaryM_VM_XM_IM<"@earlyclobber $rd">;
defm PseudoVMADC       : VPseudoBinaryM_V_X_I<"@earlyclobber $rd">;

defm PseudoVSBC        : VPseudoBinaryV_VM_XM;
defm PseudoVMSBC       : VPseudoBinaryM_VM_XM<"@earlyclobber $rd">;
defm PseudoVMSBC       : VPseudoBinaryM_V_X<"@earlyclobber $rd">;

//===----------------------------------------------------------------------===//
// 12.5. Vector Bitwise Logical Instructions
//===----------------------------------------------------------------------===//
defm PseudoVAND        : VPseudoBinaryV_VV_VX_VI;
defm PseudoVOR         : VPseudoBinaryV_VV_VX_VI;
defm PseudoVXOR        : VPseudoBinaryV_VV_VX_VI;

//===----------------------------------------------------------------------===//
// 12.6. Vector Single-Width Bit Shift Instructions
//===----------------------------------------------------------------------===//
defm PseudoVSLL        : VPseudoBinaryV_VV_VX_VI<uimm5>;
defm PseudoVSRL        : VPseudoBinaryV_VV_VX_VI<uimm5>;
defm PseudoVSRA        : VPseudoBinaryV_VV_VX_VI<uimm5>;

//===----------------------------------------------------------------------===//
// 12.7. Vector Narrowing Integer Right Shift Instructions
//===----------------------------------------------------------------------===//
defm PseudoVNSRL       : VPseudoBinaryV_WV_WX_WI;
defm PseudoVNSRA       : VPseudoBinaryV_WV_WX_WI;

//===----------------------------------------------------------------------===//
// 12.8. Vector Integer Comparison Instructions
//===----------------------------------------------------------------------===//
defm PseudoVMSEQ       : VPseudoBinaryM_VV_VX_VI;
defm PseudoVMSNE       : VPseudoBinaryM_VV_VX_VI;
defm PseudoVMSLTU      : VPseudoBinaryM_VV_VX;
defm PseudoVMSLT       : VPseudoBinaryM_VV_VX;
defm PseudoVMSLEU      : VPseudoBinaryM_VV_VX_VI;
defm PseudoVMSLE       : VPseudoBinaryM_VV_VX_VI;
defm PseudoVMSGTU      : VPseudoBinaryM_VX_VI;
defm PseudoVMSGT       : VPseudoBinaryM_VX_VI;

//===----------------------------------------------------------------------===//
// 12.9. Vector Integer Min/Max Instructions
//===----------------------------------------------------------------------===//
defm PseudoVMINU       : VPseudoBinaryV_VV_VX;
defm PseudoVMIN        : VPseudoBinaryV_VV_VX;
defm PseudoVMAXU       : VPseudoBinaryV_VV_VX;
defm PseudoVMAX        : VPseudoBinaryV_VV_VX;

//===----------------------------------------------------------------------===//
// 12.10. Vector Single-Width Integer Multiply Instructions
//===----------------------------------------------------------------------===//
defm PseudoVMUL        : VPseudoBinaryV_VV_VX;
defm PseudoVMULH       : VPseudoBinaryV_VV_VX;
defm PseudoVMULHU      : VPseudoBinaryV_VV_VX;
defm PseudoVMULHSU     : VPseudoBinaryV_VV_VX;

//===----------------------------------------------------------------------===//
// 12.11. Vector Integer Divide Instructions
//===----------------------------------------------------------------------===//
defm PseudoVDIVU       : VPseudoBinaryV_VV_VX;
defm PseudoVDIV        : VPseudoBinaryV_VV_VX;
defm PseudoVREMU       : VPseudoBinaryV_VV_VX;
defm PseudoVREM        : VPseudoBinaryV_VV_VX;

//===----------------------------------------------------------------------===//
// 12.12. Vector Widening Integer Multiply Instructions
//===----------------------------------------------------------------------===//
defm PseudoVWMUL       : VPseudoBinaryW_VV_VX;
defm PseudoVWMULU      : VPseudoBinaryW_VV_VX;
defm PseudoVWMULSU     : VPseudoBinaryW_VV_VX;

//===----------------------------------------------------------------------===//
// 12.13. Vector Single-Width Integer Multiply-Add Instructions
//===----------------------------------------------------------------------===//
defm PseudoVMACC       : VPseudoTernaryV_VV_VX_AAXA;
defm PseudoVNMSAC      : VPseudoTernaryV_VV_VX_AAXA;
defm PseudoVMADD       : VPseudoTernaryV_VV_VX_AAXA;
defm PseudoVNMSUB      : VPseudoTernaryV_VV_VX_AAXA;

//===----------------------------------------------------------------------===//
// 12.14. Vector Widening Integer Multiply-Add Instructions
//===----------------------------------------------------------------------===//
defm PseudoVWMACCU     : VPseudoTernaryW_VV_VX;
defm PseudoVWMACC      : VPseudoTernaryW_VV_VX;
defm PseudoVWMACCSU    : VPseudoTernaryW_VV_VX;
defm PseudoVWMACCUS    : VPseudoTernaryW_VX</*IsFloat*/false>;

//===----------------------------------------------------------------------===//
// 12.16. Vector Integer Merge Instructions
//===----------------------------------------------------------------------===//
defm PseudoVMERGE      : VPseudoBinaryV_VM_XM_IM;

//===----------------------------------------------------------------------===//
// 12.17. Vector Integer Move Instructions
//===----------------------------------------------------------------------===//
defm PseudoVMV_V       : VPseudoUnaryV_V_X_I_NoDummyMask;

//===----------------------------------------------------------------------===//
// 13.1. Vector Single-Width Saturating Add and Subtract
//===----------------------------------------------------------------------===//
let Defs = [VXSAT], hasSideEffects = 1 in {
  defm PseudoVSADDU      : VPseudoBinaryV_VV_VX_VI;
  defm PseudoVSADD       : VPseudoBinaryV_VV_VX_VI;
  defm PseudoVSSUBU      : VPseudoBinaryV_VV_VX;
  defm PseudoVSSUB       : VPseudoBinaryV_VV_VX;
}

//===----------------------------------------------------------------------===//
// 13.2. Vector Single-Width Averaging Add and Subtract
//===----------------------------------------------------------------------===//
let Uses = [VL, VTYPE, VXRM], hasSideEffects = 1 in {
  defm PseudoVAADDU      : VPseudoBinaryV_VV_VX;
  defm PseudoVAADD       : VPseudoBinaryV_VV_VX;
  defm PseudoVASUBU      : VPseudoBinaryV_VV_VX;
  defm PseudoVASUB       : VPseudoBinaryV_VV_VX;
}

//===----------------------------------------------------------------------===//
// 13.3. Vector Single-Width Fractional Multiply with Rounding and Saturation
//===----------------------------------------------------------------------===//
let Uses = [VL, VTYPE, VXRM], Defs = [VXSAT], hasSideEffects = 1 in {
  defm PseudoVSMUL      : VPseudoBinaryV_VV_VX;
}

//===----------------------------------------------------------------------===//
// 13.4. Vector Single-Width Scaling Shift Instructions
//===----------------------------------------------------------------------===//
let Uses = [VL, VTYPE, VXRM], hasSideEffects = 1 in {
  defm PseudoVSSRL        : VPseudoBinaryV_VV_VX_VI<uimm5>;
  defm PseudoVSSRA        : VPseudoBinaryV_VV_VX_VI<uimm5>;
}

//===----------------------------------------------------------------------===//
// 13.5. Vector Narrowing Fixed-Point Clip Instructions
//===----------------------------------------------------------------------===//
let Uses = [VL, VTYPE, VXRM], Defs = [VXSAT], hasSideEffects = 1 in {
  defm PseudoVNCLIP     : VPseudoBinaryV_WV_WX_WI;
  defm PseudoVNCLIPU    : VPseudoBinaryV_WV_WX_WI;
}

} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV, HasStdExtF] in {
//===----------------------------------------------------------------------===//
// 14.2. Vector Single-Width Floating-Point Add/Subtract Instructions
//===----------------------------------------------------------------------===//
defm PseudoVFADD       : VPseudoBinaryV_VV_VX</*IsFloat=*/1>;
defm PseudoVFSUB       : VPseudoBinaryV_VV_VX</*IsFloat=*/1>;
defm PseudoVFRSUB      : VPseudoBinaryV_VX</*IsFloat=*/1>;

//===----------------------------------------------------------------------===//
// 14.3. Vector Widening Floating-Point Add/Subtract Instructions
//===----------------------------------------------------------------------===//
defm PseudoVFWADD     : VPseudoBinaryW_VV_VX</*IsFloat=*/1>;
defm PseudoVFWSUB     : VPseudoBinaryW_VV_VX</*IsFloat=*/1>;
defm PseudoVFWADD     : VPseudoBinaryW_WV_WX</*IsFloat=*/1>;
defm PseudoVFWSUB     : VPseudoBinaryW_WV_WX</*IsFloat=*/1>;

//===----------------------------------------------------------------------===//
// 14.4. Vector Single-Width Floating-Point Multiply/Divide Instructions
//===----------------------------------------------------------------------===//
defm PseudoVFMUL       : VPseudoBinaryV_VV_VX</*IsFloat=*/1>;
defm PseudoVFDIV       : VPseudoBinaryV_VV_VX</*IsFloat=*/1>;
defm PseudoVFRDIV      : VPseudoBinaryV_VX</*IsFloat=*/1>;

//===----------------------------------------------------------------------===//
// 14.5. Vector Widening Floating-Point Multiply
//===----------------------------------------------------------------------===//
defm PseudoVFWMUL      : VPseudoBinaryW_VV_VX</*IsFloat=*/1>;

//===----------------------------------------------------------------------===//
// 14.6. Vector Single-Width Floating-Point Fused Multiply-Add Instructions
//===----------------------------------------------------------------------===//
defm PseudoVFMACC      : VPseudoTernaryV_VV_VX_AAXA</*IsFloat*/true>;
defm PseudoVFNMACC     : VPseudoTernaryV_VV_VX_AAXA</*IsFloat*/true>;
defm PseudoVFMSAC      : VPseudoTernaryV_VV_VX_AAXA</*IsFloat*/true>;
defm PseudoVFNMSAC     : VPseudoTernaryV_VV_VX_AAXA</*IsFloat*/true>;
defm PseudoVFMADD      : VPseudoTernaryV_VV_VX_AAXA</*IsFloat*/true>;
defm PseudoVFNMADD     : VPseudoTernaryV_VV_VX_AAXA</*IsFloat*/true>;
defm PseudoVFMSUB      : VPseudoTernaryV_VV_VX_AAXA</*IsFloat*/true>;
defm PseudoVFNMSUB     : VPseudoTernaryV_VV_VX_AAXA</*IsFloat*/true>;

//===----------------------------------------------------------------------===//
// 14.7. Vector Widening Floating-Point Fused Multiply-Add Instructions
//===----------------------------------------------------------------------===//
defm PseudoVFWMACC     : VPseudoTernaryW_VV_VX</*IsFloat*/true>;
defm PseudoVFWNMACC    : VPseudoTernaryW_VV_VX</*IsFloat*/true>;
defm PseudoVFWMSAC     : VPseudoTernaryW_VV_VX</*IsFloat*/true>;
defm PseudoVFWNMSAC    : VPseudoTernaryW_VV_VX</*IsFloat*/true>;

//===----------------------------------------------------------------------===//
// 14.8. Vector Floating-Point Square-Root Instruction
//===----------------------------------------------------------------------===//
defm PseudoVFSQRT      : VPseudoUnaryV_V;

//===----------------------------------------------------------------------===//
// 14.11. Vector Floating-Point Min/Max Instructions
//===----------------------------------------------------------------------===//
defm PseudoVFMIN       : VPseudoBinaryV_VV_VX</*IsFloat=*/1>;
defm PseudoVFMAX       : VPseudoBinaryV_VV_VX</*IsFloat=*/1>;

//===----------------------------------------------------------------------===//
// 14.12. Vector Floating-Point Sign-Injection Instructions
//===----------------------------------------------------------------------===//
defm PseudoVFSGNJ      : VPseudoBinaryV_VV_VX</*IsFloat=*/1>;
defm PseudoVFSGNJN     : VPseudoBinaryV_VV_VX</*IsFloat=*/1>;
defm PseudoVFSGNJX     : VPseudoBinaryV_VV_VX</*IsFloat=*/1>;

//===----------------------------------------------------------------------===//
// 14.13. Vector Floating-Point Compare Instructions
//===----------------------------------------------------------------------===//
defm PseudoVMFEQ       : VPseudoBinaryM_VV_VX</*IsFloat=*/1>;
defm PseudoVMFNE       : VPseudoBinaryM_VV_VX</*IsFloat=*/1>;
defm PseudoVMFLT       : VPseudoBinaryM_VV_VX</*IsFloat=*/1>;
defm PseudoVMFLE       : VPseudoBinaryM_VV_VX</*IsFloat=*/1>;
defm PseudoVMFGT       : VPseudoBinaryM_VX</*IsFloat=*/1>;
defm PseudoVMFGE       : VPseudoBinaryM_VX</*IsFloat=*/1>;

//===----------------------------------------------------------------------===//
// 14.14. Vector Floating-Point Classify Instruction
//===----------------------------------------------------------------------===//
defm PseudoVFCLASS     : VPseudoUnaryV_V;

//===----------------------------------------------------------------------===//
// 14.15. Vector Floating-Point Merge Instruction
//===----------------------------------------------------------------------===//
defm PseudoVFMERGE     : VPseudoBinaryV_XM</*CarryOut =*/0,/*CarryIn =*/true,
                                           /*Constraint =*/"", /*IsFloat=*/true>;

//===----------------------------------------------------------------------===//
// 14.16. Vector Floating-Point Move Instruction
//===----------------------------------------------------------------------===//
defm PseudoVFMV_V      : VPseudoUnaryV_F_NoDummyMask;

//===----------------------------------------------------------------------===//
// 14.17. Single-Width Floating-Point/Integer Type-Convert Instructions
//===----------------------------------------------------------------------===//
defm PseudoVFCVT_XU_F : VPseudoConversionV_V;
defm PseudoVFCVT_X_F : VPseudoConversionV_V;
defm PseudoVFCVT_RTZ_XU_F : VPseudoConversionV_V;
defm PseudoVFCVT_RTZ_X_F : VPseudoConversionV_V;
defm PseudoVFCVT_F_XU : VPseudoConversionV_V;
defm PseudoVFCVT_F_X : VPseudoConversionV_V;

//===----------------------------------------------------------------------===//
// 14.18. Widening Floating-Point/Integer Type-Convert Instructions
//===----------------------------------------------------------------------===//
defm PseudoVFWCVT_XU_F : VPseudoConversionW_V;
defm PseudoVFWCVT_X_F : VPseudoConversionW_V;
defm PseudoVFWCVT_RTZ_XU_F : VPseudoConversionW_V;
defm PseudoVFWCVT_RTZ_X_F : VPseudoConversionW_V;
defm PseudoVFWCVT_F_XU : VPseudoConversionW_V;
defm PseudoVFWCVT_F_X : VPseudoConversionW_V;
defm PseudoVFWCVT_F_F : VPseudoConversionW_V;

//===----------------------------------------------------------------------===//
// 14.19. Narrowing Floating-Point/Integer Type-Convert Instructions
//===----------------------------------------------------------------------===//
defm PseudoVFNCVT_XU_F : VPseudoConversionV_W;
defm PseudoVFNCVT_X_F : VPseudoConversionV_W;
defm PseudoVFNCVT_RTZ_XU_F : VPseudoConversionV_W;
defm PseudoVFNCVT_RTZ_X_F : VPseudoConversionV_W;
defm PseudoVFNCVT_F_XU : VPseudoConversionV_W;
defm PseudoVFNCVT_F_X : VPseudoConversionV_W;
defm PseudoVFNCVT_F_F : VPseudoConversionV_W;
defm PseudoVFNCVT_ROD_F_F : VPseudoConversionV_W;
} // Predicates = [HasStdExtV, HasStdExtF]

let Predicates = [HasStdExtV] in {
//===----------------------------------------------------------------------===//
// 15.1. Vector Single-Width Integer Reduction Instructions
//===----------------------------------------------------------------------===//
defm PseudoVREDSUM     : VPseudoReductionV_VS;
defm PseudoVREDAND     : VPseudoReductionV_VS;
defm PseudoVREDOR      : VPseudoReductionV_VS;
defm PseudoVREDXOR     : VPseudoReductionV_VS;
defm PseudoVREDMINU    : VPseudoReductionV_VS;
defm PseudoVREDMIN     : VPseudoReductionV_VS;
defm PseudoVREDMAXU    : VPseudoReductionV_VS;
defm PseudoVREDMAX     : VPseudoReductionV_VS;

//===----------------------------------------------------------------------===//
// 15.2. Vector Widening Integer Reduction Instructions
//===----------------------------------------------------------------------===//
defm PseudoVWREDSUMU   : VPseudoReductionV_VS;
defm PseudoVWREDSUM    : VPseudoReductionV_VS;
} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV, HasStdExtF] in {
//===----------------------------------------------------------------------===//
// 15.3. Vector Single-Width Floating-Point Reduction Instructions
//===----------------------------------------------------------------------===//
defm PseudoVFREDOSUM   : VPseudoReductionV_VS;
defm PseudoVFREDSUM    : VPseudoReductionV_VS;
defm PseudoVFREDMIN    : VPseudoReductionV_VS;
defm PseudoVFREDMAX    : VPseudoReductionV_VS;

//===----------------------------------------------------------------------===//
// 15.4. Vector Widening Floating-Point Reduction Instructions
//===----------------------------------------------------------------------===//
defm PseudoVFWREDSUM   : VPseudoReductionV_VS;
defm PseudoVFWREDOSUM  : VPseudoReductionV_VS;

} // Predicates = [HasStdExtV, HasStdExtF]

//===----------------------------------------------------------------------===//
// 16. Vector Mask Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 16.1 Vector Mask-Register Logical Instructions
//===----------------------------------------------------------------------===//

defm PseudoVMAND: VPseudoBinaryM_MM;
defm PseudoVMNAND: VPseudoBinaryM_MM;
defm PseudoVMANDNOT: VPseudoBinaryM_MM;
defm PseudoVMXOR: VPseudoBinaryM_MM;
defm PseudoVMOR: VPseudoBinaryM_MM;
defm PseudoVMNOR: VPseudoBinaryM_MM;
defm PseudoVMORNOT: VPseudoBinaryM_MM;
defm PseudoVMXNOR: VPseudoBinaryM_MM;

// Pseudo insturctions
defm PseudoVMCLR : VPseudoNullaryPseudoM<"VMXOR">;
defm PseudoVMSET : VPseudoNullaryPseudoM<"VMXNOR">;

//===----------------------------------------------------------------------===//
// 16.2. Vector mask population count vpopc
//===----------------------------------------------------------------------===//

defm PseudoVPOPC: VPseudoUnaryS_M;

//===----------------------------------------------------------------------===//
// 16.3. vfirst find-first-set mask bit
//===----------------------------------------------------------------------===//

defm PseudoVFIRST: VPseudoUnaryS_M;

//===----------------------------------------------------------------------===//
// 16.4. vmsbf.m set-before-first mask bit
//===----------------------------------------------------------------------===//
defm PseudoVMSBF: VPseudoUnaryM_M;

//===----------------------------------------------------------------------===//
// 16.5. vmsif.m set-including-first mask bit
//===----------------------------------------------------------------------===//
defm PseudoVMSIF: VPseudoUnaryM_M;

//===----------------------------------------------------------------------===//
// 16.6. vmsof.m set-only-first mask bit
//===----------------------------------------------------------------------===//
defm PseudoVMSOF: VPseudoUnaryM_M;

//===----------------------------------------------------------------------===//
// 16.8.  Vector Iota Instruction
//===----------------------------------------------------------------------===//
defm PseudoVIOTA_M: VPseudoUnaryV_M;

//===----------------------------------------------------------------------===//
// 16.9. Vector Element Index Instruction
//===----------------------------------------------------------------------===//
defm PseudoVID : VPseudoMaskNullaryV;

//===----------------------------------------------------------------------===//
// 17. Vector Permutation Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 17.1. Integer Scalar Move Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtV] in {
let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1,
    Uses = [VL, VTYPE] in {
  foreach m = MxList.m in {
    let VLMul = m.value in {
      let HasSEWOp = 1, BaseInstr = VMV_X_S in
      def PseudoVMV_X_S # "_" # m.MX: Pseudo<(outs GPR:$rd),
                                             (ins m.vrclass:$rs2, ixlenimm:$sew),
                                             []>, RISCVVPseudo;
      let HasVLOp = 1, HasSEWOp = 1, BaseInstr = VMV_S_X, WritesElement0 = 1,
          Constraints = "$rd = $rs1" in
      def PseudoVMV_S_X # "_" # m.MX: Pseudo<(outs m.vrclass:$rd),
                                             (ins m.vrclass:$rs1, GPR:$rs2,
                                                  GPR:$vl, ixlenimm:$sew),
                                             []>, RISCVVPseudo;
    }
  }
}
} // Predicates = [HasStdExtV]

//===----------------------------------------------------------------------===//
// 17.2. Floating-Point Scalar Move Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtV, HasStdExtF] in {
let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1,
    Uses = [VL, VTYPE] in {
  foreach m = MxList.m in {
    let VLMul = m.value in {
      let HasSEWOp = 1, BaseInstr = VFMV_F_S in
      def PseudoVFMV_F_S # "_" # m.MX : Pseudo<(outs FPR32:$rd),
                                               (ins m.vrclass:$rs2,
                                                    ixlenimm:$sew),
                                               []>, RISCVVPseudo;
      let HasVLOp = 1, HasSEWOp = 1, BaseInstr = VFMV_S_F, WritesElement0 = 1,
          Constraints = "$rd = $rs1" in
      def PseudoVFMV_S_F # "_" # m.MX : Pseudo<(outs m.vrclass:$rd),
                                               (ins m.vrclass:$rs1, FPR32:$rs2,
                                                    GPR:$vl, ixlenimm:$sew),
                                               []>, RISCVVPseudo;
    }
  }
}
} // Predicates = [HasStdExtV, HasStdExtF]

//===----------------------------------------------------------------------===//
// 17.3. Vector Slide Instructions
//===----------------------------------------------------------------------===//
let Predicates = [HasStdExtV] in {
  defm PseudoVSLIDEUP    : VPseudoTernaryV_VX_VI<uimm5, "@earlyclobber $rd">;
  defm PseudoVSLIDEDOWN  : VPseudoTernaryV_VX_VI<uimm5>;
  defm PseudoVSLIDE1UP   : VPseudoBinaryV_VX</*IsFloat*/false, "@earlyclobber $rd">;
  defm PseudoVSLIDE1DOWN : VPseudoBinaryV_VX</*IsFloat*/false>;
} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV, HasStdExtF] in {
  defm PseudoVFSLIDE1UP  : VPseudoBinaryV_VX</*IsFloat*/true, "@earlyclobber $rd">;
  defm PseudoVFSLIDE1DOWN : VPseudoBinaryV_VX</*IsFloat*/true>;
} // Predicates = [HasStdExtV, HasStdExtF]

//===----------------------------------------------------------------------===//
// 17.4. Vector Register Gather Instructions
//===----------------------------------------------------------------------===//
defm PseudoVRGATHER    : VPseudoBinaryV_VV_VX_VI<uimm5, "@earlyclobber $rd">;

//===----------------------------------------------------------------------===//
// 17.5. Vector Compress Instruction
//===----------------------------------------------------------------------===//
defm PseudoVCOMPRESS : VPseudoUnaryV_V_AnyMask;

//===----------------------------------------------------------------------===//
// Patterns.
//===----------------------------------------------------------------------===//
let Predicates = [HasStdExtV] in {

//===----------------------------------------------------------------------===//
// 7. Vector Loads and Stores
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 7.4 Vector Unit-Stride Instructions
//===----------------------------------------------------------------------===//

foreach vti = AllVectors in
{
  defm : VPatUSLoad<"int_riscv_vle",
                    "PseudoVLE" # vti.SEW,
                    vti.Vector, vti.Mask, vti.SEW, vti.LMul, vti.RegClass>;
  defm : VPatUSLoad<"int_riscv_vleff",
                    "PseudoVLE" # vti.SEW # "FF",
                    vti.Vector, vti.Mask, vti.SEW, vti.LMul, vti.RegClass>;
  defm : VPatUSStore<"int_riscv_vse",
                     "PseudoVSE" # vti.SEW,
                     vti.Vector, vti.Mask, vti.SEW, vti.LMul, vti.RegClass>;
}

//===----------------------------------------------------------------------===//
// 7.5 Vector Strided Instructions
//===----------------------------------------------------------------------===//

foreach vti = AllVectors in
{
  defm : VPatSLoad<"int_riscv_vlse",
                   "PseudoVLSE" # vti.SEW,
                   vti.Vector, vti.Mask, vti.SEW, vti.LMul, vti.RegClass>;
  defm : VPatSStore<"int_riscv_vsse",
                    "PseudoVSSE" # vti.SEW,
                    vti.Vector, vti.Mask, vti.SEW, vti.LMul, vti.RegClass>;
}

//===----------------------------------------------------------------------===//
// 7.6 Vector Indexed Instructions
//===----------------------------------------------------------------------===//

foreach vti = AllVectors in
foreach eew = EEWList in {
  defvar vlmul = vti.LMul;
  defvar octuple_lmul = !cond(!eq(vti.LMul.MX, "MF8") : 1,
                        !eq(vti.LMul.MX, "MF4") : 2,
                        !eq(vti.LMul.MX, "MF2") : 4,
                        !eq(vti.LMul.MX, "M1") : 8,
                        !eq(vti.LMul.MX, "M2") : 16,
                        !eq(vti.LMul.MX, "M4") : 32,
                        !eq(vti.LMul.MX, "M8") : 64);
  defvar log_sew = shift_amount<vti.SEW>.val;
  // The data vector register group has EEW=SEW, EMUL=LMUL, while the offset
  // vector register group has EEW encoding in the instruction and EMUL=(EEW/SEW)*LMUL.
  // calculate octuple elmul which is (eew * octuple_lmul) >> log_sew
  defvar octuple_elmul = !srl(!mul(eew, octuple_lmul), log_sew);
  // legal octuple elmul should be more than 0 and less than equal 64
  if !gt(octuple_elmul, 0) then {
    if !le(octuple_elmul, 64) then {
       defvar log_elmul = shift_amount<octuple_elmul>.val;
       // 0, 1, 2 -> V_MF8 ~ V_MF2
       // 3, 4, 5, 6 -> V_M1 ~ V_M8
       defvar elmul_str = !if(!eq(log_elmul, 0), "MF8",
                          !if(!eq(log_elmul, 1), "MF4",
                          !if(!eq(log_elmul, 2), "MF2",
                          "M" # !cast<string>(!shl(1, !add(log_elmul, -3))))));
       defvar elmul =!cast<LMULInfo>("V_" # elmul_str);
       defvar idx_vti = !cast<VTypeInfo>("VI" # eew # elmul_str);

       defm : VPatILoad<"int_riscv_vlxe",
                        "PseudoVLXEI"#eew,
                         vti.Vector, idx_vti.Vector, vti.Mask, vti.SEW,
                         vlmul, elmul, vti.RegClass, idx_vti.RegClass>;
       defm : VPatIStore<"int_riscv_vsxe",
                          "PseudoVSXEI"#eew,
                          vti.Vector, idx_vti.Vector, vti.Mask, vti.SEW,
                          vlmul, elmul, vti.RegClass, idx_vti.RegClass>;
       defm : VPatIStore<"int_riscv_vsuxe",
                          "PseudoVSUXEI"#eew,
                          vti.Vector, idx_vti.Vector, vti.Mask, vti.SEW,
                          vlmul, elmul, vti.RegClass, idx_vti.RegClass>;
    }
  }
}


//===----------------------------------------------------------------------===//
// 12. Vector Integer Arithmetic Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 12.1. Vector Single-Width Integer Add and Subtract
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryV_VV_VX_VI<"int_riscv_vadd", "PseudoVADD", AllIntegerVectors>;
defm "" : VPatBinaryV_VV_VX<"int_riscv_vsub", "PseudoVSUB", AllIntegerVectors>;
defm "" : VPatBinaryV_VX_VI<"int_riscv_vrsub", "PseudoVRSUB", AllIntegerVectors>;

//===----------------------------------------------------------------------===//
// 12.2. Vector Widening Integer Add/Subtract
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryW_VV_VX<"int_riscv_vwaddu", "PseudoVWADDU", AllWidenableIntVectors>;
defm "" : VPatBinaryW_VV_VX<"int_riscv_vwsubu", "PseudoVWSUBU", AllWidenableIntVectors>;
defm "" : VPatBinaryW_VV_VX<"int_riscv_vwadd", "PseudoVWADD", AllWidenableIntVectors>;
defm "" : VPatBinaryW_VV_VX<"int_riscv_vwsub", "PseudoVWSUB", AllWidenableIntVectors>;
defm "" : VPatBinaryW_WV_WX<"int_riscv_vwaddu_w", "PseudoVWADDU", AllWidenableIntVectors>;
defm "" : VPatBinaryW_WV_WX<"int_riscv_vwsubu_w", "PseudoVWSUBU", AllWidenableIntVectors>;
defm "" : VPatBinaryW_WV_WX<"int_riscv_vwadd_w", "PseudoVWADD", AllWidenableIntVectors>;
defm "" : VPatBinaryW_WV_WX<"int_riscv_vwsub_w", "PseudoVWSUB", AllWidenableIntVectors>;

//===----------------------------------------------------------------------===//
// 12.3. Vector Integer Extension
//===----------------------------------------------------------------------===//
defm "" : VPatUnaryV_VF<"int_riscv_vzext", "PseudoVZEXT", "VF2",
                        AllFractionableVF2IntVectors>;
defm "" : VPatUnaryV_VF<"int_riscv_vzext", "PseudoVZEXT", "VF4",
                        AllFractionableVF4IntVectors>;
defm "" : VPatUnaryV_VF<"int_riscv_vzext", "PseudoVZEXT", "VF8",
                        AllFractionableVF8IntVectors>;
defm "" : VPatUnaryV_VF<"int_riscv_vsext", "PseudoVSEXT", "VF2",
                        AllFractionableVF2IntVectors>;
defm "" : VPatUnaryV_VF<"int_riscv_vsext", "PseudoVSEXT", "VF4",
                        AllFractionableVF4IntVectors>;
defm "" : VPatUnaryV_VF<"int_riscv_vsext", "PseudoVSEXT", "VF8",
                        AllFractionableVF8IntVectors>;

//===----------------------------------------------------------------------===//
// 12.4. Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryV_VM_XM_IM<"int_riscv_vadc", "PseudoVADC">;
defm "" : VPatBinaryM_VM_XM_IM<"int_riscv_vmadc_carry_in", "PseudoVMADC">;
defm "" : VPatBinaryM_V_X_I<"int_riscv_vmadc", "PseudoVMADC">;

defm "" : VPatBinaryV_VM_XM<"int_riscv_vsbc", "PseudoVSBC">;
defm "" : VPatBinaryM_VM_XM<"int_riscv_vmsbc_borrow_in", "PseudoVMSBC">;
defm "" : VPatBinaryM_V_X<"int_riscv_vmsbc", "PseudoVMSBC">;

//===----------------------------------------------------------------------===//
// 12.5. Vector Bitwise Logical Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryV_VV_VX_VI<"int_riscv_vand", "PseudoVAND", AllIntegerVectors>;
defm "" : VPatBinaryV_VV_VX_VI<"int_riscv_vor", "PseudoVOR", AllIntegerVectors>;
defm "" : VPatBinaryV_VV_VX_VI<"int_riscv_vxor", "PseudoVXOR", AllIntegerVectors>;

//===----------------------------------------------------------------------===//
// 12.6. Vector Single-Width Bit Shift Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryV_VV_VX_VI<"int_riscv_vsll", "PseudoVSLL", AllIntegerVectors,
                               uimm5>;
defm "" : VPatBinaryV_VV_VX_VI<"int_riscv_vsrl", "PseudoVSRL", AllIntegerVectors,
                               uimm5>;
defm "" : VPatBinaryV_VV_VX_VI<"int_riscv_vsra", "PseudoVSRA", AllIntegerVectors,
                               uimm5>;

//===----------------------------------------------------------------------===//
// 12.7. Vector Narrowing Integer Right Shift Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryV_WV_WX_WI<"int_riscv_vnsrl", "PseudoVNSRL", AllWidenableIntVectors>;
defm "" : VPatBinaryV_WV_WX_WI<"int_riscv_vnsra", "PseudoVNSRA", AllWidenableIntVectors>;

//===----------------------------------------------------------------------===//
// 12.8. Vector Integer Comparison Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryM_VV_VX_VI<"int_riscv_vmseq", "PseudoVMSEQ", AllIntegerVectors>;
defm "" : VPatBinaryM_VV_VX_VI<"int_riscv_vmsne", "PseudoVMSNE", AllIntegerVectors>;
defm "" : VPatBinaryM_VV_VX<"int_riscv_vmsltu", "PseudoVMSLTU", AllIntegerVectors>;
defm "" : VPatBinaryM_VV_VX<"int_riscv_vmslt", "PseudoVMSLT", AllIntegerVectors>;
defm "" : VPatBinaryM_VV_VX_VI<"int_riscv_vmsleu", "PseudoVMSLEU", AllIntegerVectors>;
defm "" : VPatBinaryM_VV_VX_VI<"int_riscv_vmsle", "PseudoVMSLE", AllIntegerVectors>;

defm "" : VPatBinaryM_VX_VI<"int_riscv_vmsgtu", "PseudoVMSGTU", AllIntegerVectors>;
defm "" : VPatBinaryM_VX_VI<"int_riscv_vmsgt", "PseudoVMSGT", AllIntegerVectors>;

// Match vmslt(u).vx intrinsics to vmsle(u).vi if the scalar is -15 to 16. This
// avoids the user needing to know that there is no vmslt(u).vi instruction.
// This is limited to vmslt(u).vx as there is no vmsge().vx intrinsic or
// instruction.
foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Mask (int_riscv_vmslt (vti.Vector vti.RegClass:$rs1),
                                       (vti.Scalar simm5_plus1:$rs2), GPR:$vl)),
            (!cast<Instruction>("PseudoVMSLE_VI_"#vti.LMul.MX) vti.RegClass:$rs1,
                                                               (DecImm simm5_plus1:$rs2),
                                                               (NoX0 GPR:$vl),
                                                               vti.SEW)>;
  def : Pat<(vti.Mask (int_riscv_vmslt_mask (vti.Mask V0),
                                            (vti.Vector vti.RegClass:$rs1),
                                            (vti.Scalar simm5_plus1:$rs2),
                                            (vti.Mask VR:$merge),
                                            GPR:$vl)),
            (!cast<Instruction>("PseudoVMSLE_VI_"#vti.LMul.MX#"_MASK")
                                                      VR:$merge,
                                                      vti.RegClass:$rs1,
                                                      (DecImm simm5_plus1:$rs2),
                                                      (vti.Mask V0),
                                                      (NoX0 GPR:$vl),
                                                      vti.SEW)>;

  def : Pat<(vti.Mask (int_riscv_vmsltu (vti.Vector vti.RegClass:$rs1),
                                        (vti.Scalar simm5_plus1:$rs2), GPR:$vl)),
            (!cast<Instruction>("PseudoVMSLEU_VI_"#vti.LMul.MX) vti.RegClass:$rs1,
                                                                (DecImm simm5_plus1:$rs2),
                                                                (NoX0 GPR:$vl),
                                                                vti.SEW)>;
  def : Pat<(vti.Mask (int_riscv_vmsltu_mask (vti.Mask V0),
                                             (vti.Vector vti.RegClass:$rs1),
                                             (vti.Scalar simm5_plus1:$rs2),
                                             (vti.Mask VR:$merge),
                                             GPR:$vl)),
            (!cast<Instruction>("PseudoVMSLEU_VI_"#vti.LMul.MX#"_MASK")
                                                      VR:$merge,
                                                      vti.RegClass:$rs1,
                                                      (DecImm simm5_plus1:$rs2),
                                                      (vti.Mask V0),
                                                      (NoX0 GPR:$vl),
                                                      vti.SEW)>;

  // Special cases to avoid matching vmsltu.vi 0 (always false) to
  // vmsleu.vi -1 (always true). Instead match to vmsne.vv.
  def : Pat<(vti.Mask (int_riscv_vmsltu (vti.Vector vti.RegClass:$rs1),
                                        (vti.Scalar 0), GPR:$vl)),
            (!cast<Instruction>("PseudoVMSNE_VV_"#vti.LMul.MX) vti.RegClass:$rs1,
                                                               vti.RegClass:$rs1,
                                                               (NoX0 GPR:$vl),
                                                               vti.SEW)>;
  def : Pat<(vti.Mask (int_riscv_vmsltu_mask (vti.Mask V0),
                                            (vti.Vector vti.RegClass:$rs1),
                                            (vti.Scalar 0),
                                            (vti.Mask VR:$merge),
                                            GPR:$vl)),
            (!cast<Instruction>("PseudoVMSNE_VV_"#vti.LMul.MX#"_MASK")
                                                     VR:$merge,
                                                     vti.RegClass:$rs1,
                                                     vti.RegClass:$rs1,
                                                     (vti.Mask V0),
                                                     (NoX0 GPR:$vl),
                                                     vti.SEW)>;
}

//===----------------------------------------------------------------------===//
// 12.9. Vector Integer Min/Max Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryV_VV_VX<"int_riscv_vminu", "PseudoVMINU", AllIntegerVectors>;
defm "" : VPatBinaryV_VV_VX<"int_riscv_vmin", "PseudoVMIN", AllIntegerVectors>;
defm "" : VPatBinaryV_VV_VX<"int_riscv_vmaxu", "PseudoVMAXU", AllIntegerVectors>;
defm "" : VPatBinaryV_VV_VX<"int_riscv_vmax", "PseudoVMAX", AllIntegerVectors>;

//===----------------------------------------------------------------------===//
// 12.10. Vector Single-Width Integer Multiply Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryV_VV_VX<"int_riscv_vmul", "PseudoVMUL", AllIntegerVectors>;
defm "" : VPatBinaryV_VV_VX<"int_riscv_vmulh", "PseudoVMULH", AllIntegerVectors>;
defm "" : VPatBinaryV_VV_VX<"int_riscv_vmulhu", "PseudoVMULHU", AllIntegerVectors>;
defm "" : VPatBinaryV_VV_VX<"int_riscv_vmulhsu", "PseudoVMULHSU", AllIntegerVectors>;

//===----------------------------------------------------------------------===//
// 12.11. Vector Integer Divide Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryV_VV_VX<"int_riscv_vdivu", "PseudoVDIVU", AllIntegerVectors>;
defm "" : VPatBinaryV_VV_VX<"int_riscv_vdiv", "PseudoVDIV", AllIntegerVectors>;
defm "" : VPatBinaryV_VV_VX<"int_riscv_vremu", "PseudoVREMU", AllIntegerVectors>;
defm "" : VPatBinaryV_VV_VX<"int_riscv_vrem", "PseudoVREM", AllIntegerVectors>;

//===----------------------------------------------------------------------===//
// 12.12. Vector Widening Integer Multiply Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryW_VV_VX<"int_riscv_vwmul", "PseudoVWMUL", AllWidenableIntVectors>;
defm "" : VPatBinaryW_VV_VX<"int_riscv_vwmulu", "PseudoVWMULU", AllWidenableIntVectors>;
defm "" : VPatBinaryW_VV_VX<"int_riscv_vwmulsu", "PseudoVWMULSU", AllWidenableIntVectors>;

//===----------------------------------------------------------------------===//
// 12.13. Vector Single-Width Integer Multiply-Add Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatTernaryV_VV_VX_AAXA<"int_riscv_vmadd", "PseudoVMADD", AllIntegerVectors>;
defm "" : VPatTernaryV_VV_VX_AAXA<"int_riscv_vnmsub", "PseudoVNMSUB", AllIntegerVectors>;
defm "" : VPatTernaryV_VV_VX_AAXA<"int_riscv_vmacc", "PseudoVMACC", AllIntegerVectors>;
defm "" : VPatTernaryV_VV_VX_AAXA<"int_riscv_vnmsac", "PseudoVNMSAC", AllIntegerVectors>;

//===----------------------------------------------------------------------===//
// 12.14. Vector Widening Integer Multiply-Add Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatTernaryW_VV_VX<"int_riscv_vwmaccu", "PseudoVWMACCU", AllWidenableIntVectors>;
defm "" : VPatTernaryW_VV_VX<"int_riscv_vwmacc", "PseudoVWMACC", AllWidenableIntVectors>;
defm "" : VPatTernaryW_VV_VX<"int_riscv_vwmaccsu", "PseudoVWMACCSU", AllWidenableIntVectors>;
defm "" : VPatTernaryW_VX<"int_riscv_vwmaccus", "PseudoVWMACCUS", AllWidenableIntVectors>;

//===----------------------------------------------------------------------===//
// 12.16. Vector Integer Merge Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryV_VM_XM_IM<"int_riscv_vmerge", "PseudoVMERGE">;

//===----------------------------------------------------------------------===//
// 12.17. Vector Integer Move Instructions
//===----------------------------------------------------------------------===//
foreach vti = AllVectors in {
  def : Pat<(vti.Vector (int_riscv_vmv_v_v (vti.Vector vti.RegClass:$rs1),
                                           GPR:$vl)),
            (!cast<Instruction>("PseudoVMV_V_V_"#vti.LMul.MX)
             $rs1, (NoX0 GPR:$vl), vti.SEW)>;
}

foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (int_riscv_vmv_v_x GPR:$rs2, GPR:$vl)),
            (!cast<Instruction>("PseudoVMV_V_X_"#vti.LMul.MX)
             $rs2, (NoX0 GPR:$vl), vti.SEW)>;
  def : Pat<(vti.Vector (int_riscv_vmv_v_x simm5:$imm5, GPR:$vl)),
            (!cast<Instruction>("PseudoVMV_V_I_"#vti.LMul.MX)
             simm5:$imm5, (NoX0 GPR:$vl), vti.SEW)>;
}

//===----------------------------------------------------------------------===//
// 13.1. Vector Single-Width Saturating Add and Subtract
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryV_VV_VX_VI<"int_riscv_vsaddu", "PseudoVSADDU", AllIntegerVectors>;
defm "" : VPatBinaryV_VV_VX_VI<"int_riscv_vsadd", "PseudoVSADD", AllIntegerVectors>;
defm "" : VPatBinaryV_VV_VX<"int_riscv_vssubu", "PseudoVSSUBU", AllIntegerVectors>;
defm "" : VPatBinaryV_VV_VX<"int_riscv_vssub", "PseudoVSSUB", AllIntegerVectors>;

//===----------------------------------------------------------------------===//
// 13.2. Vector Single-Width Averaging Add and Subtract
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryV_VV_VX<"int_riscv_vaaddu", "PseudoVAADDU", AllIntegerVectors>;
defm "" : VPatBinaryV_VV_VX<"int_riscv_vaadd", "PseudoVAADD", AllIntegerVectors>;
defm "" : VPatBinaryV_VV_VX<"int_riscv_vasubu", "PseudoVASUBU", AllIntegerVectors>;
defm "" : VPatBinaryV_VV_VX<"int_riscv_vasub", "PseudoVASUB", AllIntegerVectors>;

//===----------------------------------------------------------------------===//
// 13.3. Vector Single-Width Fractional Multiply with Rounding and Saturation
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryV_VV_VX<"int_riscv_vsmul", "PseudoVSMUL", AllIntegerVectors>;

//===----------------------------------------------------------------------===//
// 13.4. Vector Single-Width Scaling Shift Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryV_VV_VX_VI<"int_riscv_vssrl", "PseudoVSSRL", AllIntegerVectors,
                               uimm5>;
defm "" : VPatBinaryV_VV_VX_VI<"int_riscv_vssra", "PseudoVSSRA", AllIntegerVectors,
                               uimm5>;

//===----------------------------------------------------------------------===//
// 13.5. Vector Narrowing Fixed-Point Clip Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryV_WV_WX_WI<"int_riscv_vnclipu", "PseudoVNCLIPU", AllWidenableIntVectors>;
defm "" : VPatBinaryV_WV_WX_WI<"int_riscv_vnclip", "PseudoVNCLIP", AllWidenableIntVectors>;

} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV, HasStdExtF] in {
//===----------------------------------------------------------------------===//
// 14.2. Vector Single-Width Floating-Point Add/Subtract Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryV_VV_VX<"int_riscv_vfadd", "PseudoVFADD", AllFloatVectors>;
defm "" : VPatBinaryV_VV_VX<"int_riscv_vfsub", "PseudoVFSUB", AllFloatVectors>;
defm "" : VPatBinaryV_VX<"int_riscv_vfrsub", "PseudoVFRSUB", AllFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.3. Vector Widening Floating-Point Add/Subtract Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryW_VV_VX<"int_riscv_vfwadd", "PseudoVFWADD", AllWidenableFloatVectors>;
defm "" : VPatBinaryW_VV_VX<"int_riscv_vfwsub", "PseudoVFWSUB", AllWidenableFloatVectors>;
defm "" : VPatBinaryW_WV_WX<"int_riscv_vfwadd_w", "PseudoVFWADD", AllWidenableFloatVectors>;
defm "" : VPatBinaryW_WV_WX<"int_riscv_vfwsub_w", "PseudoVFWSUB", AllWidenableFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.4. Vector Single-Width Floating-Point Multiply/Divide Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryV_VV_VX<"int_riscv_vfmul", "PseudoVFMUL", AllFloatVectors>;
defm "" : VPatBinaryV_VV_VX<"int_riscv_vfdiv", "PseudoVFDIV", AllFloatVectors>;
defm "" : VPatBinaryV_VX<"int_riscv_vfrdiv", "PseudoVFRDIV", AllFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.5. Vector Widening Floating-Point Multiply
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryW_VV_VX<"int_riscv_vfwmul", "PseudoVFWMUL", AllWidenableFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.6. Vector Single-Width Floating-Point Fused Multiply-Add Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatTernaryV_VV_VX_AAXA<"int_riscv_vfmacc", "PseudoVFMACC", AllFloatVectors>;
defm "" : VPatTernaryV_VV_VX_AAXA<"int_riscv_vfnmacc", "PseudoVFNMACC", AllFloatVectors>;
defm "" : VPatTernaryV_VV_VX_AAXA<"int_riscv_vfmsac", "PseudoVFMSAC", AllFloatVectors>;
defm "" : VPatTernaryV_VV_VX_AAXA<"int_riscv_vfnmsac", "PseudoVFNMSAC", AllFloatVectors>;
defm "" : VPatTernaryV_VV_VX_AAXA<"int_riscv_vfmadd", "PseudoVFMADD", AllFloatVectors>;
defm "" : VPatTernaryV_VV_VX_AAXA<"int_riscv_vfnmadd", "PseudoVFNMADD", AllFloatVectors>;
defm "" : VPatTernaryV_VV_VX_AAXA<"int_riscv_vfmsub", "PseudoVFMSUB", AllFloatVectors>;
defm "" : VPatTernaryV_VV_VX_AAXA<"int_riscv_vfnmsub", "PseudoVFNMSUB", AllFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.7. Vector Widening Floating-Point Fused Multiply-Add Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatTernaryW_VV_VX<"int_riscv_vfwmacc", "PseudoVFWMACC", AllWidenableFloatVectors>;
defm "" : VPatTernaryW_VV_VX<"int_riscv_vfwnmacc", "PseudoVFWNMACC", AllWidenableFloatVectors>;
defm "" : VPatTernaryW_VV_VX<"int_riscv_vfwmsac", "PseudoVFWMSAC", AllWidenableFloatVectors>;
defm "" : VPatTernaryW_VV_VX<"int_riscv_vfwnmsac", "PseudoVFWNMSAC", AllWidenableFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.8. Vector Floating-Point Square-Root Instruction
//===----------------------------------------------------------------------===//
defm "" : VPatUnaryV_V<"int_riscv_vfsqrt", "PseudoVFSQRT", AllFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.11. Vector Floating-Point Min/Max Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryV_VV_VX<"int_riscv_vfmin", "PseudoVFMIN", AllFloatVectors>;
defm "" : VPatBinaryV_VV_VX<"int_riscv_vfmax", "PseudoVFMAX", AllFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.12. Vector Floating-Point Sign-Injection Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryV_VV_VX<"int_riscv_vfsgnj", "PseudoVFSGNJ", AllFloatVectors>;
defm "" : VPatBinaryV_VV_VX<"int_riscv_vfsgnjn", "PseudoVFSGNJN", AllFloatVectors>;
defm "" : VPatBinaryV_VV_VX<"int_riscv_vfsgnjx", "PseudoVFSGNJX", AllFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.13. Vector Floating-Point Compare Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryM_VV_VX<"int_riscv_vmfeq", "PseudoVMFEQ", AllFloatVectors>;
defm "" : VPatBinaryM_VV_VX<"int_riscv_vmfle", "PseudoVMFLE", AllFloatVectors>;
defm "" : VPatBinaryM_VV_VX<"int_riscv_vmflt", "PseudoVMFLT", AllFloatVectors>;
defm "" : VPatBinaryM_VV_VX<"int_riscv_vmfne", "PseudoVMFNE", AllFloatVectors>;
defm "" : VPatBinaryM_VX<"int_riscv_vmfgt", "PseudoVMFGT", AllFloatVectors>;
defm "" : VPatBinaryM_VX<"int_riscv_vmfge", "PseudoVMFGE", AllFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.14. Vector Floating-Point Classify Instruction
//===----------------------------------------------------------------------===//
defm "" : VPatConversionVI_VF<"int_riscv_vfclass", "PseudoVFCLASS">;

//===----------------------------------------------------------------------===//
// 14.15. Vector Floating-Point Merge Instruction
//===----------------------------------------------------------------------===//
// We can use vmerge.vvm to support vector-vector vfmerge.
defm "" : VPatBinaryV_VM<"int_riscv_vfmerge", "PseudoVMERGE",
                         /*CarryOut = */0, /*vtilist=*/AllFloatVectors>;
defm "" : VPatBinaryV_XM<"int_riscv_vfmerge", "PseudoVFMERGE",
                         /*CarryOut = */0, /*vtilist=*/AllFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.16. Vector Floating-Point Move Instruction
//===----------------------------------------------------------------------===//
foreach fvti = AllFloatVectors in {
  // If we're splatting fpimm0, use vmv.v.x vd, x0.
  def : Pat<(fvti.Vector (int_riscv_vfmv_v_f
                         (fvti.Scalar (fpimm0)), GPR:$vl)),
            (!cast<Instruction>("PseudoVMV_V_I_"#fvti.LMul.MX)
             0, (NoX0 GPR:$vl), fvti.SEW)>;

  def : Pat<(fvti.Vector (int_riscv_vfmv_v_f
                         (fvti.Scalar fvti.ScalarRegClass:$rs2), GPR:$vl)),
            (!cast<Instruction>("PseudoVFMV_V_F_"#fvti.LMul.MX)
             ToFPR32<fvti.Scalar, fvti.ScalarRegClass, "rs2">.ret,
             (NoX0 GPR:$vl), fvti.SEW)>;
}

//===----------------------------------------------------------------------===//
// 14.17. Single-Width Floating-Point/Integer Type-Convert Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatConversionVI_VF<"int_riscv_vfcvt_xu_f_v", "PseudoVFCVT_XU_F">;
defm "" : VPatConversionVI_VF<"int_riscv_vfcvt_rtz_xu_f_v", "PseudoVFCVT_RTZ_XU_F">;
defm "" : VPatConversionVI_VF<"int_riscv_vfcvt_x_f_v", "PseudoVFCVT_X_F">;
defm "" : VPatConversionVI_VF<"int_riscv_vfcvt_rtz_x_f_v", "PseudoVFCVT_RTZ_X_F">;
defm "" : VPatConversionVF_VI<"int_riscv_vfcvt_f_x_v", "PseudoVFCVT_F_X">;
defm "" : VPatConversionVF_VI<"int_riscv_vfcvt_f_xu_v", "PseudoVFCVT_F_XU">;

//===----------------------------------------------------------------------===//
// 14.18. Widening Floating-Point/Integer Type-Convert Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatConversionWI_VF<"int_riscv_vfwcvt_xu_f_v", "PseudoVFWCVT_XU_F">;
defm "" : VPatConversionWI_VF<"int_riscv_vfwcvt_x_f_v", "PseudoVFWCVT_X_F">;
defm "" : VPatConversionWI_VF<"int_riscv_vfwcvt_rtz_xu_f_v", "PseudoVFWCVT_RTZ_XU_F">;
defm "" : VPatConversionWI_VF<"int_riscv_vfwcvt_rtz_x_f_v", "PseudoVFWCVT_RTZ_X_F">;
defm "" : VPatConversionWF_VI<"int_riscv_vfwcvt_f_xu_v", "PseudoVFWCVT_F_XU">;
defm "" : VPatConversionWF_VI<"int_riscv_vfwcvt_f_x_v", "PseudoVFWCVT_F_X">;
defm "" : VPatConversionWF_VF<"int_riscv_vfwcvt_f_f_v", "PseudoVFWCVT_F_F">;

//===----------------------------------------------------------------------===//
// 14.19. Narrowing Floating-Point/Integer Type-Convert Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatConversionVI_WF<"int_riscv_vfncvt_xu_f_w", "PseudoVFNCVT_XU_F">;
defm "" : VPatConversionVI_WF<"int_riscv_vfncvt_x_f_w", "PseudoVFNCVT_X_F">;
defm "" : VPatConversionVI_WF<"int_riscv_vfncvt_rtz_xu_f_w", "PseudoVFNCVT_RTZ_XU_F">;
defm "" : VPatConversionVI_WF<"int_riscv_vfncvt_rtz_x_f_w", "PseudoVFNCVT_RTZ_X_F">;
defm "" : VPatConversionVF_WI <"int_riscv_vfncvt_f_xu_w", "PseudoVFNCVT_F_XU">;
defm "" : VPatConversionVF_WI <"int_riscv_vfncvt_f_x_w", "PseudoVFNCVT_F_X">;
defm "" : VPatConversionVF_WF<"int_riscv_vfncvt_f_f_w", "PseudoVFNCVT_F_F">;
defm "" : VPatConversionVF_WF<"int_riscv_vfncvt_rod_f_f_w", "PseudoVFNCVT_ROD_F_F">;
} // Predicates = [HasStdExtV, HasStdExtF]

let Predicates = [HasStdExtV] in {
//===----------------------------------------------------------------------===//
// 15.1. Vector Single-Width Integer Reduction Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatReductionV_VS<"int_riscv_vredsum", "PseudoVREDSUM">;
defm "" : VPatReductionV_VS<"int_riscv_vredand", "PseudoVREDAND">;
defm "" : VPatReductionV_VS<"int_riscv_vredor", "PseudoVREDOR">;
defm "" : VPatReductionV_VS<"int_riscv_vredxor", "PseudoVREDXOR">;
defm "" : VPatReductionV_VS<"int_riscv_vredminu", "PseudoVREDMINU">;
defm "" : VPatReductionV_VS<"int_riscv_vredmin", "PseudoVREDMIN">;
defm "" : VPatReductionV_VS<"int_riscv_vredmaxu", "PseudoVREDMAXU">;
defm "" : VPatReductionV_VS<"int_riscv_vredmax", "PseudoVREDMAX">;

//===----------------------------------------------------------------------===//
// 15.2. Vector Widening Integer Reduction Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatReductionW_VS<"int_riscv_vwredsumu", "PseudoVWREDSUMU">;
defm "" : VPatReductionW_VS<"int_riscv_vwredsum", "PseudoVWREDSUM">;
} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV, HasStdExtF] in {
//===----------------------------------------------------------------------===//
// 15.3. Vector Single-Width Floating-Point Reduction Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatReductionV_VS<"int_riscv_vfredosum", "PseudoVFREDOSUM", /*IsFloat=*/1>;
defm "" : VPatReductionV_VS<"int_riscv_vfredsum", "PseudoVFREDSUM", /*IsFloat=*/1>;
defm "" : VPatReductionV_VS<"int_riscv_vfredmin", "PseudoVFREDMIN", /*IsFloat=*/1>;
defm "" : VPatReductionV_VS<"int_riscv_vfredmax", "PseudoVFREDMAX", /*IsFloat=*/1>;

//===----------------------------------------------------------------------===//
// 15.4. Vector Widening Floating-Point Reduction Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatReductionW_VS<"int_riscv_vfwredsum", "PseudoVFWREDSUM", /*IsFloat=*/1>;
defm "" : VPatReductionW_VS<"int_riscv_vfwredosum", "PseudoVFWREDOSUM", /*IsFloat=*/1>;

} // Predicates = [HasStdExtV, HasStdExtF]

//===----------------------------------------------------------------------===//
// 16. Vector Mask Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtV] in {
//===----------------------------------------------------------------------===//
// 16.1 Vector Mask-Register Logical Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryM_MM<"int_riscv_vmand", "PseudoVMAND">;
defm "" : VPatBinaryM_MM<"int_riscv_vmnand", "PseudoVMNAND">;
defm "" : VPatBinaryM_MM<"int_riscv_vmandnot", "PseudoVMANDNOT">;
defm "" : VPatBinaryM_MM<"int_riscv_vmxor", "PseudoVMXOR">;
defm "" : VPatBinaryM_MM<"int_riscv_vmor", "PseudoVMOR">;
defm "" : VPatBinaryM_MM<"int_riscv_vmnor", "PseudoVMNOR">;
defm "" : VPatBinaryM_MM<"int_riscv_vmornot", "PseudoVMORNOT">;
defm "" : VPatBinaryM_MM<"int_riscv_vmxnor", "PseudoVMXNOR">;

// pseudo instructions
defm "" : VPatNullaryM<"int_riscv_vmclr", "PseudoVMCLR">;
defm "" : VPatNullaryM<"int_riscv_vmset", "PseudoVMSET">;

//===----------------------------------------------------------------------===//
// 16.2. Vector mask population count vpopc
//===----------------------------------------------------------------------===//
defm "" : VPatUnaryS_M<"int_riscv_vpopc", "PseudoVPOPC">;

//===----------------------------------------------------------------------===//
// 16.3. vfirst find-first-set mask bit
//===----------------------------------------------------------------------===//
defm "" : VPatUnaryS_M<"int_riscv_vfirst", "PseudoVFIRST">;

//===----------------------------------------------------------------------===//
// 16.4. vmsbf.m set-before-first mask bit
//===----------------------------------------------------------------------===//
defm "" : VPatUnaryM_M<"int_riscv_vmsbf", "PseudoVMSBF">;

//===----------------------------------------------------------------------===//
// 16.5. vmsif.m set-including-first mask bit
//===----------------------------------------------------------------------===//
defm "" : VPatUnaryM_M<"int_riscv_vmsif", "PseudoVMSIF">;

//===----------------------------------------------------------------------===//
// 16.6. vmsof.m set-only-first mask bit
//===----------------------------------------------------------------------===//
defm "" : VPatUnaryM_M<"int_riscv_vmsof", "PseudoVMSOF">;

//===----------------------------------------------------------------------===//
// 16.8.  Vector Iota Instruction
//===----------------------------------------------------------------------===//
defm "" : VPatUnaryV_M<"int_riscv_viota", "PseudoVIOTA">;

//===----------------------------------------------------------------------===//
// 16.9. Vector Element Index Instruction
//===----------------------------------------------------------------------===//
defm "" : VPatNullaryV<"int_riscv_vid", "PseudoVID">;

} // Predicates = [HasStdExtV]

//===----------------------------------------------------------------------===//
// 17. Vector Permutation Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 17.1. Integer Scalar Move Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtV] in {
foreach vti = AllIntegerVectors in {
  def : Pat<(riscv_vmv_x_s (vti.Vector vti.RegClass:$rs2)),
            (!cast<Instruction>("PseudoVMV_X_S_" # vti.LMul.MX) $rs2, vti.SEW)>;
  def : Pat<(vti.Vector (int_riscv_vmv_s_x (vti.Vector vti.RegClass:$rs1),
                                           GPR:$rs2, GPR:$vl)),
            (!cast<Instruction>("PseudoVMV_S_X_" # vti.LMul.MX)
             (vti.Vector $rs1), $rs2, (NoX0 GPR:$vl), vti.SEW)>;
}
} // Predicates = [HasStdExtV]

//===----------------------------------------------------------------------===//
// 17.2. Floating-Point Scalar Move Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtV, HasStdExtF] in {
foreach fvti = AllFloatVectors in {
  defvar instr = !cast<Instruction>("PseudoVFMV_F_S_" # fvti.LMul.MX);
  def : Pat<(fvti.Scalar (int_riscv_vfmv_f_s (fvti.Vector fvti.RegClass:$rs2))),
             // Floating point instructions with a scalar result will always
             // generate the result in a register of class FPR32. When dealing
             // with the f64 variant of a pattern we need to promote the FPR32
             // subregister generated by the instruction to the FPR64 base
             // register expected by the type in the pattern
             !cond(!eq(!cast<string>(fvti.ScalarRegClass),
                       !cast<string>(FPR64)):
                      (SUBREG_TO_REG (i32 -1),
                                     (instr $rs2, fvti.SEW), sub_32),
                   !eq(!cast<string>(fvti.ScalarRegClass),
                       !cast<string>(FPR16)):
                      (EXTRACT_SUBREG (instr $rs2, fvti.SEW), sub_16),
                   !eq(1, 1):
                      (instr $rs2, fvti.SEW))>;

  def : Pat<(fvti.Vector (int_riscv_vfmv_s_f (fvti.Vector fvti.RegClass:$rs1),
                         (fvti.Scalar fvti.ScalarRegClass:$rs2), GPR:$vl)),
            (!cast<Instruction>("PseudoVFMV_S_F_" # fvti.LMul.MX)
             (fvti.Vector $rs1), ToFPR32<fvti.Scalar, fvti.ScalarRegClass, "rs2">.ret,
             (NoX0 GPR:$vl), fvti.SEW)>;
}
} // Predicates = [HasStdExtV, HasStdExtF]

//===----------------------------------------------------------------------===//
// 17.3. Vector Slide Instructions
//===----------------------------------------------------------------------===//
let Predicates = [HasStdExtV] in {
  defm "" : VPatTernaryV_VX_VI<"int_riscv_vslideup", "PseudoVSLIDEUP", AllIntegerVectors, uimm5>;
  defm "" : VPatTernaryV_VX_VI<"int_riscv_vslidedown", "PseudoVSLIDEDOWN", AllIntegerVectors, uimm5>;
  defm "" : VPatBinaryV_VX<"int_riscv_vslide1up", "PseudoVSLIDE1UP", AllIntegerVectors>;
  defm "" : VPatBinaryV_VX<"int_riscv_vslide1down", "PseudoVSLIDE1DOWN", AllIntegerVectors>;
} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV, HasStdExtF] in {
  defm "" : VPatTernaryV_VX_VI<"int_riscv_vslideup", "PseudoVSLIDEUP", AllFloatVectors, uimm5>;
  defm "" : VPatTernaryV_VX_VI<"int_riscv_vslidedown", "PseudoVSLIDEDOWN", AllFloatVectors, uimm5>;
  defm "" : VPatBinaryV_VX<"int_riscv_vfslide1up", "PseudoVFSLIDE1UP", AllFloatVectors>;
  defm "" : VPatBinaryV_VX<"int_riscv_vfslide1down", "PseudoVFSLIDE1DOWN", AllFloatVectors>;
} // Predicates = [HasStdExtV, HasStdExtF]

//===----------------------------------------------------------------------===//
// 17.4. Vector Register Gather Instructions
//===----------------------------------------------------------------------===//
let Predicates = [HasStdExtV] in {
  defm "" : VPatBinaryV_VV_VX_VI_INT<"int_riscv_vrgather", "PseudoVRGATHER",
                                     AllIntegerVectors, uimm5>;
} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV, HasStdExtF] in {
  defm "" : VPatBinaryV_VV_VX_VI_INT<"int_riscv_vrgather", "PseudoVRGATHER",
                                     AllFloatVectors, uimm5>;
} // Predicates = [HasStdExtV, HasStdExtF]

//===----------------------------------------------------------------------===//
// 17.5. Vector Compress Instruction
//===----------------------------------------------------------------------===//
let Predicates = [HasStdExtV] in {
  defm "" : VPatUnaryV_V_AnyMask<"int_riscv_vcompress", "PseudoVCOMPRESS", AllIntegerVectors>;
} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV, HasStdExtF] in {
  defm "" : VPatUnaryV_V_AnyMask<"int_riscv_vcompress", "PseudoVCOMPRESS", AllFloatVectors>;
} // Predicates = [HasStdExtV, HasStdExtF]

// Include the non-intrinsic ISel patterns
include "RISCVInstrInfoVSDPatterns.td"
