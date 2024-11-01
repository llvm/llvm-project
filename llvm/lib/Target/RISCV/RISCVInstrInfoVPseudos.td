//===-- RISCVInstrInfoVPseudos.td - RISC-V 'V' Pseudos -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file contains the required infrastructure to support code generation
/// for the standard 'V' (Vector) extension, version 1.0.
///
/// This file is included from RISCVInstrInfoV.td
///
//===----------------------------------------------------------------------===//

def riscv_vmv_x_s : SDNode<"RISCVISD::VMV_X_S",
                           SDTypeProfile<1, 1, [SDTCisInt<0>, SDTCisVec<1>,
                                                SDTCisInt<1>]>>;
def riscv_read_vlenb : SDNode<"RISCVISD::READ_VLENB",
                              SDTypeProfile<1, 0, [SDTCisVT<0, XLenVT>]>>;

// Operand that is allowed to be a register or a 5 bit immediate.
// This allows us to pick between VSETIVLI and VSETVLI opcodes using the same
// pseudo instructions.
def AVL : RegisterOperand<GPRNoX0> {
  let OperandNamespace = "RISCVOp";
  let OperandType = "OPERAND_AVL";
}

// X0 has special meaning for vsetvl/vsetvli.
//  rd | rs1 |   AVL value | Effect on vl
//--------------------------------------------------------------
// !X0 |  X0 |       VLMAX | Set vl to VLMAX
//  X0 |  X0 | Value in vl | Keep current vl, just change vtype.
def VLOp : ComplexPattern<XLenVT, 1, "selectVLOp">;

def DecImm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue() - 1, SDLoc(N),
                                   N->getValueType(0));
}]>;

defvar TAIL_UNDISTURBED_MASK_UNDISTURBED = 0;
defvar TAIL_AGNOSTIC = 1;
defvar TA_MA = 3;

//===----------------------------------------------------------------------===//
// Utilities.
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
                 !subst("_TIED", "",
                 !subst("_TU", "",
                 !subst("F16", "F",
                 !subst("F32", "F",
                 !subst("F64", "F",
                 !subst("Pseudo", "", PseudoInst)))))))))))))))))))));
}

// This class describes information associated to the LMUL.
class LMULInfo<int lmul, int oct, VReg regclass, VReg wregclass,
               VReg f2regclass, VReg f4regclass, VReg f8regclass, string mx> {
  bits<3> value = lmul; // This is encoded as the vlmul field of vtype.
  VReg vrclass = regclass;
  VReg wvrclass = wregclass;
  VReg f8vrclass = f8regclass;
  VReg f4vrclass = f4regclass;
  VReg f2vrclass = f2regclass;
  string MX = mx;
  int octuple = oct;
}

// Associate LMUL with tablegen records of register classes.
def V_M1  : LMULInfo<0b000,  8,   VR,        VRM2,   VR,   VR, VR, "M1">;
def V_M2  : LMULInfo<0b001, 16, VRM2,        VRM4,   VR,   VR, VR, "M2">;
def V_M4  : LMULInfo<0b010, 32, VRM4,        VRM8, VRM2,   VR, VR, "M4">;
def V_M8  : LMULInfo<0b011, 64, VRM8,/*NoVReg*/VR, VRM4, VRM2, VR, "M8">;

def V_MF8 : LMULInfo<0b101, 1, VR, VR,/*NoVReg*/VR,/*NoVReg*/VR,/*NoVReg*/VR, "MF8">;
def V_MF4 : LMULInfo<0b110, 2, VR, VR,          VR,/*NoVReg*/VR,/*NoVReg*/VR, "MF4">;
def V_MF2 : LMULInfo<0b111, 4, VR, VR,          VR,          VR,/*NoVReg*/VR, "MF2">;

// Used to iterate over all possible LMULs.
defvar MxList = [V_MF8, V_MF4, V_MF2, V_M1, V_M2, V_M4, V_M8];
// For floating point which don't need MF8.
defvar MxListF = [V_MF4, V_MF2, V_M1, V_M2, V_M4, V_M8];

// Used for widening and narrowing instructions as it doesn't contain M8.
defvar MxListW = [V_MF8, V_MF4, V_MF2, V_M1, V_M2, V_M4];
// For floating point which don't need MF8.
defvar MxListFW = [V_MF4, V_MF2, V_M1, V_M2, V_M4];

// Use for zext/sext.vf2
defvar MxListVF2 = [V_MF4, V_MF2, V_M1, V_M2, V_M4, V_M8];

// Use for zext/sext.vf4
defvar MxListVF4 = [V_MF2, V_M1, V_M2, V_M4, V_M8];

// Use for zext/sext.vf8
defvar MxListVF8 = [V_M1, V_M2, V_M4, V_M8];

class MxSet<int eew> {
  list<LMULInfo> m = !cond(!eq(eew, 8) : [V_MF8, V_MF4, V_MF2, V_M1, V_M2, V_M4, V_M8],
                           !eq(eew, 16) : [V_MF4, V_MF2, V_M1, V_M2, V_M4, V_M8],
                           !eq(eew, 32) : [V_MF2, V_M1, V_M2, V_M4, V_M8],
                           !eq(eew, 64) : [V_M1, V_M2, V_M4, V_M8]);
}

class FPR_Info<RegisterClass regclass, string fx, list<LMULInfo> mxlist,
               list<LMULInfo> mxlistfw> {
  RegisterClass fprclass = regclass;
  string FX = fx;
  list<LMULInfo> MxList = mxlist;
  list<LMULInfo> MxListFW = mxlistfw;
}

def SCALAR_F16 : FPR_Info<FPR16, "F16", MxSet<16>.m, [V_MF4, V_MF2, V_M1, V_M2, V_M4]>;
def SCALAR_F32 : FPR_Info<FPR32, "F32", MxSet<32>.m, [V_MF2, V_M1, V_M2, V_M4]>;
def SCALAR_F64 : FPR_Info<FPR64, "F64", MxSet<64>.m, []>;

defvar FPList = [SCALAR_F16, SCALAR_F32, SCALAR_F64];

// Used for widening instructions. It excludes F64.
defvar FPListW = [SCALAR_F16, SCALAR_F32];

class NFSet<LMULInfo m> {
  list<int> L = !cond(!eq(m.value, V_M8.value): [],
                      !eq(m.value, V_M4.value): [2],
                      !eq(m.value, V_M2.value): [2, 3, 4],
                      true: [2, 3, 4, 5, 6, 7, 8]);
}

class log2<int num> {
  int val = !if(!eq(num, 1), 0, !add(1, log2<!srl(num, 1)>.val));
}

class octuple_to_str<int octuple> {
  string ret = !if(!eq(octuple, 1), "MF8",
                   !if(!eq(octuple, 2), "MF4",
                   !if(!eq(octuple, 4), "MF2",
                   !if(!eq(octuple, 8), "M1",
                   !if(!eq(octuple, 16), "M2",
                   !if(!eq(octuple, 32), "M4",
                   !if(!eq(octuple, 64), "M8",
                   "NoDef")))))));
}

def VLOpFrag : PatFrag<(ops), (XLenVT (VLOp (XLenVT AVL:$vl)))>;

// Output pattern for X0 used to represent VLMAX in the pseudo instructions.
// We can't use X0 register becuase the AVL operands use GPRNoX0.
// This must be kept in sync with RISCV::VLMaxSentinel.
def VLMax : OutPatFrag<(ops), (XLenVT -1)>;

// List of EEW.
defvar EEWList = [8, 16, 32, 64];

class SegRegClass<LMULInfo m, int nf> {
  VReg RC = !cast<VReg>("VRN" # nf # !cond(!eq(m.value, V_MF8.value): V_M1.MX,
                                           !eq(m.value, V_MF4.value): V_M1.MX,
                                           !eq(m.value, V_MF2.value): V_M1.MX,
                                           true: m.MX));
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
  int Log2SEW = log2<Sew>.val;
  VReg RegClass = Reg;
  LMULInfo LMul = M;
  ValueType Scalar = Scal;
  RegisterClass ScalarRegClass = ScalarReg;
  // The pattern fragment which produces the AVL operand, representing the
  // "natural" vector length for this type. For scalable vectors this is VLMax.
  OutPatFrag AVL = VLMax;

  string ScalarSuffix = !cond(!eq(Scal, XLenVT) : "X",
                              !eq(Scal, f16) : "F16",
                              !eq(Scal, f32) : "F32",
                              !eq(Scal, f64) : "F64");
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
      defset list<VTypeInfo> FractionalGroupIntegerVectors = {
        def VI8MF8: VTypeInfo<vint8mf8_t,  vbool64_t,  8, VR, V_MF8>;
        def VI8MF4: VTypeInfo<vint8mf4_t,  vbool32_t,  8, VR, V_MF4>;
        def VI8MF2: VTypeInfo<vint8mf2_t,  vbool16_t,  8, VR, V_MF2>;
        def VI16MF4: VTypeInfo<vint16mf4_t, vbool64_t, 16, VR, V_MF4>;
        def VI16MF2: VTypeInfo<vint16mf2_t, vbool32_t, 16, VR, V_MF2>;
        def VI32MF2: VTypeInfo<vint32mf2_t, vbool64_t, 32, VR, V_MF2>;
      }
      def VI8M1: VTypeInfo<vint8m1_t,   vbool8_t,   8, VR, V_M1>;
      def VI16M1: VTypeInfo<vint16m1_t,  vbool16_t, 16, VR, V_M1>;
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
      defset list<VTypeInfo> FractionalGroupFloatVectors = {
        def VF16MF4: VTypeInfo<vfloat16mf4_t, vbool64_t, 16, VR, V_MF4, f16, FPR16>;
        def VF16MF2: VTypeInfo<vfloat16mf2_t, vbool32_t, 16, VR, V_MF2, f16, FPR16>;
        def VF32MF2: VTypeInfo<vfloat32mf2_t,vbool64_t, 32, VR, V_MF2, f32, FPR32>;
      }
      def VF16M1:  VTypeInfo<vfloat16m1_t,  vbool16_t, 16, VR, V_M1,  f16, FPR16>;
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
  // we assume SEW=1 and set corresponding LMUL. vsetvli insertion will
  // look for SEW=1 to optimize based on surrounding instructions.
  int SEW = 1;
  int Log2SEW = 0;
  LMULInfo LMul = M;
  string BX = Bx; // Appendix of mask operations.
  // The pattern fragment which produces the AVL operand, representing the
  // "natural" vector length for this mask type. For scalable masks this is
  // VLMax.
  OutPatFrag AVL = VLMax;
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
  Instruction BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

// The actual table.
def RISCVVPseudosTable : GenericTable {
  let FilterClass = "RISCVVPseudo";
  let CppTypeName = "PseudoInfo";
  let Fields = [ "Pseudo", "BaseInstr" ];
  let PrimaryKey = [ "Pseudo" ];
  let PrimaryKeyName = "getPseudoInfo";
  let PrimaryKeyEarlyOut = true;
}

def RISCVVInversePseudosTable : GenericTable {
  let FilterClass = "RISCVVPseudo";
  let CppTypeName = "PseudoInfo";
  let Fields = [ "Pseudo", "BaseInstr", "VLMul" ];
  let PrimaryKey = [ "BaseInstr", "VLMul" ];
  let PrimaryKeyName = "getBaseInfo";
  let PrimaryKeyEarlyOut = true;
}

def RISCVVIntrinsicsTable : GenericTable {
  let FilterClass = "RISCVVIntrinsic";
  let CppTypeName = "RISCVVIntrinsicInfo";
  let Fields = ["IntrinsicID", "ScalarOperand", "VLOperand"];
  let PrimaryKey = ["IntrinsicID"];
  let PrimaryKeyName = "getRISCVVIntrinsicInfo";
}

class RISCVMaskedPseudo<bits<4> MaskIdx, bit HasTU = true> {
  Pseudo MaskedPseudo = !cast<Pseudo>(NAME);
  Pseudo UnmaskedPseudo = !cast<Pseudo>(!subst("_MASK", "", NAME));
  Pseudo UnmaskedTUPseudo = !if(HasTU, !cast<Pseudo>(!subst("_MASK", "", NAME # "_TU")), MaskedPseudo);
  bits<4> MaskOpIdx = MaskIdx;
}

def RISCVMaskedPseudosTable : GenericTable {
  let FilterClass = "RISCVMaskedPseudo";
  let CppTypeName = "RISCVMaskedPseudoInfo";
  let Fields = ["MaskedPseudo", "UnmaskedPseudo", "UnmaskedTUPseudo", "MaskOpIdx"];
  let PrimaryKey = ["MaskedPseudo"];
  let PrimaryKeyName = "getMaskedPseudoInfo";
}

class RISCVVLE<bit M, bit TU, bit Str, bit F, bits<3> S, bits<3> L> {
  bits<1> Masked = M;
  bits<1> IsTU = TU;
  bits<1> Strided = Str;
  bits<1> FF = F;
  bits<3> Log2SEW = S;
  bits<3> LMUL = L;
  Pseudo Pseudo = !cast<Pseudo>(NAME);
}

def lookupMaskedIntrinsicByUnmaskedTA : SearchIndex {
  let Table = RISCVMaskedPseudosTable;
  let Key = ["UnmaskedPseudo"];
}

def RISCVVLETable : GenericTable {
  let FilterClass = "RISCVVLE";
  let CppTypeName = "VLEPseudo";
  let Fields = ["Masked", "IsTU", "Strided", "FF", "Log2SEW", "LMUL", "Pseudo"];
  let PrimaryKey = ["Masked", "IsTU", "Strided", "FF", "Log2SEW", "LMUL"];
  let PrimaryKeyName = "getVLEPseudo";
}

class RISCVVSE<bit M, bit Str, bits<3> S, bits<3> L> {
  bits<1> Masked = M;
  bits<1> Strided = Str;
  bits<3> Log2SEW = S;
  bits<3> LMUL = L;
  Pseudo Pseudo = !cast<Pseudo>(NAME);
}

def RISCVVSETable : GenericTable {
  let FilterClass = "RISCVVSE";
  let CppTypeName = "VSEPseudo";
  let Fields = ["Masked", "Strided", "Log2SEW", "LMUL", "Pseudo"];
  let PrimaryKey = ["Masked", "Strided", "Log2SEW", "LMUL"];
  let PrimaryKeyName = "getVSEPseudo";
}

class RISCVVLX_VSX<bit M, bit TU, bit O, bits<3> S, bits<3> L, bits<3> IL> {
  bits<1> Masked = M;
  bits<1> IsTU = TU;
  bits<1> Ordered = O;
  bits<3> Log2SEW = S;
  bits<3> LMUL = L;
  bits<3> IndexLMUL = IL;
  Pseudo Pseudo = !cast<Pseudo>(NAME);
}

class RISCVVLX<bit M, bit TU, bit O, bits<3> S, bits<3> L, bits<3> IL> :
  RISCVVLX_VSX<M, TU, O, S, L, IL>;
class RISCVVSX<bit M, bit O, bits<3> S, bits<3> L, bits<3> IL> :
  RISCVVLX_VSX<M, /*TU*/0, O, S, L, IL>;

class RISCVVLX_VSXTable : GenericTable {
  let CppTypeName = "VLX_VSXPseudo";
  let Fields = ["Masked", "IsTU", "Ordered", "Log2SEW", "LMUL", "IndexLMUL", "Pseudo"];
  let PrimaryKey = ["Masked", "IsTU", "Ordered", "Log2SEW", "LMUL", "IndexLMUL"];
}

def RISCVVLXTable : RISCVVLX_VSXTable {
  let FilterClass = "RISCVVLX";
  let PrimaryKeyName = "getVLXPseudo";
}

def RISCVVSXTable : RISCVVLX_VSXTable {
  let FilterClass = "RISCVVSX";
  let PrimaryKeyName = "getVSXPseudo";
}

class RISCVVLSEG<bits<4> N, bit M, bit TU, bit Str, bit F, bits<3> S, bits<3> L> {
  bits<4> NF = N;
  bits<1> Masked = M;
  bits<1> IsTU = TU;
  bits<1> Strided = Str;
  bits<1> FF = F;
  bits<3> Log2SEW = S;
  bits<3> LMUL = L;
  Pseudo Pseudo = !cast<Pseudo>(NAME);
}

def RISCVVLSEGTable : GenericTable {
  let FilterClass = "RISCVVLSEG";
  let CppTypeName = "VLSEGPseudo";
  let Fields = ["NF", "Masked", "IsTU", "Strided", "FF", "Log2SEW", "LMUL", "Pseudo"];
  let PrimaryKey = ["NF", "Masked", "IsTU", "Strided", "FF", "Log2SEW", "LMUL"];
  let PrimaryKeyName = "getVLSEGPseudo";
}

class RISCVVLXSEG<bits<4> N, bit M, bit TU, bit O, bits<3> S, bits<3> L, bits<3> IL> {
  bits<4> NF = N;
  bits<1> Masked = M;
  bits<1> IsTU = TU;
  bits<1> Ordered = O;
  bits<3> Log2SEW = S;
  bits<3> LMUL = L;
  bits<3> IndexLMUL = IL;
  Pseudo Pseudo = !cast<Pseudo>(NAME);
}

def RISCVVLXSEGTable : GenericTable {
  let FilterClass = "RISCVVLXSEG";
  let CppTypeName = "VLXSEGPseudo";
  let Fields = ["NF", "Masked", "IsTU", "Ordered", "Log2SEW", "LMUL", "IndexLMUL", "Pseudo"];
  let PrimaryKey = ["NF", "Masked", "IsTU", "Ordered", "Log2SEW", "LMUL", "IndexLMUL"];
  let PrimaryKeyName = "getVLXSEGPseudo";
}

class RISCVVSSEG<bits<4> N, bit M, bit Str, bits<3> S, bits<3> L> {
  bits<4> NF = N;
  bits<1> Masked = M;
  bits<1> Strided = Str;
  bits<3> Log2SEW = S;
  bits<3> LMUL = L;
  Pseudo Pseudo = !cast<Pseudo>(NAME);
}

def RISCVVSSEGTable : GenericTable {
  let FilterClass = "RISCVVSSEG";
  let CppTypeName = "VSSEGPseudo";
  let Fields = ["NF", "Masked", "Strided", "Log2SEW", "LMUL", "Pseudo"];
  let PrimaryKey = ["NF", "Masked", "Strided", "Log2SEW", "LMUL"];
  let PrimaryKeyName = "getVSSEGPseudo";
}

class RISCVVSXSEG<bits<4> N, bit M, bit O, bits<3> S, bits<3> L, bits<3> IL> {
  bits<4> NF = N;
  bits<1> Masked = M;
  bits<1> Ordered = O;
  bits<3> Log2SEW = S;
  bits<3> LMUL = L;
  bits<3> IndexLMUL = IL;
  Pseudo Pseudo = !cast<Pseudo>(NAME);
}

def RISCVVSXSEGTable : GenericTable {
  let FilterClass = "RISCVVSXSEG";
  let CppTypeName = "VSXSEGPseudo";
  let Fields = ["NF", "Masked", "Ordered", "Log2SEW", "LMUL", "IndexLMUL", "Pseudo"];
  let PrimaryKey = ["NF", "Masked", "Ordered", "Log2SEW", "LMUL", "IndexLMUL"];
  let PrimaryKeyName = "getVSXSEGPseudo";
}

//===----------------------------------------------------------------------===//
// Helpers to define the different pseudo instructions.
//===----------------------------------------------------------------------===//

// The destination vector register group for a masked vector instruction cannot
// overlap the source mask register (v0), unless the destination vector register
// is being written with a mask value (e.g., comparisons) or the scalar result
// of a reduction.
class GetVRegNoV0<VReg VRegClass> {
  VReg R = !cond(!eq(VRegClass, VR) : VRNoV0,
                 !eq(VRegClass, VRM2) : VRM2NoV0,
                 !eq(VRegClass, VRM4) : VRM4NoV0,
                 !eq(VRegClass, VRM8) : VRM8NoV0,
                 !eq(VRegClass, VRN2M1) : VRN2M1NoV0,
                 !eq(VRegClass, VRN2M2) : VRN2M2NoV0,
                 !eq(VRegClass, VRN2M4) : VRN2M4NoV0,
                 !eq(VRegClass, VRN3M1) : VRN3M1NoV0,
                 !eq(VRegClass, VRN3M2) : VRN3M2NoV0,
                 !eq(VRegClass, VRN4M1) : VRN4M1NoV0,
                 !eq(VRegClass, VRN4M2) : VRN4M2NoV0,
                 !eq(VRegClass, VRN5M1) : VRN5M1NoV0,
                 !eq(VRegClass, VRN6M1) : VRN6M1NoV0,
                 !eq(VRegClass, VRN7M1) : VRN7M1NoV0,
                 !eq(VRegClass, VRN8M1) : VRN8M1NoV0,
                 true : VRegClass);
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

class VPseudoUSLoadNoMask<VReg RetClass, int EEW, bit DummyMask = 1> :
      Pseudo<(outs RetClass:$rd),
             (ins GPRMem:$rs1, AVL:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLE</*Masked*/0, /*TU*/0, /*Strided*/0, /*FF*/0, log2<EEW>.val, VLMul> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = DummyMask;
}

class VPseudoUSLoadNoMaskTU<VReg RetClass, int EEW> :
      Pseudo<(outs RetClass:$rd),
             (ins RetClass:$dest, GPRMem:$rs1, AVL:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLE</*Masked*/0, /*TU*/1, /*Strided*/0, /*FF*/0, log2<EEW>.val, VLMul> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let HasMergeOp = 1;
  let Constraints = "$rd = $dest";
}

class VPseudoUSLoadMask<VReg RetClass, int EEW> :
      Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
              (ins GetVRegNoV0<RetClass>.R:$merge,
                   GPRMem:$rs1,
                   VMaskOp:$vm, AVL:$vl, ixlenimm:$sew, ixlenimm:$policy),[]>,
      RISCVVPseudo,
      RISCVVLE</*Masked*/1, /*TU*/1, /*Strided*/0, /*FF*/0, log2<EEW>.val, VLMul> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = "$rd = $merge";
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let HasVecPolicyOp = 1;
  let UsesMaskPolicy = 1;
}

class VPseudoUSLoadFFNoMask<VReg RetClass, int EEW, bit DummyMask = 1> :
      Pseudo<(outs RetClass:$rd, GPR:$vl),
             (ins GPRMem:$rs1, AVL:$avl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLE</*Masked*/0, /*TU*/0, /*Strided*/0, /*FF*/1, log2<EEW>.val, VLMul> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = DummyMask;
}

class VPseudoUSLoadFFNoMaskTU<VReg RetClass, int EEW> :
      Pseudo<(outs RetClass:$rd, GPR:$vl),
             (ins RetClass:$dest, GPRMem:$rs1, AVL:$avl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLE</*Masked*/0, /*TU*/1, /*Strided*/0, /*FF*/1, log2<EEW>.val, VLMul> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let HasMergeOp = 1;
  let Constraints = "$rd = $dest";
}

class VPseudoUSLoadFFMask<VReg RetClass, int EEW> :
      Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd, GPR:$vl),
              (ins GetVRegNoV0<RetClass>.R:$merge,
                   GPRMem:$rs1,
                   VMaskOp:$vm, AVL:$avl, ixlenimm:$sew, ixlenimm:$policy),[]>,
      RISCVVPseudo,
      RISCVVLE</*Masked*/1, /*TU*/1, /*Strided*/0, /*FF*/1, log2<EEW>.val, VLMul> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = "$rd = $merge";
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let HasVecPolicyOp = 1;
  let UsesMaskPolicy = 1;
}

class VPseudoSLoadNoMask<VReg RetClass, int EEW>:
      Pseudo<(outs RetClass:$rd),
             (ins GPRMem:$rs1, GPR:$rs2, AVL:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLE</*Masked*/0, /*TU*/0, /*Strided*/1, /*FF*/0, log2<EEW>.val, VLMul> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
}

class VPseudoSLoadNoMaskTU<VReg RetClass, int EEW>:
      Pseudo<(outs RetClass:$rd),
             (ins RetClass:$dest, GPRMem:$rs1, GPR:$rs2, AVL:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLE</*Masked*/0, /*TU*/1, /*Strided*/1, /*FF*/0, log2<EEW>.val, VLMul> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let HasMergeOp = 1;
  let Constraints = "$rd = $dest";
}

class VPseudoSLoadMask<VReg RetClass, int EEW>:
      Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
              (ins GetVRegNoV0<RetClass>.R:$merge,
                   GPRMem:$rs1, GPR:$rs2,
                   VMaskOp:$vm, AVL:$vl, ixlenimm:$sew, ixlenimm:$policy),[]>,
      RISCVVPseudo,
      RISCVVLE</*Masked*/1, /*TU*/1, /*Strided*/1, /*FF*/0, log2<EEW>.val, VLMul> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = "$rd = $merge";
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let HasVecPolicyOp = 1;
  let UsesMaskPolicy = 1;
}

class VPseudoILoadNoMask<VReg RetClass, VReg IdxClass, int EEW, bits<3> LMUL,
                         bit Ordered, bit EarlyClobber>:
      Pseudo<(outs RetClass:$rd),
             (ins GPRMem:$rs1, IdxClass:$rs2, AVL:$vl,
              ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLX</*Masked*/0, /*TU*/0, Ordered, log2<EEW>.val, VLMul, LMUL> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let Constraints = !if(!eq(EarlyClobber, 1), "@earlyclobber $rd", "");
}

class VPseudoILoadNoMaskTU<VReg RetClass, VReg IdxClass, int EEW, bits<3> LMUL,
                           bit Ordered, bit EarlyClobber>:
      Pseudo<(outs RetClass:$rd),
             (ins RetClass:$dest, GPRMem:$rs1, IdxClass:$rs2, AVL:$vl,
              ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLX</*Masked*/0, /*TU*/1, Ordered, log2<EEW>.val, VLMul, LMUL> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let HasMergeOp = 1;
  let Constraints = !if(!eq(EarlyClobber, 1), "@earlyclobber $rd, $rd = $dest", "$rd = $dest");
}

class VPseudoILoadMask<VReg RetClass, VReg IdxClass, int EEW, bits<3> LMUL,
                       bit Ordered, bit EarlyClobber>:
      Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
              (ins GetVRegNoV0<RetClass>.R:$merge,
                   GPRMem:$rs1, IdxClass:$rs2,
                   VMaskOp:$vm, AVL:$vl, ixlenimm:$sew, ixlenimm:$policy),[]>,
      RISCVVPseudo,
      RISCVVLX</*Masked*/1, /*TU*/1, Ordered, log2<EEW>.val, VLMul, LMUL> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = !if(!eq(EarlyClobber, 1), "@earlyclobber $rd, $rd = $merge", "$rd = $merge");
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let HasVecPolicyOp = 1;
  let UsesMaskPolicy = 1;
}

class VPseudoUSStoreNoMask<VReg StClass, int EEW, bit DummyMask = 1>:
      Pseudo<(outs),
              (ins StClass:$rd, GPRMem:$rs1, AVL:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSE</*Masked*/0, /*Strided*/0, log2<EEW>.val, VLMul> {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = DummyMask;
}

class VPseudoUSStoreMask<VReg StClass, int EEW>:
      Pseudo<(outs),
              (ins StClass:$rd, GPRMem:$rs1, VMaskOp:$vm, AVL:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSE</*Masked*/1, /*Strided*/0, log2<EEW>.val, VLMul> {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
}

class VPseudoSStoreNoMask<VReg StClass, int EEW>:
      Pseudo<(outs),
              (ins StClass:$rd, GPRMem:$rs1, GPR:$rs2, AVL:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSE</*Masked*/0, /*Strided*/1, log2<EEW>.val, VLMul> {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
}

class VPseudoSStoreMask<VReg StClass, int EEW>:
      Pseudo<(outs),
              (ins StClass:$rd, GPRMem:$rs1, GPR:$rs2, VMaskOp:$vm, AVL:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSE</*Masked*/1, /*Strided*/1, log2<EEW>.val, VLMul> {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
}

// Unary instruction that is never masked so HasDummyMask=0.
class VPseudoUnaryNoDummyMask<VReg RetClass,
                              DAGOperand Op2Class> :
        Pseudo<(outs RetClass:$rd),
               (ins Op2Class:$rs1, AVL:$vl, ixlenimm:$sew), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
}

class VPseudoUnaryNoDummyMaskTU<VReg RetClass,
                                DAGOperand Op2Class> :
        Pseudo<(outs RetClass:$rd),
               (ins RetClass:$dest, Op2Class:$rs1, AVL:$vl, ixlenimm:$sew), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let Constraints = "$rd = $dest";
}

class VPseudoNullaryNoMask<VReg RegClass>:
      Pseudo<(outs RegClass:$rd),
             (ins AVL:$vl, ixlenimm:$sew),
             []>, RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
}

class VPseudoNullaryNoMaskTU<VReg RegClass>:
      Pseudo<(outs RegClass:$rd),
             (ins RegClass:$merge, AVL:$vl, ixlenimm:$sew),
             []>, RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = "$rd = $merge";
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let HasMergeOp = 1;
}

class VPseudoNullaryMask<VReg RegClass>:
      Pseudo<(outs GetVRegNoV0<RegClass>.R:$rd),
             (ins GetVRegNoV0<RegClass>.R:$merge, VMaskOp:$vm, AVL:$vl,
              ixlenimm:$sew, ixlenimm:$policy), []>, RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints ="$rd = $merge";
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let UsesMaskPolicy = 1;
  let HasVecPolicyOp = 1;
}

// Nullary for pseudo instructions. They are expanded in
// RISCVExpandPseudoInsts pass.
class VPseudoNullaryPseudoM<string BaseInst>
       : Pseudo<(outs VR:$rd), (ins AVL:$vl, ixlenimm:$sew), []>,
       RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  // BaseInstr is not used in RISCVExpandPseudoInsts pass.
  // Just fill a corresponding real v-inst to pass tablegen check.
  let BaseInstr = !cast<Instruction>(BaseInst);
}

// RetClass could be GPR or VReg.
class VPseudoUnaryNoMask<DAGOperand RetClass, VReg OpClass, string Constraint = ""> :
        Pseudo<(outs RetClass:$rd),
               (ins OpClass:$rs2, AVL:$vl, ixlenimm:$sew), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = Constraint;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
}

// RetClass could be GPR or VReg.
class VPseudoUnaryNoMaskTU<DAGOperand RetClass, VReg OpClass, string Constraint = ""> :
      Pseudo<(outs RetClass:$rd),
        (ins RetClass:$merge, OpClass:$rs2, AVL:$vl, ixlenimm:$sew), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = Join<[Constraint, "$rd = $merge"], ",">.ret;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let HasMergeOp = 1;
}

class VPseudoUnaryMask<VReg RetClass, VReg OpClass, string Constraint = ""> :
        Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
               (ins GetVRegNoV0<RetClass>.R:$merge, OpClass:$rs2,
                    VMaskOp:$vm, AVL:$vl, ixlenimm:$sew), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = Join<[Constraint, "$rd = $merge"], ",">.ret;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let UsesMaskPolicy = 1;
}

class VPseudoUnaryMaskTA<VReg RetClass, VReg OpClass, string Constraint = ""> :
        Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
               (ins GetVRegNoV0<RetClass>.R:$merge, OpClass:$rs2,
                    VMaskOp:$vm, AVL:$vl, ixlenimm:$sew, ixlenimm:$policy), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = Join<[Constraint, "$rd = $merge"], ",">.ret;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let HasVecPolicyOp = 1;
  let UsesMaskPolicy = 1;
}

class VPseudoUnaryMaskTA_NoExcept<VReg RetClass, VReg OpClass, string Constraint = ""> :
        Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
               (ins GetVRegNoV0<RetClass>.R:$merge, OpClass:$rs2, VMaskOp:$vm,
                    AVL:$vl, ixlenimm:$sew, ixlenimm:$policy), []> {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = Join<[Constraint, "$rd = $merge"], ",">.ret;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let HasVecPolicyOp = 1;
  let UsesMaskPolicy = 1;
  let usesCustomInserter = 1;
}

class VPseudoUnaryMaskTA_FRM<VReg RetClass, VReg OpClass, string Constraint = ""> :
        Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
               (ins GetVRegNoV0<RetClass>.R:$merge, OpClass:$rs2,
                    VMaskOp:$vm, ixlenimm:$frm, AVL:$vl, ixlenimm:$sew, ixlenimm:$policy), []> {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = Join<[Constraint, "$rd = $merge"], ",">.ret;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let HasVecPolicyOp = 1;
  let UsesMaskPolicy = 1;
  let usesCustomInserter = 1;
}

// mask unary operation without maskedoff
class VPseudoMaskUnarySOutMask:
        Pseudo<(outs GPR:$rd),
               (ins VR:$rs1, VMaskOp:$vm, AVL:$vl, ixlenimm:$sew), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
}

// Mask can be V0~V31
class VPseudoUnaryAnyMask<VReg RetClass,
                          VReg Op1Class> :
      Pseudo<(outs RetClass:$rd),
             (ins RetClass:$merge,
                  Op1Class:$rs2,
                  VR:$vm, AVL:$vl, ixlenimm:$sew),
             []>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = "@earlyclobber $rd, $rd = $merge";
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
}

class VPseudoBinaryNoMask<VReg RetClass,
                          VReg Op1Class,
                          DAGOperand Op2Class,
                          string Constraint,
                          int DummyMask = 1> :
        Pseudo<(outs RetClass:$rd),
               (ins Op1Class:$rs2, Op2Class:$rs1, AVL:$vl, ixlenimm:$sew), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = Constraint;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = DummyMask;
}

class VPseudoBinaryNoMaskTU<VReg RetClass,
                            VReg Op1Class,
                            DAGOperand Op2Class,
                            string Constraint> :
        Pseudo<(outs RetClass:$rd),
               (ins RetClass:$merge, Op1Class:$rs2, Op2Class:$rs1, AVL:$vl, ixlenimm:$sew), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = Join<[Constraint, "$rd = $merge"], ",">.ret;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let HasMergeOp = 1;
}

// Special version of VPseudoBinaryNoMask where we pretend the first source is
// tied to the destination.
// This allows maskedoff and rs2 to be the same register.
class VPseudoTiedBinaryNoMask<VReg RetClass,
                              DAGOperand Op2Class,
                              string Constraint> :
        Pseudo<(outs RetClass:$rd),
               (ins RetClass:$rs2, Op2Class:$rs1, AVL:$vl, ixlenimm:$sew,
                    ixlenimm:$policy), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = Join<[Constraint, "$rd = $rs2"], ",">.ret;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let HasVecPolicyOp = 1;
  let isConvertibleToThreeAddress = 1;
}

class VPseudoIStoreNoMask<VReg StClass, VReg IdxClass, int EEW, bits<3> LMUL,
                          bit Ordered>:
      Pseudo<(outs),
              (ins StClass:$rd, GPRMem:$rs1, IdxClass:$rs2, AVL:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSX</*Masked*/0, Ordered, log2<EEW>.val, VLMul, LMUL> {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
}

class VPseudoIStoreMask<VReg StClass, VReg IdxClass, int EEW, bits<3> LMUL,
                        bit Ordered>:
      Pseudo<(outs),
              (ins StClass:$rd, GPRMem:$rs1, IdxClass:$rs2, VMaskOp:$vm, AVL:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSX</*Masked*/1, Ordered, log2<EEW>.val, VLMul, LMUL> {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
}

class VPseudoBinaryMask<VReg RetClass,
                        RegisterClass Op1Class,
                        DAGOperand Op2Class,
                        string Constraint> :
        Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
                (ins GetVRegNoV0<RetClass>.R:$merge,
                     Op1Class:$rs2, Op2Class:$rs1,
                     VMaskOp:$vm, AVL:$vl, ixlenimm:$sew), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = Join<[Constraint, "$rd = $merge"], ",">.ret;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
}

class VPseudoBinaryMaskPolicy<VReg RetClass,
                              RegisterClass Op1Class,
                              DAGOperand Op2Class,
                              string Constraint> :
        Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
                (ins GetVRegNoV0<RetClass>.R:$merge,
                     Op1Class:$rs2, Op2Class:$rs1,
                     VMaskOp:$vm, AVL:$vl, ixlenimm:$sew, ixlenimm:$policy), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = Join<[Constraint, "$rd = $merge"], ",">.ret;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let HasVecPolicyOp = 1;
  let UsesMaskPolicy = 1;
}

// Like VPseudoBinaryMask, but output can be V0.
class VPseudoBinaryMOutMask<VReg RetClass,
                            RegisterClass Op1Class,
                            DAGOperand Op2Class,
                            string Constraint> :
        Pseudo<(outs RetClass:$rd),
                (ins RetClass:$merge,
                     Op1Class:$rs2, Op2Class:$rs1,
                     VMaskOp:$vm, AVL:$vl, ixlenimm:$sew), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = Join<[Constraint, "$rd = $merge"], ",">.ret;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let UsesMaskPolicy = 1;
}

// Special version of VPseudoBinaryMask where we pretend the first source is
// tied to the destination so we can workaround the earlyclobber constraint.
// This allows maskedoff and rs2 to be the same register.
class VPseudoTiedBinaryMask<VReg RetClass,
                            DAGOperand Op2Class,
                            string Constraint> :
        Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
                (ins GetVRegNoV0<RetClass>.R:$merge,
                     Op2Class:$rs1,
                     VMaskOp:$vm, AVL:$vl, ixlenimm:$sew, ixlenimm:$policy), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = Join<[Constraint, "$rd = $merge"], ",">.ret;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 0; // Merge is also rs2.
  let HasVecPolicyOp = 1;
  let UsesMaskPolicy = 1;
}

class VPseudoBinaryCarryIn<VReg RetClass,
                           VReg Op1Class,
                           DAGOperand Op2Class,
                           LMULInfo MInfo,
                           bit CarryIn,
                           string Constraint> :
        Pseudo<(outs RetClass:$rd),
               !if(CarryIn,
                  (ins Op1Class:$rs2, Op2Class:$rs1, VMV0:$carry, AVL:$vl,
                       ixlenimm:$sew),
                  (ins Op1Class:$rs2, Op2Class:$rs1, AVL:$vl, ixlenimm:$sew)), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = Constraint;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 0;
  let VLMul = MInfo.value;
}

class VPseudoTiedBinaryCarryIn<VReg RetClass,
                               VReg Op1Class,
                               DAGOperand Op2Class,
                               LMULInfo MInfo,
                               bit CarryIn,
                               string Constraint> :
        Pseudo<(outs RetClass:$rd),
               !if(CarryIn,
                  (ins RetClass:$merge, Op1Class:$rs2, Op2Class:$rs1, VMV0:$carry, AVL:$vl,
                       ixlenimm:$sew),
                  (ins RetClass:$merge, Op1Class:$rs2, Op2Class:$rs1, AVL:$vl, ixlenimm:$sew)), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = Join<[Constraint, "$rd = $merge"], ",">.ret;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let HasVecPolicyOp = 0;
  let VLMul = MInfo.value;
}

class VPseudoTernaryNoMask<VReg RetClass,
                           RegisterClass Op1Class,
                           DAGOperand Op2Class,
                           string Constraint> :
        Pseudo<(outs RetClass:$rd),
               (ins RetClass:$rs3, Op1Class:$rs1, Op2Class:$rs2,
                    AVL:$vl, ixlenimm:$sew),
               []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = Join<[Constraint, "$rd = $rs3"], ",">.ret;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let HasDummyMask = 1;
}

class VPseudoTernaryNoMaskWithPolicy<VReg RetClass,
                                     RegisterClass Op1Class,
                                     DAGOperand Op2Class,
                                     string Constraint> :
        Pseudo<(outs RetClass:$rd),
               (ins RetClass:$rs3, Op1Class:$rs1, Op2Class:$rs2,
                    AVL:$vl, ixlenimm:$sew, ixlenimm:$policy),
               []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = Join<[Constraint, "$rd = $rs3"], ",">.ret;
  let HasVecPolicyOp = 1;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let HasDummyMask = 1;
}

class VPseudoUSSegLoadNoMask<VReg RetClass, int EEW, bits<4> NF>:
      Pseudo<(outs RetClass:$rd),
             (ins GPRMem:$rs1, AVL:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLSEG<NF, /*Masked*/0, /*TU*/0, /*Strided*/0, /*FF*/0, log2<EEW>.val, VLMul> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
}

class VPseudoUSSegLoadNoMaskTU<VReg RetClass, int EEW, bits<4> NF>:
      Pseudo<(outs RetClass:$rd),
             (ins RetClass:$dest, GPRMem:$rs1, AVL:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLSEG<NF, /*Masked*/0, /*TU*/1, /*Strided*/0, /*FF*/0, log2<EEW>.val, VLMul> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let HasMergeOp = 1;
  let Constraints = "$rd = $dest";
}

class VPseudoUSSegLoadMask<VReg RetClass, int EEW, bits<4> NF>:
      Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
             (ins GetVRegNoV0<RetClass>.R:$merge, GPRMem:$rs1,
                  VMaskOp:$vm, AVL:$vl, ixlenimm:$sew, ixlenimm:$policy),[]>,
      RISCVVPseudo,
      RISCVVLSEG<NF, /*Masked*/1, /*TU*/1, /*Strided*/0, /*FF*/0, log2<EEW>.val, VLMul> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = "$rd = $merge";
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let HasVecPolicyOp = 1;
  let UsesMaskPolicy = 1;
}

class VPseudoUSSegLoadFFNoMask<VReg RetClass, int EEW, bits<4> NF>:
      Pseudo<(outs RetClass:$rd, GPR:$vl),
             (ins GPRMem:$rs1, AVL:$avl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLSEG<NF, /*Masked*/0, /*TU*/0, /*Strided*/0, /*FF*/1, log2<EEW>.val, VLMul> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
}

class VPseudoUSSegLoadFFNoMaskTU<VReg RetClass, int EEW, bits<4> NF>:
      Pseudo<(outs RetClass:$rd, GPR:$vl),
             (ins RetClass:$dest, GPRMem:$rs1, AVL:$avl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLSEG<NF, /*Masked*/0, /*TU*/1, /*Strided*/0, /*FF*/1, log2<EEW>.val, VLMul> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let HasMergeOp = 1;
  let Constraints = "$rd = $dest";
}

class VPseudoUSSegLoadFFMask<VReg RetClass, int EEW, bits<4> NF>:
      Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd, GPR:$vl),
             (ins GetVRegNoV0<RetClass>.R:$merge, GPRMem:$rs1,
                  VMaskOp:$vm, AVL:$avl, ixlenimm:$sew, ixlenimm:$policy),[]>,
      RISCVVPseudo,
      RISCVVLSEG<NF, /*Masked*/1, /*TU*/1, /*Strided*/0, /*FF*/1, log2<EEW>.val, VLMul> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = "$rd = $merge";
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let HasVecPolicyOp = 1;
  let UsesMaskPolicy = 1;
}

class VPseudoSSegLoadNoMask<VReg RetClass, int EEW, bits<4> NF>:
      Pseudo<(outs RetClass:$rd),
             (ins GPRMem:$rs1, GPR:$offset, AVL:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLSEG<NF, /*Masked*/0, /*TU*/0, /*Strided*/1, /*FF*/0, log2<EEW>.val, VLMul> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
}

class VPseudoSSegLoadNoMaskTU<VReg RetClass, int EEW, bits<4> NF>:
      Pseudo<(outs RetClass:$rd),
             (ins RetClass:$merge, GPRMem:$rs1, GPR:$offset, AVL:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLSEG<NF, /*Masked*/0, /*TU*/1, /*Strided*/1, /*FF*/0, log2<EEW>.val, VLMul> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let HasMergeOp = 1;
  let Constraints = "$rd = $merge";
}

class VPseudoSSegLoadMask<VReg RetClass, int EEW, bits<4> NF>:
      Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
             (ins GetVRegNoV0<RetClass>.R:$merge, GPRMem:$rs1,
                  GPR:$offset, VMaskOp:$vm, AVL:$vl, ixlenimm:$sew,
                  ixlenimm:$policy),[]>,
      RISCVVPseudo,
      RISCVVLSEG<NF, /*Masked*/1, /*TU*/1, /*Strided*/1, /*FF*/0, log2<EEW>.val, VLMul> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = "$rd = $merge";
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let HasVecPolicyOp = 1;
  let UsesMaskPolicy = 1;
}

class VPseudoISegLoadNoMask<VReg RetClass, VReg IdxClass, int EEW, bits<3> LMUL,
                            bits<4> NF, bit Ordered>:
      Pseudo<(outs RetClass:$rd),
             (ins GPRMem:$rs1, IdxClass:$offset, AVL:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLXSEG<NF, /*Masked*/0, /*TU*/0, Ordered, log2<EEW>.val, VLMul, LMUL> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  // For vector indexed segment loads, the destination vector register groups
  // cannot overlap the source vector register group
  let Constraints = "@earlyclobber $rd";
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
}

class VPseudoISegLoadNoMaskTU<VReg RetClass, VReg IdxClass, int EEW, bits<3> LMUL,
                              bits<4> NF, bit Ordered>:
      Pseudo<(outs RetClass:$rd),
             (ins RetClass:$merge, GPRMem:$rs1, IdxClass:$offset, AVL:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLXSEG<NF, /*Masked*/0, /*TU*/1, Ordered, log2<EEW>.val, VLMul, LMUL> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  // For vector indexed segment loads, the destination vector register groups
  // cannot overlap the source vector register group
  let Constraints = "@earlyclobber $rd, $rd = $merge";
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let HasMergeOp = 1;
}

class VPseudoISegLoadMask<VReg RetClass, VReg IdxClass, int EEW, bits<3> LMUL,
                          bits<4> NF, bit Ordered>:
      Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
             (ins GetVRegNoV0<RetClass>.R:$merge, GPRMem:$rs1,
                  IdxClass:$offset, VMaskOp:$vm, AVL:$vl, ixlenimm:$sew,
                  ixlenimm:$policy),[]>,
      RISCVVPseudo,
      RISCVVLXSEG<NF, /*Masked*/1, /*TU*/1, Ordered, log2<EEW>.val, VLMul, LMUL> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  // For vector indexed segment loads, the destination vector register groups
  // cannot overlap the source vector register group
  let Constraints = "@earlyclobber $rd, $rd = $merge";
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let HasVecPolicyOp = 1;
  let UsesMaskPolicy = 1;
}

class VPseudoUSSegStoreNoMask<VReg ValClass, int EEW, bits<4> NF>:
      Pseudo<(outs),
             (ins ValClass:$rd, GPRMem:$rs1, AVL:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSSEG<NF, /*Masked*/0, /*Strided*/0, log2<EEW>.val, VLMul> {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
}

class VPseudoUSSegStoreMask<VReg ValClass, int EEW, bits<4> NF>:
      Pseudo<(outs),
             (ins ValClass:$rd, GPRMem:$rs1,
                  VMaskOp:$vm, AVL:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSSEG<NF, /*Masked*/1, /*Strided*/0, log2<EEW>.val, VLMul> {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
}

class VPseudoSSegStoreNoMask<VReg ValClass, int EEW, bits<4> NF>:
      Pseudo<(outs),
             (ins ValClass:$rd, GPRMem:$rs1, GPR: $offset, AVL:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSSEG<NF, /*Masked*/0, /*Strided*/1, log2<EEW>.val, VLMul> {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
}

class VPseudoSSegStoreMask<VReg ValClass, int EEW, bits<4> NF>:
      Pseudo<(outs),
             (ins ValClass:$rd, GPRMem:$rs1, GPR: $offset,
                  VMaskOp:$vm, AVL:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSSEG<NF, /*Masked*/1, /*Strided*/1, log2<EEW>.val, VLMul> {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
}

class VPseudoISegStoreNoMask<VReg ValClass, VReg IdxClass, int EEW, bits<3> LMUL,
                             bits<4> NF, bit Ordered>:
      Pseudo<(outs),
             (ins ValClass:$rd, GPRMem:$rs1, IdxClass: $index,
                  AVL:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSXSEG<NF, /*Masked*/0, Ordered, log2<EEW>.val, VLMul, LMUL> {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
}

class VPseudoISegStoreMask<VReg ValClass, VReg IdxClass, int EEW, bits<3> LMUL,
                           bits<4> NF, bit Ordered>:
      Pseudo<(outs),
             (ins ValClass:$rd, GPRMem:$rs1, IdxClass: $index,
                  VMaskOp:$vm, AVL:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSXSEG<NF, /*Masked*/1, Ordered, log2<EEW>.val, VLMul, LMUL> {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
}

multiclass VPseudoUSLoad {
  foreach eew = EEWList in {
    foreach lmul = MxSet<eew>.m in {
      defvar LInfo = lmul.MX;
      defvar vreg = lmul.vrclass;
      let VLMul = lmul.value in {
        def "E" # eew # "_V_" # LInfo :
          VPseudoUSLoadNoMask<vreg, eew>,
          VLESched;
        def "E" # eew # "_V_" # LInfo # "_TU":
          VPseudoUSLoadNoMaskTU<vreg, eew>,
          VLESched;
        def "E" # eew # "_V_" # LInfo # "_MASK" :
          VPseudoUSLoadMask<vreg, eew>,
          RISCVMaskedPseudo</*MaskOpIdx*/ 2>,
          VLESched;
      }
    }
  }
}

multiclass VPseudoFFLoad {
  foreach eew = EEWList in {
    foreach lmul = MxSet<eew>.m in {
      defvar LInfo = lmul.MX;
      defvar vreg = lmul.vrclass;
      let VLMul = lmul.value in {
        def "E" # eew # "FF_V_" # LInfo:
          VPseudoUSLoadFFNoMask<vreg, eew>,
          VLFSched;
        def "E" # eew # "FF_V_" # LInfo # "_TU":
          VPseudoUSLoadFFNoMaskTU<vreg, eew>,
          VLFSched;
        def "E" # eew # "FF_V_" # LInfo # "_MASK":
          VPseudoUSLoadFFMask<vreg, eew>,
          RISCVMaskedPseudo</*MaskOpIdx*/ 2>,
          VLFSched;
      }
    }
  }
}

multiclass VPseudoLoadMask {
  foreach mti = AllMasks in {
    let VLMul = mti.LMul.value in {
      def "_V_" # mti.BX : VPseudoUSLoadNoMask<VR, /*EEW*/1, /*DummyMask*/0>;
    }
  }
}

multiclass VPseudoSLoad {
  foreach eew = EEWList in {
    foreach lmul = MxSet<eew>.m in {
      defvar LInfo = lmul.MX;
      defvar vreg = lmul.vrclass;
      let VLMul = lmul.value in {
        def "E" # eew # "_V_" # LInfo : VPseudoSLoadNoMask<vreg, eew>,
                                        VLSSched<eew>;
        def "E" # eew # "_V_" # LInfo # "_TU": VPseudoSLoadNoMaskTU<vreg, eew>,
                                        VLSSched<eew>;
        def "E" # eew # "_V_" # LInfo # "_MASK" :
          VPseudoSLoadMask<vreg, eew>,
          RISCVMaskedPseudo</*MaskOpIdx*/ 3>,
          VLSSched<eew>;
      }
    }
  }
}

multiclass VPseudoILoad<bit Ordered> {
  foreach eew = EEWList in {
    foreach sew = EEWList in {
      foreach lmul = MxSet<sew>.m in {
        defvar octuple_lmul = lmul.octuple;
        // Calculate emul = eew * lmul / sew
        defvar octuple_emul = !srl(!mul(eew, octuple_lmul), log2<sew>.val);
        if !and(!ge(octuple_emul, 1), !le(octuple_emul, 64)) then {
          defvar LInfo = lmul.MX;
          defvar IdxLInfo = octuple_to_str<octuple_emul>.ret;
          defvar idx_lmul = !cast<LMULInfo>("V_" # IdxLInfo);
          defvar Vreg = lmul.vrclass;
          defvar IdxVreg = idx_lmul.vrclass;
          defvar HasConstraint = !ne(sew, eew);
          defvar Order = !if(Ordered, "O", "U");
          let VLMul = lmul.value in {
            def "EI" # eew # "_V_" # IdxLInfo # "_" # LInfo :
              VPseudoILoadNoMask<Vreg, IdxVreg, eew, idx_lmul.value, Ordered, HasConstraint>,
              VLXSched<eew, Order>;
            def "EI" # eew # "_V_" # IdxLInfo # "_" # LInfo # "_TU":
              VPseudoILoadNoMaskTU<Vreg, IdxVreg, eew, idx_lmul.value, Ordered, HasConstraint>,
              VLXSched<eew, Order>;
            def "EI" # eew # "_V_" # IdxLInfo # "_" # LInfo # "_MASK" :
              VPseudoILoadMask<Vreg, IdxVreg, eew, idx_lmul.value, Ordered, HasConstraint>,
              RISCVMaskedPseudo</*MaskOpIdx*/ 3>,
              VLXSched<eew, Order>;
          }
        }
      }
    }
  }
}

multiclass VPseudoUSStore {
  foreach eew = EEWList in {
    foreach lmul = MxSet<eew>.m in {
      defvar LInfo = lmul.MX;
      defvar vreg = lmul.vrclass;
      let VLMul = lmul.value in {
        def "E" # eew # "_V_" # LInfo : VPseudoUSStoreNoMask<vreg, eew>,
                                        VSESched;
        def "E" # eew # "_V_" # LInfo # "_MASK" : VPseudoUSStoreMask<vreg, eew>,
                                                  VSESched;
      }
    }
  }
}

multiclass VPseudoStoreMask {
  foreach mti = AllMasks in {
    let VLMul = mti.LMul.value in {
      def "_V_" # mti.BX : VPseudoUSStoreNoMask<VR, /*EEW*/1, /*DummyMask*/0>;
    }
  }
}

multiclass VPseudoSStore {
  foreach eew = EEWList in {
    foreach lmul = MxSet<eew>.m in {
      defvar LInfo = lmul.MX;
      defvar vreg = lmul.vrclass;
      let VLMul = lmul.value in {
        def "E" # eew # "_V_" # LInfo : VPseudoSStoreNoMask<vreg, eew>,
                                        VSSSched<eew>;
        def "E" # eew # "_V_" # LInfo # "_MASK" : VPseudoSStoreMask<vreg, eew>,
                                                  VSSSched<eew>;
      }
    }
  }
}

multiclass VPseudoIStore<bit Ordered> {
  foreach eew = EEWList in {
    foreach sew = EEWList in {
      foreach lmul = MxSet<sew>.m in {
        defvar octuple_lmul = lmul.octuple;
        // Calculate emul = eew * lmul / sew
        defvar octuple_emul = !srl(!mul(eew, octuple_lmul), log2<sew>.val);
        if !and(!ge(octuple_emul, 1), !le(octuple_emul, 64)) then {
          defvar LInfo = lmul.MX;
          defvar IdxLInfo = octuple_to_str<octuple_emul>.ret;
          defvar idx_lmul = !cast<LMULInfo>("V_" # IdxLInfo);
          defvar Vreg = lmul.vrclass;
          defvar IdxVreg = idx_lmul.vrclass;
          defvar Order = !if(Ordered, "O", "U");
          let VLMul = lmul.value in {
            def "EI" # eew # "_V_" # IdxLInfo # "_" # LInfo :
              VPseudoIStoreNoMask<Vreg, IdxVreg, eew, idx_lmul.value, Ordered>,
              VSXSched<eew, Order>;
            def "EI" # eew # "_V_" # IdxLInfo # "_" # LInfo # "_MASK" :
              VPseudoIStoreMask<Vreg, IdxVreg, eew, idx_lmul.value, Ordered>,
              VSXSched<eew, Order>;
          }
        }
      }
    }
  }
}

multiclass VPseudoVPOP_M {
  foreach mti = AllMasks in
  {
    let VLMul = mti.LMul.value in {
      def "_M_" # mti.BX : VPseudoUnaryNoMask<GPR, VR>,
                           Sched<[WriteVMPopV, ReadVMPopV, ReadVMPopV]>;
      def "_M_" # mti.BX # "_MASK" : VPseudoMaskUnarySOutMask,
                                     Sched<[WriteVMPopV, ReadVMPopV, ReadVMPopV]>;
    }
  }
}

multiclass VPseudoV1ST_M {
  foreach mti = AllMasks in
  {
    let VLMul = mti.LMul.value in {
      def "_M_" # mti.BX : VPseudoUnaryNoMask<GPR, VR>,
                           Sched<[WriteVMFFSV, ReadVMFFSV, ReadVMFFSV]>;
      def "_M_" # mti.BX # "_MASK" : VPseudoMaskUnarySOutMask,
                                     Sched<[WriteVMFFSV, ReadVMFFSV, ReadVMFFSV]>;
    }
  }
}

multiclass VPseudoVSFS_M {
  defvar constraint = "@earlyclobber $rd";
  foreach mti = AllMasks in
  {
    let VLMul = mti.LMul.value in {
      def "_M_" # mti.BX : VPseudoUnaryNoMask<VR, VR, constraint>,
                           Sched<[WriteVMSFSV, ReadVMSFSV, ReadVMask]>;
      def "_M_" # mti.BX # "_MASK" : VPseudoUnaryMask<VR, VR, constraint>,
                                     Sched<[WriteVMSFSV, ReadVMSFSV, ReadVMask]>;
    }
  }
}

multiclass VPseudoVID_V {
  foreach m = MxList in {
    let VLMul = m.value in {
      def "_V_" # m.MX : VPseudoNullaryNoMask<m.vrclass>,
                         Sched<[WriteVMIdxV, ReadVMask]>;
      def "_V_" # m.MX # "_TU": VPseudoNullaryNoMaskTU<m.vrclass>,
                                Sched<[WriteVMIdxV, ReadVMask]>;
      def "_V_" # m.MX # "_MASK" : VPseudoNullaryMask<m.vrclass>,
                                   RISCVMaskedPseudo</*MaskOpIdx*/ 1>,
                                   Sched<[WriteVMIdxV, ReadVMask]>;
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

multiclass VPseudoVIOT_M {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxList in {
    let VLMul = m.value in {
      def "_" # m.MX : VPseudoUnaryNoMask<m.vrclass, VR, constraint>,
                       Sched<[WriteVMIotV, ReadVMIotV, ReadVMask]>;
      def "_" # m.MX # "_TU" : VPseudoUnaryNoMaskTU<m.vrclass, VR, constraint>,
                               Sched<[WriteVMIotV, ReadVMIotV, ReadVMask]>;
      def "_" # m.MX # "_MASK" : VPseudoUnaryMaskTA<m.vrclass, VR, constraint>,
                                 RISCVMaskedPseudo</*MaskOpIdx*/ 2>,
                                 Sched<[WriteVMIotV, ReadVMIotV, ReadVMask]>;
    }
  }
}

multiclass VPseudoVCPR_V {
  foreach m = MxList in {
    let VLMul = m.value in
      def _VM # "_" # m.MX : VPseudoUnaryAnyMask<m.vrclass, m.vrclass>,
                             Sched<[WriteVCompressV, ReadVCompressV, ReadVCompressV]>;
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
    def "_" # MInfo.MX # "_TU" : VPseudoBinaryNoMaskTU<RetClass, Op1Class, Op2Class,
                                                       Constraint>;
    def "_" # MInfo.MX # "_MASK" : VPseudoBinaryMaskPolicy<RetClass, Op1Class, Op2Class,
                                                           Constraint>,
                                   RISCVMaskedPseudo</*MaskOpIdx*/ 3>;
  }
}

multiclass VPseudoBinaryM<VReg RetClass,
                          VReg Op1Class,
                          DAGOperand Op2Class,
                          LMULInfo MInfo,
                          string Constraint = ""> {
  let VLMul = MInfo.value in {
    def "_" # MInfo.MX : VPseudoBinaryNoMask<RetClass, Op1Class, Op2Class,
                                             Constraint>;
    let ForceTailAgnostic = true in
    def "_" # MInfo.MX # "_MASK" : VPseudoBinaryMOutMask<RetClass, Op1Class,
                                                         Op2Class, Constraint>,
                                   RISCVMaskedPseudo</*MaskOpIdx*/ 3, /*HasTU*/ false>;
  }
}

multiclass VPseudoBinaryEmul<VReg RetClass,
                             VReg Op1Class,
                             DAGOperand Op2Class,
                             LMULInfo lmul,
                             LMULInfo emul,
                             string Constraint = ""> {
  let VLMul = lmul.value in {
    def "_" # lmul.MX # "_" # emul.MX : VPseudoBinaryNoMask<RetClass, Op1Class, Op2Class,
                                                            Constraint>;
    def "_" # lmul.MX # "_" # emul.MX # "_TU": VPseudoBinaryNoMaskTU<RetClass, Op1Class, Op2Class,
                                                                     Constraint>;
    def "_" # lmul.MX # "_" # emul.MX # "_MASK" : VPseudoBinaryMaskPolicy<RetClass, Op1Class, Op2Class,
                                                                          Constraint>,
                                                  RISCVMaskedPseudo</*MaskOpIdx*/ 3>;
  }
}

multiclass VPseudoTiedBinary<VReg RetClass,
                             DAGOperand Op2Class,
                             LMULInfo MInfo,
                             string Constraint = ""> {
  let VLMul = MInfo.value in {
    def "_" # MInfo.MX # "_TIED": VPseudoTiedBinaryNoMask<RetClass, Op2Class,
                                                          Constraint>;
    def "_" # MInfo.MX # "_MASK_TIED" : VPseudoTiedBinaryMask<RetClass, Op2Class,
                                                         Constraint>;
  }
}

multiclass VPseudoBinaryV_VV<string Constraint = ""> {
  foreach m = MxList in
    defm _VV : VPseudoBinary<m.vrclass, m.vrclass, m.vrclass, m, Constraint>;
}

multiclass VPseudoBinaryV_VV_LMUL<LMULInfo m, string Constraint = ""> {
  defm _VV : VPseudoBinary<m.vrclass, m.vrclass, m.vrclass, m, Constraint>;
}

// Similar to VPseudoBinaryV_VV, but uses MxListF.
multiclass VPseudoBinaryFV_VV<LMULInfo m, string Constraint = ""> {
  defm _VV : VPseudoBinary<m.vrclass, m.vrclass, m.vrclass, m, Constraint>;
}

multiclass VPseudoVGTR_VV_EEW<int eew, string Constraint = ""> {
  foreach m = MxList in {
    foreach sew = EEWList in {
      defvar octuple_lmul = m.octuple;
      // emul = lmul * eew / sew
      defvar octuple_emul = !srl(!mul(octuple_lmul, eew), log2<sew>.val);
      if !and(!ge(octuple_emul, 1), !le(octuple_emul, 64)) then {
        defvar emulMX = octuple_to_str<octuple_emul>.ret;
        defvar emul = !cast<LMULInfo>("V_" # emulMX);
        defm _VV : VPseudoBinaryEmul<m.vrclass, m.vrclass, emul.vrclass, m, emul, Constraint>,
                   Sched<[WriteVGatherV, ReadVGatherV, ReadVGatherV]>;
      }
    }
  }
}

multiclass VPseudoBinaryV_VX<string Constraint = ""> {
  foreach m = MxList in
    defm "_VX" : VPseudoBinary<m.vrclass, m.vrclass, GPR, m, Constraint>;
}

multiclass VPseudoBinaryV_VX_LMUL<LMULInfo m, string Constraint = ""> {
  defm "_VX" : VPseudoBinary<m.vrclass, m.vrclass, GPR, m, Constraint>;
}

multiclass VPseudoVSLD1_VX<string Constraint = ""> {
  foreach m = MxList in
    defm "_VX" : VPseudoBinary<m.vrclass, m.vrclass, GPR, m, Constraint>,
                 Sched<[WriteVISlide1X, ReadVISlideV, ReadVISlideX, ReadVMask]>;
}

multiclass VPseudoBinaryV_VF<LMULInfo m, FPR_Info f, string Constraint = ""> {
  defm "_V" # f.FX : VPseudoBinary<m.vrclass, m.vrclass,
                                   f.fprclass, m, Constraint>;
}

multiclass VPseudoVSLD1_VF<string Constraint = ""> {
  foreach f = FPList in
    foreach m = f.MxList in
      defm "_V" # f.FX :
        VPseudoBinary<m.vrclass, m.vrclass, f.fprclass, m, Constraint>,
        Sched<[WriteVFSlide1F, ReadVFSlideV, ReadVFSlideF, ReadVMask]>;
}

multiclass VPseudoBinaryV_VI<Operand ImmType = simm5, string Constraint = ""> {
  foreach m = MxList in
    defm _VI : VPseudoBinary<m.vrclass, m.vrclass, ImmType, m, Constraint>;
}

multiclass VPseudoBinaryV_VI_LMUL<Operand ImmType = simm5, LMULInfo m, string Constraint = ""> {
  defm _VI : VPseudoBinary<m.vrclass, m.vrclass, ImmType, m, Constraint>;
}

multiclass VPseudoVALU_MM {
  foreach m = MxList in
    let VLMul = m.value in {
      def "_MM_" # m.MX : VPseudoBinaryNoMask<VR, VR, VR, "", /*DummyMask*/0>,
                          Sched<[WriteVMALUV, ReadVMALUV, ReadVMALUV]>;
    }
}

// We use earlyclobber here due to
// * The destination EEW is smaller than the source EEW and the overlap is
//   in the lowest-numbered part of the source register group is legal.
//   Otherwise, it is illegal.
// * The destination EEW is greater than the source EEW, the source EMUL is
//   at least 1, and the overlap is in the highest-numbered part of the
//   destination register group is legal. Otherwise, it is illegal.
multiclass VPseudoBinaryW_VV<list<LMULInfo> mxlist = MxListW> {
  foreach m = mxlist in
    defm _VV : VPseudoBinary<m.wvrclass, m.vrclass, m.vrclass, m,
                             "@earlyclobber $rd">;
}

multiclass VPseudoBinaryW_VV_LMUL<LMULInfo m> {
  defm _VV : VPseudoBinary<m.wvrclass, m.vrclass, m.vrclass, m,
                           "@earlyclobber $rd">;
}

multiclass VPseudoBinaryW_VX_LMUL<LMULInfo m> {
  defm "_VX" : VPseudoBinary<m.wvrclass, m.vrclass, GPR, m,
                             "@earlyclobber $rd">;
}

multiclass VPseudoBinaryW_VF<LMULInfo m, FPR_Info f> {
  defm "_V" # f.FX : VPseudoBinary<m.wvrclass, m.vrclass,
                                   f.fprclass, m,
                                   "@earlyclobber $rd">;
}

multiclass VPseudoBinaryW_WV<list<LMULInfo> mxlist = MxListW> {
  foreach m = mxlist in {
    defm _WV : VPseudoBinary<m.wvrclass, m.wvrclass, m.vrclass, m,
                             "@earlyclobber $rd">;
    defm _WV : VPseudoTiedBinary<m.wvrclass, m.vrclass, m,
                                 "@earlyclobber $rd">;
  }
}

multiclass VPseudoBinaryW_WV_LMUL<LMULInfo m> {
  defm _WV : VPseudoBinary<m.wvrclass, m.wvrclass, m.vrclass, m,
                           "@earlyclobber $rd">;
  defm _WV : VPseudoTiedBinary<m.wvrclass, m.vrclass, m,
                               "@earlyclobber $rd">;
}

multiclass VPseudoBinaryW_WX_LMUL<LMULInfo m> {
  defm "_WX" : VPseudoBinary<m.wvrclass, m.wvrclass, GPR, m>;
}

multiclass VPseudoBinaryW_WF<LMULInfo m, FPR_Info f> {
  defm "_W" # f.FX : VPseudoBinary<m.wvrclass, m.wvrclass,
                                   f.fprclass, m>;
}

// Narrowing instructions like vnsrl/vnsra/vnclip(u) don't need @earlyclobber
// if the source and destination have an LMUL<=1. This matches this overlap
// exception from the spec.
// "The destination EEW is smaller than the source EEW and the overlap is in the
//  lowest-numbered part of the source register group."
multiclass VPseudoBinaryV_WV {
  foreach m = MxListW in
    defm _WV : VPseudoBinary<m.vrclass, m.wvrclass, m.vrclass, m,
                             !if(!ge(m.octuple, 8), "@earlyclobber $rd", "")>;
}

multiclass VPseudoBinaryV_WV_LMUL<LMULInfo m> {
  defm _WV : VPseudoBinary<m.vrclass, m.wvrclass, m.vrclass, m,
                           !if(!ge(m.octuple, 8), "@earlyclobber $rd", "")>;
}

multiclass VPseudoBinaryV_WX {
  foreach m = MxListW in
    defm _WX : VPseudoBinary<m.vrclass, m.wvrclass, GPR, m,
                             !if(!ge(m.octuple, 8), "@earlyclobber $rd", "")>;
}

multiclass VPseudoBinaryV_WX_LMUL<LMULInfo m> {
  defm _WX : VPseudoBinary<m.vrclass, m.wvrclass, GPR, m,
                           !if(!ge(m.octuple, 8), "@earlyclobber $rd", "")>;
}

multiclass VPseudoBinaryV_WI {
  foreach m = MxListW in
    defm _WI : VPseudoBinary<m.vrclass, m.wvrclass, uimm5, m,
                             !if(!ge(m.octuple, 8), "@earlyclobber $rd", "")>;
}

multiclass VPseudoBinaryV_WI_LMUL<LMULInfo m> {
  defm _WI : VPseudoBinary<m.vrclass, m.wvrclass, uimm5, m,
                           !if(!ge(m.octuple, 8), "@earlyclobber $rd", "")>;
}

// For vadc and vsbc, the instruction encoding is reserved if the destination
// vector register is v0.
// For vadc and vsbc, CarryIn == 1 and CarryOut == 0
multiclass VPseudoBinaryV_VM<LMULInfo m, bit CarryOut = 0, bit CarryIn = 1,
                             string Constraint = ""> {
  def "_VV" # !if(CarryIn, "M", "") # "_" # m.MX :
    VPseudoBinaryCarryIn<!if(CarryOut, VR,
                         !if(!and(CarryIn, !not(CarryOut)),
                             GetVRegNoV0<m.vrclass>.R, m.vrclass)),
                         m.vrclass, m.vrclass, m, CarryIn, Constraint>;
}

multiclass VPseudoTiedBinaryV_VM<LMULInfo m, bit CarryOut = 0, bit CarryIn = 1,
                                 string Constraint = ""> {
  def "_VV" # !if(CarryIn, "M", "") # "_" # m.MX # "_TU" :
    VPseudoTiedBinaryCarryIn<!if(CarryOut, VR,
                             !if(!and(CarryIn, !not(CarryOut)),
                                 GetVRegNoV0<m.vrclass>.R, m.vrclass)),
                             m.vrclass, m.vrclass, m, CarryIn, Constraint>;
}

multiclass VPseudoBinaryV_XM<LMULInfo m, bit CarryOut = 0, bit CarryIn = 1,
                             string Constraint = ""> {
  def "_VX" # !if(CarryIn, "M", "") # "_" # m.MX :
    VPseudoBinaryCarryIn<!if(CarryOut, VR,
                         !if(!and(CarryIn, !not(CarryOut)),
                             GetVRegNoV0<m.vrclass>.R, m.vrclass)),
                         m.vrclass, GPR, m, CarryIn, Constraint>;
}

multiclass VPseudoTiedBinaryV_XM<LMULInfo m, bit CarryOut = 0, bit CarryIn = 1,
                                 string Constraint = ""> {
  def "_VX" # !if(CarryIn, "M", "") # "_" # m.MX # "_TU":
    VPseudoTiedBinaryCarryIn<!if(CarryOut, VR,
                             !if(!and(CarryIn, !not(CarryOut)),
                                 GetVRegNoV0<m.vrclass>.R, m.vrclass)),
                             m.vrclass, GPR, m, CarryIn, Constraint>;
}

multiclass VPseudoVMRG_FM {
  foreach f = FPList in {
    foreach m = f.MxList in {
      defvar mx = m.MX;
      defvar WriteVFMergeV_MX = !cast<SchedWrite>("WriteVFMergeV_" # mx);
      defvar ReadVFMergeV_MX = !cast<SchedRead>("ReadVFMergeV_" # mx);
      defvar ReadVFMergeF_MX = !cast<SchedRead>("ReadVFMergeF_" # mx);

      def "_V" # f.FX # "M_" # mx :
        VPseudoBinaryCarryIn<GetVRegNoV0<m.vrclass>.R,
                             m.vrclass, f.fprclass, m, /*CarryIn=*/1, "">,
        Sched<[WriteVFMergeV_MX, ReadVFMergeV_MX, ReadVFMergeF_MX, ReadVMask]>;
      // Tied version to allow codegen control over the tail elements
      def "_V" # f.FX # "M_" # mx # "_TU":
        VPseudoTiedBinaryCarryIn<GetVRegNoV0<m.vrclass>.R,
                                 m.vrclass, f.fprclass, m, /*CarryIn=*/1, "">,
        Sched<[WriteVFMergeV_MX, ReadVFMergeV_MX, ReadVFMergeF_MX, ReadVMask]>;
    }
  }
}

multiclass VPseudoBinaryV_IM<LMULInfo m, bit CarryOut = 0, bit CarryIn = 1,
                             string Constraint = ""> {
  def "_VI" # !if(CarryIn, "M", "") # "_" # m.MX :
    VPseudoBinaryCarryIn<!if(CarryOut, VR,
                         !if(!and(CarryIn, !not(CarryOut)),
                             GetVRegNoV0<m.vrclass>.R, m.vrclass)),
                         m.vrclass, simm5, m, CarryIn, Constraint>;
}

multiclass VPseudoTiedBinaryV_IM<LMULInfo m, bit CarryOut = 0, bit CarryIn = 1,
                                 string Constraint = ""> {
  def "_VI" # !if(CarryIn, "M", "") # "_" # m.MX # "_TU":
    VPseudoTiedBinaryCarryIn<!if(CarryOut, VR,
                             !if(!and(CarryIn, !not(CarryOut)),
                                 GetVRegNoV0<m.vrclass>.R, m.vrclass)),
                             m.vrclass, simm5, m, CarryIn, Constraint>;
}

multiclass VPseudoUnaryVMV_V_X_I {
  foreach m = MxList in {
    let VLMul = m.value in {
      defvar mx = m.MX;
      defvar WriteVIMovV_MX = !cast<SchedWrite>("WriteVIMovV_" # mx);
      defvar WriteVIMovX_MX = !cast<SchedWrite>("WriteVIMovX_" # mx);
      defvar WriteVIMovI_MX = !cast<SchedWrite>("WriteVIMovI_" # mx);
      defvar ReadVIMovV_MX = !cast<SchedRead>("ReadVIMovV_" # mx);
      defvar ReadVIMovX_MX = !cast<SchedRead>("ReadVIMovX_" # mx);

      let VLMul = m.value in {
        def "_V_" # mx : VPseudoUnaryNoDummyMask<m.vrclass, m.vrclass>,
                           Sched<[WriteVIMovV_MX, ReadVIMovV_MX]>;
        def "_X_" # mx : VPseudoUnaryNoDummyMask<m.vrclass, GPR>,
                           Sched<[WriteVIMovX_MX, ReadVIMovX_MX]>;
        def "_I_" # mx : VPseudoUnaryNoDummyMask<m.vrclass, simm5>,
                           Sched<[WriteVIMovI_MX]>;
        def "_V_" # mx # "_TU": VPseudoUnaryNoDummyMaskTU<m.vrclass, m.vrclass>,
                           Sched<[WriteVIMovV_MX, ReadVIMovV_MX]>;
        def "_X_" # mx # "_TU": VPseudoUnaryNoDummyMaskTU<m.vrclass, GPR>,
                           Sched<[WriteVIMovX_MX, ReadVIMovX_MX]>;
        def "_I_" # mx # "_TU": VPseudoUnaryNoDummyMaskTU<m.vrclass, simm5>,
                           Sched<[WriteVIMovI_MX]>;
      }
    }
  }
}

multiclass VPseudoVMV_F {
  foreach f = FPList in {
    foreach m = f.MxList in {
      defvar mx = m.MX;
      defvar WriteVFMovV_MX = !cast<SchedWrite>("WriteVFMovV_" # mx);
      defvar ReadVFMovF_MX = !cast<SchedRead>("ReadVFMovF_" # mx);

      let VLMul = m.value in {
        def "_" # f.FX # "_" # mx :
          VPseudoUnaryNoDummyMask<m.vrclass, f.fprclass>,
          Sched<[WriteVFMovV_MX, ReadVFMovF_MX]>;
        def "_" # f.FX # "_" # mx # "_TU":
          VPseudoUnaryNoDummyMaskTU<m.vrclass, f.fprclass>,
          Sched<[WriteVFMovV_MX, ReadVFMovF_MX]>;
      }
    }
  }
}

multiclass VPseudoVCLS_V {
  foreach m = MxListF in {
    defvar mx = m.MX;
    defvar WriteVFClassV_MX = !cast<SchedWrite>("WriteVFClassV_" # mx);
    defvar ReadVFClassV_MX = !cast<SchedRead>("ReadVFClassV_" # mx);

    let VLMul = m.value in {
      def "_V_" # mx : VPseudoUnaryNoMask<m.vrclass, m.vrclass>,
                       Sched<[WriteVFClassV_MX, ReadVFClassV_MX, ReadVMask]>;
      def "_V_" # mx # "_TU": VPseudoUnaryNoMaskTU<m.vrclass, m.vrclass>,
                              Sched<[WriteVFClassV_MX, ReadVFClassV_MX, ReadVMask]>;
      def "_V_" # mx # "_MASK" : VPseudoUnaryMaskTA<m.vrclass, m.vrclass>,
                                 RISCVMaskedPseudo</*MaskOpIdx*/ 2>,
                                 Sched<[WriteVFClassV_MX, ReadVFClassV_MX, ReadVMask]>;
    }
  }
}

multiclass VPseudoVSQR_V {
  foreach m = MxListF in {
    defvar mx = m.MX;
    defvar WriteVFSqrtV_MX = !cast<SchedWrite>("WriteVFSqrtV_" # mx);
    defvar ReadVFSqrtV_MX = !cast<SchedRead>("ReadVFSqrtV_" # mx);

    let VLMul = m.value in {
      def "_V_" # mx : VPseudoUnaryNoMask<m.vrclass, m.vrclass>,
                       Sched<[WriteVFSqrtV_MX, ReadVFSqrtV_MX, ReadVMask]>;
      def "_V_" # mx # "_TU": VPseudoUnaryNoMaskTU<m.vrclass, m.vrclass>,
                              Sched<[WriteVFSqrtV_MX, ReadVFSqrtV_MX, ReadVMask]>;
      def "_V_" # mx # "_MASK" : VPseudoUnaryMaskTA<m.vrclass, m.vrclass>,
                                 RISCVMaskedPseudo</*MaskOpIdx*/ 2>,
                                 Sched<[WriteVFSqrtV_MX, ReadVFSqrtV_MX, ReadVMask]>;
    }
  }
}

multiclass VPseudoVRCP_V {
  foreach m = MxListF in {
    defvar mx = m.MX;
    defvar WriteVFRecpV_MX = !cast<SchedWrite>("WriteVFRecpV_" # mx);
    defvar ReadVFRecpV_MX = !cast<SchedRead>("ReadVFRecpV_" # mx);

    let VLMul = m.value in {
      def "_V_" # mx : VPseudoUnaryNoMask<m.vrclass, m.vrclass>,
                         Sched<[WriteVFRecpV_MX, ReadVFRecpV_MX, ReadVMask]>;
      def "_V_" # mx # "_TU": VPseudoUnaryNoMaskTU<m.vrclass, m.vrclass>,
                              Sched<[WriteVFRecpV_MX, ReadVFRecpV_MX, ReadVMask]>;
      def "_V_" # mx # "_MASK" : VPseudoUnaryMaskTA<m.vrclass, m.vrclass>,
                                 RISCVMaskedPseudo</*MaskOpIdx*/ 2>,
                                 Sched<[WriteVFRecpV_MX, ReadVFRecpV_MX, ReadVMask]>;
    }
  }
}

multiclass PseudoVEXT_VF2 {
  defvar constraints = "@earlyclobber $rd";
  foreach m = MxListVF2 in
  {
    defvar mx = m.MX;
    defvar WriteVExtV_MX = !cast<SchedWrite>("WriteVExtV_" # mx);
    defvar ReadVExtV_MX = !cast<SchedRead>("ReadVExtV_" # mx);

    let VLMul = m.value in {
      def "_" # mx : VPseudoUnaryNoMask<m.vrclass, m.f2vrclass, constraints>,
                     Sched<[WriteVExtV_MX, ReadVExtV_MX, ReadVMask]>;
      def "_" # mx # "_TU": VPseudoUnaryNoMaskTU<m.vrclass, m.f2vrclass, constraints>,
                            Sched<[WriteVExtV_MX, ReadVExtV_MX, ReadVMask]>;
      def "_" # mx # "_MASK" :
        VPseudoUnaryMaskTA<m.vrclass, m.f2vrclass, constraints>,
        RISCVMaskedPseudo</*MaskOpIdx*/ 2>,
        Sched<[WriteVExtV_MX, ReadVExtV_MX, ReadVMask]>;
    }
  }
}

multiclass PseudoVEXT_VF4 {
  defvar constraints = "@earlyclobber $rd";
  foreach m = MxListVF4 in
  {
    defvar mx = m.MX;
    defvar WriteVExtV_MX = !cast<SchedWrite>("WriteVExtV_" # mx);
    defvar ReadVExtV_MX = !cast<SchedRead>("ReadVExtV_" # mx);

    let VLMul = m.value in {
      def "_" # mx : VPseudoUnaryNoMask<m.vrclass, m.f4vrclass, constraints>,
                     Sched<[WriteVExtV_MX, ReadVExtV_MX, ReadVMask]>;
      def "_" # mx # "_TU": VPseudoUnaryNoMaskTU<m.vrclass, m.f4vrclass, constraints>,
                            Sched<[WriteVExtV_MX, ReadVExtV_MX, ReadVMask]>;
      def "_" # mx # "_MASK" :
        VPseudoUnaryMaskTA<m.vrclass, m.f4vrclass, constraints>,
        RISCVMaskedPseudo</*MaskOpIdx*/ 2>,
        Sched<[WriteVExtV_MX, ReadVExtV_MX, ReadVMask]>;
    }
  }
}

multiclass PseudoVEXT_VF8 {
  defvar constraints = "@earlyclobber $rd";
  foreach m = MxListVF8 in
  {
    defvar mx = m.MX;
    defvar WriteVExtV_MX = !cast<SchedWrite>("WriteVExtV_" # mx);
    defvar ReadVExtV_MX = !cast<SchedRead>("ReadVExtV_" # mx);

    let VLMul = m.value in {
      def "_" # mx : VPseudoUnaryNoMask<m.vrclass, m.f8vrclass, constraints>,
                     Sched<[WriteVExtV_MX, ReadVExtV_MX, ReadVMask]>;
      def "_" # mx # "_TU": VPseudoUnaryNoMaskTU<m.vrclass, m.f8vrclass, constraints>,
                            Sched<[WriteVExtV_MX, ReadVExtV_MX, ReadVMask]>;
      def "_" # mx # "_MASK" :
        VPseudoUnaryMaskTA<m.vrclass, m.f8vrclass, constraints>,
        RISCVMaskedPseudo</*MaskOpIdx*/ 2>,
        Sched<[WriteVExtV_MX, ReadVExtV_MX, ReadVMask]>;
    }
  }
}

// The destination EEW is 1 since "For the purposes of register group overlap
// constraints, mask elements have EEW=1."
// The source EEW is 8, 16, 32, or 64.
// When the destination EEW is different from source EEW, we need to use
// @earlyclobber to avoid the overlap between destination and source registers.
// We don't need @earlyclobber for LMUL<=1 since that matches this overlap
// exception from the spec
// "The destination EEW is smaller than the source EEW and the overlap is in the
//  lowest-numbered part of the source register group".
// With LMUL<=1 the source and dest occupy a single register so any overlap
// is in the lowest-numbered part.
multiclass VPseudoBinaryM_VV<list<LMULInfo> mxlist = MxList> {
  foreach m = mxlist in
    defm _VV : VPseudoBinaryM<VR, m.vrclass, m.vrclass, m,
                              !if(!ge(m.octuple, 16), "@earlyclobber $rd", "")>;
}

multiclass VPseudoBinaryM_VV_LMUL<LMULInfo m> {
  defm _VV : VPseudoBinaryM<VR, m.vrclass, m.vrclass, m,
                            !if(!ge(m.octuple, 16), "@earlyclobber $rd", "")>;
}

multiclass VPseudoBinaryM_VX<LMULInfo m> {
  defm "_VX" :
    VPseudoBinaryM<VR, m.vrclass, GPR, m,
                   !if(!ge(m.octuple, 16), "@earlyclobber $rd", "")>;
}

multiclass VPseudoBinaryM_VF<LMULInfo m, FPR_Info f> {
  defm "_V" # f.FX :
    VPseudoBinaryM<VR, m.vrclass, f.fprclass, m,
                   !if(!ge(m.octuple, 16), "@earlyclobber $rd", "")>;
}

multiclass VPseudoBinaryM_VI<LMULInfo m> {
  defm _VI : VPseudoBinaryM<VR, m.vrclass, simm5, m,
                            !if(!ge(m.octuple, 16), "@earlyclobber $rd", "")>;
}

multiclass VPseudoVGTR_VV_VX_VI<Operand ImmType = simm5, string Constraint = ""> {
  defm "" : VPseudoBinaryV_VV<Constraint>,
            Sched<[WriteVGatherV, ReadVGatherV, ReadVGatherV, ReadVMask]>;
  defm "" : VPseudoBinaryV_VX<Constraint>,
            Sched<[WriteVGatherX, ReadVGatherV, ReadVGatherX, ReadVMask]>;
  defm "" : VPseudoBinaryV_VI<ImmType, Constraint>,
            Sched<[WriteVGatherI, ReadVGatherV, ReadVMask]>;
}

multiclass VPseudoVSALU_VV_VX_VI<Operand ImmType = simm5, string Constraint = ""> {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVSALUV_MX = !cast<SchedWrite>("WriteVSALUV_" # mx);
    defvar WriteVSALUX_MX = !cast<SchedWrite>("WriteVSALUX_" # mx);
    defvar WriteVSALUI_MX = !cast<SchedWrite>("WriteVSALUI_" # mx);
    defvar ReadVSALUV_MX = !cast<SchedRead>("ReadVSALUV_" # mx);
    defvar ReadVSALUX_MX = !cast<SchedRead>("ReadVSALUX_" # mx);

    defm "" : VPseudoBinaryV_VV_LMUL<m, Constraint>,
              Sched<[WriteVSALUV_MX, ReadVSALUV_MX, ReadVSALUV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_VX_LMUL<m, Constraint>,
              Sched<[WriteVSALUX_MX, ReadVSALUV_MX, ReadVSALUX_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_VI_LMUL<ImmType, m, Constraint>,
              Sched<[WriteVSALUI_MX, ReadVSALUV_MX, ReadVMask]>;
  }
}


multiclass VPseudoVSHT_VV_VX_VI<Operand ImmType = simm5, string Constraint = ""> {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVShiftV_MX = !cast<SchedWrite>("WriteVShiftV_" # mx);
    defvar WriteVShiftX_MX = !cast<SchedWrite>("WriteVShiftX_" # mx);
    defvar WriteVShiftI_MX = !cast<SchedWrite>("WriteVShiftI_" # mx);
    defvar ReadVShiftV_MX = !cast<SchedRead>("ReadVShiftV_" # mx);
    defvar ReadVShiftX_MX = !cast<SchedRead>("ReadVShiftX_" # mx);

    defm "" : VPseudoBinaryV_VV_LMUL<m, Constraint>,
              Sched<[WriteVShiftV_MX, ReadVShiftV_MX, ReadVShiftV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_VX_LMUL<m, Constraint>,
              Sched<[WriteVShiftX_MX, ReadVShiftV_MX, ReadVShiftX_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_VI_LMUL<ImmType, m, Constraint>,
              Sched<[WriteVShiftI_MX, ReadVShiftV_MX, ReadVMask]>;
  }
}

multiclass VPseudoVSSHT_VV_VX_VI<Operand ImmType = simm5, string Constraint = ""> {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVSShiftV_MX = !cast<SchedWrite>("WriteVSShiftV_" # mx);
    defvar WriteVSShiftX_MX = !cast<SchedWrite>("WriteVSShiftX_" # mx);
    defvar WriteVSShiftI_MX = !cast<SchedWrite>("WriteVSShiftI_" # mx);
    defvar ReadVSShiftV_MX = !cast<SchedRead>("ReadVSShiftV_" # mx);
    defvar ReadVSShiftX_MX = !cast<SchedRead>("ReadVSShiftX_" # mx);

    defm "" : VPseudoBinaryV_VV_LMUL<m, Constraint>,
              Sched<[WriteVSShiftV_MX, ReadVSShiftV_MX, ReadVSShiftV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_VX_LMUL<m, Constraint>,
              Sched<[WriteVSShiftX_MX, ReadVSShiftV_MX, ReadVSShiftX_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_VI_LMUL<ImmType, m, Constraint>,
              Sched<[WriteVSShiftI_MX, ReadVSShiftV_MX, ReadVMask]>;
  }
}

multiclass VPseudoVALU_VV_VX_VI<Operand ImmType = simm5, string Constraint = ""> {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVIALUV_MX = !cast<SchedWrite>("WriteVIALUV_" # mx);
    defvar WriteVIALUX_MX = !cast<SchedWrite>("WriteVIALUX_" # mx);
    defvar WriteVIALUI_MX = !cast<SchedWrite>("WriteVIALUI_" # mx);
    defvar ReadVIALUV_MX = !cast<SchedRead>("ReadVIALUV_" # mx);
    defvar ReadVIALUX_MX = !cast<SchedRead>("ReadVIALUX_" # mx);

    defm "" : VPseudoBinaryV_VV_LMUL<m, Constraint>,
            Sched<[WriteVIALUV_MX, ReadVIALUV_MX, ReadVIALUV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_VX_LMUL<m, Constraint>,
            Sched<[WriteVIALUX_MX, ReadVIALUV_MX, ReadVIALUX_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_VI_LMUL<ImmType, m, Constraint>,
            Sched<[WriteVIALUI_MX, ReadVIALUV_MX, ReadVMask]>;
  }
}

multiclass VPseudoVSALU_VV_VX {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVSALUV_MX = !cast<SchedWrite>("WriteVSALUV_" # mx);
    defvar WriteVSALUX_MX = !cast<SchedWrite>("WriteVSALUX_" # mx);
    defvar ReadVSALUV_MX = !cast<SchedRead>("ReadVSALUV_" # mx);
    defvar ReadVSALUX_MX = !cast<SchedRead>("ReadVSALUX_" # mx);

    defm "" : VPseudoBinaryV_VV_LMUL<m>,
              Sched<[WriteVSALUV_MX, ReadVSALUV_MX, ReadVSALUV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_VX_LMUL<m>,
              Sched<[WriteVSALUX_MX, ReadVSALUV_MX, ReadVSALUX_MX, ReadVMask]>;
  }
}

multiclass VPseudoVSMUL_VV_VX {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVSMulV_MX = !cast<SchedWrite>("WriteVSMulV_" # mx);
    defvar WriteVSMulX_MX = !cast<SchedWrite>("WriteVSMulX_" # mx);
    defvar ReadVSMulV_MX = !cast<SchedRead>("ReadVSMulV_" # mx);
    defvar ReadVSMulX_MX = !cast<SchedRead>("ReadVSMulX_" # mx);

    defm "" : VPseudoBinaryV_VV_LMUL<m>,
              Sched<[WriteVSMulV_MX, ReadVSMulV_MX, ReadVSMulV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_VX_LMUL<m>,
              Sched<[WriteVSMulX_MX, ReadVSMulV_MX, ReadVSMulX_MX, ReadVMask]>;
  }
}

multiclass VPseudoVAALU_VV_VX {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVAALUV_MX = !cast<SchedWrite>("WriteVAALUV_" # mx);
    defvar WriteVAALUX_MX = !cast<SchedWrite>("WriteVAALUX_" # mx);
    defvar ReadVAALUV_MX = !cast<SchedRead>("ReadVAALUV_" # mx);
    defvar ReadVAALUX_MX = !cast<SchedRead>("ReadVAALUX_" # mx);

    defm "" : VPseudoBinaryV_VV_LMUL<m>,
              Sched<[WriteVAALUV_MX, ReadVAALUV_MX, ReadVAALUV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_VX_LMUL<m>,
              Sched<[WriteVAALUX_MX, ReadVAALUV_MX, ReadVAALUX_MX, ReadVMask]>;
  }
}

multiclass VPseudoVMINMAX_VV_VX {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVICmpV_MX = !cast<SchedWrite>("WriteVICmpV_" # mx);
    defvar WriteVICmpX_MX = !cast<SchedWrite>("WriteVICmpX_" # mx);
    defvar ReadVICmpV_MX = !cast<SchedRead>("ReadVICmpV_" # mx);
    defvar ReadVICmpX_MX = !cast<SchedRead>("ReadVICmpX_" # mx);

    defm "" : VPseudoBinaryV_VV_LMUL<m>,
              Sched<[WriteVICmpV_MX, ReadVICmpV_MX, ReadVICmpV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_VX_LMUL<m>,
              Sched<[WriteVICmpX_MX, ReadVICmpV_MX, ReadVICmpX_MX, ReadVMask]>;
  }
}

multiclass VPseudoVMUL_VV_VX {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVIMulV_MX = !cast<SchedWrite>("WriteVIMulV_" # mx);
    defvar WriteVIMulX_MX = !cast<SchedWrite>("WriteVIMulX_" # mx);
    defvar ReadVIMulV_MX = !cast<SchedRead>("ReadVIMulV_" # mx);
    defvar ReadVIMulX_MX = !cast<SchedRead>("ReadVIMulX_" # mx);

    defm "" : VPseudoBinaryV_VV_LMUL<m>,
              Sched<[WriteVIMulV_MX, ReadVIMulV_MX, ReadVIMulV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_VX_LMUL<m>,
              Sched<[WriteVIMulX_MX, ReadVIMulV_MX, ReadVIMulX_MX, ReadVMask]>;
  }
}

multiclass VPseudoVDIV_VV_VX {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVIDivV_MX = !cast<SchedWrite>("WriteVIDivV_" # mx);
    defvar WriteVIDivX_MX = !cast<SchedWrite>("WriteVIDivX_" # mx);
    defvar ReadVIDivV_MX = !cast<SchedRead>("ReadVIDivV_" # mx);
    defvar ReadVIDivX_MX = !cast<SchedRead>("ReadVIDivX_" # mx);

    defm "" : VPseudoBinaryV_VV_LMUL<m>,
              Sched<[WriteVIDivV_MX, ReadVIDivV_MX, ReadVIDivV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_VX_LMUL<m>,
              Sched<[WriteVIDivX_MX, ReadVIDivV_MX, ReadVIDivX_MX, ReadVMask]>;
  }
}

multiclass VPseudoVFMUL_VV_VF {
  foreach m = MxListF in {
    defvar mx = m.MX;
    defvar WriteVFMulV_MX = !cast<SchedWrite>("WriteVFMulV_" # mx);
    defvar ReadVFMulV_MX = !cast<SchedRead>("ReadVFMulV_" # mx);

    defm "" : VPseudoBinaryFV_VV<m>,
              Sched<[WriteVFMulV_MX, ReadVFMulV_MX, ReadVFMulV_MX, ReadVMask]>;
  }

  foreach f = FPList in {
    foreach m = f.MxList in {
      defvar mx = m.MX;
      defvar WriteVFMulF_MX = !cast<SchedWrite>("WriteVFMulF_" # mx);
      defvar ReadVFMulV_MX = !cast<SchedRead>("ReadVFMulV_" # mx);
      defvar ReadVFMulF_MX = !cast<SchedRead>("ReadVFMulF_" # mx);

      defm "" : VPseudoBinaryV_VF<m, f>,
                Sched<[WriteVFMulF_MX, ReadVFMulV_MX, ReadVFMulF_MX, ReadVMask]>;
    }
  }
}

multiclass VPseudoVFDIV_VV_VF {
  foreach m = MxListF in {
    defvar mx = m.MX;
    defvar WriteVFDivV_MX = !cast<SchedWrite>("WriteVFDivV_" # mx);
    defvar ReadVFDivV_MX = !cast<SchedRead>("ReadVFDivV_" # mx);

    defm "" : VPseudoBinaryFV_VV<m>,
              Sched<[WriteVFDivV_MX, ReadVFDivV_MX, ReadVFDivV_MX, ReadVMask]>;
  }

  foreach f = FPList in {
    foreach m = f.MxList in {
      defvar mx = m.MX;
      defvar WriteVFDivF_MX = !cast<SchedWrite>("WriteVFDivF_" # mx);
      defvar ReadVFDivV_MX = !cast<SchedRead>("ReadVFDivV_" # mx);
      defvar ReadVFDivF_MX = !cast<SchedRead>("ReadVFDivF_" # mx);

      defm "" : VPseudoBinaryV_VF<m, f>,
                Sched<[WriteVFDivF_MX, ReadVFDivV_MX, ReadVFDivF_MX, ReadVMask]>;
    }
  }
}

multiclass VPseudoVFRDIV_VF {
  foreach f = FPList in {
    foreach m = f.MxList in {
      defvar mx = m.MX;
      defvar WriteVFDivF_MX = !cast<SchedWrite>("WriteVFDivF_" # mx);
      defvar ReadVFDivV_MX = !cast<SchedRead>("ReadVFDivV_" # mx);
      defvar ReadVFDivF_MX = !cast<SchedRead>("ReadVFDivF_" # mx);
        defm "" : VPseudoBinaryV_VF<m, f>,
                  Sched<[WriteVFDivF_MX, ReadVFDivV_MX, ReadVFDivF_MX, ReadVMask]>;
    }
  }
}

multiclass VPseudoVALU_VV_VX {
 foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVIALUV_MX = !cast<SchedWrite>("WriteVIALUV_" # mx);
    defvar WriteVIALUX_MX = !cast<SchedWrite>("WriteVIALUV_" # mx);
    defvar ReadVIALUV_MX = !cast<SchedRead>("ReadVIALUV_" # mx);
    defvar ReadVIALUX_MX = !cast<SchedRead>("ReadVIALUX_" # mx);

    defm "" : VPseudoBinaryV_VV_LMUL<m>,
            Sched<[WriteVIALUV_MX, ReadVIALUV_MX, ReadVIALUV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_VX_LMUL<m>,
            Sched<[WriteVIALUX_MX, ReadVIALUV_MX, ReadVIALUX_MX, ReadVMask]>;
  }
}

multiclass VPseudoVSGNJ_VV_VF {
  foreach m = MxListF in {
    defvar mx = m.MX;
    defvar WriteVFSgnjV_MX = !cast<SchedWrite>("WriteVFSgnjV_" # mx);
    defvar ReadVFSgnjV_MX = !cast<SchedRead>("ReadVFSgnjV_" # mx);

    defm "" : VPseudoBinaryFV_VV<m>,
              Sched<[WriteVFSgnjV_MX, ReadVFSgnjV_MX, ReadVFSgnjV_MX, ReadVMask]>;
  }

  foreach f = FPList in {
    foreach m = f.MxList in {
      defvar mx = m.MX;
      defvar WriteVFSgnjF_MX = !cast<SchedWrite>("WriteVFSgnjF_" # mx);
      defvar ReadVFSgnjV_MX = !cast<SchedRead>("ReadVFSgnjV_" # mx);
      defvar ReadVFSgnjF_MX = !cast<SchedRead>("ReadVFSgnjF_" # mx);

      defm "" : VPseudoBinaryV_VF<m, f>,
                Sched<[WriteVFSgnjF_MX, ReadVFSgnjV_MX, ReadVFSgnjF_MX, ReadVMask]>;
    }
  }
}

multiclass VPseudoVMAX_VV_VF {
  foreach m = MxListF in {
    defvar mx = m.MX;
    defvar WriteVFCmpV_MX = !cast<SchedWrite>("WriteVFCmpV_" # mx);
    defvar ReadVFCmpV_MX = !cast<SchedRead>("ReadVFCmpV_" # mx);

    defm "" : VPseudoBinaryFV_VV<m>,
              Sched<[WriteVFCmpV_MX, ReadVFCmpV_MX, ReadVFCmpV_MX, ReadVMask]>;
  }

  foreach f = FPList in {
    foreach m = f.MxList in {
      defvar mx = m.MX;
      defvar WriteVFCmpF_MX = !cast<SchedWrite>("WriteVFCmpF_" # mx);
      defvar ReadVFCmpV_MX = !cast<SchedRead>("ReadVFCmpV_" # mx);
      defvar ReadVFCmpF_MX = !cast<SchedRead>("ReadVFCmpF_" # mx);

      defm "" : VPseudoBinaryV_VF<m, f>,
                Sched<[WriteVFCmpF_MX, ReadVFCmpV_MX, ReadVFCmpF_MX, ReadVMask]>;
    }
  }
}

multiclass VPseudoVALU_VV_VF {
  foreach m = MxListF in {
    defvar mx = m.MX;
    defvar WriteVFALUV_MX = !cast<SchedWrite>("WriteVFALUV_" # mx);
    defvar ReadVFALUV_MX = !cast<SchedRead>("ReadVFALUV_" # mx);

    defm "" : VPseudoBinaryFV_VV<m>,
              Sched<[WriteVFALUV_MX, ReadVFALUV_MX, ReadVFALUV_MX, ReadVMask]>;
  }

  foreach f = FPList in {
    foreach m = f.MxList in {
      defvar mx = m.MX;
      defvar WriteVFALUF_MX = !cast<SchedWrite>("WriteVFALUF_" # mx);
      defvar ReadVFALUV_MX = !cast<SchedRead>("ReadVFALUV_" # mx);
      defvar ReadVFALUF_MX = !cast<SchedRead>("ReadVFALUF_" # mx);
      defm "" : VPseudoBinaryV_VF<m, f>,
                Sched<[WriteVFALUF_MX, ReadVFALUV_MX, ReadVFALUF_MX, ReadVMask]>;
    }
  }
}

multiclass VPseudoVALU_VF {
  foreach f = FPList in {
    foreach m = f.MxList in {
      defvar mx = m.MX;
      defvar WriteVFALUF_MX = !cast<SchedWrite>("WriteVFALUF_" # mx);
      defvar ReadVFALUV_MX = !cast<SchedRead>("ReadVFALUV_" # mx);
      defvar ReadVFALUF_MX = !cast<SchedRead>("ReadVFALUF_" # mx);

      defm "" : VPseudoBinaryV_VF<m, f>,
                Sched<[WriteVFALUF_MX, ReadVFALUV_MX, ReadVFALUF_MX, ReadVMask]>;
    }
  }
}

multiclass VPseudoVALU_VX_VI<Operand ImmType = simm5> {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVIALUX_MX = !cast<SchedWrite>("WriteVIALUX_" # mx);
    defvar WriteVIALUI_MX = !cast<SchedWrite>("WriteVIALUI_" # mx);
    defvar ReadVIALUV_MX = !cast<SchedRead>("ReadVIALUV_" # mx);
    defvar ReadVIALUX_MX = !cast<SchedRead>("ReadVIALUX_" # mx);

    defm "" : VPseudoBinaryV_VX_LMUL<m>,
            Sched<[WriteVIALUX_MX, ReadVIALUV_MX, ReadVIALUX_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_VI_LMUL<ImmType, m>,
            Sched<[WriteVIALUI_MX, ReadVIALUV_MX, ReadVMask]>;
  }
}

multiclass VPseudoVWALU_VV_VX {
  foreach m = MxListW in {
    defvar mx = m.MX;
    defvar WriteVIWALUV_MX = !cast<SchedWrite>("WriteVIWALUV_" # mx);
    defvar WriteVIWALUX_MX = !cast<SchedWrite>("WriteVIWALUX_" # mx);
    defvar ReadVIWALUV_MX = !cast<SchedRead>("ReadVIWALUV_" # mx);
    defvar ReadVIWALUX_MX = !cast<SchedRead>("ReadVIWALUX_" # mx);

    defm "" : VPseudoBinaryW_VV_LMUL<m>,
            Sched<[WriteVIWALUV_MX, ReadVIWALUV_MX, ReadVIWALUV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryW_VX_LMUL<m>,
            Sched<[WriteVIWALUX_MX, ReadVIWALUV_MX, ReadVIWALUX_MX, ReadVMask]>;
  }
}

multiclass VPseudoVWMUL_VV_VX {
  foreach m = MxListW in {
    defvar mx = m.MX;
    defvar WriteVIWMulV_MX = !cast<SchedWrite>("WriteVIWMulV_" # mx);
    defvar WriteVIWMulX_MX = !cast<SchedWrite>("WriteVIWMulX_" # mx);
    defvar ReadVIWMulV_MX = !cast<SchedRead>("ReadVIWMulV_" # mx);
    defvar ReadVIWMulX_MX = !cast<SchedRead>("ReadVIWMulX_" # mx);

    defm "" : VPseudoBinaryW_VV_LMUL<m>,
              Sched<[WriteVIWMulV_MX, ReadVIWMulV_MX, ReadVIWMulV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryW_VX_LMUL<m>,
              Sched<[WriteVIWMulX_MX, ReadVIWMulV_MX, ReadVIWMulX_MX, ReadVMask]>;
  }
}

multiclass VPseudoVWMUL_VV_VF {
  foreach m = MxListFW in {
    defvar mx = m.MX;
    defvar WriteVFWMulV_MX = !cast<SchedWrite>("WriteVFWMulV_" # mx);
    defvar ReadVFWMulV_MX = !cast<SchedRead>("ReadVFWMulV_" # mx);

    defm "" : VPseudoBinaryW_VV_LMUL<m>,
              Sched<[WriteVFWMulV_MX, ReadVFWMulV_MX, ReadVFWMulV_MX, ReadVMask]>;
  }

  foreach f = FPListW in {
    foreach m = f.MxListFW in {
      defvar mx = m.MX;
      defvar WriteVFWMulF_MX = !cast<SchedWrite>("WriteVFWMulF_" # mx);
      defvar ReadVFWMulV_MX = !cast<SchedRead>("ReadVFWMulV_" # mx);
      defvar ReadVFWMulF_MX = !cast<SchedRead>("ReadVFWMulF_" # mx);

      defm "" : VPseudoBinaryW_VF<m, f>,
                Sched<[WriteVFWMulF_MX, ReadVFWMulV_MX, ReadVFWMulF_MX, ReadVMask]>;
    }
  }
}

multiclass VPseudoVWALU_WV_WX {
  foreach m = MxListW in {
    defvar mx = m.MX;
    defvar WriteVIWALUV_MX = !cast<SchedWrite>("WriteVIWALUV_" # mx);
    defvar WriteVIWALUX_MX = !cast<SchedWrite>("WriteVIWALUX_" # mx);
    defvar ReadVIWALUV_MX = !cast<SchedRead>("ReadVIWALUV_" # mx);
    defvar ReadVIWALUX_MX = !cast<SchedRead>("ReadVIWALUX_" # mx);

    defm "" : VPseudoBinaryW_WV_LMUL<m>,
              Sched<[WriteVIWALUV_MX, ReadVIWALUV_MX, ReadVIWALUV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryW_WX_LMUL<m>,
              Sched<[WriteVIWALUX_MX, ReadVIWALUV_MX, ReadVIWALUX_MX, ReadVMask]>;
  }
}

multiclass VPseudoVFWALU_VV_VF {
  foreach m = MxListFW in {
    defvar mx = m.MX;
    defvar WriteVFWALUV_MX = !cast<SchedWrite>("WriteVFWALUV_" # mx);
    defvar ReadVFWALUV_MX = !cast<SchedRead>("ReadVFWALUV_" # mx);

    defm "" : VPseudoBinaryW_VV_LMUL<m>,
              Sched<[WriteVFWALUV_MX, ReadVFWALUV_MX, ReadVFWALUV_MX, ReadVMask]>;
  }

  foreach f = FPListW in {
    foreach m = f.MxListFW in {
      defvar mx = m.MX;
      defvar WriteVFWALUF_MX = !cast<SchedWrite>("WriteVFWALUF_" # mx);
      defvar ReadVFWALUV_MX = !cast<SchedRead>("ReadVFWALUV_" # mx);
      defvar ReadVFWALUF_MX = !cast<SchedRead>("ReadVFWALUF_" # mx);

      defm "" : VPseudoBinaryW_VF<m, f>,
                Sched<[WriteVFWALUF_MX, ReadVFWALUV_MX, ReadVFWALUF_MX, ReadVMask]>;
    }
  }
}

multiclass VPseudoVFWALU_WV_WF {
  foreach m = MxListFW in {
    defvar mx = m.MX;
    defvar WriteVFWALUV_MX = !cast<SchedWrite>("WriteVFWALUV_" # mx);
    defvar ReadVFWALUV_MX = !cast<SchedRead>("ReadVFWALUV_" # mx);

    defm "" : VPseudoBinaryW_WV_LMUL<m>,
              Sched<[WriteVFWALUV_MX, ReadVFWALUV_MX, ReadVFWALUV_MX, ReadVMask]>;
  }
  foreach f = FPListW in {
    foreach m = f.MxListFW in {
      defvar mx = m.MX;
      defvar WriteVFWALUF_MX = !cast<SchedWrite>("WriteVFWALUF_" # mx);
      defvar ReadVFWALUV_MX = !cast<SchedRead>("ReadVFWALUV_" # mx);
      defvar ReadVFWALUF_MX = !cast<SchedRead>("ReadVFWALUF_" # mx);

      defm "" : VPseudoBinaryW_WF<m, f>,
                Sched<[WriteVFWALUF_MX, ReadVFWALUV_MX, ReadVFWALUF_MX, ReadVMask]>;
    }
  }
}

multiclass VPseudoVMRG_VM_XM_IM {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVIMergeV_MX = !cast<SchedWrite>("WriteVIMergeV_" # mx);
    defvar WriteVIMergeX_MX = !cast<SchedWrite>("WriteVIMergeX_" # mx);
    defvar WriteVIMergeI_MX = !cast<SchedWrite>("WriteVIMergeI_" # mx);
    defvar ReadVIMergeV_MX = !cast<SchedRead>("ReadVIMergeV_" # mx);
    defvar ReadVIMergeX_MX = !cast<SchedRead>("ReadVIMergeX_" # mx);

    defm "" : VPseudoBinaryV_VM<m>,
              Sched<[WriteVIMergeV_MX, ReadVIMergeV_MX, ReadVIMergeV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_XM<m>,
              Sched<[WriteVIMergeX_MX, ReadVIMergeV_MX, ReadVIMergeX_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_IM<m>,
              Sched<[WriteVIMergeI_MX, ReadVIMergeV_MX, ReadVMask]>;
    // Tied versions to allow codegen control over the tail elements
    defm "" : VPseudoTiedBinaryV_VM<m>,
              Sched<[WriteVIMergeV_MX, ReadVIMergeV_MX, ReadVIMergeV_MX, ReadVMask]>;
    defm "" : VPseudoTiedBinaryV_XM<m>,
              Sched<[WriteVIMergeX_MX, ReadVIMergeV_MX, ReadVIMergeX_MX, ReadVMask]>;
    defm "" : VPseudoTiedBinaryV_IM<m>,
              Sched<[WriteVIMergeI_MX, ReadVIMergeV_MX, ReadVMask]>;
  }
}

multiclass VPseudoVCALU_VM_XM_IM {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVICALUV_MX = !cast<SchedWrite>("WriteVICALUV_" # mx);
    defvar WriteVICALUX_MX = !cast<SchedWrite>("WriteVICALUX_" # mx);
    defvar WriteVICALUI_MX = !cast<SchedWrite>("WriteVICALUI_" # mx);
    defvar ReadVICALUV_MX = !cast<SchedRead>("ReadVICALUV_" # mx);
    defvar ReadVICALUX_MX = !cast<SchedRead>("ReadVICALUX_" # mx);

    defm "" : VPseudoBinaryV_VM<m>,
              Sched<[WriteVICALUV_MX, ReadVICALUV_MX, ReadVICALUV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_XM<m>,
              Sched<[WriteVICALUX_MX, ReadVICALUV_MX, ReadVICALUX_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_IM<m>,
              Sched<[WriteVICALUI_MX, ReadVICALUV_MX, ReadVMask]>;
    // Tied versions to allow codegen control over the tail elements
    defm "" : VPseudoTiedBinaryV_VM<m>,
              Sched<[WriteVICALUV_MX, ReadVICALUV_MX, ReadVICALUV_MX, ReadVMask]>;
    defm "" : VPseudoTiedBinaryV_XM<m>,
              Sched<[WriteVICALUX_MX, ReadVICALUV_MX, ReadVICALUX_MX, ReadVMask]>;
    defm "" : VPseudoTiedBinaryV_IM<m>,
              Sched<[WriteVICALUI_MX, ReadVICALUV_MX, ReadVMask]>;
  }
}

multiclass VPseudoVCALU_VM_XM {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVICALUV_MX = !cast<SchedWrite>("WriteVICALUV_" # mx);
    defvar WriteVICALUX_MX = !cast<SchedWrite>("WriteVICALUX_" # mx);
    defvar ReadVICALUV_MX = !cast<SchedRead>("ReadVICALUV_" # mx);
    defvar ReadVICALUX_MX = !cast<SchedRead>("ReadVICALUX_" # mx);

    defm "" : VPseudoBinaryV_VM<m>,
              Sched<[WriteVICALUV_MX, ReadVICALUV_MX, ReadVICALUV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_XM<m>,
              Sched<[WriteVICALUX_MX, ReadVICALUV_MX, ReadVICALUX_MX, ReadVMask]>;
    // Tied versions to allow codegen control over the tail elements
    defm "" : VPseudoTiedBinaryV_VM<m>,
              Sched<[WriteVICALUV_MX, ReadVICALUV_MX, ReadVICALUV_MX, ReadVMask]>;
    defm "" : VPseudoTiedBinaryV_XM<m>,
              Sched<[WriteVICALUX_MX, ReadVICALUV_MX, ReadVICALUX_MX, ReadVMask]>;
  }
}

multiclass VPseudoVCALUM_VM_XM_IM<string Constraint> {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVICALUV_MX = !cast<SchedWrite>("WriteVICALUV_" # mx);
    defvar WriteVICALUX_MX = !cast<SchedWrite>("WriteVICALUX_" # mx);
    defvar WriteVICALUI_MX = !cast<SchedWrite>("WriteVICALUI_" # mx);
    defvar ReadVICALUV_MX = !cast<SchedRead>("ReadVICALUV_" # mx);
    defvar ReadVICALUX_MX = !cast<SchedRead>("ReadVICALUX_" # mx);

    defm "" : VPseudoBinaryV_VM<m, /*CarryOut=*/1, /*CarryIn=*/1, Constraint>,
              Sched<[WriteVICALUV_MX, ReadVICALUV_MX, ReadVICALUV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_XM<m, /*CarryOut=*/1, /*CarryIn=*/1, Constraint>,
              Sched<[WriteVICALUX_MX, ReadVICALUV_MX, ReadVICALUX_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_IM<m, /*CarryOut=*/1, /*CarryIn=*/1, Constraint>,
              Sched<[WriteVICALUI_MX, ReadVICALUV_MX, ReadVMask]>;
  }
}

multiclass VPseudoVCALUM_VM_XM<string Constraint> {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVICALUV_MX = !cast<SchedWrite>("WriteVICALUV_" # mx);
    defvar WriteVICALUX_MX = !cast<SchedWrite>("WriteVICALUX_" # mx);
    defvar ReadVICALUV_MX = !cast<SchedRead>("ReadVICALUV_" # mx);
    defvar ReadVICALUX_MX = !cast<SchedRead>("ReadVICALUX_" # mx);

    defm "" : VPseudoBinaryV_VM<m, /*CarryOut=*/1, /*CarryIn=*/1, Constraint>,
              Sched<[WriteVICALUV_MX, ReadVICALUV_MX, ReadVICALUV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_XM<m, /*CarryOut=*/1, /*CarryIn=*/1, Constraint>,
              Sched<[WriteVICALUX_MX, ReadVICALUV_MX, ReadVICALUX_MX, ReadVMask]>;
  }
}

multiclass VPseudoVCALUM_V_X_I<string Constraint> {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVICALUV_MX = !cast<SchedWrite>("WriteVICALUV_" # mx);
    defvar WriteVICALUX_MX = !cast<SchedWrite>("WriteVICALUX_" # mx);
    defvar WriteVICALUI_MX = !cast<SchedWrite>("WriteVICALUI_" # mx);
    defvar ReadVICALUV_MX = !cast<SchedRead>("ReadVICALUV_" # mx);
    defvar ReadVICALUX_MX = !cast<SchedRead>("ReadVICALUX_" # mx);

    defm "" : VPseudoBinaryV_VM<m, /*CarryOut=*/1, /*CarryIn=*/0, Constraint>,
              Sched<[WriteVICALUV_MX, ReadVICALUV_MX, ReadVICALUV_MX]>;
    defm "" : VPseudoBinaryV_XM<m, /*CarryOut=*/1, /*CarryIn=*/0, Constraint>,
              Sched<[WriteVICALUX_MX, ReadVICALUV_MX, ReadVICALUX_MX]>;
    defm "" : VPseudoBinaryV_IM<m, /*CarryOut=*/1, /*CarryIn=*/0, Constraint>,
              Sched<[WriteVICALUI_MX, ReadVICALUV_MX]>;
  }
}

multiclass VPseudoVCALUM_V_X<string Constraint> {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVICALUV_MX = !cast<SchedWrite>("WriteVICALUV_" # mx);
    defvar WriteVICALUX_MX = !cast<SchedWrite>("WriteVICALUX_" # mx);
    defvar ReadVICALUV_MX = !cast<SchedRead>("ReadVICALUV_" # mx);
    defvar ReadVICALUX_MX = !cast<SchedRead>("ReadVICALUX_" # mx);

    defm "" : VPseudoBinaryV_VM<m, /*CarryOut=*/1, /*CarryIn=*/0, Constraint>,
              Sched<[WriteVICALUV_MX, ReadVICALUV_MX, ReadVICALUV_MX]>;
    defm "" : VPseudoBinaryV_XM<m, /*CarryOut=*/1, /*CarryIn=*/0, Constraint>,
              Sched<[WriteVICALUX_MX, ReadVICALUV_MX, ReadVICALUX_MX]>;
  }
}

multiclass VPseudoVNCLP_WV_WX_WI {
  foreach m = MxListW in {
    defvar mx = m.MX;
    defvar WriteVNClipV_MX = !cast<SchedWrite>("WriteVNClipV_" # mx);
    defvar WriteVNClipX_MX = !cast<SchedWrite>("WriteVNClipX_" # mx);
    defvar WriteVNClipI_MX = !cast<SchedWrite>("WriteVNClipI_" # mx);
    defvar ReadVNClipV_MX = !cast<SchedRead>("ReadVNClipV_" # mx);
    defvar ReadVNClipX_MX = !cast<SchedRead>("ReadVNClipX_" # mx);

    defm "" : VPseudoBinaryV_WV_LMUL<m>,
              Sched<[WriteVNClipV_MX, ReadVNClipV_MX, ReadVNClipV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_WX_LMUL<m>,
              Sched<[WriteVNClipX_MX, ReadVNClipV_MX, ReadVNClipX_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_WI_LMUL<m>,
              Sched<[WriteVNClipI_MX, ReadVNClipV_MX, ReadVMask]>;
  }
}

multiclass VPseudoVNSHT_WV_WX_WI {
  foreach m = MxListW in {
    defvar mx = m.MX;
    defvar WriteVNShiftV_MX = !cast<SchedWrite>("WriteVNShiftV_" # mx);
    defvar WriteVNShiftX_MX = !cast<SchedWrite>("WriteVNShiftX_" # mx);
    defvar WriteVNShiftI_MX = !cast<SchedWrite>("WriteVNShiftI_" # mx);
    defvar ReadVNShiftV_MX = !cast<SchedRead>("ReadVNShiftV_" # mx);
    defvar ReadVNShiftX_MX = !cast<SchedRead>("ReadVNShiftX_" # mx);

    defm "" : VPseudoBinaryV_WV_LMUL<m>,
              Sched<[WriteVNShiftV_MX, ReadVNShiftV_MX, ReadVNShiftV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_WX_LMUL<m>,
              Sched<[WriteVNShiftX_MX, ReadVNShiftV_MX, ReadVNShiftX_MX, ReadVMask]>;
    defm "" : VPseudoBinaryV_WI_LMUL<m>,
              Sched<[WriteVNShiftI_MX, ReadVNShiftV_MX, ReadVMask]>;
  }
}

multiclass VPseudoTernary<VReg RetClass,
                          RegisterClass Op1Class,
                          DAGOperand Op2Class,
                          LMULInfo MInfo,
                          string Constraint = ""> {
  let VLMul = MInfo.value in {
    def "_" # MInfo.MX : VPseudoTernaryNoMask<RetClass, Op1Class, Op2Class, Constraint>;
    def "_" # MInfo.MX # "_MASK" : VPseudoBinaryMask<RetClass, Op1Class, Op2Class, Constraint>;
  }
}

multiclass VPseudoTernaryNoMaskNoPolicy<VReg RetClass,
                                        RegisterClass Op1Class,
                                        DAGOperand Op2Class,
                                        LMULInfo MInfo,
                                        string Constraint = ""> {
  let VLMul = MInfo.value in {
    def "_" # MInfo.MX : VPseudoTernaryNoMask<RetClass, Op1Class, Op2Class, Constraint>;
    def "_" # MInfo.MX # "_MASK" : VPseudoBinaryMaskPolicy<RetClass, Op1Class, Op2Class,
                                                           Constraint>;
                                   
  }
}

multiclass VPseudoTernaryWithPolicy<VReg RetClass,
                                    RegisterClass Op1Class,
                                    DAGOperand Op2Class,
                                    LMULInfo MInfo,
                                    string Constraint = "",
                                    bit Commutable = 0> {
  let VLMul = MInfo.value in {
    let isCommutable = Commutable in
    def "_" # MInfo.MX : VPseudoTernaryNoMaskWithPolicy<RetClass, Op1Class, Op2Class, Constraint>;
    def "_" # MInfo.MX # "_MASK" : VPseudoBinaryMaskPolicy<RetClass, Op1Class, Op2Class, Constraint>;
  }
}

multiclass VPseudoTernaryV_VV_AAXA<string Constraint = "",
                                   list<LMULInfo> mxlist = MxList> {
  foreach m = mxlist in {
    defm _VV : VPseudoTernaryWithPolicy<m.vrclass, m.vrclass, m.vrclass, m,
                                        Constraint, /*Commutable*/1>;
  }
}

multiclass VPseudoTernaryV_VV_AAXA_LMUL<LMULInfo m, string Constraint = ""> {
  defm _VV : VPseudoTernaryWithPolicy<m.vrclass, m.vrclass, m.vrclass, m,
                                      Constraint, /*Commutable*/1>;
}

multiclass VPseudoVSLDV_VX<string Constraint = ""> {
  foreach m = MxList in
    defm _VX : VPseudoTernaryWithPolicy<m.vrclass, m.vrclass, GPR, m, Constraint>;
}

multiclass VPseudoTernaryV_VX_AAXA<LMULInfo m, string Constraint = ""> {
  defm "_VX" : VPseudoTernaryWithPolicy<m.vrclass, GPR, m.vrclass, m,
                                        Constraint, /*Commutable*/1>;
}

multiclass VPseudoTernaryV_VF_AAXA<LMULInfo m, FPR_Info f, string Constraint = ""> {
  defm "_V" # f.FX : VPseudoTernaryWithPolicy<m.vrclass, f.fprclass,
                                              m.vrclass, m, Constraint,
                                              /*Commutable*/1>;
}

multiclass VPseudoTernaryW_VV<list<LMULInfo> mxlist = MxListW> {
  defvar constraint = "@earlyclobber $rd";
  foreach m = mxlist in
    defm _VV : VPseudoTernaryWithPolicy<m.wvrclass, m.vrclass, m.vrclass, m,
                                        constraint>;
}

multiclass VPseudoTernaryW_VV_LMUL<LMULInfo m> {
  defvar constraint = "@earlyclobber $rd";
  defm _VV : VPseudoTernaryWithPolicy<m.wvrclass, m.vrclass, m.vrclass, m,
                                      constraint>;
}

multiclass VPseudoTernaryW_VX<LMULInfo m> {
  defvar constraint = "@earlyclobber $rd";
  defm "_VX" : VPseudoTernaryWithPolicy<m.wvrclass, GPR, m.vrclass, m,
                                        constraint>;
}

multiclass VPseudoTernaryW_VF<LMULInfo m, FPR_Info f> {
  defvar constraint = "@earlyclobber $rd";
  defm "_V" # f.FX : VPseudoTernaryWithPolicy<m.wvrclass, f.fprclass,
                                              m.vrclass, m, constraint>;
}

multiclass VPseudoVSLDV_VI<Operand ImmType = simm5, string Constraint = ""> {
  foreach m = MxList in
    defm _VI : VPseudoTernaryWithPolicy<m.vrclass, m.vrclass, ImmType, m, Constraint>;
}

multiclass VPseudoVMAC_VV_VX_AAXA<string Constraint = ""> {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVIMulAddV_MX = !cast<SchedWrite>("WriteVIMulAddV_" # mx);
    defvar WriteVIMulAddX_MX = !cast<SchedWrite>("WriteVIMulAddX_" # mx);
    defvar ReadVIMulAddV_MX = !cast<SchedRead>("ReadVIMulAddV_" # mx);
    defvar ReadVIMulAddX_MX = !cast<SchedRead>("ReadVIMulAddX_" # mx);

    defm "" : VPseudoTernaryV_VV_AAXA_LMUL<m, Constraint>,
              Sched<[WriteVIMulAddV_MX, ReadVIMulAddV_MX, ReadVIMulAddV_MX,
                     ReadVIMulAddV_MX, ReadVMask]>;
    defm "" : VPseudoTernaryV_VX_AAXA<m, Constraint>,
              Sched<[WriteVIMulAddX_MX, ReadVIMulAddV_MX, ReadVIMulAddV_MX,
                     ReadVIMulAddX_MX, ReadVMask]>;
  }
}

multiclass VPseudoVMAC_VV_VF_AAXA<string Constraint = ""> {
  foreach m = MxListF in {
    defvar mx = m.MX;
    defvar WriteVFMulAddV_MX = !cast<SchedWrite>("WriteVFMulAddV_" # mx);
    defvar ReadVFMulAddV_MX = !cast<SchedRead>("ReadVFMulAddV_" # mx);

    defm "" : VPseudoTernaryV_VV_AAXA_LMUL<m, Constraint>,
              Sched<[WriteVFMulAddV_MX, ReadVFMulAddV_MX, ReadVFMulAddV_MX, ReadVFMulAddV_MX, ReadVMask]>;
  }

  foreach f = FPList in {
    foreach m = f.MxList in {
      defvar mx = m.MX;
      defvar WriteVFMulAddF_MX = !cast<SchedWrite>("WriteVFMulAddF_" # mx);
      defvar ReadVFMulAddV_MX = !cast<SchedRead>("ReadVFMulAddV_" # mx);
      defvar ReadVFMulAddF_MX = !cast<SchedRead>("ReadVFMulAddF_" # mx);

      defm "" : VPseudoTernaryV_VF_AAXA<m, f, Constraint>,
                Sched<[WriteVFMulAddF_MX, ReadVFMulAddV_MX, ReadVFMulAddV_MX, ReadVFMulAddF_MX, ReadVMask]>;
    }
  }
}

multiclass VPseudoVSLD_VX_VI<Operand ImmType = simm5, string Constraint = ""> {
  defm "" : VPseudoVSLDV_VX<Constraint>,
            Sched<[WriteVISlideX, ReadVISlideV, ReadVISlideV, ReadVISlideX, ReadVMask]>;
  defm "" : VPseudoVSLDV_VI<ImmType, Constraint>,
            Sched<[WriteVISlideI, ReadVISlideV, ReadVISlideV, ReadVMask]>;
}

multiclass VPseudoVWMAC_VV_VX {
  foreach m = MxListW in {
    defvar mx = m.MX;
    defvar WriteVIWMulAddV_MX = !cast<SchedWrite>("WriteVIWMulAddV_" # mx);
    defvar WriteVIWMulAddX_MX = !cast<SchedWrite>("WriteVIWMulAddX_" # mx);
    defvar ReadVIWMulAddV_MX = !cast<SchedRead>("ReadVIWMulAddV_" # mx);
    defvar ReadVIWMulAddX_MX = !cast<SchedRead>("ReadVIWMulAddX_" # mx);

    defm "" : VPseudoTernaryW_VV_LMUL<m>,
              Sched<[WriteVIWMulAddV_MX, ReadVIWMulAddV_MX, ReadVIWMulAddV_MX,
                     ReadVIWMulAddV_MX, ReadVMask]>;
    defm "" : VPseudoTernaryW_VX<m>,
              Sched<[WriteVIWMulAddX_MX, ReadVIWMulAddV_MX, ReadVIWMulAddV_MX,
                     ReadVIWMulAddX_MX, ReadVMask]>;
  }
}

multiclass VPseudoVWMAC_VX {
  foreach m = MxListW in {
    defvar mx = m.MX;
    defvar WriteVIWMulAddX_MX = !cast<SchedWrite>("WriteVIWMulAddX_" # mx);
    defvar ReadVIWMulAddV_MX= !cast<SchedRead>("ReadVIWMulAddV_" # mx);
    defvar ReadVIWMulAddX_MX = !cast<SchedRead>("ReadVIWMulAddX_" # mx);

    defm "" : VPseudoTernaryW_VX<m>,
              Sched<[WriteVIWMulAddX_MX, ReadVIWMulAddV_MX, ReadVIWMulAddV_MX,
                     ReadVIWMulAddX_MX, ReadVMask]>;
  }
}

multiclass VPseudoVWMAC_VV_VF {
  foreach m = MxListFW in {
    defvar mx = m.MX;
    defvar WriteVFWMulAddV_MX = !cast<SchedWrite>("WriteVFWMulAddV_" # mx);
    defvar ReadVFWMulAddV_MX = !cast<SchedRead>("ReadVFWMulAddV_" # mx);

    defm "" : VPseudoTernaryW_VV_LMUL<m>,
              Sched<[WriteVFWMulAddV_MX, ReadVFWMulAddV_MX,
                     ReadVFWMulAddV_MX, ReadVFWMulAddV_MX, ReadVMask]>;
  }

  foreach f = FPListW in {
    foreach m = f.MxListFW in {
      defvar mx = m.MX;
      defvar WriteVFWMulAddF_MX = !cast<SchedWrite>("WriteVFWMulAddF_" # mx);
      defvar ReadVFWMulAddV_MX = !cast<SchedRead>("ReadVFWMulAddV_" # mx);
      defvar ReadVFWMulAddF_MX = !cast<SchedRead>("ReadVFWMulAddF_" # mx);

      defm "" : VPseudoTernaryW_VF<m, f>,
                Sched<[WriteVFWMulAddF_MX, ReadVFWMulAddV_MX,
                       ReadVFWMulAddV_MX, ReadVFWMulAddF_MX, ReadVMask]>;
    }
  }
}

multiclass VPseudoVCMPM_VV_VX_VI {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVICmpV_MX = !cast<SchedWrite>("WriteVICmpV_" # mx);
    defvar WriteVICmpX_MX = !cast<SchedWrite>("WriteVICmpX_" # mx);
    defvar WriteVICmpI_MX = !cast<SchedWrite>("WriteVICmpI_" # mx);
    defvar ReadVICmpV_MX = !cast<SchedRead>("ReadVICmpV_" # mx);
    defvar ReadVICmpX_MX = !cast<SchedRead>("ReadVICmpX_" # mx);

    defm "" : VPseudoBinaryM_VV_LMUL<m>,
              Sched<[WriteVICmpV_MX, ReadVICmpV_MX, ReadVICmpV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryM_VX<m>,
              Sched<[WriteVICmpX_MX, ReadVICmpV_MX, ReadVICmpX_MX, ReadVMask]>;
    defm "" : VPseudoBinaryM_VI<m>,
              Sched<[WriteVICmpI_MX, ReadVICmpV_MX, ReadVMask]>;
  }
}

multiclass VPseudoVCMPM_VV_VX {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVICmpV_MX = !cast<SchedWrite>("WriteVICmpV_" # mx);
    defvar WriteVICmpX_MX = !cast<SchedWrite>("WriteVICmpX_" # mx);
    defvar ReadVICmpV_MX = !cast<SchedRead>("ReadVICmpV_" # mx);
    defvar ReadVICmpX_MX = !cast<SchedRead>("ReadVICmpX_" # mx);

    defm "" : VPseudoBinaryM_VV_LMUL<m>,
              Sched<[WriteVICmpV_MX, ReadVICmpV_MX, ReadVICmpV_MX, ReadVMask]>;
    defm "" : VPseudoBinaryM_VX<m>,
              Sched<[WriteVICmpX_MX, ReadVICmpV_MX, ReadVICmpX_MX, ReadVMask]>;
  }
}

multiclass VPseudoVCMPM_VV_VF {
  foreach m = MxListF in {
    defvar mx = m.MX;
    defvar WriteVFCmpV_MX = !cast<SchedWrite>("WriteVFCmpV_" # mx);
    defvar ReadVFCmpV_MX = !cast<SchedRead>("ReadVFCmpV_" # mx);

    defm "" : VPseudoBinaryM_VV_LMUL<m>,
              Sched<[WriteVFCmpV_MX, ReadVFCmpV_MX, ReadVFCmpV_MX, ReadVMask]>;
  }

  foreach f = FPList in {
    foreach m = f.MxList in {
      defvar mx = m.MX;
      defvar WriteVFCmpF_MX = !cast<SchedWrite>("WriteVFCmpF_" # mx);
      defvar ReadVFCmpV_MX = !cast<SchedRead>("ReadVFCmpV_" # mx);
      defvar ReadVFCmpF_MX = !cast<SchedRead>("ReadVFCmpF_" # mx);

      defm "" : VPseudoBinaryM_VF<m, f>,
                Sched<[WriteVFCmpF_MX, ReadVFCmpV_MX, ReadVFCmpF_MX, ReadVMask]>;
    }
  }
}

multiclass VPseudoVCMPM_VF {
  foreach f = FPList in {
    foreach m = f.MxList in {
      defvar mx = m.MX;
      defvar WriteVFCmpF_MX = !cast<SchedWrite>("WriteVFCmpF_" # mx);
      defvar ReadVFCmpV_MX = !cast<SchedRead>("ReadVFCmpV_" # mx);
      defvar ReadVFCmpF_MX = !cast<SchedRead>("ReadVFCmpF_" # mx);

      defm "" : VPseudoBinaryM_VF<m, f>,
                Sched<[WriteVFCmpF_MX, ReadVFCmpV_MX, ReadVFCmpF_MX, ReadVMask]>;
    }
  }
}

multiclass VPseudoVCMPM_VX_VI {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVICmpX_MX = !cast<SchedWrite>("WriteVICmpX_" # mx);
    defvar WriteVICmpI_MX = !cast<SchedWrite>("WriteVICmpI_" # mx);
    defvar ReadVICmpV_MX = !cast<SchedRead>("ReadVICmpV_" # mx);
    defvar ReadVICmpX_MX = !cast<SchedRead>("ReadVICmpX_" # mx);

    defm "" : VPseudoBinaryM_VX<m>,
              Sched<[WriteVICmpX_MX, ReadVICmpV_MX, ReadVICmpX_MX, ReadVMask]>;
    defm "" : VPseudoBinaryM_VI<m>,
              Sched<[WriteVICmpI_MX, ReadVICmpV_MX, ReadVMask]>;
  }
}

multiclass VPseudoVRED_VS {
  foreach m = MxList in {
    defm _VS : VPseudoTernary<V_M1.vrclass, m.vrclass, V_M1.vrclass, m>,
               Sched<[WriteVIRedV, ReadVIRedV, ReadVIRedV, ReadVIRedV, ReadVMask]>;
  }
}

multiclass VPseudoVWRED_VS {
  foreach m = MxList in {
    defm _VS : VPseudoTernary<V_M1.vrclass, m.vrclass, V_M1.vrclass, m>,
               Sched<[WriteVIWRedV, ReadVIWRedV, ReadVIWRedV, ReadVIWRedV, ReadVMask]>;
  }
}

multiclass VPseudoVFRED_VS {
  foreach m = MxListF in {
    defm _VS : VPseudoTernary<V_M1.vrclass, m.vrclass, V_M1.vrclass, m>,
               Sched<[WriteVFRedV, ReadVFRedV, ReadVFRedV, ReadVFRedV, ReadVMask]>;
  }
}

multiclass VPseudoVFREDO_VS {
  foreach m = MxListF in {
    defm _VS : VPseudoTernary<V_M1.vrclass, m.vrclass, V_M1.vrclass, m>,
               Sched<[WriteVFRedOV, ReadVFRedOV, ReadVFRedOV, ReadVFRedOV, ReadVMask]>;
  }
}

multiclass VPseudoVFWRED_VS {
  foreach m = MxListF in {
    defm _VS : VPseudoTernary<V_M1.vrclass, m.vrclass, V_M1.vrclass, m>,
               Sched<[WriteVFWRedV, ReadVFWRedV, ReadVFWRedV, ReadVFWRedV, ReadVMask]>;
  }
}

multiclass VPseudoConversion<VReg RetClass,
                             VReg Op1Class,
                             LMULInfo MInfo,
                             string Constraint = ""> {
  let VLMul = MInfo.value in {
    def "_" # MInfo.MX : VPseudoUnaryNoMask<RetClass, Op1Class, Constraint>;
    def "_" # MInfo.MX # "_TU": VPseudoUnaryNoMaskTU<RetClass, Op1Class, Constraint>;
    def "_" # MInfo.MX # "_MASK" : VPseudoUnaryMaskTA<RetClass, Op1Class,
                                                      Constraint>,
                                   RISCVMaskedPseudo</*MaskOpIdx*/ 2>;
  }
}

multiclass VPseudoConversionRM<VReg RetClass,
                               VReg Op1Class,
                               LMULInfo MInfo,
                               string Constraint = ""> {
  let VLMul = MInfo.value in {
    def "_" # MInfo.MX # "_MASK" : VPseudoUnaryMaskTA_FRM<RetClass, Op1Class,
                                                          Constraint>;
  }
}

multiclass VPseudoConversionNoExcept<VReg RetClass,
                                     VReg Op1Class,
                                     LMULInfo MInfo,
                                     string Constraint = ""> {
  let VLMul = MInfo.value in {
    def "_" # MInfo.MX # "_MASK" : VPseudoUnaryMaskTA_NoExcept<RetClass, Op1Class, Constraint>;
  }
}

multiclass VPseudoVCVTI_V {
  foreach m = MxListF in {
    defvar mx = m.MX;
    defvar WriteVFCvtFToIV_MX = !cast<SchedWrite>("WriteVFCvtFToIV_" # mx);
    defvar ReadVFCvtFToIV_MX = !cast<SchedRead>("ReadVFCvtFToIV_" # mx);

    defm _V : VPseudoConversion<m.vrclass, m.vrclass, m>,
              Sched<[WriteVFCvtFToIV_MX, ReadVFCvtFToIV_MX, ReadVMask]>;
  }
}

multiclass VPseudoVCVTI_RM_V {
  foreach m = MxListF in {
    defvar mx = m.MX;
    defvar WriteVFCvtFToIV_MX = !cast<SchedWrite>("WriteVFCvtFToIV_" # mx);
    defvar ReadVFCvtFToIV_MX = !cast<SchedRead>("ReadVFCvtFToIV_" # mx);

    defm _V : VPseudoConversionRM<m.vrclass, m.vrclass, m>,
              Sched<[WriteVFCvtFToIV_MX, ReadVFCvtFToIV_MX, ReadVMask]>;
  }
}

multiclass VPseudoVFROUND_NOEXCEPT_V {
  foreach m = MxListF in {
    defvar mx = m.MX;
    defvar WriteVFCvtFToIV_MX = !cast<SchedWrite>("WriteVFCvtFToIV_" # mx);
    defvar ReadVFCvtFToIV_MX = !cast<SchedRead>("ReadVFCvtFToIV_" # mx);

    defm _V : VPseudoConversionNoExcept<m.vrclass, m.vrclass, m>,
              Sched<[WriteVFCvtFToIV_MX, ReadVFCvtFToIV_MX, ReadVMask]>;
  }
}

multiclass VPseudoVCVTF_V {
  foreach m = MxListF in {
    defvar mx = m.MX;
    defvar WriteVFCvtIToFV_MX = !cast<SchedWrite>("WriteVFCvtIToFV_" # mx);
    defvar ReadVFCvtIToFV_MX = !cast<SchedRead>("ReadVFCvtIToFV_" # mx);

    defm _V : VPseudoConversion<m.vrclass, m.vrclass, m>,
              Sched<[WriteVFCvtIToFV_MX, ReadVFCvtIToFV_MX, ReadVMask]>;
  }
}

multiclass VPseudoConversionW_V {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxListW in
    defm _V : VPseudoConversion<m.wvrclass, m.vrclass, m, constraint>;
}

multiclass VPseudoVWCVTI_V {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxListFW in {
    defvar mx = m.MX;
    defvar WriteVFWCvtFToIV_MX = !cast<SchedWrite>("WriteVFWCvtFToIV_" # mx);
    defvar ReadVFWCvtFToIV_MX = !cast<SchedRead>("ReadVFWCvtFToIV_" # mx);

    defm _V : VPseudoConversion<m.wvrclass, m.vrclass, m, constraint>,
              Sched<[WriteVFWCvtFToIV_MX, ReadVFWCvtFToIV_MX, ReadVMask]>;
  }
}

multiclass VPseudoVWCVTF_V {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxListW in {
    defvar mx = m.MX;
    defvar WriteVFWCvtIToFV_MX = !cast<SchedWrite>("WriteVFWCvtIToFV_" # mx);
    defvar ReadVFWCvtIToFV_MX = !cast<SchedRead>("ReadVFWCvtIToFV_" # mx);

    defm _V : VPseudoConversion<m.wvrclass, m.vrclass, m, constraint>,
              Sched<[WriteVFWCvtIToFV_MX, ReadVFWCvtIToFV_MX, ReadVMask]>;
  }
}

multiclass VPseudoVWCVTD_V {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxListFW in {
    defvar mx = m.MX;
    defvar WriteVFWCvtFToFV_MX = !cast<SchedWrite>("WriteVFWCvtFToFV_" # mx);
    defvar ReadVFWCvtFToFV_MX = !cast<SchedRead>("ReadVFWCvtFToFV_" # mx);

    defm _V : VPseudoConversion<m.wvrclass, m.vrclass, m, constraint>,
              Sched<[WriteVFWCvtFToFV_MX, ReadVFWCvtFToFV_MX, ReadVMask]>;
  }
}

multiclass VPseudoVNCVTI_W {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxListW in {
    defvar mx = m.MX;
    defvar WriteVFNCvtFToIV_MX = !cast<SchedWrite>("WriteVFNCvtFToIV_" # mx);
    defvar ReadVFNCvtFToIV_MX = !cast<SchedRead>("ReadVFNCvtFToIV_" # mx);

    defm _W : VPseudoConversion<m.vrclass, m.wvrclass, m, constraint>,
              Sched<[WriteVFNCvtFToIV_MX, ReadVFNCvtFToIV_MX, ReadVMask]>;
  }
}

multiclass VPseudoVNCVTF_W {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxListFW in {
    defvar mx = m.MX;
    defvar WriteVFNCvtIToFV_MX = !cast<SchedWrite>("WriteVFNCvtIToFV_" # mx);
    defvar ReadVFNCvtIToFV_MX = !cast<SchedRead>("ReadVFNCvtIToFV_" # mx);

    defm _W : VPseudoConversion<m.vrclass, m.wvrclass, m, constraint>,
              Sched<[WriteVFNCvtIToFV_MX, ReadVFNCvtIToFV_MX, ReadVMask]>;
  }
}

multiclass VPseudoVNCVTD_W {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxListFW in {
    defvar mx = m.MX;
    defvar WriteVFNCvtFToFV_MX = !cast<SchedWrite>("WriteVFNCvtFToFV_" # mx);
    defvar ReadVFNCvtFToFV_MX = !cast<SchedRead>("ReadVFNCvtFToFV_" # mx);

    defm _W : VPseudoConversion<m.vrclass, m.wvrclass, m, constraint>,
              Sched<[WriteVFNCvtFToFV_MX, ReadVFNCvtFToFV_MX, ReadVMask]>;
  }
}

multiclass VPseudoUSSegLoad {
  foreach eew = EEWList in {
    foreach lmul = MxSet<eew>.m in {
      defvar LInfo = lmul.MX;
      let VLMul = lmul.value in {
        foreach nf = NFSet<lmul>.L in {
          defvar vreg = SegRegClass<lmul, nf>.RC;
          def nf # "E" # eew # "_V_" # LInfo :
            VPseudoUSSegLoadNoMask<vreg, eew, nf>, VLSEGSched<nf, eew>;
          def nf # "E" # eew # "_V_" # LInfo # "_TU" :
            VPseudoUSSegLoadNoMaskTU<vreg, eew, nf>, VLSEGSched<nf, eew>;
          def nf # "E" # eew # "_V_" # LInfo # "_MASK" :
            VPseudoUSSegLoadMask<vreg, eew, nf>, VLSEGSched<nf, eew>;
        }
      }
    }
  }
}

multiclass VPseudoUSSegLoadFF {
  foreach eew = EEWList in {
    foreach lmul = MxSet<eew>.m in {
      defvar LInfo = lmul.MX;
      let VLMul = lmul.value in {
        foreach nf = NFSet<lmul>.L in {
          defvar vreg = SegRegClass<lmul, nf>.RC;
          def nf # "E" # eew # "FF_V_" # LInfo :
            VPseudoUSSegLoadFFNoMask<vreg, eew, nf>, VLSEGFFSched<nf, eew>;
          def nf # "E" # eew # "FF_V_" # LInfo # "_TU" :
            VPseudoUSSegLoadFFNoMaskTU<vreg, eew, nf>, VLSEGFFSched<nf, eew>;
          def nf # "E" # eew # "FF_V_" # LInfo # "_MASK" :
            VPseudoUSSegLoadFFMask<vreg, eew, nf>, VLSEGFFSched<nf, eew>;
        }
      }
    }
  }
}

multiclass VPseudoSSegLoad {
  foreach eew = EEWList in {
    foreach lmul = MxSet<eew>.m in {
      defvar LInfo = lmul.MX;
      let VLMul = lmul.value in {
        foreach nf = NFSet<lmul>.L in {
          defvar vreg = SegRegClass<lmul, nf>.RC;
          def nf # "E" # eew # "_V_" # LInfo : VPseudoSSegLoadNoMask<vreg, eew, nf>,
                                               VLSSEGSched<nf, eew>;
          def nf # "E" # eew # "_V_" # LInfo # "_TU" : VPseudoSSegLoadNoMaskTU<vreg, eew, nf>,
                                                       VLSSEGSched<nf, eew>;
          def nf # "E" # eew # "_V_" # LInfo # "_MASK" : VPseudoSSegLoadMask<vreg, eew, nf>,
                                                         VLSSEGSched<nf, eew>;
        }
      }
    }
  }
}

multiclass VPseudoISegLoad<bit Ordered> {
  foreach idx_eew = EEWList in {
    foreach sew = EEWList in {
      foreach val_lmul = MxSet<sew>.m in {
        defvar octuple_lmul = val_lmul.octuple;
        // Calculate emul = eew * lmul / sew
        defvar octuple_emul = !srl(!mul(idx_eew, octuple_lmul), log2<sew>.val);
        if !and(!ge(octuple_emul, 1), !le(octuple_emul, 64)) then {
          defvar ValLInfo = val_lmul.MX;
          defvar IdxLInfo = octuple_to_str<octuple_emul>.ret;
          defvar idx_lmul = !cast<LMULInfo>("V_" # IdxLInfo);
          defvar Vreg = val_lmul.vrclass;
          defvar IdxVreg = idx_lmul.vrclass;
          defvar Order = !if(Ordered, "O", "U");
          let VLMul = val_lmul.value in {
            foreach nf = NFSet<val_lmul>.L in {
              defvar ValVreg = SegRegClass<val_lmul, nf>.RC;
              def nf # "EI" # idx_eew # "_V_" # IdxLInfo # "_" # ValLInfo :
                VPseudoISegLoadNoMask<ValVreg, IdxVreg, idx_eew, idx_lmul.value,
                                      nf, Ordered>,
                VLXSEGSched<nf, idx_eew, Order>;
              def nf # "EI" # idx_eew # "_V_" # IdxLInfo # "_" # ValLInfo # "_TU" :
                VPseudoISegLoadNoMaskTU<ValVreg, IdxVreg, idx_eew, idx_lmul.value,
                                        nf, Ordered>,
                VLXSEGSched<nf, idx_eew, Order>;
              def nf # "EI" # idx_eew # "_V_" # IdxLInfo # "_" # ValLInfo # "_MASK" :
                VPseudoISegLoadMask<ValVreg, IdxVreg, idx_eew, idx_lmul.value,
                                    nf, Ordered>,
                VLXSEGSched<nf, idx_eew, Order>;
            }
          }
        }
      }
    }
  }
}

multiclass VPseudoUSSegStore {
  foreach eew = EEWList in {
    foreach lmul = MxSet<eew>.m in {
      defvar LInfo = lmul.MX;
      let VLMul = lmul.value in {
        foreach nf = NFSet<lmul>.L in {
          defvar vreg = SegRegClass<lmul, nf>.RC;
          def nf # "E" # eew # "_V_" # LInfo : VPseudoUSSegStoreNoMask<vreg, eew, nf>,
                                               VSSEGSched<nf, eew>;
          def nf # "E" # eew # "_V_" # LInfo # "_MASK" : VPseudoUSSegStoreMask<vreg, eew, nf>,
                                                         VSSEGSched<nf, eew>;
        }
      }
    }
  }
}

multiclass VPseudoSSegStore {
  foreach eew = EEWList in {
    foreach lmul = MxSet<eew>.m in {
      defvar LInfo = lmul.MX;
      let VLMul = lmul.value in {
        foreach nf = NFSet<lmul>.L in {
          defvar vreg = SegRegClass<lmul, nf>.RC;
          def nf # "E" # eew # "_V_" # LInfo : VPseudoSSegStoreNoMask<vreg, eew, nf>,
                                               VSSSEGSched<nf, eew>;
          def nf # "E" # eew # "_V_" # LInfo # "_MASK" : VPseudoSSegStoreMask<vreg, eew, nf>,
                                                         VSSSEGSched<nf, eew>;
        }
      }
    }
  }
}

multiclass VPseudoISegStore<bit Ordered> {
  foreach idx_eew = EEWList in {
    foreach sew = EEWList in {
      foreach val_lmul = MxSet<sew>.m in {
        defvar octuple_lmul = val_lmul.octuple;
        // Calculate emul = eew * lmul / sew
        defvar octuple_emul = !srl(!mul(idx_eew, octuple_lmul), log2<sew>.val);
        if !and(!ge(octuple_emul, 1), !le(octuple_emul, 64)) then {
          defvar ValLInfo = val_lmul.MX;
          defvar IdxLInfo = octuple_to_str<octuple_emul>.ret;
          defvar idx_lmul = !cast<LMULInfo>("V_" # IdxLInfo);
          defvar Vreg = val_lmul.vrclass;
          defvar IdxVreg = idx_lmul.vrclass;
          defvar Order = !if(Ordered, "O", "U");
          let VLMul = val_lmul.value in {
            foreach nf = NFSet<val_lmul>.L in {
              defvar ValVreg = SegRegClass<val_lmul, nf>.RC;
              def nf # "EI" # idx_eew # "_V_" # IdxLInfo # "_" # ValLInfo :
                VPseudoISegStoreNoMask<ValVreg, IdxVreg, idx_eew, idx_lmul.value,
                                       nf, Ordered>,
                VSXSEGSched<nf, idx_eew, Order>;
              def nf # "EI" # idx_eew # "_V_" # IdxLInfo # "_" # ValLInfo # "_MASK" :
                VPseudoISegStoreMask<ValVreg, IdxVreg, idx_eew, idx_lmul.value,
                                     nf, Ordered>,
                VSXSEGSched<nf, idx_eew, Order>;
            }
          }
        }
      }
    }
  }
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
                   (result_type undef),
                   (op2_type op2_reg_class:$rs2),
                   VLOpFrag)),
                   (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX)
                   (op2_type op2_reg_class:$rs2),
                   GPR:$vl, sew)>;

class VPatUnaryNoMaskTU<string intrinsic_name,
                        string inst,
                        string kind,
                        ValueType result_type,
                        ValueType op2_type,
                        int sew,
                        LMULInfo vlmul,
                        VReg result_reg_class,
                        VReg op2_reg_class> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                   (result_type result_reg_class:$merge),
                   (op2_type op2_reg_class:$rs2),
                   VLOpFrag)),
                   (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX#"_TU")
                   (result_type result_reg_class:$merge),
                   (op2_type op2_reg_class:$rs2),
                   GPR:$vl, sew)>;

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
                   VLOpFrag)),
                   (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX#"_MASK")
                   (result_type result_reg_class:$merge),
                   (op2_type op2_reg_class:$rs2),
                   (mask_type V0), GPR:$vl, sew)>;

class VPatUnaryMaskTA<string intrinsic_name,
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
                   VLOpFrag, (XLenVT timm:$policy))),
                   (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX#"_MASK")
                   (result_type result_reg_class:$merge),
                   (op2_type op2_reg_class:$rs2),
                   (mask_type V0), GPR:$vl, sew, (XLenVT timm:$policy))>;

class VPatMaskUnaryNoMask<string intrinsic_name,
                          string inst,
                          MTypeInfo mti> :
  Pat<(mti.Mask (!cast<Intrinsic>(intrinsic_name)
                (mti.Mask VR:$rs2),
                VLOpFrag)),
                (!cast<Instruction>(inst#"_M_"#mti.BX)
                (mti.Mask VR:$rs2),
                GPR:$vl, mti.Log2SEW)>;

class VPatMaskUnaryMask<string intrinsic_name,
                        string inst,
                        MTypeInfo mti> :
  Pat<(mti.Mask (!cast<Intrinsic>(intrinsic_name#"_mask")
                (mti.Mask VR:$merge),
                (mti.Mask VR:$rs2),
                (mti.Mask V0),
                VLOpFrag)),
                (!cast<Instruction>(inst#"_M_"#mti.BX#"_MASK")
                (mti.Mask VR:$merge),
                (mti.Mask VR:$rs2),
                (mti.Mask V0), GPR:$vl, mti.Log2SEW)>;

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
  Pat<(result_type (!cast<Intrinsic>(intrinsic)
                   (result_type result_reg_class:$merge),
                   (op1_type op1_reg_class:$rs1),
                   (mask_type VR:$rs2),
                   VLOpFrag)),
                   (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX)
                   (result_type result_reg_class:$merge),
                   (op1_type op1_reg_class:$rs1),
                   (mask_type VR:$rs2),
                   GPR:$vl, sew)>;

class VPatBinaryM<string intrinsic_name,
                  string inst,
                  ValueType result_type,
                  ValueType op1_type,
                  ValueType op2_type,
                  int sew,
                  VReg op1_reg_class,
                  DAGOperand op2_kind> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                   (op1_type op1_reg_class:$rs1),
                   (op2_type op2_kind:$rs2),
                   VLOpFrag)),
                   (!cast<Instruction>(inst)
                   (op1_type op1_reg_class:$rs1),
                   (op2_type op2_kind:$rs2),
                   GPR:$vl, sew)>;

class VPatBinaryNoMaskTA<string intrinsic_name,
                         string inst,
                         ValueType result_type,
                         ValueType op1_type,
                         ValueType op2_type,
                         int sew,
                         VReg op1_reg_class,
                         DAGOperand op2_kind> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                   (result_type (undef)),
                   (op1_type op1_reg_class:$rs1),
                   (op2_type op2_kind:$rs2),
                   VLOpFrag)),
                   (!cast<Instruction>(inst)
                   (op1_type op1_reg_class:$rs1),
                   (op2_type op2_kind:$rs2),
                   GPR:$vl, sew)>;

class VPatBinaryNoMaskTU<string intrinsic_name,
                         string inst,
                         ValueType result_type,
                         ValueType op1_type,
                         ValueType op2_type,
                         int sew,
                         VReg result_reg_class,
                         VReg op1_reg_class,
                         DAGOperand op2_kind> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                   (result_type result_reg_class:$merge),
                   (op1_type op1_reg_class:$rs1),
                   (op2_type op2_kind:$rs2),
                   VLOpFrag)),
                   (!cast<Instruction>(inst#"_TU")
                   (result_type result_reg_class:$merge),
                   (op1_type op1_reg_class:$rs1),
                   (op2_type op2_kind:$rs2),
                   GPR:$vl, sew)>;

// Same as above but source operands are swapped.
class VPatBinaryNoMaskSwapped<string intrinsic_name,
                              string inst,
                              ValueType result_type,
                              ValueType op1_type,
                              ValueType op2_type,
                              int sew,
                              VReg op1_reg_class,
                              DAGOperand op2_kind> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                   (op2_type op2_kind:$rs2),
                   (op1_type op1_reg_class:$rs1),
                   VLOpFrag)),
                   (!cast<Instruction>(inst)
                   (op1_type op1_reg_class:$rs1),
                   (op2_type op2_kind:$rs2),
                   GPR:$vl, sew)>;

class VPatBinaryMask<string intrinsic_name,
                     string inst,
                     ValueType result_type,
                     ValueType op1_type,
                     ValueType op2_type,
                     ValueType mask_type,
                     int sew,
                     VReg result_reg_class,
                     VReg op1_reg_class,
                     DAGOperand op2_kind> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic_name#"_mask")
                   (result_type result_reg_class:$merge),
                   (op1_type op1_reg_class:$rs1),
                   (op2_type op2_kind:$rs2),
                   (mask_type V0),
                   VLOpFrag)),
                   (!cast<Instruction>(inst#"_MASK")
                   (result_type result_reg_class:$merge),
                   (op1_type op1_reg_class:$rs1),
                   (op2_type op2_kind:$rs2),
                   (mask_type V0), GPR:$vl, sew)>;

class VPatBinaryMaskTA<string intrinsic_name,
                       string inst,
                       ValueType result_type,
                       ValueType op1_type,
                       ValueType op2_type,
                       ValueType mask_type,
                       int sew,
                       VReg result_reg_class,
                       VReg op1_reg_class,
                       DAGOperand op2_kind> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic_name#"_mask")
                   (result_type result_reg_class:$merge),
                   (op1_type op1_reg_class:$rs1),
                   (op2_type op2_kind:$rs2),
                   (mask_type V0),
                   VLOpFrag, (XLenVT timm:$policy))),
                   (!cast<Instruction>(inst#"_MASK")
                   (result_type result_reg_class:$merge),
                   (op1_type op1_reg_class:$rs1),
                   (op2_type op2_kind:$rs2),
                   (mask_type V0), GPR:$vl, sew, (XLenVT timm:$policy))>;

// Same as above but source operands are swapped.
class VPatBinaryMaskSwapped<string intrinsic_name,
                            string inst,
                            ValueType result_type,
                            ValueType op1_type,
                            ValueType op2_type,
                            ValueType mask_type,
                            int sew,
                            VReg result_reg_class,
                            VReg op1_reg_class,
                            DAGOperand op2_kind> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic_name#"_mask")
                   (result_type result_reg_class:$merge),
                   (op2_type op2_kind:$rs2),
                   (op1_type op1_reg_class:$rs1),
                   (mask_type V0),
                   VLOpFrag)),
                   (!cast<Instruction>(inst#"_MASK")
                   (result_type result_reg_class:$merge),
                   (op1_type op1_reg_class:$rs1),
                   (op2_type op2_kind:$rs2),
                   (mask_type V0), GPR:$vl, sew)>;

class VPatTiedBinaryNoMask<string intrinsic_name,
                           string inst,
                           ValueType result_type,
                           ValueType op2_type,
                           int sew,
                           VReg result_reg_class,
                           DAGOperand op2_kind> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                   (result_type (undef)),
                   (result_type result_reg_class:$rs1),
                   (op2_type op2_kind:$rs2),
                   VLOpFrag)),
                   (!cast<Instruction>(inst#"_TIED")
                   (result_type result_reg_class:$rs1),
                   (op2_type op2_kind:$rs2),
                   GPR:$vl, sew, TAIL_AGNOSTIC)>;

class VPatTiedBinaryNoMaskTU<string intrinsic_name,
                             string inst,
                             ValueType result_type,
                             ValueType op2_type,
                             int sew,
                             VReg result_reg_class,
                             DAGOperand op2_kind> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                   (result_type result_reg_class:$merge),
                   (result_type result_reg_class:$merge),
                   (op2_type op2_kind:$rs2),
                   VLOpFrag)),
                   (!cast<Instruction>(inst#"_TIED")
                   (result_type result_reg_class:$merge),
                   (op2_type op2_kind:$rs2),
                   GPR:$vl, sew, TAIL_UNDISTURBED_MASK_UNDISTURBED)>;

class VPatTiedBinaryMask<string intrinsic_name,
                         string inst,
                         ValueType result_type,
                         ValueType op2_type,
                         ValueType mask_type,
                         int sew,
                         VReg result_reg_class,
                         DAGOperand op2_kind> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic_name#"_mask")
                   (result_type result_reg_class:$merge),
                   (result_type result_reg_class:$merge),
                   (op2_type op2_kind:$rs2),
                   (mask_type V0),
                   VLOpFrag, (XLenVT timm:$policy))),
                   (!cast<Instruction>(inst#"_MASK_TIED")
                   (result_type result_reg_class:$merge),
                   (op2_type op2_kind:$rs2),
                   (mask_type V0), GPR:$vl, sew, (XLenVT timm:$policy))>;

class VPatTernaryNoMask<string intrinsic,
                        string inst,
                        string kind,
                        ValueType result_type,
                        ValueType op1_type,
                        ValueType op2_type,
                        int sew,
                        LMULInfo vlmul,
                        VReg result_reg_class,
                        RegisterClass op1_reg_class,
                        DAGOperand op2_kind> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic)
                    (result_type result_reg_class:$rs3),
                    (op1_type op1_reg_class:$rs1),
                    (op2_type op2_kind:$rs2),
                    VLOpFrag)),
                   (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX)
                    result_reg_class:$rs3,
                    (op1_type op1_reg_class:$rs1),
                    op2_kind:$rs2,
                    GPR:$vl, sew)>;

class VPatTernaryNoMaskWithPolicy<string intrinsic,
                                  string inst,
                                  string kind,
                                  ValueType result_type,
                                  ValueType op1_type,
                                  ValueType op2_type,
                                  int sew,
                                  LMULInfo vlmul,
                                  VReg result_reg_class,
                                  RegisterClass op1_reg_class,
                                  DAGOperand op2_kind> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic)
                    (result_type result_reg_class:$rs3),
                    (op1_type op1_reg_class:$rs1),
                    (op2_type op2_kind:$rs2),
                    VLOpFrag, (XLenVT timm:$policy))),
                   (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX)
                    result_reg_class:$rs3,
                    (op1_type op1_reg_class:$rs1),
                    op2_kind:$rs2,
                    GPR:$vl, sew, (XLenVT timm:$policy))>;

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
                    VLOpFrag)),
                   (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX # "_MASK")
                    result_reg_class:$rs3,
                    (op1_type op1_reg_class:$rs1),
                    op2_kind:$rs2,
                    (mask_type V0),
                    GPR:$vl, sew)>;

class VPatTernaryMaskPolicy<string intrinsic,
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
                    VLOpFrag, (XLenVT timm:$policy))),
                   (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX # "_MASK")
                    result_reg_class:$rs3,
                    (op1_type op1_reg_class:$rs1),
                    op2_kind:$rs2,
                    (mask_type V0),
                    GPR:$vl, sew, (XLenVT timm:$policy))>;

multiclass VPatUnaryS_M<string intrinsic_name,
                             string inst>
{
  foreach mti = AllMasks in {
    def : Pat<(XLenVT (!cast<Intrinsic>(intrinsic_name)
                      (mti.Mask VR:$rs1), VLOpFrag)),
                      (!cast<Instruction>(inst#"_M_"#mti.BX) $rs1,
                      GPR:$vl, mti.Log2SEW)>;
    def : Pat<(XLenVT (!cast<Intrinsic>(intrinsic_name # "_mask")
                      (mti.Mask VR:$rs1), (mti.Mask V0), VLOpFrag)),
                      (!cast<Instruction>(inst#"_M_"#mti.BX#"_MASK") $rs1,
                      (mti.Mask V0), GPR:$vl, mti.Log2SEW)>;
  }
}

multiclass VPatUnaryV_V_AnyMask<string intrinsic, string instruction,
                                list<VTypeInfo> vtilist> {
  foreach vti = vtilist in {
    def : VPatUnaryAnyMask<intrinsic, instruction, "VM",
                           vti.Vector, vti.Vector, vti.Mask,
                           vti.Log2SEW, vti.LMul, vti.RegClass,
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
                          vti.Log2SEW, vti.LMul, VR>;
    def : VPatUnaryNoMaskTU<intrinsic, instruction, "M", vti.Vector, vti.Mask,
                            vti.Log2SEW, vti.LMul, vti.RegClass,VR>;
    def : VPatUnaryMaskTA<intrinsic, instruction, "M", vti.Vector, vti.Mask,
                          vti.Mask, vti.Log2SEW, vti.LMul, vti.RegClass, VR>;
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
                            vti.Log2SEW, vti.LMul, fti.RegClass>;
      def : VPatUnaryNoMaskTU<intrinsic, instruction, suffix,
                              vti.Vector, fti.Vector,
                              vti.Log2SEW, vti.LMul, vti.RegClass, fti.RegClass>;
      def : VPatUnaryMaskTA<intrinsic, instruction, suffix,
                            vti.Vector, fti.Vector, vti.Mask,
                            vti.Log2SEW, vti.LMul, vti.RegClass, fti.RegClass>;
   }
}

multiclass VPatUnaryV_V<string intrinsic, string instruction,
                        list<VTypeInfo> vtilist> {
  foreach vti = vtilist in {
    def : VPatUnaryNoMask<intrinsic, instruction, "V",
                          vti.Vector, vti.Vector,
                          vti.Log2SEW, vti.LMul, vti.RegClass>;
    def : VPatUnaryNoMaskTU<intrinsic, instruction, "V",
                            vti.Vector, vti.Vector,
                            vti.Log2SEW, vti.LMul, vti.RegClass, vti.RegClass>;
    def : VPatUnaryMaskTA<intrinsic, instruction, "V",
                          vti.Vector, vti.Vector, vti.Mask,
                          vti.Log2SEW, vti.LMul, vti.RegClass, vti.RegClass>;
  }
}

multiclass VPatNullaryV<string intrinsic, string instruction>
{
  foreach vti = AllIntegerVectors in {
    def : Pat<(vti.Vector (!cast<Intrinsic>(intrinsic)
                          (vti.Vector undef),
                          VLOpFrag)),
                          (!cast<Instruction>(instruction#"_V_" # vti.LMul.MX)
                          GPR:$vl, vti.Log2SEW)>;
    def : Pat<(vti.Vector (!cast<Intrinsic>(intrinsic)
                          (vti.Vector vti.RegClass:$merge),
                          VLOpFrag)),
                          (!cast<Instruction>(instruction#"_V_" # vti.LMul.MX # "_TU")
                          vti.RegClass:$merge, GPR:$vl, vti.Log2SEW)>;
    def : Pat<(vti.Vector (!cast<Intrinsic>(intrinsic # "_mask")
                          (vti.Vector vti.RegClass:$merge),
                          (vti.Mask V0), VLOpFrag, (XLenVT timm:$policy))),
                          (!cast<Instruction>(instruction#"_V_" # vti.LMul.MX # "_MASK")
                          vti.RegClass:$merge, (vti.Mask V0),
                          GPR:$vl, vti.Log2SEW, (XLenVT timm:$policy))>;
  }
}

multiclass VPatNullaryM<string intrinsic, string inst> {
  foreach mti = AllMasks in
    def : Pat<(mti.Mask (!cast<Intrinsic>(intrinsic)
                        VLOpFrag)),
                        (!cast<Instruction>(inst#"_M_"#mti.BX)
                        GPR:$vl, mti.Log2SEW)>;
}

multiclass VPatBinaryM<string intrinsic,
                      string inst,
                      ValueType result_type,
                      ValueType op1_type,
                      ValueType op2_type,
                      ValueType mask_type,
                      int sew,
                      VReg result_reg_class,
                      VReg op1_reg_class,
                      DAGOperand op2_kind>
{
  def : VPatBinaryM<intrinsic, inst, result_type, op1_type, op2_type,
                    sew, op1_reg_class, op2_kind>;
  def : VPatBinaryMask<intrinsic, inst, result_type, op1_type, op2_type,
                       mask_type, sew, result_reg_class, op1_reg_class,
                       op2_kind>;
}

multiclass VPatBinaryTA<string intrinsic,
                        string inst,
                        ValueType result_type,
                        ValueType op1_type,
                        ValueType op2_type,
                        ValueType mask_type,
                        int sew,
                        VReg result_reg_class,
                        VReg op1_reg_class,
                        DAGOperand op2_kind>
{
  def : VPatBinaryNoMaskTA<intrinsic, inst, result_type, op1_type, op2_type,
                           sew, op1_reg_class, op2_kind>;
  def : VPatBinaryNoMaskTU<intrinsic, inst, result_type, op1_type, op2_type,
                           sew, result_reg_class, op1_reg_class, op2_kind>;
  def : VPatBinaryMaskTA<intrinsic, inst, result_type, op1_type, op2_type,
                         mask_type, sew, result_reg_class, op1_reg_class,
                         op2_kind>;
}

multiclass VPatBinarySwapped<string intrinsic,
                      string inst,
                      ValueType result_type,
                      ValueType op1_type,
                      ValueType op2_type,
                      ValueType mask_type,
                      int sew,
                      VReg result_reg_class,
                      VReg op1_reg_class,
                      DAGOperand op2_kind>
{
  def : VPatBinaryNoMaskSwapped<intrinsic, inst, result_type, op1_type, op2_type,
                                sew, op1_reg_class, op2_kind>;
  def : VPatBinaryMaskSwapped<intrinsic, inst, result_type, op1_type, op2_type,
                              mask_type, sew, result_reg_class, op1_reg_class,
                              op2_kind>;
}

multiclass VPatBinaryCarryInTAIL<string intrinsic,
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
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic)
                         (result_type undef),
                         (op1_type op1_reg_class:$rs1),
                         (op2_type op2_kind:$rs2),
                         (mask_type V0),
                         VLOpFrag)),
                         (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX)
                         (op1_type op1_reg_class:$rs1),
                         (op2_type op2_kind:$rs2),
                         (mask_type V0), GPR:$vl, sew)>;
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic)
                         (result_type result_reg_class:$merge),
                         (op1_type op1_reg_class:$rs1),
                         (op2_type op2_kind:$rs2),
                         (mask_type V0),
                         VLOpFrag)),
                         (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX#"_TU")
                         (result_type result_reg_class:$merge),
                         (op1_type op1_reg_class:$rs1),
                         (op2_type op2_kind:$rs2),
                         (mask_type V0), GPR:$vl, sew)>;
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
                         VLOpFrag)),
                         (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX)
                         (op1_type op1_reg_class:$rs1),
                         (op2_type op2_kind:$rs2),
                         (mask_type V0), GPR:$vl, sew)>;
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
                         VLOpFrag)),
                         (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX)
                         (op1_type op1_reg_class:$rs1),
                         (op2_type op2_kind:$rs2),
                         GPR:$vl, sew)>;
}

multiclass VPatConversionTA<string intrinsic,
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
  def : VPatUnaryNoMaskTU<intrinsic, inst, kind, result_type, op1_type,
                          sew, vlmul, result_reg_class, op1_reg_class>;
  def : VPatUnaryMaskTA<intrinsic, inst, kind, result_type, op1_type,
                        mask_type, sew, vlmul, result_reg_class, op1_reg_class>;
}

multiclass VPatBinaryV_VV<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    defm : VPatBinaryTA<intrinsic, instruction # "_VV_" # vti.LMul.MX,
                        vti.Vector, vti.Vector, vti.Vector,vti.Mask,
                        vti.Log2SEW, vti.RegClass,
                        vti.RegClass, vti.RegClass>;
}

multiclass VPatBinaryV_VV_INT<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist> {
  foreach vti = vtilist in {
    defvar ivti = GetIntVTypeInfo<vti>.Vti;
    defm : VPatBinaryTA<intrinsic, instruction # "_VV_" # vti.LMul.MX,
                        vti.Vector, vti.Vector, ivti.Vector, vti.Mask,
                        vti.Log2SEW, vti.RegClass,
                        vti.RegClass, vti.RegClass>;
  }
}

multiclass VPatBinaryV_VV_INT_EEW<string intrinsic, string instruction,
                                  int eew, list<VTypeInfo> vtilist> {
  foreach vti = vtilist in {
    // emul = lmul * eew / sew
    defvar vlmul = vti.LMul;
    defvar octuple_lmul = vlmul.octuple;
    defvar octuple_emul = !srl(!mul(octuple_lmul, eew), vti.Log2SEW);
    if !and(!ge(octuple_emul, 1), !le(octuple_emul, 64)) then {
      defvar emul_str = octuple_to_str<octuple_emul>.ret;
      defvar ivti = !cast<VTypeInfo>("VI" # eew # emul_str);
      defvar inst = instruction # "_VV_" # vti.LMul.MX # "_" # emul_str;
      defm : VPatBinaryTA<intrinsic, inst,
                          vti.Vector, vti.Vector, ivti.Vector, vti.Mask,
                          vti.Log2SEW, vti.RegClass,
                          vti.RegClass, ivti.RegClass>;
    }
  }
}

multiclass VPatBinaryV_VX<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist> {
  foreach vti = vtilist in {
    defvar kind = "V"#vti.ScalarSuffix;
    defm : VPatBinaryTA<intrinsic, instruction#"_"#kind#"_"#vti.LMul.MX,
                        vti.Vector, vti.Vector, vti.Scalar, vti.Mask,
                        vti.Log2SEW, vti.RegClass,
                        vti.RegClass, vti.ScalarRegClass>;
  }
}

multiclass VPatBinaryV_VX_INT<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    defm : VPatBinaryTA<intrinsic, instruction # "_VX_" # vti.LMul.MX,
                        vti.Vector, vti.Vector, XLenVT, vti.Mask,
                        vti.Log2SEW, vti.RegClass,
                        vti.RegClass, GPR>;
}

multiclass VPatBinaryV_VI<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist, Operand imm_type> {
  foreach vti = vtilist in
    defm : VPatBinaryTA<intrinsic, instruction # "_VI_" # vti.LMul.MX,
                        vti.Vector, vti.Vector, XLenVT, vti.Mask,
                        vti.Log2SEW, vti.RegClass,
                        vti.RegClass, imm_type>;
}

multiclass VPatBinaryM_MM<string intrinsic, string instruction> {
  foreach mti = AllMasks in
    def : VPatBinaryM<intrinsic, instruction # "_MM_" # mti.LMul.MX,
                      mti.Mask, mti.Mask, mti.Mask,
                      mti.Log2SEW, VR, VR>;
}

multiclass VPatBinaryW_VV<string intrinsic, string instruction,
                          list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    defm : VPatBinaryTA<intrinsic, instruction # "_VV_" # Vti.LMul.MX,
                        Wti.Vector, Vti.Vector, Vti.Vector, Vti.Mask,
                        Vti.Log2SEW, Wti.RegClass,
                        Vti.RegClass, Vti.RegClass>;
  }
}

multiclass VPatBinaryW_VX<string intrinsic, string instruction,
                          list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    defvar kind = "V"#Vti.ScalarSuffix;
    defm : VPatBinaryTA<intrinsic, instruction#"_"#kind#"_"#Vti.LMul.MX,
                        Wti.Vector, Vti.Vector, Vti.Scalar, Vti.Mask,
                        Vti.Log2SEW, Wti.RegClass,
                        Vti.RegClass, Vti.ScalarRegClass>;
  }
}

multiclass VPatBinaryW_WV<string intrinsic, string instruction,
                          list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    def : VPatTiedBinaryNoMask<intrinsic, instruction # "_WV_" # Vti.LMul.MX,
                               Wti.Vector, Vti.Vector,
                               Vti.Log2SEW, Wti.RegClass, Vti.RegClass>;
    def : VPatBinaryNoMaskTU<intrinsic, instruction # "_WV_" # Vti.LMul.MX,
                             Wti.Vector, Wti.Vector, Vti.Vector, Vti.Log2SEW,
                             Wti.RegClass, Wti.RegClass, Vti.RegClass>;
    let AddedComplexity = 1 in {
    def : VPatTiedBinaryNoMaskTU<intrinsic, instruction # "_WV_" # Vti.LMul.MX,
                                 Wti.Vector, Vti.Vector,
                                 Vti.Log2SEW, Wti.RegClass, Vti.RegClass>;
    def : VPatTiedBinaryMask<intrinsic, instruction # "_WV_" # Vti.LMul.MX,
                             Wti.Vector, Vti.Vector, Vti.Mask,
                             Vti.Log2SEW, Wti.RegClass, Vti.RegClass>;
    }
    def : VPatBinaryMaskTA<intrinsic, instruction # "_WV_" # Vti.LMul.MX,
                           Wti.Vector, Wti.Vector, Vti.Vector, Vti.Mask,
                           Vti.Log2SEW, Wti.RegClass,
                           Wti.RegClass, Vti.RegClass>;
  }
}

multiclass VPatBinaryW_WX<string intrinsic, string instruction,
                          list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    defvar kind = "W"#Vti.ScalarSuffix;
    defm : VPatBinaryTA<intrinsic, instruction#"_"#kind#"_"#Vti.LMul.MX,
                        Wti.Vector, Wti.Vector, Vti.Scalar, Vti.Mask,
                        Vti.Log2SEW, Wti.RegClass,
                        Wti.RegClass, Vti.ScalarRegClass>;
  }
}

multiclass VPatBinaryV_WV<string intrinsic, string instruction,
                          list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    defm : VPatBinaryTA<intrinsic, instruction # "_WV_" # Vti.LMul.MX,
                        Vti.Vector, Wti.Vector, Vti.Vector, Vti.Mask,
                        Vti.Log2SEW, Vti.RegClass,
                        Wti.RegClass, Vti.RegClass>;
  }
}

multiclass VPatBinaryV_WX<string intrinsic, string instruction,
                          list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    defvar kind = "W"#Vti.ScalarSuffix;
    defm : VPatBinaryTA<intrinsic, instruction#"_"#kind#"_"#Vti.LMul.MX,
                        Vti.Vector, Wti.Vector, Vti.Scalar, Vti.Mask,
                        Vti.Log2SEW, Vti.RegClass,
                        Wti.RegClass, Vti.ScalarRegClass>;
  }
}

multiclass VPatBinaryV_WI<string intrinsic, string instruction,
                          list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    defm : VPatBinaryTA<intrinsic, instruction # "_WI_" # Vti.LMul.MX,
                        Vti.Vector, Wti.Vector, XLenVT, Vti.Mask,
                        Vti.Log2SEW, Vti.RegClass,
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
                             vti.Log2SEW, vti.LMul,
                             vti.RegClass, vti.RegClass>;
}

multiclass VPatBinaryV_XM<string intrinsic, string instruction,
                          bit CarryOut = 0,
                          list<VTypeInfo> vtilist = AllIntegerVectors> {
  foreach vti = vtilist in
    defm : VPatBinaryCarryIn<intrinsic, instruction,
                             "V"#vti.ScalarSuffix#"M",
                             !if(CarryOut, vti.Mask, vti.Vector),
                             vti.Vector, vti.Scalar, vti.Mask,
                             vti.Log2SEW, vti.LMul,
                             vti.RegClass, vti.ScalarRegClass>;
}

multiclass VPatBinaryV_IM<string intrinsic, string instruction,
                          bit CarryOut = 0> {
  foreach vti = AllIntegerVectors in
    defm : VPatBinaryCarryIn<intrinsic, instruction, "VIM",
                             !if(CarryOut, vti.Mask, vti.Vector),
                             vti.Vector, XLenVT, vti.Mask,
                             vti.Log2SEW, vti.LMul,
                             vti.RegClass, simm5>;
}

multiclass VPatBinaryV_VM_TAIL<string intrinsic, string instruction,
                               bit CarryOut = 0,
                               list<VTypeInfo> vtilist = AllIntegerVectors> {
  foreach vti = vtilist in
    defm : VPatBinaryCarryInTAIL<intrinsic, instruction, "VVM",
                                 !if(CarryOut, vti.Mask, vti.Vector),
                                 vti.Vector, vti.Vector, vti.Mask,
                                 vti.Log2SEW, vti.LMul, vti.RegClass,
                                 vti.RegClass, vti.RegClass>;
}

multiclass VPatBinaryV_XM_TAIL<string intrinsic, string instruction,
                               bit CarryOut = 0,
                               list<VTypeInfo> vtilist = AllIntegerVectors> {
  foreach vti = vtilist in
    defm : VPatBinaryCarryInTAIL<intrinsic, instruction,
                                 "V"#vti.ScalarSuffix#"M",
                                 !if(CarryOut, vti.Mask, vti.Vector),
                                 vti.Vector, vti.Scalar, vti.Mask,
                                 vti.Log2SEW, vti.LMul, vti.RegClass,
                                 vti.RegClass, vti.ScalarRegClass>;
}

multiclass VPatBinaryV_IM_TAIL<string intrinsic, string instruction,
                               bit CarryOut = 0> {
  foreach vti = AllIntegerVectors in
    defm : VPatBinaryCarryInTAIL<intrinsic, instruction, "VIM",
                                 !if(CarryOut, vti.Mask, vti.Vector),
                                 vti.Vector, XLenVT, vti.Mask,
                                 vti.Log2SEW, vti.LMul,
                                 vti.RegClass, vti.RegClass, simm5>;
}

multiclass VPatBinaryV_V<string intrinsic, string instruction> {
  foreach vti = AllIntegerVectors in
    defm : VPatBinaryMaskOut<intrinsic, instruction, "VV",
                             vti.Mask, vti.Vector, vti.Vector,
                             vti.Log2SEW, vti.LMul,
                             vti.RegClass, vti.RegClass>;
}

multiclass VPatBinaryV_X<string intrinsic, string instruction> {
  foreach vti = AllIntegerVectors in
    defm : VPatBinaryMaskOut<intrinsic, instruction, "VX",
                             vti.Mask, vti.Vector, XLenVT,
                             vti.Log2SEW, vti.LMul,
                             vti.RegClass, GPR>;
}

multiclass VPatBinaryV_I<string intrinsic, string instruction> {
  foreach vti = AllIntegerVectors in
    defm : VPatBinaryMaskOut<intrinsic, instruction, "VI",
                             vti.Mask, vti.Vector, XLenVT,
                             vti.Log2SEW, vti.LMul,
                             vti.RegClass, simm5>;
}

multiclass VPatBinaryM_VV<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    defm : VPatBinaryM<intrinsic, instruction # "_VV_" # vti.LMul.MX,
                       vti.Mask, vti.Vector, vti.Vector, vti.Mask,
                       vti.Log2SEW, VR,
                       vti.RegClass, vti.RegClass>;
}

multiclass VPatBinarySwappedM_VV<string intrinsic, string instruction,
                                 list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    defm : VPatBinarySwapped<intrinsic, instruction # "_VV_" # vti.LMul.MX,
                             vti.Mask, vti.Vector, vti.Vector, vti.Mask,
                             vti.Log2SEW, VR,
                             vti.RegClass, vti.RegClass>;
}

multiclass VPatBinaryM_VX<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist> {
  foreach vti = vtilist in {
    defvar kind = "V"#vti.ScalarSuffix;
    defm : VPatBinaryM<intrinsic, instruction#"_"#kind#"_"#vti.LMul.MX,
                       vti.Mask, vti.Vector, vti.Scalar, vti.Mask,
                       vti.Log2SEW, VR,
                       vti.RegClass, vti.ScalarRegClass>;
  }
}

multiclass VPatBinaryM_VI<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    defm : VPatBinaryM<intrinsic, instruction # "_VI_" # vti.LMul.MX,
                       vti.Mask, vti.Vector, XLenVT, vti.Mask,
                       vti.Log2SEW, VR,
                       vti.RegClass, simm5>;
}

multiclass VPatBinaryV_VV_VX_VI<string intrinsic, string instruction,
                                list<VTypeInfo> vtilist, Operand ImmType = simm5>
    : VPatBinaryV_VV<intrinsic, instruction, vtilist>,
      VPatBinaryV_VX<intrinsic, instruction, vtilist>,
      VPatBinaryV_VI<intrinsic, instruction, vtilist, ImmType>;

multiclass VPatBinaryV_VV_VX<string intrinsic, string instruction,
                             list<VTypeInfo> vtilist>
    : VPatBinaryV_VV<intrinsic, instruction, vtilist>,
      VPatBinaryV_VX<intrinsic, instruction, vtilist>;

multiclass VPatBinaryV_VX_VI<string intrinsic, string instruction,
                             list<VTypeInfo> vtilist>
    : VPatBinaryV_VX<intrinsic, instruction, vtilist>,
      VPatBinaryV_VI<intrinsic, instruction, vtilist, simm5>;

multiclass VPatBinaryW_VV_VX<string intrinsic, string instruction,
                             list<VTypeInfoToWide> vtilist>
    : VPatBinaryW_VV<intrinsic, instruction, vtilist>,
      VPatBinaryW_VX<intrinsic, instruction, vtilist>;

multiclass VPatBinaryW_WV_WX<string intrinsic, string instruction,
                             list<VTypeInfoToWide> vtilist>
    : VPatBinaryW_WV<intrinsic, instruction, vtilist>,
      VPatBinaryW_WX<intrinsic, instruction, vtilist>;

multiclass VPatBinaryV_WV_WX_WI<string intrinsic, string instruction,
                                list<VTypeInfoToWide> vtilist>
    : VPatBinaryV_WV<intrinsic, instruction, vtilist>,
      VPatBinaryV_WX<intrinsic, instruction, vtilist>,
      VPatBinaryV_WI<intrinsic, instruction, vtilist>;

multiclass VPatBinaryV_VM_XM_IM<string intrinsic, string instruction>
    : VPatBinaryV_VM_TAIL<intrinsic, instruction>,
      VPatBinaryV_XM_TAIL<intrinsic, instruction>,
      VPatBinaryV_IM_TAIL<intrinsic, instruction>;

multiclass VPatBinaryM_VM_XM_IM<string intrinsic, string instruction>
    : VPatBinaryV_VM<intrinsic, instruction, /*CarryOut=*/1>,
      VPatBinaryV_XM<intrinsic, instruction, /*CarryOut=*/1>,
      VPatBinaryV_IM<intrinsic, instruction, /*CarryOut=*/1>;

multiclass VPatBinaryM_V_X_I<string intrinsic, string instruction>
    : VPatBinaryV_V<intrinsic, instruction>,
      VPatBinaryV_X<intrinsic, instruction>,
      VPatBinaryV_I<intrinsic, instruction>;

multiclass VPatBinaryV_VM_XM<string intrinsic, string instruction>
    : VPatBinaryV_VM_TAIL<intrinsic, instruction>,
      VPatBinaryV_XM_TAIL<intrinsic, instruction>;

multiclass VPatBinaryM_VM_XM<string intrinsic, string instruction>
    : VPatBinaryV_VM<intrinsic, instruction, /*CarryOut=*/1>,
      VPatBinaryV_XM<intrinsic, instruction, /*CarryOut=*/1>;

multiclass VPatBinaryM_V_X<string intrinsic, string instruction>
    : VPatBinaryV_V<intrinsic, instruction>,
      VPatBinaryV_X<intrinsic, instruction>;

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
                          sew, vlmul, result_reg_class, op1_reg_class,
                          op2_kind>;
  def : VPatTernaryMask<intrinsic, inst, kind, result_type, op1_type, op2_type,
                        mask_type, sew, vlmul, result_reg_class, op1_reg_class,
                        op2_kind>;
}

multiclass VPatTernaryNoMaskNoPolicy<string intrinsic,
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
                          sew, vlmul, result_reg_class, op1_reg_class,
                          op2_kind>;
  def : VPatTernaryMaskPolicy<intrinsic, inst, kind, result_type, op1_type, op2_type,
                              mask_type, sew, vlmul, result_reg_class, op1_reg_class,
                              op2_kind>;
}

multiclass VPatTernaryWithPolicy<string intrinsic,
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
  def : VPatTernaryNoMaskWithPolicy<intrinsic, inst, kind, result_type, op1_type,
                                    op2_type, sew, vlmul, result_reg_class,
                                    op1_reg_class, op2_kind>;
  def : VPatTernaryMaskPolicy<intrinsic, inst, kind, result_type, op1_type, op2_type,
                              mask_type, sew, vlmul, result_reg_class, op1_reg_class,
                              op2_kind>;
}

multiclass VPatTernaryV_VV_AAXA<string intrinsic, string instruction,
                                list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    defm : VPatTernaryWithPolicy<intrinsic, instruction, "VV",
                                 vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                                 vti.Log2SEW, vti.LMul, vti.RegClass,
                                 vti.RegClass, vti.RegClass>;
}

multiclass VPatTernaryV_VX<string intrinsic, string instruction,
                           list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    defm : VPatTernaryWithPolicy<intrinsic, instruction, "VX",
                                 vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                 vti.Log2SEW, vti.LMul, vti.RegClass,
                                 vti.RegClass, GPR>;
}

multiclass VPatTernaryV_VX_AAXA<string intrinsic, string instruction,
                           list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    defm : VPatTernaryWithPolicy<intrinsic, instruction,
                                 "V"#vti.ScalarSuffix,
                                 vti.Vector, vti.Scalar, vti.Vector, vti.Mask,
                                 vti.Log2SEW, vti.LMul, vti.RegClass,
                                 vti.ScalarRegClass, vti.RegClass>;
}

multiclass VPatTernaryV_VI<string intrinsic, string instruction,
                           list<VTypeInfo> vtilist, Operand Imm_type> {
  foreach vti = vtilist in
    defm : VPatTernaryWithPolicy<intrinsic, instruction, "VI",
                                 vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                 vti.Log2SEW, vti.LMul, vti.RegClass,
                                 vti.RegClass, Imm_type>;
}

multiclass VPatTernaryW_VV<string intrinsic, string instruction,
                           list<VTypeInfoToWide> vtilist> {
  foreach vtiToWti = vtilist in {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;
    defm : VPatTernaryWithPolicy<intrinsic, instruction, "VV",
                                 wti.Vector, vti.Vector, vti.Vector,
                                 vti.Mask, vti.Log2SEW, vti.LMul,
                                 wti.RegClass, vti.RegClass, vti.RegClass>;
  }
}

multiclass VPatTernaryW_VX<string intrinsic, string instruction,
                           list<VTypeInfoToWide> vtilist> {
  foreach vtiToWti = vtilist in {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;
    defm : VPatTernaryWithPolicy<intrinsic, instruction,
                                 "V"#vti.ScalarSuffix,
                                 wti.Vector, vti.Scalar, vti.Vector,
                                 vti.Mask, vti.Log2SEW, vti.LMul,
                                 wti.RegClass, vti.ScalarRegClass, vti.RegClass>;
  }
}

multiclass VPatTernaryV_VV_VX_AAXA<string intrinsic, string instruction,
                              list<VTypeInfo> vtilist>
    : VPatTernaryV_VV_AAXA<intrinsic, instruction, vtilist>,
      VPatTernaryV_VX_AAXA<intrinsic, instruction, vtilist>;

multiclass VPatTernaryV_VX_VI<string intrinsic, string instruction,
                              list<VTypeInfo> vtilist, Operand Imm_type = simm5>
    : VPatTernaryV_VX<intrinsic, instruction, vtilist>,
      VPatTernaryV_VI<intrinsic, instruction, vtilist, Imm_type>;


multiclass VPatBinaryM_VV_VX_VI<string intrinsic, string instruction,
                                list<VTypeInfo> vtilist>
    : VPatBinaryM_VV<intrinsic, instruction, vtilist>,
      VPatBinaryM_VX<intrinsic, instruction, vtilist>,
      VPatBinaryM_VI<intrinsic, instruction, vtilist>;

multiclass VPatTernaryW_VV_VX<string intrinsic, string instruction,
                              list<VTypeInfoToWide> vtilist>
    : VPatTernaryW_VV<intrinsic, instruction, vtilist>,
      VPatTernaryW_VX<intrinsic, instruction, vtilist>;

multiclass VPatBinaryM_VV_VX<string intrinsic, string instruction,
                             list<VTypeInfo> vtilist>
    : VPatBinaryM_VV<intrinsic, instruction, vtilist>,
      VPatBinaryM_VX<intrinsic, instruction, vtilist>;

multiclass VPatBinaryM_VX_VI<string intrinsic, string instruction,
                             list<VTypeInfo> vtilist>
    : VPatBinaryM_VX<intrinsic, instruction, vtilist>,
      VPatBinaryM_VI<intrinsic, instruction, vtilist>;

multiclass VPatBinaryV_VV_VX_VI_INT<string intrinsic, string instruction,
                                    list<VTypeInfo> vtilist, Operand ImmType = simm5>
    : VPatBinaryV_VV_INT<intrinsic#"_vv", instruction, vtilist>,
      VPatBinaryV_VX_INT<intrinsic#"_vx", instruction, vtilist>,
      VPatBinaryV_VI<intrinsic#"_vx", instruction, vtilist, ImmType>;

multiclass VPatReductionV_VS<string intrinsic, string instruction, bit IsFloat = 0> {
  foreach vti = !if(IsFloat, NoGroupFloatVectors, NoGroupIntegerVectors) in
  {
    defvar vectorM1 = !cast<VTypeInfo>(!if(IsFloat, "VF", "VI") # vti.SEW # "M1");
    defm : VPatTernary<intrinsic, instruction, "VS",
                       vectorM1.Vector, vti.Vector,
                       vectorM1.Vector, vti.Mask,
                       vti.Log2SEW, vti.LMul,
                       VR, vti.RegClass, VR>;
  }
  foreach gvti = !if(IsFloat, GroupFloatVectors, GroupIntegerVectors) in
  {
    defm : VPatTernary<intrinsic, instruction, "VS",
                       gvti.VectorM1, gvti.Vector,
                       gvti.VectorM1, gvti.Mask,
                       gvti.Log2SEW, gvti.LMul,
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
                         vti.Log2SEW, vti.LMul,
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

    defm : VPatConversionTA<intrinsic, instruction, "V",
                            ivti.Vector, fvti.Vector, ivti.Mask, fvti.Log2SEW,
                            fvti.LMul, ivti.RegClass, fvti.RegClass>;
  }
}

multiclass VPatConversionVF_VI<string intrinsic,
                               string instruction>
{
  foreach fvti = AllFloatVectors in
  {
    defvar ivti = GetIntVTypeInfo<fvti>.Vti;

    defm : VPatConversionTA<intrinsic, instruction, "V",
                            fvti.Vector, ivti.Vector, fvti.Mask, ivti.Log2SEW,
                            ivti.LMul, fvti.RegClass, ivti.RegClass>;
  }
}

multiclass VPatConversionWI_VF<string intrinsic, string instruction> {
  foreach fvtiToFWti = AllWidenableFloatVectors in
  {
    defvar fvti = fvtiToFWti.Vti;
    defvar iwti = GetIntVTypeInfo<fvtiToFWti.Wti>.Vti;

    defm : VPatConversionTA<intrinsic, instruction, "V",
                            iwti.Vector, fvti.Vector, iwti.Mask, fvti.Log2SEW,
                            fvti.LMul, iwti.RegClass, fvti.RegClass>;
  }
}

multiclass VPatConversionWF_VI<string intrinsic, string instruction> {
  foreach vtiToWti = AllWidenableIntToFloatVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar fwti = vtiToWti.Wti;

    defm : VPatConversionTA<intrinsic, instruction, "V",
                            fwti.Vector, vti.Vector, fwti.Mask, vti.Log2SEW,
                            vti.LMul, fwti.RegClass, vti.RegClass>;
  }
}

multiclass VPatConversionWF_VF <string intrinsic, string instruction> {
  foreach fvtiToFWti = AllWidenableFloatVectors in
  {
    defvar fvti = fvtiToFWti.Vti;
    defvar fwti = fvtiToFWti.Wti;

    defm : VPatConversionTA<intrinsic, instruction, "V",
                            fwti.Vector, fvti.Vector, fwti.Mask, fvti.Log2SEW,
                            fvti.LMul, fwti.RegClass, fvti.RegClass>;
  }
}

multiclass VPatConversionVI_WF <string intrinsic, string instruction> {
  foreach vtiToWti = AllWidenableIntToFloatVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar fwti = vtiToWti.Wti;

    defm : VPatConversionTA<intrinsic, instruction, "W",
                            vti.Vector, fwti.Vector, vti.Mask, vti.Log2SEW,
                            vti.LMul, vti.RegClass, fwti.RegClass>;
  }
}

multiclass VPatConversionVF_WI <string intrinsic, string instruction> {
  foreach fvtiToFWti = AllWidenableFloatVectors in
  {
    defvar fvti = fvtiToFWti.Vti;
    defvar iwti = GetIntVTypeInfo<fvtiToFWti.Wti>.Vti;

    defm : VPatConversionTA<intrinsic, instruction, "W",
                            fvti.Vector, iwti.Vector, fvti.Mask, fvti.Log2SEW,
                            fvti.LMul, fvti.RegClass, iwti.RegClass>;
  }
}

multiclass VPatConversionVF_WF <string intrinsic, string instruction> {
  foreach fvtiToFWti = AllWidenableFloatVectors in
  {
    defvar fvti = fvtiToFWti.Vti;
    defvar fwti = fvtiToFWti.Wti;

    defm : VPatConversionTA<intrinsic, instruction, "W",
                            fvti.Vector, fwti.Vector, fvti.Mask, fvti.Log2SEW,
                            fvti.LMul, fvti.RegClass, fwti.RegClass>;
  }
}

multiclass VPatCompare_VI<string intrinsic, string inst,
                          ImmLeaf ImmType> {
  foreach vti = AllIntegerVectors in {
    defvar Intr = !cast<Intrinsic>(intrinsic);
    defvar Pseudo = !cast<Instruction>(inst#"_VI_"#vti.LMul.MX);
    def : Pat<(vti.Mask (Intr (vti.Vector vti.RegClass:$rs1),
                              (vti.Scalar ImmType:$rs2),
                              VLOpFrag)),
              (Pseudo vti.RegClass:$rs1, (DecImm ImmType:$rs2),
                      GPR:$vl, vti.Log2SEW)>;
    defvar IntrMask = !cast<Intrinsic>(intrinsic # "_mask");
    defvar PseudoMask = !cast<Instruction>(inst#"_VI_"#vti.LMul.MX#"_MASK");
    def : Pat<(vti.Mask (IntrMask (vti.Mask VR:$merge),
                                  (vti.Vector vti.RegClass:$rs1),
                                  (vti.Scalar ImmType:$rs2),
                                  (vti.Mask V0),
                                  VLOpFrag)),
              (PseudoMask VR:$merge, vti.RegClass:$rs1, (DecImm ImmType:$rs2),
                          (vti.Mask V0), GPR:$vl, vti.Log2SEW)>;
  }
}

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasVInstructions] in {

//===----------------------------------------------------------------------===//
// Pseudo Instructions for CodeGen
//===----------------------------------------------------------------------===//
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  def PseudoVMV1R_V : VPseudo<VMV1R_V, V_M1, (outs VR:$vd), (ins VR:$vs2)>,
                      VMVRSched<1>;
  def PseudoVMV2R_V : VPseudo<VMV2R_V, V_M2, (outs VRM2:$vd), (ins VRM2:$vs2)>,
                      VMVRSched<2>;
  def PseudoVMV4R_V : VPseudo<VMV4R_V, V_M4, (outs VRM4:$vd), (ins VRM4:$vs2)>,
                      VMVRSched<4>;
  def PseudoVMV8R_V : VPseudo<VMV8R_V, V_M8, (outs VRM8:$vd), (ins VRM8:$vs2)>,
                      VMVRSched<8>;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in {
  def PseudoReadVLENB : Pseudo<(outs GPR:$rd), (ins),
                               [(set GPR:$rd, (riscv_read_vlenb))]>,
                        Sched<[WriteRdVLENB]>;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1,
    Uses = [VL] in
def PseudoReadVL : Pseudo<(outs GPR:$rd), (ins), []>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1 in {
  def PseudoVSPILL_M1 : VPseudo<VS1R_V, V_M1, (outs), (ins VR:$rs1, GPR:$rs2)>;
  def PseudoVSPILL_M2 : VPseudo<VS2R_V, V_M2, (outs), (ins VRM2:$rs1, GPR:$rs2)>;
  def PseudoVSPILL_M4 : VPseudo<VS4R_V, V_M4, (outs), (ins VRM4:$rs1, GPR:$rs2)>;
  def PseudoVSPILL_M8 : VPseudo<VS8R_V, V_M8, (outs), (ins VRM8:$rs1, GPR:$rs2)>;
}

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1 in {
  def PseudoVRELOAD_M1 : VPseudo<VL1RE8_V, V_M1, (outs VR:$rs1), (ins GPR:$rs2)>;
  def PseudoVRELOAD_M2 : VPseudo<VL2RE8_V, V_M2, (outs VRM2:$rs1), (ins GPR:$rs2)>;
  def PseudoVRELOAD_M4 : VPseudo<VL4RE8_V, V_M4, (outs VRM4:$rs1), (ins GPR:$rs2)>;
  def PseudoVRELOAD_M8 : VPseudo<VL8RE8_V, V_M8, (outs VRM8:$rs1), (ins GPR:$rs2)>;
}

foreach lmul = MxList in {
  foreach nf = NFSet<lmul>.L in {
    defvar vreg = SegRegClass<lmul, nf>.RC;
    let hasSideEffects = 0, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1,
        Size = !mul(4, !sub(!mul(nf, 2), 1)) in {
      def "PseudoVSPILL" # nf # "_" # lmul.MX :
        Pseudo<(outs), (ins vreg:$rs1, GPR:$rs2, GPR:$vlenb), []>;
    }
    let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1,
        Size = !mul(4, !sub(!mul(nf, 2), 1)) in {
      def "PseudoVRELOAD" # nf # "_" # lmul.MX :
        Pseudo<(outs vreg:$rs1), (ins GPR:$rs2, GPR:$vlenb), []>;
    }
  }
}

//===----------------------------------------------------------------------===//
// 6. Configuration-Setting Instructions
//===----------------------------------------------------------------------===//

// Pseudos.
let hasSideEffects = 1, mayLoad = 0, mayStore = 0, Defs = [VL, VTYPE] in {
// Due to rs1=X0 having special meaning, we need a GPRNoX0 register class for
// the when we aren't using one of the special X0 encodings. Otherwise it could
// be accidentally be made X0 by MachineIR optimizations. To satisfy the
// verifier, we also need a GPRX0 instruction for the special encodings.
def PseudoVSETVLI : Pseudo<(outs GPR:$rd), (ins GPRNoX0:$rs1, VTypeIOp11:$vtypei), []>,
                    Sched<[WriteVSETVLI, ReadVSETVLI]>;
def PseudoVSETVLIX0 : Pseudo<(outs GPR:$rd), (ins GPRX0:$rs1, VTypeIOp11:$vtypei), []>,
                      Sched<[WriteVSETVLI, ReadVSETVLI]>;
def PseudoVSETIVLI : Pseudo<(outs GPR:$rd), (ins uimm5:$rs1, VTypeIOp10:$vtypei), []>,
                     Sched<[WriteVSETIVLI]>;
}

//===----------------------------------------------------------------------===//
// 7. Vector Loads and Stores
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 7.4 Vector Unit-Stride Instructions
//===----------------------------------------------------------------------===//

// Pseudos Unit-Stride Loads and Stores
defm PseudoVL : VPseudoUSLoad;
defm PseudoVS : VPseudoUSStore;

defm PseudoVLM : VPseudoLoadMask,
                 Sched<[WriteVLDM, ReadVLDX]>;
defm PseudoVSM : VPseudoStoreMask,
                 Sched<[WriteVSTM, ReadVSTX]>;

//===----------------------------------------------------------------------===//
// 7.5 Vector Strided Instructions
//===----------------------------------------------------------------------===//

// Vector Strided Loads and Stores
defm PseudoVLS : VPseudoSLoad;
defm PseudoVSS : VPseudoSStore;

//===----------------------------------------------------------------------===//
// 7.6 Vector Indexed Instructions
//===----------------------------------------------------------------------===//

// Vector Indexed Loads and Stores
defm PseudoVLUX : VPseudoILoad</*Ordered=*/false>;
defm PseudoVLOX : VPseudoILoad</*Ordered=*/true>;
defm PseudoVSOX : VPseudoIStore</*Ordered=*/true>;
defm PseudoVSUX : VPseudoIStore</*Ordered=*/false>;

//===----------------------------------------------------------------------===//
// 7.7. Unit-stride Fault-Only-First Loads
//===----------------------------------------------------------------------===//

// vleff may update VL register
let hasSideEffects = 1, Defs = [VL] in
defm PseudoVL : VPseudoFFLoad;

//===----------------------------------------------------------------------===//
// 7.8. Vector Load/Store Segment Instructions
//===----------------------------------------------------------------------===//
defm PseudoVLSEG : VPseudoUSSegLoad;
defm PseudoVLSSEG : VPseudoSSegLoad;
defm PseudoVLOXSEG : VPseudoISegLoad</*Ordered=*/true>;
defm PseudoVLUXSEG : VPseudoISegLoad</*Ordered=*/false>;
defm PseudoVSSEG : VPseudoUSSegStore;
defm PseudoVSSSEG : VPseudoSSegStore;
defm PseudoVSOXSEG : VPseudoISegStore</*Ordered=*/true>;
defm PseudoVSUXSEG : VPseudoISegStore</*Ordered=*/false>;

// vlseg<nf>e<eew>ff.v may update VL register
let hasSideEffects = 1, Defs = [VL] in {
defm PseudoVLSEG : VPseudoUSSegLoadFF;
}

//===----------------------------------------------------------------------===//
// 12. Vector Integer Arithmetic Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 12.1. Vector Single-Width Integer Add and Subtract
//===----------------------------------------------------------------------===//
defm PseudoVADD   : VPseudoVALU_VV_VX_VI;
defm PseudoVSUB   : VPseudoVALU_VV_VX;
defm PseudoVRSUB  : VPseudoVALU_VX_VI;

foreach vti = AllIntegerVectors in {
  // Match vrsub with 2 vector operands to vsub.vv by swapping operands. This
  // Occurs when legalizing vrsub.vx intrinsics for i64 on RV32 since we need
  // to use a more complex splat sequence. Add the pattern for all VTs for
  // consistency.
  def : Pat<(vti.Vector (int_riscv_vrsub (vti.Vector (undef)),
                                         (vti.Vector vti.RegClass:$rs2),
                                         (vti.Vector vti.RegClass:$rs1),
                                         VLOpFrag)),
            (!cast<Instruction>("PseudoVSUB_VV_"#vti.LMul.MX) vti.RegClass:$rs1,
                                                              vti.RegClass:$rs2,
                                                              GPR:$vl,
                                                              vti.Log2SEW)>;
  def : Pat<(vti.Vector (int_riscv_vrsub (vti.Vector vti.RegClass:$merge),
                                         (vti.Vector vti.RegClass:$rs2),
                                         (vti.Vector vti.RegClass:$rs1),
                                         VLOpFrag)),
            (!cast<Instruction>("PseudoVSUB_VV_"#vti.LMul.MX#"_TU")
                                                      vti.RegClass:$merge,
                                                      vti.RegClass:$rs1,
                                                      vti.RegClass:$rs2,
                                                      GPR:$vl,
                                                      vti.Log2SEW)>;
  def : Pat<(vti.Vector (int_riscv_vrsub_mask (vti.Vector vti.RegClass:$merge),
                                              (vti.Vector vti.RegClass:$rs2),
                                              (vti.Vector vti.RegClass:$rs1),
                                              (vti.Mask V0),
                                              VLOpFrag,
                                              (XLenVT timm:$policy))),
            (!cast<Instruction>("PseudoVSUB_VV_"#vti.LMul.MX#"_MASK")
                                                      vti.RegClass:$merge,
                                                      vti.RegClass:$rs1,
                                                      vti.RegClass:$rs2,
                                                      (vti.Mask V0),
                                                      GPR:$vl,
                                                      vti.Log2SEW,
                                                      (XLenVT timm:$policy))>;

  // Match VSUB with a small immediate to vadd.vi by negating the immediate.
  def : Pat<(vti.Vector (int_riscv_vsub (vti.Vector (undef)),
                                        (vti.Vector vti.RegClass:$rs1),
                                        (vti.Scalar simm5_plus1:$rs2),
                                        VLOpFrag)),
            (!cast<Instruction>("PseudoVADD_VI_"#vti.LMul.MX) vti.RegClass:$rs1,
                                                              (NegImm simm5_plus1:$rs2),
                                                              GPR:$vl,
                                                              vti.Log2SEW)>;
  def : Pat<(vti.Vector (int_riscv_vsub_mask (vti.Vector vti.RegClass:$merge),
                                             (vti.Vector vti.RegClass:$rs1),
                                             (vti.Scalar simm5_plus1:$rs2),
                                             (vti.Mask V0),
                                             VLOpFrag,
                                             (XLenVT timm:$policy))),
            (!cast<Instruction>("PseudoVADD_VI_"#vti.LMul.MX#"_MASK")
                                                      vti.RegClass:$merge,
                                                      vti.RegClass:$rs1,
                                                      (NegImm simm5_plus1:$rs2),
                                                      (vti.Mask V0),
                                                      GPR:$vl,
                                                      vti.Log2SEW,
                                                      (XLenVT timm:$policy))>;
}

//===----------------------------------------------------------------------===//
// 12.2. Vector Widening Integer Add/Subtract
//===----------------------------------------------------------------------===//
defm PseudoVWADDU : VPseudoVWALU_VV_VX;
defm PseudoVWSUBU : VPseudoVWALU_VV_VX;
defm PseudoVWADD  : VPseudoVWALU_VV_VX;
defm PseudoVWSUB  : VPseudoVWALU_VV_VX;
defm PseudoVWADDU : VPseudoVWALU_WV_WX;
defm PseudoVWSUBU : VPseudoVWALU_WV_WX;
defm PseudoVWADD  : VPseudoVWALU_WV_WX;
defm PseudoVWSUB  : VPseudoVWALU_WV_WX;

//===----------------------------------------------------------------------===//
// 12.3. Vector Integer Extension
//===----------------------------------------------------------------------===//
defm PseudoVZEXT_VF2 : PseudoVEXT_VF2;
defm PseudoVZEXT_VF4 : PseudoVEXT_VF4;
defm PseudoVZEXT_VF8 : PseudoVEXT_VF8;
defm PseudoVSEXT_VF2 : PseudoVEXT_VF2;
defm PseudoVSEXT_VF4 : PseudoVEXT_VF4;
defm PseudoVSEXT_VF8 : PseudoVEXT_VF8;

//===----------------------------------------------------------------------===//
// 12.4. Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
//===----------------------------------------------------------------------===//
defm PseudoVADC  : VPseudoVCALU_VM_XM_IM;
defm PseudoVMADC : VPseudoVCALUM_VM_XM_IM<"@earlyclobber $rd">;
defm PseudoVMADC : VPseudoVCALUM_V_X_I<"@earlyclobber $rd">;

defm PseudoVSBC  : VPseudoVCALU_VM_XM;
defm PseudoVMSBC : VPseudoVCALUM_VM_XM<"@earlyclobber $rd">;
defm PseudoVMSBC : VPseudoVCALUM_V_X<"@earlyclobber $rd">;

//===----------------------------------------------------------------------===//
// 12.5. Vector Bitwise Logical Instructions
//===----------------------------------------------------------------------===//
defm PseudoVAND : VPseudoVALU_VV_VX_VI;
defm PseudoVOR  : VPseudoVALU_VV_VX_VI;
defm PseudoVXOR : VPseudoVALU_VV_VX_VI;

//===----------------------------------------------------------------------===//
// 12.6. Vector Single-Width Bit Shift Instructions
//===----------------------------------------------------------------------===//
defm PseudoVSLL : VPseudoVSHT_VV_VX_VI<uimm5>;
defm PseudoVSRL : VPseudoVSHT_VV_VX_VI<uimm5>;
defm PseudoVSRA : VPseudoVSHT_VV_VX_VI<uimm5>;

//===----------------------------------------------------------------------===//
// 12.7. Vector Narrowing Integer Right Shift Instructions
//===----------------------------------------------------------------------===//
defm PseudoVNSRL : VPseudoVNSHT_WV_WX_WI;
defm PseudoVNSRA : VPseudoVNSHT_WV_WX_WI;

//===----------------------------------------------------------------------===//
// 12.8. Vector Integer Comparison Instructions
//===----------------------------------------------------------------------===//
defm PseudoVMSEQ  : VPseudoVCMPM_VV_VX_VI;
defm PseudoVMSNE  : VPseudoVCMPM_VV_VX_VI;
defm PseudoVMSLTU : VPseudoVCMPM_VV_VX;
defm PseudoVMSLT  : VPseudoVCMPM_VV_VX;
defm PseudoVMSLEU : VPseudoVCMPM_VV_VX_VI;
defm PseudoVMSLE  : VPseudoVCMPM_VV_VX_VI;
defm PseudoVMSGTU : VPseudoVCMPM_VX_VI;
defm PseudoVMSGT  : VPseudoVCMPM_VX_VI;

//===----------------------------------------------------------------------===//
// 12.9. Vector Integer Min/Max Instructions
//===----------------------------------------------------------------------===//
defm PseudoVMINU : VPseudoVMINMAX_VV_VX;
defm PseudoVMIN  : VPseudoVMINMAX_VV_VX;
defm PseudoVMAXU : VPseudoVMINMAX_VV_VX;
defm PseudoVMAX  : VPseudoVMINMAX_VV_VX;

//===----------------------------------------------------------------------===//
// 12.10. Vector Single-Width Integer Multiply Instructions
//===----------------------------------------------------------------------===//
defm PseudoVMUL    : VPseudoVMUL_VV_VX;
defm PseudoVMULH   : VPseudoVMUL_VV_VX;
defm PseudoVMULHU  : VPseudoVMUL_VV_VX;
defm PseudoVMULHSU : VPseudoVMUL_VV_VX;

//===----------------------------------------------------------------------===//
// 12.11. Vector Integer Divide Instructions
//===----------------------------------------------------------------------===//
defm PseudoVDIVU : VPseudoVDIV_VV_VX;
defm PseudoVDIV  : VPseudoVDIV_VV_VX;
defm PseudoVREMU : VPseudoVDIV_VV_VX;
defm PseudoVREM  : VPseudoVDIV_VV_VX;

//===----------------------------------------------------------------------===//
// 12.12. Vector Widening Integer Multiply Instructions
//===----------------------------------------------------------------------===//
defm PseudoVWMUL   : VPseudoVWMUL_VV_VX;
defm PseudoVWMULU  : VPseudoVWMUL_VV_VX;
defm PseudoVWMULSU : VPseudoVWMUL_VV_VX;

//===----------------------------------------------------------------------===//
// 12.13. Vector Single-Width Integer Multiply-Add Instructions
//===----------------------------------------------------------------------===//
defm PseudoVMACC  : VPseudoVMAC_VV_VX_AAXA;
defm PseudoVNMSAC : VPseudoVMAC_VV_VX_AAXA;
defm PseudoVMADD  : VPseudoVMAC_VV_VX_AAXA;
defm PseudoVNMSUB : VPseudoVMAC_VV_VX_AAXA;

//===----------------------------------------------------------------------===//
// 12.14. Vector Widening Integer Multiply-Add Instructions
//===----------------------------------------------------------------------===//
defm PseudoVWMACCU  : VPseudoVWMAC_VV_VX;
defm PseudoVWMACC   : VPseudoVWMAC_VV_VX;
defm PseudoVWMACCSU : VPseudoVWMAC_VV_VX;
defm PseudoVWMACCUS : VPseudoVWMAC_VX;

//===----------------------------------------------------------------------===//
// 12.15. Vector Integer Merge Instructions
//===----------------------------------------------------------------------===//
defm PseudoVMERGE : VPseudoVMRG_VM_XM_IM;

//===----------------------------------------------------------------------===//
// 12.16. Vector Integer Move Instructions
//===----------------------------------------------------------------------===//
defm PseudoVMV_V : VPseudoUnaryVMV_V_X_I;

//===----------------------------------------------------------------------===//
// 13.1. Vector Single-Width Saturating Add and Subtract
//===----------------------------------------------------------------------===//
let Defs = [VXSAT], hasSideEffects = 1 in {
  defm PseudoVSADDU : VPseudoVSALU_VV_VX_VI;
  defm PseudoVSADD  : VPseudoVSALU_VV_VX_VI;
  defm PseudoVSSUBU : VPseudoVSALU_VV_VX;
  defm PseudoVSSUB  : VPseudoVSALU_VV_VX;
}

//===----------------------------------------------------------------------===//
// 13.2. Vector Single-Width Averaging Add and Subtract
//===----------------------------------------------------------------------===//
let Uses = [VXRM], hasSideEffects = 1 in {
  defm PseudoVAADDU : VPseudoVAALU_VV_VX;
  defm PseudoVAADD  : VPseudoVAALU_VV_VX;
  defm PseudoVASUBU : VPseudoVAALU_VV_VX;
  defm PseudoVASUB  : VPseudoVAALU_VV_VX;
}

//===----------------------------------------------------------------------===//
// 13.3. Vector Single-Width Fractional Multiply with Rounding and Saturation
//===----------------------------------------------------------------------===//
let Uses = [VXRM], Defs = [VXSAT], hasSideEffects = 1 in {
  defm PseudoVSMUL : VPseudoVSMUL_VV_VX;
}

//===----------------------------------------------------------------------===//
// 13.4. Vector Single-Width Scaling Shift Instructions
//===----------------------------------------------------------------------===//
let Uses = [VXRM], hasSideEffects = 1 in {
  defm PseudoVSSRL : VPseudoVSSHT_VV_VX_VI<uimm5>;
  defm PseudoVSSRA : VPseudoVSSHT_VV_VX_VI<uimm5>;
}

//===----------------------------------------------------------------------===//
// 13.5. Vector Narrowing Fixed-Point Clip Instructions
//===----------------------------------------------------------------------===//
let Uses = [VXRM], Defs = [VXSAT], hasSideEffects = 1 in {
  defm PseudoVNCLIP  : VPseudoVNCLP_WV_WX_WI;
  defm PseudoVNCLIPU : VPseudoVNCLP_WV_WX_WI;
}

} // Predicates = [HasVInstructions]

let Predicates = [HasVInstructionsAnyF] in {
//===----------------------------------------------------------------------===//
// 14.2. Vector Single-Width Floating-Point Add/Subtract Instructions
//===----------------------------------------------------------------------===//
let Uses = [FRM], mayRaiseFPException = true in {
defm PseudoVFADD  : VPseudoVALU_VV_VF;
defm PseudoVFSUB  : VPseudoVALU_VV_VF;
defm PseudoVFRSUB : VPseudoVALU_VF;
}

//===----------------------------------------------------------------------===//
// 14.3. Vector Widening Floating-Point Add/Subtract Instructions
//===----------------------------------------------------------------------===//
let Uses = [FRM], mayRaiseFPException = true in {
defm PseudoVFWADD : VPseudoVFWALU_VV_VF;
defm PseudoVFWSUB : VPseudoVFWALU_VV_VF;
defm PseudoVFWADD : VPseudoVFWALU_WV_WF;
defm PseudoVFWSUB : VPseudoVFWALU_WV_WF;
}

//===----------------------------------------------------------------------===//
// 14.4. Vector Single-Width Floating-Point Multiply/Divide Instructions
//===----------------------------------------------------------------------===//
let Uses = [FRM], mayRaiseFPException = true in {
defm PseudoVFMUL  : VPseudoVFMUL_VV_VF;
defm PseudoVFDIV  : VPseudoVFDIV_VV_VF;
defm PseudoVFRDIV : VPseudoVFRDIV_VF;
}

//===----------------------------------------------------------------------===//
// 14.5. Vector Widening Floating-Point Multiply
//===----------------------------------------------------------------------===//
let Uses = [FRM], mayRaiseFPException = true in {
defm PseudoVFWMUL : VPseudoVWMUL_VV_VF;
}

//===----------------------------------------------------------------------===//
// 14.6. Vector Single-Width Floating-Point Fused Multiply-Add Instructions
//===----------------------------------------------------------------------===//
let Uses = [FRM], mayRaiseFPException = true in {
defm PseudoVFMACC  : VPseudoVMAC_VV_VF_AAXA;
defm PseudoVFNMACC : VPseudoVMAC_VV_VF_AAXA;
defm PseudoVFMSAC  : VPseudoVMAC_VV_VF_AAXA;
defm PseudoVFNMSAC : VPseudoVMAC_VV_VF_AAXA;
defm PseudoVFMADD  : VPseudoVMAC_VV_VF_AAXA;
defm PseudoVFNMADD : VPseudoVMAC_VV_VF_AAXA;
defm PseudoVFMSUB  : VPseudoVMAC_VV_VF_AAXA;
defm PseudoVFNMSUB : VPseudoVMAC_VV_VF_AAXA;
}

//===----------------------------------------------------------------------===//
// 14.7. Vector Widening Floating-Point Fused Multiply-Add Instructions
//===----------------------------------------------------------------------===//
let Uses = [FRM], mayRaiseFPException = true in {
defm PseudoVFWMACC  : VPseudoVWMAC_VV_VF;
defm PseudoVFWNMACC : VPseudoVWMAC_VV_VF;
defm PseudoVFWMSAC  : VPseudoVWMAC_VV_VF;
defm PseudoVFWNMSAC : VPseudoVWMAC_VV_VF;
}

//===----------------------------------------------------------------------===//
// 14.8. Vector Floating-Point Square-Root Instruction
//===----------------------------------------------------------------------===//
let Uses = [FRM], mayRaiseFPException = true in
defm PseudoVFSQRT : VPseudoVSQR_V;

//===----------------------------------------------------------------------===//
// 14.9. Vector Floating-Point Reciprocal Square-Root Estimate Instruction
//===----------------------------------------------------------------------===//
let mayRaiseFPException = true in
defm PseudoVFRSQRT7 : VPseudoVRCP_V;

//===----------------------------------------------------------------------===//
// 14.10. Vector Floating-Point Reciprocal Estimate Instruction
//===----------------------------------------------------------------------===//
let Uses = [FRM], mayRaiseFPException = true in
defm PseudoVFREC7 : VPseudoVRCP_V;

//===----------------------------------------------------------------------===//
// 14.11. Vector Floating-Point Min/Max Instructions
//===----------------------------------------------------------------------===//
let mayRaiseFPException = true in {
defm PseudoVFMIN : VPseudoVMAX_VV_VF;
defm PseudoVFMAX : VPseudoVMAX_VV_VF;
}

//===----------------------------------------------------------------------===//
// 14.12. Vector Floating-Point Sign-Injection Instructions
//===----------------------------------------------------------------------===//
defm PseudoVFSGNJ  : VPseudoVSGNJ_VV_VF;
defm PseudoVFSGNJN : VPseudoVSGNJ_VV_VF;
defm PseudoVFSGNJX : VPseudoVSGNJ_VV_VF;

//===----------------------------------------------------------------------===//
// 14.13. Vector Floating-Point Compare Instructions
//===----------------------------------------------------------------------===//
let mayRaiseFPException = true in {
defm PseudoVMFEQ : VPseudoVCMPM_VV_VF;
defm PseudoVMFNE : VPseudoVCMPM_VV_VF;
defm PseudoVMFLT : VPseudoVCMPM_VV_VF;
defm PseudoVMFLE : VPseudoVCMPM_VV_VF;
defm PseudoVMFGT : VPseudoVCMPM_VF;
defm PseudoVMFGE : VPseudoVCMPM_VF;
}

//===----------------------------------------------------------------------===//
// 14.14. Vector Floating-Point Classify Instruction
//===----------------------------------------------------------------------===//
defm PseudoVFCLASS : VPseudoVCLS_V;

//===----------------------------------------------------------------------===//
// 14.15. Vector Floating-Point Merge Instruction
//===----------------------------------------------------------------------===//
defm PseudoVFMERGE : VPseudoVMRG_FM;

//===----------------------------------------------------------------------===//
// 14.16. Vector Floating-Point Move Instruction
//===----------------------------------------------------------------------===//
defm PseudoVFMV_V : VPseudoVMV_F;

//===----------------------------------------------------------------------===//
// 14.17. Single-Width Floating-Point/Integer Type-Convert Instructions
//===----------------------------------------------------------------------===//
let mayRaiseFPException = true in {
let Uses = [FRM] in {
defm PseudoVFCVT_XU_F : VPseudoVCVTI_V;
defm PseudoVFCVT_X_F : VPseudoVCVTI_V;
}
defm PseudoVFCVT_RTZ_XU_F : VPseudoVCVTI_V;
defm PseudoVFCVT_RTZ_X_F : VPseudoVCVTI_V;
defm PseudoVFCVT_RM_X_F : VPseudoVCVTI_RM_V;
defm PseudoVFROUND_NOEXCEPT : VPseudoVFROUND_NOEXCEPT_V;
let Uses = [FRM] in {
defm PseudoVFCVT_F_XU : VPseudoVCVTF_V;
defm PseudoVFCVT_F_X : VPseudoVCVTF_V;
}
} // mayRaiseFPException = true

//===----------------------------------------------------------------------===//
// 14.18. Widening Floating-Point/Integer Type-Convert Instructions
//===----------------------------------------------------------------------===//
let mayRaiseFPException = true in {
let Uses = [FRM] in {
defm PseudoVFWCVT_XU_F     : VPseudoVWCVTI_V;
defm PseudoVFWCVT_X_F      : VPseudoVWCVTI_V;
}
defm PseudoVFWCVT_RTZ_XU_F : VPseudoVWCVTI_V;
defm PseudoVFWCVT_RTZ_X_F  : VPseudoVWCVTI_V;
defm PseudoVFWCVT_F_XU     : VPseudoVWCVTF_V;
defm PseudoVFWCVT_F_X      : VPseudoVWCVTF_V;
defm PseudoVFWCVT_F_F      : VPseudoVWCVTD_V;
} // mayRaiseFPException = true

//===----------------------------------------------------------------------===//
// 14.19. Narrowing Floating-Point/Integer Type-Convert Instructions
//===----------------------------------------------------------------------===//
let mayRaiseFPException = true in {
let Uses = [FRM] in {
defm PseudoVFNCVT_XU_F     : VPseudoVNCVTI_W;
defm PseudoVFNCVT_X_F      : VPseudoVNCVTI_W;
}
defm PseudoVFNCVT_RTZ_XU_F : VPseudoVNCVTI_W;
defm PseudoVFNCVT_RTZ_X_F  : VPseudoVNCVTI_W;
let Uses = [FRM] in {
defm PseudoVFNCVT_F_XU     : VPseudoVNCVTF_W;
defm PseudoVFNCVT_F_X      : VPseudoVNCVTF_W;
defm PseudoVFNCVT_F_F      : VPseudoVNCVTD_W;
}
defm PseudoVFNCVT_ROD_F_F  : VPseudoVNCVTD_W;
} // mayRaiseFPException = true
} // Predicates = [HasVInstructionsAnyF]

let Predicates = [HasVInstructions] in {
//===----------------------------------------------------------------------===//
// 15.1. Vector Single-Width Integer Reduction Instructions
//===----------------------------------------------------------------------===//
defm PseudoVREDSUM  : VPseudoVRED_VS;
defm PseudoVREDAND  : VPseudoVRED_VS;
defm PseudoVREDOR   : VPseudoVRED_VS;
defm PseudoVREDXOR  : VPseudoVRED_VS;
defm PseudoVREDMINU : VPseudoVRED_VS;
defm PseudoVREDMIN  : VPseudoVRED_VS;
defm PseudoVREDMAXU : VPseudoVRED_VS;
defm PseudoVREDMAX  : VPseudoVRED_VS;

//===----------------------------------------------------------------------===//
// 15.2. Vector Widening Integer Reduction Instructions
//===----------------------------------------------------------------------===//
let IsRVVWideningReduction = 1 in {
defm PseudoVWREDSUMU   : VPseudoVWRED_VS;
defm PseudoVWREDSUM    : VPseudoVWRED_VS;
}
} // Predicates = [HasVInstructions]

let Predicates = [HasVInstructionsAnyF] in {
//===----------------------------------------------------------------------===//
// 15.3. Vector Single-Width Floating-Point Reduction Instructions
//===----------------------------------------------------------------------===//
let Uses = [FRM], mayRaiseFPException = true in {
defm PseudoVFREDOSUM : VPseudoVFREDO_VS;
defm PseudoVFREDUSUM : VPseudoVFRED_VS;
}
let mayRaiseFPException = true in {
defm PseudoVFREDMIN  : VPseudoVFRED_VS;
defm PseudoVFREDMAX  : VPseudoVFRED_VS;
}

//===----------------------------------------------------------------------===//
// 15.4. Vector Widening Floating-Point Reduction Instructions
//===----------------------------------------------------------------------===//
let IsRVVWideningReduction = 1,
    Uses = [FRM],
    mayRaiseFPException = true in {
defm PseudoVFWREDUSUM  : VPseudoVFWRED_VS;
defm PseudoVFWREDOSUM  : VPseudoVFWRED_VS;
}

} // Predicates = [HasVInstructionsAnyF]

//===----------------------------------------------------------------------===//
// 16. Vector Mask Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 16.1 Vector Mask-Register Logical Instructions
//===----------------------------------------------------------------------===//

defm PseudoVMAND: VPseudoVALU_MM;
defm PseudoVMNAND: VPseudoVALU_MM;
defm PseudoVMANDN: VPseudoVALU_MM;
defm PseudoVMXOR: VPseudoVALU_MM;
defm PseudoVMOR: VPseudoVALU_MM;
defm PseudoVMNOR: VPseudoVALU_MM;
defm PseudoVMORN: VPseudoVALU_MM;
defm PseudoVMXNOR: VPseudoVALU_MM;

// Pseudo instructions
defm PseudoVMCLR : VPseudoNullaryPseudoM<"VMXOR">,
                   Sched<[WriteVMALUV, ReadVMALUV, ReadVMALUV]>;
defm PseudoVMSET : VPseudoNullaryPseudoM<"VMXNOR">,
                   Sched<[WriteVMALUV, ReadVMALUV, ReadVMALUV]>;

//===----------------------------------------------------------------------===//
// 16.2. Vector mask population count vcpop
//===----------------------------------------------------------------------===//

defm PseudoVCPOP: VPseudoVPOP_M;

//===----------------------------------------------------------------------===//
// 16.3. vfirst find-first-set mask bit
//===----------------------------------------------------------------------===//

defm PseudoVFIRST: VPseudoV1ST_M;

//===----------------------------------------------------------------------===//
// 16.4. vmsbf.m set-before-first mask bit
//===----------------------------------------------------------------------===//
defm PseudoVMSBF: VPseudoVSFS_M;

//===----------------------------------------------------------------------===//
// 16.5. vmsif.m set-including-first mask bit
//===----------------------------------------------------------------------===//
defm PseudoVMSIF: VPseudoVSFS_M;

//===----------------------------------------------------------------------===//
// 16.6. vmsof.m set-only-first mask bit
//===----------------------------------------------------------------------===//
defm PseudoVMSOF: VPseudoVSFS_M;

//===----------------------------------------------------------------------===//
// 16.8.  Vector Iota Instruction
//===----------------------------------------------------------------------===//
defm PseudoVIOTA_M: VPseudoVIOT_M;

//===----------------------------------------------------------------------===//
// 16.9. Vector Element Index Instruction
//===----------------------------------------------------------------------===//
defm PseudoVID : VPseudoVID_V;

//===----------------------------------------------------------------------===//
// 17. Vector Permutation Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 17.1. Integer Scalar Move Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasVInstructions] in {
let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in {
  foreach m = MxList in {
    let VLMul = m.value in {
      let HasSEWOp = 1, BaseInstr = VMV_X_S in
      def PseudoVMV_X_S # "_" # m.MX:
        Pseudo<(outs GPR:$rd), (ins m.vrclass:$rs2, ixlenimm:$sew), []>,
        Sched<[WriteVIMovVX, ReadVIMovVX]>,
        RISCVVPseudo;
      let HasVLOp = 1, HasSEWOp = 1, BaseInstr = VMV_S_X,
          Constraints = "$rd = $rs1" in
      def PseudoVMV_S_X # "_" # m.MX: Pseudo<(outs m.vrclass:$rd),
                                             (ins m.vrclass:$rs1, GPR:$rs2,
                                                  AVL:$vl, ixlenimm:$sew),
                                             []>,
        Sched<[WriteVIMovXV, ReadVIMovXV, ReadVIMovXX]>,
        RISCVVPseudo;
    }
  }
}
} // Predicates = [HasVInstructions]

//===----------------------------------------------------------------------===//
// 17.2. Floating-Point Scalar Move Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasVInstructionsAnyF] in {
let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in {
  foreach f = FPList in {
    foreach m = f.MxList in {
      let VLMul = m.value in {
        let HasSEWOp = 1, BaseInstr = VFMV_F_S in
        def "PseudoVFMV_" # f.FX # "_S_" # m.MX :
          Pseudo<(outs f.fprclass:$rd),
                 (ins m.vrclass:$rs2, ixlenimm:$sew), []>,
          Sched<[WriteVFMovVF, ReadVFMovVF]>,
          RISCVVPseudo;
        let HasVLOp = 1, HasSEWOp = 1, BaseInstr = VFMV_S_F,
            Constraints = "$rd = $rs1" in
        def "PseudoVFMV_S_" # f.FX # "_" # m.MX :
                                          Pseudo<(outs m.vrclass:$rd),
                                                 (ins m.vrclass:$rs1, f.fprclass:$rs2,
                                                      AVL:$vl, ixlenimm:$sew),
                                                 []>,
          Sched<[WriteVFMovFV, ReadVFMovFV, ReadVFMovFX]>,
          RISCVVPseudo;
      }
    }
  }
}
} // Predicates = [HasVInstructionsAnyF]

//===----------------------------------------------------------------------===//
// 17.3. Vector Slide Instructions
//===----------------------------------------------------------------------===//
let Predicates = [HasVInstructions] in {
  defm PseudoVSLIDEUP    : VPseudoVSLD_VX_VI<uimm5, "@earlyclobber $rd">;
  defm PseudoVSLIDEDOWN  : VPseudoVSLD_VX_VI<uimm5>;
  defm PseudoVSLIDE1UP   : VPseudoVSLD1_VX<"@earlyclobber $rd">;
  defm PseudoVSLIDE1DOWN : VPseudoVSLD1_VX;
} // Predicates = [HasVInstructions]

let Predicates = [HasVInstructionsAnyF] in {
  defm PseudoVFSLIDE1UP  : VPseudoVSLD1_VF<"@earlyclobber $rd">;
  defm PseudoVFSLIDE1DOWN : VPseudoVSLD1_VF;
} // Predicates = [HasVInstructionsAnyF]

//===----------------------------------------------------------------------===//
// 17.4. Vector Register Gather Instructions
//===----------------------------------------------------------------------===//
defm PseudoVRGATHER     : VPseudoVGTR_VV_VX_VI<uimm5, "@earlyclobber $rd">;
defm PseudoVRGATHEREI16 : VPseudoVGTR_VV_EEW</* eew */ 16, "@earlyclobber $rd">;

//===----------------------------------------------------------------------===//
// 17.5. Vector Compress Instruction
//===----------------------------------------------------------------------===//
defm PseudoVCOMPRESS : VPseudoVCPR_V;

//===----------------------------------------------------------------------===//
// Patterns.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 12. Vector Integer Arithmetic Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasVInstructions] in {
//===----------------------------------------------------------------------===//
// 12.1. Vector Single-Width Integer Add and Subtract
//===----------------------------------------------------------------------===//
defm : VPatBinaryV_VV_VX_VI<"int_riscv_vadd", "PseudoVADD", AllIntegerVectors>;
defm : VPatBinaryV_VV_VX<"int_riscv_vsub", "PseudoVSUB", AllIntegerVectors>;
defm : VPatBinaryV_VX_VI<"int_riscv_vrsub", "PseudoVRSUB", AllIntegerVectors>;

//===----------------------------------------------------------------------===//
// 12.2. Vector Widening Integer Add/Subtract
//===----------------------------------------------------------------------===//
defm : VPatBinaryW_VV_VX<"int_riscv_vwaddu", "PseudoVWADDU", AllWidenableIntVectors>;
defm : VPatBinaryW_VV_VX<"int_riscv_vwsubu", "PseudoVWSUBU", AllWidenableIntVectors>;
defm : VPatBinaryW_VV_VX<"int_riscv_vwadd", "PseudoVWADD", AllWidenableIntVectors>;
defm : VPatBinaryW_VV_VX<"int_riscv_vwsub", "PseudoVWSUB", AllWidenableIntVectors>;
defm : VPatBinaryW_WV_WX<"int_riscv_vwaddu_w", "PseudoVWADDU", AllWidenableIntVectors>;
defm : VPatBinaryW_WV_WX<"int_riscv_vwsubu_w", "PseudoVWSUBU", AllWidenableIntVectors>;
defm : VPatBinaryW_WV_WX<"int_riscv_vwadd_w", "PseudoVWADD", AllWidenableIntVectors>;
defm : VPatBinaryW_WV_WX<"int_riscv_vwsub_w", "PseudoVWSUB", AllWidenableIntVectors>;

//===----------------------------------------------------------------------===//
// 12.3. Vector Integer Extension
//===----------------------------------------------------------------------===//
defm : VPatUnaryV_VF<"int_riscv_vzext", "PseudoVZEXT", "VF2",
                     AllFractionableVF2IntVectors>;
defm : VPatUnaryV_VF<"int_riscv_vzext", "PseudoVZEXT", "VF4",
                     AllFractionableVF4IntVectors>;
defm : VPatUnaryV_VF<"int_riscv_vzext", "PseudoVZEXT", "VF8",
                     AllFractionableVF8IntVectors>;
defm : VPatUnaryV_VF<"int_riscv_vsext", "PseudoVSEXT", "VF2",
                     AllFractionableVF2IntVectors>;
defm : VPatUnaryV_VF<"int_riscv_vsext", "PseudoVSEXT", "VF4",
                     AllFractionableVF4IntVectors>;
defm : VPatUnaryV_VF<"int_riscv_vsext", "PseudoVSEXT", "VF8",
                     AllFractionableVF8IntVectors>;

//===----------------------------------------------------------------------===//
// 12.4. Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
//===----------------------------------------------------------------------===//
defm : VPatBinaryV_VM_XM_IM<"int_riscv_vadc", "PseudoVADC">;
defm : VPatBinaryM_VM_XM_IM<"int_riscv_vmadc_carry_in", "PseudoVMADC">;
defm : VPatBinaryM_V_X_I<"int_riscv_vmadc", "PseudoVMADC">;

defm : VPatBinaryV_VM_XM<"int_riscv_vsbc", "PseudoVSBC">;
defm : VPatBinaryM_VM_XM<"int_riscv_vmsbc_borrow_in", "PseudoVMSBC">;
defm : VPatBinaryM_V_X<"int_riscv_vmsbc", "PseudoVMSBC">;

//===----------------------------------------------------------------------===//
// 12.5. Vector Bitwise Logical Instructions
//===----------------------------------------------------------------------===//
defm : VPatBinaryV_VV_VX_VI<"int_riscv_vand", "PseudoVAND", AllIntegerVectors>;
defm : VPatBinaryV_VV_VX_VI<"int_riscv_vor", "PseudoVOR", AllIntegerVectors>;
defm : VPatBinaryV_VV_VX_VI<"int_riscv_vxor", "PseudoVXOR", AllIntegerVectors>;

//===----------------------------------------------------------------------===//
// 12.6. Vector Single-Width Bit Shift Instructions
//===----------------------------------------------------------------------===//
defm : VPatBinaryV_VV_VX_VI<"int_riscv_vsll", "PseudoVSLL", AllIntegerVectors,
                            uimm5>;
defm : VPatBinaryV_VV_VX_VI<"int_riscv_vsrl", "PseudoVSRL", AllIntegerVectors,
                            uimm5>;
defm : VPatBinaryV_VV_VX_VI<"int_riscv_vsra", "PseudoVSRA", AllIntegerVectors,
                            uimm5>;

foreach vti = AllIntegerVectors in {
  // Emit shift by 1 as an add since it might be faster.
  def : Pat<(vti.Vector (int_riscv_vsll (vti.Vector undef),
                                        (vti.Vector vti.RegClass:$rs1),
                                        (XLenVT 1), VLOpFrag)),
            (!cast<Instruction>("PseudoVADD_VV_"#vti.LMul.MX) vti.RegClass:$rs1,
                                                              vti.RegClass:$rs1,
                                                              GPR:$vl,
                                                              vti.Log2SEW)>;
  def : Pat<(vti.Vector (int_riscv_vsll_mask (vti.Vector vti.RegClass:$merge),
                                             (vti.Vector vti.RegClass:$rs1),
                                             (XLenVT 1),
                                             (vti.Mask V0),
                                             VLOpFrag,
                                             (XLenVT timm:$policy))),
            (!cast<Instruction>("PseudoVADD_VV_"#vti.LMul.MX#"_MASK")
                                                        vti.RegClass:$merge,
                                                        vti.RegClass:$rs1,
                                                        vti.RegClass:$rs1,
                                                        (vti.Mask V0),
                                                        GPR:$vl,
                                                        vti.Log2SEW,
                                                        (XLenVT timm:$policy))>;
}

//===----------------------------------------------------------------------===//
// 12.7. Vector Narrowing Integer Right Shift Instructions
//===----------------------------------------------------------------------===//
defm : VPatBinaryV_WV_WX_WI<"int_riscv_vnsrl", "PseudoVNSRL", AllWidenableIntVectors>;
defm : VPatBinaryV_WV_WX_WI<"int_riscv_vnsra", "PseudoVNSRA", AllWidenableIntVectors>;

//===----------------------------------------------------------------------===//
// 12.8. Vector Integer Comparison Instructions
//===----------------------------------------------------------------------===//
defm : VPatBinaryM_VV_VX_VI<"int_riscv_vmseq", "PseudoVMSEQ", AllIntegerVectors>;
defm : VPatBinaryM_VV_VX_VI<"int_riscv_vmsne", "PseudoVMSNE", AllIntegerVectors>;
defm : VPatBinaryM_VV_VX<"int_riscv_vmsltu", "PseudoVMSLTU", AllIntegerVectors>;
defm : VPatBinaryM_VV_VX<"int_riscv_vmslt", "PseudoVMSLT", AllIntegerVectors>;
defm : VPatBinaryM_VV_VX_VI<"int_riscv_vmsleu", "PseudoVMSLEU", AllIntegerVectors>;
defm : VPatBinaryM_VV_VX_VI<"int_riscv_vmsle", "PseudoVMSLE", AllIntegerVectors>;

defm : VPatBinaryM_VX_VI<"int_riscv_vmsgtu", "PseudoVMSGTU", AllIntegerVectors>;
defm : VPatBinaryM_VX_VI<"int_riscv_vmsgt", "PseudoVMSGT", AllIntegerVectors>;

// Match vmsgt with 2 vector operands to vmslt with the operands swapped.
defm : VPatBinarySwappedM_VV<"int_riscv_vmsgtu", "PseudoVMSLTU", AllIntegerVectors>;
defm : VPatBinarySwappedM_VV<"int_riscv_vmsgt", "PseudoVMSLT", AllIntegerVectors>;

defm : VPatBinarySwappedM_VV<"int_riscv_vmsgeu", "PseudoVMSLEU", AllIntegerVectors>;
defm : VPatBinarySwappedM_VV<"int_riscv_vmsge", "PseudoVMSLE", AllIntegerVectors>;

// Match vmslt(u).vx intrinsics to vmsle(u).vi if the scalar is -15 to 16 and
// non-zero. Zero can be .vx with x0. This avoids the user needing to know that
// there is no vmslt(u).vi instruction. Similar for vmsge(u).vx intrinsics
// using vmslt(u).vi.
defm : VPatCompare_VI<"int_riscv_vmslt", "PseudoVMSLE", simm5_plus1_nonzero>;
defm : VPatCompare_VI<"int_riscv_vmsltu", "PseudoVMSLEU", simm5_plus1_nonzero>;

// We need to handle 0 for vmsge.vi using vmslt.vi because there is no vmsge.vx.
defm : VPatCompare_VI<"int_riscv_vmsge", "PseudoVMSGT", simm5_plus1>;
defm : VPatCompare_VI<"int_riscv_vmsgeu", "PseudoVMSGTU", simm5_plus1_nonzero>;

//===----------------------------------------------------------------------===//
// 12.9. Vector Integer Min/Max Instructions
//===----------------------------------------------------------------------===//
defm : VPatBinaryV_VV_VX<"int_riscv_vminu", "PseudoVMINU", AllIntegerVectors>;
defm : VPatBinaryV_VV_VX<"int_riscv_vmin", "PseudoVMIN", AllIntegerVectors>;
defm : VPatBinaryV_VV_VX<"int_riscv_vmaxu", "PseudoVMAXU", AllIntegerVectors>;
defm : VPatBinaryV_VV_VX<"int_riscv_vmax", "PseudoVMAX", AllIntegerVectors>;

//===----------------------------------------------------------------------===//
// 12.10. Vector Single-Width Integer Multiply Instructions
//===----------------------------------------------------------------------===//
defm : VPatBinaryV_VV_VX<"int_riscv_vmul", "PseudoVMUL", AllIntegerVectors>;
defm : VPatBinaryV_VV_VX<"int_riscv_vmulh", "PseudoVMULH", AllIntegerVectors>;
defm : VPatBinaryV_VV_VX<"int_riscv_vmulhu", "PseudoVMULHU", AllIntegerVectors>;
defm : VPatBinaryV_VV_VX<"int_riscv_vmulhsu", "PseudoVMULHSU", AllIntegerVectors>;

//===----------------------------------------------------------------------===//
// 12.11. Vector Integer Divide Instructions
//===----------------------------------------------------------------------===//
defm : VPatBinaryV_VV_VX<"int_riscv_vdivu", "PseudoVDIVU", AllIntegerVectors>;
defm : VPatBinaryV_VV_VX<"int_riscv_vdiv", "PseudoVDIV", AllIntegerVectors>;
defm : VPatBinaryV_VV_VX<"int_riscv_vremu", "PseudoVREMU", AllIntegerVectors>;
defm : VPatBinaryV_VV_VX<"int_riscv_vrem", "PseudoVREM", AllIntegerVectors>;

//===----------------------------------------------------------------------===//
// 12.12. Vector Widening Integer Multiply Instructions
//===----------------------------------------------------------------------===//
defm : VPatBinaryW_VV_VX<"int_riscv_vwmul", "PseudoVWMUL", AllWidenableIntVectors>;
defm : VPatBinaryW_VV_VX<"int_riscv_vwmulu", "PseudoVWMULU", AllWidenableIntVectors>;
defm : VPatBinaryW_VV_VX<"int_riscv_vwmulsu", "PseudoVWMULSU", AllWidenableIntVectors>;

//===----------------------------------------------------------------------===//
// 12.13. Vector Single-Width Integer Multiply-Add Instructions
//===----------------------------------------------------------------------===//
defm : VPatTernaryV_VV_VX_AAXA<"int_riscv_vmadd", "PseudoVMADD", AllIntegerVectors>;
defm : VPatTernaryV_VV_VX_AAXA<"int_riscv_vnmsub", "PseudoVNMSUB", AllIntegerVectors>;
defm : VPatTernaryV_VV_VX_AAXA<"int_riscv_vmacc", "PseudoVMACC", AllIntegerVectors>;
defm : VPatTernaryV_VV_VX_AAXA<"int_riscv_vnmsac", "PseudoVNMSAC", AllIntegerVectors>;

//===----------------------------------------------------------------------===//
// 12.14. Vector Widening Integer Multiply-Add Instructions
//===----------------------------------------------------------------------===//
defm : VPatTernaryW_VV_VX<"int_riscv_vwmaccu", "PseudoVWMACCU", AllWidenableIntVectors>;
defm : VPatTernaryW_VV_VX<"int_riscv_vwmacc", "PseudoVWMACC", AllWidenableIntVectors>;
defm : VPatTernaryW_VV_VX<"int_riscv_vwmaccsu", "PseudoVWMACCSU", AllWidenableIntVectors>;
defm : VPatTernaryW_VX<"int_riscv_vwmaccus", "PseudoVWMACCUS", AllWidenableIntVectors>;

//===----------------------------------------------------------------------===//
// 12.15. Vector Integer Merge Instructions
//===----------------------------------------------------------------------===//
defm : VPatBinaryV_VM_XM_IM<"int_riscv_vmerge", "PseudoVMERGE">;

//===----------------------------------------------------------------------===//
// 12.16. Vector Integer Move Instructions
//===----------------------------------------------------------------------===//
foreach vti = AllVectors in {
  def : Pat<(vti.Vector (int_riscv_vmv_v_v (vti.Vector undef),
                                           (vti.Vector vti.RegClass:$rs1),
                                           VLOpFrag)),
            (!cast<Instruction>("PseudoVMV_V_V_"#vti.LMul.MX)
             $rs1, GPR:$vl, vti.Log2SEW)>;
  def : Pat<(vti.Vector (int_riscv_vmv_v_v (vti.Vector vti.RegClass:$passthru),
                                           (vti.Vector vti.RegClass:$rs1),
                                           VLOpFrag)),
            (!cast<Instruction>("PseudoVMV_V_V_"#vti.LMul.MX#"_TU")
             $passthru, $rs1, GPR:$vl, vti.Log2SEW)>;

  // vmv.v.x/vmv.v.i are handled in RISCInstrVInstrInfoVVLPatterns.td
}

//===----------------------------------------------------------------------===//
// 13.1. Vector Single-Width Saturating Add and Subtract
//===----------------------------------------------------------------------===//
defm : VPatBinaryV_VV_VX_VI<"int_riscv_vsaddu", "PseudoVSADDU", AllIntegerVectors>;
defm : VPatBinaryV_VV_VX_VI<"int_riscv_vsadd", "PseudoVSADD", AllIntegerVectors>;
defm : VPatBinaryV_VV_VX<"int_riscv_vssubu", "PseudoVSSUBU", AllIntegerVectors>;
defm : VPatBinaryV_VV_VX<"int_riscv_vssub", "PseudoVSSUB", AllIntegerVectors>;

//===----------------------------------------------------------------------===//
// 13.2. Vector Single-Width Averaging Add and Subtract
//===----------------------------------------------------------------------===//
defm : VPatBinaryV_VV_VX<"int_riscv_vaaddu", "PseudoVAADDU", AllIntegerVectors>;
defm : VPatBinaryV_VV_VX<"int_riscv_vaadd", "PseudoVAADD", AllIntegerVectors>;
defm : VPatBinaryV_VV_VX<"int_riscv_vasubu", "PseudoVASUBU", AllIntegerVectors>;
defm : VPatBinaryV_VV_VX<"int_riscv_vasub", "PseudoVASUB", AllIntegerVectors>;

//===----------------------------------------------------------------------===//
// 13.3. Vector Single-Width Fractional Multiply with Rounding and Saturation
//===----------------------------------------------------------------------===//
defm : VPatBinaryV_VV_VX<"int_riscv_vsmul", "PseudoVSMUL", AllIntegerVectors>;

//===----------------------------------------------------------------------===//
// 13.4. Vector Single-Width Scaling Shift Instructions
//===----------------------------------------------------------------------===//
defm : VPatBinaryV_VV_VX_VI<"int_riscv_vssrl", "PseudoVSSRL", AllIntegerVectors,
                            uimm5>;
defm : VPatBinaryV_VV_VX_VI<"int_riscv_vssra", "PseudoVSSRA", AllIntegerVectors,
                            uimm5>;

//===----------------------------------------------------------------------===//
// 13.5. Vector Narrowing Fixed-Point Clip Instructions
//===----------------------------------------------------------------------===//
defm : VPatBinaryV_WV_WX_WI<"int_riscv_vnclipu", "PseudoVNCLIPU", AllWidenableIntVectors>;
defm : VPatBinaryV_WV_WX_WI<"int_riscv_vnclip", "PseudoVNCLIP", AllWidenableIntVectors>;

} // Predicates = [HasVInstructions]

let Predicates = [HasVInstructionsAnyF] in {
//===----------------------------------------------------------------------===//
// 14.2. Vector Single-Width Floating-Point Add/Subtract Instructions
//===----------------------------------------------------------------------===//
defm : VPatBinaryV_VV_VX<"int_riscv_vfadd", "PseudoVFADD", AllFloatVectors>;
defm : VPatBinaryV_VV_VX<"int_riscv_vfsub", "PseudoVFSUB", AllFloatVectors>;
defm : VPatBinaryV_VX<"int_riscv_vfrsub", "PseudoVFRSUB", AllFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.3. Vector Widening Floating-Point Add/Subtract Instructions
//===----------------------------------------------------------------------===//
defm : VPatBinaryW_VV_VX<"int_riscv_vfwadd", "PseudoVFWADD", AllWidenableFloatVectors>;
defm : VPatBinaryW_VV_VX<"int_riscv_vfwsub", "PseudoVFWSUB", AllWidenableFloatVectors>;
defm : VPatBinaryW_WV_WX<"int_riscv_vfwadd_w", "PseudoVFWADD", AllWidenableFloatVectors>;
defm : VPatBinaryW_WV_WX<"int_riscv_vfwsub_w", "PseudoVFWSUB", AllWidenableFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.4. Vector Single-Width Floating-Point Multiply/Divide Instructions
//===----------------------------------------------------------------------===//
defm : VPatBinaryV_VV_VX<"int_riscv_vfmul", "PseudoVFMUL", AllFloatVectors>;
defm : VPatBinaryV_VV_VX<"int_riscv_vfdiv", "PseudoVFDIV", AllFloatVectors>;
defm : VPatBinaryV_VX<"int_riscv_vfrdiv", "PseudoVFRDIV", AllFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.5. Vector Widening Floating-Point Multiply
//===----------------------------------------------------------------------===//
defm : VPatBinaryW_VV_VX<"int_riscv_vfwmul", "PseudoVFWMUL", AllWidenableFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.6. Vector Single-Width Floating-Point Fused Multiply-Add Instructions
//===----------------------------------------------------------------------===//
defm : VPatTernaryV_VV_VX_AAXA<"int_riscv_vfmacc", "PseudoVFMACC", AllFloatVectors>;
defm : VPatTernaryV_VV_VX_AAXA<"int_riscv_vfnmacc", "PseudoVFNMACC", AllFloatVectors>;
defm : VPatTernaryV_VV_VX_AAXA<"int_riscv_vfmsac", "PseudoVFMSAC", AllFloatVectors>;
defm : VPatTernaryV_VV_VX_AAXA<"int_riscv_vfnmsac", "PseudoVFNMSAC", AllFloatVectors>;
defm : VPatTernaryV_VV_VX_AAXA<"int_riscv_vfmadd", "PseudoVFMADD", AllFloatVectors>;
defm : VPatTernaryV_VV_VX_AAXA<"int_riscv_vfnmadd", "PseudoVFNMADD", AllFloatVectors>;
defm : VPatTernaryV_VV_VX_AAXA<"int_riscv_vfmsub", "PseudoVFMSUB", AllFloatVectors>;
defm : VPatTernaryV_VV_VX_AAXA<"int_riscv_vfnmsub", "PseudoVFNMSUB", AllFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.7. Vector Widening Floating-Point Fused Multiply-Add Instructions
//===----------------------------------------------------------------------===//
defm : VPatTernaryW_VV_VX<"int_riscv_vfwmacc", "PseudoVFWMACC", AllWidenableFloatVectors>;
defm : VPatTernaryW_VV_VX<"int_riscv_vfwnmacc", "PseudoVFWNMACC", AllWidenableFloatVectors>;
defm : VPatTernaryW_VV_VX<"int_riscv_vfwmsac", "PseudoVFWMSAC", AllWidenableFloatVectors>;
defm : VPatTernaryW_VV_VX<"int_riscv_vfwnmsac", "PseudoVFWNMSAC", AllWidenableFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.8. Vector Floating-Point Square-Root Instruction
//===----------------------------------------------------------------------===//
defm : VPatUnaryV_V<"int_riscv_vfsqrt", "PseudoVFSQRT", AllFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.9. Vector Floating-Point Reciprocal Square-Root Estimate Instruction
//===----------------------------------------------------------------------===//
defm : VPatUnaryV_V<"int_riscv_vfrsqrt7", "PseudoVFRSQRT7", AllFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.10. Vector Floating-Point Reciprocal Estimate Instruction
//===----------------------------------------------------------------------===//
defm : VPatUnaryV_V<"int_riscv_vfrec7", "PseudoVFREC7", AllFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.11. Vector Floating-Point Min/Max Instructions
//===----------------------------------------------------------------------===//
defm : VPatBinaryV_VV_VX<"int_riscv_vfmin", "PseudoVFMIN", AllFloatVectors>;
defm : VPatBinaryV_VV_VX<"int_riscv_vfmax", "PseudoVFMAX", AllFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.12. Vector Floating-Point Sign-Injection Instructions
//===----------------------------------------------------------------------===//
defm : VPatBinaryV_VV_VX<"int_riscv_vfsgnj", "PseudoVFSGNJ", AllFloatVectors>;
defm : VPatBinaryV_VV_VX<"int_riscv_vfsgnjn", "PseudoVFSGNJN", AllFloatVectors>;
defm : VPatBinaryV_VV_VX<"int_riscv_vfsgnjx", "PseudoVFSGNJX", AllFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.13. Vector Floating-Point Compare Instructions
//===----------------------------------------------------------------------===//
defm : VPatBinaryM_VV_VX<"int_riscv_vmfeq", "PseudoVMFEQ", AllFloatVectors>;
defm : VPatBinaryM_VV_VX<"int_riscv_vmfle", "PseudoVMFLE", AllFloatVectors>;
defm : VPatBinaryM_VV_VX<"int_riscv_vmflt", "PseudoVMFLT", AllFloatVectors>;
defm : VPatBinaryM_VV_VX<"int_riscv_vmfne", "PseudoVMFNE", AllFloatVectors>;
defm : VPatBinaryM_VX<"int_riscv_vmfgt", "PseudoVMFGT", AllFloatVectors>;
defm : VPatBinaryM_VX<"int_riscv_vmfge", "PseudoVMFGE", AllFloatVectors>;
defm : VPatBinarySwappedM_VV<"int_riscv_vmfgt", "PseudoVMFLT", AllFloatVectors>;
defm : VPatBinarySwappedM_VV<"int_riscv_vmfge", "PseudoVMFLE", AllFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.14. Vector Floating-Point Classify Instruction
//===----------------------------------------------------------------------===//
defm : VPatConversionVI_VF<"int_riscv_vfclass", "PseudoVFCLASS">;

//===----------------------------------------------------------------------===//
// 14.15. Vector Floating-Point Merge Instruction
//===----------------------------------------------------------------------===//
// We can use vmerge.vvm to support vector-vector vfmerge.
// NOTE: Clang previously used int_riscv_vfmerge for vector-vector, but now uses
// int_riscv_vmerge. Support both for compatibility.
defm : VPatBinaryV_VM_TAIL<"int_riscv_vmerge", "PseudoVMERGE",
                           /*CarryOut = */0, /*vtilist=*/AllFloatVectors>;
defm : VPatBinaryV_VM_TAIL<"int_riscv_vfmerge", "PseudoVMERGE",
                           /*CarryOut = */0, /*vtilist=*/AllFloatVectors>;
defm : VPatBinaryV_XM_TAIL<"int_riscv_vfmerge", "PseudoVFMERGE",
                           /*CarryOut = */0, /*vtilist=*/AllFloatVectors>;

foreach fvti = AllFloatVectors in {
  defvar instr = !cast<Instruction>("PseudoVMERGE_VIM_"#fvti.LMul.MX);
  def : Pat<(fvti.Vector (int_riscv_vfmerge (fvti.Vector undef),
                                            (fvti.Vector fvti.RegClass:$rs2),
                                            (fvti.Scalar (fpimm0)),
                                            (fvti.Mask V0), VLOpFrag)),
            (instr fvti.RegClass:$rs2, 0, (fvti.Mask V0), GPR:$vl, fvti.Log2SEW)>;
  defvar instr_tu = !cast<Instruction>("PseudoVMERGE_VIM_"#fvti.LMul.MX#"_TU");
  def : Pat<(fvti.Vector (int_riscv_vfmerge (fvti.Vector fvti.RegClass:$merge),
                                            (fvti.Vector fvti.RegClass:$rs2),
                                            (fvti.Scalar (fpimm0)),
                                            (fvti.Mask V0), VLOpFrag)),
            (instr_tu fvti.RegClass:$merge, fvti.RegClass:$rs2, 0,
                      (fvti.Mask V0), GPR:$vl, fvti.Log2SEW)>;
}

//===----------------------------------------------------------------------===//
// 14.17. Single-Width Floating-Point/Integer Type-Convert Instructions
//===----------------------------------------------------------------------===//
defm : VPatConversionVI_VF<"int_riscv_vfcvt_xu_f_v", "PseudoVFCVT_XU_F">;
defm : VPatConversionVI_VF<"int_riscv_vfcvt_rtz_xu_f_v", "PseudoVFCVT_RTZ_XU_F">;
defm : VPatConversionVI_VF<"int_riscv_vfcvt_x_f_v", "PseudoVFCVT_X_F">;
defm : VPatConversionVI_VF<"int_riscv_vfcvt_rtz_x_f_v", "PseudoVFCVT_RTZ_X_F">;
defm : VPatConversionVF_VI<"int_riscv_vfcvt_f_x_v", "PseudoVFCVT_F_X">;
defm : VPatConversionVF_VI<"int_riscv_vfcvt_f_xu_v", "PseudoVFCVT_F_XU">;

//===----------------------------------------------------------------------===//
// 14.18. Widening Floating-Point/Integer Type-Convert Instructions
//===----------------------------------------------------------------------===//
defm : VPatConversionWI_VF<"int_riscv_vfwcvt_xu_f_v", "PseudoVFWCVT_XU_F">;
defm : VPatConversionWI_VF<"int_riscv_vfwcvt_x_f_v", "PseudoVFWCVT_X_F">;
defm : VPatConversionWI_VF<"int_riscv_vfwcvt_rtz_xu_f_v", "PseudoVFWCVT_RTZ_XU_F">;
defm : VPatConversionWI_VF<"int_riscv_vfwcvt_rtz_x_f_v", "PseudoVFWCVT_RTZ_X_F">;
defm : VPatConversionWF_VI<"int_riscv_vfwcvt_f_xu_v", "PseudoVFWCVT_F_XU">;
defm : VPatConversionWF_VI<"int_riscv_vfwcvt_f_x_v", "PseudoVFWCVT_F_X">;
defm : VPatConversionWF_VF<"int_riscv_vfwcvt_f_f_v", "PseudoVFWCVT_F_F">;

//===----------------------------------------------------------------------===//
// 14.19. Narrowing Floating-Point/Integer Type-Convert Instructions
//===----------------------------------------------------------------------===//
defm : VPatConversionVI_WF<"int_riscv_vfncvt_xu_f_w", "PseudoVFNCVT_XU_F">;
defm : VPatConversionVI_WF<"int_riscv_vfncvt_x_f_w", "PseudoVFNCVT_X_F">;
defm : VPatConversionVI_WF<"int_riscv_vfncvt_rtz_xu_f_w", "PseudoVFNCVT_RTZ_XU_F">;
defm : VPatConversionVI_WF<"int_riscv_vfncvt_rtz_x_f_w", "PseudoVFNCVT_RTZ_X_F">;
defm : VPatConversionVF_WI <"int_riscv_vfncvt_f_xu_w", "PseudoVFNCVT_F_XU">;
defm : VPatConversionVF_WI <"int_riscv_vfncvt_f_x_w", "PseudoVFNCVT_F_X">;
defm : VPatConversionVF_WF<"int_riscv_vfncvt_f_f_w", "PseudoVFNCVT_F_F">;
defm : VPatConversionVF_WF<"int_riscv_vfncvt_rod_f_f_w", "PseudoVFNCVT_ROD_F_F">;
} // Predicates = [HasVInstructionsAnyF]

let Predicates = [HasVInstructions] in {
//===----------------------------------------------------------------------===//
// 15.1. Vector Single-Width Integer Reduction Instructions
//===----------------------------------------------------------------------===//
defm : VPatReductionV_VS<"int_riscv_vredsum", "PseudoVREDSUM">;
defm : VPatReductionV_VS<"int_riscv_vredand", "PseudoVREDAND">;
defm : VPatReductionV_VS<"int_riscv_vredor", "PseudoVREDOR">;
defm : VPatReductionV_VS<"int_riscv_vredxor", "PseudoVREDXOR">;
defm : VPatReductionV_VS<"int_riscv_vredminu", "PseudoVREDMINU">;
defm : VPatReductionV_VS<"int_riscv_vredmin", "PseudoVREDMIN">;
defm : VPatReductionV_VS<"int_riscv_vredmaxu", "PseudoVREDMAXU">;
defm : VPatReductionV_VS<"int_riscv_vredmax", "PseudoVREDMAX">;

//===----------------------------------------------------------------------===//
// 15.2. Vector Widening Integer Reduction Instructions
//===----------------------------------------------------------------------===//
defm : VPatReductionW_VS<"int_riscv_vwredsumu", "PseudoVWREDSUMU">;
defm : VPatReductionW_VS<"int_riscv_vwredsum", "PseudoVWREDSUM">;
} // Predicates = [HasVInstructions]

let Predicates = [HasVInstructionsAnyF] in {
//===----------------------------------------------------------------------===//
// 15.3. Vector Single-Width Floating-Point Reduction Instructions
//===----------------------------------------------------------------------===//
defm : VPatReductionV_VS<"int_riscv_vfredosum", "PseudoVFREDOSUM", /*IsFloat=*/1>;
defm : VPatReductionV_VS<"int_riscv_vfredusum", "PseudoVFREDUSUM", /*IsFloat=*/1>;
defm : VPatReductionV_VS<"int_riscv_vfredmin", "PseudoVFREDMIN", /*IsFloat=*/1>;
defm : VPatReductionV_VS<"int_riscv_vfredmax", "PseudoVFREDMAX", /*IsFloat=*/1>;

//===----------------------------------------------------------------------===//
// 15.4. Vector Widening Floating-Point Reduction Instructions
//===----------------------------------------------------------------------===//
defm : VPatReductionW_VS<"int_riscv_vfwredusum", "PseudoVFWREDUSUM", /*IsFloat=*/1>;
defm : VPatReductionW_VS<"int_riscv_vfwredosum", "PseudoVFWREDOSUM", /*IsFloat=*/1>;

} // Predicates = [HasVInstructionsAnyF]

//===----------------------------------------------------------------------===//
// 16. Vector Mask Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasVInstructions] in {
//===----------------------------------------------------------------------===//
// 16.1 Vector Mask-Register Logical Instructions
//===----------------------------------------------------------------------===//
defm : VPatBinaryM_MM<"int_riscv_vmand", "PseudoVMAND">;
defm : VPatBinaryM_MM<"int_riscv_vmnand", "PseudoVMNAND">;
defm : VPatBinaryM_MM<"int_riscv_vmandn", "PseudoVMANDN">;
defm : VPatBinaryM_MM<"int_riscv_vmxor", "PseudoVMXOR">;
defm : VPatBinaryM_MM<"int_riscv_vmor", "PseudoVMOR">;
defm : VPatBinaryM_MM<"int_riscv_vmnor", "PseudoVMNOR">;
defm : VPatBinaryM_MM<"int_riscv_vmorn", "PseudoVMORN">;
defm : VPatBinaryM_MM<"int_riscv_vmxnor", "PseudoVMXNOR">;

// pseudo instructions
defm : VPatNullaryM<"int_riscv_vmclr", "PseudoVMCLR">;
defm : VPatNullaryM<"int_riscv_vmset", "PseudoVMSET">;

//===----------------------------------------------------------------------===//
// 16.2. Vector count population in mask vcpop.m
//===----------------------------------------------------------------------===//
defm : VPatUnaryS_M<"int_riscv_vcpop", "PseudoVCPOP">;

//===----------------------------------------------------------------------===//
// 16.3. vfirst find-first-set mask bit
//===----------------------------------------------------------------------===//
defm : VPatUnaryS_M<"int_riscv_vfirst", "PseudoVFIRST">;

//===----------------------------------------------------------------------===//
// 16.4. vmsbf.m set-before-first mask bit
//===----------------------------------------------------------------------===//
defm : VPatUnaryM_M<"int_riscv_vmsbf", "PseudoVMSBF">;

//===----------------------------------------------------------------------===//
// 16.5. vmsif.m set-including-first mask bit
//===----------------------------------------------------------------------===//
defm : VPatUnaryM_M<"int_riscv_vmsif", "PseudoVMSIF">;

//===----------------------------------------------------------------------===//
// 16.6. vmsof.m set-only-first mask bit
//===----------------------------------------------------------------------===//
defm : VPatUnaryM_M<"int_riscv_vmsof", "PseudoVMSOF">;

//===----------------------------------------------------------------------===//
// 16.8.  Vector Iota Instruction
//===----------------------------------------------------------------------===//
defm : VPatUnaryV_M<"int_riscv_viota", "PseudoVIOTA">;

//===----------------------------------------------------------------------===//
// 16.9. Vector Element Index Instruction
//===----------------------------------------------------------------------===//
defm : VPatNullaryV<"int_riscv_vid", "PseudoVID">;

} // Predicates = [HasVInstructions]

//===----------------------------------------------------------------------===//
// 17. Vector Permutation Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 17.1. Integer Scalar Move Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasVInstructions] in {
foreach vti = AllIntegerVectors in {
  def : Pat<(riscv_vmv_x_s (vti.Vector vti.RegClass:$rs2)),
            (!cast<Instruction>("PseudoVMV_X_S_" # vti.LMul.MX) $rs2, vti.Log2SEW)>;
  // vmv.s.x is handled with a custom node in RISCVInstrInfoVVLPatterns.td
}
} // Predicates = [HasVInstructions]

//===----------------------------------------------------------------------===//
// 17.2. Floating-Point Scalar Move Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasVInstructionsAnyF] in {
foreach fvti = AllFloatVectors in {
  defvar instr = !cast<Instruction>("PseudoVFMV_"#fvti.ScalarSuffix#"_S_" #
                                    fvti.LMul.MX);
  def : Pat<(fvti.Scalar (int_riscv_vfmv_f_s (fvti.Vector fvti.RegClass:$rs2))),
                         (instr $rs2, fvti.Log2SEW)>;

  def : Pat<(fvti.Vector (int_riscv_vfmv_s_f (fvti.Vector fvti.RegClass:$rs1),
                         (fvti.Scalar fvti.ScalarRegClass:$rs2), VLOpFrag)),
            (!cast<Instruction>("PseudoVFMV_S_"#fvti.ScalarSuffix#"_" #
                                fvti.LMul.MX)
             (fvti.Vector $rs1),
             (fvti.Scalar fvti.ScalarRegClass:$rs2),
             GPR:$vl, fvti.Log2SEW)>;

  def : Pat<(fvti.Vector (int_riscv_vfmv_s_f (fvti.Vector fvti.RegClass:$rs1),
                         (fvti.Scalar (fpimm0)), VLOpFrag)),
            (!cast<Instruction>("PseudoVMV_S_X_" # fvti.LMul.MX)
             (fvti.Vector $rs1), X0, GPR:$vl, fvti.Log2SEW)>;
}
} // Predicates = [HasVInstructionsAnyF]

//===----------------------------------------------------------------------===//
// 17.3. Vector Slide Instructions
//===----------------------------------------------------------------------===//
let Predicates = [HasVInstructions] in {
  defm : VPatTernaryV_VX_VI<"int_riscv_vslideup", "PseudoVSLIDEUP", AllIntegerVectors, uimm5>;
  defm : VPatTernaryV_VX_VI<"int_riscv_vslidedown", "PseudoVSLIDEDOWN", AllIntegerVectors, uimm5>;
  defm : VPatBinaryV_VX<"int_riscv_vslide1up", "PseudoVSLIDE1UP", AllIntegerVectors>;
  defm : VPatBinaryV_VX<"int_riscv_vslide1down", "PseudoVSLIDE1DOWN", AllIntegerVectors>;
} // Predicates = [HasVInstructions]

let Predicates = [HasVInstructionsAnyF] in {
  defm : VPatTernaryV_VX_VI<"int_riscv_vslideup", "PseudoVSLIDEUP", AllFloatVectors, uimm5>;
  defm : VPatTernaryV_VX_VI<"int_riscv_vslidedown", "PseudoVSLIDEDOWN", AllFloatVectors, uimm5>;
  defm : VPatBinaryV_VX<"int_riscv_vfslide1up", "PseudoVFSLIDE1UP", AllFloatVectors>;
  defm : VPatBinaryV_VX<"int_riscv_vfslide1down", "PseudoVFSLIDE1DOWN", AllFloatVectors>;
} // Predicates = [HasVInstructionsAnyF]

//===----------------------------------------------------------------------===//
// 17.4. Vector Register Gather Instructions
//===----------------------------------------------------------------------===//
let Predicates = [HasVInstructions] in {
  defm : VPatBinaryV_VV_VX_VI_INT<"int_riscv_vrgather", "PseudoVRGATHER",
                                  AllIntegerVectors, uimm5>;
  defm : VPatBinaryV_VV_INT_EEW<"int_riscv_vrgatherei16_vv", "PseudoVRGATHEREI16",
                                /* eew */ 16, AllIntegerVectors>;
} // Predicates = [HasVInstructions]

let Predicates = [HasVInstructionsAnyF] in {
  defm : VPatBinaryV_VV_VX_VI_INT<"int_riscv_vrgather", "PseudoVRGATHER",
                                  AllFloatVectors, uimm5>;
  defm : VPatBinaryV_VV_INT_EEW<"int_riscv_vrgatherei16_vv", "PseudoVRGATHEREI16",
                                /* eew */ 16, AllFloatVectors>;
} // Predicates = [HasVInstructionsAnyF]

//===----------------------------------------------------------------------===//
// 17.5. Vector Compress Instruction
//===----------------------------------------------------------------------===//
let Predicates = [HasVInstructions] in {
  defm : VPatUnaryV_V_AnyMask<"int_riscv_vcompress", "PseudoVCOMPRESS", AllIntegerVectors>;
} // Predicates = [HasVInstructions]

let Predicates = [HasVInstructionsAnyF] in {
  defm : VPatUnaryV_V_AnyMask<"int_riscv_vcompress", "PseudoVCOMPRESS", AllFloatVectors>;
} // Predicates = [HasVInstructionsAnyF]

// Include the non-intrinsic ISel patterns
include "RISCVInstrInfoVVLPatterns.td"
include "RISCVInstrInfoVSDPatterns.td"
