//===-- RISCVInstrInfoVPseudos.td - RISC-V 'V' Pseudos -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file contains the required infrastructure to support code generation
/// for the standard 'V' (Vector) extension, version 0.10.  This version is still
/// experimental as the 'V' extension hasn't been ratified yet.
///
/// This file is included from RISCVInstrInfoV.td
///
//===----------------------------------------------------------------------===//

def riscv_vmv_x_s : SDNode<"RISCVISD::VMV_X_S",
                           SDTypeProfile<1, 1, [SDTCisInt<0>, SDTCisVec<1>,
                                                SDTCisInt<1>]>>;
def riscv_read_vlenb : SDNode<"RISCVISD::READ_VLENB",
                              SDTypeProfile<1, 0, [SDTCisVT<0, XLenVT>]>>;

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

class FPR_Info<RegisterClass regclass, string fx> {
  RegisterClass fprclass = regclass;
  string FX = fx;
}

def SCALAR_F16 : FPR_Info<FPR16, "F16">;
def SCALAR_F32 : FPR_Info<FPR32, "F32">;
def SCALAR_F64 : FPR_Info<FPR64, "F64">;

def FPList {
  list<FPR_Info> fpinfo = [SCALAR_F16, SCALAR_F32, SCALAR_F64];
}

class MxSet<int eew> {
  list<LMULInfo> m = !cond(!eq(eew, 8) : [V_MF8, V_MF4, V_MF2, V_M1, V_M2, V_M4, V_M8],
                           !eq(eew, 16) : [V_MF4, V_MF2, V_M1, V_M2, V_M4, V_M8],
                           !eq(eew, 32) : [V_MF2, V_M1, V_M2, V_M4, V_M8],
                           !eq(eew, 64) : [V_M1, V_M2, V_M4, V_M8]);
}

class NFSet<LMULInfo m> {
  list<int> L = !cond(!eq(m.value, V_M8.value): [],
                      !eq(m.value, V_M4.value): [2],
                      !eq(m.value, V_M2.value): [2, 3, 4],
                      true: [2, 3, 4, 5, 6, 7, 8]);
}

class shift_amount<int num> {
  int val = !if(!eq(num, 1), 0, !add(1, shift_amount<!srl(num, 1)>.val));
}

class octuple_from_str<string MX> {
  int ret = !cond(!eq(MX, "MF8") : 1,
                        !eq(MX, "MF4") : 2,
                        !eq(MX, "MF2") : 4,
                        !eq(MX, "M1") : 8,
                        !eq(MX, "M2") : 16,
                        !eq(MX, "M4") : 32,
                        !eq(MX, "M8") : 64);
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

// Output pattern for X0 used to represent VLMAX in the pseudo instructions.
def VLMax : OutPatFrag<(ops), (XLenVT X0)>;

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
  // we assume SEW=8 and set corresponding LMUL.
  int SEW = 8;
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
  Instruction BaseInstr;
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

def RISCVVIntrinsicsTable : GenericTable {
  let FilterClass = "RISCVVIntrinsic";
  let CppTypeName = "RISCVVIntrinsicInfo";
  let Fields = ["IntrinsicID", "SplatOperand"];
  let PrimaryKey = ["IntrinsicID"];
  let PrimaryKeyName = "getRISCVVIntrinsicInfo";
}

class RISCVVLE<bit M, bit Str, bit F, bits<7> S, bits<3> L> {
  bits<1> Masked = M;
  bits<1> Strided = Str;
  bits<1> FF = F;
  bits<7> SEW = S;
  bits<3> LMUL = L;
  Pseudo Pseudo = !cast<Pseudo>(NAME);
}

def RISCVVLETable : GenericTable {
  let FilterClass = "RISCVVLE";
  let CppTypeName = "VLEPseudo";
  let Fields = ["Masked", "Strided", "FF", "SEW", "LMUL", "Pseudo"];
  let PrimaryKey = ["Masked", "Strided", "FF", "SEW", "LMUL"];
  let PrimaryKeyName = "getVLEPseudo";
}

class RISCVVSE<bit M, bit Str, bits<7> S, bits<3> L> {
  bits<1> Masked = M;
  bits<1> Strided = Str;
  bits<7> SEW = S;
  bits<3> LMUL = L;
  Pseudo Pseudo = !cast<Pseudo>(NAME);
}

def RISCVVSETable : GenericTable {
  let FilterClass = "RISCVVSE";
  let CppTypeName = "VSEPseudo";
  let Fields = ["Masked", "Strided", "SEW", "LMUL", "Pseudo"];
  let PrimaryKey = ["Masked", "Strided", "SEW", "LMUL"];
  let PrimaryKeyName = "getVSEPseudo";
}

class RISCVVLX_VSX<bit M, bit O, bits<7> S, bits<3> L, bits<3> IL> {
  bits<1> Masked = M;
  bits<1> Ordered = O;
  bits<7> SEW = S;
  bits<3> LMUL = L;
  bits<3> IndexLMUL = IL;
  Pseudo Pseudo = !cast<Pseudo>(NAME);
}

class RISCVVLX<bit M, bit O, bits<7> S, bits<3> L, bits<3> IL> :
  RISCVVLX_VSX<M, O, S, L, IL>;
class RISCVVSX<bit M, bit O, bits<7> S, bits<3> L, bits<3> IL> :
  RISCVVLX_VSX<M, O, S, L, IL>;

class RISCVVLX_VSXTable : GenericTable {
  let CppTypeName = "VLX_VSXPseudo";
  let Fields = ["Masked", "Ordered", "SEW", "LMUL", "IndexLMUL", "Pseudo"];
  let PrimaryKey = ["Masked", "Ordered", "SEW", "LMUL", "IndexLMUL"];
}

def RISCVVLXTable : RISCVVLX_VSXTable {
  let FilterClass = "RISCVVLX";
  let PrimaryKeyName = "getVLXPseudo";
}

def RISCVVSXTable : RISCVVLX_VSXTable {
  let FilterClass = "RISCVVSX";
  let PrimaryKeyName = "getVSXPseudo";
}

class RISCVVLSEG<bits<4> N, bit M, bit Str, bit F, bits<7> S, bits<3> L> {
  bits<4> NF = N;
  bits<1> Masked = M;
  bits<1> Strided = Str;
  bits<1> FF = F;
  bits<7> SEW = S;
  bits<3> LMUL = L;
  Pseudo Pseudo = !cast<Pseudo>(NAME);
}

def RISCVVLSEGTable : GenericTable {
  let FilterClass = "RISCVVLSEG";
  let CppTypeName = "VLSEGPseudo";
  let Fields = ["NF", "Masked", "Strided", "FF", "SEW", "LMUL", "Pseudo"];
  let PrimaryKey = ["NF", "Masked", "Strided", "FF", "SEW", "LMUL"];
  let PrimaryKeyName = "getVLSEGPseudo";
}

class RISCVVLXSEG<bits<4> N, bit M, bit O, bits<7> S, bits<3> L, bits<3> IL> {
  bits<4> NF = N;
  bits<1> Masked = M;
  bits<1> Ordered = O;
  bits<7> SEW = S;
  bits<3> LMUL = L;
  bits<3> IndexLMUL = IL;
  Pseudo Pseudo = !cast<Pseudo>(NAME);
}

def RISCVVLXSEGTable : GenericTable {
  let FilterClass = "RISCVVLXSEG";
  let CppTypeName = "VLXSEGPseudo";
  let Fields = ["NF", "Masked", "Ordered", "SEW", "LMUL", "IndexLMUL", "Pseudo"];
  let PrimaryKey = ["NF", "Masked", "Ordered", "SEW", "LMUL", "IndexLMUL"];
  let PrimaryKeyName = "getVLXSEGPseudo";
}

class RISCVVSSEG<bits<4> N, bit M, bit Str, bits<7> S, bits<3> L> {
  bits<4> NF = N;
  bits<1> Masked = M;
  bits<1> Strided = Str;
  bits<7> SEW = S;
  bits<3> LMUL = L;
  Pseudo Pseudo = !cast<Pseudo>(NAME);
}

def RISCVVSSEGTable : GenericTable {
  let FilterClass = "RISCVVSSEG";
  let CppTypeName = "VSSEGPseudo";
  let Fields = ["NF", "Masked", "Strided", "SEW", "LMUL", "Pseudo"];
  let PrimaryKey = ["NF", "Masked", "Strided", "SEW", "LMUL"];
  let PrimaryKeyName = "getVSSEGPseudo";
}

class RISCVVSXSEG<bits<4> N, bit M, bit O, bits<7> S, bits<3> L, bits<3> IL> {
  bits<4> NF = N;
  bits<1> Masked = M;
  bits<1> Ordered = O;
  bits<7> SEW = S;
  bits<3> LMUL = L;
  bits<3> IndexLMUL = IL;
  Pseudo Pseudo = !cast<Pseudo>(NAME);
}

def RISCVVSXSEGTable : GenericTable {
  let FilterClass = "RISCVVSXSEG";
  let CppTypeName = "VSXSEGPseudo";
  let Fields = ["NF", "Masked", "Ordered", "SEW", "LMUL", "IndexLMUL", "Pseudo"];
  let PrimaryKey = ["NF", "Masked", "Ordered", "SEW", "LMUL", "IndexLMUL"];
  let PrimaryKeyName = "getVSXSEGPseudo";
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
                 !subst("_COMMUTABLE", "",
                 !subst("F16", "F",
                 !subst("F32", "F",
                 !subst("F64", "F",
                 !subst("Pseudo", "", PseudoInst))))))))))))))))))));
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

class VPseudoUSLoadNoMask<VReg RetClass, bits<7> EEW, bit isFF> :
      Pseudo<(outs RetClass:$rd),
             (ins GPR:$rs1, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLE</*Masked*/0, /*Strided*/0, /*FF*/isFF, EEW, VLMul> {
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

class VPseudoUSLoadMask<VReg RetClass, bits<7> EEW, bit isFF> :
      Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
              (ins GetVRegNoV0<RetClass>.R:$merge,
                   GPR:$rs1,
                   VMaskOp:$vm, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLE</*Masked*/1, /*Strided*/0, /*FF*/isFF, EEW, VLMul> {
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

class VPseudoSLoadNoMask<VReg RetClass, bits<7> EEW>:
      Pseudo<(outs RetClass:$rd),
             (ins GPR:$rs1, GPR:$rs2, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLE</*Masked*/0, /*Strided*/1, /*FF*/0, EEW, VLMul> {
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

class VPseudoSLoadMask<VReg RetClass, bits<7> EEW>:
      Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
              (ins GetVRegNoV0<RetClass>.R:$merge,
                   GPR:$rs1, GPR:$rs2,
                   VMaskOp:$vm, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLE</*Masked*/1, /*Strided*/1, /*FF*/0, EEW, VLMul> {
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

class VPseudoILoadNoMask<VReg RetClass, VReg IdxClass, bits<7> EEW, bits<3> LMUL,
                         bit Ordered>:
      Pseudo<(outs RetClass:$rd),
             (ins GPR:$rs1, IdxClass:$rs2, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLX</*Masked*/0, Ordered, EEW, VLMul, LMUL> {
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

class VPseudoILoadMask<VReg RetClass, VReg IdxClass, bits<7> EEW, bits<3> LMUL,
                       bit Ordered>:
      Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
              (ins GetVRegNoV0<RetClass>.R:$merge,
                   GPR:$rs1, IdxClass:$rs2,
                   VMaskOp:$vm, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLX</*Masked*/1, Ordered, EEW, VLMul, LMUL> {
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

class VPseudoUSStoreNoMask<VReg StClass, bits<7> EEW>:
      Pseudo<(outs),
              (ins StClass:$rd, GPR:$rs1, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSE</*Masked*/0, /*Strided*/0, EEW, VLMul> {
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

class VPseudoUSStoreMask<VReg StClass, bits<7> EEW>:
      Pseudo<(outs),
              (ins StClass:$rd, GPR:$rs1, VMaskOp:$vm, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSE</*Masked*/1, /*Strided*/0, EEW, VLMul> {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoSStoreNoMask<VReg StClass, bits<7> EEW>:
      Pseudo<(outs),
              (ins StClass:$rd, GPR:$rs1, GPR:$rs2, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSE</*Masked*/0, /*Strided*/1, EEW, VLMul> {
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

class VPseudoSStoreMask<VReg StClass, bits<7> EEW>:
      Pseudo<(outs),
              (ins StClass:$rd, GPR:$rs1, GPR:$rs2, VMaskOp:$vm, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSE</*Masked*/1, /*Strided*/1, EEW, VLMul> {
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

class VPseudoIStoreNoMask<VReg StClass, VReg IdxClass, bits<7> EEW, bits<3> LMUL,
                          bit Ordered>:
      Pseudo<(outs),
              (ins StClass:$rd, GPR:$rs1, IdxClass:$rs2, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSX</*Masked*/0, Ordered, EEW, VLMul, LMUL> {
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

class VPseudoIStoreMask<VReg StClass, VReg IdxClass, bits<7> EEW, bits<3> LMUL,
                        bit Ordered>:
      Pseudo<(outs),
              (ins StClass:$rd, GPR:$rs1, IdxClass:$rs2, VMaskOp:$vm, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSX</*Masked*/1, Ordered, EEW, VLMul, LMUL> {
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
                        RegisterClass Op1Class,
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
                           RegisterClass Op1Class,
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

class VPseudoAMOWDNoMask<VReg RetClass,
                         VReg Op1Class> :
        Pseudo<(outs GetVRegNoV0<RetClass>.R:$vd_wd),
               (ins GPR:$rs1,
                    Op1Class:$vs2,
                    GetVRegNoV0<RetClass>.R:$vd,
                    GPR:$vl, ixlenimm:$sew), []>,
        RISCVVPseudo {
  let mayLoad = 1;
  let mayStore = 1;
  let hasSideEffects = 1;
  let usesCustomInserter = 1;
  let Constraints = "$vd_wd = $vd";
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoAMOWDMask<VReg RetClass,
                       VReg Op1Class> :
        Pseudo<(outs GetVRegNoV0<RetClass>.R:$vd_wd),
               (ins GPR:$rs1,
                    Op1Class:$vs2,
                    GetVRegNoV0<RetClass>.R:$vd,
                    VMaskOp:$vm, GPR:$vl, ixlenimm:$sew), []>,
        RISCVVPseudo {
  let mayLoad = 1;
  let mayStore = 1;
  let hasSideEffects = 1;
  let usesCustomInserter = 1;
  let Constraints = "$vd_wd = $vd";
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

multiclass VPseudoAMOEI<int eew> {
  // Standard scalar AMO supports 32, 64, and 128 Mem data bits,
  // and in the base vector "V" extension, only SEW up to ELEN = max(XLEN, FLEN)
  // are required to be supported.
  // therefore only [32, 64] is allowed here.
  foreach sew = [32, 64] in {
    foreach lmul = MxSet<sew>.m in {
      defvar octuple_lmul = octuple_from_str<lmul.MX>.ret;
      // Calculate emul = eew * lmul / sew
      defvar octuple_emul = !srl(!mul(eew, octuple_lmul), shift_amount<sew>.val);
      if !and(!ge(octuple_emul, 1), !le(octuple_emul, 64)) then {
        defvar emulMX = octuple_to_str<octuple_emul>.ret;
        defvar emul= !cast<LMULInfo>("V_" # emulMX);
        let VLMul = lmul.value in {
          def "_WD_" # lmul.MX # "_" # emulMX : VPseudoAMOWDNoMask<lmul.vrclass, emul.vrclass>;
          def "_WD_" # lmul.MX # "_" # emulMX # "_MASK" : VPseudoAMOWDMask<lmul.vrclass, emul.vrclass>;
        }
      }
    }
  }
}

multiclass VPseudoAMO {
  foreach eew = EEWList in
    defm "EI" # eew : VPseudoAMOEI<eew>;
}

class VPseudoUSSegLoadNoMask<VReg RetClass, bits<7> EEW, bits<4> NF, bit isFF>:
      Pseudo<(outs RetClass:$rd),
             (ins GPR:$rs1, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLSEG<NF, /*Masked*/0, /*Strided*/0, /*FF*/isFF, EEW, VLMul> {
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

class VPseudoUSSegLoadMask<VReg RetClass, bits<7> EEW, bits<4> NF, bit isFF>:
      Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
             (ins GetVRegNoV0<RetClass>.R:$merge, GPR:$rs1,
                  VMaskOp:$vm, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLSEG<NF, /*Masked*/1, /*Strided*/0, /*FF*/isFF, EEW, VLMul> {
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

class VPseudoSSegLoadNoMask<VReg RetClass, bits<7> EEW, bits<4> NF>:
      Pseudo<(outs RetClass:$rd),
             (ins GPR:$rs1, GPR:$offset, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLSEG<NF, /*Masked*/0, /*Strided*/1, /*FF*/0, EEW, VLMul> {
  let mayLoad = 1;
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

class VPseudoSSegLoadMask<VReg RetClass, bits<7> EEW, bits<4> NF>:
      Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
             (ins GetVRegNoV0<RetClass>.R:$merge, GPR:$rs1,
                  GPR:$offset, VMaskOp:$vm, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLSEG<NF, /*Masked*/1, /*Strided*/1, /*FF*/0, EEW, VLMul> {
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

class VPseudoISegLoadNoMask<VReg RetClass, VReg IdxClass, bits<7> EEW, bits<3> LMUL,
                            bits<4> NF, bit Ordered>:
      Pseudo<(outs RetClass:$rd),
             (ins GPR:$rs1, IdxClass:$offset, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLXSEG<NF, /*Masked*/0, Ordered, EEW, VLMul, LMUL> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  // For vector indexed segment loads, the destination vector register groups
  // cannot overlap the source vector register group
  let Constraints = "@earlyclobber $rd";
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasDummyMask = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoISegLoadMask<VReg RetClass, VReg IdxClass, bits<7> EEW, bits<3> LMUL,
                          bits<4> NF, bit Ordered>:
      Pseudo<(outs GetVRegNoV0<RetClass>.R:$rd),
             (ins GetVRegNoV0<RetClass>.R:$merge, GPR:$rs1,
                  IdxClass:$offset, VMaskOp:$vm, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVLXSEG<NF, /*Masked*/1, Ordered, EEW, VLMul, LMUL> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  // For vector indexed segment loads, the destination vector register groups
  // cannot overlap the source vector register group
  let Constraints = "@earlyclobber $rd, $rd = $merge";
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasMergeOp = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoUSSegStoreNoMask<VReg ValClass, bits<7> EEW, bits<4> NF>:
      Pseudo<(outs),
             (ins ValClass:$rd, GPR:$rs1, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSSEG<NF, /*Masked*/0, /*Strided*/0, EEW, VLMul> {
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

class VPseudoUSSegStoreMask<VReg ValClass, bits<7> EEW, bits<4> NF>:
      Pseudo<(outs),
             (ins ValClass:$rd, GPR:$rs1,
                  VMaskOp:$vm, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSSEG<NF, /*Masked*/1, /*Strided*/0, EEW, VLMul> {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoSSegStoreNoMask<VReg ValClass, bits<7> EEW, bits<4> NF>:
      Pseudo<(outs),
             (ins ValClass:$rd, GPR:$rs1, GPR: $offset, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSSEG<NF, /*Masked*/0, /*Strided*/1, EEW, VLMul> {
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

class VPseudoSSegStoreMask<VReg ValClass, bits<7> EEW, bits<4> NF>:
      Pseudo<(outs),
             (ins ValClass:$rd, GPR:$rs1, GPR: $offset,
                  VMaskOp:$vm, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSSEG<NF, /*Masked*/1, /*Strided*/1, EEW, VLMul> {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoISegStoreNoMask<VReg ValClass, VReg IdxClass, bits<7> EEW, bits<3> LMUL,
                             bits<4> NF, bit Ordered>:
      Pseudo<(outs),
             (ins ValClass:$rd, GPR:$rs1, IdxClass: $index,
                  GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSXSEG<NF, /*Masked*/0, Ordered, EEW, VLMul, LMUL> {
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

class VPseudoISegStoreMask<VReg ValClass, VReg IdxClass, bits<7> EEW, bits<3> LMUL,
                           bits<4> NF, bit Ordered>:
      Pseudo<(outs),
             (ins ValClass:$rd, GPR:$rs1, IdxClass: $index,
                  VMaskOp:$vm, GPR:$vl, ixlenimm:$sew),[]>,
      RISCVVPseudo,
      RISCVVSXSEG<NF, /*Masked*/1, Ordered, EEW, VLMul, LMUL> {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let usesCustomInserter = 1;
  let Uses = [VL, VTYPE];
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

multiclass VPseudoUSLoad<bit isFF> {
  foreach eew = EEWList in {
    foreach lmul = MxSet<eew>.m in {
      defvar LInfo = lmul.MX;
      defvar vreg = lmul.vrclass;
      defvar FFStr = !if(isFF, "FF", "");
      let VLMul = lmul.value in {
        def "E" # eew # FFStr # "_V_" # LInfo :
          VPseudoUSLoadNoMask<vreg, eew, isFF>;
        def "E" # eew # FFStr # "_V_" # LInfo # "_MASK" :
          VPseudoUSLoadMask<vreg, eew, isFF>;
      }
    }
  }
}

multiclass VPseudoLoadMask {
  foreach mti = AllMasks in {
    let VLMul = mti.LMul.value in {
      def "_V_" # mti.BX : VPseudoUSLoadNoMask<VR, /*EEW*/1, /*isFF*/0>;
    }
  }
}

multiclass VPseudoSLoad {
  foreach eew = EEWList in {
    foreach lmul = MxSet<eew>.m in {
      defvar LInfo = lmul.MX;
      defvar vreg = lmul.vrclass;
      let VLMul = lmul.value in {
        def "E" # eew # "_V_" # LInfo : VPseudoSLoadNoMask<vreg, eew>;
        def "E" # eew # "_V_" # LInfo # "_MASK" : VPseudoSLoadMask<vreg, eew>;
      }
    }
  }
}

multiclass VPseudoILoad<bit Ordered> {
  foreach eew = EEWList in {
    foreach sew = EEWList in {
      foreach lmul = MxSet<sew>.m in {
        defvar octuple_lmul = octuple_from_str<lmul.MX>.ret;
        // Calculate emul = eew * lmul / sew
        defvar octuple_emul = !srl(!mul(eew, octuple_lmul), shift_amount<sew>.val);
        if !and(!ge(octuple_emul, 1), !le(octuple_emul, 64)) then {
          defvar LInfo = lmul.MX;
          defvar IdxLInfo = octuple_to_str<octuple_emul>.ret;
          defvar idx_lmul = !cast<LMULInfo>("V_" # IdxLInfo);
          defvar Vreg = lmul.vrclass;
          defvar IdxVreg = idx_lmul.vrclass;
          let VLMul = lmul.value in {
            def "EI" # eew # "_V_" # IdxLInfo # "_" # LInfo :
              VPseudoILoadNoMask<Vreg, IdxVreg, eew, idx_lmul.value, Ordered>;
            def "EI" # eew # "_V_" # IdxLInfo # "_" # LInfo # "_MASK" :
              VPseudoILoadMask<Vreg, IdxVreg, eew, idx_lmul.value, Ordered>;
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
        def "E" # eew # "_V_" # LInfo : VPseudoUSStoreNoMask<vreg, eew>;
        def "E" # eew # "_V_" # LInfo # "_MASK" : VPseudoUSStoreMask<vreg, eew>;
      }
    }
  }
}

multiclass VPseudoStoreMask {
  foreach mti = AllMasks in {
    let VLMul = mti.LMul.value in {
      def "_V_" # mti.BX : VPseudoUSStoreNoMask<VR, /*EEW*/1>;
    }
  }
}

multiclass VPseudoSStore {
  foreach eew = EEWList in {
    foreach lmul = MxSet<eew>.m in {
      defvar LInfo = lmul.MX;
      defvar vreg = lmul.vrclass;
      let VLMul = lmul.value in {
        def "E" # eew # "_V_" # LInfo : VPseudoSStoreNoMask<vreg, eew>;
        def "E" # eew # "_V_" # LInfo # "_MASK" : VPseudoSStoreMask<vreg, eew>;
      }
    }
  }
}

multiclass VPseudoIStore<bit Ordered> {
  foreach eew = EEWList in {
    foreach sew = EEWList in {
      foreach lmul = MxSet<sew>.m in {
        defvar octuple_lmul = octuple_from_str<lmul.MX>.ret;
        // Calculate emul = eew * lmul / sew
        defvar octuple_emul = !srl(!mul(eew, octuple_lmul), shift_amount<sew>.val);
        if !and(!ge(octuple_emul, 1), !le(octuple_emul, 64)) then {
          defvar LInfo = lmul.MX;
          defvar IdxLInfo = octuple_to_str<octuple_emul>.ret;
          defvar idx_lmul = !cast<LMULInfo>("V_" # IdxLInfo);
          defvar Vreg = lmul.vrclass;
          defvar IdxVreg = idx_lmul.vrclass;
          let VLMul = lmul.value in {
            def "EI" # eew # "_V_" # IdxLInfo # "_" # LInfo :
              VPseudoIStoreNoMask<Vreg, IdxVreg, eew, idx_lmul.value, Ordered>;
            def "EI" # eew # "_V_" # IdxLInfo # "_" # LInfo # "_MASK" :
              VPseudoIStoreMask<Vreg, IdxVreg, eew, idx_lmul.value, Ordered>;
          }
        }
      }
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
      def _VM # "_" # m.MX : VPseudoUnaryAnyMask<m.vrclass, m.vrclass>;
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

multiclass VPseudoBinaryEmul<VReg RetClass,
                             VReg Op1Class,
                             DAGOperand Op2Class,
                             LMULInfo lmul,
                             LMULInfo emul,
                             string Constraint = ""> {
  let VLMul = lmul.value in {
    def "_" # lmul.MX # "_" # emul.MX : VPseudoBinaryNoMask<RetClass, Op1Class, Op2Class,
                                                            Constraint>;
    def "_" # lmul.MX # "_" # emul.MX # "_MASK" : VPseudoBinaryMask<RetClass, Op1Class, Op2Class,
                                                                    Constraint>;
  }
}

multiclass VPseudoBinaryV_VV<string Constraint = ""> {
  foreach m = MxList.m in
    defm _VV : VPseudoBinary<m.vrclass, m.vrclass, m.vrclass, m, Constraint>;
}

multiclass VPseudoBinaryV_VV_EEW<int eew, string Constraint = ""> {
  foreach m = MxList.m in {
    foreach sew = EEWList in {
      defvar octuple_lmul = octuple_from_str<m.MX>.ret;
      // emul = lmul * eew / sew
      defvar octuple_emul = !srl(!mul(octuple_lmul, eew), shift_amount<sew>.val);
      if !and(!ge(octuple_emul, 1), !le(octuple_emul, 64)) then {
        defvar emulMX = octuple_to_str<octuple_emul>.ret;
        defvar emul = !cast<LMULInfo>("V_" # emulMX);
        defm _VV : VPseudoBinaryEmul<m.vrclass, m.vrclass, emul.vrclass, m, emul, Constraint>;
      }
    }
  }
}

multiclass VPseudoBinaryV_VX<string Constraint = ""> {
  foreach m = MxList.m in
    defm "_VX" : VPseudoBinary<m.vrclass, m.vrclass, GPR, m, Constraint>;
}

multiclass VPseudoBinaryV_VF<string Constraint = ""> {
  foreach m = MxList.m in
    foreach f = FPList.fpinfo in
      defm "_V" # f.FX : VPseudoBinary<m.vrclass, m.vrclass,
                                       f.fprclass, m, Constraint>;
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

multiclass VPseudoBinaryW_VX {
  foreach m = MxList.m[0-5] in
    defm "_VX" : VPseudoBinary<m.wvrclass, m.vrclass, GPR, m,
                               "@earlyclobber $rd">;
}

multiclass VPseudoBinaryW_VF {
  foreach m = MxList.m[0-5] in
    foreach f = FPList.fpinfo[0-1] in
      defm "_V" # f.FX : VPseudoBinary<m.wvrclass, m.vrclass,
                                       f.fprclass, m,
                                       "@earlyclobber $rd">;
}

multiclass VPseudoBinaryW_WV {
  foreach m = MxList.m[0-5] in
    defm _WV : VPseudoBinary<m.wvrclass, m.wvrclass, m.vrclass, m,
                             "@earlyclobber $rd">;
}

multiclass VPseudoBinaryW_WX {
  foreach m = MxList.m[0-5] in
    defm "_WX" : VPseudoBinary<m.wvrclass, m.wvrclass, GPR, m,
                               "@earlyclobber $rd">;
}

multiclass VPseudoBinaryW_WF {
  foreach m = MxList.m[0-5] in
    foreach f = FPList.fpinfo[0-1] in
      defm "_W" # f.FX : VPseudoBinary<m.wvrclass, m.wvrclass,
                                       f.fprclass, m,
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
                             string Constraint = ""> {
  foreach m = MxList.m in
    def "_VX" # !if(CarryIn, "M", "") # "_" # m.MX :
      VPseudoBinaryCarryIn<!if(CarryOut, VR,
                           !if(!and(CarryIn, !not(CarryOut)),
                               GetVRegNoV0<m.vrclass>.R, m.vrclass)),
                           m.vrclass, GPR, m, CarryIn, Constraint>;
}

multiclass VPseudoBinaryV_FM {
  foreach m = MxList.m in
    foreach f = FPList.fpinfo in
      def "_V" # f.FX # "M_" # m.MX :
        VPseudoBinaryCarryIn<GetVRegNoV0<m.vrclass>.R,
                             m.vrclass, f.fprclass, m, /*CarryIn=*/1, "">;
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
    foreach f = FPList.fpinfo in {
      let VLMul = m.value in {
        def "_" # f.FX # "_" # m.MX : VPseudoUnaryNoDummyMask<m.vrclass, f.fprclass>;
      }
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

multiclass VPseudoBinaryM_VX {
  foreach m = MxList.m in
    defm "_VX" :
      VPseudoBinary<VR, m.vrclass, GPR, m, "@earlyclobber $rd">;
}

multiclass VPseudoBinaryM_VF {
  foreach m = MxList.m in
    foreach f = FPList.fpinfo in
      defm "_V" # f.FX :
        VPseudoBinary<VR, m.vrclass, f.fprclass, m, "@earlyclobber $rd">;
}

multiclass VPseudoBinaryM_VI {
  foreach m = MxList.m in
    defm _VI : VPseudoBinary<VR, m.vrclass, simm5, m, "@earlyclobber $rd">;
}

multiclass VPseudoBinaryV_VV_VX_VI<Operand ImmType = simm5, string Constraint = ""> {
  defm "" : VPseudoBinaryV_VV<Constraint>;
  defm "" : VPseudoBinaryV_VX<Constraint>;
  defm "" : VPseudoBinaryV_VI<ImmType, Constraint>;
}

multiclass VPseudoBinaryV_VV_VX {
  defm "" : VPseudoBinaryV_VV;
  defm "" : VPseudoBinaryV_VX;
}

multiclass VPseudoBinaryV_VV_VF {
  defm "" : VPseudoBinaryV_VV;
  defm "" : VPseudoBinaryV_VF;
}

multiclass VPseudoBinaryV_VX_VI<Operand ImmType = simm5> {
  defm "" : VPseudoBinaryV_VX;
  defm "" : VPseudoBinaryV_VI<ImmType>;
}

multiclass VPseudoBinaryW_VV_VX {
  defm "" : VPseudoBinaryW_VV;
  defm "" : VPseudoBinaryW_VX;
}

multiclass VPseudoBinaryW_VV_VF {
  defm "" : VPseudoBinaryW_VV;
  defm "" : VPseudoBinaryW_VF;
}

multiclass VPseudoBinaryW_WV_WX {
  defm "" : VPseudoBinaryW_WV;
  defm "" : VPseudoBinaryW_WX;
}

multiclass VPseudoBinaryW_WV_WF {
  defm "" : VPseudoBinaryW_WV;
  defm "" : VPseudoBinaryW_WF;
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
                          RegisterClass Op1Class,
                          DAGOperand Op2Class,
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

multiclass VPseudoTernaryV_VX_AAXA<string Constraint = ""> {
  foreach m = MxList.m in
    defm "_VX" : VPseudoTernary<m.vrclass, GPR, m.vrclass, m, Constraint>;
}

multiclass VPseudoTernaryV_VF_AAXA<string Constraint = ""> {
  foreach m = MxList.m in
    foreach f = FPList.fpinfo in
      defm "_V" # f.FX : VPseudoTernary<m.vrclass, f.fprclass, m.vrclass,
                                        m, Constraint>;
}

multiclass VPseudoTernaryW_VV {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxList.m[0-5] in
    defm _VV : VPseudoTernary<m.wvrclass, m.vrclass, m.vrclass, m, constraint>;
}

multiclass VPseudoTernaryW_VX {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxList.m[0-5] in
    defm "_VX" : VPseudoTernary<m.wvrclass, GPR, m.vrclass, m, constraint>;
}

multiclass VPseudoTernaryW_VF {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxList.m[0-5] in
    foreach f = FPList.fpinfo[0-1] in
      defm "_V" # f.FX : VPseudoTernary<m.wvrclass, f.fprclass, m.vrclass, m,
                                        constraint>;
}

multiclass VPseudoTernaryV_VI<Operand ImmType = simm5, string Constraint = ""> {
  foreach m = MxList.m in
    defm _VI : VPseudoTernary<m.vrclass, m.vrclass, ImmType, m, Constraint>;
}

multiclass VPseudoTernaryV_VV_VX_AAXA<string Constraint = ""> {
  defm "" : VPseudoTernaryV_VV<Constraint>;
  defm "" : VPseudoTernaryV_VX_AAXA<Constraint>;
}

multiclass VPseudoTernaryV_VV_VF_AAXA<string Constraint = ""> {
  defm "" : VPseudoTernaryV_VV<Constraint>;
  defm "" : VPseudoTernaryV_VF_AAXA<Constraint>;

  foreach m = MxList.m in {
    // Add a commutable version for use by IR fma.
    // NOTE: We need this because we use a tail undisturbed policy on the
    // intrinsic version so we can't commute those instructions since it would
    // change which input operand is tied to the destination. That would
    // remove user control of the tail elements.
    let isCommutable = 1, ForceTailAgnostic = true, VLMul = m.value in {
      def "_VV_" # m.MX # "_COMMUTABLE" : VPseudoTernaryNoMask<m.vrclass,
                                                               m.vrclass,
                                                               m.vrclass,
                                                               Constraint>;
      foreach f = FPList.fpinfo in
        def "_V" # f.FX # "_" # m.MX # "_COMMUTABLE" :
           VPseudoTernaryNoMask<m.vrclass, f.fprclass, m.vrclass, Constraint>;
    }
  }
}

multiclass VPseudoTernaryV_VX_VI<Operand ImmType = simm5, string Constraint = ""> {
  defm "" : VPseudoTernaryV_VX<Constraint>;
  defm "" : VPseudoTernaryV_VI<ImmType, Constraint>;
}

multiclass VPseudoTernaryW_VV_VX {
  defm "" : VPseudoTernaryW_VV;
  defm "" : VPseudoTernaryW_VX;
}

multiclass VPseudoTernaryW_VV_VF {
  defm "" : VPseudoTernaryW_VV;
  defm "" : VPseudoTernaryW_VF;
}

multiclass VPseudoBinaryM_VV_VX_VI {
  defm "" : VPseudoBinaryM_VV;
  defm "" : VPseudoBinaryM_VX;
  defm "" : VPseudoBinaryM_VI;
}

multiclass VPseudoBinaryM_VV_VX {
  defm "" : VPseudoBinaryM_VV;
  defm "" : VPseudoBinaryM_VX;
}

multiclass VPseudoBinaryM_VV_VF {
  defm "" : VPseudoBinaryM_VV;
  defm "" : VPseudoBinaryM_VF;
}

multiclass VPseudoBinaryM_VX_VI {
  defm "" : VPseudoBinaryM_VX;
  defm "" : VPseudoBinaryM_VI;
}

multiclass VPseudoReductionV_VS {
  foreach m = MxList.m in {
    let ForceTailAgnostic = true in
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

multiclass VPseudoUSSegLoad<bit isFF> {
  foreach eew = EEWList in {
    foreach lmul = MxSet<eew>.m in {
      defvar LInfo = lmul.MX;
      let VLMul = lmul.value in {
        foreach nf = NFSet<lmul>.L in {
          defvar vreg = SegRegClass<lmul, nf>.RC;
          defvar FFStr = !if(isFF, "FF", "");
          def nf # "E" # eew # FFStr # "_V_" # LInfo :
            VPseudoUSSegLoadNoMask<vreg, eew, nf, isFF>;
          def nf # "E" # eew # FFStr # "_V_" # LInfo # "_MASK" :
            VPseudoUSSegLoadMask<vreg, eew, nf, isFF>;
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
          def nf # "E" # eew # "_V_" # LInfo : VPseudoSSegLoadNoMask<vreg, eew, nf>;
          def nf # "E" # eew # "_V_" # LInfo # "_MASK" : VPseudoSSegLoadMask<vreg, eew, nf>;
        }
      }
    }
  }
}

multiclass VPseudoISegLoad<bit Ordered> {
  foreach idx_eew = EEWList in {
    foreach sew = EEWList in {
      foreach val_lmul = MxSet<sew>.m in {
        defvar octuple_lmul = octuple_from_str<val_lmul.MX>.ret;
        // Calculate emul = eew * lmul / sew
        defvar octuple_emul = !srl(!mul(idx_eew, octuple_lmul), shift_amount<sew>.val);
        if !and(!ge(octuple_emul, 1), !le(octuple_emul, 64)) then {
          defvar ValLInfo = val_lmul.MX;
          defvar IdxLInfo = octuple_to_str<octuple_emul>.ret;
          defvar idx_lmul = !cast<LMULInfo>("V_" # IdxLInfo);
          defvar Vreg = val_lmul.vrclass;
          defvar IdxVreg = idx_lmul.vrclass;
          let VLMul = val_lmul.value in {
            foreach nf = NFSet<val_lmul>.L in {
              defvar ValVreg = SegRegClass<val_lmul, nf>.RC;
              def nf # "EI" # idx_eew # "_V_" # IdxLInfo # "_" # ValLInfo :
                VPseudoISegLoadNoMask<ValVreg, IdxVreg, idx_eew, idx_lmul.value,
                                      nf, Ordered>;
              def nf # "EI" # idx_eew # "_V_" # IdxLInfo # "_" # ValLInfo # "_MASK" :
                VPseudoISegLoadMask<ValVreg, IdxVreg, idx_eew, idx_lmul.value,
                                    nf, Ordered>;
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
          def nf # "E" # eew # "_V_" # LInfo : VPseudoUSSegStoreNoMask<vreg, eew, nf>;
          def nf # "E" # eew # "_V_" # LInfo # "_MASK" : VPseudoUSSegStoreMask<vreg, eew, nf>;
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
          def nf # "E" # eew # "_V_" # LInfo : VPseudoSSegStoreNoMask<vreg, eew, nf>;
          def nf # "E" # eew # "_V_" # LInfo # "_MASK" : VPseudoSSegStoreMask<vreg, eew, nf>;
        }
      }
    }
  }
}

multiclass VPseudoISegStore<bit Ordered> {
  foreach idx_eew = EEWList in {
    foreach sew = EEWList in {
      foreach val_lmul = MxSet<sew>.m in {
        defvar octuple_lmul = octuple_from_str<val_lmul.MX>.ret;
        // Calculate emul = eew * lmul / sew
        defvar octuple_emul = !srl(!mul(idx_eew, octuple_lmul), shift_amount<sew>.val);
        if !and(!ge(octuple_emul, 1), !le(octuple_emul, 64)) then {
          defvar ValLInfo = val_lmul.MX;
          defvar IdxLInfo = octuple_to_str<octuple_emul>.ret;
          defvar idx_lmul = !cast<LMULInfo>("V_" # IdxLInfo);
          defvar Vreg = val_lmul.vrclass;
          defvar IdxVreg = idx_lmul.vrclass;
          let VLMul = val_lmul.value in {
            foreach nf = NFSet<val_lmul>.L in {
              defvar ValVreg = SegRegClass<val_lmul, nf>.RC;
              def nf # "EI" # idx_eew # "_V_" # IdxLInfo # "_" # ValLInfo :
                VPseudoISegStoreNoMask<ValVreg, IdxVreg, idx_eew, idx_lmul.value,
                                       nf, Ordered>;
              def nf # "EI" # idx_eew # "_V_" # IdxLInfo # "_" # ValLInfo # "_MASK" :
                VPseudoISegStoreMask<ValVreg, IdxVreg, idx_eew, idx_lmul.value,
                                     nf, Ordered>;
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
                   (op2_type op2_reg_class:$rs2),
                   (XLenVT (VLOp GPR:$vl)))),
                   (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX)
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
                   (XLenVT (VLOp GPR:$vl)))),
                   (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX#"_MASK")
                   (result_type result_reg_class:$merge),
                   (op2_type op2_reg_class:$rs2),
                   (mask_type V0), GPR:$vl, sew)>;

class VPatMaskUnaryNoMask<string intrinsic_name,
                          string inst,
                          MTypeInfo mti> :
  Pat<(mti.Mask (!cast<Intrinsic>(intrinsic_name)
                (mti.Mask VR:$rs2),
                (XLenVT (VLOp GPR:$vl)))),
                (!cast<Instruction>(inst#"_M_"#mti.BX)
                (mti.Mask VR:$rs2),
                GPR:$vl, mti.SEW)>;

class VPatMaskUnaryMask<string intrinsic_name,
                        string inst,
                        MTypeInfo mti> :
  Pat<(mti.Mask (!cast<Intrinsic>(intrinsic_name#"_mask")
                (mti.Mask VR:$merge),
                (mti.Mask VR:$rs2),
                (mti.Mask V0),
                (XLenVT (VLOp GPR:$vl)))),
                (!cast<Instruction>(inst#"_M_"#mti.BX#"_MASK")
                (mti.Mask VR:$merge),
                (mti.Mask VR:$rs2),
                (mti.Mask V0), GPR:$vl, mti.SEW)>;

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
                   (XLenVT (VLOp GPR:$vl)))),
                   (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX)
                   (result_type result_reg_class:$merge),
                   (op1_type op1_reg_class:$rs1),
                   (mask_type VR:$rs2),
                   GPR:$vl, sew)>;

class VPatBinaryNoMask<string intrinsic_name,
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
                   (XLenVT (VLOp GPR:$vl)))),
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
                   (XLenVT (VLOp GPR:$vl)))),
                   (!cast<Instruction>(inst#"_MASK")
                   (result_type result_reg_class:$merge),
                   (op1_type op1_reg_class:$rs1),
                   (op2_type op2_kind:$rs2),
                   (mask_type V0), GPR:$vl, sew)>;

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
                    (XLenVT (VLOp GPR:$vl)))),
                   (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX)
                    result_reg_class:$rs3,
                    (op1_type op1_reg_class:$rs1),
                    op2_kind:$rs2,
                    GPR:$vl, sew)>;

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
                    (XLenVT (VLOp GPR:$vl)))),
                   (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX # "_MASK")
                    result_reg_class:$rs3,
                    (op1_type op1_reg_class:$rs1),
                    op2_kind:$rs2,
                    (mask_type V0),
                    GPR:$vl, sew)>;

class VPatAMOWDNoMask<string intrinsic_name,
                    string inst,
                    ValueType result_type,
                    ValueType op1_type,
                    int sew,
                    LMULInfo vlmul,
                    LMULInfo emul,
                    VReg op1_reg_class> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                    GPR:$rs1,
                    (op1_type op1_reg_class:$vs2),
                    (result_type vlmul.vrclass:$vd),
                    (XLenVT (VLOp GPR:$vl)))),
                   (!cast<Instruction>(inst # "_WD_" # vlmul.MX # "_" # emul.MX)
                    $rs1, $vs2, $vd,
                    GPR:$vl, sew)>;

class VPatAMOWDMask<string intrinsic_name,
                    string inst,
                    ValueType result_type,
                    ValueType op1_type,
                    ValueType mask_type,
                    int sew,
                    LMULInfo vlmul,
                    LMULInfo emul,
                    VReg op1_reg_class> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic_name # "_mask")
                    GPR:$rs1,
                    (op1_type op1_reg_class:$vs2),
                    (result_type vlmul.vrclass:$vd),
                    (mask_type V0),
                    (XLenVT (VLOp GPR:$vl)))),
                   (!cast<Instruction>(inst # "_WD_" # vlmul.MX # "_" # emul.MX # "_MASK")
                    $rs1, $vs2, $vd,
                    (mask_type V0), GPR:$vl, sew)>;

multiclass VPatUnaryS_M<string intrinsic_name,
                             string inst>
{
  foreach mti = AllMasks in {
    def : Pat<(XLenVT (!cast<Intrinsic>(intrinsic_name)
                      (mti.Mask VR:$rs1), (XLenVT (VLOp GPR:$vl)))),
                      (!cast<Instruction>(inst#"_M_"#mti.BX) $rs1,
                      GPR:$vl, mti.SEW)>;
    def : Pat<(XLenVT (!cast<Intrinsic>(intrinsic_name # "_mask")
                      (mti.Mask VR:$rs1), (mti.Mask V0), (XLenVT (VLOp GPR:$vl)))),
                      (!cast<Instruction>(inst#"_M_"#mti.BX#"_MASK") $rs1,
                      (mti.Mask V0), GPR:$vl, mti.SEW)>;
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
                          (XLenVT (VLOp GPR:$vl)))),
                          (!cast<Instruction>(instruction#"_V_" # vti.LMul.MX)
                          GPR:$vl, vti.SEW)>;
    def : Pat<(vti.Vector (!cast<Intrinsic>(intrinsic # "_mask")
                          (vti.Vector vti.RegClass:$merge),
                          (vti.Mask V0), (XLenVT (VLOp GPR:$vl)))),
                          (!cast<Instruction>(instruction#"_V_" # vti.LMul.MX # "_MASK")
                          vti.RegClass:$merge, (vti.Mask V0),
                          GPR:$vl, vti.SEW)>;
  }
}

multiclass VPatNullaryM<string intrinsic, string inst> {
  foreach mti = AllMasks in
    def : Pat<(mti.Mask (!cast<Intrinsic>(intrinsic)
                        (XLenVT (VLOp GPR:$vl)))),
                        (!cast<Instruction>(inst#"_M_"#mti.BX)
                        GPR:$vl, mti.SEW)>;
}

multiclass VPatBinary<string intrinsic,
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
  def : VPatBinaryNoMask<intrinsic, inst, result_type, op1_type, op2_type,
                         sew, op1_reg_class, op2_kind>;
  def : VPatBinaryMask<intrinsic, inst, result_type, op1_type, op2_type,
                       mask_type, sew, result_reg_class, op1_reg_class,
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
                         (XLenVT (VLOp GPR:$vl)))),
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
                         (XLenVT (VLOp GPR:$vl)))),
                         (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX)
                         (op1_type op1_reg_class:$rs1),
                         (op2_type op2_kind:$rs2),
                         GPR:$vl, sew)>;
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
    defm : VPatBinary<intrinsic, instruction # "_VV_" # vti.LMul.MX,
                      vti.Vector, vti.Vector, vti.Vector,vti.Mask,
                      vti.SEW, vti.RegClass,
                      vti.RegClass, vti.RegClass>;
}

multiclass VPatBinaryV_VV_INT<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist> {
  foreach vti = vtilist in {
    defvar ivti = GetIntVTypeInfo<vti>.Vti;
    defm : VPatBinary<intrinsic, instruction # "_VV_" # vti.LMul.MX,
                      vti.Vector, vti.Vector, ivti.Vector, vti.Mask,
                      vti.SEW, vti.RegClass,
                      vti.RegClass, vti.RegClass>;
  }
}

multiclass VPatBinaryV_VV_INT_EEW<string intrinsic, string instruction,
                                  int eew, list<VTypeInfo> vtilist> {
  foreach vti = vtilist in {
    // emul = lmul * eew / sew
    defvar vlmul = vti.LMul;
    defvar octuple_lmul = octuple_from_str<vlmul.MX>.ret;
    defvar octuple_emul = !srl(!mul(octuple_lmul, eew), shift_amount<vti.SEW>.val);
    if !and(!ge(octuple_emul, 1), !le(octuple_emul, 64)) then {
      defvar emul_str = octuple_to_str<octuple_emul>.ret;
      defvar ivti = !cast<VTypeInfo>("VI" # eew # emul_str);
      defvar inst = instruction # "_VV_" # vti.LMul.MX # "_" # emul_str;
      defm : VPatBinary<intrinsic, inst,
                        vti.Vector, vti.Vector, ivti.Vector, vti.Mask,
                        vti.SEW, vti.RegClass,
                        vti.RegClass, ivti.RegClass>;
    }
  }
}

multiclass VPatBinaryV_VX<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist> {
  foreach vti = vtilist in {
    defvar kind = "V"#vti.ScalarSuffix;
    defm : VPatBinary<intrinsic, instruction#"_"#kind#"_"#vti.LMul.MX,
                      vti.Vector, vti.Vector, vti.Scalar, vti.Mask,
                      vti.SEW, vti.RegClass,
                      vti.RegClass, vti.ScalarRegClass>;
  }
}

multiclass VPatBinaryV_VX_INT<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    defm : VPatBinary<intrinsic, instruction # "_VX_" # vti.LMul.MX,
                      vti.Vector, vti.Vector, XLenVT, vti.Mask,
                      vti.SEW, vti.RegClass,
                      vti.RegClass, GPR>;
}

multiclass VPatBinaryV_VI<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist, Operand imm_type> {
  foreach vti = vtilist in
    defm : VPatBinary<intrinsic, instruction # "_VI_" # vti.LMul.MX,
                      vti.Vector, vti.Vector, XLenVT, vti.Mask,
                      vti.SEW, vti.RegClass,
                      vti.RegClass, imm_type>;
}

multiclass VPatBinaryM_MM<string intrinsic, string instruction> {
  foreach mti = AllMasks in
    def : VPatBinaryNoMask<intrinsic, instruction # "_MM_" # mti.LMul.MX,
                           mti.Mask, mti.Mask, mti.Mask,
                           mti.SEW, VR, VR>;
}

multiclass VPatBinaryW_VV<string intrinsic, string instruction,
                          list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    defm : VPatBinary<intrinsic, instruction # "_VV_" # Vti.LMul.MX,
                      Wti.Vector, Vti.Vector, Vti.Vector, Vti.Mask,
                      Vti.SEW, Wti.RegClass,
                      Vti.RegClass, Vti.RegClass>;
  }
}

multiclass VPatBinaryW_VX<string intrinsic, string instruction,
                          list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    defvar kind = "V"#Vti.ScalarSuffix;
    defm : VPatBinary<intrinsic, instruction#"_"#kind#"_"#Vti.LMul.MX,
                      Wti.Vector, Vti.Vector, Vti.Scalar, Vti.Mask,
                      Vti.SEW, Wti.RegClass,
                      Vti.RegClass, Vti.ScalarRegClass>;
  }
}

multiclass VPatBinaryW_WV<string intrinsic, string instruction,
                          list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    defm : VPatBinary<intrinsic, instruction # "_WV_" # Vti.LMul.MX,
                      Wti.Vector, Wti.Vector, Vti.Vector, Vti.Mask,
                      Vti.SEW, Wti.RegClass,
                      Wti.RegClass, Vti.RegClass>;
  }
}

multiclass VPatBinaryW_WX<string intrinsic, string instruction,
                          list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    defvar kind = "W"#Vti.ScalarSuffix;
    defm : VPatBinary<intrinsic, instruction#"_"#kind#"_"#Vti.LMul.MX,
                      Wti.Vector, Wti.Vector, Vti.Scalar, Vti.Mask,
                      Vti.SEW, Wti.RegClass,
                      Wti.RegClass, Vti.ScalarRegClass>;
  }
}

multiclass VPatBinaryV_WV<string intrinsic, string instruction,
                          list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    defm : VPatBinary<intrinsic, instruction # "_WV_" # Vti.LMul.MX,
                      Vti.Vector, Wti.Vector, Vti.Vector, Vti.Mask,
                      Vti.SEW, Vti.RegClass,
                      Wti.RegClass, Vti.RegClass>;
  }
}

multiclass VPatBinaryV_WX<string intrinsic, string instruction,
                          list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    defvar kind = "W"#Vti.ScalarSuffix;
    defm : VPatBinary<intrinsic, instruction#"_"#kind#"_"#Vti.LMul.MX,
                      Vti.Vector, Wti.Vector, Vti.Scalar, Vti.Mask,
                      Vti.SEW, Vti.RegClass,
                      Wti.RegClass, Vti.ScalarRegClass>;
  }
}

multiclass VPatBinaryV_WI<string intrinsic, string instruction,
                          list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    defm : VPatBinary<intrinsic, instruction # "_WI_" # Vti.LMul.MX,
                      Vti.Vector, Wti.Vector, XLenVT, Vti.Mask,
                      Vti.SEW, Vti.RegClass,
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
                             "V"#vti.ScalarSuffix#"M",
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
    defm : VPatBinary<intrinsic, instruction # "_VV_" # vti.LMul.MX,
                      vti.Mask, vti.Vector, vti.Vector, vti.Mask,
                      vti.SEW, VR,
                      vti.RegClass, vti.RegClass>;
}

multiclass VPatBinaryM_VX<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist> {
  foreach vti = vtilist in {
    defvar kind = "V"#vti.ScalarSuffix;
    defm : VPatBinary<intrinsic, instruction#"_"#kind#"_"#vti.LMul.MX,
                      vti.Mask, vti.Vector, vti.Scalar, vti.Mask,
                      vti.SEW, VR,
                      vti.RegClass, vti.ScalarRegClass>;
  }
}

multiclass VPatBinaryM_VI<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    defm : VPatBinary<intrinsic, instruction # "_VI_" # vti.LMul.MX,
                      vti.Mask, vti.Vector, XLenVT, vti.Mask,
                      vti.SEW, VR,
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
                       "V"#vti.ScalarSuffix,
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
                       "V"#vti.ScalarSuffix,
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
  defm "" : VPatBinaryV_VV_INT<intrinsic#"_vv", instruction, vtilist>;
  defm "" : VPatBinaryV_VX_INT<intrinsic#"_vx", instruction, vtilist>;
  defm "" : VPatBinaryV_VI<intrinsic#"_vx", instruction, vtilist, ImmType>;
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

multiclass VPatAMOWD<string intrinsic,
                     string inst,
                     ValueType result_type,
                     ValueType offset_type,
                     ValueType mask_type,
                     int sew,
                     LMULInfo vlmul,
                     LMULInfo emul,
                     VReg op1_reg_class>
{
  def : VPatAMOWDNoMask<intrinsic, inst, result_type, offset_type,
                        sew, vlmul, emul, op1_reg_class>;
  def : VPatAMOWDMask<intrinsic, inst, result_type, offset_type,
                      mask_type, sew, vlmul, emul, op1_reg_class>;
}

multiclass VPatAMOV_WD<string intrinsic,
                       string inst,
                       list<VTypeInfo> vtilist> {
  foreach eew = EEWList in {
    foreach vti = vtilist in {
      if !or(!eq(vti.SEW, 32), !eq(vti.SEW, 64)) then {
        defvar octuple_lmul = octuple_from_str<vti.LMul.MX>.ret;
        // Calculate emul = eew * lmul / sew
        defvar octuple_emul = !srl(!mul(eew, octuple_lmul), shift_amount<vti.SEW>.val);
        if !and(!ge(octuple_emul, 1), !le(octuple_emul, 64)) then {
          defvar emulMX = octuple_to_str<octuple_emul>.ret;
          defvar offsetVti = !cast<VTypeInfo>("VI" # eew # emulMX);
          defvar inst_ei = inst # "EI" # eew;
          defm : VPatAMOWD<intrinsic, inst_ei,
                           vti.Vector, offsetVti.Vector,
                           vti.Mask, vti.SEW, vti.LMul, offsetVti.LMul, offsetVti.RegClass>;
        }
      }
    }
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

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in {
  def PseudoReadVLENB : Pseudo<(outs GPR:$rd), (ins),
                               [(set GPR:$rd, (riscv_read_vlenb))]>;
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

foreach lmul = MxList.m in {
  foreach nf = NFSet<lmul>.L in {
    defvar vreg = SegRegClass<lmul, nf>.RC;
    let hasSideEffects = 0, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1 in {
      def "PseudoVSPILL" # nf # "_" # lmul.MX :
        Pseudo<(outs), (ins vreg:$rs1, GPR:$rs2, GPR:$vlenb), []>;
    }
    let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1 in {
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
def PseudoVSETVLI : Pseudo<(outs GPR:$rd), (ins GPR:$rs1, VTypeIOp:$vtypei), []>;
def PseudoVSETIVLI : Pseudo<(outs GPR:$rd), (ins uimm5:$rs1, VTypeIOp:$vtypei), []>;
}

//===----------------------------------------------------------------------===//
// 7. Vector Loads and Stores
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 7.4 Vector Unit-Stride Instructions
//===----------------------------------------------------------------------===//

// Pseudos Unit-Stride Loads and Stores
defm PseudoVL : VPseudoUSLoad</*isFF=*/false>;
defm PseudoVS : VPseudoUSStore;

defm PseudoVLE1 : VPseudoLoadMask;
defm PseudoVSE1 : VPseudoStoreMask;

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
defm PseudoVL : VPseudoUSLoad</*isFF=*/true>;

//===----------------------------------------------------------------------===//
// 7.8. Vector Load/Store Segment Instructions
//===----------------------------------------------------------------------===//
defm PseudoVLSEG : VPseudoUSSegLoad</*isFF=*/false>;
defm PseudoVLSSEG : VPseudoSSegLoad;
defm PseudoVLOXSEG : VPseudoISegLoad</*Ordered=*/true>;
defm PseudoVLUXSEG : VPseudoISegLoad</*Ordered=*/false>;
defm PseudoVSSEG : VPseudoUSSegStore;
defm PseudoVSSSEG : VPseudoSSegStore;
defm PseudoVSOXSEG : VPseudoISegStore</*Ordered=*/true>;
defm PseudoVSUXSEG : VPseudoISegStore</*Ordered=*/false>;

// vlseg<nf>e<eew>ff.v may update VL register
let hasSideEffects = 1, Defs = [VL] in
defm PseudoVLSEG : VPseudoUSSegLoad</*isFF=*/true>;

//===----------------------------------------------------------------------===//
// 8. Vector AMO Operations
//===----------------------------------------------------------------------===//
defm PseudoVAMOSWAP : VPseudoAMO;
defm PseudoVAMOADD : VPseudoAMO;
defm PseudoVAMOXOR : VPseudoAMO;
defm PseudoVAMOAND : VPseudoAMO;
defm PseudoVAMOOR : VPseudoAMO;
defm PseudoVAMOMIN : VPseudoAMO;
defm PseudoVAMOMAX : VPseudoAMO;
defm PseudoVAMOMINU : VPseudoAMO;
defm PseudoVAMOMAXU : VPseudoAMO;

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
defm PseudoVWMACCUS    : VPseudoTernaryW_VX;

//===----------------------------------------------------------------------===//
// 12.15. Vector Integer Merge Instructions
//===----------------------------------------------------------------------===//
defm PseudoVMERGE      : VPseudoBinaryV_VM_XM_IM;

//===----------------------------------------------------------------------===//
// 12.16. Vector Integer Move Instructions
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
defm PseudoVFADD       : VPseudoBinaryV_VV_VF;
defm PseudoVFSUB       : VPseudoBinaryV_VV_VF;
defm PseudoVFRSUB      : VPseudoBinaryV_VF;

//===----------------------------------------------------------------------===//
// 14.3. Vector Widening Floating-Point Add/Subtract Instructions
//===----------------------------------------------------------------------===//
defm PseudoVFWADD     : VPseudoBinaryW_VV_VF;
defm PseudoVFWSUB     : VPseudoBinaryW_VV_VF;
defm PseudoVFWADD     : VPseudoBinaryW_WV_WF;
defm PseudoVFWSUB     : VPseudoBinaryW_WV_WF;

//===----------------------------------------------------------------------===//
// 14.4. Vector Single-Width Floating-Point Multiply/Divide Instructions
//===----------------------------------------------------------------------===//
defm PseudoVFMUL       : VPseudoBinaryV_VV_VF;
defm PseudoVFDIV       : VPseudoBinaryV_VV_VF;
defm PseudoVFRDIV      : VPseudoBinaryV_VF;

//===----------------------------------------------------------------------===//
// 14.5. Vector Widening Floating-Point Multiply
//===----------------------------------------------------------------------===//
defm PseudoVFWMUL      : VPseudoBinaryW_VV_VF;

//===----------------------------------------------------------------------===//
// 14.6. Vector Single-Width Floating-Point Fused Multiply-Add Instructions
//===----------------------------------------------------------------------===//
defm PseudoVFMACC      : VPseudoTernaryV_VV_VF_AAXA;
defm PseudoVFNMACC     : VPseudoTernaryV_VV_VF_AAXA;
defm PseudoVFMSAC      : VPseudoTernaryV_VV_VF_AAXA;
defm PseudoVFNMSAC     : VPseudoTernaryV_VV_VF_AAXA;
defm PseudoVFMADD      : VPseudoTernaryV_VV_VF_AAXA;
defm PseudoVFNMADD     : VPseudoTernaryV_VV_VF_AAXA;
defm PseudoVFMSUB      : VPseudoTernaryV_VV_VF_AAXA;
defm PseudoVFNMSUB     : VPseudoTernaryV_VV_VF_AAXA;

//===----------------------------------------------------------------------===//
// 14.7. Vector Widening Floating-Point Fused Multiply-Add Instructions
//===----------------------------------------------------------------------===//
defm PseudoVFWMACC     : VPseudoTernaryW_VV_VF;
defm PseudoVFWNMACC    : VPseudoTernaryW_VV_VF;
defm PseudoVFWMSAC     : VPseudoTernaryW_VV_VF;
defm PseudoVFWNMSAC    : VPseudoTernaryW_VV_VF;

//===----------------------------------------------------------------------===//
// 14.8. Vector Floating-Point Square-Root Instruction
//===----------------------------------------------------------------------===//
defm PseudoVFSQRT      : VPseudoUnaryV_V;

//===----------------------------------------------------------------------===//
// 14.9. Vector Floating-Point Reciprocal Square-Root Estimate Instruction
//===----------------------------------------------------------------------===//
defm PseudoVFRSQRT7    : VPseudoUnaryV_V;

//===----------------------------------------------------------------------===//
// 14.10. Vector Floating-Point Reciprocal Estimate Instruction
//===----------------------------------------------------------------------===//
defm PseudoVFREC7      : VPseudoUnaryV_V;

//===----------------------------------------------------------------------===//
// 14.11. Vector Floating-Point Min/Max Instructions
//===----------------------------------------------------------------------===//
defm PseudoVFMIN       : VPseudoBinaryV_VV_VF;
defm PseudoVFMAX       : VPseudoBinaryV_VV_VF;

//===----------------------------------------------------------------------===//
// 14.12. Vector Floating-Point Sign-Injection Instructions
//===----------------------------------------------------------------------===//
defm PseudoVFSGNJ      : VPseudoBinaryV_VV_VF;
defm PseudoVFSGNJN     : VPseudoBinaryV_VV_VF;
defm PseudoVFSGNJX     : VPseudoBinaryV_VV_VF;

//===----------------------------------------------------------------------===//
// 14.13. Vector Floating-Point Compare Instructions
//===----------------------------------------------------------------------===//
defm PseudoVMFEQ       : VPseudoBinaryM_VV_VF;
defm PseudoVMFNE       : VPseudoBinaryM_VV_VF;
defm PseudoVMFLT       : VPseudoBinaryM_VV_VF;
defm PseudoVMFLE       : VPseudoBinaryM_VV_VF;
defm PseudoVMFGT       : VPseudoBinaryM_VF;
defm PseudoVMFGE       : VPseudoBinaryM_VF;

//===----------------------------------------------------------------------===//
// 14.14. Vector Floating-Point Classify Instruction
//===----------------------------------------------------------------------===//
defm PseudoVFCLASS     : VPseudoUnaryV_V;

//===----------------------------------------------------------------------===//
// 14.15. Vector Floating-Point Merge Instruction
//===----------------------------------------------------------------------===//
defm PseudoVFMERGE     : VPseudoBinaryV_FM;

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

// Pseudo instructions
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
      let HasVLOp = 1, HasSEWOp = 1, BaseInstr = VMV_S_X,
          ForceTailAgnostic = true, Constraints = "$rd = $rs1" in
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
    foreach f = FPList.fpinfo in {
      let VLMul = m.value in {
        let HasSEWOp = 1, BaseInstr = VFMV_F_S in
        def "PseudoVFMV_" # f.FX # "_S_" # m.MX :
                                          Pseudo<(outs f.fprclass:$rd),
                                                 (ins m.vrclass:$rs2,
                                                      ixlenimm:$sew),
                                                 []>, RISCVVPseudo;
        let HasVLOp = 1, HasSEWOp = 1, BaseInstr = VFMV_S_F,
            ForceTailAgnostic = true, Constraints = "$rd = $rs1" in
        def "PseudoVFMV_S_" # f.FX # "_" # m.MX :
                                          Pseudo<(outs m.vrclass:$rd),
                                                 (ins m.vrclass:$rs1, f.fprclass:$rs2,
                                                      GPR:$vl, ixlenimm:$sew),
                                                 []>, RISCVVPseudo;
      }
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
  defm PseudoVSLIDE1UP   : VPseudoBinaryV_VX<"@earlyclobber $rd">;
  defm PseudoVSLIDE1DOWN : VPseudoBinaryV_VX;
} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV, HasStdExtF] in {
  defm PseudoVFSLIDE1UP  : VPseudoBinaryV_VF<"@earlyclobber $rd">;
  defm PseudoVFSLIDE1DOWN : VPseudoBinaryV_VF;
} // Predicates = [HasStdExtV, HasStdExtF]

//===----------------------------------------------------------------------===//
// 17.4. Vector Register Gather Instructions
//===----------------------------------------------------------------------===//
defm PseudoVRGATHER    : VPseudoBinaryV_VV_VX_VI<uimm5, "@earlyclobber $rd">;
defm PseudoVRGATHEREI16 : VPseudoBinaryV_VV_EEW</* eew */ 16, "@earlyclobber $rd">;

//===----------------------------------------------------------------------===//
// 17.5. Vector Compress Instruction
//===----------------------------------------------------------------------===//
defm PseudoVCOMPRESS : VPseudoUnaryV_V_AnyMask;

//===----------------------------------------------------------------------===//
// Patterns.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 8. Vector AMO Operations
//===----------------------------------------------------------------------===//
let Predicates = [HasStdExtZvamo] in {
  defm "" : VPatAMOV_WD<"int_riscv_vamoswap", "PseudoVAMOSWAP", AllIntegerVectors>;
  defm "" : VPatAMOV_WD<"int_riscv_vamoadd", "PseudoVAMOADD", AllIntegerVectors>;
  defm "" : VPatAMOV_WD<"int_riscv_vamoxor", "PseudoVAMOXOR", AllIntegerVectors>;
  defm "" : VPatAMOV_WD<"int_riscv_vamoand", "PseudoVAMOAND", AllIntegerVectors>;
  defm "" : VPatAMOV_WD<"int_riscv_vamoor", "PseudoVAMOOR", AllIntegerVectors>;
  defm "" : VPatAMOV_WD<"int_riscv_vamomin", "PseudoVAMOMIN", AllIntegerVectors>;
  defm "" : VPatAMOV_WD<"int_riscv_vamomax", "PseudoVAMOMAX", AllIntegerVectors>;
  defm "" : VPatAMOV_WD<"int_riscv_vamominu", "PseudoVAMOMINU", AllIntegerVectors>;
  defm "" : VPatAMOV_WD<"int_riscv_vamomaxu", "PseudoVAMOMAXU", AllIntegerVectors>;
} // Predicates = [HasStdExtZvamo]

let Predicates = [HasStdExtZvamo, HasStdExtF] in {
  defm "" : VPatAMOV_WD<"int_riscv_vamoswap", "PseudoVAMOSWAP", AllFloatVectors>;
} // Predicates = [HasStdExtZvamo, HasStdExtF]

//===----------------------------------------------------------------------===//
// 12. Vector Integer Arithmetic Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtV] in {
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
                                       (vti.Scalar simm5_plus1:$rs2), (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMSLE_VI_"#vti.LMul.MX) vti.RegClass:$rs1,
                                                               (DecImm simm5_plus1:$rs2),
                                                               GPR:$vl,
                                                               vti.SEW)>;
  def : Pat<(vti.Mask (int_riscv_vmslt_mask (vti.Mask VR:$merge),
                                            (vti.Vector vti.RegClass:$rs1),
                                            (vti.Scalar simm5_plus1:$rs2),
                                            (vti.Mask V0),
                                            (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMSLE_VI_"#vti.LMul.MX#"_MASK")
                                                      VR:$merge,
                                                      vti.RegClass:$rs1,
                                                      (DecImm simm5_plus1:$rs2),
                                                      (vti.Mask V0),
                                                      GPR:$vl,
                                                      vti.SEW)>;

  def : Pat<(vti.Mask (int_riscv_vmsltu (vti.Vector vti.RegClass:$rs1),
                                        (vti.Scalar simm5_plus1:$rs2),
                                        (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMSLEU_VI_"#vti.LMul.MX) vti.RegClass:$rs1,
                                                                (DecImm simm5_plus1:$rs2),
                                                                GPR:$vl,
                                                                vti.SEW)>;
  def : Pat<(vti.Mask (int_riscv_vmsltu_mask (vti.Mask VR:$merge),
                                             (vti.Vector vti.RegClass:$rs1),
                                             (vti.Scalar simm5_plus1:$rs2),
                                             (vti.Mask V0),
                                             (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMSLEU_VI_"#vti.LMul.MX#"_MASK")
                                                      VR:$merge,
                                                      vti.RegClass:$rs1,
                                                      (DecImm simm5_plus1:$rs2),
                                                      (vti.Mask V0),
                                                      GPR:$vl,
                                                      vti.SEW)>;

  // Special cases to avoid matching vmsltu.vi 0 (always false) to
  // vmsleu.vi -1 (always true). Instead match to vmsne.vv.
  def : Pat<(vti.Mask (int_riscv_vmsltu (vti.Vector vti.RegClass:$rs1),
                                        (vti.Scalar 0), (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMSNE_VV_"#vti.LMul.MX) vti.RegClass:$rs1,
                                                               vti.RegClass:$rs1,
                                                               GPR:$vl,
                                                               vti.SEW)>;
  def : Pat<(vti.Mask (int_riscv_vmsltu_mask (vti.Mask VR:$merge),
                                             (vti.Vector vti.RegClass:$rs1),
                                             (vti.Scalar 0),
                                             (vti.Mask V0),
                                             (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMSNE_VV_"#vti.LMul.MX#"_MASK")
                                                     VR:$merge,
                                                     vti.RegClass:$rs1,
                                                     vti.RegClass:$rs1,
                                                     (vti.Mask V0),
                                                     GPR:$vl,
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
// 12.15. Vector Integer Merge Instructions
//===----------------------------------------------------------------------===//
defm "" : VPatBinaryV_VM_XM_IM<"int_riscv_vmerge", "PseudoVMERGE">;

//===----------------------------------------------------------------------===//
// 12.16. Vector Integer Move Instructions
//===----------------------------------------------------------------------===//
foreach vti = AllVectors in {
  def : Pat<(vti.Vector (int_riscv_vmv_v_v (vti.Vector vti.RegClass:$rs1),
                                           (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMV_V_V_"#vti.LMul.MX)
             $rs1, GPR:$vl, vti.SEW)>;

  // vmv.v.x/vmv.v.i are handled in RISCInstrVInstrInfoVVLPatterns.td
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
// 14.9. Vector Floating-Point Reciprocal Square-Root Estimate Instruction
//===----------------------------------------------------------------------===//
defm "" : VPatUnaryV_V<"int_riscv_vfrsqrt7", "PseudoVFRSQRT7", AllFloatVectors>;

//===----------------------------------------------------------------------===//
// 14.10. Vector Floating-Point Reciprocal Estimate Instruction
//===----------------------------------------------------------------------===//
defm "" : VPatUnaryV_V<"int_riscv_vfrec7", "PseudoVFREC7", AllFloatVectors>;

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

foreach fvti = AllFloatVectors in {
  defvar instr = !cast<Instruction>("PseudoVMERGE_VIM_"#fvti.LMul.MX);
  def : Pat<(fvti.Vector (int_riscv_vfmerge (fvti.Vector fvti.RegClass:$rs2),
                                            (fvti.Scalar (fpimm0)),
                                            (fvti.Mask V0), (XLenVT (VLOp GPR:$vl)))),
            (instr fvti.RegClass:$rs2, 0, (fvti.Mask V0), GPR:$vl, fvti.SEW)>;
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
  // vmv.s.x is handled with a custom node in RISCVInstrInfoVVLPatterns.td
}
} // Predicates = [HasStdExtV]

//===----------------------------------------------------------------------===//
// 17.2. Floating-Point Scalar Move Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtV, HasStdExtF] in {
foreach fvti = AllFloatVectors in {
  defvar instr = !cast<Instruction>("PseudoVFMV_"#fvti.ScalarSuffix#"_S_" #
                                    fvti.LMul.MX);
  def : Pat<(fvti.Scalar (int_riscv_vfmv_f_s (fvti.Vector fvti.RegClass:$rs2))),
                         (instr $rs2, fvti.SEW)>;

  def : Pat<(fvti.Vector (int_riscv_vfmv_s_f (fvti.Vector fvti.RegClass:$rs1),
                         (fvti.Scalar fvti.ScalarRegClass:$rs2), (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVFMV_S_"#fvti.ScalarSuffix#"_" #
                                fvti.LMul.MX)
             (fvti.Vector $rs1),
             (fvti.Scalar fvti.ScalarRegClass:$rs2),
             GPR:$vl, fvti.SEW)>;
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
  defm "" : VPatBinaryV_VV_INT_EEW<"int_riscv_vrgatherei16_vv", "PseudoVRGATHEREI16",
                                   /* eew */ 16, AllIntegerVectors>;
} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV, HasStdExtF] in {
  defm "" : VPatBinaryV_VV_VX_VI_INT<"int_riscv_vrgather", "PseudoVRGATHER",
                                     AllFloatVectors, uimm5>;
  defm "" : VPatBinaryV_VV_INT_EEW<"int_riscv_vrgatherei16_vv", "PseudoVRGATHEREI16",
                                   /* eew */ 16, AllFloatVectors>;
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
include "RISCVInstrInfoVVLPatterns.td"
