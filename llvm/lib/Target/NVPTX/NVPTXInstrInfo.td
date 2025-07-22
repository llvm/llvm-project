//===- NVPTXInstrInfo.td - NVPTX Instruction defs -------------*- tblgen-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the PTX instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

include "NVPTXInstrFormats.td"

let OperandType = "OPERAND_IMMEDIATE" in {
  def f16imm : Operand<f16>;
  def bf16imm : Operand<bf16>;

}

// List of vector specific properties
def isVecLD      : VecInstTypeEnum<1>;
def isVecST      : VecInstTypeEnum<2>;
def isVecBuild   : VecInstTypeEnum<3>;
def isVecShuffle : VecInstTypeEnum<4>;
def isVecExtract : VecInstTypeEnum<5>;
def isVecInsert  : VecInstTypeEnum<6>;
def isVecDest    : VecInstTypeEnum<7>;
def isVecOther   : VecInstTypeEnum<15>;

//===----------------------------------------------------------------------===//
// NVPTX Operand Definitions.
//===----------------------------------------------------------------------===//

def brtarget    : Operand<OtherVT>;

// CVT conversion modes
// These must match the enum in NVPTX.h
def CvtNONE : PatLeaf<(i32 0x0)>;
def CvtRNI  : PatLeaf<(i32 0x1)>;
def CvtRZI  : PatLeaf<(i32 0x2)>;
def CvtRMI  : PatLeaf<(i32 0x3)>;
def CvtRPI  : PatLeaf<(i32 0x4)>;
def CvtRN   : PatLeaf<(i32 0x5)>;
def CvtRZ   : PatLeaf<(i32 0x6)>;
def CvtRM   : PatLeaf<(i32 0x7)>;
def CvtRP   : PatLeaf<(i32 0x8)>;
def CvtRNA   : PatLeaf<(i32 0x9)>;

def CvtNONE_FTZ : PatLeaf<(i32 0x10)>;
def CvtRNI_FTZ  : PatLeaf<(i32 0x11)>;
def CvtRZI_FTZ  : PatLeaf<(i32 0x12)>;
def CvtRMI_FTZ  : PatLeaf<(i32 0x13)>;
def CvtRPI_FTZ  : PatLeaf<(i32 0x14)>;
def CvtRN_FTZ   : PatLeaf<(i32 0x15)>;
def CvtRZ_FTZ   : PatLeaf<(i32 0x16)>;
def CvtRM_FTZ   : PatLeaf<(i32 0x17)>;
def CvtRP_FTZ   : PatLeaf<(i32 0x18)>;

def CvtSAT      : PatLeaf<(i32 0x20)>;
def CvtSAT_FTZ  : PatLeaf<(i32 0x30)>;

def CvtNONE_RELU   : PatLeaf<(i32 0x40)>;
def CvtRN_RELU   : PatLeaf<(i32 0x45)>;
def CvtRZ_RELU   : PatLeaf<(i32 0x46)>;

def CvtMode : Operand<i32> {
  let PrintMethod = "printCvtMode";
}

// Compare modes
// These must match the enum in NVPTX.h
def CmpEQ   : PatLeaf<(i32 0)>;
def CmpNE   : PatLeaf<(i32 1)>;
def CmpLT   : PatLeaf<(i32 2)>;
def CmpLE   : PatLeaf<(i32 3)>;
def CmpGT   : PatLeaf<(i32 4)>;
def CmpGE   : PatLeaf<(i32 5)>;
def CmpLO   : PatLeaf<(i32 6)>;
def CmpLS   : PatLeaf<(i32 7)>;
def CmpHI   : PatLeaf<(i32 8)>;
def CmpHS   : PatLeaf<(i32 9)>;
def CmpEQU  : PatLeaf<(i32 10)>;
def CmpNEU  : PatLeaf<(i32 11)>;
def CmpLTU  : PatLeaf<(i32 12)>;
def CmpLEU  : PatLeaf<(i32 13)>;
def CmpGTU  : PatLeaf<(i32 14)>;
def CmpGEU  : PatLeaf<(i32 15)>;
def CmpNUM  : PatLeaf<(i32 16)>;
def CmpNAN  : PatLeaf<(i32 17)>;

def CmpEQ_FTZ   : PatLeaf<(i32 0x100)>;
def CmpNE_FTZ   : PatLeaf<(i32 0x101)>;
def CmpLT_FTZ   : PatLeaf<(i32 0x102)>;
def CmpLE_FTZ   : PatLeaf<(i32 0x103)>;
def CmpGT_FTZ   : PatLeaf<(i32 0x104)>;
def CmpGE_FTZ   : PatLeaf<(i32 0x105)>;
def CmpEQU_FTZ  : PatLeaf<(i32 0x10A)>;
def CmpNEU_FTZ  : PatLeaf<(i32 0x10B)>;
def CmpLTU_FTZ  : PatLeaf<(i32 0x10C)>;
def CmpLEU_FTZ  : PatLeaf<(i32 0x10D)>;
def CmpGTU_FTZ  : PatLeaf<(i32 0x10E)>;
def CmpGEU_FTZ  : PatLeaf<(i32 0x10F)>;
def CmpNUM_FTZ  : PatLeaf<(i32 0x110)>;
def CmpNAN_FTZ  : PatLeaf<(i32 0x111)>;

def CmpMode : Operand<i32> {
  let PrintMethod = "printCmpMode";
}
def VecElement : Operand<i32> {
  let PrintMethod = "printVecElement";
}

// PRMT modes
// These must match the enum in NVPTX.h
def PrmtNONE : PatLeaf<(i32 0x0)>;
def PrmtF4E  : PatLeaf<(i32 0x1)>;
def PrmtB4E  : PatLeaf<(i32 0x2)>;
def PrmtRC8  : PatLeaf<(i32 0x3)>;
def PrmtECL  : PatLeaf<(i32 0x4)>;
def PrmtECR  : PatLeaf<(i32 0x5)>;
def PrmtRC16 : PatLeaf<(i32 0x6)>;

def PrmtMode : Operand<i32> {
  let PrintMethod = "printPrmtMode";
}


//===----------------------------------------------------------------------===//
// NVPTX Instruction Predicate Definitions
//===----------------------------------------------------------------------===//


def hasAtomAddF64 : Predicate<"Subtarget->hasAtomAddF64()">;
def hasAtomScope : Predicate<"Subtarget->hasAtomScope()">;
def hasAtomBitwise64 : Predicate<"Subtarget->hasAtomBitwise64()">;
def hasAtomMinMax64 : Predicate<"Subtarget->hasAtomMinMax64()">;
def hasVote : Predicate<"Subtarget->hasVote()">;
def hasDouble : Predicate<"Subtarget->hasDouble()">;
def hasClusters : Predicate<"Subtarget->hasClusters()">;
def hasLDG : Predicate<"Subtarget->hasLDG()">;
def hasLDU : Predicate<"Subtarget->hasLDU()">;
def hasPTXASUnreachableBug : Predicate<"Subtarget->hasPTXASUnreachableBug()">;
def noPTXASUnreachableBug : Predicate<"!Subtarget->hasPTXASUnreachableBug()">;
def hasOptEnabled : Predicate<"TM.getOptLevel() != CodeGenOptLevel::None">;
def hasArchAccelFeatures : Predicate<"Subtarget->hasArchAccelFeatures()">;

def doF32FTZ : Predicate<"useF32FTZ()">;
def doNoF32FTZ : Predicate<"!useF32FTZ()">;
def doRsqrtOpt : Predicate<"doRsqrtOpt()">;

def doMulWide      : Predicate<"doMulWide">;

def do_SQRTF32_APPROX : Predicate<"!usePrecSqrtF32()">;
def do_SQRTF32_RN : Predicate<"usePrecSqrtF32()">;

def hasHWROT32 : Predicate<"Subtarget->hasHWROT32()">;
def noHWROT32 : Predicate<"!Subtarget->hasHWROT32()">;
def hasDotInstructions : Predicate<"Subtarget->hasDotInstructions()">;
def hasTcgen05Instructions : Predicate<"Subtarget->hasTcgen05Instructions()">;

def True : Predicate<"true">;
def False : Predicate<"false">;

class hasPTX<int version>: Predicate<"Subtarget->getPTXVersion() >= " # version>;
class hasSM<int version>: Predicate<"Subtarget->getSmVersion() >= " # version>;

// Explicit records for arch-accelerated SM versions
def hasSM90a : Predicate<"Subtarget->getFullSmVersion() == 901">;
def hasSM100a : Predicate<"Subtarget->getFullSmVersion() == 1001">;
def hasSM101a : Predicate<"Subtarget->getFullSmVersion() == 1011">;
def hasSM120a : Predicate<"Subtarget->getFullSmVersion() == 1201">;

// non-sync shfl instructions are not available on sm_70+ in PTX6.4+
def hasSHFL : Predicate<"!(Subtarget->getSmVersion() >= 70"
                          "&& Subtarget->getPTXVersion() >= 64)">;

def useFP16Math: Predicate<"Subtarget->allowFP16Math()">;
def hasBF16Math: Predicate<"Subtarget->hasBF16Math()">;

// Helper class to aid conversion between ValueType and a matching RegisterClass.

class ValueToRegClass<ValueType T> {
   string name = !cast<string>(T);
   NVPTXRegClass ret = !cond(
     !eq(name, "i1"): Int1Regs,
     !eq(name, "i16"): Int16Regs,
     !eq(name, "v2i16"): Int32Regs,
     !eq(name, "i32"): Int32Regs,
     !eq(name, "i64"): Int64Regs,
     !eq(name, "f16"): Int16Regs,
     !eq(name, "v2f16"): Int32Regs,
     !eq(name, "bf16"): Int16Regs,
     !eq(name, "v2bf16"): Int32Regs,
     !eq(name, "f32"): Float32Regs,
     !eq(name, "f64"): Float64Regs,
     !eq(name, "ai32"): Int32ArgRegs,
     !eq(name, "ai64"): Int64ArgRegs,
     !eq(name, "af32"): Float32ArgRegs,
     !eq(name, "if64"): Float64ArgRegs,
    );
}


//===----------------------------------------------------------------------===//
// Some Common Instruction Class Templates
//===----------------------------------------------------------------------===//

class OneUse1<SDPatternOperator operator>
    : PatFrag<(ops node:$A), (operator node:$A), [{ return N->hasOneUse(); }]>;

class fpimm_pos_inf<ValueType vt>
    : FPImmLeaf<vt, [{ return Imm.isPosInfinity(); }]>;

// Utility class to wrap up information about a register and DAG type for more
// convenient iteration and parameterization
class RegTyInfo<ValueType ty, NVPTXRegClass rc, Operand imm, SDNode imm_node,
                bit supports_imm = 1> {
  ValueType Ty = ty;
  NVPTXRegClass RC = rc;
  Operand Imm = imm;
  SDNode ImmNode = imm_node;
  bit SupportsImm = supports_imm;
  int Size = ty.Size;
}

def I1RT     : RegTyInfo<i1,  Int1Regs,  i1imm,  imm>;
def I16RT    : RegTyInfo<i16, Int16Regs, i16imm, imm>;
def I32RT    : RegTyInfo<i32, Int32Regs, i32imm, imm>;
def I64RT    : RegTyInfo<i64, Int64Regs, i64imm, imm>;

def F32RT    : RegTyInfo<f32, Float32Regs, f32imm, fpimm>;
def F64RT    : RegTyInfo<f64, Float64Regs, f64imm, fpimm>;
def F16RT    : RegTyInfo<f16, Int16Regs, f16imm, fpimm, supports_imm = 0>;
def BF16RT   : RegTyInfo<bf16, Int16Regs, bf16imm, fpimm, supports_imm = 0>;

def F16X2RT  : RegTyInfo<v2f16, Int32Regs, ?, ?, supports_imm = 0>;
def BF16X2RT : RegTyInfo<v2bf16, Int32Regs, ?, ?, supports_imm = 0>;


multiclass I3Inst<string op_str, SDPatternOperator op_node, RegTyInfo t,
                  bit commutative, list<Predicate> requires = []> {
  defvar asmstr = op_str # " \t$dst, $a, $b;";

  def rr :
    NVPTXInst<(outs t.RC:$dst), (ins t.RC:$a, t.RC:$b),
              asmstr,
              [(set t.Ty:$dst, (op_node t.Ty:$a, t.Ty:$b))]>,
              Requires<requires>;
  def ri :
    NVPTXInst<(outs t.RC:$dst), (ins t.RC:$a, t.Imm:$b),
              asmstr,
              [(set t.Ty:$dst, (op_node t.Ty:$a, (t.Ty imm:$b)))]>,
              Requires<requires>;
  if !not(commutative) then
    def ir :
      NVPTXInst<(outs t.RC:$dst), (ins t.Imm:$a, t.RC:$b),
                asmstr,
                [(set t.Ty:$dst, (op_node (t.Ty imm:$a), t.Ty:$b))]>,
                Requires<requires>;
}

// Template for instructions which take three int64, int32, or int16 args.
// The instructions are named "<OpcStr><Width>" (e.g. "add.s64").
multiclass I3<string op_str, SDPatternOperator op_node, bit commutative> {
  foreach t = [I16RT, I32RT, I64RT] in
    defm t.Ty# : I3Inst<op_str # t.Size, op_node, t, commutative>;
}

class I16x2<string OpcStr, SDNode OpNode> :
 NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a, Int32Regs:$b),
              !strconcat(OpcStr, "16x2 \t$dst, $a, $b;"),
              [(set v2i16:$dst, (OpNode v2i16:$a, v2i16:$b))]>,
              Requires<[hasPTX<80>, hasSM<90>]>;

// Template for instructions which take 3 int args.  The instructions are
// named "<OpcStr>.s32" (e.g. "addc.cc.s32").
multiclass ADD_SUB_INT_CARRY<string op_str, SDNode op_node, bit commutative> {
  let hasSideEffects = 1 in {
    defm i32 : I3Inst<op_str # ".s32", op_node, I32RT, commutative>;
    defm i64 : I3Inst<op_str # ".s64", op_node, I64RT, commutative,
                     requires = [hasPTX<43>]>;
  }
}

// Template for minimum/maximum instructions.
//
// Also defines ftz (flush subnormal inputs and results to sign-preserving
// zero) variants for fp32 functions.
multiclass FMINIMUMMAXIMUM<string OpcStr, bit NaN, SDNode OpNode> {
  if !not(NaN) then {
   def f64rr :
     NVPTXInst<(outs Float64Regs:$dst),
               (ins Float64Regs:$a, Float64Regs:$b),
               !strconcat(OpcStr, ".f64 \t$dst, $a, $b;"),
               [(set f64:$dst, (OpNode f64:$a, f64:$b))]>;
   def f64ri :
     NVPTXInst<(outs Float64Regs:$dst),
               (ins Float64Regs:$a, f64imm:$b),
               !strconcat(OpcStr, ".f64 \t$dst, $a, $b;"),
               [(set f64:$dst, (OpNode f64:$a, fpimm:$b))]>;
  }
   def f32rr_ftz :
     NVPTXInst<(outs Float32Regs:$dst),
               (ins Float32Regs:$a, Float32Regs:$b),
               !strconcat(OpcStr, ".ftz.f32 \t$dst, $a, $b;"),
               [(set f32:$dst, (OpNode f32:$a, f32:$b))]>,
               Requires<[doF32FTZ]>;
   def f32ri_ftz :
     NVPTXInst<(outs Float32Regs:$dst),
               (ins Float32Regs:$a, f32imm:$b),
               !strconcat(OpcStr, ".ftz.f32 \t$dst, $a, $b;"),
               [(set f32:$dst, (OpNode f32:$a, fpimm:$b))]>,
               Requires<[doF32FTZ]>;
   def f32rr :
     NVPTXInst<(outs Float32Regs:$dst),
               (ins Float32Regs:$a, Float32Regs:$b),
               !strconcat(OpcStr, ".f32 \t$dst, $a, $b;"),
               [(set f32:$dst, (OpNode f32:$a, f32:$b))]>;
   def f32ri :
     NVPTXInst<(outs Float32Regs:$dst),
               (ins Float32Regs:$a, f32imm:$b),
               !strconcat(OpcStr, ".f32 \t$dst, $a, $b;"),
               [(set f32:$dst, (OpNode f32:$a, fpimm:$b))]>;

   def f16rr_ftz :
     NVPTXInst<(outs Int16Regs:$dst),
               (ins Int16Regs:$a, Int16Regs:$b),
               !strconcat(OpcStr, ".ftz.f16 \t$dst, $a, $b;"),
               [(set f16:$dst, (OpNode f16:$a, f16:$b))]>,
               Requires<[useFP16Math, doF32FTZ]>;
   def f16rr :
     NVPTXInst<(outs Int16Regs:$dst),
               (ins Int16Regs:$a, Int16Regs:$b),
               !strconcat(OpcStr, ".f16 \t$dst, $a, $b;"),
               [(set f16:$dst, (OpNode f16:$a, f16:$b))]>,
               Requires<[useFP16Math, hasSM<80>, hasPTX<70>]>;

   def f16x2rr_ftz :
     NVPTXInst<(outs Int32Regs:$dst),
               (ins Int32Regs:$a, Int32Regs:$b),
               !strconcat(OpcStr, ".ftz.f16x2 \t$dst, $a, $b;"),
               [(set v2f16:$dst, (OpNode v2f16:$a, v2f16:$b))]>,
               Requires<[useFP16Math, hasSM<80>, hasPTX<70>, doF32FTZ]>;
   def f16x2rr :
     NVPTXInst<(outs Int32Regs:$dst),
               (ins Int32Regs:$a, Int32Regs:$b),
               !strconcat(OpcStr, ".f16x2 \t$dst, $a, $b;"),
               [(set v2f16:$dst, (OpNode v2f16:$a, v2f16:$b))]>,
               Requires<[useFP16Math, hasSM<80>, hasPTX<70>]>;
   def bf16rr :
     NVPTXInst<(outs Int16Regs:$dst),
               (ins Int16Regs:$a, Int16Regs:$b),
               !strconcat(OpcStr, ".bf16 \t$dst, $a, $b;"),
               [(set bf16:$dst, (OpNode bf16:$a, bf16:$b))]>,
               Requires<[hasBF16Math, hasSM<80>, hasPTX<70>]>;
   def bf16x2rr :
     NVPTXInst<(outs Int32Regs:$dst),
               (ins Int32Regs:$a, Int32Regs:$b),
               !strconcat(OpcStr, ".bf16x2 \t$dst, $a, $b;"),
               [(set v2bf16:$dst, (OpNode v2bf16:$a, v2bf16:$b))]>,
               Requires<[hasBF16Math, hasSM<80>, hasPTX<70>]>;
}

// Template for instructions which take three FP args.  The
// instructions are named "<OpcStr>.f<Width>" (e.g. "add.f64").
//
// Also defines ftz (flush subnormal inputs and results to sign-preserving
// zero) variants for fp32/fp16 functions.
//
// This multiclass should be used for nodes that can be folded to make fma ops.
// In this case, we use the ".rn" variant when FMA is disabled, as this behaves
// just like the non ".rn" op, but prevents ptxas from creating FMAs.
multiclass F3<string op_str, SDPatternOperator op_pat> {
  def f64rr :
    NVPTXInst<(outs Float64Regs:$dst),
              (ins Float64Regs:$a, Float64Regs:$b),
              op_str # ".f64 \t$dst, $a, $b;",
              [(set f64:$dst, (op_pat f64:$a, f64:$b))]>;
  def f64ri :
    NVPTXInst<(outs Float64Regs:$dst),
              (ins Float64Regs:$a, f64imm:$b),
              op_str # ".f64 \t$dst, $a, $b;",
              [(set f64:$dst, (op_pat f64:$a, fpimm:$b))]>;
  def f32rr_ftz :
    NVPTXInst<(outs Float32Regs:$dst),
              (ins Float32Regs:$a, Float32Regs:$b),
              op_str # ".ftz.f32 \t$dst, $a, $b;",
              [(set f32:$dst, (op_pat f32:$a, f32:$b))]>,
              Requires<[doF32FTZ]>;
  def f32ri_ftz :
    NVPTXInst<(outs Float32Regs:$dst),
              (ins Float32Regs:$a, f32imm:$b),
              op_str # ".ftz.f32 \t$dst, $a, $b;",
              [(set f32:$dst, (op_pat f32:$a, fpimm:$b))]>,
              Requires<[doF32FTZ]>;
  def f32rr :
    NVPTXInst<(outs Float32Regs:$dst),
              (ins Float32Regs:$a, Float32Regs:$b),
              op_str # ".f32 \t$dst, $a, $b;",
              [(set f32:$dst, (op_pat f32:$a, f32:$b))]>;
  def f32ri :
    NVPTXInst<(outs Float32Regs:$dst),
              (ins Float32Regs:$a, f32imm:$b),
              op_str # ".f32 \t$dst, $a, $b;",
              [(set f32:$dst, (op_pat f32:$a, fpimm:$b))]>;

  def f16rr_ftz :
    NVPTXInst<(outs Int16Regs:$dst),
              (ins Int16Regs:$a, Int16Regs:$b),
              op_str # ".ftz.f16 \t$dst, $a, $b;",
              [(set f16:$dst, (op_pat f16:$a, f16:$b))]>,
              Requires<[useFP16Math, doF32FTZ]>;
  def f16rr :
    NVPTXInst<(outs Int16Regs:$dst),
              (ins Int16Regs:$a, Int16Regs:$b),
              op_str # ".f16 \t$dst, $a, $b;",
              [(set f16:$dst, (op_pat f16:$a, f16:$b))]>,
              Requires<[useFP16Math]>;

  def f16x2rr_ftz :
    NVPTXInst<(outs Int32Regs:$dst),
              (ins Int32Regs:$a, Int32Regs:$b),
              op_str # ".ftz.f16x2 \t$dst, $a, $b;",
              [(set v2f16:$dst, (op_pat v2f16:$a, v2f16:$b))]>,
              Requires<[useFP16Math, doF32FTZ]>;
  def f16x2rr :
    NVPTXInst<(outs Int32Regs:$dst),
              (ins Int32Regs:$a, Int32Regs:$b),
              op_str # ".f16x2 \t$dst, $a, $b;",
              [(set v2f16:$dst, (op_pat v2f16:$a, v2f16:$b))]>,
              Requires<[useFP16Math]>;
  def bf16rr :
    NVPTXInst<(outs Int16Regs:$dst),
              (ins Int16Regs:$a, Int16Regs:$b),
              op_str # ".bf16 \t$dst, $a, $b;",
              [(set bf16:$dst, (op_pat bf16:$a, bf16:$b))]>,
              Requires<[hasBF16Math]>;

  def bf16x2rr :
    NVPTXInst<(outs Int32Regs:$dst),
              (ins Int32Regs:$a, Int32Regs:$b),
              op_str # ".bf16x2 \t$dst, $a, $b;",
              [(set v2bf16:$dst, (op_pat v2bf16:$a, v2bf16:$b))]>,
              Requires<[hasBF16Math]>;
}

class BinOpAllowsFMA<SDPatternOperator operator>
    : PatFrag<(ops node:$A, node:$B),
              (operator node:$A, node:$B), [{
  return allowFMA() || N->getFlags().hasAllowContract();
}]>;

multiclass F3_fma_component<string op_str, SDNode op_node> {
  defm "" : F3<op_str, BinOpAllowsFMA<op_node>>;
  defm _rn : F3<op_str # ".rn", op_node>;
}

// Template for operations which take two f32 or f64 operands.  Provides three
// instructions: <OpcStr>.f64, <OpcStr>.f32, and <OpcStr>.ftz.f32 (flush
// subnormal inputs and results to zero).
multiclass F2<string OpcStr, SDNode OpNode> {
   def f64 :     NVPTXInst<(outs Float64Regs:$dst), (ins Float64Regs:$a),
                           !strconcat(OpcStr, ".f64 \t$dst, $a;"),
                           [(set f64:$dst, (OpNode f64:$a))]>;
   def f32_ftz : NVPTXInst<(outs Float32Regs:$dst), (ins Float32Regs:$a),
                           !strconcat(OpcStr, ".ftz.f32 \t$dst, $a;"),
                           [(set f32:$dst, (OpNode f32:$a))]>,
                           Requires<[doF32FTZ]>;
   def f32 :     NVPTXInst<(outs Float32Regs:$dst), (ins Float32Regs:$a),
                           !strconcat(OpcStr, ".f32 \t$dst, $a;"),
                           [(set f32:$dst, (OpNode f32:$a))]>;
}

multiclass F2_Support_Half<string OpcStr, SDNode OpNode> {
   def bf16 :      NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$a),
                           !strconcat(OpcStr, ".bf16 \t$dst, $a;"),
                           [(set bf16:$dst, (OpNode bf16:$a))]>,
                           Requires<[hasSM<80>, hasPTX<70>]>;
   def bf16x2 :    NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a),
                           !strconcat(OpcStr, ".bf16x2 \t$dst, $a;"),
                           [(set v2bf16:$dst, (OpNode v2bf16:$a))]>,
                           Requires<[hasSM<80>, hasPTX<70>]>;
   def f16_ftz :   NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$a),
                           !strconcat(OpcStr, ".ftz.f16 \t$dst, $a;"),
                           [(set f16:$dst, (OpNode f16:$a))]>,
                           Requires<[hasSM<53>, hasPTX<65>, doF32FTZ]>;
   def f16x2_ftz : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a),
                           !strconcat(OpcStr, ".ftz.f16x2 \t$dst, $a;"),
                           [(set v2f16:$dst, (OpNode v2f16:$a))]>,
                           Requires<[hasSM<53>, hasPTX<65>, doF32FTZ]>;
   def f16 :       NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$a),
                           !strconcat(OpcStr, ".f16 \t$dst, $a;"),
                           [(set f16:$dst, (OpNode f16:$a))]>,
                           Requires<[hasSM<53>, hasPTX<65>]>;
   def f16x2 :     NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a),
                           !strconcat(OpcStr, ".f16x2 \t$dst, $a;"),
                           [(set v2f16:$dst, (OpNode v2f16:$a))]>,
                           Requires<[hasSM<53>, hasPTX<65>]>;

}

// Variant where only .ftz.bf16 is supported.
multiclass F2_Support_Half_BF<string OpcStr, SDNode OpNode> {
   def bf16_ftz :  NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$a),
                           OpcStr # ".ftz.bf16 \t$dst, $a;",
                           [(set bf16:$dst, (OpNode bf16:$a))]>,
                           Requires<[hasSM<90>, hasPTX<78>]>;
   def bf16x2_ftz: NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a),
                           OpcStr # ".ftz.bf16x2 \t$dst, $a;",
                           [(set v2bf16:$dst, (OpNode v2bf16:$a))]>,
                           Requires<[hasSM<90>, hasPTX<78>]>;
}

//===----------------------------------------------------------------------===//
// NVPTX Instructions.
//===----------------------------------------------------------------------===//

//-----------------------------------
// Type Conversion
//-----------------------------------

let hasSideEffects = false in {
  // Generate a cvt to the given type from all possible types.  Each instance
  // takes a CvtMode immediate that defines the conversion mode to use.  It can
  // be CvtNONE to omit a conversion mode.
  multiclass CVT_FROM_ALL<string ToType, RegisterClass RC, list<Predicate> Preds = []> {
    def _s8 :
      NVPTXInst<(outs RC:$dst),
                (ins Int16Regs:$src, CvtMode:$mode),
                !strconcat("cvt${mode:base}${mode:ftz}${mode:sat}.",
                ToType, ".s8 \t$dst, $src;"), []>,
      Requires<Preds>;
    def _u8 :
      NVPTXInst<(outs RC:$dst),
                (ins Int16Regs:$src, CvtMode:$mode),
                !strconcat("cvt${mode:base}${mode:ftz}${mode:sat}.",
                ToType, ".u8 \t$dst, $src;"), []>,
      Requires<Preds>;
    def _s16 :
      NVPTXInst<(outs RC:$dst),
                (ins Int16Regs:$src, CvtMode:$mode),
                !strconcat("cvt${mode:base}${mode:ftz}${mode:sat}.",
                ToType, ".s16 \t$dst, $src;"), []>,
      Requires<Preds>;
    def _u16 :
      NVPTXInst<(outs RC:$dst),
                (ins Int16Regs:$src, CvtMode:$mode),
                !strconcat("cvt${mode:base}${mode:ftz}${mode:sat}.",
                ToType, ".u16 \t$dst, $src;"), []>,
      Requires<Preds>;
    def _s32 :
      NVPTXInst<(outs RC:$dst),
                (ins Int32Regs:$src, CvtMode:$mode),
                !strconcat("cvt${mode:base}${mode:ftz}${mode:sat}.",
                ToType, ".s32 \t$dst, $src;"), []>,
      Requires<Preds>;
    def _u32 :
      NVPTXInst<(outs RC:$dst),
                (ins Int32Regs:$src, CvtMode:$mode),
                !strconcat("cvt${mode:base}${mode:ftz}${mode:sat}.",
                ToType, ".u32 \t$dst, $src;"), []>,
      Requires<Preds>;
    def _s64 :
      NVPTXInst<(outs RC:$dst),
                (ins Int64Regs:$src, CvtMode:$mode),
                !strconcat("cvt${mode:base}${mode:ftz}${mode:sat}.",
                ToType, ".s64 \t$dst, $src;"), []>,
      Requires<Preds>;
    def _u64 :
      NVPTXInst<(outs RC:$dst),
                (ins Int64Regs:$src, CvtMode:$mode),
                !strconcat("cvt${mode:base}${mode:ftz}${mode:sat}.",
                ToType, ".u64 \t$dst, $src;"), []>,
      Requires<Preds>;
    def _f16 :
      NVPTXInst<(outs RC:$dst),
                (ins Int16Regs:$src, CvtMode:$mode),
                !strconcat("cvt${mode:base}${mode:ftz}${mode:sat}.",
                ToType, ".f16 \t$dst, $src;"), []>,
      Requires<Preds>;
    def _bf16 :
      NVPTXInst<(outs RC:$dst),
                (ins Int16Regs:$src, CvtMode:$mode),
                !strconcat("cvt${mode:base}${mode:ftz}${mode:relu}${mode:sat}.",
                ToType, ".bf16 \t$dst, $src;"), []>,
      Requires<!if(!eq(ToType, "f32"),
                   // bf16->f32 was introduced early.
                   [hasPTX<71>, hasSM<80>],
                   // bf16->everything else needs sm90/ptx78
                   [hasPTX<78>, hasSM<90>])>;
    def _f32 :
      NVPTXInst<(outs RC:$dst),
                (ins Float32Regs:$src, CvtMode:$mode),
                !strconcat("cvt${mode:base}${mode:ftz}${mode:relu}${mode:sat}.",
                ToType, ".f32 \t$dst, $src;"), []>,
      Requires<!if(!eq(ToType, "bf16"),
                   // f32->bf16 was introduced early.
                   [hasPTX<70>, hasSM<80>],
                   Preds)>;
    def _f64 :
      NVPTXInst<(outs RC:$dst),
                (ins Float64Regs:$src, CvtMode:$mode),
                !strconcat("cvt${mode:base}${mode:ftz}${mode:sat}.",
                ToType, ".f64 \t$dst, $src;"), []>,
      Requires<Preds>;
  }

  // Generate cvts from all types to all types.
  defm CVT_s8  : CVT_FROM_ALL<"s8",  Int16Regs>;
  defm CVT_u8  : CVT_FROM_ALL<"u8",  Int16Regs>;
  defm CVT_s16 : CVT_FROM_ALL<"s16", Int16Regs>;
  defm CVT_u16 : CVT_FROM_ALL<"u16", Int16Regs>;
  defm CVT_s32 : CVT_FROM_ALL<"s32", Int32Regs>;
  defm CVT_u32 : CVT_FROM_ALL<"u32", Int32Regs>;
  defm CVT_s64 : CVT_FROM_ALL<"s64", Int64Regs>;
  defm CVT_u64 : CVT_FROM_ALL<"u64", Int64Regs>;
  defm CVT_f16 : CVT_FROM_ALL<"f16", Int16Regs>;
  defm CVT_bf16 : CVT_FROM_ALL<"bf16", Int16Regs, [hasPTX<78>, hasSM<90>]>;
  defm CVT_f32 : CVT_FROM_ALL<"f32", Float32Regs>;
  defm CVT_f64 : CVT_FROM_ALL<"f64", Float64Regs>;

  // These cvts are different from those above: The source and dest registers
  // are of the same type.
  def CVT_INREG_s16_s8 :  NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$src),
                                    "cvt.s16.s8 \t$dst, $src;", []>;
  def CVT_INREG_s32_s8 :  NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$src),
                                    "cvt.s32.s8 \t$dst, $src;", []>;
  def CVT_INREG_s32_s16 : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$src),
                                    "cvt.s32.s16 \t$dst, $src;", []>;
  def CVT_INREG_s64_s8 :  NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$src),
                                    "cvt.s64.s8 \t$dst, $src;", []>;
  def CVT_INREG_s64_s16 : NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$src),
                                    "cvt.s64.s16 \t$dst, $src;", []>;
  def CVT_INREG_s64_s32 : NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$src),
                                    "cvt.s64.s32 \t$dst, $src;", []>;

  multiclass CVT_FROM_FLOAT_V2_SM80<string FromName, RegisterClass RC> {
    def _f32 :
      NVPTXInst<(outs RC:$dst),
                (ins Float32Regs:$src1, Float32Regs:$src2,  CvtMode:$mode),
                !strconcat("cvt${mode:base}${mode:relu}.",
                FromName, ".f32 \t$dst, $src1, $src2;"), []>,
    Requires<[hasPTX<70>, hasSM<80>]>;
  }

  defm CVT_f16x2 : CVT_FROM_FLOAT_V2_SM80<"f16x2", Int32Regs>;
  defm CVT_bf16x2 : CVT_FROM_FLOAT_V2_SM80<"bf16x2", Int32Regs>;

  // FP8 conversions.
  multiclass CVT_TO_F8X2<string F8Name> {
    def _f32 :
      NVPTXInst<(outs Int16Regs:$dst),
                (ins Float32Regs:$src1, Float32Regs:$src2, CvtMode:$mode),
                !strconcat("cvt${mode:base}.satfinite${mode:relu}.",
                F8Name, "x2.f32 \t$dst, $src1, $src2;"), []>,
      Requires<[hasPTX<81>, hasSM<89>]>;
    def _f16x2 :
      NVPTXInst<(outs Int16Regs:$dst),
                (ins Int32Regs:$src, CvtMode:$mode),
                !strconcat("cvt${mode:base}.satfinite${mode:relu}.",
                F8Name, "x2.f16x2 \t$dst, $src;"), []>,
      Requires<[hasPTX<81>, hasSM<89>]>;
  }

  defm CVT_e4m3x2 : CVT_TO_F8X2<"e4m3">;
  defm CVT_e5m2x2 : CVT_TO_F8X2<"e5m2">;

  class CVT_f16x2_fp8<string F8Name> :
    NVPTXInst<(outs Int32Regs:$dst),
              (ins Int16Regs:$src, CvtMode:$mode),
              !strconcat("cvt${mode:base}${mode:relu}.f16x2.",
              F8Name, "x2 \t$dst, $src;"), []>,
    Requires<[hasPTX<81>, hasSM<89>]>;

  def CVT_f16x2_e4m3x2 : CVT_f16x2_fp8<"e4m3">;
  def CVT_f16x2_e5m2x2 : CVT_f16x2_fp8<"e5m2">;

  // Float to TF32 conversions
  multiclass CVT_TO_TF32<string Modifier, list<Predicate> Preds = [hasPTX<78>, hasSM<90>]> {
    defvar Intr = !cast<Intrinsic>("int_nvvm_f2tf32_" # !subst(".", "_", Modifier));

    def NAME : NVPTXInst<(outs Int32Regs:$dst), (ins Float32Regs:$src),
               "cvt." # Modifier # ".tf32.f32 \t$dst, $src;",
               [(set i32:$dst, (Intr f32:$src))]>,
               Requires<Preds>;
  }

  defm CVT_to_tf32_rn : CVT_TO_TF32<"rn">;
  defm CVT_to_tf32_rz : CVT_TO_TF32<"rz">;
  defm CVT_to_tf32_rn_relu  : CVT_TO_TF32<"rn.relu">;
  defm CVT_to_tf32_rz_relu  : CVT_TO_TF32<"rz.relu">;
  defm CVT_to_tf32_rna      : CVT_TO_TF32<"rna", [hasPTX<70>, hasSM<80>]>;
  defm CVT_to_tf32_rna_satf : CVT_TO_TF32<"rna.satfinite", [hasPTX<81>, hasSM<89>]>;

  defm CVT_to_tf32_rn_satf : CVT_TO_TF32<"rn.satfinite", [hasPTX<86>, hasSM<100>]>;
  defm CVT_to_tf32_rz_satf : CVT_TO_TF32<"rz.satfinite", [hasPTX<86>, hasSM<100>]>;
  defm CVT_to_tf32_rn_relu_satf  : CVT_TO_TF32<"rn.relu.satfinite", [hasPTX<86>, hasSM<100>]>;
  defm CVT_to_tf32_rz_relu_satf  : CVT_TO_TF32<"rz.relu.satfinite", [hasPTX<86>, hasSM<100>]>;

  // FP6 conversions.
  foreach type = ["e2m3x2", "e3m2x2"] in {
    def CVT_ # type # _f32_sf : NVPTXInst<(outs Int16Regs:$dst),
                                          (ins Float32Regs:$src1, 
                                            Float32Regs:$src2, CvtMode:$mode),
                                          "cvt${mode:base}.satfinite${mode:relu}." 
                                            # type # ".f32 \t$dst, $src1, $src2;", []>;
    def CVT_f16x2_ # type : NVPTXInst<(outs Int32Regs:$dst),
                                      (ins Int16Regs:$src, CvtMode:$mode),
                                      "cvt${mode:base}${mode:relu}.f16x2." 
                                        # type # " \t$dst, $src;", []>;
  }
  
  // FP4 conversions.
  def CVT_e2m1x2_f32_sf : NVPTXInst<(outs Int16Regs:$dst),
      (ins Float32Regs:$src1, Float32Regs:$src2, CvtMode:$mode),
      !strconcat("{{ \n\t",
                 ".reg .b8 \t%e2m1x2_out; \n\t",
                 "cvt${mode:base}.satfinite${mode:relu}.e2m1x2.f32 \t%e2m1x2_out, $src1, $src2; \n\t",
                 "cvt.u16.u8 \t$dst, %e2m1x2_out; \n\t",
                 "}}"), []>;

  def CVT_f16x2_e2m1x2 : NVPTXInst<(outs Int32Regs:$dst),
      (ins Int16Regs:$src, CvtMode:$mode),
      !strconcat("{{ \n\t",
                 ".reg .b8 \t%e2m1x2_in; \n\t",
                 "cvt.u8.u16 \t%e2m1x2_in, $src; \n\t",
                 "cvt${mode:base}${mode:relu}.f16x2.e2m1x2 \t$dst, %e2m1x2_in; \n\t",
                 "}}"), []>;

  // UE8M0x2 conversions.
  class CVT_f32_to_ue8m0x2<string sat = ""> :
    NVPTXInst<(outs Int16Regs:$dst),
              (ins Float32Regs:$src1, Float32Regs:$src2, CvtMode:$mode),
              "cvt${mode:base}" # sat # ".ue8m0x2.f32 \t$dst, $src1, $src2;", []>;
  
  class CVT_bf16x2_to_ue8m0x2<string sat = ""> :
    NVPTXInst<(outs Int16Regs:$dst),
              (ins Int32Regs:$src, CvtMode:$mode),
              "cvt${mode:base}" # sat # ".ue8m0x2.bf16x2 \t$dst, $src;", []>;
              
  def CVT_ue8m0x2_f32 : CVT_f32_to_ue8m0x2;
  def CVT_ue8m0x2_f32_sf : CVT_f32_to_ue8m0x2<".satfinite">;
  def CVT_ue8m0x2_bf16x2 : CVT_bf16x2_to_ue8m0x2;
  def CVT_ue8m0x2_bf16x2_sf : CVT_bf16x2_to_ue8m0x2<".satfinite">;

  def CVT_bf16x2_ue8m0x2 :
      NVPTXInst<(outs Int32Regs:$dst),
                (ins Int16Regs:$src),
                "cvt.rn.bf16x2.ue8m0x2 \t$dst, $src;", []>;

}

def fpround_oneuse : OneUse1<fpround>;
def : Pat<(v2bf16 (build_vector (bf16 (fpround_oneuse f32:$lo)),
                                (bf16 (fpround_oneuse f32:$hi)))),
          (CVT_bf16x2_f32 $hi, $lo, CvtRN)>,
      Requires<[hasPTX<70>, hasSM<80>, hasBF16Math]>;

def : Pat<(v2f16 (build_vector (f16 (fpround_oneuse f32:$lo)),
                               (f16 (fpround_oneuse f32:$hi)))),
          (CVT_f16x2_f32 $hi, $lo, CvtRN)>,
      Requires<[hasPTX<70>, hasSM<80>, useFP16Math]>;

//-----------------------------------
// Selection instructions (selp)
//-----------------------------------

// TODO: Missing slct

// selp instructions that don't have any pattern matches; we explicitly use
// them within this file.
let hasSideEffects = false in {
  multiclass SELP_PATTERN<string TypeStr, RegTyInfo t> {
    defvar asm_str = "selp." # TypeStr # " \t$dst, $a, $b, $p;";
    def rr :
      NVPTXInst<(outs t.RC:$dst),
                (ins t.RC:$a, t.RC:$b, Int1Regs:$p),
                asm_str,
                [(set t.Ty:$dst, (select i1:$p, t.Ty:$a, t.Ty:$b))]>;
    def ri :
      NVPTXInst<(outs t.RC:$dst),
                (ins t.RC:$a, t.Imm:$b, Int1Regs:$p),
                asm_str,
                [(set t.Ty:$dst, (select i1:$p, t.Ty:$a, t.ImmNode:$b))]>;
    def ir :
      NVPTXInst<(outs t.RC:$dst),
                (ins t.Imm:$a, t.RC:$b, Int1Regs:$p),
                asm_str,
                [(set t.Ty:$dst, (select i1:$p, t.ImmNode:$a, t.Ty:$b))]>;
    def ii :
      NVPTXInst<(outs t.RC:$dst),
                (ins t.Imm:$a, t.Imm:$b, Int1Regs:$p),
                asm_str,
                [(set t.Ty:$dst, (select i1:$p, t.ImmNode:$a, t.ImmNode:$b))]>;
  }
}

// Don't pattern match on selp.{s,u}{16,32,64} -- selp.b{16,32,64} is just as
// good.
defm SELP_b16  : SELP_PATTERN<"b16", I16RT>;
defm SELP_b32  : SELP_PATTERN<"b32", I32RT>;
defm SELP_b64  : SELP_PATTERN<"b64", I64RT>;
defm SELP_f16  : SELP_PATTERN<"b16", F16RT>;
defm SELP_bf16 : SELP_PATTERN<"b16", BF16RT>;
defm SELP_f32  : SELP_PATTERN<"f32", F32RT>;
defm SELP_f64  : SELP_PATTERN<"f64", F64RT>;

// This does not work as tablegen fails to infer the type of 'imm'.
// def v2f16imm : Operand<v2f16>;
// defm SELP_f16x2 : SELP_PATTERN<"b32", v2f16, Int32Regs, v2f16imm, imm>;

foreach vt = [v2f16, v2bf16, v2i16, v4i8] in {
def : Pat<(vt (select i1:$p, vt:$a, vt:$b)),
          (SELP_b32rr $a, $b, $p)>;
}

//-----------------------------------
// Test Instructions
//-----------------------------------

def fabs_oneuse : OneUse1<fabs>;

def TESTINF_f32r : NVPTXInst<(outs Int1Regs:$p), (ins Float32Regs:$a),
                             "testp.infinite.f32 \t$p, $a;",
                             [(set i1:$p, (seteq (fabs_oneuse f32:$a), fpimm_pos_inf<f32>))]>;
def TESTINF_f64r : NVPTXInst<(outs Int1Regs:$p), (ins Float64Regs:$a),
                             "testp.infinite.f64 \t$p, $a;",
                             [(set i1:$p, (seteq (fabs_oneuse f64:$a), fpimm_pos_inf<f64>))]>;

//-----------------------------------
// Integer Arithmetic
//-----------------------------------

// Template for xor masquerading as int1 arithmetic.
multiclass ADD_SUB_i1<SDNode OpNode> {
   def _rr: NVPTXInst<(outs Int1Regs:$dst), (ins Int1Regs:$a, Int1Regs:$b),
                      "xor.pred \t$dst, $a, $b;",
                      [(set i1:$dst, (OpNode i1:$a, i1:$b))]>;
   def _ri: NVPTXInst<(outs Int1Regs:$dst), (ins Int1Regs:$a, i1imm:$b),
                      "xor.pred \t$dst, $a, $b;",
                      [(set i1:$dst, (OpNode i1:$a, (imm):$b))]>;
}

// int1 addition and subtraction are both just xor.
defm ADD_i1 : ADD_SUB_i1<add>;
defm SUB_i1 : ADD_SUB_i1<sub>;

// int16, int32, and int64 signed addition.  Since nvptx is 2's complement, we
// also use these for unsigned arithmetic.
defm ADD : I3<"add.s", add, commutative = true>;
defm SUB : I3<"sub.s", sub, commutative = false>;

def ADD16x2 : I16x2<"add.s", add>;

// in32 and int64 addition and subtraction with carry-out.
defm ADDCC : ADD_SUB_INT_CARRY<"add.cc", addc, commutative = true>;
defm SUBCC : ADD_SUB_INT_CARRY<"sub.cc", subc, commutative = false>;

// int32 and int64 addition and subtraction with carry-in and carry-out.
defm ADDCCC : ADD_SUB_INT_CARRY<"addc.cc", adde, commutative = true>;
defm SUBCCC : ADD_SUB_INT_CARRY<"subc.cc", sube, commutative = false>;

defm MULT : I3<"mul.lo.s", mul, commutative = true>;

defm MULTHS : I3<"mul.hi.s", mulhs, commutative = true>;
defm MULTHU : I3<"mul.hi.u", mulhu, commutative = true>;

defm SDIV : I3<"div.s", sdiv, commutative = false>;
defm UDIV : I3<"div.u", udiv, commutative = false>;

// The ri versions of rem.s and rem.u won't be selected; DAGCombiner::visitSREM
// will lower it.
defm SREM : I3<"rem.s", srem, commutative = false>;
defm UREM : I3<"rem.u", urem, commutative = false>;

// Integer absolute value.  NumBits should be one minus the bit width of RC.
// This idiom implements the algorithm at
// http://graphics.stanford.edu/~seander/bithacks.html#IntegerAbs.
multiclass ABS<ValueType T, RegisterClass RC, string SizeName> {
  def : NVPTXInst<(outs RC:$dst), (ins RC:$a),
                  !strconcat("abs", SizeName, " \t$dst, $a;"),
                  [(set T:$dst, (abs T:$a))]>;
}
defm ABS_16 : ABS<i16, Int16Regs, ".s16">;
defm ABS_32 : ABS<i32, Int32Regs, ".s32">;
defm ABS_64 : ABS<i64, Int64Regs, ".s64">;

// Integer min/max.
defm SMAX : I3<"max.s", smax, commutative = true>;
defm UMAX : I3<"max.u", umax, commutative = true>;
defm SMIN : I3<"min.s", smin, commutative = true>;
defm UMIN : I3<"min.u", umin, commutative = true>;

def SMAX16x2 : I16x2<"max.s", smax>;
def UMAX16x2 : I16x2<"max.u", umax>;
def SMIN16x2 : I16x2<"min.s", smin>;
def UMIN16x2 : I16x2<"min.u", umin>;


//
// Wide multiplication
//
def MULWIDES64 :
  NVPTXInst<(outs Int64Regs:$dst), (ins Int32Regs:$a, Int32Regs:$b),
            "mul.wide.s32 \t$dst, $a, $b;", []>;
def MULWIDES64Imm :
  NVPTXInst<(outs Int64Regs:$dst), (ins Int32Regs:$a, i32imm:$b),
            "mul.wide.s32 \t$dst, $a, $b;", []>;
def MULWIDES64Imm64 :
  NVPTXInst<(outs Int64Regs:$dst), (ins Int32Regs:$a, i64imm:$b),
            "mul.wide.s32 \t$dst, $a, $b;", []>;

def MULWIDEU64 :
  NVPTXInst<(outs Int64Regs:$dst), (ins Int32Regs:$a, Int32Regs:$b),
            "mul.wide.u32 \t$dst, $a, $b;", []>;
def MULWIDEU64Imm :
  NVPTXInst<(outs Int64Regs:$dst), (ins Int32Regs:$a, i32imm:$b),
            "mul.wide.u32 \t$dst, $a, $b;", []>;
def MULWIDEU64Imm64 :
  NVPTXInst<(outs Int64Regs:$dst), (ins Int32Regs:$a, i64imm:$b),
            "mul.wide.u32 \t$dst, $a, $b;", []>;

def MULWIDES32 :
  NVPTXInst<(outs Int32Regs:$dst), (ins Int16Regs:$a, Int16Regs:$b),
            "mul.wide.s16 \t$dst, $a, $b;", []>;
def MULWIDES32Imm :
  NVPTXInst<(outs Int32Regs:$dst), (ins Int16Regs:$a, i16imm:$b),
            "mul.wide.s16 \t$dst, $a, $b;", []>;
def MULWIDES32Imm32 :
  NVPTXInst<(outs Int32Regs:$dst), (ins Int16Regs:$a, i32imm:$b),
            "mul.wide.s16 \t$dst, $a, $b;", []>;

def MULWIDEU32 :
  NVPTXInst<(outs Int32Regs:$dst), (ins Int16Regs:$a, Int16Regs:$b),
            "mul.wide.u16 \t$dst, $a, $b;", []>;
def MULWIDEU32Imm :
  NVPTXInst<(outs Int32Regs:$dst), (ins Int16Regs:$a, i16imm:$b),
            "mul.wide.u16 \t$dst, $a, $b;", []>;
def MULWIDEU32Imm32 :
  NVPTXInst<(outs Int32Regs:$dst), (ins Int16Regs:$a, i32imm:$b),
            "mul.wide.u16 \t$dst, $a, $b;", []>;

def SDTMulWide : SDTypeProfile<1, 2, [SDTCisSameAs<1, 2>]>;
def mul_wide_signed : SDNode<"NVPTXISD::MUL_WIDE_SIGNED", SDTMulWide>;
def mul_wide_unsigned : SDNode<"NVPTXISD::MUL_WIDE_UNSIGNED", SDTMulWide>;

// Matchers for signed, unsigned mul.wide ISD nodes.
def : Pat<(i32 (mul_wide_signed i16:$a, i16:$b)),
          (MULWIDES32 $a, $b)>,
      Requires<[doMulWide]>;
def : Pat<(i32 (mul_wide_signed i16:$a, imm:$b)),
          (MULWIDES32Imm $a, imm:$b)>,
      Requires<[doMulWide]>;
def : Pat<(i32 (mul_wide_unsigned i16:$a, i16:$b)),
          (MULWIDEU32 $a, $b)>,
      Requires<[doMulWide]>;
def : Pat<(i32 (mul_wide_unsigned i16:$a, imm:$b)),
          (MULWIDEU32Imm $a, imm:$b)>,
      Requires<[doMulWide]>;

def : Pat<(i64 (mul_wide_signed i32:$a, i32:$b)),
          (MULWIDES64 $a, $b)>,
      Requires<[doMulWide]>;
def : Pat<(i64 (mul_wide_signed i32:$a, imm:$b)),
          (MULWIDES64Imm $a, imm:$b)>,
      Requires<[doMulWide]>;
def : Pat<(i64 (mul_wide_unsigned i32:$a, i32:$b)),
          (MULWIDEU64 $a, $b)>,
      Requires<[doMulWide]>;
def : Pat<(i64 (mul_wide_unsigned i32:$a, imm:$b)),
          (MULWIDEU64Imm $a, imm:$b)>,
      Requires<[doMulWide]>;

// Predicates used for converting some patterns to mul.wide.
def SInt32Const : PatLeaf<(imm), [{
  const APInt &v = N->getAPIntValue();
  return v.isSignedIntN(32);
}]>;

def UInt32Const : PatLeaf<(imm), [{
  const APInt &v = N->getAPIntValue();
  return v.isIntN(32);
}]>;

def SInt16Const : PatLeaf<(imm), [{
  const APInt &v = N->getAPIntValue();
  return v.isSignedIntN(16);
}]>;

def UInt16Const : PatLeaf<(imm), [{
  const APInt &v = N->getAPIntValue();
  return v.isIntN(16);
}]>;

def IntConst_0_30 : PatLeaf<(imm), [{
  // Check if 0 <= v < 31; only then will the result of (x << v) be an int32.
  const APInt &v = N->getAPIntValue();
  return v.sge(0) && v.slt(31);
}]>;

def IntConst_0_14 : PatLeaf<(imm), [{
  // Check if 0 <= v < 15; only then will the result of (x << v) be an int16.
  const APInt &v = N->getAPIntValue();
  return v.sge(0) && v.slt(15);
}]>;

def SHL2MUL32 : SDNodeXForm<imm, [{
  const APInt &v = N->getAPIntValue();
  APInt temp(32, 1);
  return CurDAG->getTargetConstant(temp.shl(v), SDLoc(N), MVT::i32);
}]>;

def SHL2MUL16 : SDNodeXForm<imm, [{
  const APInt &v = N->getAPIntValue();
  APInt temp(16, 1);
  return CurDAG->getTargetConstant(temp.shl(v), SDLoc(N), MVT::i16);
}]>;

// Convert "sign/zero-extend, then shift left by an immediate" to mul.wide.
def : Pat<(shl (sext i32:$a), (i32 IntConst_0_30:$b)),
          (MULWIDES64Imm $a, (SHL2MUL32 $b))>,
      Requires<[doMulWide]>;
def : Pat<(shl (zext i32:$a), (i32 IntConst_0_30:$b)),
          (MULWIDEU64Imm $a, (SHL2MUL32 $b))>,
      Requires<[doMulWide]>;

def : Pat<(shl (sext i16:$a), (i16 IntConst_0_14:$b)),
          (MULWIDES32Imm $a, (SHL2MUL16 $b))>,
      Requires<[doMulWide]>;
def : Pat<(shl (zext i16:$a), (i16 IntConst_0_14:$b)),
          (MULWIDEU32Imm $a, (SHL2MUL16 $b))>,
      Requires<[doMulWide]>;

// Convert "sign/zero-extend then multiply" to mul.wide.
def : Pat<(mul (sext i32:$a), (sext i32:$b)),
          (MULWIDES64 $a, $b)>,
      Requires<[doMulWide]>;
def : Pat<(mul (sext i32:$a), (i64 SInt32Const:$b)),
          (MULWIDES64Imm64 $a, (i64 SInt32Const:$b))>,
      Requires<[doMulWide]>;

def : Pat<(mul (zext i32:$a), (zext i32:$b)),
          (MULWIDEU64 $a, $b)>,
      Requires<[doMulWide]>;
def : Pat<(mul (zext i32:$a), (i64 UInt32Const:$b)),
          (MULWIDEU64Imm64 $a, (i64 UInt32Const:$b))>,
      Requires<[doMulWide]>;

def : Pat<(mul (sext i16:$a), (sext i16:$b)),
          (MULWIDES32 $a, $b)>,
      Requires<[doMulWide]>;
def : Pat<(mul (sext i16:$a), (i32 SInt16Const:$b)),
          (MULWIDES32Imm32 $a, (i32 SInt16Const:$b))>,
      Requires<[doMulWide]>;

def : Pat<(mul (zext i16:$a), (zext i16:$b)),
          (MULWIDEU32 $a, $b)>,
      Requires<[doMulWide]>;
def : Pat<(mul (zext i16:$a), (i32 UInt16Const:$b)),
          (MULWIDEU32Imm32 $a, (i32 UInt16Const:$b))>,
      Requires<[doMulWide]>;

//
// Integer multiply-add
//
def mul_oneuse : PatFrag<(ops node:$a, node:$b), (mul node:$a, node:$b), [{
  return N->hasOneUse();
}]>;

multiclass MAD<string Ptx, ValueType VT, NVPTXRegClass Reg, Operand Imm> {
  def rrr:
    NVPTXInst<(outs Reg:$dst),
              (ins Reg:$a, Reg:$b, Reg:$c),
              Ptx # " \t$dst, $a, $b, $c;",
              [(set VT:$dst, (add (mul_oneuse VT:$a, VT:$b), VT:$c))]>;

  def rir:
    NVPTXInst<(outs Reg:$dst),
              (ins Reg:$a, Imm:$b, Reg:$c),
              Ptx # " \t$dst, $a, $b, $c;",
              [(set VT:$dst, (add (mul_oneuse VT:$a, imm:$b), VT:$c))]>;
  def rri:
    NVPTXInst<(outs Reg:$dst),
              (ins Reg:$a, Reg:$b, Imm:$c),
              Ptx # " \t$dst, $a, $b, $c;",
              [(set VT:$dst, (add (mul_oneuse VT:$a, VT:$b), imm:$c))]>;
  def rii:
    NVPTXInst<(outs Reg:$dst),
              (ins Reg:$a, Imm:$b, Imm:$c),
              Ptx # " \t$dst, $a, $b, $c;",
              [(set VT:$dst, (add (mul_oneuse VT:$a, imm:$b), imm:$c))]>;
}

let Predicates = [hasOptEnabled] in {
defm MAD16 : MAD<"mad.lo.s16", i16, Int16Regs, i16imm>;
defm MAD32 : MAD<"mad.lo.s32", i32, Int32Regs, i32imm>;
defm MAD64 : MAD<"mad.lo.s64", i64, Int64Regs, i64imm>;
}

def INEG16 :
  NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$src),
            "neg.s16 \t$dst, $src;",
            [(set i16:$dst, (ineg i16:$src))]>;
def INEG32 :
  NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$src),
            "neg.s32 \t$dst, $src;",
            [(set i32:$dst, (ineg i32:$src))]>;
def INEG64 :
  NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$src),
            "neg.s64 \t$dst, $src;",
            [(set i64:$dst, (ineg i64:$src))]>;

//-----------------------------------
// Floating Point Arithmetic
//-----------------------------------

// Constant 1.0f
def f32imm_1 : FPImmLeaf<f32, [{
  return &Imm.getSemantics() == &llvm::APFloat::IEEEsingle() &&
         Imm.convertToFloat() == 1.0f;
}]>;
// Constant 1.0 (double)
def f64imm_1 : FPImmLeaf<f64, [{
  return &Imm.getSemantics() == &llvm::APFloat::IEEEdouble() &&
         Imm.convertToDouble() == 1.0;
}]>;
// Constant -1.0 (double)
def f64imm_neg1 : FPImmLeaf<f64, [{
  return &Imm.getSemantics() == &llvm::APFloat::IEEEdouble() &&
         Imm.convertToDouble() == -1.0;
}]>;

defm FADD : F3_fma_component<"add", fadd>;
defm FSUB : F3_fma_component<"sub", fsub>;
defm FMUL : F3_fma_component<"mul", fmul>;

defm FMIN : FMINIMUMMAXIMUM<"min", /* NaN */ false, fminnum>;
defm FMAX : FMINIMUMMAXIMUM<"max", /* NaN */ false, fmaxnum>;
defm FMINNAN : FMINIMUMMAXIMUM<"min.NaN", /* NaN */ true, fminimum>;
defm FMAXNAN : FMINIMUMMAXIMUM<"max.NaN", /* NaN */ true, fmaximum>;

defm FABS  : F2<"abs", fabs>;
defm FNEG  : F2<"neg", fneg>;
defm FABS_H: F2_Support_Half<"abs", fabs>;
defm FNEG_H: F2_Support_Half<"neg", fneg>;

defm FSQRT : F2<"sqrt.rn", fsqrt>;

defm FEXP2_H: F2_Support_Half_BF<"ex2.approx", fexp2>;

//
// F16 NEG
//
class FNEG_F16_F16X2<string OpcStr, ValueType T, RegisterClass RC, Predicate Pred> :
      NVPTXInst<(outs RC:$dst), (ins RC:$src),
                !strconcat(OpcStr, " \t$dst, $src;"),
                [(set T:$dst, (fneg T:$src))]>,
                Requires<[useFP16Math, hasPTX<60>, hasSM<53>, Pred]>;
def FNEG16_ftz   : FNEG_F16_F16X2<"neg.ftz.f16", f16, Int16Regs, doF32FTZ>;
def FNEG16       : FNEG_F16_F16X2<"neg.f16", f16, Int16Regs, True>;
def FNEG16x2_ftz : FNEG_F16_F16X2<"neg.ftz.f16x2", v2f16, Int32Regs, doF32FTZ>;
def FNEG16x2     : FNEG_F16_F16X2<"neg.f16x2", v2f16, Int32Regs, True>;

//
// BF16 NEG
//

class FNEG_BF16_F16X2<string OpcStr, ValueType T, RegisterClass RC, Predicate Pred> :
      NVPTXInst<(outs RC:$dst), (ins RC:$src),
                !strconcat(OpcStr, " \t$dst, $src;"),
                [(set T:$dst, (fneg T:$src))]>,
                Requires<[hasBF16Math, hasPTX<70>, hasSM<80>, Pred]>;
def BFNEG16_ftz   : FNEG_BF16_F16X2<"neg.ftz.bf16", bf16, Int16Regs, doF32FTZ>;
def BFNEG16       : FNEG_BF16_F16X2<"neg.bf16", bf16, Int16Regs, True>;
def BFNEG16x2_ftz : FNEG_BF16_F16X2<"neg.ftz.bf16x2", v2bf16, Int32Regs, doF32FTZ>;
def BFNEG16x2     : FNEG_BF16_F16X2<"neg.bf16x2", v2bf16, Int32Regs, True>;

//
// F64 division
//
def FRCP64r :
  NVPTXInst<(outs Float64Regs:$dst),
            (ins Float64Regs:$b),
            "rcp.rn.f64 \t$dst, $b;",
            [(set f64:$dst, (fdiv f64imm_1, f64:$b))]>;
def FDIV64rr :
  NVPTXInst<(outs Float64Regs:$dst),
            (ins Float64Regs:$a, Float64Regs:$b),
            "div.rn.f64 \t$dst, $a, $b;",
            [(set f64:$dst, (fdiv f64:$a, f64:$b))]>;
def FDIV64ri :
  NVPTXInst<(outs Float64Regs:$dst),
            (ins Float64Regs:$a, f64imm:$b),
            "div.rn.f64 \t$dst, $a, $b;",
            [(set f64:$dst, (fdiv f64:$a, fpimm:$b))]>;

// fdiv will be converted to rcp
// fneg (fdiv 1.0, X) => fneg (rcp.rn X)
def : Pat<(fdiv f64imm_neg1, f64:$b),
          (FNEGf64 (FRCP64r $b))>;

//
// F32 Approximate reciprocal
//

def fdiv_approx : PatFrag<(ops node:$a, node:$b),
                          (fdiv node:$a, node:$b), [{
  return getDivF32Level(N) == NVPTX::DivPrecisionLevel::Approx;
}]>;


def FRCP32_approx_r_ftz :
  NVPTXInst<(outs Float32Regs:$dst),
            (ins Float32Regs:$b),
            "rcp.approx.ftz.f32 \t$dst, $b;",
            [(set f32:$dst, (fdiv_approx f32imm_1, f32:$b))]>,
            Requires<[doF32FTZ]>;
def FRCP32_approx_r :
  NVPTXInst<(outs Float32Regs:$dst),
            (ins Float32Regs:$b),
            "rcp.approx.f32 \t$dst, $b;",
            [(set f32:$dst, (fdiv_approx f32imm_1, f32:$b))]>;

//
// F32 Approximate division
//
def FDIV32approxrr_ftz :
  NVPTXInst<(outs Float32Regs:$dst),
            (ins Float32Regs:$a, Float32Regs:$b),
            "div.approx.ftz.f32 \t$dst, $a, $b;",
            [(set f32:$dst, (fdiv_approx f32:$a, f32:$b))]>,
            Requires<[doF32FTZ]>;
def FDIV32approxri_ftz :
  NVPTXInst<(outs Float32Regs:$dst),
            (ins Float32Regs:$a, f32imm:$b),
            "div.approx.ftz.f32 \t$dst, $a, $b;",
            [(set f32:$dst, (fdiv_approx f32:$a, fpimm:$b))]>,
            Requires<[doF32FTZ]>;
def FDIV32approxrr :
  NVPTXInst<(outs Float32Regs:$dst),
            (ins Float32Regs:$a, Float32Regs:$b),
            "div.approx.f32 \t$dst, $a, $b;",
            [(set f32:$dst, (fdiv_approx f32:$a, f32:$b))]>;
def FDIV32approxri :
  NVPTXInst<(outs Float32Regs:$dst),
            (ins Float32Regs:$a, f32imm:$b),
            "div.approx.f32 \t$dst, $a, $b;",
            [(set f32:$dst, (fdiv_approx f32:$a, fpimm:$b))]>;
//
// F32 Semi-accurate reciprocal
//
// rcp.approx gives the same result as div.full(1.0f, a) and is faster.
//

def fdiv_full : PatFrag<(ops node:$a, node:$b),
                        (fdiv node:$a, node:$b), [{
  return getDivF32Level(N) == NVPTX::DivPrecisionLevel::Full;
}]>;


def : Pat<(fdiv_full f32imm_1, f32:$b),
          (FRCP32_approx_r_ftz $b)>,
      Requires<[doF32FTZ]>;

def : Pat<(fdiv_full f32imm_1, f32:$b),
          (FRCP32_approx_r $b)>;

//
// F32 Semi-accurate division
//
def FDIV32rr_ftz :
  NVPTXInst<(outs Float32Regs:$dst),
            (ins Float32Regs:$a, Float32Regs:$b),
            "div.full.ftz.f32 \t$dst, $a, $b;",
            [(set f32:$dst, (fdiv_full f32:$a, f32:$b))]>,
            Requires<[doF32FTZ]>;
def FDIV32ri_ftz :
  NVPTXInst<(outs Float32Regs:$dst),
            (ins Float32Regs:$a, f32imm:$b),
            "div.full.ftz.f32 \t$dst, $a, $b;",
            [(set f32:$dst, (fdiv_full f32:$a, fpimm:$b))]>,
            Requires<[doF32FTZ]>;
def FDIV32rr :
  NVPTXInst<(outs Float32Regs:$dst),
            (ins Float32Regs:$a, Float32Regs:$b),
            "div.full.f32 \t$dst, $a, $b;",
            [(set f32:$dst, (fdiv_full f32:$a, f32:$b))]>;
def FDIV32ri :
  NVPTXInst<(outs Float32Regs:$dst),
            (ins Float32Regs:$a, f32imm:$b),
            "div.full.f32 \t$dst, $a, $b;",
            [(set f32:$dst, (fdiv_full f32:$a, fpimm:$b))]>;
//
// F32 Accurate reciprocal
//
def FRCP32r_prec_ftz :
  NVPTXInst<(outs Float32Regs:$dst),
            (ins Float32Regs:$b),
            "rcp.rn.ftz.f32 \t$dst, $b;",
            [(set f32:$dst, (fdiv f32imm_1, f32:$b))]>,
            Requires<[doF32FTZ]>;
def FRCP32r_prec :
  NVPTXInst<(outs Float32Regs:$dst),
            (ins Float32Regs:$b),
            "rcp.rn.f32 \t$dst, $b;",
            [(set f32:$dst, (fdiv f32imm_1, f32:$b))]>;
//
// F32 Accurate division
//
def FDIV32rr_prec_ftz :
  NVPTXInst<(outs Float32Regs:$dst),
            (ins Float32Regs:$a, Float32Regs:$b),
            "div.rn.ftz.f32 \t$dst, $a, $b;",
            [(set f32:$dst, (fdiv f32:$a, f32:$b))]>,
            Requires<[doF32FTZ]>;
def FDIV32ri_prec_ftz :
  NVPTXInst<(outs Float32Regs:$dst),
            (ins Float32Regs:$a, f32imm:$b),
            "div.rn.ftz.f32 \t$dst, $a, $b;",
            [(set f32:$dst, (fdiv f32:$a, fpimm:$b))]>,
            Requires<[doF32FTZ]>;
def FDIV32rr_prec :
  NVPTXInst<(outs Float32Regs:$dst),
            (ins Float32Regs:$a, Float32Regs:$b),
            "div.rn.f32 \t$dst, $a, $b;",
            [(set f32:$dst, (fdiv f32:$a, f32:$b))]>;
def FDIV32ri_prec :
  NVPTXInst<(outs Float32Regs:$dst),
            (ins Float32Regs:$a, f32imm:$b),
            "div.rn.f32 \t$dst, $a, $b;",
            [(set f32:$dst, (fdiv f32:$a, fpimm:$b))]>;

//
// FMA
//

multiclass FMA<string OpcStr, RegTyInfo t, list<Predicate> Preds = []> {
  defvar asmstr = OpcStr # " \t$dst, $a, $b, $c;";
  def rrr : NVPTXInst<(outs t.RC:$dst), (ins t.RC:$a, t.RC:$b, t.RC:$c),
                      asmstr,
                      [(set t.Ty:$dst, (fma t.Ty:$a, t.Ty:$b, t.Ty:$c))]>,
                      Requires<Preds>;

  if t.SupportsImm then {
    def rri : NVPTXInst<(outs t.RC:$dst),
                        (ins t.RC:$a, t.RC:$b, t.Imm:$c),
                        asmstr,
                        [(set t.Ty:$dst, (fma t.Ty:$a, t.Ty:$b, fpimm:$c))]>,
                        Requires<Preds>;
    def rir : NVPTXInst<(outs t.RC:$dst),
                        (ins t.RC:$a, t.Imm:$b, t.RC:$c),
                        asmstr,
                        [(set t.Ty:$dst, (fma t.Ty:$a, fpimm:$b, t.Ty:$c))]>,
                        Requires<Preds>;
    def rii : NVPTXInst<(outs t.RC:$dst),
                        (ins t.RC:$a, t.Imm:$b, t.Imm:$c),
                        asmstr,
                        [(set t.Ty:$dst, (fma t.Ty:$a, fpimm:$b, fpimm:$c))]>,
                        Requires<Preds>;
    def iir : NVPTXInst<(outs t.RC:$dst),
                        (ins t.Imm:$a, t.Imm:$b, t.RC:$c),
                        asmstr,
                        [(set t.Ty:$dst, (fma fpimm:$a, fpimm:$b, t.Ty:$c))]>,
                        Requires<Preds>;
  }
}

defm FMA16_ftz    : FMA<"fma.rn.ftz.f16", F16RT, [useFP16Math, doF32FTZ]>;
defm FMA16        : FMA<"fma.rn.f16", F16RT, [useFP16Math]>;
defm FMA16x2_ftz  : FMA<"fma.rn.ftz.f16x2", F16X2RT, [useFP16Math, doF32FTZ]>;
defm FMA16x2      : FMA<"fma.rn.f16x2", F16X2RT, [useFP16Math]>;
defm BFMA16       : FMA<"fma.rn.bf16", BF16RT, [hasBF16Math]>;
defm BFMA16x2     : FMA<"fma.rn.bf16x2", BF16X2RT, [hasBF16Math]>;
defm FMA32_ftz    : FMA<"fma.rn.ftz.f32", F32RT, [doF32FTZ]>;
defm FMA32        : FMA<"fma.rn.f32", F32RT>;
defm FMA64        : FMA<"fma.rn.f64", F64RT>;

// sin/cos

class UnaryOpAllowsApproxFn<SDPatternOperator operator>
    : PatFrag<(ops node:$A),
              (operator node:$A), [{
  return allowUnsafeFPMath() || N->getFlags().hasApproximateFuncs();
}]>;

def SINF:  NVPTXInst<(outs Float32Regs:$dst), (ins Float32Regs:$src),
                      "sin.approx.f32 \t$dst, $src;",
                      [(set f32:$dst, (UnaryOpAllowsApproxFn<fsin> f32:$src))]>;
def COSF:  NVPTXInst<(outs Float32Regs:$dst), (ins Float32Regs:$src),
                      "cos.approx.f32 \t$dst, $src;",
                      [(set f32:$dst, (UnaryOpAllowsApproxFn<fcos> f32:$src))]>;

//-----------------------------------
// Bitwise operations
//-----------------------------------

// Template for three-arg bitwise operations.  Takes three args, Creates .b16,
// .b32, .b64, and .pred (predicate registers -- i.e., i1) versions of OpcStr.
multiclass BITWISE<string OpcStr, SDNode OpNode> {
  defm b1 : I3Inst<OpcStr # ".pred", OpNode, I1RT, commutative = true>;
  defm b16 : I3Inst<OpcStr # ".b16", OpNode, I16RT, commutative = true>;
  defm b32 : I3Inst<OpcStr # ".b32", OpNode, I32RT, commutative = true>;
  defm b64 : I3Inst<OpcStr # ".b64", OpNode, I64RT, commutative = true>;
}

defm OR  : BITWISE<"or", or>;
defm AND : BITWISE<"and", and>;
defm XOR : BITWISE<"xor", xor>;

// PTX does not support mul on predicates, convert to and instructions
def : Pat<(mul i1:$a, i1:$b), (ANDb1rr $a, $b)>;
def : Pat<(mul i1:$a, imm:$b), (ANDb1ri $a, imm:$b)>;

// These transformations were once reliably performed by instcombine, but thanks
// to poison semantics they are no longer safe for LLVM IR, perform them here
// instead.
def : Pat<(select i1:$a, i1:$b, 0), (ANDb1rr $a, $b)>;
def : Pat<(select i1:$a, 1, i1:$b), (ORb1rr $a, $b)>;

// Lower logical v2i16/v4i8 ops as bitwise ops on b32.
foreach vt = [v2i16, v4i8] in {
  def: Pat<(or vt:$a, vt:$b),
           (ORb32rr $a, $b)>;
  def: Pat<(xor vt:$a, vt:$b),
           (XORb32rr $a, $b)>;
  def: Pat<(and vt:$a, vt:$b),
           (ANDb32rr $a, $b)>;

  // The constants get legalized into a bitcast from i32, so that's what we need
  // to match here.
  def: Pat<(or vt:$a, (vt (bitconvert (i32 imm:$b)))),
           (ORb32ri $a, imm:$b)>;
  def: Pat<(xor vt:$a, (vt (bitconvert (i32 imm:$b)))),
           (XORb32ri $a, imm:$b)>;
  def: Pat<(and vt:$a, (vt (bitconvert (i32 imm:$b)))),
           (ANDb32ri $a, imm:$b)>;
}

def NOT1  : NVPTXInst<(outs Int1Regs:$dst), (ins Int1Regs:$src),
                      "not.pred \t$dst, $src;",
                      [(set i1:$dst, (not i1:$src))]>;
def NOT16 : NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$src),
                      "not.b16 \t$dst, $src;",
                      [(set i16:$dst, (not i16:$src))]>;
def NOT32 : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$src),
                      "not.b32 \t$dst, $src;",
                      [(set i32:$dst, (not i32:$src))]>;
def NOT64 : NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$src),
                       "not.b64 \t$dst, $src;",
                       [(set i64:$dst, (not i64:$src))]>;

// Template for left/right shifts.  Takes three operands,
//   [dest (reg), src (reg), shift (reg or imm)].
// dest and src may be int64, int32, or int16, but shift is always int32.
//
// This template also defines a 32-bit shift (imm, imm) instruction.
multiclass SHIFT<string OpcStr, SDNode OpNode> {
   def i64rr :
     NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$a, Int32Regs:$b),
               !strconcat(OpcStr, "64 \t$dst, $a, $b;"),
               [(set i64:$dst, (OpNode i64:$a, i32:$b))]>;
   def i64ri :
     NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$a, i32imm:$b),
               !strconcat(OpcStr, "64 \t$dst, $a, $b;"),
               [(set i64:$dst, (OpNode i64:$a, (i32 imm:$b)))]>;
   def i32rr :
     NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a, Int32Regs:$b),
               !strconcat(OpcStr, "32 \t$dst, $a, $b;"),
               [(set i32:$dst, (OpNode i32:$a, i32:$b))]>;
   def i32ri :
     NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a, i32imm:$b),
               !strconcat(OpcStr, "32 \t$dst, $a, $b;"),
               [(set i32:$dst, (OpNode i32:$a, (i32 imm:$b)))]>;
   def i32ii :
     NVPTXInst<(outs Int32Regs:$dst), (ins i32imm:$a, i32imm:$b),
               !strconcat(OpcStr, "32 \t$dst, $a, $b;"),
               [(set i32:$dst, (OpNode (i32 imm:$a), (i32 imm:$b)))]>;
   def i16rr :
     NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$a, Int32Regs:$b),
               !strconcat(OpcStr, "16 \t$dst, $a, $b;"),
               [(set i16:$dst, (OpNode i16:$a, i32:$b))]>;
   def i16ri :
     NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$a, i32imm:$b),
               !strconcat(OpcStr, "16 \t$dst, $a, $b;"),
               [(set i16:$dst, (OpNode i16:$a, (i32 imm:$b)))]>;
}

defm SHL : SHIFT<"shl.b", shl>;
defm SRA : SHIFT<"shr.s", sra>;
defm SRL : SHIFT<"shr.u", srl>;

// Bit-reverse
def BREV32 :
  NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a),
             "brev.b32 \t$dst, $a;",
             [(set i32:$dst, (bitreverse i32:$a))]>;
def BREV64 :
  NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$a),
             "brev.b64 \t$dst, $a;",
             [(set i64:$dst, (bitreverse i64:$a))]>;


//
// BFE - bit-field extract
//

// Template for BFE/BFI instructions.
// Args: [dest (reg), src (reg), start (reg or imm), end (reg or imm)].
// Start may be an imm only if end is also an imm.  FIXME: Is this a
// restriction in PTX?
//
// dest and src may be int32 or int64, but start and end are always int32.
def SDTBFE :
  SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>, SDTCisInt<0>,
                       SDTCisVT<2, i32>, SDTCisVT<3, i32>]>;
def bfe : SDNode<"NVPTXISD::BFE", SDTBFE>;

def SDTBFI :
  SDTypeProfile<1, 4, [SDTCisInt<0>, SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, 
                       SDTCisVT<3, i32>, SDTCisVT<4, i32>]>;
def bfi : SDNode<"NVPTXISD::BFI", SDTBFI>;

def SDTPRMT :
  SDTypeProfile<1, 4, [SDTCisVT<0, i32>, SDTCisVT<1, i32>,
                       SDTCisVT<2, i32>, SDTCisVT<3, i32>, SDTCisVT<4, i32>,]>;
def prmt : SDNode<"NVPTXISD::PRMT", SDTPRMT>;

multiclass BFE<string Instr, ValueType T, RegisterClass RC> {
  def rrr
    : NVPTXInst<(outs RC:$d),
                (ins RC:$a, Int32Regs:$b, Int32Regs:$c),
                !strconcat(Instr, " \t$d, $a, $b, $c;"),
                [(set T:$d, (bfe T:$a, i32:$b, i32:$c))]>;
  def rri
    : NVPTXInst<(outs RC:$d),
                (ins RC:$a, Int32Regs:$b, i32imm:$c),
                !strconcat(Instr, " \t$d, $a, $b, $c;"),
                [(set T:$d, (bfe T:$a, i32:$b, imm:$c))]>;
  def rii
    : NVPTXInst<(outs RC:$d),
                (ins RC:$a, i32imm:$b, i32imm:$c),
                !strconcat(Instr, " \t$d, $a, $b, $c;"),
                [(set T:$d, (bfe T:$a, imm:$b, imm:$c))]>;
}

multiclass BFI<string Instr, ValueType T, RegisterClass RC, Operand ImmCls> {
  def rrrr
    : NVPTXInst<(outs RC:$f),
                (ins RC:$a, RC:$b, Int32Regs:$c, Int32Regs:$d),
                !strconcat(Instr, " \t$f, $a, $b, $c, $d;"),
                [(set T:$f, (bfi T:$a, T:$b, i32:$c, i32:$d))]>;
  def rrri
    : NVPTXInst<(outs RC:$f),
                (ins RC:$a, RC:$b, Int32Regs:$c, i32imm:$d),
                !strconcat(Instr, " \t$f, $a, $b, $c, $d;"),
                [(set T:$f, (bfi T:$a, T:$b, i32:$c, imm:$d))]>;
  def rrii
    : NVPTXInst<(outs RC:$f),
                (ins RC:$a, RC:$b, i32imm:$c, i32imm:$d),
                !strconcat(Instr, " \t$f, $a, $b, $c, $d;"),
                [(set T:$f, (bfi T:$a, T:$b, imm:$c, imm:$d))]>;
  def irrr
    : NVPTXInst<(outs RC:$f),
                (ins ImmCls:$a, RC:$b, Int32Regs:$c, Int32Regs:$d),
                !strconcat(Instr, " \t$f, $a, $b, $c, $d;"),
                [(set T:$f, (bfi (T imm:$a), T:$b, i32:$c, i32:$d))]>;
  def irri
    : NVPTXInst<(outs RC:$f),
                (ins ImmCls:$a, RC:$b, Int32Regs:$c, i32imm:$d),
                !strconcat(Instr, " \t$f, $a, $b, $c, $d;"),
                [(set T:$f, (bfi (T imm:$a), T:$b, i32:$c, imm:$d))]>;
  def irii
    : NVPTXInst<(outs RC:$f),
                (ins ImmCls:$a, RC:$b, i32imm:$c, i32imm:$d),
                !strconcat(Instr, " \t$f, $a, $b, $c, $d;"),
                [(set T:$f, (bfi (T imm:$a), T:$b, imm:$c, imm:$d))]>;
}

def Hexu32imm : Operand<i32> {
  let PrintMethod = "printHexu32imm";
}

multiclass PRMT<ValueType T, RegisterClass RC> {
  def rrr
    : NVPTXInst<(outs RC:$d),
                (ins RC:$a, Int32Regs:$b, Int32Regs:$c, PrmtMode:$mode),
                !strconcat("prmt.b32${mode}", " \t$d, $a, $b, $c;"),
                [(set T:$d, (prmt T:$a, T:$b, i32:$c, imm:$mode))]>;
  def rri
    : NVPTXInst<(outs RC:$d),
                (ins RC:$a, Int32Regs:$b, Hexu32imm:$c, PrmtMode:$mode),
                !strconcat("prmt.b32${mode}", " \t$d, $a, $b, $c;"),
                [(set T:$d, (prmt T:$a, T:$b, imm:$c, imm:$mode))]>;
  def rii
    : NVPTXInst<(outs RC:$d),
                (ins RC:$a, i32imm:$b, Hexu32imm:$c, PrmtMode:$mode),
                !strconcat("prmt.b32${mode}", " \t$d, $a, $b, $c;"),
                [(set T:$d, (prmt T:$a, imm:$b, imm:$c, imm:$mode))]>;
}

let hasSideEffects = false in {
  // order is somewhat important here. signed/unsigned variants match
  // the same patterns, so the first one wins. Having unsigned byte extraction
  // has the benefit of always having zero in unused bits, which makes some
  // optimizations easier (e.g. no need to mask them).
  defm BFE_U32 : BFE<"bfe.u32", i32, Int32Regs>;
  defm BFE_S32 : BFE<"bfe.s32", i32, Int32Regs>;
  defm BFE_U64 : BFE<"bfe.u64", i64, Int64Regs>;
  defm BFE_S64 : BFE<"bfe.s64", i64, Int64Regs>;

  defm BFI_B32 : BFI<"bfi.b32", i32, Int32Regs, i32imm>;
  defm BFI_B64 : BFI<"bfi.b64", i64, Int64Regs, i64imm>;

  defm PRMT_B32 : PRMT<i32, Int32Regs>;
}


// byte extraction + signed/unsigned extension to i32.
def : Pat<(i32 (sext_inreg (bfe i32:$s, i32:$o, 8), i8)),
          (BFE_S32rri $s, $o, 8)>;
def : Pat<(i32 (sext_inreg (bfe i32:$s, imm:$o, 8), i8)),
          (BFE_S32rii $s, imm:$o, 8)>;
def : Pat<(i32 (and (bfe i32:$s, i32:$o, 8), 255)),
          (BFE_U32rri $s, $o, 8)>;
def : Pat<(i32 (and (bfe i32:$s, imm:$o, 8), 255)),
          (BFE_U32rii $s, imm:$o, 8)>;

// byte extraction + signed extension to i16
def : Pat<(i16 (sext_inreg (trunc (bfe i32:$s, imm:$o, 8)), i8)),
          (CVT_s8_s32 (BFE_S32rii $s, imm:$o, 8), CvtNONE)>;


// Byte extraction via shift/trunc/sext
def : Pat<(i16 (sext_inreg (trunc i32:$s), i8)),
          (CVT_s8_s32 $s, CvtNONE)>;
def : Pat<(i16 (sext_inreg (trunc (srl i32:$s,  (i32 imm:$o))), i8)),
          (CVT_s8_s32 (BFE_S32rii $s, imm:$o, 8), CvtNONE)>;
def : Pat<(sext_inreg (srl i32:$s,  (i32 imm:$o)), i8),
          (BFE_S32rii $s, imm:$o, 8)>;
def : Pat<(i16 (sra (i16 (trunc i32:$s)), (i32 8))),
          (CVT_s8_s32 (BFE_S32rii $s, 8, 8), CvtNONE)>;
def : Pat<(sext_inreg (srl i64:$s,  (i32 imm:$o)), i8),
          (BFE_S64rii $s, imm:$o, 8)>;
def : Pat<(i16 (sext_inreg (trunc i64:$s), i8)),
          (CVT_s8_s64 $s, CvtNONE)>;
def : Pat<(i16 (sext_inreg (trunc (srl i64:$s,  (i32 imm:$o))), i8)),
          (CVT_s8_s64 (BFE_S64rii $s, imm:$o, 8), CvtNONE)>;

//-----------------------------------
// Comparison instructions (setp, set)
//-----------------------------------

// FIXME: This doesn't cover versions of set and setp that combine with a
// boolean predicate, e.g. setp.eq.and.b16.

let hasSideEffects = false in {
  multiclass SETP<string TypeStr, RegisterClass RC, Operand ImmCls> {
    def rr :
      NVPTXInst<(outs Int1Regs:$dst), (ins RC:$a, RC:$b, CmpMode:$cmp),
                !strconcat("setp${cmp:base}${cmp:ftz}.", TypeStr,
                           " \t$dst, $a, $b;"), []>;
    def ri :
      NVPTXInst<(outs Int1Regs:$dst), (ins RC:$a, ImmCls:$b, CmpMode:$cmp),
                !strconcat("setp${cmp:base}${cmp:ftz}.", TypeStr,
                           " \t$dst, $a, $b;"), []>;
    def ir :
      NVPTXInst<(outs Int1Regs:$dst), (ins ImmCls:$a, RC:$b, CmpMode:$cmp),
                !strconcat("setp${cmp:base}${cmp:ftz}.", TypeStr,
                           " \t$dst, $a, $b;"), []>;
  }
}

defm SETP_b16 : SETP<"b16", Int16Regs, i16imm>;
defm SETP_s16 : SETP<"s16", Int16Regs, i16imm>;
defm SETP_u16 : SETP<"u16", Int16Regs, i16imm>;
defm SETP_b32 : SETP<"b32", Int32Regs, i32imm>;
defm SETP_s32 : SETP<"s32", Int32Regs, i32imm>;
defm SETP_u32 : SETP<"u32", Int32Regs, i32imm>;
defm SETP_b64 : SETP<"b64", Int64Regs, i64imm>;
defm SETP_s64 : SETP<"s64", Int64Regs, i64imm>;
defm SETP_u64 : SETP<"u64", Int64Regs, i64imm>;
defm SETP_f32 : SETP<"f32", Float32Regs, f32imm>;
defm SETP_f64 : SETP<"f64", Float64Regs, f64imm>;
def SETP_f16rr :
      NVPTXInst<(outs Int1Regs:$dst),
                (ins Int16Regs:$a, Int16Regs:$b, CmpMode:$cmp),
                "setp${cmp:base}${cmp:ftz}.f16 \t$dst, $a, $b;",
                []>, Requires<[useFP16Math]>;

def SETP_f16x2rr :
      NVPTXInst<(outs Int1Regs:$p, Int1Regs:$q),
                (ins Int32Regs:$a, Int32Regs:$b, CmpMode:$cmp),
                "setp${cmp:base}${cmp:ftz}.f16x2 \t$p|$q, $a, $b;",
                []>,
                Requires<[useFP16Math]>;
def SETP_bf16rr :
      NVPTXInst<(outs Int1Regs:$dst),
                (ins Int16Regs:$a, Int16Regs:$b, CmpMode:$cmp),
                "setp${cmp:base}${cmp:ftz}.bf16 \t$dst, $a, $b;",
                []>, Requires<[hasBF16Math, hasPTX<78>, hasSM<90>]>;

def SETP_bf16x2rr :
      NVPTXInst<(outs Int1Regs:$p, Int1Regs:$q),
                (ins Int32Regs:$a, Int32Regs:$b, CmpMode:$cmp),
                "setp${cmp:base}${cmp:ftz}.bf16x2 \t$p|$q, $a, $b;",
                []>,
                Requires<[hasBF16Math, hasPTX<78>, hasSM<90>]>;

//-----------------------------------
// Data Movement (Load / Store, Move)
//-----------------------------------

def addr : ComplexPattern<pAny, 2, "SelectADDR">;

def ADDR_base : Operand<pAny> {
  let PrintMethod = "printOperand";
}

def ADDR : Operand<pAny> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops ADDR_base, i32imm);
}

def LdStCode : Operand<i32> {
  let PrintMethod = "printLdStCode";
}

def MmaCode : Operand<i32> {
  let PrintMethod = "printMmaCode";
}

def Offseti32imm : Operand<i32> {
  let PrintMethod = "printOffseti32imm";
}

def SDTWrapper : SDTypeProfile<1, 1, [SDTCisSameAs<0, 1>, SDTCisPtrTy<0>]>;
def Wrapper    : SDNode<"NVPTXISD::Wrapper", SDTWrapper>;

// Load a memory address into a u32 or u64 register.
def MOV_ADDR : NVPTXInst<(outs Int32Regs:$dst), (ins ADDR_base:$a),
                         "mov.b32 \t$dst, $a;",
                         [(set i32:$dst, (Wrapper tglobaladdr:$a))]>;
def MOV_ADDR64 : NVPTXInst<(outs Int64Regs:$dst), (ins ADDR_base:$a),
                           "mov.b64 \t$dst, $a;",
                           [(set i64:$dst, (Wrapper tglobaladdr:$a))]>;

// Get pointer to local stack.
let hasSideEffects = false in {
  def MOV_DEPOT_ADDR :    NVPTXInst<(outs Int32Regs:$d), (ins i32imm:$num),
                                     "mov.b32 \t$d, __local_depot$num;", []>;
  def MOV_DEPOT_ADDR_64 : NVPTXInst<(outs Int64Regs:$d), (ins i32imm:$num),
                                    "mov.b64 \t$d, __local_depot$num;", []>;
}


// copyPhysreg is hard-coded in NVPTXInstrInfo.cpp
let hasSideEffects = false, isAsCheapAsAMove = true in {
  // Class for register-to-register moves
  class MOVr<RegisterClass RC, string OpStr> :
    NVPTXInst<(outs RC:$dst), (ins RC:$src),
             "mov." # OpStr # " \t$dst, $src;", []>;
  
  // Class for immediate-to-register moves
  class MOVi<RegisterClass RC, string OpStr, ValueType VT, Operand IMMType, SDNode ImmNode> :
    NVPTXInst<(outs RC:$dst), (ins IMMType:$src),
             "mov." # OpStr # " \t$dst, $src;",
             [(set VT:$dst, ImmNode:$src)]>;
}

def IMOV1r : MOVr<Int1Regs, "pred">;
def IMOV1i : MOVi<Int1Regs, "pred", i1, i1imm, imm>;
def MOV16r : MOVr<Int16Regs, "b16">;
def IMOV16i : MOVi<Int16Regs, "b16", i16, i16imm, imm>;
def IMOV32r : MOVr<Int32Regs, "b32">;
def IMOV32i : MOVi<Int32Regs, "b32", i32, i32imm, imm>;
def IMOV64r : MOVr<Int64Regs, "b64">;
def IMOV64i : MOVi<Int64Regs, "b64", i64, i64imm, imm>;
def IMOV128r : MOVr<Int128Regs, "b128">;
def FMOV16i : MOVi<Int16Regs, "b16", f16, f16imm, fpimm>;
def BFMOV16i : MOVi<Int16Regs, "b16", bf16, bf16imm, fpimm>;
def FMOV32r : MOVr<Float32Regs, "b32">;
def FMOV32i : MOVi<Float32Regs, "b32", f32, f32imm, fpimm>;
def FMOV64r : MOVr<Float64Regs, "b64">;
def FMOV64i : MOVi<Float64Regs, "b64", f64, f64imm, fpimm>;

def : Pat<(i32 (Wrapper texternalsym:$dst)), (IMOV32i texternalsym:$dst)>;
def : Pat<(i64 (Wrapper texternalsym:$dst)), (IMOV64i texternalsym:$dst)>;

//---- Copy Frame Index ----
def LEA_ADDRi :   NVPTXInst<(outs Int32Regs:$dst), (ins ADDR:$addr),
                            "add.u32 \t$dst, ${addr:add};", []>;
def LEA_ADDRi64 : NVPTXInst<(outs Int64Regs:$dst), (ins ADDR:$addr),
                            "add.u64 \t$dst, ${addr:add};", []>;

def to_tframeindex : SDNodeXForm<frameindex, [{
  return CurDAG->getTargetFrameIndex(N->getIndex(), N->getValueType(0));
}]>;

def : Pat<(i32 frameindex:$fi), (LEA_ADDRi (to_tframeindex $fi), 0)>;
def : Pat<(i64 frameindex:$fi), (LEA_ADDRi64 (to_tframeindex $fi), 0)>;

//-----------------------------------
// Comparison and Selection
//-----------------------------------

multiclass ISET_FORMAT<PatFrag OpNode, PatLeaf Mode,
                       Instruction setp_16rr,
                       Instruction setp_16ri,
                       Instruction setp_16ir,
                       Instruction setp_32rr,
                       Instruction setp_32ri,
                       Instruction setp_32ir,
                       Instruction setp_64rr,
                       Instruction setp_64ri,
                       Instruction setp_64ir> {
  // i16 -> pred
  def : Pat<(i1 (OpNode i16:$a, i16:$b)),
            (setp_16rr $a, $b, Mode)>;
  def : Pat<(i1 (OpNode i16:$a, imm:$b)),
            (setp_16ri $a, imm:$b, Mode)>;
  def : Pat<(i1 (OpNode imm:$a, i16:$b)),
            (setp_16ir imm:$a, $b, Mode)>;
  // i32 -> pred
  def : Pat<(i1 (OpNode i32:$a, i32:$b)),
            (setp_32rr $a, $b, Mode)>;
  def : Pat<(i1 (OpNode i32:$a, imm:$b)),
            (setp_32ri $a, imm:$b, Mode)>;
  def : Pat<(i1 (OpNode imm:$a, i32:$b)),
            (setp_32ir imm:$a, $b, Mode)>;
  // i64 -> pred
  def : Pat<(i1 (OpNode i64:$a, i64:$b)),
            (setp_64rr $a, $b, Mode)>;
  def : Pat<(i1 (OpNode i64:$a, imm:$b)),
            (setp_64ri $a, imm:$b, Mode)>;
  def : Pat<(i1 (OpNode imm:$a, i64:$b)),
            (setp_64ir imm:$a, $b, Mode)>;
}

multiclass ISET_FORMAT_SIGNED<PatFrag OpNode, PatLeaf Mode>
  : ISET_FORMAT<OpNode, Mode,
                SETP_s16rr, SETP_s16ri, SETP_s16ir,
                SETP_s32rr, SETP_s32ri, SETP_s32ir,
                SETP_s64rr, SETP_s64ri, SETP_s64ir> {
  // TableGen doesn't like empty multiclasses.
  def : PatLeaf<(i32 0)>;
}

multiclass ISET_FORMAT_UNSIGNED<PatFrag OpNode, PatLeaf Mode>
  : ISET_FORMAT<OpNode, Mode,
                SETP_u16rr, SETP_u16ri, SETP_u16ir,
                SETP_u32rr, SETP_u32ri, SETP_u32ir,
                SETP_u64rr, SETP_u64ri, SETP_u64ir> {
  // TableGen doesn't like empty multiclasses.
  def : PatLeaf<(i32 0)>;
}

defm : ISET_FORMAT_SIGNED<setgt, CmpGT>;
defm : ISET_FORMAT_SIGNED<setlt, CmpLT>;
defm : ISET_FORMAT_SIGNED<setge, CmpGE>;
defm : ISET_FORMAT_SIGNED<setle, CmpLE>;
defm : ISET_FORMAT_SIGNED<seteq, CmpEQ>;
defm : ISET_FORMAT_SIGNED<setne, CmpNE>;
defm : ISET_FORMAT_UNSIGNED<setugt, CmpGT>;
defm : ISET_FORMAT_UNSIGNED<setult, CmpLT>;
defm : ISET_FORMAT_UNSIGNED<setuge, CmpGE>;
defm : ISET_FORMAT_UNSIGNED<setule, CmpLE>;
defm : ISET_FORMAT_UNSIGNED<setueq, CmpEQ>;
defm : ISET_FORMAT_UNSIGNED<setune, CmpNE>;

// comparisons of i8 extracted with BFE as i32
// It's faster to do comparison directly on i32 extracted by BFE,
// instead of the long conversion and sign extending.
def: Pat<(setgt (i16 (sext_inreg (i16 (trunc (bfe Int32Regs:$a, Int32Regs:$oa, 8))), i8)),
                (i16 (sext_inreg (i16 (trunc (bfe Int32Regs:$b, Int32Regs:$ob, 8))), i8))),
         (SETP_s32rr (BFE_S32rri $a, $oa, 8), (BFE_S32rri $b, $ob, 8), CmpGT)>;
def: Pat<(setgt (i16 (sext_inreg (trunc (bfe Int32Regs:$a, imm:$oa, 8)), i8)),
                (i16 (sext_inreg (trunc (bfe Int32Regs:$b, imm:$ob, 8)), i8))),
         (SETP_s32rr (BFE_S32rii $a, imm:$oa, 8), (BFE_S32rii $b, imm:$ob, 8), CmpGT)>;
def: Pat<(setge (i16 (sext_inreg (i16 (trunc (bfe Int32Regs:$a, Int32Regs:$oa, 8))), i8)),
                (i16 (sext_inreg (i16 (trunc (bfe Int32Regs:$b, Int32Regs:$ob, 8))), i8))),
         (SETP_s32rr (BFE_S32rri $a, $oa, 8), (BFE_S32rri $b, $ob, 8), CmpGE)>;
def: Pat<(setge (i16 (sext_inreg (trunc (bfe Int32Regs:$a, imm:$oa, 8)), i8)),
                (i16 (sext_inreg (trunc (bfe Int32Regs:$b, imm:$ob, 8)), i8))),
         (SETP_s32rr (BFE_S32rii $a, imm:$oa, 8), (BFE_S32rii $b, imm:$ob, 8), CmpGE)>;
def: Pat<(setlt (i16 (sext_inreg (i16 (trunc (bfe Int32Regs:$a, Int32Regs:$oa, 8))), i8)),
                (i16 (sext_inreg (i16 (trunc (bfe Int32Regs:$b, Int32Regs:$ob, 8))), i8))),
         (SETP_s32rr (BFE_S32rri $a, $oa, 8), (BFE_S32rri $b, $ob, 8), CmpLT)>;
def: Pat<(setlt (i16 (sext_inreg (trunc (bfe Int32Regs:$a, imm:$oa, 8)), i8)),
                (i16 (sext_inreg (trunc (bfe Int32Regs:$b, imm:$ob, 8)), i8))),
         (SETP_s32rr (BFE_S32rii $a, imm:$oa, 8), (BFE_S32rii $b, imm:$ob, 8), CmpLT)>;
def: Pat<(setle (i16 (sext_inreg (i16 (trunc (bfe Int32Regs:$a, Int32Regs:$oa, 8))), i8)),
                (i16 (sext_inreg (i16 (trunc (bfe Int32Regs:$b, Int32Regs:$ob, 8))), i8))),
         (SETP_s32rr (BFE_S32rri $a, $oa, 8), (BFE_S32rri $b, $ob, 8), CmpLE)>;
def: Pat<(setle (i16 (sext_inreg (trunc (bfe Int32Regs:$a, imm:$oa, 8)), i8)),
                (i16 (sext_inreg (trunc (bfe Int32Regs:$b, imm:$ob, 8)), i8))),
         (SETP_s32rr (BFE_S32rii $a, imm:$oa, 8), (BFE_S32rii $b, imm:$ob, 8), CmpLE)>;

def: Pat<(setugt (i16 (and (trunc (bfe Int32Regs:$a, Int32Regs:$oa, 8)), 255)),
                 (i16 (and (trunc (bfe Int32Regs:$b, Int32Regs:$ob, 8)), 255))),
         (SETP_u32rr (BFE_U32rri $a, $oa, 8), (BFE_U32rri $b, $ob, 8), CmpHI)>;
def: Pat<(setugt (i16 (and (trunc (bfe Int32Regs:$a, imm:$oa, 8)), 255)),
                 (i16 (and (trunc (bfe Int32Regs:$b, imm:$ob, 8)), 255))),
         (SETP_u32rr (BFE_U32rii $a, imm:$oa, 8), (BFE_U32rii $b, imm:$ob, 8), CmpHI)>;
def: Pat<(setuge (i16 (and (trunc (bfe Int32Regs:$a, Int32Regs:$oa, 8)), 255)),
                 (i16 (and (trunc (bfe Int32Regs:$b, Int32Regs:$ob, 8)), 255))),
         (SETP_u32rr (BFE_U32rri $a, $oa, 8), (BFE_U32rri $b, $ob, 8), CmpHS)>;
def: Pat<(setuge (i16 (and (trunc (bfe Int32Regs:$a, imm:$oa, 8)), 255)),
                 (i16 (and (trunc (bfe Int32Regs:$b, imm:$ob, 8)), 255))),
         (SETP_u32rr (BFE_U32rii $a, imm:$oa, 8), (BFE_U32rii $b, imm:$ob, 8), CmpHS)>;
def: Pat<(setult (i16 (and (trunc (bfe Int32Regs:$a, Int32Regs:$oa, 8)), 255)),
                 (i16 (and (trunc (bfe Int32Regs:$b, Int32Regs:$ob, 8)), 255))),
         (SETP_u32rr (BFE_U32rri $a, $oa, 8), (BFE_U32rri $b, $ob, 8), CmpLO)>;
def: Pat<(setult (i16 (and (trunc (bfe Int32Regs:$a, imm:$oa, 8)), 255)),
                 (i16 (and (trunc (bfe Int32Regs:$b, imm:$ob, 8)), 255))),
         (SETP_u32rr (BFE_U32rii $a, imm:$oa, 8), (BFE_U32rii $b, imm:$ob, 8), CmpLO)>;
def: Pat<(setule (i16 (and (trunc (bfe Int32Regs:$a, Int32Regs:$oa, 8)), 255)),
                 (i16 (and (trunc (bfe Int32Regs:$b, Int32Regs:$ob, 8)), 255))),
         (SETP_u32rr (BFE_U32rri $a, $oa, 8), (BFE_U32rri $b, $ob, 8), CmpLS)>;
def: Pat<(setule (i16 (and (trunc (bfe Int32Regs:$a, imm:$oa, 8)), 255)),
                 (i16 (and (trunc (bfe Int32Regs:$b, imm:$ob, 8)), 255))),
         (SETP_u32rr (BFE_U32rii $a, imm:$oa, 8), (BFE_U32rii $b, imm:$ob, 8), CmpLS)>;
def: Pat<(seteq (i16 (and (trunc (bfe Int32Regs:$a, Int32Regs:$oa, 8)), 255)),
                 (i16 (and (trunc (bfe Int32Regs:$b, Int32Regs:$ob, 8)), 255))),
         (SETP_u32rr (BFE_U32rri $a, $oa, 8), (BFE_U32rri $b, $ob, 8), CmpEQ)>;
def: Pat<(seteq (i16 (and (trunc (bfe Int32Regs:$a, imm:$oa, 8)), 255)),
                 (i16 (and (trunc (bfe Int32Regs:$b, imm:$ob, 8)), 255))),
         (SETP_u32rr (BFE_U32rii $a, imm:$oa, 8), (BFE_U32rii $b, imm:$ob, 8), CmpEQ)>;
def: Pat<(setne (i16 (and (trunc (bfe Int32Regs:$a, Int32Regs:$oa, 8)), 255)),
                 (i16 (and (trunc (bfe Int32Regs:$b, Int32Regs:$ob, 8)), 255))),
         (SETP_u32rr (BFE_U32rri $a, $oa, 8), (BFE_U32rri $b, $ob, 8), CmpNE)>;
def: Pat<(setne (i16 (and (trunc (bfe Int32Regs:$a, imm:$oa, 8)), 255)),
                 (i16 (and (trunc (bfe Int32Regs:$b, imm:$ob, 8)), 255))),
         (SETP_u32rr (BFE_U32rii $a, imm:$oa, 8), (BFE_U32rii $b, imm:$ob, 8), CmpNE)>;

// i1 compare -> i32
def : Pat<(i32 (setne i1:$a, i1:$b)),
          (SELP_b32ii -1, 0, (XORb1rr $a, $b))>;
def : Pat<(i32 (setne i1:$a, i1:$b)),
          (SELP_b32ii 0, -1, (XORb1rr $a, $b))>;



multiclass FSET_FORMAT<PatFrag OpNode, PatLeaf Mode, PatLeaf ModeFTZ> {
  // f16 -> pred
  def : Pat<(i1 (OpNode f16:$a, f16:$b)),
            (SETP_f16rr $a, $b, ModeFTZ)>,
        Requires<[useFP16Math,doF32FTZ]>;
  def : Pat<(i1 (OpNode f16:$a, f16:$b)),
            (SETP_f16rr $a, $b, Mode)>,
        Requires<[useFP16Math]>;

  // bf16 -> pred
  def : Pat<(i1 (OpNode bf16:$a, bf16:$b)),
            (SETP_bf16rr $a, $b, ModeFTZ)>,
        Requires<[hasBF16Math,doF32FTZ]>;
  def : Pat<(i1 (OpNode bf16:$a, bf16:$b)),
            (SETP_bf16rr $a, $b, Mode)>,
        Requires<[hasBF16Math]>;

  // f32 -> pred
  def : Pat<(i1 (OpNode f32:$a, f32:$b)),
            (SETP_f32rr $a, $b, ModeFTZ)>,
        Requires<[doF32FTZ]>;
  def : Pat<(i1 (OpNode f32:$a, f32:$b)),
            (SETP_f32rr $a, $b, Mode)>;
  def : Pat<(i1 (OpNode f32:$a, fpimm:$b)),
            (SETP_f32ri $a, fpimm:$b, ModeFTZ)>,
        Requires<[doF32FTZ]>;
  def : Pat<(i1 (OpNode f32:$a, fpimm:$b)),
            (SETP_f32ri $a, fpimm:$b, Mode)>;
  def : Pat<(i1 (OpNode fpimm:$a, f32:$b)),
            (SETP_f32ir fpimm:$a, $b, ModeFTZ)>,
        Requires<[doF32FTZ]>;
  def : Pat<(i1 (OpNode fpimm:$a, f32:$b)),
            (SETP_f32ir fpimm:$a, $b, Mode)>;

  // f64 -> pred
  def : Pat<(i1 (OpNode f64:$a, f64:$b)),
            (SETP_f64rr $a, $b, Mode)>;
  def : Pat<(i1 (OpNode f64:$a, fpimm:$b)),
            (SETP_f64ri $a, fpimm:$b, Mode)>;
  def : Pat<(i1 (OpNode fpimm:$a, f64:$b)),
            (SETP_f64ir fpimm:$a, $b, Mode)>;
}

defm FSetOGT : FSET_FORMAT<setogt, CmpGT, CmpGT_FTZ>;
defm FSetOLT : FSET_FORMAT<setolt, CmpLT, CmpLT_FTZ>;
defm FSetOGE : FSET_FORMAT<setoge, CmpGE, CmpGE_FTZ>;
defm FSetOLE : FSET_FORMAT<setole, CmpLE, CmpLE_FTZ>;
defm FSetOEQ : FSET_FORMAT<setoeq, CmpEQ, CmpEQ_FTZ>;
defm FSetONE : FSET_FORMAT<setone, CmpNE, CmpNE_FTZ>;

defm FSetUGT : FSET_FORMAT<setugt, CmpGTU, CmpGTU_FTZ>;
defm FSetULT : FSET_FORMAT<setult, CmpLTU, CmpLTU_FTZ>;
defm FSetUGE : FSET_FORMAT<setuge, CmpGEU, CmpGEU_FTZ>;
defm FSetULE : FSET_FORMAT<setule, CmpLEU, CmpLEU_FTZ>;
defm FSetUEQ : FSET_FORMAT<setueq, CmpEQU, CmpEQU_FTZ>;
defm FSetUNE : FSET_FORMAT<setune, CmpNEU, CmpNEU_FTZ>;

defm FSetGT : FSET_FORMAT<setgt, CmpGT, CmpGT_FTZ>;
defm FSetLT : FSET_FORMAT<setlt, CmpLT, CmpLT_FTZ>;
defm FSetGE : FSET_FORMAT<setge, CmpGE, CmpGE_FTZ>;
defm FSetLE : FSET_FORMAT<setle, CmpLE, CmpLE_FTZ>;
defm FSetEQ : FSET_FORMAT<seteq, CmpEQ, CmpEQ_FTZ>;
defm FSetNE : FSET_FORMAT<setne, CmpNE, CmpNE_FTZ>;

defm FSetNUM : FSET_FORMAT<seto, CmpNUM, CmpNUM_FTZ>;
defm FSetNAN : FSET_FORMAT<setuo, CmpNAN, CmpNAN_FTZ>;

def SDTDeclareParamProfile :
  SDTypeProfile<0, 3, [SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>]>;
def SDTDeclareScalarParamProfile :
  SDTypeProfile<0, 3, [SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>]>;
def SDTLoadParamProfile : SDTypeProfile<1, 2, [SDTCisInt<1>, SDTCisInt<2>]>;
def SDTLoadParamV2Profile : SDTypeProfile<2, 2, [SDTCisSameAs<0, 1>, SDTCisInt<2>, SDTCisInt<3>]>;
def SDTLoadParamV4Profile : SDTypeProfile<4, 2, [SDTCisInt<4>, SDTCisInt<5>]>;
def SDTPrintCallProfile : SDTypeProfile<0, 1, [SDTCisInt<0>]>;
def SDTPrintCallUniProfile : SDTypeProfile<0, 1, [SDTCisInt<0>]>;
def SDTStoreParamProfile : SDTypeProfile<0, 3, [SDTCisInt<0>, SDTCisInt<1>]>;
def SDTStoreParamV2Profile : SDTypeProfile<0, 4, [SDTCisInt<0>, SDTCisInt<1>]>;
def SDTStoreParamV4Profile : SDTypeProfile<0, 6, [SDTCisInt<0>, SDTCisInt<1>]>;
def SDTStoreParam32Profile : SDTypeProfile<0, 3, [SDTCisInt<0>, SDTCisInt<1>]>;
def SDTCallArgProfile : SDTypeProfile<0, 2, [SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;
def SDTCallArgMarkProfile : SDTypeProfile<0, 0, []>;
def SDTCallVoidProfile : SDTypeProfile<0, 1, []>;
def SDTCallValProfile : SDTypeProfile<1, 0, []>;
def SDTMoveParamProfile : SDTypeProfile<1, 1, [SDTCisInt<0>, SDTCisInt<1>]>;
def SDTStoreRetvalProfile : SDTypeProfile<0, 2, [SDTCisInt<0>]>;
def SDTStoreRetvalV2Profile : SDTypeProfile<0, 3, [SDTCisInt<0>]>;
def SDTStoreRetvalV4Profile : SDTypeProfile<0, 5, [SDTCisInt<0>]>;
def SDTPseudoUseParamProfile : SDTypeProfile<0, 1, []>;
def SDTProxyRegProfile : SDTypeProfile<1, 1, []>;

def DeclareParam :
  SDNode<"NVPTXISD::DeclareParam", SDTDeclareParamProfile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def DeclareScalarParam :
  SDNode<"NVPTXISD::DeclareScalarParam", SDTDeclareScalarParamProfile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def DeclareRetParam :
  SDNode<"NVPTXISD::DeclareRetParam", SDTDeclareParamProfile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def DeclareRet :
  SDNode<"NVPTXISD::DeclareRet", SDTDeclareScalarParamProfile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def LoadParam :
  SDNode<"NVPTXISD::LoadParam", SDTLoadParamProfile,
         [SDNPHasChain, SDNPMayLoad, SDNPOutGlue, SDNPInGlue]>;
def LoadParamV2 :
  SDNode<"NVPTXISD::LoadParamV2", SDTLoadParamV2Profile,
         [SDNPHasChain, SDNPMayLoad, SDNPOutGlue, SDNPInGlue]>;
def LoadParamV4 :
  SDNode<"NVPTXISD::LoadParamV4", SDTLoadParamV4Profile,
         [SDNPHasChain, SDNPMayLoad, SDNPOutGlue, SDNPInGlue]>;
def PrintCall :
  SDNode<"NVPTXISD::PrintCall", SDTPrintCallProfile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def PrintConvergentCall :
  SDNode<"NVPTXISD::PrintConvergentCall", SDTPrintCallProfile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def PrintCallUni :
  SDNode<"NVPTXISD::PrintCallUni", SDTPrintCallUniProfile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def PrintConvergentCallUni :
  SDNode<"NVPTXISD::PrintConvergentCallUni", SDTPrintCallUniProfile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def StoreParam :
  SDNode<"NVPTXISD::StoreParam", SDTStoreParamProfile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def StoreParamV2 :
  SDNode<"NVPTXISD::StoreParamV2", SDTStoreParamV2Profile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def StoreParamV4 :
  SDNode<"NVPTXISD::StoreParamV4", SDTStoreParamV4Profile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def StoreParamU32 :
  SDNode<"NVPTXISD::StoreParamU32", SDTStoreParam32Profile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def StoreParamS32 :
  SDNode<"NVPTXISD::StoreParamS32", SDTStoreParam32Profile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def CallArgBegin :
  SDNode<"NVPTXISD::CallArgBegin", SDTCallArgMarkProfile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def CallArg :
  SDNode<"NVPTXISD::CallArg", SDTCallArgProfile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def LastCallArg :
  SDNode<"NVPTXISD::LastCallArg", SDTCallArgProfile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def CallArgEnd :
  SDNode<"NVPTXISD::CallArgEnd", SDTCallVoidProfile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def CallVoid :
  SDNode<"NVPTXISD::CallVoid", SDTCallVoidProfile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def Prototype :
  SDNode<"NVPTXISD::Prototype", SDTCallVoidProfile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def CallVal :
  SDNode<"NVPTXISD::CallVal", SDTCallValProfile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def MoveParam :
  SDNode<"NVPTXISD::MoveParam", SDTMoveParamProfile, []>;
def StoreRetval :
  SDNode<"NVPTXISD::StoreRetval", SDTStoreRetvalProfile,
         [SDNPHasChain, SDNPSideEffect]>;
def StoreRetvalV2 :
  SDNode<"NVPTXISD::StoreRetvalV2", SDTStoreRetvalV2Profile,
         [SDNPHasChain, SDNPSideEffect]>;
def StoreRetvalV4 :
  SDNode<"NVPTXISD::StoreRetvalV4", SDTStoreRetvalV4Profile,
         [SDNPHasChain, SDNPSideEffect]>;
def PseudoUseParam :
  SDNode<"NVPTXISD::PseudoUseParam", SDTPseudoUseParamProfile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def RETURNNode :
  SDNode<"NVPTXISD::RETURN", SDTCallArgMarkProfile,
         [SDNPHasChain, SDNPSideEffect]>;
def ProxyReg :
  SDNode<"NVPTXISD::ProxyReg", SDTProxyRegProfile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;

let mayLoad = true in {
  class LoadParamMemInst<NVPTXRegClass regclass, string opstr> :
        NVPTXInst<(outs regclass:$dst), (ins Offseti32imm:$b),
                  !strconcat("ld.param", opstr, " \t$dst, [retval0$b];"),
                  []>;

  class LoadParamV2MemInst<NVPTXRegClass regclass, string opstr> :
        NVPTXInst<(outs regclass:$dst, regclass:$dst2), (ins Offseti32imm:$b),
                  !strconcat("ld.param.v2", opstr,
                             " \t{{$dst, $dst2}}, [retval0$b];"), []>;

  class LoadParamV4MemInst<NVPTXRegClass regclass, string opstr> :
        NVPTXInst<(outs regclass:$dst, regclass:$dst2, regclass:$dst3,
                        regclass:$dst4),
                  (ins Offseti32imm:$b),
                  !strconcat("ld.param.v4", opstr,
                             " \t{{$dst, $dst2, $dst3, $dst4}}, [retval0$b];"),
                  []>;
}

class LoadParamRegInst<NVPTXRegClass regclass, string opstr> :
      NVPTXInst<(outs regclass:$dst), (ins i32imm:$b),
                !strconcat("mov", opstr, " \t$dst, retval$b;"),
                [(set regclass:$dst, (LoadParam (i32 0), (i32 imm:$b)))]>;

let mayStore = true in {

  multiclass StoreParamInst<NVPTXRegClass regclass, Operand IMMType, string opstr, bit support_imm = true> {
    foreach op = [IMMType, regclass] in
      if !or(support_imm, !isa<NVPTXRegClass>(op)) then
        def _ # !if(!isa<NVPTXRegClass>(op), "r", "i")
          : NVPTXInst<(outs),
                      (ins op:$val, i32imm:$a, Offseti32imm:$b),
                      "st.param" # opstr # " \t[param$a$b], $val;",
                      []>;
  }

  multiclass StoreParamV2Inst<NVPTXRegClass regclass, Operand IMMType, string opstr> {
    foreach op1 = [IMMType, regclass] in
      foreach op2 = [IMMType, regclass] in
        def _ # !if(!isa<NVPTXRegClass>(op1), "r", "i")
              # !if(!isa<NVPTXRegClass>(op2), "r", "i")
          : NVPTXInst<(outs),
                      (ins op1:$val1, op2:$val2,
                           i32imm:$a, Offseti32imm:$b),
                      "st.param.v2" # opstr # " \t[param$a$b], {{$val1, $val2}};",
                      []>;
  }

  multiclass StoreParamV4Inst<NVPTXRegClass regclass, Operand IMMType, string opstr> {
    foreach op1 = [IMMType, regclass] in
      foreach op2 = [IMMType, regclass] in
        foreach op3 = [IMMType, regclass] in
          foreach op4 = [IMMType, regclass] in
            def _ # !if(!isa<NVPTXRegClass>(op1), "r", "i")
                  # !if(!isa<NVPTXRegClass>(op2), "r", "i")
                  # !if(!isa<NVPTXRegClass>(op3), "r", "i")
                  # !if(!isa<NVPTXRegClass>(op4), "r", "i")

              : NVPTXInst<(outs),
                          (ins op1:$val1, op2:$val2, op3:$val3, op4:$val4,
                               i32imm:$a, Offseti32imm:$b),
                          "st.param.v4" # opstr #
                          " \t[param$a$b], {{$val1, $val2, $val3, $val4}};",
                          []>;
  }

  class StoreRetvalInst<NVPTXRegClass regclass, string opstr> :
        NVPTXInst<(outs), (ins regclass:$val, Offseti32imm:$a),
                  !strconcat("st.param", opstr, " \t[func_retval0$a], $val;"),
                  []>;

  class StoreRetvalV2Inst<NVPTXRegClass regclass, string opstr> :
        NVPTXInst<(outs), (ins regclass:$val, regclass:$val2, Offseti32imm:$a),
                  !strconcat("st.param.v2", opstr,
                             " \t[func_retval0$a], {{$val, $val2}};"),
                  []>;

  class StoreRetvalV4Inst<NVPTXRegClass regclass, string opstr> :
        NVPTXInst<(outs),
                  (ins regclass:$val, regclass:$val2, regclass:$val3,
                       regclass:$val4, Offseti32imm:$a),
                  !strconcat("st.param.v4", opstr,
                             " \t[func_retval0$a], {{$val, $val2, $val3, $val4}};"),
                  []>;
}

let isCall=1 in {
  multiclass CALL<string OpcStr, SDNode OpNode> {
     def PrintCallNoRetInst : NVPTXInst<(outs), (ins),
       !strconcat(OpcStr, " "), [(OpNode (i32 0))]>;
     def PrintCallRetInst1 : NVPTXInst<(outs), (ins),
       !strconcat(OpcStr, " (retval0), "), [(OpNode (i32 1))]>;
     def PrintCallRetInst2 : NVPTXInst<(outs), (ins),
       !strconcat(OpcStr, " (retval0, retval1), "), [(OpNode (i32 2))]>;
     def PrintCallRetInst3 : NVPTXInst<(outs), (ins),
       !strconcat(OpcStr, " (retval0, retval1, retval2), "), [(OpNode (i32 3))]>;
     def PrintCallRetInst4 : NVPTXInst<(outs), (ins),
       !strconcat(OpcStr, " (retval0, retval1, retval2, retval3), "),
       [(OpNode (i32 4))]>;
     def PrintCallRetInst5 : NVPTXInst<(outs), (ins),
       !strconcat(OpcStr, " (retval0, retval1, retval2, retval3, retval4), "),
       [(OpNode (i32 5))]>;
     def PrintCallRetInst6 : NVPTXInst<(outs), (ins),
       !strconcat(OpcStr, " (retval0, retval1, retval2, retval3, retval4, "
                            "retval5), "),
       [(OpNode (i32 6))]>;
     def PrintCallRetInst7 : NVPTXInst<(outs), (ins),
       !strconcat(OpcStr, " (retval0, retval1, retval2, retval3, retval4, "
                            "retval5, retval6), "),
       [(OpNode (i32 7))]>;
     def PrintCallRetInst8 : NVPTXInst<(outs), (ins),
       !strconcat(OpcStr, " (retval0, retval1, retval2, retval3, retval4, "
                            "retval5, retval6, retval7), "),
       [(OpNode (i32 8))]>;
  }
}

defm Call : CALL<"call", PrintCall>;
defm CallUni : CALL<"call.uni", PrintCallUni>;

// Convergent call instructions.  These are identical to regular calls, except
// they have the isConvergent bit set.
let isConvergent=1 in {
  defm ConvergentCall : CALL<"call", PrintConvergentCall>;
  defm ConvergentCallUni : CALL<"call.uni", PrintConvergentCallUni>;
}

def LoadParamMemI64    : LoadParamMemInst<Int64Regs, ".b64">;
def LoadParamMemI32    : LoadParamMemInst<Int32Regs, ".b32">;
def LoadParamMemI16    : LoadParamMemInst<Int16Regs, ".b16">;
def LoadParamMemI8     : LoadParamMemInst<Int16Regs, ".b8">;
def LoadParamMemV2I64  : LoadParamV2MemInst<Int64Regs, ".b64">;
def LoadParamMemV2I32  : LoadParamV2MemInst<Int32Regs, ".b32">;
def LoadParamMemV2I16  : LoadParamV2MemInst<Int16Regs, ".b16">;
def LoadParamMemV2I8   : LoadParamV2MemInst<Int16Regs, ".b8">;
def LoadParamMemV4I32  : LoadParamV4MemInst<Int32Regs, ".b32">;
def LoadParamMemV4I16  : LoadParamV4MemInst<Int16Regs, ".b16">;
def LoadParamMemV4I8   : LoadParamV4MemInst<Int16Regs, ".b8">;
def LoadParamMemF32    : LoadParamMemInst<Float32Regs, ".b32">;
def LoadParamMemF64    : LoadParamMemInst<Float64Regs, ".b64">;
def LoadParamMemV2F32  : LoadParamV2MemInst<Float32Regs, ".b32">;
def LoadParamMemV2F64  : LoadParamV2MemInst<Float64Regs, ".b64">;
def LoadParamMemV4F32  : LoadParamV4MemInst<Float32Regs, ".b32">;

defm StoreParamI64    : StoreParamInst<Int64Regs, i64imm, ".b64">;
defm StoreParamI32    : StoreParamInst<Int32Regs, i32imm, ".b32">;
defm StoreParamI16    : StoreParamInst<Int16Regs, i16imm, ".b16">;
defm StoreParamI8     : StoreParamInst<Int16Regs, i8imm,  ".b8">;

defm StoreParamI8TruncI32 : StoreParamInst<Int32Regs, i8imm, ".b8", /* support_imm */ false>;
defm StoreParamI8TruncI64 : StoreParamInst<Int64Regs, i8imm, ".b8", /* support_imm */ false>;

defm StoreParamV2I64  : StoreParamV2Inst<Int64Regs, i64imm, ".b64">;
defm StoreParamV2I32  : StoreParamV2Inst<Int32Regs, i32imm, ".b32">;
defm StoreParamV2I16  : StoreParamV2Inst<Int16Regs, i16imm, ".b16">;
defm StoreParamV2I8   : StoreParamV2Inst<Int16Regs, i8imm,  ".b8">;

defm StoreParamV4I32  : StoreParamV4Inst<Int32Regs, i32imm, ".b32">;
defm StoreParamV4I16  : StoreParamV4Inst<Int16Regs, i16imm, ".b16">;
defm StoreParamV4I8   : StoreParamV4Inst<Int16Regs, i8imm,  ".b8">;

defm StoreParamF32    : StoreParamInst<Float32Regs, f32imm, ".b32">;
defm StoreParamF64    : StoreParamInst<Float64Regs, f64imm, ".b64">;

defm StoreParamV2F32  : StoreParamV2Inst<Float32Regs, f32imm, ".b32">;
defm StoreParamV2F64  : StoreParamV2Inst<Float64Regs, f64imm, ".b64">;

defm StoreParamV4F32  : StoreParamV4Inst<Float32Regs, f32imm, ".b32">;

def StoreRetvalI64    : StoreRetvalInst<Int64Regs, ".b64">;
def StoreRetvalI32    : StoreRetvalInst<Int32Regs, ".b32">;
def StoreRetvalI16    : StoreRetvalInst<Int16Regs, ".b16">;
def StoreRetvalI8     : StoreRetvalInst<Int16Regs, ".b8">;
def StoreRetvalI8TruncI32 : StoreRetvalInst<Int32Regs, ".b8">;
def StoreRetvalI8TruncI64 : StoreRetvalInst<Int64Regs, ".b8">;
def StoreRetvalV2I64  : StoreRetvalV2Inst<Int64Regs, ".b64">;
def StoreRetvalV2I32  : StoreRetvalV2Inst<Int32Regs, ".b32">;
def StoreRetvalV2I16  : StoreRetvalV2Inst<Int16Regs, ".b16">;
def StoreRetvalV2I8   : StoreRetvalV2Inst<Int16Regs, ".b8">;
def StoreRetvalV4I32  : StoreRetvalV4Inst<Int32Regs, ".b32">;
def StoreRetvalV4I16  : StoreRetvalV4Inst<Int16Regs, ".b16">;
def StoreRetvalV4I8   : StoreRetvalV4Inst<Int16Regs, ".b8">;

def StoreRetvalF64    : StoreRetvalInst<Float64Regs, ".b64">;
def StoreRetvalF32    : StoreRetvalInst<Float32Regs, ".b32">;
def StoreRetvalV2F64  : StoreRetvalV2Inst<Float64Regs, ".b64">;
def StoreRetvalV2F32  : StoreRetvalV2Inst<Float32Regs, ".b32">;
def StoreRetvalV4F32  : StoreRetvalV4Inst<Float32Regs, ".b32">;

def CallArgBeginInst : NVPTXInst<(outs), (ins), "(", [(CallArgBegin)]>;
def CallArgEndInst1  : NVPTXInst<(outs), (ins), ");", [(CallArgEnd (i32 1))]>;
def CallArgEndInst0  : NVPTXInst<(outs), (ins), ")", [(CallArgEnd (i32 0))]>;
def RETURNInst       : NVPTXInst<(outs), (ins), "ret;", [(RETURNNode)]>;

def CallArgParam : NVPTXInst<(outs), (ins i32imm:$a), "param$a, ",
                             [(CallArg 1, imm:$a)]>;
def LastCallArgParam : NVPTXInst<(outs), (ins i32imm:$a), "param$a",
                                 [(LastCallArg 1, imm:$a)]>;

def CallVoidInst :      NVPTXInst<(outs), (ins ADDR_base:$addr), "$addr, ",
                                  [(CallVoid (Wrapper tglobaladdr:$addr))]>;
def CallVoidInstReg :   NVPTXInst<(outs), (ins Int32Regs:$addr), "$addr, ",
                                  [(CallVoid i32:$addr)]>;
def CallVoidInstReg64 : NVPTXInst<(outs), (ins Int64Regs:$addr), "$addr, ",
                                  [(CallVoid i64:$addr)]>;
def PrototypeInst :     NVPTXInst<(outs), (ins i32imm:$val), ", prototype_$val;",
                                  [(Prototype (i32 imm:$val))]>;

def DeclareRetMemInst :
  NVPTXInst<(outs), (ins i32imm:$align, i32imm:$size, i32imm:$num),
            ".param .align $align .b8 retval$num[$size];",
            [(DeclareRetParam (i32 imm:$align), (i32 imm:$size), (i32 imm:$num))]>;
def DeclareRetScalarInst :
  NVPTXInst<(outs), (ins i32imm:$size, i32imm:$num),
            ".param .b$size retval$num;",
            [(DeclareRet (i32 1), (i32 imm:$size), (i32 imm:$num))]>;
def DeclareRetRegInst :
  NVPTXInst<(outs), (ins i32imm:$size, i32imm:$num),
            ".reg .b$size retval$num;",
            [(DeclareRet (i32 2), (i32 imm:$size), (i32 imm:$num))]>;

def DeclareParamInst :
  NVPTXInst<(outs), (ins i32imm:$align, i32imm:$a, i32imm:$size),
            ".param .align $align .b8 param$a[$size];",
            [(DeclareParam (i32 imm:$align), (i32 imm:$a), (i32 imm:$size))]>;
def DeclareScalarParamInst :
  NVPTXInst<(outs), (ins i32imm:$a, i32imm:$size),
            ".param .b$size param$a;",
            [(DeclareScalarParam (i32 imm:$a), (i32 imm:$size), (i32 0))]>;
def DeclareScalarRegInst :
  NVPTXInst<(outs), (ins i32imm:$a, i32imm:$size),
            ".reg .b$size param$a;",
            [(DeclareScalarParam (i32 imm:$a), (i32 imm:$size), (i32 1))]>;

class MoveParamSymbolInst<NVPTXRegClass regclass, Operand srcty, ValueType vt,
                          string asmstr> :
  NVPTXInst<(outs regclass:$dst), (ins srcty:$src),
            !strconcat("mov", asmstr, " \t$dst, $src;"),
            [(set vt:$dst, (MoveParam texternalsym:$src))]>;

def MOV64_PARAM : MoveParamSymbolInst<Int64Regs, i64imm, i64, ".b64">;
def MOV32_PARAM : MoveParamSymbolInst<Int32Regs, i32imm, i32, ".b32">;

class PseudoUseParamInst<NVPTXRegClass regclass, ValueType vt> :
  NVPTXInst<(outs), (ins regclass:$src),
            "// Pseudo use of $src",
            [(PseudoUseParam vt:$src)]>;

def PseudoUseParamI64 : PseudoUseParamInst<Int64Regs, i64>;
def PseudoUseParamI32 : PseudoUseParamInst<Int32Regs, i32>;
def PseudoUseParamI16 : PseudoUseParamInst<Int16Regs, i16>;
def PseudoUseParamF64 : PseudoUseParamInst<Float64Regs, f64>;
def PseudoUseParamF32 : PseudoUseParamInst<Float32Regs, f32>;

class ProxyRegInst<string SzStr, ValueType T, NVPTXRegClass regclass> :
  NVPTXInst<(outs regclass:$dst), (ins regclass:$src),
            !strconcat("mov.", SzStr, " \t$dst, $src;"),
            [(set T:$dst, (ProxyReg T:$src))]>;

def ProxyRegI1    : ProxyRegInst<"pred", i1, Int1Regs>;
def ProxyRegI16   : ProxyRegInst<"b16",  i16, Int16Regs>;
def ProxyRegI32   : ProxyRegInst<"b32",  i32, Int32Regs>;
def ProxyRegI64   : ProxyRegInst<"b64",  i64, Int64Regs>;
def ProxyRegF32   : ProxyRegInst<"b32",  f32, Float32Regs>;
def ProxyRegF64   : ProxyRegInst<"b64",  f64, Float64Regs>;

foreach vt = [f16, bf16] in {
  def: Pat<(vt (ProxyReg  vt:$src)), (ProxyRegI16 $src)>;
}

foreach vt = [v2f16, v2bf16, v2i16, v4i8] in {
  def: Pat<(vt (ProxyReg  vt:$src)), (ProxyRegI32 $src)>;
}

//
// Load / Store Handling
//
class LD<NVPTXRegClass regclass>
  : NVPTXInst<
    (outs regclass:$dst),
    (ins LdStCode:$sem, LdStCode:$scope, LdStCode:$addsp, LdStCode:$Sign,
         i32imm:$fromWidth, ADDR:$addr),
    "ld${sem:sem}${scope:scope}${addsp:addsp}.${Sign:sign}$fromWidth "
    "\t$dst, [$addr];", []>;

let mayLoad=1, hasSideEffects=0 in {
  def LD_i8  : LD<Int16Regs>;
  def LD_i16 : LD<Int16Regs>;
  def LD_i32 : LD<Int32Regs>;
  def LD_i64 : LD<Int64Regs>;
  def LD_f32 : LD<Float32Regs>;
  def LD_f64 : LD<Float64Regs>;
}

class ST<NVPTXRegClass regclass>
  : NVPTXInst<
    (outs),
    (ins regclass:$src, LdStCode:$sem, LdStCode:$scope, LdStCode:$addsp,
         LdStCode:$Sign, i32imm:$toWidth, ADDR:$addr),
    "st${sem:sem}${scope:scope}${addsp:addsp}.${Sign:sign}$toWidth"
    " \t[$addr], $src;", []>;

let mayStore=1, hasSideEffects=0 in {
  def ST_i8  : ST<Int16Regs>;
  def ST_i16 : ST<Int16Regs>;
  def ST_i32 : ST<Int32Regs>;
  def ST_i64 : ST<Int64Regs>;
  def ST_f32 : ST<Float32Regs>;
  def ST_f64 : ST<Float64Regs>;
}

// The following is used only in and after vector elementizations.  Vector
// elementization happens at the machine instruction level, so the following
// instructions never appear in the DAG.
multiclass LD_VEC<NVPTXRegClass regclass, bit support_v8 = false> {
  def _v2 : NVPTXInst<
    (outs regclass:$dst1, regclass:$dst2),
    (ins LdStCode:$sem, LdStCode:$scope, LdStCode:$addsp,
         LdStCode:$Sign, i32imm:$fromWidth, ADDR:$addr),
    "ld${sem:sem}${scope:scope}${addsp:addsp}.v2.${Sign:sign}$fromWidth "
    "\t{{$dst1, $dst2}}, [$addr];", []>;
  def _v4 : NVPTXInst<
    (outs regclass:$dst1, regclass:$dst2, regclass:$dst3, regclass:$dst4),
    (ins LdStCode:$sem, LdStCode:$scope, LdStCode:$addsp,
         LdStCode:$Sign, i32imm:$fromWidth, ADDR:$addr),
    "ld${sem:sem}${scope:scope}${addsp:addsp}.v4.${Sign:sign}$fromWidth "
    "\t{{$dst1, $dst2, $dst3, $dst4}}, [$addr];", []>;
  if support_v8 then
    def _v8 : NVPTXInst<
      (outs regclass:$dst1, regclass:$dst2, regclass:$dst3, regclass:$dst4,
            regclass:$dst5, regclass:$dst6, regclass:$dst7, regclass:$dst8),
      (ins LdStCode:$sem, LdStCode:$scope, LdStCode:$addsp, LdStCode:$Sign,
           i32imm:$fromWidth, ADDR:$addr),
      "ld${sem:sem}${scope:scope}${addsp:addsp}.v8.${Sign:sign}$fromWidth "
      "\t{{$dst1, $dst2, $dst3, $dst4, $dst5, $dst6, $dst7, $dst8}}, "
      "[$addr];", []>;
}
let mayLoad=1, hasSideEffects=0 in {
  defm LDV_i8  : LD_VEC<Int16Regs>;
  defm LDV_i16 : LD_VEC<Int16Regs>;
  defm LDV_i32 : LD_VEC<Int32Regs, support_v8 = true>;
  defm LDV_i64 : LD_VEC<Int64Regs>;
  defm LDV_f32 : LD_VEC<Float32Regs, support_v8 = true>;
  defm LDV_f64 : LD_VEC<Float64Regs>;
}

multiclass ST_VEC<NVPTXRegClass regclass, bit support_v8 = false> {
  def _v2 : NVPTXInst<
    (outs),
    (ins regclass:$src1, regclass:$src2, LdStCode:$sem, LdStCode:$scope,
         LdStCode:$addsp, LdStCode:$Sign, i32imm:$fromWidth,
         ADDR:$addr),
    "st${sem:sem}${scope:scope}${addsp:addsp}.v2.${Sign:sign}$fromWidth "
    "\t[$addr], {{$src1, $src2}};", []>;
  def _v4 : NVPTXInst<
    (outs),
    (ins regclass:$src1, regclass:$src2, regclass:$src3, regclass:$src4,
         LdStCode:$sem, LdStCode:$scope, LdStCode:$addsp,
         LdStCode:$Sign, i32imm:$fromWidth, ADDR:$addr),
    "st${sem:sem}${scope:scope}${addsp:addsp}.v4.${Sign:sign}$fromWidth "
    "\t[$addr], {{$src1, $src2, $src3, $src4}};", []>;
  if support_v8 then
    def _v8 : NVPTXInst<
      (outs),
      (ins regclass:$src1, regclass:$src2, regclass:$src3, regclass:$src4,
           regclass:$src5, regclass:$src6, regclass:$src7, regclass:$src8,
           LdStCode:$sem, LdStCode:$scope, LdStCode:$addsp, LdStCode:$Sign,
           i32imm:$fromWidth, ADDR:$addr),
      "st${sem:sem}${scope:scope}${addsp:addsp}.v8.${Sign:sign}$fromWidth "
      "\t[$addr], "
      "{{$src1, $src2, $src3, $src4, $src5, $src6, $src7, $src8}};", []>;
}

let mayStore=1, hasSideEffects=0 in {
  defm STV_i8  : ST_VEC<Int16Regs>;
  defm STV_i16 : ST_VEC<Int16Regs>;
  defm STV_i32 : ST_VEC<Int32Regs, support_v8 = true>;
  defm STV_i64 : ST_VEC<Int64Regs>;
  defm STV_f32 : ST_VEC<Float32Regs, support_v8 = true>;
  defm STV_f64 : ST_VEC<Float64Regs>;
}

//---- Conversion ----

class F_BITCONVERT<string SzStr, ValueType TIn, ValueType TOut,
  NVPTXRegClass regclassIn = ValueToRegClass<TIn>.ret,
  NVPTXRegClass regclassOut = ValueToRegClass<TOut>.ret> :
           NVPTXInst<(outs regclassOut:$d), (ins regclassIn:$a),
           !strconcat("mov.b", SzStr, " \t$d, $a;"),
     [(set TOut:$d, (bitconvert TIn:$a))]>;

def BITCONVERT_32_I2F : F_BITCONVERT<"32", i32, f32>;
def BITCONVERT_32_F2I : F_BITCONVERT<"32", f32, i32>;
def BITCONVERT_64_I2F : F_BITCONVERT<"64", i64, f64>;
def BITCONVERT_64_F2I : F_BITCONVERT<"64", f64, i64>;

foreach vt = [v2f16, v2bf16, v2i16, v4i8] in {
def: Pat<(vt (bitconvert (f32 Float32Regs:$a))),
         (BITCONVERT_32_F2I $a)>;
def: Pat<(f32 (bitconvert vt:$a)),
         (BITCONVERT_32_I2F $a)>;
}
foreach vt = [f16, bf16] in {
  def: Pat<(vt (bitconvert i16:$a)),
           (vt Int16Regs:$a)>;
  def: Pat<(i16 (bitconvert vt:$a)),
           (i16 Int16Regs:$a)>;
}

foreach ta = [v2f16, v2bf16, v2i16, v4i8, i32] in {
  foreach tb = [v2f16, v2bf16, v2i16, v4i8, i32] in {
    if !ne(ta, tb) then {
      def: Pat<(ta (bitconvert tb:$a)),
               (ta Int32Regs:$a)>;
    }
  }
}

// NOTE: pred->fp are currently sub-optimal due to an issue in TableGen where
// we cannot specify floating-point literals in isel patterns.  Therefore, we
// use an integer selp to select either 1 (or -1 in case of signed) or 0
// and then cvt to floating-point.

// sint -> f16
def : Pat<(f16 (sint_to_fp i1:$a)),
          (CVT_f16_s32 (SELP_b32ii -1, 0, $a), CvtRN)>;
def : Pat<(f16 (sint_to_fp Int16Regs:$a)),
          (CVT_f16_s16 $a, CvtRN)>;
def : Pat<(f16 (sint_to_fp i32:$a)),
          (CVT_f16_s32 $a, CvtRN)>;
def : Pat<(f16 (sint_to_fp i64:$a)),
          (CVT_f16_s64 $a, CvtRN)>;

// uint -> f16
def : Pat<(f16 (uint_to_fp i1:$a)),
          (CVT_f16_u32 (SELP_b32ii 1, 0, $a), CvtRN)>;
def : Pat<(f16 (uint_to_fp Int16Regs:$a)),
          (CVT_f16_u16 $a, CvtRN)>;
def : Pat<(f16 (uint_to_fp i32:$a)),
          (CVT_f16_u32 $a, CvtRN)>;
def : Pat<(f16 (uint_to_fp i64:$a)),
          (CVT_f16_u64 $a, CvtRN)>;

// sint -> bf16
def : Pat<(bf16 (sint_to_fp i1:$a)),
          (CVT_bf16_s32 (SELP_b32ii 1, 0, $a), CvtRN)>, Requires<[hasPTX<78>, hasSM<90>]>;
def : Pat<(bf16 (sint_to_fp i16:$a)),
          (CVT_bf16_s16 $a, CvtRN)>, Requires<[hasPTX<78>, hasSM<90>]>;
def : Pat<(bf16 (sint_to_fp i32:$a)),
          (CVT_bf16_s32 $a, CvtRN)>, Requires<[hasPTX<78>, hasSM<90>]>;
def : Pat<(bf16 (sint_to_fp i64:$a)),
          (CVT_bf16_s64 $a, CvtRN)>, Requires<[hasPTX<78>, hasSM<90>]>;

// uint -> bf16
def : Pat<(bf16 (uint_to_fp i1:$a)),
          (CVT_bf16_u32 (SELP_b32ii 1, 0, $a), CvtRN)>, Requires<[hasPTX<78>, hasSM<90>]>;
def : Pat<(bf16 (uint_to_fp i16:$a)),
          (CVT_bf16_u16 $a, CvtRN)>, Requires<[hasPTX<78>, hasSM<90>]>;
def : Pat<(bf16 (uint_to_fp i32:$a)),
          (CVT_bf16_u32 $a, CvtRN)>, Requires<[hasPTX<78>, hasSM<90>]>;
def : Pat<(bf16 (uint_to_fp i64:$a)),
          (CVT_bf16_u64 $a, CvtRN)>, Requires<[hasPTX<78>, hasSM<90>]>;

// sint -> f32
def : Pat<(f32 (sint_to_fp i1:$a)),
          (CVT_f32_s32 (SELP_b32ii -1, 0, $a), CvtRN)>;
def : Pat<(f32 (sint_to_fp i16:$a)),
          (CVT_f32_s16 $a, CvtRN)>;
def : Pat<(f32 (sint_to_fp i32:$a)),
          (CVT_f32_s32 $a, CvtRN)>;
def : Pat<(f32 (sint_to_fp i64:$a)),
          (CVT_f32_s64 $a, CvtRN)>;

// uint -> f32
def : Pat<(f32 (uint_to_fp i1:$a)),
          (CVT_f32_u32 (SELP_b32ii 1, 0, $a), CvtRN)>;
def : Pat<(f32 (uint_to_fp i16:$a)),
          (CVT_f32_u16 $a, CvtRN)>;
def : Pat<(f32 (uint_to_fp i32:$a)),
          (CVT_f32_u32 $a, CvtRN)>;
def : Pat<(f32 (uint_to_fp i64:$a)),
          (CVT_f32_u64 $a, CvtRN)>;

// sint -> f64
def : Pat<(f64 (sint_to_fp i1:$a)),
          (CVT_f64_s32 (SELP_b32ii -1, 0, $a), CvtRN)>;
def : Pat<(f64 (sint_to_fp i16:$a)),
          (CVT_f64_s16 $a, CvtRN)>;
def : Pat<(f64 (sint_to_fp i32:$a)),
          (CVT_f64_s32 $a, CvtRN)>;
def : Pat<(f64 (sint_to_fp i64:$a)),
          (CVT_f64_s64 $a, CvtRN)>;

// uint -> f64
def : Pat<(f64 (uint_to_fp i1:$a)),
          (CVT_f64_u32 (SELP_b32ii 1, 0, $a), CvtRN)>;
def : Pat<(f64 (uint_to_fp i16:$a)),
          (CVT_f64_u16 $a, CvtRN)>;
def : Pat<(f64 (uint_to_fp i32:$a)),
          (CVT_f64_u32 $a, CvtRN)>;
def : Pat<(f64 (uint_to_fp i64:$a)),
          (CVT_f64_u64 $a, CvtRN)>;


// f16 -> sint
def : Pat<(i1 (fp_to_sint f16:$a)),
          (SETP_b16ri $a, 0, CmpEQ)>;
def : Pat<(i16 (fp_to_sint f16:$a)),
          (CVT_s16_f16 $a, CvtRZI)>;
def : Pat<(i32 (fp_to_sint f16:$a)),
          (CVT_s32_f16 $a, CvtRZI)>;
def : Pat<(i64 (fp_to_sint f16:$a)),
          (CVT_s64_f16 $a, CvtRZI)>;

// f16 -> uint
def : Pat<(i1 (fp_to_uint f16:$a)),
          (SETP_b16ri $a, 0, CmpEQ)>;
def : Pat<(i16 (fp_to_uint f16:$a)),
          (CVT_u16_f16 $a, CvtRZI)>;
def : Pat<(i32 (fp_to_uint f16:$a)),
          (CVT_u32_f16 $a, CvtRZI)>;
def : Pat<(i64 (fp_to_uint f16:$a)),
          (CVT_u64_f16 $a, CvtRZI)>;

// bf16 -> sint
def : Pat<(i1 (fp_to_sint bf16:$a)),
          (SETP_b16ri $a, 0, CmpEQ)>;
def : Pat<(i16 (fp_to_sint bf16:$a)),
          (CVT_s16_bf16 $a, CvtRZI)>;
def : Pat<(i32 (fp_to_sint bf16:$a)),
          (CVT_s32_bf16 $a, CvtRZI)>;
def : Pat<(i64 (fp_to_sint bf16:$a)),
          (CVT_s64_bf16 $a, CvtRZI)>;

// bf16 -> uint
def : Pat<(i1 (fp_to_uint bf16:$a)),
          (SETP_b16ri $a, 0, CmpEQ)>;
def : Pat<(i16 (fp_to_uint bf16:$a)),
          (CVT_u16_bf16 $a, CvtRZI)>;
def : Pat<(i32 (fp_to_uint bf16:$a)),
          (CVT_u32_bf16 $a, CvtRZI)>;
def : Pat<(i64 (fp_to_uint bf16:$a)),
          (CVT_u64_bf16 $a, CvtRZI)>;
// f32 -> sint
def : Pat<(i1 (fp_to_sint f32:$a)),
          (SETP_b32ri (BITCONVERT_32_F2I $a), 0, CmpEQ)>;
def : Pat<(i16 (fp_to_sint f32:$a)),
          (CVT_s16_f32 $a, CvtRZI_FTZ)>, Requires<[doF32FTZ]>;
def : Pat<(i16 (fp_to_sint f32:$a)),
          (CVT_s16_f32 $a, CvtRZI)>;
def : Pat<(i32 (fp_to_sint f32:$a)),
          (CVT_s32_f32 $a, CvtRZI_FTZ)>, Requires<[doF32FTZ]>;
def : Pat<(i32 (fp_to_sint f32:$a)),
          (CVT_s32_f32 $a, CvtRZI)>;
def : Pat<(i64 (fp_to_sint f32:$a)),
          (CVT_s64_f32 $a, CvtRZI_FTZ)>, Requires<[doF32FTZ]>;
def : Pat<(i64 (fp_to_sint f32:$a)),
          (CVT_s64_f32 $a, CvtRZI)>;

// f32 -> uint
def : Pat<(i1 (fp_to_uint f32:$a)),
          (SETP_b32ri (BITCONVERT_32_F2I $a), 0, CmpEQ)>;
def : Pat<(i16 (fp_to_uint f32:$a)),
          (CVT_u16_f32 $a, CvtRZI_FTZ)>, Requires<[doF32FTZ]>;
def : Pat<(i16 (fp_to_uint f32:$a)),
          (CVT_u16_f32 $a, CvtRZI)>;
def : Pat<(i32 (fp_to_uint f32:$a)),
          (CVT_u32_f32 $a, CvtRZI_FTZ)>, Requires<[doF32FTZ]>;
def : Pat<(i32 (fp_to_uint f32:$a)),
          (CVT_u32_f32 $a, CvtRZI)>;
def : Pat<(i64 (fp_to_uint f32:$a)),
          (CVT_u64_f32 $a, CvtRZI_FTZ)>, Requires<[doF32FTZ]>;
def : Pat<(i64 (fp_to_uint f32:$a)),
          (CVT_u64_f32 $a, CvtRZI)>;

// f64 -> sint
def : Pat<(i1 (fp_to_sint f64:$a)),
          (SETP_b64ri (BITCONVERT_64_F2I $a), 0, CmpEQ)>;
def : Pat<(i16 (fp_to_sint f64:$a)),
          (CVT_s16_f64 $a, CvtRZI)>;
def : Pat<(i32 (fp_to_sint f64:$a)),
          (CVT_s32_f64 $a, CvtRZI)>;
def : Pat<(i64 (fp_to_sint f64:$a)),
          (CVT_s64_f64 $a, CvtRZI)>;

// f64 -> uint
def : Pat<(i1 (fp_to_uint f64:$a)),
          (SETP_b64ri (BITCONVERT_64_F2I $a), 0, CmpEQ)>;
def : Pat<(i16 (fp_to_uint f64:$a)),
          (CVT_u16_f64 $a, CvtRZI)>;
def : Pat<(i32 (fp_to_uint f64:$a)),
          (CVT_u32_f64 $a, CvtRZI)>;
def : Pat<(i64 (fp_to_uint f64:$a)),
          (CVT_u64_f64 $a, CvtRZI)>;

// sext i1
def : Pat<(i16 (sext i1:$a)),
          (SELP_b16ii -1, 0, $a)>;
def : Pat<(i32 (sext i1:$a)),
          (SELP_b32ii -1, 0, $a)>;
def : Pat<(i64 (sext i1:$a)),
          (SELP_b64ii -1, 0, $a)>;

// zext i1
def : Pat<(i16 (zext i1:$a)),
          (SELP_b16ii 1, 0, $a)>;
def : Pat<(i32 (zext i1:$a)),
          (SELP_b32ii 1, 0, $a)>;
def : Pat<(i64 (zext i1:$a)),
          (SELP_b64ii 1, 0, $a)>;

// anyext i1
def : Pat<(i16 (anyext i1:$a)),
          (SELP_b16ii -1, 0, $a)>;
def : Pat<(i32 (anyext i1:$a)),
          (SELP_b32ii -1, 0, $a)>;
def : Pat<(i64 (anyext i1:$a)),
          (SELP_b64ii -1, 0, $a)>;

// sext i16
def : Pat<(i32 (sext i16:$a)),
          (CVT_s32_s16 $a, CvtNONE)>;
def : Pat<(i64 (sext i16:$a)),
          (CVT_s64_s16 $a, CvtNONE)>;

// zext i16
def : Pat<(i32 (zext i16:$a)),
          (CVT_u32_u16 $a, CvtNONE)>;
def : Pat<(i64 (zext i16:$a)),
          (CVT_u64_u16 $a, CvtNONE)>;

// anyext i16
def : Pat<(i32 (anyext i16:$a)),
          (CVT_u32_u16 $a, CvtNONE)>;
def : Pat<(i64 (anyext i16:$a)),
          (CVT_u64_u16 $a, CvtNONE)>;

// sext i32
def : Pat<(i64 (sext i32:$a)),
          (CVT_s64_s32 $a, CvtNONE)>;

// zext i32
def : Pat<(i64 (zext i32:$a)),
          (CVT_u64_u32 $a, CvtNONE)>;

// anyext i32
def : Pat<(i64 (anyext i32:$a)),
          (CVT_u64_u32 $a, CvtNONE)>;


// truncate i64
def : Pat<(i32 (trunc i64:$a)),
          (CVT_u32_u64 $a, CvtNONE)>;
def : Pat<(i16 (trunc i64:$a)),
          (CVT_u16_u64 $a, CvtNONE)>;
def : Pat<(i1 (trunc i64:$a)),
          (SETP_b64ri (ANDb64ri $a, 1), 0, CmpNE)>;

// truncate i32
def : Pat<(i16 (trunc i32:$a)),
          (CVT_u16_u32 $a, CvtNONE)>;
def : Pat<(i1 (trunc i32:$a)),
          (SETP_b32ri (ANDb32ri $a, 1), 0, CmpNE)>;

// truncate i16
def : Pat<(i1 (trunc i16:$a)),
          (SETP_b16ri (ANDb16ri $a, 1), 0, CmpNE)>;

// sext_inreg
def : Pat<(sext_inreg i16:$a, i8), (CVT_INREG_s16_s8 $a)>;
def : Pat<(sext_inreg i32:$a, i8), (CVT_INREG_s32_s8 $a)>;
def : Pat<(sext_inreg i32:$a, i16), (CVT_INREG_s32_s16 $a)>;
def : Pat<(sext_inreg i64:$a, i8), (CVT_INREG_s64_s8 $a)>;
def : Pat<(sext_inreg i64:$a, i16), (CVT_INREG_s64_s16 $a)>;
def : Pat<(sext_inreg i64:$a, i32), (CVT_INREG_s64_s32 $a)>;

let hasSideEffects = false in {
  // pack a set of smaller int registers to a larger int register
  def V4I16toI64 : NVPTXInst<(outs Int64Regs:$d),
                             (ins Int16Regs:$s1, Int16Regs:$s2,
                                  Int16Regs:$s3, Int16Regs:$s4),
                             "mov.b64 \t$d, {{$s1, $s2, $s3, $s4}};", []>;
  def V2I16toI32 : NVPTXInst<(outs Int32Regs:$d),
                             (ins Int16Regs:$s1, Int16Regs:$s2),
                             "mov.b32 \t$d, {{$s1, $s2}};", []>;
  def V2I32toI64 : NVPTXInst<(outs Int64Regs:$d),
                             (ins Int32Regs:$s1, Int32Regs:$s2),
                             "mov.b64 \t$d, {{$s1, $s2}};", []>;
  def V2I64toI128 : NVPTXInst<(outs Int128Regs:$d),
                              (ins Int64Regs:$s1, Int64Regs:$s2),
                              "mov.b128 \t$d, {{$s1, $s2}};", []>;
  def V2F32toF64 : NVPTXInst<(outs Float64Regs:$d),
                             (ins Float32Regs:$s1, Float32Regs:$s2),
                             "mov.b64 \t$d, {{$s1, $s2}};", []>;

  // unpack a larger int register to a set of smaller int registers
  def I64toV4I16 : NVPTXInst<(outs Int16Regs:$d1, Int16Regs:$d2,
                                   Int16Regs:$d3, Int16Regs:$d4),
                             (ins Int64Regs:$s),
                             "mov.b64 \t{{$d1, $d2, $d3, $d4}}, $s;", []>;
  def I32toV2I16 : NVPTXInst<(outs Int16Regs:$d1, Int16Regs:$d2),
                             (ins Int32Regs:$s),
                             "mov.b32 \t{{$d1, $d2}}, $s;", []>;
  def I64toV2I32 : NVPTXInst<(outs Int32Regs:$d1, Int32Regs:$d2),
                             (ins Int64Regs:$s),
                             "mov.b64 \t{{$d1, $d2}}, $s;", []>;
  def I128toV2I64: NVPTXInst<(outs Int64Regs:$d1, Int64Regs:$d2),
                              (ins Int128Regs:$s),
                              "mov.b128 \t{{$d1, $d2}}, $s;", []>;
  def F64toV2F32 : NVPTXInst<(outs Float32Regs:$d1, Float32Regs:$d2),
                             (ins Float64Regs:$s),
                             "mov.b64 \t{{$d1, $d2}}, $s;", []>;

  def I32toI16H  : NVPTXInst<(outs Int16Regs:$high),
                             (ins Int32Regs:$s),
                             "{{ .reg .b16 tmp; mov.b32 {tmp, $high}, $s; }}",
                             []>;
  def I32toI16L  : NVPTXInst<(outs Int16Regs:$low),
                             (ins Int32Regs:$s),
                             "{{ .reg .b16 tmp; mov.b32 {$low, tmp}, $s; }}",
                             []>;
  def I64toI32H  : NVPTXInst<(outs Int32Regs:$high),
                             (ins Int64Regs:$s),
                             "{{ .reg .b32 tmp; mov.b64 {tmp, $high}, $s; }}",
                             []>;
  def I64toI32L  : NVPTXInst<(outs Int32Regs:$low),
                             (ins Int64Regs:$s),
                             "{{ .reg .b32 tmp; mov.b64 {$low, tmp}, $s; }}",
                             []>;

  // PTX 7.1 lets you avoid a temp register and just use _ as a "sink" for the
  // unused high/low part.
  def I32toI16H_Sink  : NVPTXInst<(outs Int16Regs:$high),
                             (ins Int32Regs:$s),
                             "mov.b32 \t{{_, $high}}, $s;",
                             []>, Requires<[hasPTX<71>]>;
  def I32toI16L_Sink  : NVPTXInst<(outs Int16Regs:$low),
                             (ins Int32Regs:$s),
                             "mov.b32 \t{{$low, _}}, $s;",
                             []>, Requires<[hasPTX<71>]>;
  def I64toI32H_Sink  : NVPTXInst<(outs Int32Regs:$high),
                             (ins Int64Regs:$s),
                             "mov.b64 \t{{_, $high}}, $s;",
                             []>, Requires<[hasPTX<71>]>;
  def I64toI32L_Sink  : NVPTXInst<(outs Int32Regs:$low),
                             (ins Int64Regs:$s),
                             "mov.b64 \t{{$low, _}}, $s;",
                             []>, Requires<[hasPTX<71>]>;
}

def : Pat<(i16 (trunc (srl i32:$s, (i32 16)))), (I32toI16H_Sink i32:$s)>, Requires<[hasPTX<71>]>;
def : Pat<(i16 (trunc (sra i32:$s, (i32 16)))), (I32toI16H_Sink i32:$s)>, Requires<[hasPTX<71>]>;
def : Pat<(i32 (trunc (srl i64:$s, (i32 32)))), (I64toI32H_Sink i64:$s)>, Requires<[hasPTX<71>]>;
def : Pat<(i32 (trunc (sra i64:$s, (i32 32)))), (I64toI32H_Sink i64:$s)>, Requires<[hasPTX<71>]>;

// Fall back to the old way if we don't have PTX 7.1.
def : Pat<(i16 (trunc (srl i32:$s, (i32 16)))), (I32toI16H $s)>;
def : Pat<(i16 (trunc (sra i32:$s, (i32 16)))), (I32toI16H $s)>;
def : Pat<(i32 (trunc (srl i64:$s, (i32 32)))), (I64toI32H $s)>;
def : Pat<(i32 (trunc (sra i64:$s, (i32 32)))), (I64toI32H $s)>;

def: Pat<(i32 (sext (extractelt v2i16:$src, 0))),
         (CVT_INREG_s32_s16 $src)>;

foreach vt = [v2f16, v2bf16, v2i16] in {
  def : Pat<(extractelt vt:$src, 0), (I32toI16L_Sink $src)>, Requires<[hasPTX<71>]>;
  def : Pat<(extractelt vt:$src, 1), (I32toI16H_Sink $src)>, Requires<[hasPTX<71>]>;

  def : Pat<(extractelt vt:$src, 0), (I32toI16L $src)>;
  def : Pat<(extractelt vt:$src, 1), (I32toI16H $src)>;
}
def : Pat<(v2f16 (build_vector f16:$a, f16:$b)),
          (V2I16toI32 $a, $b)>;
def : Pat<(v2bf16 (build_vector bf16:$a, bf16:$b)),
          (V2I16toI32 $a, $b)>;
def : Pat<(v2i16 (build_vector i16:$a, i16:$b)),
          (V2I16toI32 $a, $b)>;

def: Pat<(v2i16 (scalar_to_vector i16:$a)),
         (CVT_u32_u16 $a, CvtNONE)>;


def nvptx_build_vector : SDNode<"NVPTXISD::BUILD_VECTOR", SDTypeProfile<1, 2, []>, []>;

def : Pat<(i64 (nvptx_build_vector i32:$a, i32:$b)),
          (V2I32toI64 $a, $b)>;

//
// Funnel-Shift
//

// Create SDNodes so they can be used in the DAG code, e.g.
// NVPTXISelLowering (LowerShiftLeftParts and LowerShiftRightParts)
def fshl_clamp : SDNode<"NVPTXISD::FSHL_CLAMP", SDTIntShiftDOp, []>;
def fshr_clamp : SDNode<"NVPTXISD::FSHR_CLAMP", SDTIntShiftDOp, []>;

// Funnel shift, requires >= sm_32.  Does not trap if amt is out of range, so
// no side effects.
let hasSideEffects = false in {
  multiclass ShfInst<string mode, SDNode op> {
    def _i
      : NVPTXInst<(outs Int32Regs:$dst),
                  (ins  Int32Regs:$lo, Int32Regs:$hi, i32imm:$amt),
                  "shf." # mode # ".b32 \t$dst, $lo, $hi, $amt;",
                  [(set i32:$dst,
                      (op i32:$hi, i32:$lo, (i32 imm:$amt)))]>,
        Requires<[hasHWROT32]>;

    def _r
      : NVPTXInst<(outs Int32Regs:$dst),
                  (ins  Int32Regs:$lo, Int32Regs:$hi, Int32Regs:$amt),
                  "shf." # mode # ".b32 \t$dst, $lo, $hi, $amt;",
                  [(set i32:$dst,
                      (op i32:$hi, i32:$lo, i32:$amt))]>,
        Requires<[hasHWROT32]>;
  }

  defm SHF_L_CLAMP : ShfInst<"l.clamp", fshl_clamp>;
  defm SHF_R_CLAMP : ShfInst<"r.clamp", fshr_clamp>;
  defm SHF_L_WRAP  : ShfInst<"l.wrap", fshl>;
  defm SHF_R_WRAP  : ShfInst<"r.wrap", fshr>;
}

def : Pat<(i32 (int_nvvm_fshl_clamp i32:$hi, i32:$lo, i32:$amt)),
          (SHF_L_CLAMP_r $lo, $hi, $amt)>;
def : Pat<(i32 (int_nvvm_fshl_clamp i32:$hi, i32:$lo, (i32 imm:$amt))),
          (SHF_L_CLAMP_i $lo, $hi, imm:$amt)>;
def : Pat<(i32 (int_nvvm_fshr_clamp i32:$hi, i32:$lo, i32:$amt)),
          (SHF_R_CLAMP_r $lo, $hi, $amt)>;
def : Pat<(i32 (int_nvvm_fshr_clamp i32:$hi, i32:$lo, (i32 imm:$amt))),
          (SHF_R_CLAMP_i $lo, $hi, imm:$amt)>;

let hasSideEffects = false in {
  foreach RT = [I32RT, I64RT] in {
    // Count leading zeros
    def CLZr # RT.Size : NVPTXInst<(outs Int32Regs:$d), (ins RT.RC:$a),
                                   "clz.b" # RT.Size # " \t$d, $a;",
                                   [(set i32:$d, (ctlz RT.Ty:$a))]>;

    // Population count
    def POPCr # RT.Size : NVPTXInst<(outs Int32Regs:$d), (ins RT.RC:$a),
                                    "popc.b" # RT.Size # " \t$d, $a;",
                                    [(set i32:$d, (ctpop RT.Ty:$a))]>;
  }
}

// fpround f32 -> f16
def : Pat<(f16 (fpround f32:$a)),
          (CVT_f16_f32 $a, CvtRN)>;

// fpround f32 -> bf16
def : Pat<(bf16 (fpround f32:$a)),
          (CVT_bf16_f32 $a, CvtRN)>, Requires<[hasPTX<70>, hasSM<80>]>;

// fpround f64 -> f16
def : Pat<(f16 (fpround f64:$a)),
          (CVT_f16_f64 $a, CvtRN)>;

// fpround f64 -> bf16
def : Pat<(bf16 (fpround f64:$a)),
          (CVT_bf16_f64 $a, CvtRN)>, Requires<[hasPTX<78>, hasSM<90>]>;
// fpround f64 -> f32
def : Pat<(f32 (fpround f64:$a)),
          (CVT_f32_f64 $a, CvtRN_FTZ)>, Requires<[doF32FTZ]>;
def : Pat<(f32 (fpround f64:$a)),
          (CVT_f32_f64 $a, CvtRN)>;

// fpextend f16 -> f32
def : Pat<(f32 (fpextend f16:$a)),
          (CVT_f32_f16 $a, CvtNONE_FTZ)>, Requires<[doF32FTZ]>;
def : Pat<(f32 (fpextend f16:$a)),
          (CVT_f32_f16 $a, CvtNONE)>;
// fpextend bf16 -> f32
def : Pat<(f32 (fpextend bf16:$a)),
          (CVT_f32_bf16 $a, CvtNONE_FTZ)>, Requires<[doF32FTZ]>;
def : Pat<(f32 (fpextend bf16:$a)),
          (CVT_f32_bf16 $a, CvtNONE)>, Requires<[hasPTX<71>, hasSM<80>]>;

// fpextend f16 -> f64
def : Pat<(f64 (fpextend f16:$a)),
          (CVT_f64_f16 $a, CvtNONE)>;

// fpextend bf16 -> f64
def : Pat<(f64 (fpextend bf16:$a)),
          (CVT_f64_bf16 $a, CvtNONE)>, Requires<[hasPTX<78>, hasSM<90>]>;

// fpextend f32 -> f64
def : Pat<(f64 (fpextend f32:$a)),
          (CVT_f64_f32 $a, CvtNONE_FTZ)>, Requires<[doF32FTZ]>;
def : Pat<(f64 (fpextend f32:$a)),
          (CVT_f64_f32 $a, CvtNONE)>;

def retglue : SDNode<"NVPTXISD::RET_GLUE", SDTNone,
                     [SDNPHasChain, SDNPOptInGlue]>;

// fceil, ffloor, froundeven, ftrunc.

multiclass CVT_ROUND<SDNode OpNode, PatLeaf Mode, PatLeaf ModeFTZ> {
  def : Pat<(OpNode f16:$a),
            (CVT_f16_f16 $a, Mode)>;
  def : Pat<(OpNode bf16:$a),
            (CVT_bf16_bf16 $a, Mode)>;
  def : Pat<(OpNode f32:$a),
            (CVT_f32_f32 $a, ModeFTZ)>, Requires<[doF32FTZ]>;
  def : Pat<(OpNode f32:$a),
            (CVT_f32_f32 $a, Mode)>, Requires<[doNoF32FTZ]>;
  def : Pat<(OpNode f64:$a),
            (CVT_f64_f64 $a, Mode)>;
}

defm : CVT_ROUND<fceil, CvtRPI, CvtRPI_FTZ>;
defm : CVT_ROUND<ffloor, CvtRMI, CvtRMI_FTZ>;
defm : CVT_ROUND<froundeven, CvtRNI, CvtRNI_FTZ>;
defm : CVT_ROUND<ftrunc, CvtRZI, CvtRZI_FTZ>;

// nearbyint and rint are implemented as rounding to nearest even.  This isn't
// strictly correct, because it causes us to ignore the rounding mode.  But it
// matches what CUDA's "libm" does.

defm : CVT_ROUND<fnearbyint, CvtRNI, CvtRNI_FTZ>;
defm : CVT_ROUND<frint, CvtRNI, CvtRNI_FTZ>;

//-----------------------------------
// Control-flow
//-----------------------------------

let isTerminator=1 in {
   let isReturn=1, isBarrier=1 in
      def Return : NVPTXInst<(outs), (ins), "ret;", [(retglue)]>;

   let isBranch=1 in
      def CBranch : NVPTXInst<(outs), (ins Int1Regs:$a, brtarget:$target),
                              "@$a bra \t$target;",
                              [(brcond i1:$a, bb:$target)]>;
   let isBranch=1 in
      def CBranchOther : NVPTXInst<(outs), (ins Int1Regs:$a, brtarget:$target),
                                   "@!$a bra \t$target;", []>;

   let isBranch=1, isBarrier=1 in
      def GOTO : NVPTXInst<(outs), (ins brtarget:$target),
                           "bra.uni \t$target;", [(br bb:$target)]>;
}

def : Pat<(brcond i32:$a, bb:$target),
          (CBranch (SETP_u32ri $a, 0, CmpNE), bb:$target)>;

// SelectionDAGBuilder::visitSWitchCase() will invert the condition of a
// conditional branch if the target block is the next block so that the code
// can fall through to the target block.  The invertion is done by 'xor
// condition, 1', which will be translated to (setne condition, -1).  Since ptx
// supports '@!pred bra target', we should use it.
def : Pat<(brcond (i1 (setne i1:$a, -1)), bb:$target),
          (CBranchOther $a, bb:$target)>;

// Call
def SDT_NVPTXCallSeqStart : SDCallSeqStart<[SDTCisVT<0, i32>,
                                            SDTCisVT<1, i32>]>;
def SDT_NVPTXCallSeqEnd   : SDCallSeqEnd<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;

def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_NVPTXCallSeqStart,
                           [SDNPHasChain, SDNPOutGlue, SDNPSideEffect]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END", SDT_NVPTXCallSeqEnd,
                           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                            SDNPSideEffect]>;

def SDT_NVPTXCall : SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>;
def call          : SDNode<"NVPTXISD::CALL", SDT_NVPTXCall,
                           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;
def calltarget : Operand<i32>;
let isCall=1 in {
   def CALL : NVPTXInst<(outs), (ins calltarget:$dst), "call \t$dst, (1);", []>;
}

def : Pat<(call tglobaladdr:$dst), (CALL tglobaladdr:$dst)>;
def : Pat<(call texternalsym:$dst), (CALL texternalsym:$dst)>;

// Pseudo instructions.
class Pseudo<dag outs, dag ins, string asmstr, list<dag> pattern>
   : NVPTXInst<outs, ins, asmstr, pattern>;

def Callseq_Start :
  NVPTXInst<(outs), (ins i32imm:$amt1, i32imm:$amt2),
            "\\{ // callseq $amt1, $amt2",
            [(callseq_start timm:$amt1, timm:$amt2)]>;
def Callseq_End :
  NVPTXInst<(outs), (ins i32imm:$amt1, i32imm:$amt2),
            "\\} // callseq $amt1",
            [(callseq_end timm:$amt1, timm:$amt2)]>;

// trap instruction
def trapinst : NVPTXInst<(outs), (ins), "trap;", [(trap)]>, Requires<[noPTXASUnreachableBug]>;
// Emit an `exit` as well to convey to ptxas that `trap` exits the CFG.
// This won't be necessary in a future version of ptxas.
def trapexitinst : NVPTXInst<(outs), (ins), "trap; exit;", [(trap)]>, Requires<[hasPTXASUnreachableBug]>;
// brkpt instruction
def debugtrapinst : NVPTXInst<(outs), (ins), "brkpt;", [(debugtrap)]>;

// Call prototype wrapper
def SDTCallPrototype : SDTypeProfile<0, 1, [SDTCisInt<0>]>;
def CallPrototype :
  SDNode<"NVPTXISD::CallPrototype", SDTCallPrototype,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def ProtoIdent : Operand<i32> {
  let PrintMethod = "printProtoIdent";
}
def CALL_PROTOTYPE :
  NVPTXInst<(outs), (ins ProtoIdent:$ident),
            "$ident", [(CallPrototype (i32 texternalsym:$ident))]>;

def SDTDynAllocaOp :
  SDTypeProfile<1, 2, [SDTCisSameAs<0, 1>, SDTCisInt<1>, SDTCisVT<2, i32>]>;

def dyn_alloca :
  SDNode<"NVPTXISD::DYNAMIC_STACKALLOC", SDTDynAllocaOp,
         [SDNPHasChain, SDNPSideEffect]>;

foreach t = [I32RT, I64RT] in {
  def DYNAMIC_STACKALLOC # t.Size :
    NVPTXInst<(outs t.RC:$ptr),
              (ins t.RC:$size, i32imm:$align),
              "alloca.u" # t.Size # " \t$ptr, $size, $align;",
              [(set t.Ty:$ptr, (dyn_alloca t.Ty:$size, timm:$align))]>,
              Requires<[hasPTX<73>, hasSM<52>]>;
}

//
// BRX
//

def SDTBrxStartProfile : SDTypeProfile<0, 1, [SDTCisInt<0>]>;
def SDTBrxItemProfile : SDTypeProfile<0, 1, [SDTCisVT<0, OtherVT>]>;
def SDTBrxEndProfile : SDTypeProfile<0, 3, [SDTCisVT<0, OtherVT>, SDTCisInt<1>, SDTCisInt<2>]>;

def brx_start :
  SDNode<"NVPTXISD::BrxStart", SDTBrxStartProfile,
         [SDNPHasChain, SDNPOutGlue, SDNPSideEffect]>;
def brx_item :
  SDNode<"NVPTXISD::BrxItem", SDTBrxItemProfile,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def brx_end :
  SDNode<"NVPTXISD::BrxEnd", SDTBrxEndProfile,
         [SDNPHasChain, SDNPInGlue, SDNPSideEffect]>;

let isTerminator = 1, isBranch = 1, isIndirectBranch = 1, isNotDuplicable = 1 in {

  def BRX_START :
    NVPTXInst<(outs), (ins i32imm:$id),
              "$$L_brx_$id: .branchtargets",
              [(brx_start (i32 imm:$id))]>;

  def BRX_ITEM :
    NVPTXInst<(outs), (ins brtarget:$target),
              "\t$target,",
              [(brx_item bb:$target)]>;

  def BRX_END :
    NVPTXInst<(outs), (ins brtarget:$target, Int32Regs:$val, i32imm:$id),
              "\t$target;\n\tbrx.idx \t$val, $$L_brx_$id;",
              [(brx_end bb:$target, i32:$val, (i32 imm:$id))]> {
      let isBarrier = 1;
    }
}


foreach a_type = ["s", "u"] in {
  foreach b_type = ["s", "u"] in {

    def DOT4_ # a_type # b_type :
      NVPTXInst<(outs Int32Regs:$dst),
                (ins Int32Regs:$a, Int32Regs:$b, Int32Regs:$c),
                "dp4a." # a_type # "32." # b_type # "32 \t$dst, $a, $b, $c;",
                [(set i32:$dst,
                    (!cast<Intrinsic>("int_nvvm_idp4a_" # a_type # "_" # b_type)
                     i32:$a, i32:$b, i32:$c))]>,
                Requires<[hasDotInstructions]>;

    foreach is_hi = [0, -1] in {
      defvar lohi_suffix = !if(is_hi, "hi", "lo");

      def DOT2_ # lohi_suffix # _ # a_type # b_type :
        NVPTXInst<(outs Int32Regs:$dst),
                  (ins Int32Regs:$a, Int32Regs:$b, Int32Regs:$c),
                  "dp2a." # lohi_suffix # "." # a_type # "32." # b_type # "32 \t$dst, $a, $b, $c;",
                  [(set i32:$dst,
                      (!cast<Intrinsic>("int_nvvm_idp2a_" # a_type # "_" # b_type)
                       i32:$a, i32:$b, is_hi, i32:$c))]>,
                  Requires<[hasDotInstructions]>;
    }
  }
}

//
// Stack Manipulation
//

def SDTStackRestore : SDTypeProfile<0, 1, [SDTCisInt<0>]>;

def stackrestore :
  SDNode<"NVPTXISD::STACKRESTORE", SDTStackRestore,
         [SDNPHasChain, SDNPSideEffect]>;

def stacksave :
  SDNode<"NVPTXISD::STACKSAVE", SDTIntLeaf,
         [SDNPHasChain, SDNPSideEffect]>;

def STACKRESTORE_32 :
  NVPTXInst<(outs), (ins Int32Regs:$ptr),
            "stackrestore.u32 \t$ptr;",
            [(stackrestore i32:$ptr)]>,
            Requires<[hasPTX<73>, hasSM<52>]>;

def STACKSAVE_32 :
  NVPTXInst<(outs Int32Regs:$dst), (ins),
            "stacksave.u32 \t$dst;",
            [(set i32:$dst, (i32 stacksave))]>,
            Requires<[hasPTX<73>, hasSM<52>]>;

def STACKRESTORE_64 :
  NVPTXInst<(outs), (ins Int64Regs:$ptr),
            "stackrestore.u64 \t$ptr;",
            [(stackrestore i64:$ptr)]>,
            Requires<[hasPTX<73>, hasSM<52>]>;

def STACKSAVE_64 :
  NVPTXInst<(outs Int64Regs:$dst), (ins),
            "stacksave.u64 \t$dst;",
            [(set i64:$dst, (i64 stacksave))]>,
            Requires<[hasPTX<73>, hasSM<52>]>;

include "NVPTXIntrinsics.td"

//-----------------------------------
// Notes
//-----------------------------------
// BSWAP is currently expanded. The following is a more efficient
// - for < sm_20, use vector scalar mov, as tesla support native 16-bit register
// - for sm_20, use pmpt (use vector scalar mov to get the pack and
//   unpack). sm_20 supports native 32-bit register, but not native 16-bit
// register.

def : Pat <
  (i32 (bswap i32:$a)),
  (INT_NVVM_PRMT $a, (i32 0), (i32 0x0123))>;

def : Pat <
  (v2i16 (bswap v2i16:$a)),
  (INT_NVVM_PRMT $a, (i32 0), (i32 0x2301))>;

def : Pat <
  (i64 (bswap i64:$a)),
  (V2I32toI64
    (INT_NVVM_PRMT (I64toI32H_Sink $a), (i32 0), (i32 0x0123)),
    (INT_NVVM_PRMT (I64toI32L_Sink $a), (i32 0), (i32 0x0123)))>,
  Requires<[hasPTX<71>]>;

// Fall back to the old way if we don't have PTX 7.1.
def : Pat <
  (i64 (bswap i64:$a)),
  (V2I32toI64
    (INT_NVVM_PRMT (I64toI32H $a), (i32 0), (i32 0x0123)),
    (INT_NVVM_PRMT (I64toI32L $a), (i32 0), (i32 0x0123)))>;


////////////////////////////////////////////////////////////////////////////////
// PTX Fence instructions
////////////////////////////////////////////////////////////////////////////////

class NVPTXFenceInst<string scope, string sem, Predicate ptx>:
    NVPTXInst<(outs), (ins), "fence."#sem#"."#scope#";", []>,
    Requires<[ptx, hasSM<70>]>;

foreach scope = ["sys", "gpu", "cluster", "cta"] in {
  def atomic_thread_fence_seq_cst_#scope: NVPTXFenceInst<scope, "sc", hasPTX<60>>;
  def atomic_thread_fence_acq_rel_#scope: NVPTXFenceInst<scope, "acq_rel", hasPTX<60>>;
  def atomic_thread_fence_acquire_#scope: NVPTXFenceInst<scope, "acquire", hasPTX<87>>;
  def atomic_thread_fence_release_#scope: NVPTXFenceInst<scope, "release", hasPTX<87>>;
}

def fpimm_any_zero : FPImmLeaf<fAny, [{
  return Imm.isZero();
}]>;

def fpimm_positive_zero_v2f16 : PatFrag<(ops), (v2f16 (bitconvert (i32 0)))>;
def fpimm_positive_zero_v2bf16 : PatFrag<(ops), (v2bf16 (bitconvert (i32 0)))>;

// Perform substitution if fma only has one use, and also if instruction has
// nnan instruction flag or if the TM has NoNaNsFPMath
def NVPTX_fma_oneuse_and_nnan : PatFrag<(ops node:$a, node:$b, node:$c),
                                  (fma node:$a, node:$b, node:$c), [{
  return N->hasOneUse() &&
    (N->getFlags().hasNoNaNs() || TM.Options.NoNaNsFPMath);
}]>;
// fmaxnum will differentiate between signed and unsigned zeros soon, so this
// PatFrag is for a fmaxnum node with nsz
def NVPTX_fmaxnum_nsz : PatFrag<(ops node:$a, node:$b),
                                  (fmaxnum node:$a, node:$b), [{
  return N->getFlags().hasNoSignedZeros() || TM.Options.NoSignedZerosFPMath;
}]>;

class NVPTXInst_rrr<RegisterClass RC, string Instruction, list<Predicate> Preds>
  : NVPTXInst<(outs RC:$dst), (ins RC:$a, RC:$b, RC:$c),
      !strconcat(Instruction, "\t$dst, $a, $b, $c;"), []>,
      Requires<Preds>;

def FMARELU_F16 : NVPTXInst_rrr<Int16Regs, "fma.rn.relu.f16", [useFP16Math, hasPTX<70>, hasSM<80>]>;
def FMARELU_F16_FTZ : NVPTXInst_rrr<Int16Regs, "fma.rn.ftz.relu.f16", [useFP16Math, hasPTX<70>, hasSM<80>]>;
def FMARELU_BF16 : NVPTXInst_rrr<Int16Regs, "fma.rn.relu.bf16", [hasBF16Math, hasPTX<70>, hasSM<80>]>;
def FMARELU_F16X2 : NVPTXInst_rrr<Int32Regs, "fma.rn.relu.f16x2", [useFP16Math, hasPTX<70>, hasSM<80>]>;
def FMARELU_F16X2_FTZ : NVPTXInst_rrr<Int32Regs, "fma.rn.ftz.relu.f16x2", [useFP16Math, hasPTX<70>, hasSM<80>]>;
def FMARELU_BF16X2 : NVPTXInst_rrr<Int32Regs, "fma.rn.relu.bf16x2", [hasBF16Math, hasPTX<70>, hasSM<80>]>;

// FTZ
def : Pat<(f16 (NVPTX_fmaxnum_nsz (NVPTX_fma_oneuse_and_nnan f16:$a, f16:$b, f16:$c), fpimm_any_zero)),
  (FMARELU_F16_FTZ $a, $b, $c)>,
  Requires<[doF32FTZ]>;
def : Pat<(v2f16 (NVPTX_fmaxnum_nsz (NVPTX_fma_oneuse_and_nnan v2f16:$a, v2f16:$b, v2f16:$c), fpimm_positive_zero_v2f16)),
  (FMARELU_F16X2_FTZ $a, $b, $c)>,
  Requires<[doF32FTZ]>;

// NO FTZ
def : Pat<(f16 (NVPTX_fmaxnum_nsz (NVPTX_fma_oneuse_and_nnan f16:$a, f16:$b, f16:$c), fpimm_any_zero)),
  (FMARELU_F16 $a, $b, $c)>;
def : Pat<(bf16 (NVPTX_fmaxnum_nsz (NVPTX_fma_oneuse_and_nnan bf16:$a, bf16:$b, bf16:$c), fpimm_any_zero)),
  (FMARELU_BF16 $a, $b, $c)>;
def : Pat<(v2f16 (NVPTX_fmaxnum_nsz (NVPTX_fma_oneuse_and_nnan v2f16:$a, v2f16:$b, v2f16:$c), fpimm_positive_zero_v2f16)),
  (FMARELU_F16X2 $a, $b, $c)>;
def : Pat<(v2bf16 (NVPTX_fmaxnum_nsz (NVPTX_fma_oneuse_and_nnan v2bf16:$a, v2bf16:$b, v2bf16:$c), fpimm_positive_zero_v2bf16)),
  (FMARELU_BF16X2 $a, $b, $c)>;
