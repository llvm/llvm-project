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

// FTZ flag

def FTZ : PatLeaf<(i1 1)>;
def NoFTZ : PatLeaf<(i1 0)>;

def getFTZFlag : SDNodeXForm<imm, [{
  (void)N;
  return CurDAG->getTargetConstant(useF32FTZ() ? 1 : 0, SDLoc(), MVT::i1);
}]>;

def FTZFlag : OperandWithDefaultOps<i1, (ops (getFTZFlag (i1 0)))> {
  let PrintMethod = "printFTZFlag";
}

// Compare modes
// These must match the enum in NVPTX.h
def CmpEQ : PatLeaf<(i32 0)>;
def CmpNE : PatLeaf<(i32 1)>;

def CmpMode : Operand<i32> {
  let PrintMethod = "printCmpMode";
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
def hasClusters : Predicate<"Subtarget->hasClusters()">;
def hasPTXASUnreachableBug : Predicate<"Subtarget->hasPTXASUnreachableBug()">;
def noPTXASUnreachableBug : Predicate<"!Subtarget->hasPTXASUnreachableBug()">;
def hasOptEnabled : Predicate<"TM.getOptLevel() != CodeGenOptLevel::None">;
def hasArchAccelFeatures : Predicate<"Subtarget->hasArchAccelFeatures()">;

def doF32FTZ : Predicate<"useF32FTZ()">;
def doNoF32FTZ : Predicate<"!useF32FTZ()">;
def doRsqrtOpt : Predicate<"doRsqrtOpt()">;

def hasHWROT32 : Predicate<"Subtarget->hasHWROT32()">;
def noHWROT32 : Predicate<"!Subtarget->hasHWROT32()">;
def hasDotInstructions : Predicate<"Subtarget->hasDotInstructions()">;
def hasTcgen05Instructions : Predicate<"Subtarget->hasTcgen05Instructions()">;
def hasTMACTAGroupSupport  : Predicate<"Subtarget->hasCpAsyncBulkTensorCTAGroupSupport()">;
def hasF32x2Instructions : Predicate<"Subtarget->hasF32x2Instructions()">;

class hasPTX<int version>: Predicate<"Subtarget->getPTXVersion() >= " # version>;
class hasSM<int version>: Predicate<"Subtarget->getSmVersion() >= " # version>;

// Explicit records for arch-accelerated SM versions
def hasSM90a : Predicate<"Subtarget->getSmVersion() == 90 && Subtarget->hasArchAccelFeatures()">;
def hasSM100a : Predicate<"Subtarget->getSmVersion() == 100 && Subtarget->hasArchAccelFeatures()">;
def hasSM101a : Predicate<"Subtarget->getSmVersion() == 101 && Subtarget->hasArchAccelFeatures()">;
def hasSM120a : Predicate<"Subtarget->getSmVersion() == 120 && Subtarget->hasArchAccelFeatures()">;

// non-sync shfl instructions are not available on sm_70+ in PTX6.4+
def hasSHFL : Predicate<"!(Subtarget->getSmVersion() >= 70"
                          "&& Subtarget->getPTXVersion() >= 64)">;

def useFP16Math: Predicate<"Subtarget->allowFP16Math()">;
def hasBF16Math: Predicate<"Subtarget->hasBF16Math()">;


//===----------------------------------------------------------------------===//
// Some Common Instruction Class Templates
//===----------------------------------------------------------------------===//

class OneUse1<SDPatternOperator operator>
    : PatFrag<(ops node:$A), (operator node:$A), [{ return N->hasOneUse(); }]>;
class OneUse2<SDPatternOperator operator>
    : PatFrag<(ops node:$A, node:$B), (operator node:$A, node:$B), [{ return N->hasOneUse(); }]>;


class zeroinitializer<ValueType vt> : 
  PatLeaf<(vt (bitconvert (!cast<ValueType>("i" # vt.Size) 0)))>;


def fpimm_pos_inf : FPImmLeaf<fAny, [{ return Imm.isPosInfinity(); }]>;
def fpimm_0 : FPImmLeaf<fAny, [{ return Imm.isZero(); }]>;
def fpimm_1 : FPImmLeaf<fAny, [{ return Imm.isExactlyValue(1.0); }]>;
def fpimm_neg_1 : FPImmLeaf<fAny, [{ return Imm.isExactlyValue(-1.0); }]>;


// Operands which can hold a Register or an Immediate.
//
// Unfortunately, since most register classes can hold multiple types, we must
// use the 'Any' type for these.

def RI1  : Operand<i1>;
def RI16 : Operand<Any>;
def RI32 : Operand<Any>;
def RI64 : Operand<Any>;

// Utility class to wrap up information about a register and DAG type for more
// convenient iteration and parameterization
class RegTyInfo<ValueType ty, NVPTXRegClass rc, string ptx_type, Operand imm, SDNode imm_node,
                bit supports_imm = 1> {
  ValueType Ty = ty;
  NVPTXRegClass RC = rc;
  Operand Imm = imm;
  SDNode ImmNode = imm_node;
  bit SupportsImm = supports_imm;
  int Size = ty.Size;
  string PtxType = ptx_type;
}

def I1RT     : RegTyInfo<i1,  B1,  "pred", i1imm,  imm>;
def I16RT    : RegTyInfo<i16, B16, "b16",  i16imm, imm>;
def I32RT    : RegTyInfo<i32, B32, "b32",  i32imm, imm>;
def I64RT    : RegTyInfo<i64, B64, "b64",  i64imm, imm>;

def F32RT    : RegTyInfo<f32,  B32, "f32",  f32imm,  fpimm>;
def F64RT    : RegTyInfo<f64,  B64, "f64",  f64imm,  fpimm>;
def F16RT    : RegTyInfo<f16,  B16, "f16",  f16imm,  fpimm, supports_imm = 0>;
def BF16RT   : RegTyInfo<bf16, B16, "bf16", bf16imm, fpimm, supports_imm = 0>;

def F16X2RT  : RegTyInfo<v2f16, B32, "f16x2", ?, ?, supports_imm = 0>;
def BF16X2RT : RegTyInfo<v2bf16, B32, "bf16x2", ?, ?, supports_imm = 0>;
def F32X2RT  : RegTyInfo<v2f32, B64, "f32x2", ?, ?, supports_imm = 0>;


// This class provides a basic wrapper around an NVPTXInst that abstracts the
// specific syntax of most PTX instructions. It automatically handles the
// construction of the asm string based on the provided dag arguments.
// For example, the following asm-strings would be computed:
//
//   * BasicFlagsNVPTXInst<(outs B32:$dst),
//                         (ins B32:$a, B32:$b), (ins),
//                         "add.s32">;
//         ---> "add.s32 \t$dst, $a, $b;"
//
//   * BasicFlagsNVPTXInst<(outs B32:$d),
//                         (ins B32:$a, B32:$b, Hexu32imm:$c),
//                         (ins PrmtMode:$mode),
//                         "prmt.b32${mode}">;
//         ---> "prmt.b32${mode} \t$d, $a, $b, $c;"
//
//   * BasicFlagsNVPTXInst<(outs B64:$state),
//                         (ins ADDR:$addr),
//                         "mbarrier.arrive.b64">;
//         ---> "mbarrier.arrive.b64 \t$state, [$addr];"
//
class BasicFlagsNVPTXInst<dag outs_dag, dag ins_dag, dag flags_dag, string asmstr,
                          list<dag> pattern = []>
  : NVPTXInst<
      outs_dag,
      !con(ins_dag, flags_dag),
      !strconcat(
        asmstr,
        !if(!and(!empty(ins_dag), !empty(outs_dag)), "",
          !strconcat(
            " \t",
            !interleave(
              !foreach(i, !range(!size(outs_dag)),
                "$" # !getdagname(outs_dag, i)),
              "|"),
            !if(!or(!empty(ins_dag), !empty(outs_dag)), "", ", "),
            !interleave(
              !foreach(i, !range(!size(ins_dag)),
                 !if(!eq(!cast<string>(!getdagarg<DAGOperand>(ins_dag, i)), "ADDR"),
                    "[$" # !getdagname(ins_dag, i) # "]",
                    "$" # !getdagname(ins_dag, i)
                 )
                ),
              ", "))),
        ";"),
      pattern>;

class BasicNVPTXInst<dag outs, dag insv, string asmstr, list<dag> pattern = []>
  : BasicFlagsNVPTXInst<outs, insv, (ins), asmstr, pattern>;


multiclass I3Inst<string op_str, SDPatternOperator op_node, RegTyInfo t,
                  bit commutative, list<Predicate> requires = []> {
  def rr :
    BasicNVPTXInst<(outs t.RC:$dst), (ins t.RC:$a, t.RC:$b),
              op_str,
              [(set t.Ty:$dst, (op_node t.Ty:$a, t.Ty:$b))]>,
              Requires<requires>;
  def ri :
    BasicNVPTXInst<(outs t.RC:$dst), (ins t.RC:$a, t.Imm:$b),
              op_str,
              [(set t.Ty:$dst, (op_node t.Ty:$a, (t.Ty imm:$b)))]>,
              Requires<requires>;
  if !not(commutative) then
    def ir :
      BasicNVPTXInst<(outs t.RC:$dst), (ins t.Imm:$a, t.RC:$b),
                op_str,
                [(set t.Ty:$dst, (op_node (t.Ty imm:$a), t.Ty:$b))]>,
                Requires<requires>;
}

// Template for instructions which take three int64, int32, or int16 args.
// The instructions are named "<OpcStr><Width>" (e.g. "add.s64").
multiclass I3<string op_str, SDPatternOperator op_node, bit commutative> {
  foreach t = [I16RT, I32RT, I64RT] in
    defm t.Size# : I3Inst<op_str # t.Size, op_node, t, commutative>;
}

class I16x2<string OpcStr, SDNode OpNode> :
  BasicNVPTXInst<(outs B32:$dst), (ins B32:$a, B32:$b),
              OpcStr # "16x2",
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
  defvar nan_str = !if(NaN, ".NaN", "");
  if !not(NaN) then {
   def _f64_rr :
     BasicNVPTXInst<(outs B64:$dst),
               (ins B64:$a, B64:$b),
               OpcStr # ".f64",
               [(set f64:$dst, (OpNode f64:$a, f64:$b))]>;
   def _f64_ri :
     BasicNVPTXInst<(outs B64:$dst),
               (ins B64:$a, f64imm:$b),
               OpcStr # ".f64",
               [(set f64:$dst, (OpNode f64:$a, fpimm:$b))]>;
  }
   def _f32_rr :
     BasicFlagsNVPTXInst<(outs B32:$dst),
               (ins B32:$a, B32:$b),
               (ins FTZFlag:$ftz),
               OpcStr # "$ftz" # nan_str # ".f32",
               [(set f32:$dst, (OpNode f32:$a, f32:$b))]>;
   def _f32_ri :
     BasicFlagsNVPTXInst<(outs B32:$dst),
               (ins B32:$a, f32imm:$b),
               (ins FTZFlag:$ftz),
               OpcStr # "$ftz" # nan_str # ".f32",
               [(set f32:$dst, (OpNode f32:$a, fpimm:$b))]>;

   def _f16_rr :
     BasicFlagsNVPTXInst<(outs B16:$dst),
               (ins B16:$a, B16:$b),
               (ins FTZFlag:$ftz),
               OpcStr # "$ftz" # nan_str # ".f16",
               [(set f16:$dst, (OpNode f16:$a, f16:$b))]>,
               Requires<[useFP16Math]>;

   def _f16x2_rr :
     BasicFlagsNVPTXInst<(outs B32:$dst),
               (ins B32:$a, B32:$b),
               (ins FTZFlag:$ftz),
               OpcStr # "$ftz" # nan_str # ".f16x2",
               [(set v2f16:$dst, (OpNode v2f16:$a, v2f16:$b))]>,
               Requires<[useFP16Math, hasSM<80>, hasPTX<70>]>;
   def _bf16_rr :
     BasicNVPTXInst<(outs B16:$dst),
               (ins B16:$a, B16:$b),
               OpcStr # nan_str # ".bf16",
               [(set bf16:$dst, (OpNode bf16:$a, bf16:$b))]>,
               Requires<[hasBF16Math, hasSM<80>, hasPTX<70>]>;
   def _bf16x2_rr :
     BasicNVPTXInst<(outs B32:$dst),
               (ins B32:$a, B32:$b),
               OpcStr # nan_str # ".bf16x2",
               [(set v2bf16:$dst, (OpNode v2bf16:$a, v2bf16:$b))]>,
               Requires<[hasBF16Math, hasSM<80>, hasPTX<70>]>;
}

// Template for 3-input minimum/maximum instructions
// (sm_100+/PTX 8.8 and f32 only)
//
// Also defines ftz (flush subnormal inputs and results to sign-preserving
// zero) variants for fp32 functions.
multiclass FMINIMUMMAXIMUM3<string OpcStr, bit NaN, SDNode OpNode> {
  defvar nan_str = !if(NaN, ".NaN", "");
   def f32rrr :
     BasicFlagsNVPTXInst<(outs B32:$dst),
               (ins B32:$a, B32:$b, B32:$c),
               (ins FTZFlag:$ftz),
               OpcStr # "$ftz" # nan_str # ".f32",
               [(set f32:$dst, (OpNode f32:$a, f32:$b, f32:$c))]>,
               Requires<[hasPTX<88>, hasSM<100>]>;
   def f32rri :
     BasicFlagsNVPTXInst<(outs B32:$dst),
               (ins B32:$a, B32:$b, f32imm:$c),
               (ins FTZFlag:$ftz),
               OpcStr # "$ftz" # nan_str # ".f32",
               [(set f32:$dst, (OpNode f32:$a, f32:$b, fpimm:$c))]>,
               Requires<[hasPTX<88>, hasSM<100>]>;
   def f32rii :
     BasicFlagsNVPTXInst<(outs B32:$dst),
               (ins B32:$a, f32imm:$b, f32imm:$c),
               (ins FTZFlag:$ftz),
               OpcStr # "$ftz" # nan_str # ".f32",
               [(set f32:$dst, (OpNode f32:$a, fpimm:$b, fpimm:$c))]>,
               Requires<[hasPTX<88>, hasSM<100>]>;
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
    BasicNVPTXInst<(outs B64:$dst),
              (ins B64:$a, B64:$b),
              op_str # ".f64",
              [(set f64:$dst, (op_pat f64:$a, f64:$b))]>;
  def f64ri :
    BasicNVPTXInst<(outs B64:$dst),
              (ins B64:$a, f64imm:$b),
              op_str # ".f64",
              [(set f64:$dst, (op_pat f64:$a, fpimm:$b))]>;
  def f32rr :
    BasicFlagsNVPTXInst<(outs B32:$dst),
              (ins B32:$a, B32:$b),
              (ins FTZFlag:$ftz),
              op_str # "$ftz.f32",
              [(set f32:$dst, (op_pat f32:$a, f32:$b))]>;
  def f32ri :
    BasicFlagsNVPTXInst<(outs B32:$dst),
              (ins B32:$a, f32imm:$b),
              (ins FTZFlag:$ftz),
              op_str # "$ftz.f32",
              [(set f32:$dst, (op_pat f32:$a, fpimm:$b))]>;

  def f16rr :
    BasicFlagsNVPTXInst<(outs B16:$dst),
              (ins B16:$a, B16:$b),
              (ins FTZFlag:$ftz),
              op_str # "$ftz.f16",
              [(set f16:$dst, (op_pat f16:$a, f16:$b))]>,
              Requires<[useFP16Math]>;
  def f32x2rr :
    BasicFlagsNVPTXInst<(outs B64:$dst),
              (ins B64:$a, B64:$b),
              (ins FTZFlag:$ftz),
              op_str # "$ftz.f32x2",
              [(set v2f32:$dst, (op_pat v2f32:$a, v2f32:$b))]>,
              Requires<[hasF32x2Instructions]>;
  def f16x2rr :
    BasicFlagsNVPTXInst<(outs B32:$dst),
              (ins B32:$a, B32:$b),
              (ins FTZFlag:$ftz),
              op_str # "$ftz.f16x2",
              [(set v2f16:$dst, (op_pat v2f16:$a, v2f16:$b))]>,
              Requires<[useFP16Math]>;
  def bf16rr :
    BasicNVPTXInst<(outs B16:$dst),
              (ins B16:$a, B16:$b),
              op_str # ".bf16",
              [(set bf16:$dst, (op_pat bf16:$a, bf16:$b))]>,
              Requires<[hasBF16Math]>;

  def bf16x2rr :
    BasicNVPTXInst<(outs B32:$dst),
              (ins B32:$a, B32:$b),
              op_str # ".bf16x2",
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
   def f64 : BasicNVPTXInst<(outs B64:$dst), (ins B64:$a),
                           OpcStr # ".f64",
                           [(set f64:$dst, (OpNode f64:$a))]>;
   def f32 : BasicFlagsNVPTXInst<(outs B32:$dst), (ins B32:$a),
                           (ins FTZFlag:$ftz),
                           OpcStr # "$ftz.f32",
                           [(set f32:$dst, (OpNode f32:$a))]>;
}

multiclass F2_Support_Half<string OpcStr, SDNode OpNode> {
   def bf16 :      BasicNVPTXInst<(outs B16:$dst), (ins B16:$a),
                           OpcStr # ".bf16",
                           [(set bf16:$dst, (OpNode bf16:$a))]>,
                           Requires<[hasSM<80>, hasPTX<70>]>;
   def bf16x2 :    BasicNVPTXInst<(outs B32:$dst), (ins B32:$a),
                           OpcStr # ".bf16x2",
                           [(set v2bf16:$dst, (OpNode v2bf16:$a))]>,
                           Requires<[hasSM<80>, hasPTX<70>]>;
   def f16 :       BasicFlagsNVPTXInst<(outs B16:$dst), (ins B16:$a),
                           (ins FTZFlag:$ftz),
                           OpcStr # "$ftz.f16",
                           [(set f16:$dst, (OpNode f16:$a))]>,
                           Requires<[hasSM<53>, hasPTX<65>]>;
   def f16x2 :     BasicFlagsNVPTXInst<(outs B32:$dst), (ins B32:$a),
                           (ins FTZFlag:$ftz),
                           OpcStr # "$ftz.f16x2",
                           [(set v2f16:$dst, (OpNode v2f16:$a))]>,
                           Requires<[hasSM<53>, hasPTX<65>]>;

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
    foreach sign = ["s", "u"] in {
      def _ # sign # "8" :
        BasicFlagsNVPTXInst<(outs RC:$dst),
                  (ins B16:$src), (ins CvtMode:$mode),
                  "cvt${mode:base}${mode:ftz}${mode:sat}." # ToType # "." # sign # "8">,
        Requires<Preds>;
      def _ # sign # "16" :
        BasicFlagsNVPTXInst<(outs RC:$dst),
                  (ins B16:$src), (ins CvtMode:$mode),
                  "cvt${mode:base}${mode:ftz}${mode:sat}." # ToType # "." # sign # "16">,
        Requires<Preds>;
      def _ # sign # "32" :
        BasicFlagsNVPTXInst<(outs RC:$dst),
                  (ins B32:$src), (ins CvtMode:$mode),
                  "cvt${mode:base}${mode:ftz}${mode:sat}." # ToType # "." # sign # "32">,
        Requires<Preds>;
      def _ # sign # "64" :
        BasicFlagsNVPTXInst<(outs RC:$dst),
                  (ins B64:$src), (ins CvtMode:$mode),
                  "cvt${mode:base}${mode:ftz}${mode:sat}." # ToType # "." # sign # "64">,
        Requires<Preds>;
    }
    def _f16 :
      BasicFlagsNVPTXInst<(outs RC:$dst),
                (ins B16:$src), (ins CvtMode:$mode),
                "cvt${mode:base}${mode:ftz}${mode:sat}." # ToType # ".f16">,
      Requires<Preds>;
    def _bf16 :
      BasicFlagsNVPTXInst<(outs RC:$dst),
                (ins B16:$src), (ins CvtMode:$mode),
                "cvt${mode:base}${mode:ftz}${mode:relu}${mode:sat}." # ToType # ".bf16">,
      Requires<!if(!eq(ToType, "f32"),
                   // bf16->f32 was introduced early.
                   [hasPTX<71>, hasSM<80>],
                   // bf16->everything else needs sm90/ptx78
                   [hasPTX<78>, hasSM<90>])>;
    def _f32 :
      BasicFlagsNVPTXInst<(outs RC:$dst),
                (ins B32:$src), (ins CvtMode:$mode),
                "cvt${mode:base}${mode:ftz}${mode:relu}${mode:sat}." # ToType # ".f32">,
      Requires<!if(!eq(ToType, "bf16"),
                   // f32->bf16 was introduced early.
                   [hasPTX<70>, hasSM<80>],
                   Preds)>;
    def _f64 :
      BasicFlagsNVPTXInst<(outs RC:$dst),
                (ins B64:$src), (ins CvtMode:$mode),
                "cvt${mode:base}${mode:ftz}${mode:sat}." # ToType # ".f64">,
      Requires<Preds>;
  }

  // Generate cvts from all types to all types.
  foreach sign = ["s", "u"] in {
    defm CVT_ # sign # "8"  : CVT_FROM_ALL<sign # "8",  B16>;
    defm CVT_ # sign # "16" : CVT_FROM_ALL<sign # "16", B16>;
    defm CVT_ # sign # "32" : CVT_FROM_ALL<sign # "32", B32>;
    defm CVT_ # sign # "64" : CVT_FROM_ALL<sign # "64", B64>;
  }
  defm CVT_f16 : CVT_FROM_ALL<"f16", B16>;
  defm CVT_bf16 : CVT_FROM_ALL<"bf16", B16, [hasPTX<78>, hasSM<90>]>;
  defm CVT_f32 : CVT_FROM_ALL<"f32", B32>;
  defm CVT_f64 : CVT_FROM_ALL<"f64", B64>;

  // These cvts are different from those above: The source and dest registers
  // are of the same type.
  def CVT_INREG_s16_s8  : BasicNVPTXInst<(outs B16:$dst), (ins B16:$src), "cvt.s16.s8">;
  def CVT_INREG_s32_s8  : BasicNVPTXInst<(outs B32:$dst), (ins B32:$src), "cvt.s32.s8">;
  def CVT_INREG_s32_s16 : BasicNVPTXInst<(outs B32:$dst), (ins B32:$src), "cvt.s32.s16">;
  def CVT_INREG_s64_s8  : BasicNVPTXInst<(outs B64:$dst), (ins B64:$src), "cvt.s64.s8">;
  def CVT_INREG_s64_s16 : BasicNVPTXInst<(outs B64:$dst), (ins B64:$src), "cvt.s64.s16">;
  def CVT_INREG_s64_s32 : BasicNVPTXInst<(outs B64:$dst), (ins B64:$src), "cvt.s64.s32">;

  multiclass CVT_FROM_FLOAT_V2_SM80<string FromName, RegisterClass RC> {
    def _f32 :
      BasicFlagsNVPTXInst<(outs RC:$dst),
                (ins B32:$src1, B32:$src2), (ins CvtMode:$mode),
                "cvt${mode:base}${mode:relu}." # FromName # ".f32">,
    Requires<[hasPTX<70>, hasSM<80>]>;
  }

  defm CVT_f16x2 : CVT_FROM_FLOAT_V2_SM80<"f16x2", B32>;
  defm CVT_bf16x2 : CVT_FROM_FLOAT_V2_SM80<"bf16x2", B32>;

  // FP8 conversions.
  multiclass CVT_TO_F8X2<string F8Name> {
    def _f32 :
      BasicFlagsNVPTXInst<(outs B16:$dst),
                (ins B32:$src1, B32:$src2), (ins CvtMode:$mode),
                "cvt${mode:base}.satfinite${mode:relu}." # F8Name # "x2.f32">,
      Requires<[hasPTX<81>, hasSM<89>]>;
    def _f16x2 :
      BasicFlagsNVPTXInst<(outs B16:$dst),
                (ins B32:$src), (ins CvtMode:$mode),
                "cvt${mode:base}.satfinite${mode:relu}." # F8Name # "x2.f16x2">,
      Requires<[hasPTX<81>, hasSM<89>]>;
  }

  defm CVT_e4m3x2 : CVT_TO_F8X2<"e4m3">;
  defm CVT_e5m2x2 : CVT_TO_F8X2<"e5m2">;

  class CVT_f16x2_fp8<string F8Name> :
    BasicFlagsNVPTXInst<(outs B32:$dst),
              (ins B16:$src), (ins CvtMode:$mode),
              "cvt${mode:base}${mode:relu}.f16x2." # F8Name # "x2">,
    Requires<[hasPTX<81>, hasSM<89>]>;

  def CVT_f16x2_e4m3x2 : CVT_f16x2_fp8<"e4m3">;
  def CVT_f16x2_e5m2x2 : CVT_f16x2_fp8<"e5m2">;

  // Float to TF32 conversions
  multiclass CVT_TO_TF32<string Modifier, list<Predicate> Preds = [hasPTX<78>, hasSM<90>]> {
    defvar Intr = !cast<Intrinsic>("int_nvvm_f2tf32_" # !subst(".", "_", Modifier));

    def NAME : BasicNVPTXInst<(outs B32:$dst), (ins B32:$src),
               "cvt." # Modifier # ".tf32.f32",
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
    def CVT_ # type # _f32_sf : BasicFlagsNVPTXInst<(outs B16:$dst),
                                          (ins B32:$src1, B32:$src2), (ins CvtMode:$mode),
                                          "cvt${mode:base}.satfinite${mode:relu}." # type # ".f32">;
    def CVT_f16x2_ # type : BasicFlagsNVPTXInst<(outs B32:$dst),
                                      (ins B16:$src), (ins CvtMode:$mode),
                                      "cvt${mode:base}${mode:relu}.f16x2." # type>;
  }
  
  // FP4 conversions.
  def CVT_e2m1x2_f32_sf : NVPTXInst<(outs B16:$dst),
      (ins B32:$src1, B32:$src2, CvtMode:$mode),
      !strconcat("{{ \n\t",
                 ".reg .b8 \t%e2m1x2_out; \n\t",
                 "cvt${mode:base}.satfinite${mode:relu}.e2m1x2.f32 \t%e2m1x2_out, $src1, $src2; \n\t",
                 "cvt.u16.u8 \t$dst, %e2m1x2_out; \n\t",
                 "}}"), []>;

  def CVT_f16x2_e2m1x2 : NVPTXInst<(outs B32:$dst),
      (ins B16:$src, CvtMode:$mode),
      !strconcat("{{ \n\t",
                 ".reg .b8 \t%e2m1x2_in; \n\t",
                 "cvt.u8.u16 \t%e2m1x2_in, $src; \n\t",
                 "cvt${mode:base}${mode:relu}.f16x2.e2m1x2 \t$dst, %e2m1x2_in; \n\t",
                 "}}"), []>;

  // UE8M0x2 conversions.
  class CVT_f32_to_ue8m0x2<string sat = ""> :
    BasicFlagsNVPTXInst<(outs B16:$dst),
              (ins B32:$src1, B32:$src2), (ins CvtMode:$mode),
              "cvt${mode:base}" # sat # ".ue8m0x2.f32">;
  
  class CVT_bf16x2_to_ue8m0x2<string sat = ""> :
    BasicFlagsNVPTXInst<(outs B16:$dst),
              (ins B32:$src), (ins CvtMode:$mode),
              "cvt${mode:base}" # sat # ".ue8m0x2.bf16x2">;
              
  def CVT_ue8m0x2_f32 : CVT_f32_to_ue8m0x2;
  def CVT_ue8m0x2_f32_sf : CVT_f32_to_ue8m0x2<".satfinite">;
  def CVT_ue8m0x2_bf16x2 : CVT_bf16x2_to_ue8m0x2;
  def CVT_ue8m0x2_bf16x2_sf : CVT_bf16x2_to_ue8m0x2<".satfinite">;

  def CVT_bf16x2_ue8m0x2 :
    BasicNVPTXInst<(outs B32:$dst),
                   (ins B16:$src),
                   "cvt.rn.bf16x2.ue8m0x2">;

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
    defvar asm_str = "selp." # TypeStr;
    def rr :
      BasicNVPTXInst<(outs t.RC:$dst),
                (ins t.RC:$a, t.RC:$b, B1:$p),
                asm_str,
                [(set t.Ty:$dst, (select i1:$p, t.Ty:$a, t.Ty:$b))]>;
    def ri :
      BasicNVPTXInst<(outs t.RC:$dst),
                (ins t.RC:$a, t.Imm:$b, B1:$p),
                asm_str,
                [(set t.Ty:$dst, (select i1:$p, t.Ty:$a, t.ImmNode:$b))]>;
    def ir :
      BasicNVPTXInst<(outs t.RC:$dst),
                (ins t.Imm:$a, t.RC:$b, B1:$p),
                asm_str,
                [(set t.Ty:$dst, (select i1:$p, t.ImmNode:$a, t.Ty:$b))]>;
    def ii :
      BasicNVPTXInst<(outs t.RC:$dst),
                (ins t.Imm:$a, t.Imm:$b, B1:$p),
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
// defm SELP_f16x2 : SELP_PATTERN<"b32", v2f16, B32, v2f16imm, imm>;

foreach vt = [v2f16, v2bf16, v2i16, v4i8] in {
def : Pat<(vt (select i1:$p, vt:$a, vt:$b)),
          (SELP_b32rr $a, $b, $p)>;
}

def : Pat<(v2f32 (select i1:$p, v2f32:$a, v2f32:$b)),
          (SELP_b64rr $a, $b, $p)>;

//-----------------------------------
// Test Instructions
//-----------------------------------

def fabs_oneuse : OneUse1<fabs>;

def TESTINF_f32r : BasicNVPTXInst<(outs B1:$p), (ins B32:$a),
                             "testp.infinite.f32",
                             [(set i1:$p, (seteq (fabs_oneuse f32:$a), fpimm_pos_inf))]>;
def TESTINF_f64r : BasicNVPTXInst<(outs B1:$p), (ins B64:$a),
                             "testp.infinite.f64",
                             [(set i1:$p, (seteq (fabs_oneuse f64:$a), fpimm_pos_inf))]>;

//-----------------------------------
// Integer Arithmetic
//-----------------------------------

// int16, int32, and int64 signed addition.  Since nvptx is 2's complement, we
// also use these for unsigned arithmetic.
defm ADD : I3<"add.s", add, commutative = true>;
defm SUB : I3<"sub.s", sub, commutative = false>;

def ADD16x2 : I16x2<"add.s", add>;

// int32 and int64 addition and subtraction with carry-out.
defm ADDCC : ADD_SUB_INT_CARRY<"add.cc", addc, commutative = true>;
defm SUBCC : ADD_SUB_INT_CARRY<"sub.cc", subc, commutative = false>;

// int32 and int64 addition and subtraction with carry-in and carry-out.
defm ADDCCC : ADD_SUB_INT_CARRY<"addc.cc", adde, commutative = true>;
defm SUBCCC : ADD_SUB_INT_CARRY<"subc.cc", sube, commutative = false>;

defm MULT : I3<"mul.lo.s", mul, commutative = true>;

defm MUL_HI_S : I3<"mul.hi.s", mulhs, commutative = true>;
defm MUL_HI_U : I3<"mul.hi.u", mulhu, commutative = true>;

defm SDIV : I3<"div.s", sdiv, commutative = false>;
defm UDIV : I3<"div.u", udiv, commutative = false>;

// The ri versions of rem.s and rem.u won't be selected; DAGCombiner::visitSREM
// will lower it.
defm SREM : I3<"rem.s", srem, commutative = false>;
defm UREM : I3<"rem.u", urem, commutative = false>;

foreach t = [I16RT, I32RT, I64RT] in {
  def ABS_S # t.Size :
    BasicNVPTXInst<(outs t.RC:$dst), (ins t.RC:$a),
                   "abs.s" # t.Size,
                   [(set t.Ty:$dst, (abs t.Ty:$a))]>;

  def NEG_S # t.Size :
    BasicNVPTXInst<(outs t.RC:$dst), (ins t.RC:$src),
                   "neg.s" # t.Size,
                   [(set t.Ty:$dst, (ineg t.Ty:$src))]>;
}

// Integer min/max.
defm SMAX : I3<"max.s", smax, commutative = true>;
defm UMAX : I3<"max.u", umax, commutative = true>;
defm SMIN : I3<"min.s", smin, commutative = true>;
defm UMIN : I3<"min.u", umin, commutative = true>;

def SMAX16x2 : I16x2<"max.s", smax>;
def UMAX16x2 : I16x2<"max.u", umax>;
def SMIN16x2 : I16x2<"min.s", smin>;
def UMIN16x2 : I16x2<"min.u", umin>;

let Predicates = [hasPTX<80>, hasSM<90>] in {

  def MIN_RELU_S32 : BasicNVPTXInst<(outs B32:$dst), (ins B32:$a, B32:$b),
                     "min.relu.s32",
                     [(set i32:$dst, (smax (smin i32:$a, i32:$b), 0))]>;
  def MAX_RELU_S32 : BasicNVPTXInst<(outs B32:$dst), (ins B32:$a, B32:$b),
                     "max.relu.s32",
                     [(set i32:$dst, (smax (smax i32:$a, i32:$b), 0))]>;
  def MIN_RELU_S16x2 : BasicNVPTXInst<(outs B32:$dst), (ins B32:$a, B32:$b),
                     "min.relu.s16x2",
                     [(set v2i16:$dst, (smax (smin v2i16:$a, v2i16:$b),
                                             zeroinitializer<v2i16>))]>;
  def MAX_RELU_S16x2 : BasicNVPTXInst<(outs B32:$dst), (ins B32:$a, B32:$b),
                     "max.relu.s16x2",
                     [(set v2i16:$dst, (smax (smax v2i16:$a, v2i16:$b),
                                             zeroinitializer<v2i16>))]>;
}

//
// Wide multiplication
//

def SDTMulWide : SDTypeProfile<1, 2, [SDTCisInt<0>, SDTCisInt<1>, SDTCisSameAs<1, 2>]>;
def smul_wide : SDNode<"NVPTXISD::MUL_WIDE_SIGNED", SDTMulWide, [SDNPCommutative]>;
def umul_wide : SDNode<"NVPTXISD::MUL_WIDE_UNSIGNED", SDTMulWide, [SDNPCommutative]>;


multiclass MULWIDEInst<string suffix, SDPatternOperator op, RegTyInfo big_t, RegTyInfo small_t> {
  def suffix # _rr :
    BasicNVPTXInst<(outs big_t.RC:$dst), (ins small_t.RC:$a, small_t.RC:$b), 
                   "mul.wide." # suffix,
                   [(set big_t.Ty:$dst, (op small_t.Ty:$a, small_t.Ty:$b))]>;
  def suffix # _ri :
    BasicNVPTXInst<(outs big_t.RC:$dst), (ins small_t.RC:$a, small_t.Imm:$b), 
                   "mul.wide." # suffix,
                   [(set big_t.Ty:$dst, (op small_t.Ty:$a, imm:$b))]>;
}

defm MUL_WIDE : MULWIDEInst<"s32", smul_wide, I64RT, I32RT>;
defm MUL_WIDE : MULWIDEInst<"u32", umul_wide, I64RT, I32RT>;
defm MUL_WIDE : MULWIDEInst<"s16", smul_wide, I32RT, I16RT>;
defm MUL_WIDE : MULWIDEInst<"u16", umul_wide, I32RT, I16RT>;

//
// Integer multiply-add
//
multiclass MADInst<string suffix, SDPatternOperator op, RegTyInfo big_t, RegTyInfo small_t> {
  def rrr:
    BasicNVPTXInst<(outs big_t.RC:$dst),
              (ins small_t.RC:$a, small_t.RC:$b, big_t.RC:$c),
              "mad." # suffix,
              [(set big_t.Ty:$dst, (add (OneUse2<op> small_t.Ty:$a, small_t.Ty:$b), big_t.Ty:$c))]>;
  def rri:
    BasicNVPTXInst<(outs big_t.RC:$dst),
              (ins small_t.RC:$a, small_t.RC:$b, big_t.Imm:$c),
              "mad." # suffix,
              [(set big_t.Ty:$dst, (add (OneUse2<op> small_t.Ty:$a, small_t.Ty:$b), imm:$c))]>;
  def rir:
    BasicNVPTXInst<(outs big_t.RC:$dst),
              (ins small_t.RC:$a, small_t.Imm:$b, big_t.RC:$c),
              "mad." # suffix,
              [(set big_t.Ty:$dst, (add (OneUse2<op> small_t.Ty:$a, imm:$b), big_t.Ty:$c))]>;
  def rii:
    BasicNVPTXInst<(outs big_t.RC:$dst),
              (ins small_t.RC:$a, small_t.Imm:$b, big_t.Imm:$c),
              "mad." # suffix,
              [(set big_t.Ty:$dst, (add (OneUse2<op> small_t.Ty:$a, imm:$b), imm:$c))]>;
}

let Predicates = [hasOptEnabled] in {
  defm MAD_LO_S16 : MADInst<"lo.s16", mul, I16RT, I16RT>;
  defm MAD_LO_S32 : MADInst<"lo.s32", mul, I32RT, I32RT>;
  defm MAD_LO_S64 : MADInst<"lo.s64", mul, I64RT, I64RT>;

  defm MAD_WIDE_U16 : MADInst<"wide.u16", umul_wide, I32RT, I16RT>;
  defm MAD_WIDE_S16 : MADInst<"wide.s16", smul_wide, I32RT, I16RT>;
  defm MAD_WIDE_U32 : MADInst<"wide.u32", umul_wide, I64RT, I32RT>;
  defm MAD_WIDE_S32 : MADInst<"wide.s32", smul_wide, I64RT, I32RT>;
}

//-----------------------------------
// Floating Point Arithmetic
//-----------------------------------

defm FADD : F3_fma_component<"add", fadd>;
defm FSUB : F3_fma_component<"sub", fsub>;
defm FMUL : F3_fma_component<"mul", fmul>;

defm MIN : FMINIMUMMAXIMUM<"min", /* NaN */ false, fminnum>;
defm MAX : FMINIMUMMAXIMUM<"max", /* NaN */ false, fmaxnum>;
defm MIN_NAN : FMINIMUMMAXIMUM<"min", /* NaN */ true, fminimum>;
defm MAX_NAN : FMINIMUMMAXIMUM<"max", /* NaN */ true, fmaximum>;

def nvptx_fminnum3 : SDNode<"NVPTXISD::FMINNUM3", SDTFPTernaryOp,
                            [SDNPCommutative]>;
def nvptx_fmaxnum3 : SDNode<"NVPTXISD::FMAXNUM3", SDTFPTernaryOp,
                             [SDNPCommutative]>;
def nvptx_fminimum3 : SDNode<"NVPTXISD::FMINIMUM3", SDTFPTernaryOp,
                             [SDNPCommutative]>;
def nvptx_fmaximum3 : SDNode<"NVPTXISD::FMAXIMUM3", SDTFPTernaryOp,
                             [SDNPCommutative]>;

defm FMIN3 : FMINIMUMMAXIMUM3<"min", /* NaN */ false, nvptx_fminnum3>;
defm FMAX3 : FMINIMUMMAXIMUM3<"max", /* NaN */ false, nvptx_fmaxnum3>;
defm FMINNAN3 : FMINIMUMMAXIMUM3<"min", /* NaN */ true, nvptx_fminimum3>;
defm FMAXNAN3 : FMINIMUMMAXIMUM3<"max", /* NaN */ true, nvptx_fmaximum3>;

defm FABS  : F2<"abs", fabs>;
defm FNEG  : F2<"neg", fneg>;
defm FABS_H: F2_Support_Half<"abs", fabs>;
defm FNEG_H: F2_Support_Half<"neg", fneg>;

defm FSQRT : F2<"sqrt.rn", fsqrt>;

//
// F16 NEG
//
class FNEG16<RegTyInfo t> :
      BasicFlagsNVPTXInst<(outs t.RC:$dst), (ins t.RC:$src), (ins FTZFlag:$ftz),
                "neg$ftz." # t.PtxType,
                [(set t.Ty:$dst, (fneg t.Ty:$src))]>;

let Predicates = [useFP16Math, hasPTX<60>, hasSM<53>] in {
  def NEG_F16    : FNEG16<F16RT>;
  def NEG_F16x2  : FNEG16<F16X2RT>;
}
let Predicates = [hasBF16Math, hasPTX<70>, hasSM<80>] in {
  def NEG_BF16   : FNEG16<BF16RT>;
  def NEG_BF16x2 : FNEG16<BF16X2RT>;
}

//
// EX2
//

class FEXP2Inst<RegTyInfo t, dag flags, string flag_str> :
      BasicFlagsNVPTXInst<(outs t.RC:$dst), (ins t.RC:$src),
                flags, "ex2.approx" # flag_str # "." # t.PtxType,
                [(set t.Ty:$dst, (fexp2 t.Ty:$src))]>;

def EX2_APPROX_f32 : FEXP2Inst<F32RT, (ins FTZFlag:$ftz), "$ftz">;

let Predicates = [useFP16Math, hasPTX<70>, hasSM<75>] in {
  def EX2_APPROX_f16 : FEXP2Inst<F16RT, (ins), "">;
  def EX2_APPROX_f16x2 : FEXP2Inst<F16X2RT, (ins), "">;
}
let Predicates = [hasPTX<78>, hasSM<90>] in {
  def EX2_APPROX_bf16 : FEXP2Inst<BF16RT, (ins), ".ftz">;
  def EX2_APPROX_bf16x2 : FEXP2Inst<BF16X2RT, (ins), ".ftz">;
}

// F64 division
//
def FRCP64r :
  BasicNVPTXInst<(outs B64:$dst),
                 (ins B64:$b),
                 "rcp.rn.f64",
                 [(set f64:$dst, (fdiv fpimm_1, f64:$b))]>;
def FDIV64rr :
  BasicNVPTXInst<(outs B64:$dst),
                 (ins B64:$a, B64:$b),
                 "div.rn.f64",
                 [(set f64:$dst, (fdiv f64:$a, f64:$b))]>;
def FDIV64ri :
  BasicNVPTXInst<(outs B64:$dst),
                 (ins B64:$a, f64imm:$b),
                 "div.rn.f64",
                 [(set f64:$dst, (fdiv f64:$a, fpimm:$b))]>;

// fdiv will be converted to rcp
// fneg (fdiv 1.0, X) => fneg (rcp.rn X)
def : Pat<(fdiv fpimm_neg_1, f64:$b),
          (FNEGf64 (FRCP64r $b))>;

//
// F32 Approximate reciprocal
//

def fdiv_approx : PatFrag<(ops node:$a, node:$b),
                          (fdiv node:$a, node:$b), [{
  return getDivF32Level(N) == NVPTX::DivPrecisionLevel::Approx;
}]>;


def RCP_APPROX_F32_r :
  BasicFlagsNVPTXInst<(outs B32:$dst),
                 (ins B32:$b), (ins FTZFlag:$ftz),
                 "rcp.approx$ftz.f32",
                 [(set f32:$dst, (fdiv_approx fpimm_1, f32:$b))]>;

//
// F32 Approximate division
//
def DIV_APPROX_F32_rr :
  BasicFlagsNVPTXInst<(outs B32:$dst),
                 (ins B32:$a, B32:$b), (ins FTZFlag:$ftz),
                 "div.approx$ftz.f32",
                 [(set f32:$dst, (fdiv_approx f32:$a, f32:$b))]>;
def DIV_APPROX_F32_ri :
  BasicFlagsNVPTXInst<(outs B32:$dst),
                 (ins B32:$a, f32imm:$b), (ins FTZFlag:$ftz),
                 "div.approx$ftz.f32",
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


def : Pat<(fdiv_full fpimm_1, f32:$b),
          (RCP_APPROX_F32_r $b)>;

//
// F32 Semi-accurate division
//
def FDIV32rr :
  BasicFlagsNVPTXInst<(outs B32:$dst),
                 (ins B32:$a, B32:$b), (ins FTZFlag:$ftz),
                 "div.full$ftz.f32",
                 [(set f32:$dst, (fdiv_full f32:$a, f32:$b))]>;
def FDIV32ri :
  BasicFlagsNVPTXInst<(outs B32:$dst),
                 (ins B32:$a, f32imm:$b), (ins FTZFlag:$ftz),
                 "div.full$ftz.f32",
                 [(set f32:$dst, (fdiv_full f32:$a, fpimm:$b))]>;
//
// F32 Accurate reciprocal
//

def fdiv_ftz : PatFrag<(ops node:$a, node:$b),
                       (fdiv node:$a, node:$b), [{
  return getDivF32Level(N) == NVPTX::DivPrecisionLevel::IEEE754;
}]>;

def FRCP32r_prec :
  BasicFlagsNVPTXInst<(outs B32:$dst),
                 (ins B32:$b), (ins FTZFlag:$ftz),
                 "rcp.rn$ftz.f32",
                 [(set f32:$dst, (fdiv_ftz fpimm_1, f32:$b))]>;
//
// F32 Accurate division
//
def FDIV32rr_prec :
  BasicFlagsNVPTXInst<(outs B32:$dst),
                 (ins B32:$a, B32:$b), (ins FTZFlag:$ftz),
                 "div.rn$ftz.f32",
                 [(set f32:$dst, (fdiv_ftz f32:$a, f32:$b))]>;
def FDIV32ri_prec :
  BasicFlagsNVPTXInst<(outs B32:$dst),
                 (ins B32:$a, f32imm:$b), (ins FTZFlag:$ftz),
                 "div.rn$ftz.f32",
                 [(set f32:$dst, (fdiv_ftz f32:$a, fpimm:$b))]>;

def : Pat<(fdiv fpimm_1, f32:$b), (FRCP32r_prec $b, NoFTZ)>;
def : Pat<(fdiv f32:$a, f32:$b), (FDIV32rr_prec $a, $b, NoFTZ)>;
def : Pat<(fdiv f32:$a, fpimm:$b), (FDIV32ri_prec $a, fpimm:$b, NoFTZ)>;

//
// FMA
//

multiclass FMA<RegTyInfo t, bit allow_ftz = true, list<Predicate> preds = []> {
  defvar flag_str = !if(allow_ftz, "$ftz", "");
  defvar flag_ops = !if(allow_ftz, (ins FTZFlag:$ftz), (ins));
  defvar op_str = "fma.rn" # flag_str # "." # t.PtxType;

  let Predicates = preds in {
    def rrr : BasicFlagsNVPTXInst<(outs t.RC:$dst), (ins t.RC:$a, t.RC:$b, t.RC:$c),
                        flag_ops, op_str,
                        [(set t.Ty:$dst, (fma t.Ty:$a, t.Ty:$b, t.Ty:$c))]>;

    if t.SupportsImm then {
      def rri : BasicFlagsNVPTXInst<(outs t.RC:$dst),
                          (ins t.RC:$a, t.RC:$b, t.Imm:$c),
                          flag_ops, op_str,
                          [(set t.Ty:$dst, (fma t.Ty:$a, t.Ty:$b, fpimm:$c))]>;
      def rir : BasicFlagsNVPTXInst<(outs t.RC:$dst),
                          (ins t.RC:$a, t.Imm:$b, t.RC:$c),
                          flag_ops, op_str,
                          [(set t.Ty:$dst, (fma t.Ty:$a, fpimm:$b, t.Ty:$c))]>;
      def rii : BasicFlagsNVPTXInst<(outs t.RC:$dst),
                          (ins t.RC:$a, t.Imm:$b, t.Imm:$c),
                          flag_ops, op_str,
                          [(set t.Ty:$dst, (fma t.Ty:$a, fpimm:$b, fpimm:$c))]>;
      def iir : BasicFlagsNVPTXInst<(outs t.RC:$dst),
                          (ins t.Imm:$a, t.Imm:$b, t.RC:$c),
                          flag_ops, op_str,
                          [(set t.Ty:$dst, (fma fpimm:$a, fpimm:$b, t.Ty:$c))]>;
    }
  }
}

defm FMA_F16    : FMA<F16RT,    allow_ftz = true, preds = [useFP16Math]>;
defm FMA_F16x2  : FMA<F16X2RT,  allow_ftz = true, preds = [useFP16Math]>;
defm FMA_BF16   : FMA<BF16RT,   allow_ftz = false, preds = [hasBF16Math]>;
defm FMA_BF16x2 : FMA<BF16X2RT, allow_ftz = false, preds = [hasBF16Math]>;
defm FMA_F32    : FMA<F32RT,    allow_ftz = true>;
defm FMA_F32x2  : FMA<F32X2RT,  allow_ftz = true, preds = [hasF32x2Instructions]>;
defm FMA_F64    : FMA<F64RT,    allow_ftz = false>;

// sin/cos/tanh

class UnaryOpAllowsApproxFn<SDPatternOperator operator>
    : PatFrag<(ops node:$A), (operator node:$A), [{
  return N->getFlags().hasApproximateFuncs();
}]>;

def SIN_APPROX_f32 :
  BasicFlagsNVPTXInst<(outs B32:$dst), (ins B32:$src), (ins FTZFlag:$ftz),
                      "sin.approx$ftz.f32",
                      [(set f32:$dst, (UnaryOpAllowsApproxFn<fsin> f32:$src))]>;
def COS_APPROX_f32 :
  BasicFlagsNVPTXInst<(outs B32:$dst), (ins B32:$src), (ins FTZFlag:$ftz),
                      "cos.approx$ftz.f32",
                      [(set f32:$dst, (UnaryOpAllowsApproxFn<fcos> f32:$src))]>;
def TANH_APPROX_f32 :
  BasicNVPTXInst<(outs B32:$dst), (ins B32:$src), "tanh.approx.f32",
                 [(set f32:$dst, (UnaryOpAllowsApproxFn<ftanh> f32:$src))]>,
                 Requires<[hasPTX<70>, hasSM<75>]>;

//-----------------------------------
// Bitwise operations
//-----------------------------------

// Template for three-arg bitwise operations.  Takes three args, Creates .b16,
// .b32, .b64, and .pred (predicate registers -- i.e., i1) versions of OpcStr.
multiclass BITWISE<string OpcStr, SDNode OpNode> {
  foreach t = [I1RT, I16RT, I32RT, I64RT] in
    defm _ # t.PtxType : I3Inst<OpcStr # "." # t.PtxType, OpNode, t, commutative = true>;
}

defm OR  : BITWISE<"or", or>;
defm AND : BITWISE<"and", and>;
defm XOR : BITWISE<"xor", xor>;

// PTX does not support mul on predicates, convert to and instructions
def : Pat<(mul i1:$a, i1:$b), (AND_predrr $a, $b)>;
def : Pat<(mul i1:$a, imm:$b), (AND_predri $a, imm:$b)>;

foreach op = [add, sub] in {
  def : Pat<(op i1:$a, i1:$b), (XOR_predrr $a, $b)>;
  def : Pat<(op i1:$a, imm:$b), (XOR_predri $a, imm:$b)>;
}

// These transformations were once reliably performed by instcombine, but thanks
// to poison semantics they are no longer safe for LLVM IR, perform them here
// instead.
def : Pat<(select i1:$a, i1:$b, 0), (AND_predrr $a, $b)>;
def : Pat<(select i1:$a, 1, i1:$b), (OR_predrr $a, $b)>;

// Lower logical v2i16/v4i8 ops as bitwise ops on b32.
foreach vt = [v2i16, v4i8] in {
  def : Pat<(or vt:$a, vt:$b), (OR_b32rr $a, $b)>;
  def : Pat<(xor vt:$a, vt:$b), (XOR_b32rr $a, $b)>;
  def : Pat<(and vt:$a, vt:$b), (AND_b32rr $a, $b)>;

  // The constants get legalized into a bitcast from i32, so that's what we need
  // to match here.
  def: Pat<(or vt:$a, (vt (bitconvert (i32 imm:$b)))),
           (OR_b32ri $a, imm:$b)>;
  def: Pat<(xor vt:$a, (vt (bitconvert (i32 imm:$b)))),
           (XOR_b32ri $a, imm:$b)>;
  def: Pat<(and vt:$a, (vt (bitconvert (i32 imm:$b)))),
           (AND_b32ri $a, imm:$b)>;
}

foreach t = [I1RT, I16RT, I32RT, I64RT] in
  def NOT_ # t.PtxType : BasicNVPTXInst<(outs t.RC:$dst), (ins t.RC:$src),
                                        "not." # t.PtxType,
                                        [(set t.Ty:$dst, (not t.Ty:$src))]>;

// Template for left/right shifts.  Takes three operands,
//   [dest (reg), src (reg), shift (reg or imm)].
// dest and src may be int64, int32, or int16, but shift is always int32.
//
// This template also defines a 32-bit shift (imm, imm) instruction.
multiclass SHIFT<string OpcStr, SDNode OpNode> {
  let hasSideEffects = false in {
    foreach t = [I64RT, I32RT, I16RT] in {
      def t.Size # _rr :
        BasicNVPTXInst<(outs t.RC:$dst), (ins t.RC:$a, B32:$b),
                  OpcStr # t.Size,
                  [(set t.Ty:$dst, (OpNode t.Ty:$a, i32:$b))]>;
      def t.Size # _ri :
        BasicNVPTXInst<(outs t.RC:$dst), (ins t.RC:$a, i32imm:$b),
                  OpcStr # t.Size,
                  [(set t.Ty:$dst, (OpNode t.Ty:$a, (i32 imm:$b)))]>;
      def t.Size # _ii :
        BasicNVPTXInst<(outs t.RC:$dst), (ins t.RC:$a, i32imm:$b),
                  OpcStr # t.Size,
                  [(set t.Ty:$dst, (OpNode (t.Ty imm:$a), (i32 imm:$b)))]>;
    }
  }
}

defm SHL : SHIFT<"shl.b", shl>;
defm SRA : SHIFT<"shr.s", sra>;
defm SRL : SHIFT<"shr.u", srl>;

// Bit-reverse
foreach t = [I64RT, I32RT] in
  def BREV_ # t.PtxType :
    BasicNVPTXInst<(outs t.RC:$dst), (ins t.RC:$a),
               "brev." # t.PtxType,
               [(set t.Ty:$dst, (bitreverse t.Ty:$a))]>;


//
// BFE - bit-field extract
//

// Template for BFE/BFI instructions.
// Args: [dest (reg), src (reg), start (reg or imm), end (reg or imm)].
// Start may be an imm only if end is also an imm.  FIXME: Is this a
// restriction in PTX?
//
// dest and src may be int32 or int64, but start and end are always int32.
def SDTBFI :
  SDTypeProfile<1, 4, [SDTCisInt<0>, SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, 
                       SDTCisVT<3, i32>, SDTCisVT<4, i32>]>;
def bfi : SDNode<"NVPTXISD::BFI", SDTBFI>;

def SDTPRMT :
  SDTypeProfile<1, 4, [SDTCisVT<0, i32>, SDTCisVT<1, i32>,
                       SDTCisVT<2, i32>, SDTCisVT<3, i32>, SDTCisVT<4, i32>]>;
def prmt : SDNode<"NVPTXISD::PRMT", SDTPRMT>;

multiclass BFE<string Instr, RegisterClass RC> {
  def rrr
    : BasicNVPTXInst<(outs RC:$d), (ins RC:$a, B32:$b, B32:$c), Instr>;
  def rri
    : BasicNVPTXInst<(outs RC:$d), (ins RC:$a, B32:$b, i32imm:$c), Instr>;
  def rii
    : BasicNVPTXInst<(outs RC:$d), (ins RC:$a, i32imm:$b, i32imm:$c), Instr>;
}

multiclass BFI<string Instr, ValueType T, RegisterClass RC, Operand ImmCls> {
  def rrrr
    : BasicNVPTXInst<(outs RC:$f),
                (ins RC:$a, RC:$b, B32:$c, B32:$d),
                Instr,
                [(set T:$f, (bfi T:$a, T:$b, i32:$c, i32:$d))]>;
  def rrri
    : BasicNVPTXInst<(outs RC:$f),
                (ins RC:$a, RC:$b, B32:$c, i32imm:$d),
                Instr,
                [(set T:$f, (bfi T:$a, T:$b, i32:$c, imm:$d))]>;
  def rrii
    : BasicNVPTXInst<(outs RC:$f),
                (ins RC:$a, RC:$b, i32imm:$c, i32imm:$d),
                Instr,
                [(set T:$f, (bfi T:$a, T:$b, imm:$c, imm:$d))]>;
  def irrr
    : BasicNVPTXInst<(outs RC:$f),
                (ins ImmCls:$a, RC:$b, B32:$c, B32:$d),
                Instr,
                [(set T:$f, (bfi (T imm:$a), T:$b, i32:$c, i32:$d))]>;
  def irri
    : BasicNVPTXInst<(outs RC:$f),
                (ins ImmCls:$a, RC:$b, B32:$c, i32imm:$d),
                Instr,
                [(set T:$f, (bfi (T imm:$a), T:$b, i32:$c, imm:$d))]>;
  def irii
    : BasicNVPTXInst<(outs RC:$f),
                (ins ImmCls:$a, RC:$b, i32imm:$c, i32imm:$d),
                Instr,
                [(set T:$f, (bfi (T imm:$a), T:$b, imm:$c, imm:$d))]>;
}

def Hexu32imm : Operand<i32> {
  let PrintMethod = "printHexu32imm";
}

let hasSideEffects = false in {
  // order is somewhat important here. signed/unsigned variants match
  // the same patterns, so the first one wins. Having unsigned byte extraction
  // has the benefit of always having zero in unused bits, which makes some
  // optimizations easier (e.g. no need to mask them).
  defm BFE_U32 : BFE<"bfe.u32", B32>;
  defm BFE_S32 : BFE<"bfe.s32", B32>;
  defm BFE_U64 : BFE<"bfe.u64", B64>;
  defm BFE_S64 : BFE<"bfe.s64", B64>;

  defm BFI_B32 : BFI<"bfi.b32", i32, B32, i32imm>;
  defm BFI_B64 : BFI<"bfi.b64", i64, B64, i64imm>;

  def PRMT_B32rrr
    : BasicFlagsNVPTXInst<(outs B32:$d),
                (ins B32:$a, B32:$b, B32:$c),
                (ins PrmtMode:$mode),
                "prmt.b32$mode",
                [(set i32:$d, (prmt i32:$a, i32:$b, i32:$c, imm:$mode))]>;
  def PRMT_B32rri
    : BasicFlagsNVPTXInst<(outs B32:$d),
                (ins B32:$a, B32:$b, Hexu32imm:$c),
                (ins PrmtMode:$mode),
                "prmt.b32$mode",
                [(set i32:$d, (prmt i32:$a, i32:$b, imm:$c, imm:$mode))]>;
  def PRMT_B32rir
  : BasicFlagsNVPTXInst<(outs B32:$d),
              (ins B32:$a, i32imm:$b, B32:$c),
              (ins PrmtMode:$mode),
              "prmt.b32$mode",
              [(set i32:$d, (prmt i32:$a, imm:$b, i32:$c, imm:$mode))]>;
  def PRMT_B32rii
    : BasicFlagsNVPTXInst<(outs B32:$d),
                (ins B32:$a, i32imm:$b, Hexu32imm:$c),
                (ins PrmtMode:$mode),
                "prmt.b32$mode",
                [(set i32:$d, (prmt i32:$a, imm:$b, imm:$c, imm:$mode))]>;
  def PRMT_B32irr
    : BasicFlagsNVPTXInst<(outs B32:$d),
                (ins i32imm:$a, B32:$b, B32:$c), (ins PrmtMode:$mode),
                "prmt.b32$mode",
                [(set i32:$d, (prmt imm:$a, i32:$b, i32:$c, imm:$mode))]>;
  def PRMT_B32iri
    : BasicFlagsNVPTXInst<(outs B32:$d),
                (ins i32imm:$a, B32:$b, Hexu32imm:$c), (ins PrmtMode:$mode),
                "prmt.b32$mode",
                [(set i32:$d, (prmt imm:$a, i32:$b, imm:$c, imm:$mode))]>;
  def PRMT_B32iir
    : BasicFlagsNVPTXInst<(outs B32:$d),
                (ins i32imm:$a, i32imm:$b, B32:$c), (ins PrmtMode:$mode),
                "prmt.b32$mode",
                [(set i32:$d, (prmt imm:$a, imm:$b, i32:$c, imm:$mode))]>;

}

// PRMT folding patterns
def : Pat<(fshr i32:$hi, i32:$lo, (shl i32:$amt, (i32 3))),
          (PRMT_B32rrr $lo, $hi, $amt, PrmtF4E)>;


def byte_extract_prmt : ImmLeaf<i32, [{
  return (Imm == 0x7770) || (Imm == 0x7771) || (Imm == 0x7772) || (Imm == 0x7773);
}]>;

def to_sign_extend_selector : SDNodeXForm<imm, [{
  const APInt &V = N->getAPIntValue();
  const APInt B = V.trunc(4);
  const APInt BSext = B | 8;
  const APInt R = BSext.concat(BSext).concat(BSext).concat(B).zext(32);
  return CurDAG->getTargetConstant(R, SDLoc(N), MVT::i32);
}]>;


// byte extraction + signed/unsigned extension to i32.
def : Pat<(i32 (sext_inreg (prmt i32:$s, 0, byte_extract_prmt:$sel, PrmtNONE), i8)),
          (PRMT_B32rii $s, 0, (to_sign_extend_selector $sel), PrmtNONE)>;

// byte extraction + signed extension to i16
def : Pat<(i16 (sext_inreg (trunc (prmt i32:$s, 0, byte_extract_prmt:$sel, PrmtNONE)), i8)),
          (CVT_u16_u32 (PRMT_B32rii $s, 0, (to_sign_extend_selector $sel), PrmtNONE), CvtNONE)>;


// Byte extraction via shift/trunc/sext
def : Pat<(i16 (sext_inreg (trunc i32:$s), i8)), (CVT_s8_s32 $s, CvtNONE)>;
def : Pat<(i16 (sext_inreg (trunc i64:$s), i8)), (CVT_s8_s64 $s, CvtNONE)>;

def : Pat<(sext_inreg (srl i32:$s, (i32 imm:$o)), i8), (BFE_S32rii $s, imm:$o, 8)>;
def : Pat<(sext_inreg (srl i64:$s, (i32 imm:$o)), i8), (BFE_S64rii $s, imm:$o, 8)>;

def : Pat<(i16 (sext_inreg (trunc (srl i32:$s, (i32 imm:$o))), i8)),
          (CVT_s8_s32 (BFE_S32rii $s, imm:$o, 8), CvtNONE)>;
def : Pat<(i16 (sext_inreg (trunc (srl i64:$s, (i32 imm:$o))), i8)),
          (CVT_s8_s64 (BFE_S64rii $s, imm:$o, 8), CvtNONE)>;

def : Pat<(i16 (sra (i16 (trunc i32:$s)), (i32 8))),
          (CVT_s8_s32 (BFE_S32rii $s, 8, 8), CvtNONE)>;

//-----------------------------------
// Comparison instructions (setp, set)
//-----------------------------------

// FIXME: This doesn't cover versions of set and setp that combine with a
// boolean predicate, e.g. setp.eq.and.b16.
def cond2cc : SDNodeXForm<cond, [{
  return getPTXCmpMode(*N);
}]>;

multiclass FSETP<RegTyInfo t, bit allow_ftz = true> {
  defvar ftz_str = !if(allow_ftz, "$ftz", "");
  defvar op_str = "setp.${cmp:FCmp}" # ftz_str # "." # t.PtxType;
  defvar flags = !con((ins CmpMode:$cmp), !if(allow_ftz, (ins  FTZFlag:$ftz), (ins)));
  let hasSideEffects = false in {
    def rr :
      BasicFlagsNVPTXInst<(outs B1:$dst), (ins t.RC:$a, t.RC:$b),
                          flags, op_str>;
    
    if t.SupportsImm then {
      def ri :
        BasicFlagsNVPTXInst<(outs B1:$dst), (ins t.RC:$a, t.Imm:$b),
                            flags, op_str>;
      def ir :
        BasicFlagsNVPTXInst<(outs B1:$dst), (ins t.Imm:$a, t.RC:$b),
                            flags, op_str>;
    }
  }
  def : Pat<(i1 (setcc t.Ty:$a, t.Ty:$b, cond:$cc)),
            (!cast<NVPTXInst>(NAME # "rr") $a, $b, (cond2cc $cc))>;
  if t.SupportsImm then {
    def : Pat<(i1 (setcc t.Ty:$a, fpimm:$b, cond:$cc)),
              (!cast<NVPTXInst>(NAME # "ri") $a, fpimm:$b, (cond2cc $cc))>;
    def : Pat<(i1 (setcc fpimm:$a, t.Ty:$b, cond:$cc)),
              (!cast<NVPTXInst>(NAME # "ir") fpimm:$a, $b, (cond2cc $cc))>;
  }
}

multiclass ISETP<RegTyInfo t> {
  defvar op_str = "setp.${cmp:ICmp}.${cmp:IType}" # t.Size;
  let hasSideEffects = false in {
    def rr :
      BasicFlagsNVPTXInst<(outs B1:$dst), (ins t.RC:$a, t.RC:$b),
                          (ins CmpMode:$cmp), op_str>;
    def ri :
      BasicFlagsNVPTXInst<(outs B1:$dst), (ins t.RC:$a, t.Imm:$b),
                          (ins CmpMode:$cmp), op_str>;
    def ir :
      BasicFlagsNVPTXInst<(outs B1:$dst), (ins t.Imm:$a, t.RC:$b),
                          (ins CmpMode:$cmp), op_str>;
  }
  def : Pat<(i1 (setcc t.Ty:$a, t.Ty:$b, cond:$cc)),
            (!cast<NVPTXInst>(NAME # "rr") $a, $b, (cond2cc $cc))>;
  def : Pat<(i1 (setcc t.Ty:$a, imm:$b, cond:$cc)),
            (!cast<NVPTXInst>(NAME # "ri") $a, imm:$b, (cond2cc $cc))>;
  def : Pat<(i1 (setcc imm:$a, t.Ty:$b, cond:$cc)),
            (!cast<NVPTXInst>(NAME # "ir") imm:$a, $b, (cond2cc $cc))>;
}

defm SETP_i16 : ISETP<I16RT>;
defm SETP_i32 : ISETP<I32RT>;
defm SETP_i64 : ISETP<I64RT>;

defm SETP_f32 : FSETP<F32RT>;
defm SETP_f64 : FSETP<F64RT, allow_ftz = false>;
let Predicates = [useFP16Math] in
  defm SETP_f16 : FSETP<F16RT>;
let Predicates = [hasBF16Math, hasPTX<78>, hasSM<90>] in
  defm SETP_bf16 : FSETP<BF16RT>;

def SETP_f16x2rr :
      BasicFlagsNVPTXInst<(outs B1:$p, B1:$q),
                (ins B32:$a, B32:$b), (ins CmpMode:$cmp, FTZFlag:$ftz),
                "setp.${cmp:FCmp}$ftz.f16x2">,
                Requires<[useFP16Math]>;

def SETP_bf16x2rr :
      BasicFlagsNVPTXInst<(outs B1:$p, B1:$q),
                (ins B32:$a, B32:$b), (ins CmpMode:$cmp, FTZFlag:$ftz),
                "setp.${cmp:FCmp}$ftz.bf16x2">,
                Requires<[hasBF16Math, hasPTX<78>, hasSM<90>]>;

//-----------------------------------
// Data Movement (Load / Store, Move)
//-----------------------------------

def addr : ComplexPattern<pAny, 2, "SelectADDR">;

def ADDR_base : Operand<pAny>;
def ADDR : Operand<pAny> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops ADDR_base, i32imm);
}

def AtomicCode : Operand<i32> {
  let PrintMethod = "printAtomicCode";
}

def MmaCode : Operand<i32> {
  let PrintMethod = "printMmaCode";
}

// Get pointer to local stack.
let hasSideEffects = false in {
  def MOV_DEPOT_ADDR :    NVPTXInst<(outs B32:$d), (ins i32imm:$num),
                                     "mov.b32 \t$d, __local_depot$num;">;
  def MOV_DEPOT_ADDR_64 : NVPTXInst<(outs B64:$d), (ins i32imm:$num),
                                    "mov.b64 \t$d, __local_depot$num;">;
}

let hasSideEffects = false in {
  let isMoveReg = true, isAsCheapAsAMove = true in
    class MOVr<RegisterClass RC, string OpStr> :
      BasicNVPTXInst<(outs RC:$dst), (ins RC:$src), "mov." # OpStr>;

  let isMoveImm = true, isAsCheapAsAMove = true in
    class MOVi<RegTyInfo t, string suffix> :
      BasicNVPTXInst<(outs t.RC:$dst), (ins t.Imm:$src),
              "mov." # suffix,
              [(set t.Ty:$dst, t.ImmNode:$src)]>;

  // We don't want to set isAsCheapAsAMove to true for these instructions as
  // this would prevent CSE and resulted in regressions (see discussion after
  // PR-145581 in llvm-project).
  class MovSymInst<RegTyInfo t> :
    BasicNVPTXInst<(outs t.RC:$dst), (ins Operand<t.Ty>:$src),
                   "mov.b" # t.Size>;
}

def MOV_B1_r : MOVr<B1, "pred">;
def MOV_B16_r : MOVr<B16, "b16">;
def MOV_B32_r : MOVr<B32, "b32">;
def MOV_B64_r : MOVr<B64, "b64">;
def MOV_B128_r : MOVr<B128, "b128">;

def MOV_B1_i   : MOVi<I1RT, "pred">;
def MOV_B16_i  : MOVi<I16RT, "b16">;
def MOV_B32_i  : MOVi<I32RT, "b32">;
def MOV_B64_i  : MOVi<I64RT, "b64">;
def MOV_F16_i  : MOVi<F16RT, "b16">;
def MOV_BF16_i : MOVi<BF16RT, "b16">;
def MOV_F32_i  : MOVi<F32RT, "b32">;
def MOV_F64_i  : MOVi<F64RT, "b64">;

def MOV_B32_sym : MovSymInst<I32RT>;
def MOV_B64_sym : MovSymInst<I64RT>;


def to_tglobaladdr : SDNodeXForm<globaladdr, [{
  return CurDAG->getTargetGlobalAddress(N->getGlobal(), SDLoc(N),
                                        N->getValueType(0), N->getOffset(),
                                        N->getTargetFlags());
}]>;

def to_texternsym : SDNodeXForm<externalsym, [{
  return CurDAG->getTargetExternalSymbol(N->getSymbol(), N->getValueType(0),
                                         N->getTargetFlags());
}]>;

def to_tframeindex : SDNodeXForm<frameindex, [{
  return CurDAG->getTargetFrameIndex(N->getIndex(), N->getValueType(0));
}]>;

def : Pat<(i32 globaladdr:$dst), (MOV_B32_sym (to_tglobaladdr $dst))>;
def : Pat<(i64 globaladdr:$dst), (MOV_B64_sym (to_tglobaladdr $dst))>;

def : Pat<(i32 externalsym:$dst), (MOV_B32_sym (to_texternsym $dst))>;
def : Pat<(i64 externalsym:$dst), (MOV_B64_sym (to_texternsym $dst))>;

//---- Copy Frame Index ----
def LEA_ADDRi :   NVPTXInst<(outs B32:$dst), (ins ADDR:$addr),
                            "add.u32 \t$dst, ${addr:add};">;
def LEA_ADDRi64 : NVPTXInst<(outs B64:$dst), (ins ADDR:$addr),
                            "add.u64 \t$dst, ${addr:add};">;

def : Pat<(i32 frameindex:$fi), (LEA_ADDRi (to_tframeindex $fi), 0)>;
def : Pat<(i64 frameindex:$fi), (LEA_ADDRi64 (to_tframeindex $fi), 0)>;

//-----------------------------------
// Comparison and Selection
//-----------------------------------
// TODO: These patterns seem very specific and brittle. We should try to find
// a more general solution.

def cond_signed : PatLeaf<(cond), [{
  return isSignedIntSetCC(N->get());
}]>;

// A 16-bit signed comparison of sign-extended byte extracts can be converted
// to 32-bit comparison if we change the PRMT to sign-extend the extracted
// bytes.
def : Pat<(setcc (i16 (sext_inreg (trunc (prmt i32:$a, 0, byte_extract_prmt:$sel_a, PrmtNONE)), i8)),
                 (i16 (sext_inreg (trunc (prmt i32:$b, 0, byte_extract_prmt:$sel_b, PrmtNONE)), i8)),
                 cond_signed:$cc),
          (SETP_i32rr (PRMT_B32rii i32:$a, 0, (to_sign_extend_selector $sel_a), PrmtNONE),
                      (PRMT_B32rii i32:$b, 0, (to_sign_extend_selector $sel_b), PrmtNONE),
                      (cond2cc $cc))>;

def SDTDeclareArrayParam :
  SDTypeProfile<0, 3, [SDTCisVT<0, i32>, SDTCisVT<1, i32>, SDTCisVT<2, i32>]>;
def SDTDeclareScalarParam :
  SDTypeProfile<0, 2, [SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;
def SDTMoveParamProfile : SDTypeProfile<1, 1, [SDTCisInt<0>, SDTCisSameAs<0, 1>]>;

def SDTProxyReg : SDTypeProfile<1, 1, [SDTCisSameAs<0, 1>]>;


def declare_array_param :
  SDNode<"NVPTXISD::DeclareArrayParam", SDTDeclareArrayParam,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def declare_scalar_param :
  SDNode<"NVPTXISD::DeclareScalarParam", SDTDeclareScalarParam,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def MoveParam :
  SDNode<"NVPTXISD::MoveParam", SDTMoveParamProfile, []>;
def proxy_reg :
  SDNode<"NVPTXISD::ProxyReg", SDTProxyReg, [SDNPHasChain]>;

  /// CALL(Chain, IsConvergent, IsIndirectCall/IsUniform, NumReturns,
  ///      NumParams, Callee, Proto)
def SDTCallProfile : SDTypeProfile<0, 6,
                       [SDTCisVT<0, i32>, SDTCisVT<1, i32>, SDTCisVT<2, i32>,
                        SDTCisVT<3, i32>, SDTCisVT<5, i32>]>;
def call : SDNode<"NVPTXISD::CALL", SDTCallProfile, [SDNPHasChain, SDNPSideEffect]>;

/// CALL(Chain, IsConvergent, IsIndirectCall/IsUniform, NumReturns,
///      NumParams, Callee, Proto)

def CallOperand : Operand<i32> { let PrintMethod = "printCallOperand"; }

foreach is_convergent = [0, 1] in {
  defvar convergent_suffix = !if(is_convergent, "_conv", "");

  let isCall = 1, isConvergent = is_convergent in {
    def CALL # convergent_suffix :
      NVPTXInst<(outs),
                (ins ADDR_base:$addr, CallOperand:$rets, CallOperand:$params, 
                     i32imm:$proto),
                "call${rets:RetList} $addr, (${params:ParamList}), prototype_$proto;">;

    def CALL_UNI # convergent_suffix :
      NVPTXInst<(outs),
                (ins ADDR_base:$addr, CallOperand:$rets, CallOperand:$params),
                "call.uni${rets:RetList} $addr, (${params:ParamList});">;
  }

  defvar call_inst = !cast<NVPTXInst>("CALL" # convergent_suffix);
  def : Pat<(call is_convergent, 1, imm:$rets, imm:$params, i32:$addr, imm:$proto),
            (call_inst $addr, imm:$rets, imm:$params, imm:$proto)>;
  def : Pat<(call is_convergent, 1, imm:$rets, imm:$params, i64:$addr, imm:$proto),
            (call_inst $addr, imm:$rets, imm:$params, imm:$proto)>;

  defvar call_uni_inst = !cast<NVPTXInst>("CALL_UNI" # convergent_suffix);
  def : Pat<(call is_convergent, 0, imm:$rets, imm:$params, globaladdr:$addr, 0),
            (call_uni_inst (to_tglobaladdr $addr), imm:$rets, imm:$params)>;
}

def DECLARE_PARAM_array :
  NVPTXInst<(outs), (ins i32imm:$a, i32imm:$align, i32imm:$size),
            ".param .align $align .b8 \t$a[$size];">;
def DECLARE_PARAM_scalar :
  NVPTXInst<(outs), (ins i32imm:$a, i32imm:$size),
            ".param .b$size \t$a;">;

def : Pat<(declare_array_param externalsym:$a, imm:$align, imm:$size),
          (DECLARE_PARAM_array (to_texternsym $a), imm:$align, imm:$size)>;
def : Pat<(declare_scalar_param externalsym:$a, imm:$size),
          (DECLARE_PARAM_scalar (to_texternsym $a), imm:$size)>;

// Call prototype wrapper, this is a dummy instruction that just prints it's
// operand which is string defining the prototype.
def SDTCallPrototype : SDTypeProfile<0, 1, [SDTCisInt<0>]>;
def CallPrototype :
  SDNode<"NVPTXISD::CallPrototype", SDTCallPrototype,
         [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def ProtoIdent : Operand<i32> { let PrintMethod = "printProtoIdent"; }
def CALL_PROTOTYPE :
  NVPTXInst<(outs), (ins ProtoIdent:$ident),
            "$ident", [(CallPrototype (i32 texternalsym:$ident))]>;


foreach t = [I32RT, I64RT] in {
  defvar inst_name = "MOV" # t.Size # "_PARAM";
  def inst_name : BasicNVPTXInst<(outs t.RC:$dst), (ins t.RC:$src), "mov.b" # t.Size>;
  def : Pat<(MoveParam (t.Ty externalsym:$src)),
            (!cast<NVPTXInst>(inst_name) (t.Ty (to_texternsym $src)))>;
}

multiclass ProxyRegInst<string SzStr, NVPTXRegClass rc> {
  def NAME : BasicNVPTXInst<(outs rc:$dst), (ins rc:$src),
                 "mov." # SzStr>;
  foreach vt = rc.RegTypes in
    def : Pat<(vt (proxy_reg vt:$src)), (!cast<NVPTXInst>(NAME) $src)>;
}

defm ProxyRegB1  : ProxyRegInst<"pred", B1>;
defm ProxyRegB16 : ProxyRegInst<"b16",  B16>;
defm ProxyRegB32 : ProxyRegInst<"b32",  B32>;
defm ProxyRegB64 : ProxyRegInst<"b64",  B64>;


// Callseq start and end

// Note: these nodes are marked as SDNPMayStore and SDNPMayLoad because
// they define the scope in which the declared params may be used. Therefore
// we add these flags to ensure ld.param and st.param are not sunk or hoisted
// out of that scope.

def callseq_start : SDNode<"ISD::CALLSEQ_START",
                           SDCallSeqStart<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>,
                           [SDNPHasChain, SDNPOutGlue,
                            SDNPSideEffect, SDNPMayStore, SDNPMayLoad]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END",
                           SDCallSeqEnd<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>,
                           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                            SDNPSideEffect, SDNPMayStore, SDNPMayLoad]>;

def Callseq_Start :
  NVPTXInst<(outs), (ins i32imm:$amt1, i32imm:$amt2),
            "\\{ // callseq $amt1, $amt2",
            [(callseq_start timm:$amt1, timm:$amt2)]>;
def Callseq_End :
  NVPTXInst<(outs), (ins i32imm:$amt1, i32imm:$amt2),
            "\\} // callseq $amt1",
            [(callseq_end timm:$amt1, timm:$amt2)]>;

//
// Load / Store Handling
//
class LD<NVPTXRegClass regclass>
  : NVPTXInst<
    (outs regclass:$dst),
    (ins AtomicCode:$sem, AtomicCode:$scope, AtomicCode:$addsp, AtomicCode:$Sign,
         i32imm:$fromWidth, ADDR:$addr),
    "ld${sem:sem}${scope:scope}${addsp:addsp}.${Sign:sign}$fromWidth "
    "\t$dst, [$addr];">;

let mayLoad=1, hasSideEffects=0 in {
  def LD_i16 : LD<B16>;
  def LD_i32 : LD<B32>;
  def LD_i64 : LD<B64>;
}

class ST<DAGOperand O>
  : NVPTXInst<
    (outs),
    (ins O:$src,
         AtomicCode:$sem, AtomicCode:$scope, AtomicCode:$addsp, i32imm:$toWidth,
         ADDR:$addr),
    "st${sem:sem}${scope:scope}${addsp:addsp}.b$toWidth"
    " \t[$addr], $src;">;

let mayStore=1, hasSideEffects=0 in {
  def ST_i16 : ST<RI16>;
  def ST_i32 : ST<RI32>;
  def ST_i64 : ST<RI64>;
}

// The following is used only in and after vector elementizations.  Vector
// elementization happens at the machine instruction level, so the following
// instructions never appear in the DAG.
multiclass LD_VEC<NVPTXRegClass regclass, bit support_v8 = false> {
  def _v2 : NVPTXInst<
    (outs regclass:$dst1, regclass:$dst2),
    (ins AtomicCode:$sem, AtomicCode:$scope, AtomicCode:$addsp,
         AtomicCode:$Sign, i32imm:$fromWidth, ADDR:$addr),
    "ld${sem:sem}${scope:scope}${addsp:addsp}.v2.${Sign:sign}$fromWidth "
    "\t{{$dst1, $dst2}}, [$addr];">;
  def _v4 : NVPTXInst<
    (outs regclass:$dst1, regclass:$dst2, regclass:$dst3, regclass:$dst4),
    (ins AtomicCode:$sem, AtomicCode:$scope, AtomicCode:$addsp,
         AtomicCode:$Sign, i32imm:$fromWidth, ADDR:$addr),
    "ld${sem:sem}${scope:scope}${addsp:addsp}.v4.${Sign:sign}$fromWidth "
    "\t{{$dst1, $dst2, $dst3, $dst4}}, [$addr];">;
  if support_v8 then
    def _v8 : NVPTXInst<
      (outs regclass:$dst1, regclass:$dst2, regclass:$dst3, regclass:$dst4,
            regclass:$dst5, regclass:$dst6, regclass:$dst7, regclass:$dst8),
      (ins AtomicCode:$sem, AtomicCode:$scope, AtomicCode:$addsp, AtomicCode:$Sign,
           i32imm:$fromWidth, ADDR:$addr),
      "ld${sem:sem}${scope:scope}${addsp:addsp}.v8.${Sign:sign}$fromWidth "
      "\t{{$dst1, $dst2, $dst3, $dst4, $dst5, $dst6, $dst7, $dst8}}, "
      "[$addr];">;
}
let mayLoad=1, hasSideEffects=0 in {
  defm LDV_i16 : LD_VEC<B16>;
  defm LDV_i32 : LD_VEC<B32, support_v8 = true>;
  defm LDV_i64 : LD_VEC<B64>;
}

multiclass ST_VEC<DAGOperand O, bit support_v8 = false> {
  def _v2 : NVPTXInst<
    (outs),
    (ins O:$src1, O:$src2,
         AtomicCode:$sem, AtomicCode:$scope, AtomicCode:$addsp, i32imm:$fromWidth,
         ADDR:$addr),
    "st${sem:sem}${scope:scope}${addsp:addsp}.v2.b$fromWidth "
    "\t[$addr], {{$src1, $src2}};">;
  def _v4 : NVPTXInst<
    (outs),
    (ins O:$src1, O:$src2, O:$src3, O:$src4,
         AtomicCode:$sem, AtomicCode:$scope, AtomicCode:$addsp, i32imm:$fromWidth,
         ADDR:$addr),
    "st${sem:sem}${scope:scope}${addsp:addsp}.v4.b$fromWidth "
    "\t[$addr], {{$src1, $src2, $src3, $src4}};">;
  if support_v8 then
    def _v8 : NVPTXInst<
      (outs),
      (ins O:$src1, O:$src2, O:$src3, O:$src4,
           O:$src5, O:$src6, O:$src7, O:$src8,
           AtomicCode:$sem, AtomicCode:$scope, AtomicCode:$addsp, i32imm:$fromWidth,
           ADDR:$addr),
      "st${sem:sem}${scope:scope}${addsp:addsp}.v8.b$fromWidth "
      "\t[$addr], "
      "{{$src1, $src2, $src3, $src4, $src5, $src6, $src7, $src8}};">;
}

let mayStore=1, hasSideEffects=0 in {
  defm STV_i16 : ST_VEC<RI16>;
  defm STV_i32 : ST_VEC<RI32, support_v8 = true>;
  defm STV_i64 : ST_VEC<RI64>;
}

//---- Conversion ----

foreach rc = [B16, B32, B64] in
  foreach ta = rc.RegTypes in
    foreach tb = rc.RegTypes in
      if !ne(ta, tb) then
        def : Pat<(ta (bitconvert tb:$a)),
                  (ta rc:$a)>;

// NOTE: pred->fp are currently sub-optimal due to an issue in TableGen where
// we cannot specify floating-point literals in isel patterns.  Therefore, we
// use an integer selp to select either 1 (or -1 in case of signed) or 0
// and then cvt to floating-point.

// sint -> f16
def : Pat<(f16 (sint_to_fp  i1:$a)), (CVT_f16_s32 (SELP_b32ii -1, 0, $a), CvtRN)>;
def : Pat<(f16 (sint_to_fp i16:$a)), (CVT_f16_s16 $a, CvtRN)>;
def : Pat<(f16 (sint_to_fp i32:$a)), (CVT_f16_s32 $a, CvtRN)>;
def : Pat<(f16 (sint_to_fp i64:$a)), (CVT_f16_s64 $a, CvtRN)>;

// uint -> f16
def : Pat<(f16 (uint_to_fp  i1:$a)), (CVT_f16_u32 (SELP_b32ii 1, 0, $a), CvtRN)>;
def : Pat<(f16 (uint_to_fp i16:$a)), (CVT_f16_u16 $a, CvtRN)>;
def : Pat<(f16 (uint_to_fp i32:$a)), (CVT_f16_u32 $a, CvtRN)>;
def : Pat<(f16 (uint_to_fp i64:$a)), (CVT_f16_u64 $a, CvtRN)>;

// sint -> bf16
let Predicates = [hasPTX<78>, hasSM<90>] in {
  def : Pat<(bf16 (sint_to_fp i1:$a)), (CVT_bf16_s32 (SELP_b32ii 1, 0, $a), CvtRN)>;
  def : Pat<(bf16 (sint_to_fp i16:$a)), (CVT_bf16_s16 $a, CvtRN)>;
  def : Pat<(bf16 (sint_to_fp i32:$a)), (CVT_bf16_s32 $a, CvtRN)>;
  def : Pat<(bf16 (sint_to_fp i64:$a)), (CVT_bf16_s64 $a, CvtRN)>;
}

// uint -> bf16
let Predicates = [hasPTX<78>, hasSM<90>] in {
  def : Pat<(bf16 (uint_to_fp i1:$a)), (CVT_bf16_u32 (SELP_b32ii 1, 0, $a), CvtRN)>;
  def : Pat<(bf16 (uint_to_fp i16:$a)), (CVT_bf16_u16 $a, CvtRN)>;
  def : Pat<(bf16 (uint_to_fp i32:$a)), (CVT_bf16_u32 $a, CvtRN)>;
  def : Pat<(bf16 (uint_to_fp i64:$a)), (CVT_bf16_u64 $a, CvtRN)>;
}

// sint -> f32
def : Pat<(f32 (sint_to_fp  i1:$a)), (CVT_f32_s32 (SELP_b32ii -1, 0, $a), CvtRN)>;
def : Pat<(f32 (sint_to_fp i16:$a)), (CVT_f32_s16 $a, CvtRN)>;
def : Pat<(f32 (sint_to_fp i32:$a)), (CVT_f32_s32 $a, CvtRN)>;
def : Pat<(f32 (sint_to_fp i64:$a)), (CVT_f32_s64 $a, CvtRN)>;

// uint -> f32
def : Pat<(f32 (uint_to_fp  i1:$a)), (CVT_f32_u32 (SELP_b32ii 1, 0, $a), CvtRN)>;
def : Pat<(f32 (uint_to_fp i16:$a)), (CVT_f32_u16 $a, CvtRN)>;
def : Pat<(f32 (uint_to_fp i32:$a)), (CVT_f32_u32 $a, CvtRN)>;
def : Pat<(f32 (uint_to_fp i64:$a)), (CVT_f32_u64 $a, CvtRN)>;

// sint -> f64
def : Pat<(f64 (sint_to_fp i1:$a)), (CVT_f64_s32 (SELP_b32ii -1, 0, $a), CvtRN)>;
def : Pat<(f64 (sint_to_fp i16:$a)), (CVT_f64_s16 $a, CvtRN)>;
def : Pat<(f64 (sint_to_fp i32:$a)), (CVT_f64_s32 $a, CvtRN)>;
def : Pat<(f64 (sint_to_fp i64:$a)), (CVT_f64_s64 $a, CvtRN)>;

// uint -> f64
def : Pat<(f64 (uint_to_fp i1:$a)),  (CVT_f64_u32 (SELP_b32ii 1, 0, $a), CvtRN)>;
def : Pat<(f64 (uint_to_fp i16:$a)), (CVT_f64_u16 $a, CvtRN)>;
def : Pat<(f64 (uint_to_fp i32:$a)), (CVT_f64_u32 $a, CvtRN)>;
def : Pat<(f64 (uint_to_fp i64:$a)), (CVT_f64_u64 $a, CvtRN)>;


// f16 -> sint
def : Pat<(i1  (fp_to_sint f16:$a)), (SETP_i16ri $a, 0, CmpEQ)>;
def : Pat<(i16 (fp_to_sint f16:$a)), (CVT_s16_f16 $a, CvtRZI)>;
def : Pat<(i32 (fp_to_sint f16:$a)), (CVT_s32_f16 $a, CvtRZI)>;
def : Pat<(i64 (fp_to_sint f16:$a)), (CVT_s64_f16 $a, CvtRZI)>;

// f16 -> uint
def : Pat<(i1  (fp_to_uint f16:$a)), (SETP_i16ri $a, 0, CmpEQ)>;
def : Pat<(i16 (fp_to_uint f16:$a)), (CVT_u16_f16 $a, CvtRZI)>;
def : Pat<(i32 (fp_to_uint f16:$a)), (CVT_u32_f16 $a, CvtRZI)>;
def : Pat<(i64 (fp_to_uint f16:$a)), (CVT_u64_f16 $a, CvtRZI)>;

// bf16 -> sint
def : Pat<(i1  (fp_to_sint bf16:$a)), (SETP_i16ri $a, 0, CmpEQ)>;
def : Pat<(i16 (fp_to_sint bf16:$a)), (CVT_s16_bf16 $a, CvtRZI)>;
def : Pat<(i32 (fp_to_sint bf16:$a)), (CVT_s32_bf16 $a, CvtRZI)>;
def : Pat<(i64 (fp_to_sint bf16:$a)), (CVT_s64_bf16 $a, CvtRZI)>;

// bf16 -> uint
def : Pat<(i1 (fp_to_uint bf16:$a)),  (SETP_i16ri $a, 0, CmpEQ)>;
def : Pat<(i16 (fp_to_uint bf16:$a)), (CVT_u16_bf16 $a, CvtRZI)>;
def : Pat<(i32 (fp_to_uint bf16:$a)), (CVT_u32_bf16 $a, CvtRZI)>;
def : Pat<(i64 (fp_to_uint bf16:$a)), (CVT_u64_bf16 $a, CvtRZI)>;
// f32 -> sint
let Predicates = [doF32FTZ] in {
  def : Pat<(i16 (fp_to_sint f32:$a)), (CVT_s16_f32 $a, CvtRZI_FTZ)>;
  def : Pat<(i32 (fp_to_sint f32:$a)), (CVT_s32_f32 $a, CvtRZI_FTZ)>;
  def : Pat<(i64 (fp_to_sint f32:$a)), (CVT_s64_f32 $a, CvtRZI_FTZ)>;
}
def : Pat<(i1  (fp_to_sint f32:$a)), (SETP_i32ri $a, 0, CmpEQ)>;
def : Pat<(i16 (fp_to_sint f32:$a)), (CVT_s16_f32 $a, CvtRZI)>;
def : Pat<(i32 (fp_to_sint f32:$a)), (CVT_s32_f32 $a, CvtRZI)>;
def : Pat<(i64 (fp_to_sint f32:$a)), (CVT_s64_f32 $a, CvtRZI)>;

// f32 -> uint
let Predicates = [doF32FTZ] in {
  def : Pat<(i16 (fp_to_uint f32:$a)), (CVT_u16_f32 $a, CvtRZI_FTZ)>;
  def : Pat<(i32 (fp_to_uint f32:$a)), (CVT_u32_f32 $a, CvtRZI_FTZ)>;
  def : Pat<(i64 (fp_to_uint f32:$a)), (CVT_u64_f32 $a, CvtRZI_FTZ)>;
}
def : Pat<(i1  (fp_to_uint f32:$a)), (SETP_i32ri $a, 0, CmpEQ)>;
def : Pat<(i16 (fp_to_uint f32:$a)), (CVT_u16_f32 $a, CvtRZI)>;
def : Pat<(i32 (fp_to_uint f32:$a)), (CVT_u32_f32 $a, CvtRZI)>;
def : Pat<(i64 (fp_to_uint f32:$a)), (CVT_u64_f32 $a, CvtRZI)>;

// f64 -> sint
def : Pat<(i1  (fp_to_sint f64:$a)), (SETP_i64ri $a, 0, CmpEQ)>;
def : Pat<(i16 (fp_to_sint f64:$a)), (CVT_s16_f64 $a, CvtRZI)>;
def : Pat<(i32 (fp_to_sint f64:$a)), (CVT_s32_f64 $a, CvtRZI)>;
def : Pat<(i64 (fp_to_sint f64:$a)), (CVT_s64_f64 $a, CvtRZI)>;

// f64 -> uint
def : Pat<(i1  (fp_to_uint f64:$a)), (SETP_i64ri $a, 0, CmpEQ)>;
def : Pat<(i16 (fp_to_uint f64:$a)), (CVT_u16_f64 $a, CvtRZI)>;
def : Pat<(i32 (fp_to_uint f64:$a)), (CVT_u32_f64 $a, CvtRZI)>;
def : Pat<(i64 (fp_to_uint f64:$a)), (CVT_u64_f64 $a, CvtRZI)>;

// sext i1
def : Pat<(i16 (sext i1:$a)), (SELP_b16ii -1, 0, $a)>;
def : Pat<(i32 (sext i1:$a)), (SELP_b32ii -1, 0, $a)>;
def : Pat<(i64 (sext i1:$a)), (SELP_b64ii -1, 0, $a)>;

// zext i1
def : Pat<(i16 (zext i1:$a)), (SELP_b16ii 1, 0, $a)>;
def : Pat<(i32 (zext i1:$a)), (SELP_b32ii 1, 0, $a)>;
def : Pat<(i64 (zext i1:$a)), (SELP_b64ii 1, 0, $a)>;

// anyext i1
def : Pat<(i16 (anyext i1:$a)), (SELP_b16ii -1, 0, $a)>;
def : Pat<(i32 (anyext i1:$a)), (SELP_b32ii -1, 0, $a)>;
def : Pat<(i64 (anyext i1:$a)), (SELP_b64ii -1, 0, $a)>;

// sext i16
def : Pat<(i32 (sext i16:$a)), (CVT_s32_s16 $a, CvtNONE)>;
def : Pat<(i64 (sext i16:$a)), (CVT_s64_s16 $a, CvtNONE)>;

// zext i16
def : Pat<(i32 (zext i16:$a)), (CVT_u32_u16 $a, CvtNONE)>;
def : Pat<(i64 (zext i16:$a)), (CVT_u64_u16 $a, CvtNONE)>;

// anyext i16
def : Pat<(i32 (anyext i16:$a)), (CVT_u32_u16 $a, CvtNONE)>;
def : Pat<(i64 (anyext i16:$a)), (CVT_u64_u16 $a, CvtNONE)>;

// sext i32
def : Pat<(i64 (sext i32:$a)), (CVT_s64_s32 $a, CvtNONE)>;

// zext i32
def : Pat<(i64 (zext i32:$a)), (CVT_u64_u32 $a, CvtNONE)>;

// anyext i32
def : Pat<(i64 (anyext i32:$a)), (CVT_u64_u32 $a, CvtNONE)>;


// truncate i64
def : Pat<(i32 (trunc i64:$a)), (CVT_u32_u64 $a, CvtNONE)>;
def : Pat<(i16 (trunc i64:$a)), (CVT_u16_u64 $a, CvtNONE)>;
def : Pat<(i1  (trunc i64:$a)), (SETP_i64ri (AND_b64ri $a, 1), 0, CmpNE)>;

// truncate i32
def : Pat<(i16 (trunc i32:$a)), (CVT_u16_u32 $a, CvtNONE)>;
def : Pat<(i1  (trunc i32:$a)), (SETP_i32ri (AND_b32ri $a, 1), 0, CmpNE)>;

// truncate i16
def : Pat<(i1 (trunc i16:$a)), (SETP_i16ri (AND_b16ri $a, 1), 0, CmpNE)>;

// sext_inreg
def : Pat<(sext_inreg i16:$a, i8), (CVT_INREG_s16_s8 $a)>;
def : Pat<(sext_inreg i32:$a, i8), (CVT_INREG_s32_s8 $a)>;
def : Pat<(sext_inreg i32:$a, i16), (CVT_INREG_s32_s16 $a)>;
def : Pat<(sext_inreg i64:$a, i8), (CVT_INREG_s64_s8 $a)>;
def : Pat<(sext_inreg i64:$a, i16), (CVT_INREG_s64_s16 $a)>;
def : Pat<(sext_inreg i64:$a, i32), (CVT_INREG_s64_s32 $a)>;

let hasSideEffects = false in {
  // pack a set of smaller int registers to a larger int register
  def V4I16toI64 : NVPTXInst<(outs B64:$d),
                             (ins B16:$s1, B16:$s2,
                                  B16:$s3, B16:$s4),
                             "mov.b64 \t$d, {{$s1, $s2, $s3, $s4}};">;
  def V2I16toI32 : NVPTXInst<(outs B32:$d),
                             (ins B16:$s1, B16:$s2),
                             "mov.b32 \t$d, {{$s1, $s2}};">;
  def V2I32toI64 : NVPTXInst<(outs B64:$d),
                             (ins B32:$s1, B32:$s2),
                             "mov.b64 \t$d, {{$s1, $s2}};">;
  def V2I64toI128 : NVPTXInst<(outs B128:$d),
                              (ins B64:$s1, B64:$s2),
                              "mov.b128 \t$d, {{$s1, $s2}};">;

  // unpack a larger int register to a set of smaller int registers
  def I64toV4I16 : NVPTXInst<(outs B16:$d1, B16:$d2,
                                   B16:$d3, B16:$d4),
                             (ins B64:$s),
                             "mov.b64 \t{{$d1, $d2, $d3, $d4}}, $s;">;
  def I32toV2I16 : NVPTXInst<(outs B16:$d1, B16:$d2),
                             (ins B32:$s),
                             "mov.b32 \t{{$d1, $d2}}, $s;">;
  def I64toV2I32 : NVPTXInst<(outs B32:$d1, B32:$d2),
                             (ins B64:$s),
                             "mov.b64 \t{{$d1, $d2}}, $s;">;
  def I128toV2I64: NVPTXInst<(outs B64:$d1, B64:$d2),
                              (ins B128:$s),
                              "mov.b128 \t{{$d1, $d2}}, $s;">;

  def I32toI16H  : NVPTXInst<(outs B16:$high), (ins B32:$s),
                             "{{ .reg .b16 tmp; mov.b32 {tmp, $high}, $s; }}">;
  def I32toI16L  : NVPTXInst<(outs B16:$low), (ins B32:$s),
                             "{{ .reg .b16 tmp; mov.b32 {$low, tmp}, $s; }}">;
  def I64toI32H  : NVPTXInst<(outs B32:$high), (ins B64:$s),
                            "{{ .reg .b32 tmp; mov.b64 {tmp, $high}, $s; }}">;
  def I64toI32L  : NVPTXInst<(outs B32:$low), (ins B64:$s),
                             "{{ .reg .b32 tmp; mov.b64 {$low, tmp}, $s; }}">;

  // PTX 7.1 lets you avoid a temp register and just use _ as a "sink" for the
  // unused high/low part.
  let Predicates = [hasPTX<71>] in {
    def I32toI16H_Sink  : NVPTXInst<(outs B16:$high), (ins B32:$s),
                              "mov.b32 \t{{_, $high}}, $s;">;
    def I32toI16L_Sink  : NVPTXInst<(outs B16:$low), (ins B32:$s),
                              "mov.b32 \t{{$low, _}}, $s;">;
    def I64toI32H_Sink  : NVPTXInst<(outs B32:$high), (ins B64:$s),
                              "mov.b64 \t{{_, $high}}, $s;">;
    def I64toI32L_Sink  : NVPTXInst<(outs B32:$low), (ins B64:$s),
                              "mov.b64 \t{{$low, _}}, $s;">;
  }
}

let Predicates = [hasPTX<71>] in {
  def : Pat<(i16 (trunc (srl i32:$s, (i32 16)))), (I32toI16H_Sink i32:$s)>;
  def : Pat<(i16 (trunc (sra i32:$s, (i32 16)))), (I32toI16H_Sink i32:$s)>;
  def : Pat<(i32 (trunc (srl i64:$s, (i32 32)))), (I64toI32H_Sink i64:$s)>;
  def : Pat<(i32 (trunc (sra i64:$s, (i32 32)))), (I64toI32H_Sink i64:$s)>;
}

// Fall back to the old way if we don't have PTX 7.1.
def : Pat<(i16 (trunc (srl i32:$s, (i32 16)))), (I32toI16H $s)>;
def : Pat<(i16 (trunc (sra i32:$s, (i32 16)))), (I32toI16H $s)>;
def : Pat<(i32 (trunc (srl i64:$s, (i32 32)))), (I64toI32H $s)>;
def : Pat<(i32 (trunc (sra i64:$s, (i32 32)))), (I64toI32H $s)>;

def: Pat<(i32 (sext (extractelt v2i16:$src, 0))),
         (CVT_INREG_s32_s16 $src)>;

// Handle extracting one element from the pair (32-bit types)
foreach vt = [v2f16, v2bf16, v2i16] in {
  def : Pat<(extractelt vt:$src, 0), (I32toI16L_Sink $src)>, Requires<[hasPTX<71>]>;
  def : Pat<(extractelt vt:$src, 1), (I32toI16H_Sink $src)>, Requires<[hasPTX<71>]>;

  def : Pat<(extractelt vt:$src, 0), (I32toI16L $src)>;
  def : Pat<(extractelt vt:$src, 1), (I32toI16H $src)>;

  def : Pat<(vt (build_vector vt.ElementType:$a, vt.ElementType:$b)), 
            (V2I16toI32 $a, $b)>;
}

// Same thing for the 64-bit type v2f32.
foreach vt = [v2f32] in {
  def : Pat<(extractelt vt:$src, 0), (I64toI32L_Sink $src)>, Requires<[hasPTX<71>]>;
  def : Pat<(extractelt vt:$src, 1), (I64toI32H_Sink $src)>, Requires<[hasPTX<71>]>;

  def : Pat<(extractelt vt:$src, 0), (I64toI32L $src)>;
  def : Pat<(extractelt vt:$src, 1), (I64toI32H $src)>;

  def : Pat<(vt (build_vector vt.ElementType:$a, vt.ElementType:$b)), 
            (V2I32toI64 $a, $b)>;
}

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
      : BasicNVPTXInst<(outs B32:$dst),
                  (ins  B32:$lo, B32:$hi, i32imm:$amt),
                  "shf." # mode # ".b32",
                  [(set i32:$dst,
                      (op i32:$hi, i32:$lo, (i32 imm:$amt)))]>,
        Requires<[hasHWROT32]>;

    def _r
      : BasicNVPTXInst<(outs B32:$dst),
                  (ins  B32:$lo, B32:$hi, B32:$amt),
                  "shf." # mode # ".b32",
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
    def CLZr # RT.Size : BasicNVPTXInst<(outs B32:$d), (ins RT.RC:$a),
                                   "clz.b" # RT.Size,
                                   [(set i32:$d, (ctlz RT.Ty:$a))]>;

    // Population count
    def POPCr # RT.Size : BasicNVPTXInst<(outs B32:$d), (ins RT.RC:$a),
                                    "popc.b" # RT.Size,
                                    [(set i32:$d, (ctpop RT.Ty:$a))]>;
  }
}

// fpround f32 -> f16
def : Pat<(f16 (fpround f32:$a)), (CVT_f16_f32 $a, CvtRN)>;

// fpround f32 -> bf16
def : Pat<(bf16 (fpround f32:$a)), (CVT_bf16_f32 $a, CvtRN)>, 
      Requires<[hasPTX<70>, hasSM<80>]>;

// fpround f64 -> f16
def : Pat<(f16 (fpround f64:$a)), (CVT_f16_f64 $a, CvtRN)>;

// fpround f64 -> bf16
def : Pat<(bf16 (fpround f64:$a)), (CVT_bf16_f64 $a, CvtRN)>, 
      Requires<[hasPTX<78>, hasSM<90>]>;

// fpround f64 -> f32
def : Pat<(f32 (fpround f64:$a)), (CVT_f32_f64 $a, CvtRN_FTZ)>, Requires<[doF32FTZ]>;
def : Pat<(f32 (fpround f64:$a)), (CVT_f32_f64 $a, CvtRN)>;

// fpextend f16 -> f32
def : Pat<(f32 (fpextend f16:$a)), (CVT_f32_f16 $a, CvtNONE_FTZ)>, Requires<[doF32FTZ]>;
def : Pat<(f32 (fpextend f16:$a)), (CVT_f32_f16 $a, CvtNONE)>;
// fpextend bf16 -> f32
def : Pat<(f32 (fpextend bf16:$a)), (CVT_f32_bf16 $a, CvtNONE_FTZ)>, Requires<[doF32FTZ]>;
def : Pat<(f32 (fpextend bf16:$a)), (CVT_f32_bf16 $a, CvtNONE)>, Requires<[hasPTX<71>, hasSM<80>]>;

// fpextend f16 -> f64
def : Pat<(f64 (fpextend f16:$a)), (CVT_f64_f16 $a, CvtNONE)>;

// fpextend bf16 -> f64
def : Pat<(f64 (fpextend bf16:$a)), (CVT_f64_bf16 $a, CvtNONE)>, Requires<[hasPTX<78>, hasSM<90>]>;

// fpextend f32 -> f64
def : Pat<(f64 (fpextend f32:$a)), (CVT_f64_f32 $a, CvtNONE_FTZ)>, Requires<[doF32FTZ]>;
def : Pat<(f64 (fpextend f32:$a)), (CVT_f64_f32 $a, CvtNONE)>;

def retglue : SDNode<"NVPTXISD::RET_GLUE", SDTNone,
                     [SDNPHasChain, SDNPOptInGlue]>;

// fceil, ffloor, froundeven, ftrunc.

multiclass CVT_ROUND<SDNode OpNode, PatLeaf Mode, PatLeaf ModeFTZ> {
  def : Pat<(OpNode  f16:$a), (CVT_f16_f16 $a, Mode)>;
  def : Pat<(OpNode bf16:$a), (CVT_bf16_bf16 $a, Mode)>;
  def : Pat<(OpNode  f32:$a), (CVT_f32_f32 $a, ModeFTZ)>, Requires<[doF32FTZ]>;
  def : Pat<(OpNode  f32:$a), (CVT_f32_f32 $a, Mode)>, Requires<[doNoF32FTZ]>;
  def : Pat<(OpNode  f64:$a), (CVT_f64_f64 $a, Mode)>;
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
      def Return : BasicNVPTXInst<(outs), (ins), "ret", [(retglue)]>;

  let isBranch=1 in {
    def CBranch : NVPTXInst<(outs), (ins B1:$a, brtarget:$target),
                              "@$a bra \t$target;",
                              [(brcond i1:$a, bb:$target)]>;

    let isBarrier=1 in
      def GOTO : BasicNVPTXInst<(outs), (ins brtarget:$target),
                            "bra.uni", [(br bb:$target)]>;
  }
}


// trap instruction
def trapinst : BasicNVPTXInst<(outs), (ins), "trap", [(trap)]>, Requires<[noPTXASUnreachableBug]>;
// Emit an `exit` as well to convey to ptxas that `trap` exits the CFG.
// This won't be necessary in a future version of ptxas.
def trapexitinst : NVPTXInst<(outs), (ins), "trap; exit;", [(trap)]>, Requires<[hasPTXASUnreachableBug]>;
// brkpt instruction
def debugtrapinst : BasicNVPTXInst<(outs), (ins), "brkpt", [(debugtrap)]>;

def SDTDynAllocaOp :
  SDTypeProfile<1, 2, [SDTCisSameAs<0, 1>, SDTCisInt<1>, SDTCisVT<2, i32>]>;

def dyn_alloca :
  SDNode<"NVPTXISD::DYNAMIC_STACKALLOC", SDTDynAllocaOp,
         [SDNPHasChain, SDNPSideEffect]>;

foreach t = [I32RT, I64RT] in {
  def DYNAMIC_STACKALLOC # t.Size :
    BasicNVPTXInst<(outs t.RC:$ptr),
              (ins t.RC:$size, i32imm:$align),
              "alloca.u" # t.Size,
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
    NVPTXInst<(outs), (ins brtarget:$target, B32:$val, i32imm:$id),
              "\t$target;\n\tbrx.idx \t$val, $$L_brx_$id;",
              [(brx_end bb:$target, i32:$val, (i32 imm:$id))]> {
      let isBarrier = 1;
    }
}


foreach a_type = ["s", "u"] in {
  foreach b_type = ["s", "u"] in {

    def DOT4_ # a_type # b_type :
      BasicNVPTXInst<(outs B32:$dst),
                (ins B32:$a, B32:$b, B32:$c),
                "dp4a." # a_type # "32." # b_type # "32",
                [(set i32:$dst,
                    (!cast<Intrinsic>("int_nvvm_idp4a_" # a_type # "_" # b_type)
                     i32:$a, i32:$b, i32:$c))]>,
                Requires<[hasDotInstructions]>;

    foreach is_hi = [0, -1] in {
      defvar lohi_suffix = !if(is_hi, "hi", "lo");

      def DOT2_ # lohi_suffix # _ # a_type # b_type :
        BasicNVPTXInst<(outs B32:$dst),
                  (ins B32:$a, B32:$b, B32:$c),
                  "dp2a." # lohi_suffix # "." # a_type # "32." # b_type # "32",
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

let Predicates = [hasPTX<73>, hasSM<52>] in {
  foreach t = [I32RT, I64RT] in {
    def STACKRESTORE_ # t.Size :
      BasicNVPTXInst<(outs), (ins t.RC:$ptr),
                "stackrestore.u" # t.Size,
              [(stackrestore t.Ty:$ptr)]>;

    def STACKSAVE_ # t.Size :
      BasicNVPTXInst<(outs t.RC:$dst), (ins),
                "stacksave.u" # t.Size,
              [(set t.Ty:$dst, (t.Ty stacksave))]>;
  }
}

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
  (PRMT_B32rii $a, (i32 0), (i32 0x0123), PrmtNONE)>;

def : Pat <
  (v2i16 (bswap v2i16:$a)),
  (PRMT_B32rii $a, (i32 0), (i32 0x2301), PrmtNONE)>;

def : Pat <
  (i64 (bswap i64:$a)),
  (V2I32toI64
    (PRMT_B32rii (I64toI32H_Sink $a), (i32 0), (i32 0x0123), PrmtNONE),
    (PRMT_B32rii (I64toI32L_Sink $a), (i32 0), (i32 0x0123), PrmtNONE))>,
  Requires<[hasPTX<71>]>;

// Fall back to the old way if we don't have PTX 7.1.
def : Pat <
  (i64 (bswap i64:$a)),
  (V2I32toI64
    (PRMT_B32rii (I64toI32H $a), (i32 0), (i32 0x0123), PrmtNONE),
    (PRMT_B32rii (I64toI32L $a), (i32 0), (i32 0x0123), PrmtNONE))>;


////////////////////////////////////////////////////////////////////////////////
// PTX Fence instructions
////////////////////////////////////////////////////////////////////////////////

class NVPTXFenceInst<string scope, string sem, Predicate ptx>:
    BasicNVPTXInst<(outs), (ins), "fence."#sem#"."#scope>,
    Requires<[ptx, hasSM<70>]>;

foreach scope = ["sys", "gpu", "cluster", "cta"] in {
  def atomic_thread_fence_seq_cst_#scope: NVPTXFenceInst<scope, "sc", hasPTX<60>>;
  def atomic_thread_fence_acq_rel_#scope: NVPTXFenceInst<scope, "acq_rel", hasPTX<60>>;
  def atomic_thread_fence_acquire_#scope: NVPTXFenceInst<scope, "acquire", hasPTX<87>>;
  def atomic_thread_fence_release_#scope: NVPTXFenceInst<scope, "release", hasPTX<87>>;
}

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

class FMARELUInst<RegTyInfo t, bit allow_ftz, PatFrag zero_pat>
  : BasicFlagsNVPTXInst<(outs t.RC:$dst), (ins t.RC:$a, t.RC:$b, t.RC:$c),
                   !if(allow_ftz, (ins FTZFlag:$ftz), (ins)),
                   "fma.rn" # !if(allow_ftz, "$ftz", "") # ".relu." # t.PtxType,
                   [(set t.Ty:$dst, (NVPTX_fmaxnum_nsz (NVPTX_fma_oneuse_and_nnan t.Ty:$a, t.Ty:$b, t.Ty:$c), zero_pat))]>;

let Predicates = [useFP16Math, hasPTX<70>, hasSM<80>] in {
  def FMARELU_F16 : FMARELUInst<F16RT, true, fpimm_0>;
  def FMARELU_F16X2 : FMARELUInst<F16X2RT, true, zeroinitializer<v2f16>>;
}

let Predicates = [hasBF16Math, hasPTX<70>, hasSM<80>] in {
  def FMARELU_BF16 : FMARELUInst<BF16RT, false, fpimm_0>;
  def FMARELU_BF16X2 : FMARELUInst<BF16X2RT, false, zeroinitializer<v2bf16>>;
}
