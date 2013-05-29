//===- NVPTXInstrInfo.td - NVPTX Instruction defs -------------*- tblgen-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the PTX instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

include "NVPTXInstrFormats.td"

// A NOP instruction
def NOP : NVPTXInst<(outs), (ins), "", []>;

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

//===----------------------------------------------------------------------===//
// NVPTX Instruction Predicate Definitions
//===----------------------------------------------------------------------===//


def hasAtomRedG32 : Predicate<"Subtarget.hasAtomRedG32()">;
def hasAtomRedS32 : Predicate<"Subtarget.hasAtomRedS32()">;
def hasAtomRedGen32 : Predicate<"Subtarget.hasAtomRedGen32()">;
def useAtomRedG32forGen32 :
  Predicate<"!Subtarget.hasAtomRedGen32() && Subtarget.hasAtomRedG32()">;
def hasBrkPt : Predicate<"Subtarget.hasBrkPt()">;
def hasAtomRedG64 : Predicate<"Subtarget.hasAtomRedG64()">;
def hasAtomRedS64 : Predicate<"Subtarget.hasAtomRedS64()">;
def hasAtomRedGen64 : Predicate<"Subtarget.hasAtomRedGen64()">;
def useAtomRedG64forGen64 :
  Predicate<"!Subtarget.hasAtomRedGen64() && Subtarget.hasAtomRedG64()">;
def hasAtomAddF32 : Predicate<"Subtarget.hasAtomAddF32()">;
def hasVote : Predicate<"Subtarget.hasVote()">;
def hasDouble : Predicate<"Subtarget.hasDouble()">;
def reqPTX20 : Predicate<"Subtarget.reqPTX20()">;
def hasLDG : Predicate<"Subtarget.hasLDG()">;
def hasLDU : Predicate<"Subtarget.hasLDU()">;
def hasGenericLdSt : Predicate<"Subtarget.hasGenericLdSt()">;

def doF32FTZ : Predicate<"UseF32FTZ">;

def doFMAF32      : Predicate<"doFMAF32">;
def doFMAF32_ftz  : Predicate<"(doFMAF32 && UseF32FTZ)">;
def doFMAF32AGG      : Predicate<"doFMAF32AGG">;
def doFMAF32AGG_ftz  : Predicate<"(doFMAF32AGG && UseF32FTZ)">;
def doFMAF64      : Predicate<"doFMAF64">;
def doFMAF64AGG      : Predicate<"doFMAF64AGG">;
def doFMADF32     : Predicate<"doFMADF32">;
def doFMADF32_ftz : Predicate<"(doFMADF32 && UseF32FTZ)">;

def doMulWide      : Predicate<"doMulWide">;

def allowFMA : Predicate<"allowFMA">;
def allowFMA_ftz : Predicate<"(allowFMA && UseF32FTZ)">;

def do_DIVF32_APPROX : Predicate<"do_DIVF32_PREC==0">;
def do_DIVF32_FULL : Predicate<"do_DIVF32_PREC==1">;

def do_SQRTF32_APPROX : Predicate<"do_SQRTF32_PREC==0">;
def do_SQRTF32_RN : Predicate<"do_SQRTF32_PREC==1">;

def hasHWROT32 : Predicate<"Subtarget.hasHWROT32()">;

def true : Predicate<"1">;

//===----------------------------------------------------------------------===//
// Special Handling for 8-bit Operands and Operations
//
// PTX supports 8-bit signed and unsigned types, but does not support 8-bit
// operations (like add, shift, etc) except for ld/st/cvt. SASS does not have
// 8-bit registers.
//
// PTX ld, st and cvt instructions permit source and destination data operands
// to be wider than the instruction-type size, so that narrow values may be
// loaded, stored, and converted using regular-width registers.
//
// So in PTX generation, we
// - always use 16-bit registers in place in 8-bit registers.
//   (8-bit variables should stay as 8-bit as they represent memory layout.)
// - for the following 8-bit operations, we sign-ext/zero-ext the 8-bit values
//   before operation
//   . div
//   . rem
//   . neg (sign)
//   . set, setp
//   . shr
//
// We are patching the operations by inserting the cvt instructions in the
// asm strings of the affected instructions.
//
// Since vector operations, except for ld/st, are eventually elementized. We
// do not need to special-hand the vector 8-bit operations.
//
//
//===----------------------------------------------------------------------===//

// Generate string block like
// {
//   .reg .s16 %temp1;
//   .reg .s16 %temp2;
//   cvt.s16.s8 %temp1, %a;
//   cvt.s16.s8 %temp2, %b;
//   opc.s16    %dst, %temp1, %temp2;
// }
// when OpcStr=opc.s TypeStr=s16 CVTStr=cvt.s16.s8
class Handle_i8rr<string OpcStr, string TypeStr, string CVTStr> {
  string s = !strconcat("{{\n\t",
             !strconcat(".reg .", !strconcat(TypeStr,
             !strconcat(" \t%temp1;\n\t",
             !strconcat(".reg .", !strconcat(TypeStr,
             !strconcat(" \t%temp2;\n\t",
             !strconcat(CVTStr, !strconcat(" \t%temp1, $a;\n\t",
             !strconcat(CVTStr, !strconcat(" \t%temp2, $b;\n\t",
             !strconcat(OpcStr, "16 \t$dst, %temp1, %temp2;\n\t}}"))))))))))));
}

// Generate string block like
// {
//   .reg .s16 %temp1;
//   .reg .s16 %temp2;
//   cvt.s16.s8 %temp1, %a;
//   mov.b16    %temp2, %b;
//   cvt.s16.s8 %temp2, %temp2;
//   opc.s16    %dst, %temp1, %temp2;
// }
// when OpcStr=opc.s TypeStr=s16 CVTStr=cvt.s16.s8
class Handle_i8ri<string OpcStr, string TypeStr, string CVTStr> {
  string s = !strconcat("{{\n\t",
             !strconcat(".reg .", !strconcat(TypeStr,
             !strconcat(" \t%temp1;\n\t",
             !strconcat(".reg .",
             !strconcat(TypeStr, !strconcat(" \t%temp2;\n\t",
             !strconcat(CVTStr, !strconcat(" \t%temp1, $a;\n\t",
             !strconcat("mov.b16 \t%temp2, $b;\n\t",
             !strconcat(CVTStr, !strconcat(" \t%temp2, %temp2;\n\t",
             !strconcat(OpcStr, "16 \t$dst, %temp1, %temp2;\n\t}}")))))))))))));
}

// Generate string block like
// {
//   .reg .s16 %temp1;
//   .reg .s16 %temp2;
//   mov.b16    %temp1, %b;
//   cvt.s16.s8 %temp1, %temp1;
//   cvt.s16.s8 %temp2, %a;
//   opc.s16    %dst, %temp1, %temp2;
// }
// when OpcStr=opc.s TypeStr=s16 CVTStr=cvt.s16.s8
class Handle_i8ir<string OpcStr, string TypeStr, string CVTStr> {
  string s = !strconcat("{{\n\t",
             !strconcat(".reg .", !strconcat(TypeStr,
             !strconcat(" \t%temp1;\n\t",
             !strconcat(".reg .", !strconcat(TypeStr,
             !strconcat(" \t%temp2;\n\t",
             !strconcat("mov.b16 \t%temp1, $a;\n\t",
             !strconcat(CVTStr, !strconcat(" \t%temp1, %temp1;\n\t",
             !strconcat(CVTStr, !strconcat(" \t%temp2, $b;\n\t",
             !strconcat(OpcStr, "16 \t$dst, %temp1, %temp2;\n\t}}")))))))))))));
}


//===----------------------------------------------------------------------===//
// Some Common Instruction Class Templates
//===----------------------------------------------------------------------===//

multiclass I3<string OpcStr, SDNode OpNode> {
  def i64rr : NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$a, Int64Regs:$b),
                     !strconcat(OpcStr, "64 \t$dst, $a, $b;"),
                     [(set Int64Regs:$dst, (OpNode Int64Regs:$a,
                       Int64Regs:$b))]>;
  def i64ri : NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$a, i64imm:$b),
                     !strconcat(OpcStr, "64 \t$dst, $a, $b;"),
                     [(set Int64Regs:$dst, (OpNode Int64Regs:$a, imm:$b))]>;
  def i32rr : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a, Int32Regs:$b),
                     !strconcat(OpcStr, "32 \t$dst, $a, $b;"),
                     [(set Int32Regs:$dst, (OpNode Int32Regs:$a,
                       Int32Regs:$b))]>;
  def i32ri : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a, i32imm:$b),
                     !strconcat(OpcStr, "32 \t$dst, $a, $b;"),
                     [(set Int32Regs:$dst, (OpNode Int32Regs:$a, imm:$b))]>;
  def i16rr : NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$a, Int16Regs:$b),
                     !strconcat(OpcStr, "16 \t$dst, $a, $b;"),
                     [(set Int16Regs:$dst, (OpNode Int16Regs:$a,
                       Int16Regs:$b))]>;
  def i16ri : NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$a, i16imm:$b),
                     !strconcat(OpcStr, "16 \t$dst, $a, $b;"),
                     [(set Int16Regs:$dst, (OpNode Int16Regs:$a, (imm):$b))]>;
  def i8rr : NVPTXInst<(outs Int8Regs:$dst), (ins Int8Regs:$a, Int8Regs:$b),
                     !strconcat(OpcStr, "16 \t$dst, $a, $b;"),
                     [(set Int8Regs:$dst, (OpNode Int8Regs:$a, Int8Regs:$b))]>;
  def i8ri : NVPTXInst<(outs Int8Regs:$dst), (ins Int8Regs:$a, i8imm:$b),
                     !strconcat(OpcStr, "16 \t$dst, $a, $b;"),
                     [(set Int8Regs:$dst, (OpNode Int8Regs:$a, (imm):$b))]>;
}

multiclass I3_i8<string OpcStr, SDNode OpNode, string TypeStr, string CVTStr> {
  def i64rr : NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$a, Int64Regs:$b),
                     !strconcat(OpcStr, "64 \t$dst, $a, $b;"),
                     [(set Int64Regs:$dst, (OpNode Int64Regs:$a,
                       Int64Regs:$b))]>;
  def i64ri : NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$a, i64imm:$b),
                     !strconcat(OpcStr, "64 \t$dst, $a, $b;"),
                     [(set Int64Regs:$dst, (OpNode Int64Regs:$a, imm:$b))]>;
  def i32rr : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a, Int32Regs:$b),
                     !strconcat(OpcStr, "32 \t$dst, $a, $b;"),
                     [(set Int32Regs:$dst, (OpNode Int32Regs:$a,
                       Int32Regs:$b))]>;
  def i32ri : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a, i32imm:$b),
                     !strconcat(OpcStr, "32 \t$dst, $a, $b;"),
                     [(set Int32Regs:$dst, (OpNode Int32Regs:$a, imm:$b))]>;
  def i16rr : NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$a, Int16Regs:$b),
                     !strconcat(OpcStr, "16 \t$dst, $a, $b;"),
                     [(set Int16Regs:$dst, (OpNode Int16Regs:$a,
                       Int16Regs:$b))]>;
  def i16ri : NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$a, i16imm:$b),
                     !strconcat(OpcStr, "16 \t$dst, $a, $b;"),
                     [(set Int16Regs:$dst, (OpNode Int16Regs:$a, (imm):$b))]>;
  def i8rr : NVPTXInst<(outs Int8Regs:$dst), (ins Int8Regs:$a, Int8Regs:$b),
                     Handle_i8rr<OpcStr, TypeStr, CVTStr>.s,
                     [(set Int8Regs:$dst, (OpNode Int8Regs:$a, Int8Regs:$b))]>;
  def i8ri : NVPTXInst<(outs Int8Regs:$dst), (ins Int8Regs:$a, i8imm:$b),
                     Handle_i8ri<OpcStr, TypeStr, CVTStr>.s,
                     [(set Int8Regs:$dst, (OpNode Int8Regs:$a, (imm):$b))]>;
}

multiclass I3_noi8<string OpcStr, SDNode OpNode> {
  def i64rr : NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$a, Int64Regs:$b),
                     !strconcat(OpcStr, "64 \t$dst, $a, $b;"),
                     [(set Int64Regs:$dst, (OpNode Int64Regs:$a,
                       Int64Regs:$b))]>;
  def i64ri : NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$a, i64imm:$b),
                     !strconcat(OpcStr, "64 \t$dst, $a, $b;"),
                     [(set Int64Regs:$dst, (OpNode Int64Regs:$a, imm:$b))]>;
  def i32rr : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a, Int32Regs:$b),
                     !strconcat(OpcStr, "32 \t$dst, $a, $b;"),
                     [(set Int32Regs:$dst, (OpNode Int32Regs:$a,
                       Int32Regs:$b))]>;
  def i32ri : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a, i32imm:$b),
                     !strconcat(OpcStr, "32 \t$dst, $a, $b;"),
                     [(set Int32Regs:$dst, (OpNode Int32Regs:$a, imm:$b))]>;
  def i16rr : NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$a, Int16Regs:$b),
                     !strconcat(OpcStr, "16 \t$dst, $a, $b;"),
                     [(set Int16Regs:$dst, (OpNode Int16Regs:$a,
                       Int16Regs:$b))]>;
  def i16ri : NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$a, i16imm:$b),
                     !strconcat(OpcStr, "16 \t$dst, $a, $b;"),
                     [(set Int16Regs:$dst, (OpNode Int16Regs:$a, (imm):$b))]>;
}

multiclass ADD_SUB_INT_32<string OpcStr, SDNode OpNode> {
   def i32rr : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a,
       Int32Regs:$b),
                      !strconcat(OpcStr, ".s32 \t$dst, $a, $b;"),
                      [(set Int32Regs:$dst, (OpNode Int32Regs:$a,
                        Int32Regs:$b))]>;
   def i32ri : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a, i32imm:$b),
                      !strconcat(OpcStr, ".s32 \t$dst, $a, $b;"),
                      [(set Int32Regs:$dst, (OpNode Int32Regs:$a, imm:$b))]>;
}

multiclass F3<string OpcStr, SDNode OpNode> {
   def f64rr : NVPTXInst<(outs Float64Regs:$dst),
                      (ins Float64Regs:$a, Float64Regs:$b),
                      !strconcat(OpcStr, ".f64 \t$dst, $a, $b;"),
                      [(set Float64Regs:$dst,
                        (OpNode Float64Regs:$a, Float64Regs:$b))]>,
                      Requires<[allowFMA]>;
   def f64ri : NVPTXInst<(outs Float64Regs:$dst),
                      (ins Float64Regs:$a, f64imm:$b),
                      !strconcat(OpcStr, ".f64 \t$dst, $a, $b;"),
                      [(set Float64Regs:$dst,
                        (OpNode Float64Regs:$a, fpimm:$b))]>,
                      Requires<[allowFMA]>;
   def f32rr_ftz : NVPTXInst<(outs Float32Regs:$dst),
                      (ins Float32Regs:$a, Float32Regs:$b),
                      !strconcat(OpcStr, ".ftz.f32 \t$dst, $a, $b;"),
                      [(set Float32Regs:$dst,
                        (OpNode Float32Regs:$a, Float32Regs:$b))]>,
                      Requires<[allowFMA_ftz]>;
   def f32ri_ftz : NVPTXInst<(outs Float32Regs:$dst),
                      (ins Float32Regs:$a, f32imm:$b),
                      !strconcat(OpcStr, ".ftz.f32 \t$dst, $a, $b;"),
                      [(set Float32Regs:$dst,
                        (OpNode Float32Regs:$a, fpimm:$b))]>,
                      Requires<[allowFMA_ftz]>;
   def f32rr : NVPTXInst<(outs Float32Regs:$dst),
                      (ins Float32Regs:$a, Float32Regs:$b),
                      !strconcat(OpcStr, ".f32 \t$dst, $a, $b;"),
                      [(set Float32Regs:$dst,
                        (OpNode Float32Regs:$a, Float32Regs:$b))]>,
                      Requires<[allowFMA]>;
   def f32ri : NVPTXInst<(outs Float32Regs:$dst),
                      (ins Float32Regs:$a, f32imm:$b),
                      !strconcat(OpcStr, ".f32 \t$dst, $a, $b;"),
                      [(set Float32Regs:$dst,
                        (OpNode Float32Regs:$a, fpimm:$b))]>,
                      Requires<[allowFMA]>;
}

multiclass F3_rn<string OpcStr, SDNode OpNode> {
   def f64rr : NVPTXInst<(outs Float64Regs:$dst),
                      (ins Float64Regs:$a, Float64Regs:$b),
                      !strconcat(OpcStr, ".rn.f64 \t$dst, $a, $b;"),
                      [(set Float64Regs:$dst,
                        (OpNode Float64Regs:$a, Float64Regs:$b))]>;
   def f64ri : NVPTXInst<(outs Float64Regs:$dst),
                      (ins Float64Regs:$a, f64imm:$b),
                      !strconcat(OpcStr, ".rn.f64 \t$dst, $a, $b;"),
                      [(set Float64Regs:$dst,
                        (OpNode Float64Regs:$a, fpimm:$b))]>;
   def f32rr_ftz : NVPTXInst<(outs Float32Regs:$dst),
                      (ins Float32Regs:$a, Float32Regs:$b),
                      !strconcat(OpcStr, ".rn.ftz.f32 \t$dst, $a, $b;"),
                      [(set Float32Regs:$dst,
                        (OpNode Float32Regs:$a, Float32Regs:$b))]>,
                      Requires<[doF32FTZ]>;
   def f32ri_ftz : NVPTXInst<(outs Float32Regs:$dst),
                      (ins Float32Regs:$a, f32imm:$b),
                      !strconcat(OpcStr, ".rn.ftz.f32 \t$dst, $a, $b;"),
                      [(set Float32Regs:$dst,
                        (OpNode Float32Regs:$a, fpimm:$b))]>,
                      Requires<[doF32FTZ]>;
   def f32rr : NVPTXInst<(outs Float32Regs:$dst),
                      (ins Float32Regs:$a, Float32Regs:$b),
                      !strconcat(OpcStr, ".rn.f32 \t$dst, $a, $b;"),
                      [(set Float32Regs:$dst,
                        (OpNode Float32Regs:$a, Float32Regs:$b))]>;
   def f32ri : NVPTXInst<(outs Float32Regs:$dst),
                      (ins Float32Regs:$a, f32imm:$b),
                      !strconcat(OpcStr, ".rn.f32 \t$dst, $a, $b;"),
                      [(set Float32Regs:$dst,
                        (OpNode Float32Regs:$a, fpimm:$b))]>;
}

multiclass F2<string OpcStr, SDNode OpNode> {
   def f64 : NVPTXInst<(outs Float64Regs:$dst), (ins Float64Regs:$a),
                      !strconcat(OpcStr, ".f64 \t$dst, $a;"),
                      [(set Float64Regs:$dst, (OpNode Float64Regs:$a))]>;
   def f32_ftz : NVPTXInst<(outs Float32Regs:$dst), (ins Float32Regs:$a),
                      !strconcat(OpcStr, ".ftz.f32 \t$dst, $a;"),
                      [(set Float32Regs:$dst, (OpNode Float32Regs:$a))]>,
                      Requires<[doF32FTZ]>;
   def f32 : NVPTXInst<(outs Float32Regs:$dst), (ins Float32Regs:$a),
                      !strconcat(OpcStr, ".f32 \t$dst, $a;"),
                      [(set Float32Regs:$dst, (OpNode Float32Regs:$a))]>;
}

//===----------------------------------------------------------------------===//
// NVPTX Instructions.
//===----------------------------------------------------------------------===//

//-----------------------------------
// Integer Arithmetic
//-----------------------------------

multiclass ADD_SUB_i1<SDNode OpNode> {
   def _rr: NVPTXInst<(outs Int1Regs:$dst), (ins Int1Regs:$a, Int1Regs:$b),
          "xor.pred \t$dst, $a, $b;",
      [(set Int1Regs:$dst, (OpNode Int1Regs:$a, Int1Regs:$b))]>;
   def _ri: NVPTXInst<(outs Int1Regs:$dst), (ins Int1Regs:$a, i1imm:$b),
          "xor.pred \t$dst, $a, $b;",
      [(set Int1Regs:$dst, (OpNode Int1Regs:$a, (imm):$b))]>;
}

defm ADD_i1 : ADD_SUB_i1<add>;
defm SUB_i1 : ADD_SUB_i1<sub>;


defm ADD : I3<"add.s", add>;
defm SUB : I3<"sub.s", sub>;

defm ADDCC : ADD_SUB_INT_32<"add.cc", addc>;
defm SUBCC : ADD_SUB_INT_32<"sub.cc", subc>;

defm ADDCCC : ADD_SUB_INT_32<"addc.cc", adde>;
defm SUBCCC : ADD_SUB_INT_32<"subc.cc", sube>;

//mul.wide PTX instruction
def SInt32Const : PatLeaf<(imm), [{
  const APInt &v = N->getAPIntValue();
  if (v.isSignedIntN(32))
    return true;
  return false;
}]>;

def UInt32Const : PatLeaf<(imm), [{
  const APInt &v = N->getAPIntValue();
  if (v.isIntN(32))
    return true;
  return false;
}]>;

def SInt16Const : PatLeaf<(imm), [{
  const APInt &v = N->getAPIntValue();
  if (v.isSignedIntN(16))
    return true;
  return false;
}]>;

def UInt16Const : PatLeaf<(imm), [{
  const APInt &v = N->getAPIntValue();
  if (v.isIntN(16))
    return true;
  return false;
}]>;

def Int5Const : PatLeaf<(imm), [{
  const APInt &v = N->getAPIntValue();
  // Check if 0 <= v < 32
  // Only then the result from (x << v) will be i32
  if (v.sge(0) && v.slt(32))
    return true;
  return false;
}]>;

def Int4Const : PatLeaf<(imm), [{
  const APInt &v = N->getAPIntValue();
  // Check if 0 <= v < 16
  // Only then the result from (x << v) will be i16
  if (v.sge(0) && v.slt(16))
    return true;
  return false;
}]>;

def SHL2MUL32 : SDNodeXForm<imm, [{
  const APInt &v = N->getAPIntValue();
  APInt temp(32, 1);
  return CurDAG->getTargetConstant(temp.shl(v), MVT::i32);
}]>;

def SHL2MUL16 : SDNodeXForm<imm, [{
  const APInt &v = N->getAPIntValue();
  APInt temp(16, 1);
  return CurDAG->getTargetConstant(temp.shl(v), MVT::i16);
}]>;

def MULWIDES64 : NVPTXInst<(outs Int64Regs:$dst),
                           (ins Int32Regs:$a, Int32Regs:$b),
                           "mul.wide.s32 \t$dst, $a, $b;", []>;
def MULWIDES64Imm : NVPTXInst<(outs Int64Regs:$dst),
                            (ins Int32Regs:$a, i64imm:$b),
                           "mul.wide.s32 \t$dst, $a, $b;", []>;

def MULWIDEU64 : NVPTXInst<(outs Int64Regs:$dst),
                           (ins Int32Regs:$a, Int32Regs:$b),
                           "mul.wide.u32 \t$dst, $a, $b;", []>;
def MULWIDEU64Imm : NVPTXInst<(outs Int64Regs:$dst),
                            (ins Int32Regs:$a, i64imm:$b),
                           "mul.wide.u32 \t$dst, $a, $b;", []>;

def MULWIDES32 : NVPTXInst<(outs Int32Regs:$dst),
                            (ins Int16Regs:$a, Int16Regs:$b),
                           "mul.wide.s16 \t$dst, $a, $b;", []>;
def MULWIDES32Imm : NVPTXInst<(outs Int32Regs:$dst),
                            (ins Int16Regs:$a, i32imm:$b),
                           "mul.wide.s16 \t$dst, $a, $b;", []>;

def MULWIDEU32 : NVPTXInst<(outs Int32Regs:$dst),
                            (ins Int16Regs:$a, Int16Regs:$b),
                           "mul.wide.u16 \t$dst, $a, $b;", []>;
def MULWIDEU32Imm : NVPTXInst<(outs Int32Regs:$dst),
                            (ins Int16Regs:$a, i32imm:$b),
                           "mul.wide.u16 \t$dst, $a, $b;", []>;

def : Pat<(shl (sext Int32Regs:$a), (i32 Int5Const:$b)),
          (MULWIDES64Imm Int32Regs:$a, (SHL2MUL32 node:$b))>,
          Requires<[doMulWide]>;
def : Pat<(shl (zext Int32Regs:$a), (i32 Int5Const:$b)),
          (MULWIDEU64Imm Int32Regs:$a, (SHL2MUL32 node:$b))>,
          Requires<[doMulWide]>;

def : Pat<(shl (sext Int16Regs:$a), (i16 Int4Const:$b)),
          (MULWIDES32Imm Int16Regs:$a, (SHL2MUL16 node:$b))>,
          Requires<[doMulWide]>;
def : Pat<(shl (zext Int16Regs:$a), (i16 Int4Const:$b)),
          (MULWIDEU32Imm Int16Regs:$a, (SHL2MUL16 node:$b))>,
          Requires<[doMulWide]>;

def : Pat<(mul (sext Int32Regs:$a), (sext Int32Regs:$b)),
          (MULWIDES64 Int32Regs:$a, Int32Regs:$b)>,
          Requires<[doMulWide]>;
def : Pat<(mul (sext Int32Regs:$a), (i64 SInt32Const:$b)),
          (MULWIDES64Imm Int32Regs:$a, (i64 SInt32Const:$b))>,
          Requires<[doMulWide]>;

def : Pat<(mul (zext Int32Regs:$a), (zext Int32Regs:$b)),
          (MULWIDEU64 Int32Regs:$a, Int32Regs:$b)>, Requires<[doMulWide]>;
def : Pat<(mul (zext Int32Regs:$a), (i64 UInt32Const:$b)),
          (MULWIDEU64Imm Int32Regs:$a, (i64 UInt32Const:$b))>,
          Requires<[doMulWide]>;

def : Pat<(mul (sext Int16Regs:$a), (sext Int16Regs:$b)),
          (MULWIDES32 Int16Regs:$a, Int16Regs:$b)>, Requires<[doMulWide]>;
def : Pat<(mul (sext Int16Regs:$a), (i32 SInt16Const:$b)),
          (MULWIDES32Imm Int16Regs:$a, (i32 SInt16Const:$b))>,
          Requires<[doMulWide]>;

def : Pat<(mul (zext Int16Regs:$a), (zext Int16Regs:$b)),
          (MULWIDEU32 Int16Regs:$a, Int16Regs:$b)>, Requires<[doMulWide]>;
def : Pat<(mul (zext Int16Regs:$a), (i32 UInt16Const:$b)),
          (MULWIDEU32Imm Int16Regs:$a, (i32 UInt16Const:$b))>,
          Requires<[doMulWide]>;

defm MULT : I3<"mul.lo.s", mul>;

defm MULTHS : I3_noi8<"mul.hi.s", mulhs>;
defm MULTHU : I3_noi8<"mul.hi.u", mulhu>;
def MULTHSi8rr : NVPTXInst<(outs Int8Regs:$dst), (ins Int8Regs:$a, Int8Regs:$b),
            !strconcat("{{ \n\t",
            !strconcat(".reg \t.s16 temp1; \n\t",
            !strconcat(".reg \t.s16 temp2; \n\t",
            !strconcat("cvt.s16.s8 \ttemp1, $a; \n\t",
            !strconcat("cvt.s16.s8 \ttemp2, $b; \n\t",
            !strconcat("mul.lo.s16 \t$dst, temp1, temp2; \n\t",
            !strconcat("shr.s16 \t$dst, $dst, 8; \n\t",
            !strconcat("}}", "")))))))),
      [(set Int8Regs:$dst, (mulhs Int8Regs:$a, Int8Regs:$b))]>;
def MULTHSi8ri : NVPTXInst<(outs Int8Regs:$dst), (ins Int8Regs:$a, i8imm:$b),
            !strconcat("{{ \n\t",
            !strconcat(".reg \t.s16 temp1; \n\t",
            !strconcat(".reg \t.s16 temp2; \n\t",
            !strconcat("cvt.s16.s8 \ttemp1, $a; \n\t",
            !strconcat("mov.b16 \ttemp2, $b; \n\t",
            !strconcat("cvt.s16.s8 \ttemp2, temp2; \n\t",
            !strconcat("mul.lo.s16 \t$dst, temp1, temp2; \n\t",
            !strconcat("shr.s16 \t$dst, $dst, 8; \n\t",
            !strconcat("}}", ""))))))))),
      [(set Int8Regs:$dst, (mulhs Int8Regs:$a, imm:$b))]>;
def MULTHUi8rr : NVPTXInst<(outs Int8Regs:$dst), (ins Int8Regs:$a, Int8Regs:$b),
            !strconcat("{{ \n\t",
            !strconcat(".reg \t.u16 temp1; \n\t",
            !strconcat(".reg \t.u16 temp2; \n\t",
            !strconcat("cvt.u16.u8 \ttemp1, $a; \n\t",
            !strconcat("cvt.u16.u8 \ttemp2, $b; \n\t",
            !strconcat("mul.lo.u16 \t$dst, temp1, temp2; \n\t",
            !strconcat("shr.u16 \t$dst, $dst, 8; \n\t",
            !strconcat("}}", "")))))))),
      [(set Int8Regs:$dst, (mulhu Int8Regs:$a, Int8Regs:$b))]>;
def MULTHUi8ri : NVPTXInst<(outs Int8Regs:$dst), (ins Int8Regs:$a, i8imm:$b),
            !strconcat("{{ \n\t",
            !strconcat(".reg \t.u16 temp1; \n\t",
            !strconcat(".reg \t.u16 temp2; \n\t",
            !strconcat("cvt.u16.u8 \ttemp1, $a; \n\t",
            !strconcat("mov.b16 \ttemp2, $b; \n\t",
            !strconcat("cvt.u16.u8 \ttemp2, temp2; \n\t",
            !strconcat("mul.lo.u16 \t$dst, temp1, temp2; \n\t",
            !strconcat("shr.u16 \t$dst, $dst, 8; \n\t",
            !strconcat("}}", ""))))))))),
      [(set Int8Regs:$dst, (mulhu Int8Regs:$a, imm:$b))]>;


defm SDIV : I3_i8<"div.s", sdiv, "s16", "cvt.s16.s8">;
defm UDIV : I3_i8<"div.u", udiv, "u16", "cvt.u16.u8">;

defm SREM : I3_i8<"rem.s", srem, "s16", "cvt.s16.s8">;
// The ri version will not be selected as DAGCombiner::visitSREM will lower it.
defm UREM : I3_i8<"rem.u", urem, "u16", "cvt.u16.u8">;
// The ri version will not be selected as DAGCombiner::visitUREM will lower it.

def MAD8rrr : NVPTXInst<(outs Int8Regs:$dst),
                      (ins Int8Regs:$a, Int8Regs:$b, Int8Regs:$c),
                      "mad.lo.s16 \t$dst, $a, $b, $c;",
                      [(set Int8Regs:$dst, (add (mul Int8Regs:$a, Int8Regs:$b),
                        Int8Regs:$c))]>;
def MAD8rri : NVPTXInst<(outs Int8Regs:$dst),
                      (ins Int8Regs:$a, Int8Regs:$b, i8imm:$c),
                      "mad.lo.s16 \t$dst, $a, $b, $c;",
                      [(set Int8Regs:$dst, (add (mul Int8Regs:$a, Int8Regs:$b),
                        imm:$c))]>;
def MAD8rir : NVPTXInst<(outs Int8Regs:$dst),
                      (ins Int8Regs:$a, i8imm:$b, Int8Regs:$c),
                      "mad.lo.s16 \t$dst, $a, $b, $c;",
                      [(set Int8Regs:$dst, (add (mul Int8Regs:$a, imm:$b),
                        Int8Regs:$c))]>;
def MAD8rii : NVPTXInst<(outs Int8Regs:$dst),
                      (ins Int8Regs:$a, i8imm:$b, i8imm:$c),
                      "mad.lo.s16 \t$dst, $a, $b, $c;",
                      [(set Int8Regs:$dst, (add (mul Int8Regs:$a, imm:$b),
                        imm:$c))]>;

def MAD16rrr : NVPTXInst<(outs Int16Regs:$dst),
                      (ins Int16Regs:$a, Int16Regs:$b, Int16Regs:$c),
                      "mad.lo.s16 \t$dst, $a, $b, $c;",
                      [(set Int16Regs:$dst, (add
                        (mul Int16Regs:$a, Int16Regs:$b), Int16Regs:$c))]>;
def MAD16rri : NVPTXInst<(outs Int16Regs:$dst),
                      (ins Int16Regs:$a, Int16Regs:$b, i16imm:$c),
                      "mad.lo.s16 \t$dst, $a, $b, $c;",
                      [(set Int16Regs:$dst, (add
                        (mul Int16Regs:$a, Int16Regs:$b), imm:$c))]>;
def MAD16rir : NVPTXInst<(outs Int16Regs:$dst),
                      (ins Int16Regs:$a, i16imm:$b, Int16Regs:$c),
                      "mad.lo.s16 \t$dst, $a, $b, $c;",
                      [(set Int16Regs:$dst, (add
                        (mul Int16Regs:$a, imm:$b), Int16Regs:$c))]>;
def MAD16rii : NVPTXInst<(outs Int16Regs:$dst),
    (ins Int16Regs:$a, i16imm:$b, i16imm:$c),
                      "mad.lo.s16 \t$dst, $a, $b, $c;",
                      [(set Int16Regs:$dst, (add (mul Int16Regs:$a, imm:$b),
                        imm:$c))]>;

def MAD32rrr : NVPTXInst<(outs Int32Regs:$dst),
                      (ins Int32Regs:$a, Int32Regs:$b, Int32Regs:$c),
                      "mad.lo.s32 \t$dst, $a, $b, $c;",
                      [(set Int32Regs:$dst, (add
                        (mul Int32Regs:$a, Int32Regs:$b), Int32Regs:$c))]>;
def MAD32rri : NVPTXInst<(outs Int32Regs:$dst),
                      (ins Int32Regs:$a, Int32Regs:$b, i32imm:$c),
                      "mad.lo.s32 \t$dst, $a, $b, $c;",
                      [(set Int32Regs:$dst, (add
                        (mul Int32Regs:$a, Int32Regs:$b), imm:$c))]>;
def MAD32rir : NVPTXInst<(outs Int32Regs:$dst),
                      (ins Int32Regs:$a, i32imm:$b, Int32Regs:$c),
                      "mad.lo.s32 \t$dst, $a, $b, $c;",
                      [(set Int32Regs:$dst, (add
                        (mul Int32Regs:$a, imm:$b), Int32Regs:$c))]>;
def MAD32rii : NVPTXInst<(outs Int32Regs:$dst),
                      (ins Int32Regs:$a, i32imm:$b, i32imm:$c),
                      "mad.lo.s32 \t$dst, $a, $b, $c;",
                      [(set Int32Regs:$dst, (add
                        (mul Int32Regs:$a, imm:$b), imm:$c))]>;

def MAD64rrr : NVPTXInst<(outs Int64Regs:$dst),
                      (ins Int64Regs:$a, Int64Regs:$b, Int64Regs:$c),
                      "mad.lo.s64 \t$dst, $a, $b, $c;",
                      [(set Int64Regs:$dst, (add
                        (mul Int64Regs:$a, Int64Regs:$b), Int64Regs:$c))]>;
def MAD64rri : NVPTXInst<(outs Int64Regs:$dst),
                      (ins Int64Regs:$a, Int64Regs:$b, i64imm:$c),
                      "mad.lo.s64 \t$dst, $a, $b, $c;",
                      [(set Int64Regs:$dst, (add
                        (mul Int64Regs:$a, Int64Regs:$b), imm:$c))]>;
def MAD64rir : NVPTXInst<(outs Int64Regs:$dst),
                      (ins Int64Regs:$a, i64imm:$b, Int64Regs:$c),
                      "mad.lo.s64 \t$dst, $a, $b, $c;",
                      [(set Int64Regs:$dst, (add
                        (mul Int64Regs:$a, imm:$b), Int64Regs:$c))]>;
def MAD64rii : NVPTXInst<(outs Int64Regs:$dst),
                      (ins Int64Regs:$a, i64imm:$b, i64imm:$c),
                      "mad.lo.s64 \t$dst, $a, $b, $c;",
                      [(set Int64Regs:$dst, (add
                        (mul Int64Regs:$a, imm:$b), imm:$c))]>;


def INEG8 : NVPTXInst<(outs Int8Regs:$dst), (ins Int8Regs:$src),
                     !strconcat("cvt.s16.s8 \t$dst, $src;\n\t",
                                 "neg.s16 \t$dst, $dst;"),
         [(set Int8Regs:$dst, (ineg Int8Regs:$src))]>;
def INEG16 : NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$src),
                     "neg.s16 \t$dst, $src;",
         [(set Int16Regs:$dst, (ineg Int16Regs:$src))]>;
def INEG32 : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$src),
                     "neg.s32 \t$dst, $src;",
         [(set Int32Regs:$dst, (ineg Int32Regs:$src))]>;
def INEG64 : NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$src),
                     "neg.s64 \t$dst, $src;",
         [(set Int64Regs:$dst, (ineg Int64Regs:$src))]>;

//-----------------------------------
// Floating Point Arithmetic
//-----------------------------------

// Constant 1.0f
def FloatConst1 : PatLeaf<(fpimm), [{
    if (&(N->getValueAPF().getSemantics()) != &llvm::APFloat::IEEEsingle)
      return false;
    float f = (float)N->getValueAPF().convertToFloat();
    return (f==1.0f);
}]>;
// Constand (double)1.0
def DoubleConst1 : PatLeaf<(fpimm), [{
    if (&(N->getValueAPF().getSemantics()) != &llvm::APFloat::IEEEdouble)
      return false;
    double d = (double)N->getValueAPF().convertToDouble();
    return (d==1.0);
}]>;

defm FADD : F3<"add", fadd>;
defm FSUB : F3<"sub", fsub>;
defm FMUL : F3<"mul", fmul>;

defm FADD_rn : F3_rn<"add", fadd>;
defm FSUB_rn : F3_rn<"sub", fsub>;
defm FMUL_rn : F3_rn<"mul", fmul>;

defm FABS : F2<"abs", fabs>;
defm FNEG : F2<"neg", fneg>;
defm FSQRT : F2<"sqrt.rn", fsqrt>;

//
// F64 division
//
def FDIV641r : NVPTXInst<(outs Float64Regs:$dst),
                      (ins f64imm:$a, Float64Regs:$b),
                      "rcp.rn.f64 \t$dst, $b;",
                      [(set Float64Regs:$dst,
                        (fdiv DoubleConst1:$a, Float64Regs:$b))]>;
def FDIV64rr : NVPTXInst<(outs Float64Regs:$dst),
                      (ins Float64Regs:$a, Float64Regs:$b),
                      "div.rn.f64 \t$dst, $a, $b;",
                      [(set Float64Regs:$dst,
                        (fdiv Float64Regs:$a, Float64Regs:$b))]>;
def FDIV64ri : NVPTXInst<(outs Float64Regs:$dst),
                      (ins Float64Regs:$a, f64imm:$b),
                      "div.rn.f64 \t$dst, $a, $b;",
                      [(set Float64Regs:$dst,
                        (fdiv Float64Regs:$a, fpimm:$b))]>;

//
// F32 Approximate reciprocal
//
def FDIV321r_ftz : NVPTXInst<(outs Float32Regs:$dst),
                      (ins f32imm:$a, Float32Regs:$b),
                      "rcp.approx.ftz.f32 \t$dst, $b;",
                      [(set Float32Regs:$dst,
                        (fdiv FloatConst1:$a, Float32Regs:$b))]>,
                      Requires<[do_DIVF32_APPROX, doF32FTZ]>;
def FDIV321r : NVPTXInst<(outs Float32Regs:$dst),
                        (ins f32imm:$a, Float32Regs:$b),
                       "rcp.approx.f32 \t$dst, $b;",
                      [(set Float32Regs:$dst,
                        (fdiv FloatConst1:$a, Float32Regs:$b))]>,
                      Requires<[do_DIVF32_APPROX]>;
//
// F32 Approximate division
//
def FDIV32approxrr_ftz : NVPTXInst<(outs Float32Regs:$dst),
                      (ins Float32Regs:$a, Float32Regs:$b),
                      "div.approx.ftz.f32 \t$dst, $a, $b;",
                      [(set Float32Regs:$dst,
                        (fdiv Float32Regs:$a, Float32Regs:$b))]>,
                      Requires<[do_DIVF32_APPROX, doF32FTZ]>;
def FDIV32approxrr     : NVPTXInst<(outs Float32Regs:$dst),
                      (ins Float32Regs:$a, Float32Regs:$b),
                      "div.approx.f32 \t$dst, $a, $b;",
                      [(set Float32Regs:$dst,
                        (fdiv Float32Regs:$a, Float32Regs:$b))]>,
                      Requires<[do_DIVF32_APPROX]>;
//
// F32 Semi-accurate reciprocal
//
// rcp.approx gives the same result as div.full(1.0f, a) and is faster.
//
def FDIV321r_approx_ftz : NVPTXInst<(outs Float32Regs:$dst),
                      (ins f32imm:$a, Float32Regs:$b),
                      "rcp.approx.ftz.f32 \t$dst, $b;",
                      [(set Float32Regs:$dst,
                        (fdiv FloatConst1:$a, Float32Regs:$b))]>,
                      Requires<[do_DIVF32_FULL, doF32FTZ]>;
def FDIV321r_approx : NVPTXInst<(outs Float32Regs:$dst),
                      (ins f32imm:$a, Float32Regs:$b),
                      "rcp.approx.f32 \t$dst, $b;",
                      [(set Float32Regs:$dst,
                        (fdiv FloatConst1:$a, Float32Regs:$b))]>,
                      Requires<[do_DIVF32_FULL]>;
//
// F32 Semi-accurate division
//
def FDIV32rr_ftz : NVPTXInst<(outs Float32Regs:$dst),
                      (ins Float32Regs:$a, Float32Regs:$b),
                      "div.full.ftz.f32 \t$dst, $a, $b;",
                      [(set Float32Regs:$dst,
                        (fdiv Float32Regs:$a, Float32Regs:$b))]>,
                      Requires<[do_DIVF32_FULL, doF32FTZ]>;
def FDIV32ri_ftz : NVPTXInst<(outs Float32Regs:$dst),
                      (ins Float32Regs:$a, f32imm:$b),
                      "div.full.ftz.f32 \t$dst, $a, $b;",
                      [(set Float32Regs:$dst,
                        (fdiv Float32Regs:$a, fpimm:$b))]>,
                      Requires<[do_DIVF32_FULL, doF32FTZ]>;
def FDIV32rr : NVPTXInst<(outs Float32Regs:$dst),
                      (ins Float32Regs:$a, Float32Regs:$b),
                      "div.full.f32 \t$dst, $a, $b;",
                      [(set Float32Regs:$dst,
                        (fdiv Float32Regs:$a, Float32Regs:$b))]>,
                      Requires<[do_DIVF32_FULL]>;
def FDIV32ri : NVPTXInst<(outs Float32Regs:$dst),
                      (ins Float32Regs:$a, f32imm:$b),
                      "div.full.f32 \t$dst, $a, $b;",
                      [(set Float32Regs:$dst,
                        (fdiv Float32Regs:$a, fpimm:$b))]>,
                      Requires<[do_DIVF32_FULL]>;
//
// F32 Accurate reciprocal
//
def FDIV321r_prec_ftz : NVPTXInst<(outs Float32Regs:$dst),
                        (ins f32imm:$a, Float32Regs:$b),
                       "rcp.rn.ftz.f32 \t$dst, $b;",
                      [(set Float32Regs:$dst,
                        (fdiv FloatConst1:$a, Float32Regs:$b))]>,
                      Requires<[reqPTX20, doF32FTZ]>;
def FDIV321r_prec : NVPTXInst<(outs Float32Regs:$dst),
                      (ins f32imm:$a, Float32Regs:$b),
                       "rcp.rn.f32 \t$dst, $b;",
                      [(set Float32Regs:$dst,
                        (fdiv FloatConst1:$a, Float32Regs:$b))]>,
                      Requires<[reqPTX20]>;
//
// F32 Accurate division
//
def FDIV32rr_prec_ftz : NVPTXInst<(outs Float32Regs:$dst),
                      (ins Float32Regs:$a, Float32Regs:$b),
                      "div.rn.ftz.f32 \t$dst, $a, $b;",
                      [(set Float32Regs:$dst,
                        (fdiv Float32Regs:$a, Float32Regs:$b))]>,
                      Requires<[doF32FTZ, reqPTX20]>;
def FDIV32ri_prec_ftz : NVPTXInst<(outs Float32Regs:$dst),
                      (ins Float32Regs:$a, f32imm:$b),
                      "div.rn.ftz.f32 \t$dst, $a, $b;",
                      [(set Float32Regs:$dst,
                        (fdiv Float32Regs:$a, fpimm:$b))]>,
                      Requires<[doF32FTZ, reqPTX20]>;
def FDIV32rr_prec : NVPTXInst<(outs Float32Regs:$dst),
                      (ins Float32Regs:$a, Float32Regs:$b),
                      "div.rn.f32 \t$dst, $a, $b;",
                      [(set Float32Regs:$dst,
                        (fdiv Float32Regs:$a, Float32Regs:$b))]>,
                      Requires<[reqPTX20]>;
def FDIV32ri_prec : NVPTXInst<(outs Float32Regs:$dst),
                      (ins Float32Regs:$a, f32imm:$b),
                      "div.rn.f32 \t$dst, $a, $b;",
                      [(set Float32Regs:$dst,
                        (fdiv Float32Regs:$a, fpimm:$b))]>,
                      Requires<[reqPTX20]>;


multiclass FPCONTRACT32<string OpcStr, Predicate Pred> {
   def rrr : NVPTXInst<(outs Float32Regs:$dst),
                      (ins Float32Regs:$a, Float32Regs:$b, Float32Regs:$c),
                      !strconcat(OpcStr, " \t$dst, $a, $b, $c;"),
                      [(set Float32Regs:$dst, (fadd
                        (fmul Float32Regs:$a, Float32Regs:$b),
                        Float32Regs:$c))]>, Requires<[Pred]>;
   // This is to WAR a weird bug in Tablegen that does not automatically
   // generate the following permutated rule rrr2 from the above rrr.
   // So we explicitly add it here. This happens to FMA32 only.
   // See the comments at FMAD32 and FMA32 for more information.
   def rrr2 : NVPTXInst<(outs Float32Regs:$dst),
                        (ins Float32Regs:$a, Float32Regs:$b, Float32Regs:$c),
                      !strconcat(OpcStr, " \t$dst, $a, $b, $c;"),
                      [(set Float32Regs:$dst, (fadd Float32Regs:$c,
                        (fmul Float32Regs:$a, Float32Regs:$b)))]>,
                      Requires<[Pred]>;
   def rri : NVPTXInst<(outs Float32Regs:$dst),
                      (ins Float32Regs:$a, Float32Regs:$b, f32imm:$c),
                      !strconcat(OpcStr, " \t$dst, $a, $b, $c;"),
                      [(set Float32Regs:$dst, (fadd
                        (fmul Float32Regs:$a, Float32Regs:$b), fpimm:$c))]>,
                      Requires<[Pred]>;
   def rir : NVPTXInst<(outs Float32Regs:$dst),
                      (ins Float32Regs:$a, f32imm:$b, Float32Regs:$c),
                      !strconcat(OpcStr, " \t$dst, $a, $b, $c;"),
                      [(set Float32Regs:$dst, (fadd
                        (fmul Float32Regs:$a, fpimm:$b), Float32Regs:$c))]>,
                      Requires<[Pred]>;
   def rii : NVPTXInst<(outs Float32Regs:$dst),
                      (ins Float32Regs:$a, f32imm:$b, f32imm:$c),
                      !strconcat(OpcStr, " \t$dst, $a, $b, $c;"),
                      [(set Float32Regs:$dst, (fadd
                        (fmul Float32Regs:$a, fpimm:$b), fpimm:$c))]>,
                      Requires<[Pred]>;
}

multiclass FPCONTRACT64<string OpcStr, Predicate Pred> {
   def rrr : NVPTXInst<(outs Float64Regs:$dst),
                      (ins Float64Regs:$a, Float64Regs:$b, Float64Regs:$c),
                      !strconcat(OpcStr, " \t$dst, $a, $b, $c;"),
                      [(set Float64Regs:$dst, (fadd
                        (fmul Float64Regs:$a, Float64Regs:$b),
                        Float64Regs:$c))]>, Requires<[Pred]>;
   def rri : NVPTXInst<(outs Float64Regs:$dst),
                      (ins Float64Regs:$a, Float64Regs:$b, f64imm:$c),
                      !strconcat(OpcStr, " \t$dst, $a, $b, $c;"),
                      [(set Float64Regs:$dst, (fadd (fmul Float64Regs:$a,
                        Float64Regs:$b), fpimm:$c))]>, Requires<[Pred]>;
   def rir : NVPTXInst<(outs Float64Regs:$dst),
                      (ins Float64Regs:$a, f64imm:$b, Float64Regs:$c),
                      !strconcat(OpcStr, " \t$dst, $a, $b, $c;"),
                      [(set Float64Regs:$dst, (fadd
                        (fmul Float64Regs:$a, fpimm:$b), Float64Regs:$c))]>,
                      Requires<[Pred]>;
   def rii : NVPTXInst<(outs Float64Regs:$dst),
                      (ins Float64Regs:$a, f64imm:$b, f64imm:$c),
                      !strconcat(OpcStr, " \t$dst, $a, $b, $c;"),
                      [(set Float64Regs:$dst, (fadd
                        (fmul Float64Regs:$a, fpimm:$b), fpimm:$c))]>,
                      Requires<[Pred]>;
}

// Due to a unknown reason (most likely a bug in tablegen), tablegen does not
// automatically generate the rrr2 rule from
// the rrr rule (see FPCONTRACT32) for FMA32, though it does for FMAD32.
// If we reverse the order of the following two lines, then rrr2 rule will be
// generated for FMA32, but not for rrr.
// Therefore, we manually write the rrr2 rule in FPCONTRACT32.
defm FMAD32_ftz : FPCONTRACT32<"mad.ftz.f32", doFMADF32_ftz>;
defm FMAD32 : FPCONTRACT32<"mad.f32", doFMADF32>;
defm FMA32_ftz  : FPCONTRACT32<"fma.rn.ftz.f32", doFMAF32_ftz>;
defm FMA32  : FPCONTRACT32<"fma.rn.f32", doFMAF32>;
defm FMA64  : FPCONTRACT64<"fma.rn.f64", doFMAF64>;

// b*c-a => fmad(b, c, -a)
multiclass FPCONTRACT32_SUB_PAT_MAD<NVPTXInst Inst, Predicate Pred> {
  def : Pat<(fsub (fmul Float32Regs:$b, Float32Regs:$c), Float32Regs:$a),
          (Inst Float32Regs:$b, Float32Regs:$c, (FNEGf32 Float32Regs:$a))>,
          Requires<[Pred]>;
}

// a-b*c => fmad(-b,c, a)
// - legal because a-b*c <=> a+(-b*c) <=> a+(-b)*c
// b*c-a => fmad(b, c, -a)
// - legal because b*c-a <=> b*c+(-a)
multiclass FPCONTRACT32_SUB_PAT<NVPTXInst Inst, Predicate Pred> {
  def : Pat<(fsub Float32Regs:$a, (fmul Float32Regs:$b, Float32Regs:$c)),
          (Inst (FNEGf32 Float32Regs:$b), Float32Regs:$c, Float32Regs:$a)>,
          Requires<[Pred]>;
  def : Pat<(fsub (fmul Float32Regs:$b, Float32Regs:$c), Float32Regs:$a),
          (Inst Float32Regs:$b, Float32Regs:$c, (FNEGf32 Float32Regs:$a))>,
          Requires<[Pred]>;
}

// a-b*c => fmad(-b,c, a)
// b*c-a => fmad(b, c, -a)
multiclass FPCONTRACT64_SUB_PAT<NVPTXInst Inst, Predicate Pred> {
  def : Pat<(fsub Float64Regs:$a, (fmul Float64Regs:$b, Float64Regs:$c)),
          (Inst (FNEGf64 Float64Regs:$b), Float64Regs:$c, Float64Regs:$a)>,
          Requires<[Pred]>;

  def : Pat<(fsub (fmul Float64Regs:$b, Float64Regs:$c), Float64Regs:$a),
          (Inst Float64Regs:$b, Float64Regs:$c, (FNEGf64 Float64Regs:$a))>,
          Requires<[Pred]>;
}

defm FMAF32ext_ftz  : FPCONTRACT32_SUB_PAT<FMA32_ftzrrr, doFMAF32AGG_ftz>;
defm FMAF32ext  : FPCONTRACT32_SUB_PAT<FMA32rrr, doFMAF32AGG>;
defm FMADF32ext_ftz : FPCONTRACT32_SUB_PAT_MAD<FMAD32_ftzrrr, doFMADF32_ftz>;
defm FMADF32ext : FPCONTRACT32_SUB_PAT_MAD<FMAD32rrr, doFMADF32>;
defm FMAF64ext  : FPCONTRACT64_SUB_PAT<FMA64rrr, doFMAF64AGG>;

def SINF:  NVPTXInst<(outs Float32Regs:$dst), (ins Float32Regs:$src),
                      "sin.approx.f32 \t$dst, $src;",
                      [(set Float32Regs:$dst, (fsin Float32Regs:$src))]>;
def COSF:  NVPTXInst<(outs Float32Regs:$dst), (ins Float32Regs:$src),
                      "cos.approx.f32 \t$dst, $src;",
                      [(set Float32Regs:$dst, (fcos Float32Regs:$src))]>;

//-----------------------------------
// Logical Arithmetic
//-----------------------------------

multiclass LOG_FORMAT<string OpcStr, SDNode OpNode> {
  def b1rr:  NVPTXInst<(outs Int1Regs:$dst), (ins Int1Regs:$a, Int1Regs:$b),
                      !strconcat(OpcStr, ".pred  \t$dst, $a, $b;"),
                      [(set Int1Regs:$dst, (OpNode Int1Regs:$a, Int1Regs:$b))]>;
  def b1ri:  NVPTXInst<(outs Int1Regs:$dst), (ins Int1Regs:$a, i1imm:$b),
                      !strconcat(OpcStr, ".pred  \t$dst, $a, $b;"),
                      [(set Int1Regs:$dst, (OpNode Int1Regs:$a, imm:$b))]>;
  def b8rr:  NVPTXInst<(outs Int8Regs:$dst), (ins Int8Regs:$a, Int8Regs:$b),
                      !strconcat(OpcStr, ".b16  \t$dst, $a, $b;"),
                      [(set Int8Regs:$dst, (OpNode Int8Regs:$a, Int8Regs:$b))]>;
  def b8ri:  NVPTXInst<(outs Int8Regs:$dst), (ins Int8Regs:$a, i8imm:$b),
                      !strconcat(OpcStr, ".b16  \t$dst, $a, $b;"),
                      [(set Int8Regs:$dst, (OpNode Int8Regs:$a, imm:$b))]>;
  def b16rr:  NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$a, Int16Regs:$b),
                      !strconcat(OpcStr, ".b16  \t$dst, $a, $b;"),
                      [(set Int16Regs:$dst, (OpNode Int16Regs:$a,
                        Int16Regs:$b))]>;
  def b16ri:  NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$a, i16imm:$b),
                      !strconcat(OpcStr, ".b16  \t$dst, $a, $b;"),
                      [(set Int16Regs:$dst, (OpNode Int16Regs:$a, imm:$b))]>;
  def b32rr:  NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a, Int32Regs:$b),
                      !strconcat(OpcStr, ".b32  \t$dst, $a, $b;"),
                      [(set Int32Regs:$dst, (OpNode Int32Regs:$a,
                        Int32Regs:$b))]>;
  def b32ri:  NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a, i32imm:$b),
                      !strconcat(OpcStr, ".b32  \t$dst, $a, $b;"),
                      [(set Int32Regs:$dst, (OpNode Int32Regs:$a, imm:$b))]>;
  def b64rr:  NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$a, Int64Regs:$b),
                      !strconcat(OpcStr, ".b64  \t$dst, $a, $b;"),
                      [(set Int64Regs:$dst, (OpNode Int64Regs:$a,
                        Int64Regs:$b))]>;
  def b64ri:  NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$a, i64imm:$b),
                      !strconcat(OpcStr, ".b64  \t$dst, $a, $b;"),
                      [(set Int64Regs:$dst, (OpNode Int64Regs:$a, imm:$b))]>;
}

defm OR  : LOG_FORMAT<"or", or>;
defm AND : LOG_FORMAT<"and", and>;
defm XOR : LOG_FORMAT<"xor", xor>;

def NOT1:  NVPTXInst<(outs Int1Regs:$dst), (ins Int1Regs:$src),
                      "not.pred \t$dst, $src;",
                      [(set Int1Regs:$dst, (not Int1Regs:$src))]>;
def NOT8:  NVPTXInst<(outs Int8Regs:$dst), (ins Int8Regs:$src),
                      "not.b16 \t$dst, $src;",
                      [(set Int8Regs:$dst, (not Int8Regs:$src))]>;
def NOT16:  NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$src),
                      "not.b16 \t$dst, $src;",
                      [(set Int16Regs:$dst, (not Int16Regs:$src))]>;
def NOT32:  NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$src),
                      "not.b32 \t$dst, $src;",
                      [(set Int32Regs:$dst, (not Int32Regs:$src))]>;
def NOT64:  NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$src),
                      "not.b64 \t$dst, $src;",
                      [(set Int64Regs:$dst, (not Int64Regs:$src))]>;

// For shifts, the second src operand must be 32-bit value
multiclass LSHIFT_FORMAT<string OpcStr, SDNode OpNode> {
   def i64rr : NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$a,
                      Int32Regs:$b),
                      !strconcat(OpcStr, "64 \t$dst, $a, $b;"),
                      [(set Int64Regs:$dst, (OpNode Int64Regs:$a,
                        Int32Regs:$b))]>;
   def i64ri : NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$a, i32imm:$b),
                      !strconcat(OpcStr, "64 \t$dst, $a, $b;"),
                      [(set Int64Regs:$dst, (OpNode Int64Regs:$a,
                        (i32 imm:$b)))]>;
   def i32rr : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a,
                      Int32Regs:$b),
                      !strconcat(OpcStr, "32 \t$dst, $a, $b;"),
                      [(set Int32Regs:$dst, (OpNode Int32Regs:$a,
                        Int32Regs:$b))]>;
   def i32ri : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a, i32imm:$b),
                      !strconcat(OpcStr, "32 \t$dst, $a, $b;"),
                      [(set Int32Regs:$dst, (OpNode Int32Regs:$a,
                        (i32 imm:$b)))]>;
   def i32ii : NVPTXInst<(outs Int32Regs:$dst), (ins i32imm:$a, i32imm:$b),
                      !strconcat(OpcStr, "32 \t$dst, $a, $b;"),
                      [(set Int32Regs:$dst, (OpNode (i32 imm:$a),
                        (i32 imm:$b)))]>;
   def i16rr : NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$a,
                      Int32Regs:$b),
                      !strconcat(OpcStr, "16 \t$dst, $a, $b;"),
                      [(set Int16Regs:$dst, (OpNode Int16Regs:$a,
                        Int32Regs:$b))]>;
   def i16ri : NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$a, i32imm:$b),
                      !strconcat(OpcStr, "16 \t$dst, $a, $b;"),
                      [(set Int16Regs:$dst, (OpNode Int16Regs:$a,
                        (i32 imm:$b)))]>;
   def i8rr : NVPTXInst<(outs Int8Regs:$dst), (ins Int8Regs:$a, Int32Regs:$b),
                      !strconcat(OpcStr, "16 \t$dst, $a, $b;"),
                      [(set Int8Regs:$dst, (OpNode Int8Regs:$a,
                        Int32Regs:$b))]>;
   def i8ri : NVPTXInst<(outs Int8Regs:$dst), (ins Int8Regs:$a, i32imm:$b),
                      !strconcat(OpcStr, "16 \t$dst, $a, $b;"),
                      [(set Int8Regs:$dst, (OpNode Int8Regs:$a,
                        (i32 imm:$b)))]>;
}

defm SHL : LSHIFT_FORMAT<"shl.b", shl>;

// For shifts, the second src operand must be 32-bit value
// Need to add cvt for the 8-bits.
multiclass RSHIFT_FORMAT<string OpcStr, SDNode OpNode, string CVTStr> {
   def i64rr : NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$a,
                      Int32Regs:$b),
                      !strconcat(OpcStr, "64 \t$dst, $a, $b;"),
                      [(set Int64Regs:$dst, (OpNode Int64Regs:$a,
                        Int32Regs:$b))]>;
   def i64ri : NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$a, i32imm:$b),
                      !strconcat(OpcStr, "64 \t$dst, $a, $b;"),
                      [(set Int64Regs:$dst, (OpNode Int64Regs:$a,
                        (i32 imm:$b)))]>;
   def i32rr : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a,
                      Int32Regs:$b),
                      !strconcat(OpcStr, "32 \t$dst, $a, $b;"),
                      [(set Int32Regs:$dst, (OpNode Int32Regs:$a,
                        Int32Regs:$b))]>;
   def i32ri : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a, i32imm:$b),
                      !strconcat(OpcStr, "32 \t$dst, $a, $b;"),
                      [(set Int32Regs:$dst, (OpNode Int32Regs:$a,
                        (i32 imm:$b)))]>;
   def i32ii : NVPTXInst<(outs Int32Regs:$dst), (ins i32imm:$a, i32imm:$b),
                      !strconcat(OpcStr, "32 \t$dst, $a, $b;"),
                      [(set Int32Regs:$dst, (OpNode (i32 imm:$a),
                        (i32 imm:$b)))]>;
   def i16rr : NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$a,
                      Int32Regs:$b),
                      !strconcat(OpcStr, "16 \t$dst, $a, $b;"),
                      [(set Int16Regs:$dst, (OpNode Int16Regs:$a,
                        Int32Regs:$b))]>;
   def i16ri : NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$a, i32imm:$b),
                      !strconcat(OpcStr, "16 \t$dst, $a, $b;"),
                      [(set Int16Regs:$dst, (OpNode Int16Regs:$a,
                        (i32 imm:$b)))]>;
   def i8rr : NVPTXInst<(outs Int8Regs:$dst), (ins Int8Regs:$a, Int32Regs:$b),
                      !strconcat(CVTStr, !strconcat(" \t$dst, $a;\n\t",
                      !strconcat(OpcStr, "16 \t$dst, $dst, $b;"))),
                      [(set Int8Regs:$dst, (OpNode Int8Regs:$a,
                        Int32Regs:$b))]>;
   def i8ri : NVPTXInst<(outs Int8Regs:$dst), (ins Int8Regs:$a, i32imm:$b),
                      !strconcat(CVTStr, !strconcat(" \t$dst, $a;\n\t",
                      !strconcat(OpcStr, "16 \t$dst, $dst, $b;"))),
                      [(set Int8Regs:$dst, (OpNode Int8Regs:$a,
                        (i32 imm:$b)))]>;
}

defm SRA : RSHIFT_FORMAT<"shr.s", sra, "cvt.s16.s8">;
defm SRL : RSHIFT_FORMAT<"shr.u", srl, "cvt.u16.u8">;

// 32bit
def ROT32imm_sw : NVPTXInst<(outs Int32Regs:$dst),
  (ins Int32Regs:$src, i32imm:$amt1, i32imm:$amt2),
    !strconcat("{{\n\t",
    !strconcat(".reg .b32 %lhs;\n\t",
    !strconcat(".reg .b32 %rhs;\n\t",
    !strconcat("shl.b32 \t%lhs, $src, $amt1;\n\t",
    !strconcat("shr.b32 \t%rhs, $src, $amt2;\n\t",
    !strconcat("add.u32 \t$dst, %lhs, %rhs;\n\t",
    !strconcat("}}", ""))))))),
    []>;

def SUB_FRM_32 : SDNodeXForm<imm, [{
    return CurDAG->getTargetConstant(32-N->getZExtValue(), MVT::i32);
}]>;

def : Pat<(rotl Int32Regs:$src, (i32 imm:$amt)),
          (ROT32imm_sw Int32Regs:$src, imm:$amt, (SUB_FRM_32 node:$amt))>;
def : Pat<(rotr Int32Regs:$src, (i32 imm:$amt)),
          (ROT32imm_sw Int32Regs:$src, (SUB_FRM_32 node:$amt), imm:$amt)>;

def ROTL32reg_sw : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$src,
    Int32Regs:$amt),
    !strconcat("{{\n\t",
    !strconcat(".reg .b32 %lhs;\n\t",
    !strconcat(".reg .b32 %rhs;\n\t",
    !strconcat(".reg .b32 %amt2;\n\t",
    !strconcat("shl.b32 \t%lhs, $src, $amt;\n\t",
    !strconcat("sub.s32 \t%amt2, 32, $amt;\n\t",
    !strconcat("shr.b32 \t%rhs, $src, %amt2;\n\t",
    !strconcat("add.u32 \t$dst, %lhs, %rhs;\n\t",
    !strconcat("}}", ""))))))))),
    [(set Int32Regs:$dst, (rotl Int32Regs:$src, Int32Regs:$amt))]>;

def ROTR32reg_sw : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$src,
    Int32Regs:$amt),
    !strconcat("{{\n\t",
    !strconcat(".reg .b32 %lhs;\n\t",
    !strconcat(".reg .b32 %rhs;\n\t",
    !strconcat(".reg .b32 %amt2;\n\t",
    !strconcat("shr.b32 \t%lhs, $src, $amt;\n\t",
    !strconcat("sub.s32 \t%amt2, 32, $amt;\n\t",
    !strconcat("shl.b32 \t%rhs, $src, %amt2;\n\t",
    !strconcat("add.u32 \t$dst, %lhs, %rhs;\n\t",
    !strconcat("}}", ""))))))))),
    [(set Int32Regs:$dst, (rotr Int32Regs:$src, Int32Regs:$amt))]>;

// 64bit
def ROT64imm_sw : NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$src,
    i32imm:$amt1, i32imm:$amt2),
    !strconcat("{{\n\t",
    !strconcat(".reg .b64 %lhs;\n\t",
    !strconcat(".reg .b64 %rhs;\n\t",
    !strconcat("shl.b64 \t%lhs, $src, $amt1;\n\t",
    !strconcat("shr.b64 \t%rhs, $src, $amt2;\n\t",
    !strconcat("add.u64 \t$dst, %lhs, %rhs;\n\t",
    !strconcat("}}", ""))))))),
    []>;

def SUB_FRM_64 : SDNodeXForm<imm, [{
    return CurDAG->getTargetConstant(64-N->getZExtValue(), MVT::i32);
}]>;

def : Pat<(rotl Int64Regs:$src, (i32 imm:$amt)),
          (ROT64imm_sw Int64Regs:$src, imm:$amt, (SUB_FRM_64 node:$amt))>;
def : Pat<(rotr Int64Regs:$src, (i32 imm:$amt)),
          (ROT64imm_sw Int64Regs:$src, (SUB_FRM_64 node:$amt), imm:$amt)>;

def ROTL64reg_sw : NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$src,
    Int32Regs:$amt),
    !strconcat("{{\n\t",
    !strconcat(".reg .b64 %lhs;\n\t",
    !strconcat(".reg .b64 %rhs;\n\t",
    !strconcat(".reg .u32 %amt2;\n\t",
    !strconcat("shl.b64 \t%lhs, $src, $amt;\n\t",
    !strconcat("sub.u32 \t%amt2, 64, $amt;\n\t",
    !strconcat("shr.b64 \t%rhs, $src, %amt2;\n\t",
    !strconcat("add.u64 \t$dst, %lhs, %rhs;\n\t",
    !strconcat("}}", ""))))))))),
    [(set Int64Regs:$dst, (rotl Int64Regs:$src, Int32Regs:$amt))]>;

def ROTR64reg_sw : NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$src,
    Int32Regs:$amt),
    !strconcat("{{\n\t",
    !strconcat(".reg .b64 %lhs;\n\t",
    !strconcat(".reg .b64 %rhs;\n\t",
    !strconcat(".reg .u32 %amt2;\n\t",
    !strconcat("shr.b64 \t%lhs, $src, $amt;\n\t",
    !strconcat("sub.u32 \t%amt2, 64, $amt;\n\t",
    !strconcat("shl.b64 \t%rhs, $src, %amt2;\n\t",
    !strconcat("add.u64 \t$dst, %lhs, %rhs;\n\t",
    !strconcat("}}", ""))))))))),
    [(set Int64Regs:$dst, (rotr Int64Regs:$src, Int32Regs:$amt))]>;


//-----------------------------------
// Data Movement (Load / Store, Move)
//-----------------------------------

def ADDRri : ComplexPattern<i32, 2, "SelectADDRri", [frameindex],
  [SDNPWantRoot]>;
def ADDRri64 : ComplexPattern<i64, 2, "SelectADDRri64", [frameindex],
  [SDNPWantRoot]>;

def MEMri : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops Int32Regs, i32imm);
}
def MEMri64 : Operand<i64> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops Int64Regs, i64imm);
}

def imem : Operand<iPTR> {
    let PrintMethod = "printOperand";
}

def imemAny : Operand<iPTRAny> {
    let PrintMethod = "printOperand";
}

def LdStCode : Operand<i32> {
    let PrintMethod = "printLdStCode";
}

def SDTWrapper : SDTypeProfile<1, 1, [SDTCisSameAs<0, 1>, SDTCisPtrTy<0>]>;
def Wrapper    : SDNode<"NVPTXISD::Wrapper", SDTWrapper>;

def MOV_ADDR : NVPTXInst<(outs Int32Regs:$dst), (ins imem:$a),
                     "mov.u32 \t$dst, $a;",
                     [(set Int32Regs:$dst, (Wrapper tglobaladdr:$a))]>;

def MOV_ADDR64 : NVPTXInst<(outs Int64Regs:$dst), (ins imem:$a),
                     "mov.u64 \t$dst, $a;",
                     [(set Int64Regs:$dst, (Wrapper tglobaladdr:$a))]>;

// copyPhysreg is hard-coded in NVPTXInstrInfo.cpp
let IsSimpleMove=1 in {
def IMOV1rr: NVPTXInst<(outs Int1Regs:$dst), (ins Int1Regs:$sss),
                   "mov.pred \t$dst, $sss;", []>;
def IMOV8rr: NVPTXInst<(outs Int8Regs:$dst), (ins Int8Regs:$sss),
                    "mov.u16 \t$dst, $sss;", []>;
def IMOV16rr: NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$sss),
                    "mov.u16 \t$dst, $sss;", []>;
def IMOV32rr: NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$sss),
                    "mov.u32 \t$dst, $sss;", []>;
def IMOV64rr: NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$sss),
                    "mov.u64 \t$dst, $sss;", []>;

def FMOV32rr: NVPTXInst<(outs Float32Regs:$dst), (ins Float32Regs:$src),
                    "mov.f32 \t$dst, $src;", []>;
def FMOV64rr: NVPTXInst<(outs Float64Regs:$dst), (ins Float64Regs:$src),
                    "mov.f64 \t$dst, $src;", []>;
}
def IMOV1ri: NVPTXInst<(outs Int1Regs:$dst), (ins i1imm:$src),
                    "mov.pred \t$dst, $src;",
          [(set Int1Regs:$dst, imm:$src)]>;
def IMOV8ri: NVPTXInst<(outs Int8Regs:$dst), (ins i8imm:$src),
                    "mov.u16 \t$dst, $src;",
          [(set Int8Regs:$dst, imm:$src)]>;
def IMOV16ri: NVPTXInst<(outs Int16Regs:$dst), (ins i16imm:$src),
                    "mov.u16 \t$dst, $src;",
          [(set Int16Regs:$dst, imm:$src)]>;
def IMOV32ri: NVPTXInst<(outs Int32Regs:$dst), (ins i32imm:$src),
                    "mov.u32 \t$dst, $src;",
          [(set Int32Regs:$dst, imm:$src)]>;
def IMOV64i: NVPTXInst<(outs Int64Regs:$dst), (ins i64imm:$src),
                    "mov.u64 \t$dst, $src;",
          [(set Int64Regs:$dst, imm:$src)]>;

def FMOV32ri: NVPTXInst<(outs Float32Regs:$dst), (ins f32imm:$src),
                    "mov.f32 \t$dst, $src;",
          [(set Float32Regs:$dst, fpimm:$src)]>;
def FMOV64ri: NVPTXInst<(outs Float64Regs:$dst), (ins f64imm:$src),
                    "mov.f64 \t$dst, $src;",
          [(set Float64Regs:$dst, fpimm:$src)]>;

def : Pat<(i32 (Wrapper texternalsym:$dst)), (IMOV32ri texternalsym:$dst)>;

//---- Copy Frame Index ----
def LEA_ADDRi : NVPTXInst<(outs Int32Regs:$dst), (ins MEMri:$addr),
                        "add.u32 \t$dst, ${addr:add};",
                        [(set Int32Regs:$dst, ADDRri:$addr)]>;
def LEA_ADDRi64 : NVPTXInst<(outs Int64Regs:$dst), (ins MEMri64:$addr),
                        "add.u64 \t$dst, ${addr:add};",
                        [(set Int64Regs:$dst, ADDRri64:$addr)]>;

//-----------------------------------
// Comparison and Selection
//-----------------------------------

// Generate string block like
// {
//   .reg .pred p;
//   setp.gt.s16 p, %a, %b;
//   selp.s16 %dst, -1, 0, p;
// }
// when OpcStr=setp.gt.s sz1=16 sz2=16 d=%dst a=%a b=%b
class Set_Str<string OpcStr, string sz1, string sz2, string d, string a,
  string b> {
  string t1  = "{{\n\t.reg .pred p;\n\t";
  string t2  = !strconcat(t1 , OpcStr);
  string t3  = !strconcat(t2 , sz1);
  string t4  = !strconcat(t3 , " \tp, ");
  string t5  = !strconcat(t4 , a);
  string t6  = !strconcat(t5 , ", ");
  string t7  = !strconcat(t6 , b);
  string t8  = !strconcat(t7 , ";\n\tselp.s");
  string t9  = !strconcat(t8 , sz2);
  string t10 = !strconcat(t9, " \t");
  string t11 = !strconcat(t10, d);
  string s   = !strconcat(t11, ", -1, 0, p;\n\t}}");
}

// Generate string block like
// {
//   .reg .pred p;
//   .reg .s16 %temp1;
//   .reg .s16 %temp2;
//   cvt.s16.s8 %temp1, %a;
//   cvt s16.s8 %temp1, %b;
//   setp.gt.s16 p, %temp1, %temp2;
//   selp.s16 %dst, -1, 0, p;
// }
// when OpcStr=setp.gt.s d=%dst a=%a b=%b type=s16 cvt=cvt.s16.s8
class Set_Stri8<string OpcStr, string d, string a, string b, string type,
  string cvt> {
  string t1  = "{{\n\t.reg .pred p;\n\t";
  string t2  = !strconcat(t1, ".reg .");
  string t3  = !strconcat(t2, type);
  string t4  = !strconcat(t3, " %temp1;\n\t");
  string t5  = !strconcat(t4, ".reg .");
  string t6  = !strconcat(t5, type);
  string t7  = !strconcat(t6, " %temp2;\n\t");
  string t8  = !strconcat(t7, cvt);
  string t9  = !strconcat(t8, " \t%temp1, ");
  string t10 = !strconcat(t9, a);
  string t11 = !strconcat(t10, ";\n\t");
  string t12 = !strconcat(t11, cvt);
  string t13 = !strconcat(t12, " \t%temp2, ");
  string t14 = !strconcat(t13, b);
  string t15 = !strconcat(t14, ";\n\t");
  string t16 = !strconcat(t15, OpcStr);
  string t17 = !strconcat(t16, "16");
  string t18 = !strconcat(t17, " \tp, %temp1, %temp2;\n\t");
  string t19 = !strconcat(t18, "selp.s16 \t");
  string t20 = !strconcat(t19, d);
  string s   = !strconcat(t20, ", -1, 0, p;\n\t}}");
}

multiclass ISET_FORMAT<string OpcStr, string OpcStr_u32, PatFrag OpNode,
  string TypeStr, string CVTStr> {
  def i8rr_toi8: NVPTXInst<(outs Int8Regs:$dst), (ins Int8Regs:$a, Int8Regs:$b),
                     Set_Stri8<OpcStr, "$dst", "$a", "$b", TypeStr, CVTStr>.s,
               []>;
  def i16rr_toi16: NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$a,
      Int16Regs:$b),
                     Set_Str<OpcStr, "16", "16", "$dst", "$a", "$b">.s,
               []>;
  def i32rr_toi32: NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a,
      Int32Regs:$b),
                     Set_Str<OpcStr, "32", "32", "$dst", "$a", "$b">.s,
               []>;
  def i64rr_toi64: NVPTXInst<(outs Int64Regs:$dst), (ins Int64Regs:$a,
      Int64Regs:$b),
                     Set_Str<OpcStr, "64", "64", "$dst", "$a", "$b">.s,
               []>;

  def i8rr_p: NVPTXInst<(outs Int1Regs:$dst), (ins Int8Regs:$a, Int8Regs:$b),
                     Handle_i8rr<OpcStr, TypeStr, CVTStr>.s,
               [(set Int1Regs:$dst, (OpNode Int8Regs:$a, Int8Regs:$b))]>;
  def i8ri_p: NVPTXInst<(outs Int1Regs:$dst), (ins Int8Regs:$a, i8imm:$b),
                     Handle_i8ri<OpcStr, TypeStr, CVTStr>.s,
               [(set Int1Regs:$dst, (OpNode Int8Regs:$a, imm:$b))]>;
  def i8ir_p: NVPTXInst<(outs Int1Regs:$dst), (ins i8imm:$a, Int8Regs:$b),
                     Handle_i8ir<OpcStr, TypeStr, CVTStr>.s,
               [(set Int1Regs:$dst, (OpNode imm:$a, Int8Regs:$b))]>;
  def i16rr_p: NVPTXInst<(outs Int1Regs:$dst), (ins Int16Regs:$a, Int16Regs:$b),
                 !strconcat(OpcStr, "16 \t$dst, $a, $b;"),
               [(set Int1Regs:$dst, (OpNode Int16Regs:$a, Int16Regs:$b))]>;
  def i16ri_p: NVPTXInst<(outs Int1Regs:$dst), (ins Int16Regs:$a, i16imm:$b),
                 !strconcat(OpcStr, "16 \t$dst, $a, $b;"),
               [(set Int1Regs:$dst, (OpNode Int16Regs:$a, imm:$b))]>;
  def i16ir_p: NVPTXInst<(outs Int1Regs:$dst), (ins i16imm:$a, Int16Regs:$b),
                 !strconcat(OpcStr, "16 \t$dst, $a, $b;"),
               [(set Int1Regs:$dst, (OpNode imm:$a, Int16Regs:$b))]>;
  def i32rr_p: NVPTXInst<(outs Int1Regs:$dst), (ins Int32Regs:$a, Int32Regs:$b),
                 !strconcat(OpcStr, "32 \t$dst, $a, $b;"),
               [(set Int1Regs:$dst, (OpNode Int32Regs:$a, Int32Regs:$b))]>;
  def i32ri_p: NVPTXInst<(outs Int1Regs:$dst), (ins Int32Regs:$a, i32imm:$b),
                 !strconcat(OpcStr, "32 \t$dst, $a, $b;"),
               [(set Int1Regs:$dst, (OpNode Int32Regs:$a, imm:$b))]>;
  def i32ir_p: NVPTXInst<(outs Int1Regs:$dst), (ins i32imm:$a, Int32Regs:$b),
                 !strconcat(OpcStr, "32 \t$dst, $a, $b;"),
               [(set Int1Regs:$dst, (OpNode imm:$a, Int32Regs:$b))]>;
  def i64rr_p: NVPTXInst<(outs Int1Regs:$dst), (ins Int64Regs:$a, Int64Regs:$b),
                 !strconcat(OpcStr, "64 \t$dst, $a, $b;"),
               [(set Int1Regs:$dst, (OpNode Int64Regs:$a, Int64Regs:$b))]>;
  def i64ri_p: NVPTXInst<(outs Int1Regs:$dst), (ins Int64Regs:$a, i64imm:$b),
                 !strconcat(OpcStr, "64 \t$dst, $a, $b;"),
               [(set Int1Regs:$dst, (OpNode Int64Regs:$a, imm:$b))]>;
  def i64ir_p: NVPTXInst<(outs Int1Regs:$dst), (ins i64imm:$a, Int64Regs:$b),
                 !strconcat(OpcStr, "64 \t$dst, $a, $b;"),
               [(set Int1Regs:$dst, (OpNode imm:$a, Int64Regs:$b))]>;

  def i8rr_u32: NVPTXInst<(outs Int32Regs:$dst), (ins Int8Regs:$a, Int8Regs:$b),
                     Handle_i8rr<OpcStr_u32, TypeStr, CVTStr>.s,
               [(set Int32Regs:$dst, (OpNode Int8Regs:$a, Int8Regs:$b))]>;
  def i8ri_u32: NVPTXInst<(outs Int32Regs:$dst), (ins Int8Regs:$a, i8imm:$b),
                     Handle_i8ri<OpcStr_u32, TypeStr, CVTStr>.s,
               [(set Int32Regs:$dst, (OpNode Int8Regs:$a, imm:$b))]>;
  def i8ir_u32: NVPTXInst<(outs Int32Regs:$dst), (ins i8imm:$a, Int8Regs:$b),
                     Handle_i8ir<OpcStr_u32, TypeStr, CVTStr>.s,
               [(set Int32Regs:$dst, (OpNode imm:$a, Int8Regs:$b))]>;
  def i16rr_u32: NVPTXInst<(outs Int32Regs:$dst), (ins Int16Regs:$a,
      Int16Regs:$b),
                 !strconcat(OpcStr_u32, "16 \t$dst, $a, $b;"),
               [(set Int32Regs:$dst, (OpNode Int16Regs:$a, Int16Regs:$b))]>;
  def i16ri_u32: NVPTXInst<(outs Int32Regs:$dst), (ins Int16Regs:$a, i16imm:$b),
                 !strconcat(OpcStr_u32, "16 \t$dst, $a, $b;"),
               [(set Int32Regs:$dst, (OpNode Int16Regs:$a, imm:$b))]>;
  def i16ir_u32: NVPTXInst<(outs Int32Regs:$dst), (ins i16imm:$a, Int16Regs:$b),
                 !strconcat(OpcStr_u32, "16 \t$dst, $a, $b;"),
               [(set Int32Regs:$dst, (OpNode imm:$a, Int16Regs:$b))]>;
  def i32rr_u32: NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a,
      Int32Regs:$b),
                 !strconcat(OpcStr_u32, "32 \t$dst, $a, $b;"),
               [(set Int32Regs:$dst, (OpNode Int32Regs:$a, Int32Regs:$b))]>;
  def i32ri_u32: NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$a, i32imm:$b),
                 !strconcat(OpcStr_u32, "32 \t$dst, $a, $b;"),
               [(set Int32Regs:$dst, (OpNode Int32Regs:$a, imm:$b))]>;
  def i32ir_u32: NVPTXInst<(outs Int32Regs:$dst), (ins i32imm:$a, Int32Regs:$b),
                 !strconcat(OpcStr_u32, "32 \t$dst, $a, $b;"),
               [(set Int32Regs:$dst, (OpNode imm:$a, Int32Regs:$b))]>;
  def i64rr_u32: NVPTXInst<(outs Int32Regs:$dst), (ins Int64Regs:$a,
      Int64Regs:$b),
                 !strconcat(OpcStr_u32, "64 \t$dst, $a, $b;"),
               [(set Int32Regs:$dst, (OpNode Int64Regs:$a, Int64Regs:$b))]>;
  def i64ri_u32: NVPTXInst<(outs Int32Regs:$dst), (ins Int64Regs:$a, i64imm:$b),
                 !strconcat(OpcStr_u32, "64 \t$dst, $a, $b;"),
               [(set Int32Regs:$dst, (OpNode Int64Regs:$a, imm:$b))]>;
  def i64ir_u32: NVPTXInst<(outs Int32Regs:$dst), (ins i64imm:$a, Int64Regs:$b),
                 !strconcat(OpcStr_u32, "64 \t$dst, $a, $b;"),
               [(set Int32Regs:$dst, (OpNode imm:$a, Int64Regs:$b))]>;
}

multiclass FSET_FORMAT<string OpcStr, string OpcStr_u32, PatFrag OpNode> {
  def f32rr_toi32_ftz: NVPTXInst<(outs Int32Regs:$dst), (ins Float32Regs:$a,
      Float32Regs:$b),
                     Set_Str<OpcStr, "ftz.f32", "32", "$dst", "$a", "$b">.s,
               []>, Requires<[doF32FTZ]>;
  def f32rr_toi32: NVPTXInst<(outs Int32Regs:$dst), (ins Float32Regs:$a,
      Float32Regs:$b),
                     Set_Str<OpcStr, "f32", "32", "$dst", "$a", "$b">.s,
               []>;
  def f64rr_toi64: NVPTXInst<(outs Int64Regs:$dst), (ins Float64Regs:$a,
      Float64Regs:$b),
                     Set_Str<OpcStr, "f64", "64", "$dst", "$a", "$b">.s,
               []>;
  def f64rr_toi32: NVPTXInst<(outs Int32Regs:$dst), (ins Float64Regs:$a,
      Float64Regs:$b),
                     Set_Str<OpcStr, "f64", "32", "$dst", "$a", "$b">.s,
               []>;

  def f32rr_p_ftz: NVPTXInst<(outs Int1Regs:$dst), (ins Float32Regs:$a
      , Float32Regs:$b),
                 !strconcat(OpcStr, "ftz.f32 \t$dst, $a, $b;"),
               [(set Int1Regs:$dst, (OpNode Float32Regs:$a, Float32Regs:$b))]>
  , Requires<[doF32FTZ]>;
  def f32rr_p: NVPTXInst<(outs Int1Regs:$dst),
    (ins Float32Regs:$a, Float32Regs:$b),
                 !strconcat(OpcStr, "f32 \t$dst, $a, $b;"),
               [(set Int1Regs:$dst, (OpNode Float32Regs:$a, Float32Regs:$b))]>;
  def f32ri_p_ftz: NVPTXInst<(outs Int1Regs:$dst),
    (ins Float32Regs:$a, f32imm:$b),
                 !strconcat(OpcStr, "ftz.f32 \t$dst, $a, $b;"),
               [(set Int1Regs:$dst, (OpNode Float32Regs:$a, fpimm:$b))]>,
  Requires<[doF32FTZ]>;
  def f32ri_p: NVPTXInst<(outs Int1Regs:$dst), (ins Float32Regs:$a, f32imm:$b),
                 !strconcat(OpcStr, "f32 \t$dst, $a, $b;"),
               [(set Int1Regs:$dst, (OpNode Float32Regs:$a, fpimm:$b))]>;
  def f32ir_p_ftz: NVPTXInst<(outs Int1Regs:$dst),
    (ins f32imm:$a, Float32Regs:$b),
                 !strconcat(OpcStr, "ftz.f32 \t$dst, $a, $b;"),
               [(set Int1Regs:$dst, (OpNode fpimm:$a, Float32Regs:$b))]>,
  Requires<[doF32FTZ]>;
  def f32ir_p: NVPTXInst<(outs Int1Regs:$dst), (ins f32imm:$a, Float32Regs:$b),
                 !strconcat(OpcStr, "f32 \t$dst, $a, $b;"),
               [(set Int1Regs:$dst, (OpNode fpimm:$a, Float32Regs:$b))]>;
  def f64rr_p: NVPTXInst<(outs Int1Regs:$dst),
    (ins Float64Regs:$a, Float64Regs:$b),
                 !strconcat(OpcStr, "f64 \t$dst, $a, $b;"),
               [(set Int1Regs:$dst, (OpNode Float64Regs:$a, Float64Regs:$b))]>;
  def f64ri_p: NVPTXInst<(outs Int1Regs:$dst), (ins Float64Regs:$a, f64imm:$b),
                 !strconcat(OpcStr, "f64 \t$dst, $a, $b;"),
               [(set Int1Regs:$dst, (OpNode Float64Regs:$a, fpimm:$b))]>;
  def f64ir_p: NVPTXInst<(outs Int1Regs:$dst), (ins f64imm:$a, Float64Regs:$b),
                 !strconcat(OpcStr, "f64 \t$dst, $a, $b;"),
               [(set Int1Regs:$dst, (OpNode fpimm:$a, Float64Regs:$b))]>;

  def f32rr_u32_ftz: NVPTXInst<(outs Int32Regs:$dst),
    (ins Float32Regs:$a, Float32Regs:$b),
                 !strconcat(OpcStr_u32, "ftz.f32 \t$dst, $a, $b;"),
               [(set Int32Regs:$dst, (OpNode Float32Regs:$a, Float32Regs:$b))]>;
  def f32rr_u32: NVPTXInst<(outs Int32Regs:$dst),
    (ins Float32Regs:$a, Float32Regs:$b),
                 !strconcat(OpcStr_u32, "f32 \t$dst, $a, $b;"),
               [(set Int32Regs:$dst, (OpNode Float32Regs:$a, Float32Regs:$b))]>;
  def f32ri_u32_ftz: NVPTXInst<(outs Int32Regs:$dst),
    (ins Float32Regs:$a, f32imm:$b),
                 !strconcat(OpcStr_u32, "ftz.f32 \t$dst, $a, $b;"),
               [(set Int32Regs:$dst, (OpNode Float32Regs:$a, fpimm:$b))]>;
  def f32ri_u32: NVPTXInst<(outs Int32Regs:$dst),
    (ins Float32Regs:$a, f32imm:$b),
                 !strconcat(OpcStr_u32, "f32 \t$dst, $a, $b;"),
               [(set Int32Regs:$dst, (OpNode Float32Regs:$a, fpimm:$b))]>;
  def f32ir_u32_ftz: NVPTXInst<(outs Int32Regs:$dst),
    (ins f32imm:$a, Float32Regs:$b),
                 !strconcat(OpcStr_u32, "ftz.f32 \t$dst, $a, $b;"),
               [(set Int32Regs:$dst, (OpNode fpimm:$a, Float32Regs:$b))]>;
  def f32ir_u32: NVPTXInst<(outs Int32Regs:$dst),
    (ins f32imm:$a, Float32Regs:$b),
                 !strconcat(OpcStr_u32, "f32 \t$dst, $a, $b;"),
               [(set Int32Regs:$dst, (OpNode fpimm:$a, Float32Regs:$b))]>;
  def f64rr_u32: NVPTXInst<(outs Int32Regs:$dst),
    (ins Float64Regs:$a, Float64Regs:$b),
                 !strconcat(OpcStr_u32, "f64 \t$dst, $a, $b;"),
               [(set Int32Regs:$dst, (OpNode Float64Regs:$a, Float64Regs:$b))]>;
  def f64ri_u32: NVPTXInst<(outs Int32Regs:$dst),
    (ins Float64Regs:$a, f64imm:$b),
                 !strconcat(OpcStr_u32, "f64 \t$dst, $a, $b;"),
               [(set Int32Regs:$dst, (OpNode Float64Regs:$a, fpimm:$b))]>;
  def f64ir_u32: NVPTXInst<(outs Int32Regs:$dst),
    (ins f64imm:$a, Float64Regs:$b),
                 !strconcat(OpcStr_u32, "f64 \t$dst, $a, $b;"),
               [(set Int32Regs:$dst, (OpNode fpimm:$a, Float64Regs:$b))]>;
}

defm ISetSGT
: ISET_FORMAT<"setp.gt.s", "set.gt.u32.s", setgt, "s16", "cvt.s16.s8">;
defm ISetUGT
: ISET_FORMAT<"setp.gt.u", "set.gt.u32.u", setugt, "u16", "cvt.u16.u8">;
defm ISetSLT
: ISET_FORMAT<"setp.lt.s", "set.lt.u32.s", setlt, "s16", "cvt.s16.s8">;
defm ISetULT
: ISET_FORMAT<"setp.lt.u", "set.lt.u32.u", setult, "u16", "cvt.u16.u8">;
defm ISetSGE
: ISET_FORMAT<"setp.ge.s", "set.ge.u32.s", setge, "s16", "cvt.s16.s8">;
defm ISetUGE
: ISET_FORMAT<"setp.ge.u", "set.ge.u32.u", setuge, "u16", "cvt.u16.u8">;
defm ISetSLE
: ISET_FORMAT<"setp.le.s", "set.le.u32.s", setle, "s16", "cvt.s16.s8">;
defm ISetULE
: ISET_FORMAT<"setp.le.u", "set.le.u32.u", setule, "u16", "cvt.u16.u8">;
defm ISetSEQ
: ISET_FORMAT<"setp.eq.s", "set.eq.u32.s", seteq, "s16", "cvt.s16.s8">;
defm ISetUEQ
: ISET_FORMAT<"setp.eq.u", "set.eq.u32.u", setueq, "u16", "cvt.u16.u8">;
defm ISetSNE
: ISET_FORMAT<"setp.ne.s", "set.ne.u32.s", setne, "s16", "cvt.s16.s8">;
defm ISetUNE
: ISET_FORMAT<"setp.ne.u", "set.ne.u32.u", setune, "u16", "cvt.u16.u8">;

def ISetSNEi1rr_p : NVPTXInst<(outs Int1Regs:$dst),
  (ins Int1Regs:$a, Int1Regs:$b),
                      "xor.pred \t$dst, $a, $b;",
            [(set Int1Regs:$dst, (setne Int1Regs:$a, Int1Regs:$b))]>;
def ISetUNEi1rr_p : NVPTXInst<(outs Int1Regs:$dst),
  (ins Int1Regs:$a, Int1Regs:$b),
                      "xor.pred \t$dst, $a, $b;",
            [(set Int1Regs:$dst, (setune Int1Regs:$a, Int1Regs:$b))]>;
def ISetSEQi1rr_p : NVPTXInst<(outs Int1Regs:$dst),
  (ins Int1Regs:$a, Int1Regs:$b),
            !strconcat("{{\n\t",
            !strconcat(".reg .pred temp;\n\t",
            !strconcat("xor.pred \ttemp, $a, $b;\n\t",
            !strconcat("not.pred \t$dst, temp;\n\t}}","")))),
            [(set Int1Regs:$dst, (seteq Int1Regs:$a, Int1Regs:$b))]>;
def ISetUEQi1rr_p : NVPTXInst<(outs Int1Regs:$dst),
  (ins Int1Regs:$a, Int1Regs:$b),
            !strconcat("{{\n\t",
            !strconcat(".reg .pred temp;\n\t",
            !strconcat("xor.pred \ttemp, $a, $b;\n\t",
            !strconcat("not.pred \t$dst, temp;\n\t}}","")))),
            [(set Int1Regs:$dst, (setueq Int1Regs:$a, Int1Regs:$b))]>;

// Compare 2 i1's and produce a u32
def ISETSNEi1rr_u32 : NVPTXInst<(outs Int32Regs:$dst),
  (ins Int1Regs:$a, Int1Regs:$b),
                  !strconcat("{{\n\t",
                  !strconcat(".reg .pred temp;\n\t",
                  !strconcat("xor.pred \ttemp, $a, $b;\n\t",
                  !strconcat("selp.u32 \t$dst, -1, 0, temp;", "\n\t}}")))),
                  [(set Int32Regs:$dst, (setne Int1Regs:$a, Int1Regs:$b))]>;
def ISETSEQi1rr_u32 : NVPTXInst<(outs Int32Regs:$dst),
  (ins Int1Regs:$a, Int1Regs:$b),
                  !strconcat("{{\n\t",
                  !strconcat(".reg .pred temp;\n\t",
                  !strconcat("xor.pred \ttemp, $a, $b;\n\t",
                  !strconcat("selp.u32 \t$dst, 0, -1, temp;", "\n\t}}")))),
                  [(set Int32Regs:$dst, (seteq Int1Regs:$a, Int1Regs:$b))]>;

defm FSetGT : FSET_FORMAT<"setp.gt.", "set.gt.u32.", setogt>;
defm FSetLT : FSET_FORMAT<"setp.lt.", "set.lt.u32.", setolt>;
defm FSetGE : FSET_FORMAT<"setp.ge.", "set.ge.u32.", setoge>;
defm FSetLE : FSET_FORMAT<"setp.le.", "set.le.u32.", setole>;
defm FSetEQ : FSET_FORMAT<"setp.eq.", "set.eq.u32.", setoeq>;
defm FSetNE : FSET_FORMAT<"setp.ne.", "set.ne.u32.", setone>;

defm FSetUGT : FSET_FORMAT<"setp.gtu.", "set.gtu.u32.", setugt>;
defm FSetULT : FSET_FORMAT<"setp.ltu.", "set.ltu.u32.",setult>;
defm FSetUGE : FSET_FORMAT<"setp.geu.", "set.geu.u32.",setuge>;
defm FSetULE : FSET_FORMAT<"setp.leu.", "set.leu.u32.",setule>;
defm FSetUEQ : FSET_FORMAT<"setp.equ.", "set.equ.u32.",setueq>;
defm FSetUNE : FSET_FORMAT<"setp.neu.", "set.neu.u32.",setune>;

defm FSetNUM : FSET_FORMAT<"setp.num.", "set.num.u32.",seto>;
defm FSetNAN : FSET_FORMAT<"setp.nan.", "set.nan.u32.",setuo>;

def SELECTi1rr : Pat<(i1 (select Int1Regs:$p, Int1Regs:$a, Int1Regs:$b)),
                     (ORb1rr (ANDb1rr Int1Regs:$p, Int1Regs:$a),
                             (ANDb1rr (NOT1 Int1Regs:$p), Int1Regs:$b))>;
def SELECTi8rr : NVPTXInst<(outs Int8Regs:$dst),
  (ins Int8Regs:$a, Int8Regs:$b, Int1Regs:$p),
                      "selp.b16 \t$dst, $a, $b, $p;",
      [(set Int8Regs:$dst, (select Int1Regs:$p, Int8Regs:$a, Int8Regs:$b))]>;
def SELECTi8ri : NVPTXInst<(outs Int8Regs:$dst),
  (ins Int8Regs:$a, i8imm:$b, Int1Regs:$p),
                      "selp.b16 \t$dst, $a, $b, $p;",
      [(set Int8Regs:$dst, (select Int1Regs:$p, Int8Regs:$a, imm:$b))]>;
def SELECTi8ir : NVPTXInst<(outs Int8Regs:$dst),
  (ins i8imm:$a, Int8Regs:$b, Int1Regs:$p),
                      "selp.b16 \t$dst, $a, $b, $p;",
      [(set Int8Regs:$dst, (select Int1Regs:$p, imm:$a, Int8Regs:$b))]>;
def SELECTi8ii : NVPTXInst<(outs Int8Regs:$dst),
  (ins i8imm:$a, i8imm:$b, Int1Regs:$p),
                      "selp.b16 \t$dst, $a, $b, $p;",
      [(set Int8Regs:$dst, (select Int1Regs:$p, imm:$a, imm:$b))]>;

def SELECTi16rr : NVPTXInst<(outs Int16Regs:$dst),
  (ins Int16Regs:$a, Int16Regs:$b, Int1Regs:$p),
                      "selp.b16 \t$dst, $a, $b, $p;",
      [(set Int16Regs:$dst, (select Int1Regs:$p, Int16Regs:$a, Int16Regs:$b))]>;
def SELECTi16ri : NVPTXInst<(outs Int16Regs:$dst),
  (ins Int16Regs:$a, i16imm:$b, Int1Regs:$p),
                      "selp.b16 \t$dst, $a, $b, $p;",
      [(set Int16Regs:$dst, (select Int1Regs:$p, Int16Regs:$a, imm:$b))]>;
def SELECTi16ir : NVPTXInst<(outs Int16Regs:$dst),
  (ins i16imm:$a, Int16Regs:$b, Int1Regs:$p),
                      "selp.b16 \t$dst, $a, $b, $p;",
      [(set Int16Regs:$dst, (select Int1Regs:$p, imm:$a, Int16Regs:$b))]>;
def SELECTi16ii : NVPTXInst<(outs Int16Regs:$dst),
  (ins i16imm:$a, i16imm:$b, Int1Regs:$p),
                      "selp.b16 \t$dst, $a, $b, $p;",
      [(set Int16Regs:$dst, (select Int1Regs:$p, imm:$a, imm:$b))]>;

def SELECTi32rr : NVPTXInst<(outs Int32Regs:$dst),
  (ins Int32Regs:$a, Int32Regs:$b, Int1Regs:$p),
                      "selp.b32 \t$dst, $a, $b, $p;",
      [(set Int32Regs:$dst, (select Int1Regs:$p, Int32Regs:$a, Int32Regs:$b))]>;
def SELECTi32ri : NVPTXInst<(outs Int32Regs:$dst),
  (ins Int32Regs:$a, i32imm:$b, Int1Regs:$p),
                      "selp.b32 \t$dst, $a, $b, $p;",
      [(set Int32Regs:$dst, (select Int1Regs:$p, Int32Regs:$a, imm:$b))]>;
def SELECTi32ir : NVPTXInst<(outs Int32Regs:$dst),
  (ins i32imm:$a, Int32Regs:$b, Int1Regs:$p),
                      "selp.b32 \t$dst, $a, $b, $p;",
      [(set Int32Regs:$dst, (select Int1Regs:$p, imm:$a, Int32Regs:$b))]>;
def SELECTi32ii : NVPTXInst<(outs Int32Regs:$dst),
  (ins i32imm:$a, i32imm:$b, Int1Regs:$p),
                      "selp.b32 \t$dst, $a, $b, $p;",
      [(set Int32Regs:$dst, (select Int1Regs:$p, imm:$a, imm:$b))]>;

def SELECTi64rr : NVPTXInst<(outs Int64Regs:$dst),
  (ins Int64Regs:$a, Int64Regs:$b, Int1Regs:$p),
                      "selp.b64 \t$dst, $a, $b, $p;",
      [(set Int64Regs:$dst, (select Int1Regs:$p, Int64Regs:$a, Int64Regs:$b))]>;
def SELECTi64ri : NVPTXInst<(outs Int64Regs:$dst),
  (ins Int64Regs:$a, i64imm:$b, Int1Regs:$p),
                      "selp.b64 \t$dst, $a, $b, $p;",
      [(set Int64Regs:$dst, (select Int1Regs:$p, Int64Regs:$a, imm:$b))]>;
def SELECTi64ir : NVPTXInst<(outs Int64Regs:$dst),
  (ins i64imm:$a, Int64Regs:$b, Int1Regs:$p),
                      "selp.b64 \t$dst, $a, $b, $p;",
      [(set Int64Regs:$dst, (select Int1Regs:$p, imm:$a, Int64Regs:$b))]>;
def SELECTi64ii : NVPTXInst<(outs Int64Regs:$dst),
  (ins i64imm:$a, i64imm:$b, Int1Regs:$p),
                      "selp.b64 \t$dst, $a, $b, $p;",
      [(set Int64Regs:$dst, (select Int1Regs:$p, imm:$a, imm:$b))]>;

def SELECTf32rr : NVPTXInst<(outs Float32Regs:$dst),
  (ins Float32Regs:$a, Float32Regs:$b, Int1Regs:$p),
                      "selp.f32 \t$dst, $a, $b, $p;",
      [(set Float32Regs:$dst,
        (select Int1Regs:$p, Float32Regs:$a, Float32Regs:$b))]>;
def SELECTf32ri : NVPTXInst<(outs Float32Regs:$dst),
  (ins Float32Regs:$a, f32imm:$b, Int1Regs:$p),
                      "selp.f32 \t$dst, $a, $b, $p;",
      [(set Float32Regs:$dst, (select Int1Regs:$p, Float32Regs:$a, fpimm:$b))]>;
def SELECTf32ir : NVPTXInst<(outs Float32Regs:$dst),
  (ins f32imm:$a, Float32Regs:$b, Int1Regs:$p),
                      "selp.f32 \t$dst, $a, $b, $p;",
      [(set Float32Regs:$dst, (select Int1Regs:$p, fpimm:$a, Float32Regs:$b))]>;
def SELECTf32ii : NVPTXInst<(outs Float32Regs:$dst),
  (ins f32imm:$a, f32imm:$b, Int1Regs:$p),
                      "selp.f32 \t$dst, $a, $b, $p;",
      [(set Float32Regs:$dst, (select Int1Regs:$p, fpimm:$a, fpimm:$b))]>;

def SELECTf64rr : NVPTXInst<(outs Float64Regs:$dst),
  (ins Float64Regs:$a, Float64Regs:$b, Int1Regs:$p),
                      "selp.f64 \t$dst, $a, $b, $p;",
      [(set Float64Regs:$dst,
        (select Int1Regs:$p, Float64Regs:$a, Float64Regs:$b))]>;
def SELECTf64ri : NVPTXInst<(outs Float64Regs:$dst),
  (ins Float64Regs:$a, f64imm:$b, Int1Regs:$p),
                      "selp.f64 \t$dst, $a, $b, $p;",
      [(set Float64Regs:$dst, (select Int1Regs:$p, Float64Regs:$a, fpimm:$b))]>;
def SELECTf64ir : NVPTXInst<(outs Float64Regs:$dst),
  (ins f64imm:$a, Float64Regs:$b, Int1Regs:$p),
                      "selp.f64 \t$dst, $a, $b, $p;",
      [(set Float64Regs:$dst, (select Int1Regs:$p, fpimm:$a, Float64Regs:$b))]>;
def SELECTf64ii : NVPTXInst<(outs Float64Regs:$dst),
  (ins f64imm:$a, f64imm:$b, Int1Regs:$p),
                      "selp.f64 \t $dst, $a, $b, $p;",
      [(set Float64Regs:$dst, (select Int1Regs:$p, fpimm:$a, fpimm:$b))]>;

//def ld_param         : SDNode<"NVPTXISD::LOAD_PARAM", SDTLoad,
//                        [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;

def SDTDeclareParamProfile : SDTypeProfile<0, 3, [SDTCisInt<0>, SDTCisInt<1>,
  SDTCisInt<2>]>;
def SDTDeclareScalarParamProfile : SDTypeProfile<0, 3, [SDTCisInt<0>,
  SDTCisInt<1>, SDTCisInt<2>]>;
def SDTLoadParamProfile : SDTypeProfile<1, 2, [SDTCisInt<1>, SDTCisInt<2>]>;
def SDTPrintCallProfile : SDTypeProfile<0, 1, [SDTCisInt<0>]>;
def SDTPrintCallUniProfile : SDTypeProfile<0, 1, [SDTCisInt<0>]>;
def SDTStoreParamProfile : SDTypeProfile<0, 3, [SDTCisInt<0>, SDTCisInt<1>]>;
def SDTStoreParam32Profile : SDTypeProfile<0, 3, [SDTCisInt<0>, SDTCisInt<1>]>;
def SDTCallArgProfile : SDTypeProfile<0, 2, [SDTCisInt<0>]>;
def SDTCallArgMarkProfile : SDTypeProfile<0, 0, []>;
def SDTCallVoidProfile : SDTypeProfile<0, 1, []>;
def SDTCallValProfile : SDTypeProfile<1, 0, []>;
def SDTMoveParamProfile : SDTypeProfile<1, 1, []>;
def SDTMoveRetvalProfile : SDTypeProfile<0, 1, []>;
def SDTStoreRetvalProfile : SDTypeProfile<0, 2, [SDTCisInt<0>]>;
def SDTPseudoUseParamProfile : SDTypeProfile<0, 1, []>;

def DeclareParam : SDNode<"NVPTXISD::DeclareParam", SDTDeclareParamProfile,
                       [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def DeclareScalarParam : SDNode<"NVPTXISD::DeclareScalarParam",
  SDTDeclareScalarParamProfile,
                       [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def DeclareRetParam : SDNode<"NVPTXISD::DeclareRetParam",
  SDTDeclareParamProfile,
                       [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def DeclareRet   : SDNode<"NVPTXISD::DeclareRet", SDTDeclareScalarParamProfile,
                       [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def LoadParam    : SDNode<"NVPTXISD::LoadParam", SDTLoadParamProfile,
                         [SDNPHasChain, SDNPMayLoad, SDNPOutGlue, SDNPInGlue]>;
def PrintCall    : SDNode<"NVPTXISD::PrintCall", SDTPrintCallProfile,
                       [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def PrintCallUni : SDNode<"NVPTXISD::PrintCallUni", SDTPrintCallUniProfile,
                       [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def StoreParam   : SDNode<"NVPTXISD::StoreParam", SDTStoreParamProfile,
                       [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def StoreParamU32 : SDNode<"NVPTXISD::StoreParamU32", SDTStoreParam32Profile,
                       [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def StoreParamS32 : SDNode<"NVPTXISD::StoreParamS32", SDTStoreParam32Profile,
                       [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def MoveToParam  : SDNode<"NVPTXISD::MoveToParam", SDTStoreParamProfile,
                       [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def CallArgBegin : SDNode<"NVPTXISD::CallArgBegin", SDTCallArgMarkProfile,
                       [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def CallArg      : SDNode<"NVPTXISD::CallArg", SDTCallArgProfile,
                       [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def LastCallArg  : SDNode<"NVPTXISD::LastCallArg", SDTCallArgProfile,
                       [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def CallArgEnd   : SDNode<"NVPTXISD::CallArgEnd", SDTCallVoidProfile,
                       [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def CallVoid     : SDNode<"NVPTXISD::CallVoid", SDTCallVoidProfile,
                       [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def Prototype    : SDNode<"NVPTXISD::Prototype", SDTCallVoidProfile,
                       [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def CallVal      : SDNode<"NVPTXISD::CallVal", SDTCallValProfile,
                       [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def MoveParam    : SDNode<"NVPTXISD::MoveParam", SDTMoveParamProfile,
                         []>;
def MoveRetval   : SDNode<"NVPTXISD::MoveRetval", SDTMoveRetvalProfile,
                         [SDNPHasChain, SDNPSideEffect]>;
def StoreRetval  : SDNode<"NVPTXISD::StoreRetval", SDTStoreRetvalProfile,
                         [SDNPHasChain, SDNPSideEffect]>;
def MoveToRetval : SDNode<"NVPTXISD::MoveToRetval", SDTStoreRetvalProfile,
                         [SDNPHasChain, SDNPSideEffect]>;
def PseudoUseParam : SDNode<"NVPTXISD::PseudoUseParam",
  SDTPseudoUseParamProfile,
                       [SDNPHasChain, SDNPOutGlue, SDNPInGlue, SDNPSideEffect]>;
def RETURNNode   : SDNode<"NVPTXISD::RETURN", SDTCallArgMarkProfile,
                         [SDNPHasChain, SDNPSideEffect]>;

class LoadParamMemInst<NVPTXRegClass regclass, string opstr> :
      NVPTXInst<(outs regclass:$dst), (ins i32imm:$b),
                !strconcat(!strconcat("ld.param", opstr),
                "\t$dst, [retval0+$b];"),
                [(set regclass:$dst, (LoadParam (i32 1), (i32 imm:$b)))]>;

class LoadParamRegInst<NVPTXRegClass regclass, string opstr> :
      NVPTXInst<(outs regclass:$dst), (ins i32imm:$b),
                !strconcat(!strconcat("mov", opstr),
                "\t$dst, retval$b;"),
                [(set regclass:$dst, (LoadParam (i32 0), (i32 imm:$b)))]>;

class StoreParamInst<NVPTXRegClass regclass, string opstr> :
      NVPTXInst<(outs), (ins regclass:$val, i32imm:$a, i32imm:$b),
                !strconcat(!strconcat("st.param", opstr),
                "\t[param$a+$b], $val;"),
                [(StoreParam (i32 imm:$a), (i32 imm:$b), regclass:$val)]>;

class MoveToParamInst<NVPTXRegClass regclass, string opstr> :
      NVPTXInst<(outs), (ins regclass:$val, i32imm:$a, i32imm:$b),
                !strconcat(!strconcat("mov", opstr),
                "\tparam$a, $val;"),
                [(MoveToParam (i32 imm:$a), (i32 imm:$b), regclass:$val)]>;

class StoreRetvalInst<NVPTXRegClass regclass, string opstr> :
      NVPTXInst<(outs), (ins regclass:$val, i32imm:$a),
                !strconcat(!strconcat("st.param", opstr),
                "\t[func_retval0+$a], $val;"),
                [(StoreRetval (i32 imm:$a), regclass:$val)]>;

class MoveToRetvalInst<NVPTXRegClass regclass, string opstr> :
      NVPTXInst<(outs), (ins i32imm:$num, regclass:$val),
                !strconcat(!strconcat("mov", opstr),
                "\tfunc_retval$num, $val;"),
                [(MoveToRetval (i32 imm:$num), regclass:$val)]>;

class MoveRetvalInst<NVPTXRegClass regclass, string opstr> :
      NVPTXInst<(outs), (ins regclass:$val),
                !strconcat(!strconcat("mov", opstr),
                "\tfunc_retval0, $val;"),
                [(MoveRetval regclass:$val)]>;

def PrintCallRetInst1 : NVPTXInst<(outs), (ins),
"call (retval0), ",
                                [(PrintCall (i32 1))]>;
def PrintCallRetInst2 : NVPTXInst<(outs), (ins),
"call (retval0, retval1), ",
                                [(PrintCall (i32 2))]>;
def PrintCallRetInst3 : NVPTXInst<(outs), (ins),
"call (retval0, retval1, retval2), ",
                                [(PrintCall (i32 3))]>;
def PrintCallRetInst4 : NVPTXInst<(outs), (ins),
"call (retval0, retval1, retval2, retval3), ",
                                [(PrintCall (i32 4))]>;
def PrintCallRetInst5 : NVPTXInst<(outs), (ins),
"call (retval0, retval1, retval2, retval3, retval4), ",
                                [(PrintCall (i32 5))]>;
def PrintCallRetInst6 : NVPTXInst<(outs), (ins),
"call (retval0, retval1, retval2, retval3, retval4, retval5), ",
                                [(PrintCall (i32 6))]>;
def PrintCallRetInst7 : NVPTXInst<(outs), (ins),
"call (retval0, retval1, retval2, retval3, retval4, retval5, retval6), ",
                                [(PrintCall (i32 7))]>;
def PrintCallRetInst8 : NVPTXInst<(outs), (ins),
!strconcat("call (retval0, retval1, retval2, retval3, retval4",
           ", retval5, retval6, retval7), "),
                                [(PrintCall (i32 8))]>;

def PrintCallNoRetInst : NVPTXInst<(outs), (ins), "call ",
                                [(PrintCall (i32 0))]>;

def PrintCallUniRetInst1 : NVPTXInst<(outs), (ins),
"call.uni (retval0), ",
                                [(PrintCallUni (i32 1))]>;
def PrintCallUniRetInst2 : NVPTXInst<(outs), (ins),
"call.uni (retval0, retval1), ",
                                [(PrintCallUni (i32 2))]>;
def PrintCallUniRetInst3 : NVPTXInst<(outs), (ins),
"call.uni (retval0, retval1, retval2), ",
                                [(PrintCallUni (i32 3))]>;
def PrintCallUniRetInst4 : NVPTXInst<(outs), (ins),
"call.uni (retval0, retval1, retval2, retval3), ",
                                [(PrintCallUni (i32 4))]>;
def PrintCallUniRetInst5 : NVPTXInst<(outs), (ins),
"call.uni (retval0, retval1, retval2, retval3, retval4), ",
                                [(PrintCallUni (i32 5))]>;
def PrintCallUniRetInst6 : NVPTXInst<(outs), (ins),
"call.uni (retval0, retval1, retval2, retval3, retval4, retval5), ",
                                [(PrintCallUni (i32 6))]>;
def PrintCallUniRetInst7 : NVPTXInst<(outs), (ins),
"call.uni (retval0, retval1, retval2, retval3, retval4, retval5, retval6), ",
                                [(PrintCallUni (i32 7))]>;
def PrintCallUniRetInst8 : NVPTXInst<(outs), (ins),
!strconcat("call.uni (retval0, retval1, retval2, retval3, retval4",
           ", retval5, retval6, retval7), "),
                                [(PrintCallUni (i32 8))]>;

def PrintCallUniNoRetInst : NVPTXInst<(outs), (ins), "call.uni ",
                                [(PrintCallUni (i32 0))]>;

def LoadParamMemI64    : LoadParamMemInst<Int64Regs, ".b64">;
def LoadParamMemI32    : LoadParamMemInst<Int32Regs, ".b32">;
def LoadParamMemI16    : LoadParamMemInst<Int16Regs, ".b16">;
def LoadParamMemI8     : LoadParamMemInst<Int8Regs, ".b8">;

//def LoadParamMemI16    : NVPTXInst<(outs Int16Regs:$dst), (ins i32imm:$b),
//                !strconcat("ld.param.b32\ttemp_param_reg, [retval0+$b];\n\t",
//                "cvt.u16.u32\t$dst, temp_param_reg;"),
//                [(set Int16Regs:$dst, (LoadParam (i32 1), (i32 imm:$b)))]>;
//def LoadParamMemI8     : NVPTXInst<(outs Int8Regs:$dst), (ins i32imm:$b),
//                !strconcat("ld.param.b32\ttemp_param_reg, [retval0+$b];\n\t",
//                "cvt.u16.u32\t$dst, temp_param_reg;"),
//                [(set Int8Regs:$dst, (LoadParam (i32 1), (i32 imm:$b)))]>;

def LoadParamMemF32    : LoadParamMemInst<Float32Regs, ".f32">;
def LoadParamMemF64    : LoadParamMemInst<Float64Regs, ".f64">;

def LoadParamRegI64    : LoadParamRegInst<Int64Regs, ".b64">;
def LoadParamRegI32    : LoadParamRegInst<Int32Regs, ".b32">;
def LoadParamRegI16    : NVPTXInst<(outs Int16Regs:$dst), (ins i32imm:$b),
                         "cvt.u16.u32\t$dst, retval$b;",
                         [(set Int16Regs:$dst,
                           (LoadParam (i32 0), (i32 imm:$b)))]>;
def LoadParamRegI8     : NVPTXInst<(outs Int8Regs:$dst), (ins i32imm:$b),
                         "cvt.u16.u32\t$dst, retval$b;",
                         [(set Int8Regs:$dst,
                           (LoadParam (i32 0), (i32 imm:$b)))]>;

def LoadParamRegF32    : LoadParamRegInst<Float32Regs, ".f32">;
def LoadParamRegF64    : LoadParamRegInst<Float64Regs, ".f64">;

def StoreParamI64    : StoreParamInst<Int64Regs, ".b64">;
def StoreParamI32    : StoreParamInst<Int32Regs, ".b32">;

def StoreParamI16    : NVPTXInst<(outs),
  (ins Int16Regs:$val, i32imm:$a, i32imm:$b),
                       "st.param.b16\t[param$a+$b], $val;",
           [(StoreParam (i32 imm:$a), (i32 imm:$b), Int16Regs:$val)]>;

def StoreParamI8     : NVPTXInst<(outs),
  (ins Int8Regs:$val, i32imm:$a, i32imm:$b),
                       "st.param.b8\t[param$a+$b], $val;",
                       [(StoreParam
                         (i32 imm:$a), (i32 imm:$b), Int8Regs:$val)]>;

def StoreParamS32I16 : NVPTXInst<(outs),
  (ins Int16Regs:$val, i32imm:$a, i32imm:$b),
                 !strconcat("cvt.s32.s16\ttemp_param_reg, $val;\n\t",
                            "st.param.b32\t[param$a+$b], temp_param_reg;"),
                 [(StoreParamS32 (i32 imm:$a), (i32 imm:$b), Int16Regs:$val)]>;
def StoreParamU32I16 : NVPTXInst<(outs),
  (ins Int16Regs:$val, i32imm:$a, i32imm:$b),
                 !strconcat("cvt.u32.u16\ttemp_param_reg, $val;\n\t",
                            "st.param.b32\t[param$a+$b], temp_param_reg;"),
                 [(StoreParamU32 (i32 imm:$a), (i32 imm:$b), Int16Regs:$val)]>;

def StoreParamU32I8   : NVPTXInst<(outs),
  (ins Int8Regs:$val, i32imm:$a, i32imm:$b),
                 !strconcat("cvt.u32.u8\ttemp_param_reg, $val;\n\t",
                            "st.param.b32\t[param$a+$b], temp_param_reg;"),
                 [(StoreParamU32 (i32 imm:$a), (i32 imm:$b), Int8Regs:$val)]>;
def StoreParamS32I8   : NVPTXInst<(outs),
  (ins Int8Regs:$val, i32imm:$a, i32imm:$b),
                 !strconcat("cvt.s32.s8\ttemp_param_reg, $val;\n\t",
                            "st.param.b32\t[param$a+$b], temp_param_reg;"),
                 [(StoreParamS32 (i32 imm:$a), (i32 imm:$b), Int8Regs:$val)]>;

def StoreParamF32    : StoreParamInst<Float32Regs, ".f32">;
def StoreParamF64    : StoreParamInst<Float64Regs, ".f64">;

def MoveToParamI64   : MoveToParamInst<Int64Regs, ".b64">;
def MoveToParamI32   : MoveToParamInst<Int32Regs, ".b32">;
def MoveToParamF64   : MoveToParamInst<Float64Regs, ".f64">;
def MoveToParamF32   : MoveToParamInst<Float32Regs, ".f32">;
def MoveToParamI16   : NVPTXInst<(outs),
  (ins Int16Regs:$val, i32imm:$a, i32imm:$b),
                   !strconcat("cvt.u32.u16\ttemp_param_reg, $val;\n\t",
                              "mov.b32\tparam$a, temp_param_reg;"),
                   [(MoveToParam (i32 imm:$a), (i32 imm:$b), Int16Regs:$val)]>;
def MoveToParamI8    : NVPTXInst<(outs),
  (ins Int8Regs:$val, i32imm:$a, i32imm:$b),
                   !strconcat("cvt.u32.u16\ttemp_param_reg, $val;\n\t",
                              "mov.b32\tparam$a, temp_param_reg;"),
                   [(MoveToParam (i32 imm:$a), (i32 imm:$b), Int8Regs:$val)]>;

def StoreRetvalI64    : StoreRetvalInst<Int64Regs, ".b64">;
def StoreRetvalI32    : StoreRetvalInst<Int32Regs, ".b32">;
def StoreRetvalI16    : StoreRetvalInst<Int16Regs, ".b16">;
def StoreRetvalI8     : StoreRetvalInst<Int8Regs, ".b8">;

//def StoreRetvalI16    : NVPTXInst<(outs), (ins Int16Regs:$val, i32imm:$a),
//     !strconcat("\{\n\t",
//     !strconcat(".reg .b32 temp_retval_reg;\n\t",
//     !strconcat("cvt.u32.u16\ttemp_retval_reg, $val;\n\t",
//                "st.param.b32\t[func_retval0+$a], temp_retval_reg;\n\t\}"))),
//     [(StoreRetval (i32 imm:$a), Int16Regs:$val)]>;
//def StoreRetvalI8     : NVPTXInst<(outs), (ins Int8Regs:$val, i32imm:$a),
//     !strconcat("\{\n\t",
//     !strconcat(".reg .b32 temp_retval_reg;\n\t",
//     !strconcat("cvt.u32.u16\ttemp_retval_reg, $val;\n\t",
//                "st.param.b32\t[func_retval0+$a], temp_retval_reg;\n\t\}"))),
//     [(StoreRetval (i32 imm:$a), Int8Regs:$val)]>;

def StoreRetvalF64    : StoreRetvalInst<Float64Regs, ".f64">;
def StoreRetvalF32    : StoreRetvalInst<Float32Regs, ".f32">;

def MoveRetvalI64    : MoveRetvalInst<Int64Regs, ".b64">;
def MoveRetvalI32    : MoveRetvalInst<Int32Regs, ".b32">;
def MoveRetvalI16    : MoveRetvalInst<Int16Regs, ".b16">;
def MoveRetvalI8     : MoveRetvalInst<Int8Regs, ".b8">;
def MoveRetvalF64    : MoveRetvalInst<Float64Regs, ".f64">;
def MoveRetvalF32    : MoveRetvalInst<Float32Regs, ".f32">;

def MoveToRetvalI64    : MoveToRetvalInst<Int64Regs, ".b64">;
def MoveToRetvalI32    : MoveToRetvalInst<Int32Regs, ".b32">;
def MoveToRetvalF64    : MoveToRetvalInst<Float64Regs, ".f64">;
def MoveToRetvalF32    : MoveToRetvalInst<Float32Regs, ".f32">;
def MoveToRetvalI16    : NVPTXInst<(outs), (ins i32imm:$num, Int16Regs:$val),
                         "cvt.u32.u16\tfunc_retval$num, $val;",
                         [(MoveToRetval (i32 imm:$num), Int16Regs:$val)]>;
def MoveToRetvalI8     : NVPTXInst<(outs), (ins i32imm:$num, Int8Regs:$val),
                         "cvt.u32.u16\tfunc_retval$num, $val;",
                         [(MoveToRetval (i32 imm:$num), Int8Regs:$val)]>;

def CallArgBeginInst : NVPTXInst<(outs), (ins), "(", [(CallArgBegin)]>;
def CallArgEndInst1  : NVPTXInst<(outs), (ins), ");", [(CallArgEnd (i32 1))]>;
def CallArgEndInst0  : NVPTXInst<(outs), (ins), ")", [(CallArgEnd (i32 0))]>;
def RETURNInst       : NVPTXInst<(outs), (ins), "ret;", [(RETURNNode)]>;

class CallArgInst<NVPTXRegClass regclass> :
      NVPTXInst<(outs), (ins regclass:$a), "$a, ",
                [(CallArg (i32 0), regclass:$a)]>;

class LastCallArgInst<NVPTXRegClass regclass> :
      NVPTXInst<(outs), (ins regclass:$a), "$a",
                [(LastCallArg (i32 0), regclass:$a)]>;

def CallArgI64     : CallArgInst<Int64Regs>;
def CallArgI32     : CallArgInst<Int32Regs>;
def CallArgI16     : CallArgInst<Int16Regs>;
def CallArgI8      : CallArgInst<Int8Regs>;

def CallArgF64     : CallArgInst<Float64Regs>;
def CallArgF32     : CallArgInst<Float32Regs>;

def LastCallArgI64 : LastCallArgInst<Int64Regs>;
def LastCallArgI32 : LastCallArgInst<Int32Regs>;
def LastCallArgI16 : LastCallArgInst<Int16Regs>;
def LastCallArgI8  : LastCallArgInst<Int8Regs>;

def LastCallArgF64 : LastCallArgInst<Float64Regs>;
def LastCallArgF32 : LastCallArgInst<Float32Regs>;

def CallArgI32imm : NVPTXInst<(outs), (ins i32imm:$a), "$a, ",
                              [(CallArg (i32 0), (i32 imm:$a))]>;
def LastCallArgI32imm : NVPTXInst<(outs), (ins i32imm:$a), "$a",
                              [(LastCallArg (i32 0), (i32 imm:$a))]>;

def CallArgParam : NVPTXInst<(outs), (ins i32imm:$a), "param$a, ",
                             [(CallArg (i32 1), (i32 imm:$a))]>;
def LastCallArgParam : NVPTXInst<(outs), (ins i32imm:$a), "param$a",
                             [(LastCallArg (i32 1), (i32 imm:$a))]>;

def CallVoidInst : NVPTXInst<(outs), (ins imem:$addr),
                             "$addr, ",
                             [(CallVoid (Wrapper tglobaladdr:$addr))]>;
def CallVoidInstReg : NVPTXInst<(outs), (ins Int32Regs:$addr),
                             "$addr, ",
                             [(CallVoid Int32Regs:$addr)]>;
def CallVoidInstReg64 : NVPTXInst<(outs), (ins Int64Regs:$addr),
                             "$addr, ",
                             [(CallVoid Int64Regs:$addr)]>;
def PrototypeInst : NVPTXInst<(outs), (ins i32imm:$val),
                             ", prototype_$val;",
                             [(Prototype (i32 imm:$val))]>;

def DeclareRetMemInst : NVPTXInst<(outs),
  (ins i32imm:$align, i32imm:$size, i32imm:$num),
         ".param .align $align .b8 retval$num[$size];",
         [(DeclareRetParam (i32 imm:$align), (i32 imm:$size), (i32 imm:$num))]>;
def DeclareRetScalarInst : NVPTXInst<(outs), (ins i32imm:$size, i32imm:$num),
         ".param .b$size retval$num;",
         [(DeclareRet (i32 1), (i32 imm:$size), (i32 imm:$num))]>;
def DeclareRetRegInst : NVPTXInst<(outs), (ins i32imm:$size, i32imm:$num),
         ".reg .b$size retval$num;",
         [(DeclareRet (i32 2), (i32 imm:$size), (i32 imm:$num))]>;

def DeclareParamInst : NVPTXInst<(outs),
  (ins i32imm:$align, i32imm:$a, i32imm:$size),
         ".param .align $align .b8 param$a[$size];",
         [(DeclareParam (i32 imm:$align), (i32 imm:$a), (i32 imm:$size))]>;
def DeclareScalarParamInst : NVPTXInst<(outs), (ins i32imm:$a, i32imm:$size),
         ".param .b$size param$a;",
         [(DeclareScalarParam (i32 imm:$a), (i32 imm:$size), (i32 0))]>;
def DeclareScalarRegInst : NVPTXInst<(outs), (ins i32imm:$a, i32imm:$size),
         ".reg .b$size param$a;",
         [(DeclareScalarParam (i32 imm:$a), (i32 imm:$size), (i32 1))]>;

class MoveParamInst<NVPTXRegClass regclass, string asmstr> :
      NVPTXInst<(outs regclass:$dst), (ins regclass:$src),
                !strconcat(!strconcat("mov", asmstr), "\t$dst, $src;"),
                [(set regclass:$dst, (MoveParam regclass:$src))]>;

def MoveParamI64 : MoveParamInst<Int64Regs, ".b64">;
def MoveParamI32 : MoveParamInst<Int32Regs, ".b32">;
def MoveParamI16 : NVPTXInst<(outs Int16Regs:$dst), (ins Int16Regs:$src),
                   "cvt.u16.u32\t$dst, $src;",
                   [(set Int16Regs:$dst, (MoveParam Int16Regs:$src))]>;
def MoveParamI8  : NVPTXInst<(outs Int8Regs:$dst), (ins Int8Regs:$src),
                   "cvt.u16.u32\t$dst, $src;",
                   [(set Int8Regs:$dst, (MoveParam Int8Regs:$src))]>;
def MoveParamF64 : MoveParamInst<Float64Regs, ".f64">;
def MoveParamF32 : MoveParamInst<Float32Regs, ".f32">;

class PseudoUseParamInst<NVPTXRegClass regclass> :
      NVPTXInst<(outs), (ins regclass:$src),
      "// Pseudo use of $src",
      [(PseudoUseParam regclass:$src)]>;

def PseudoUseParamI64 : PseudoUseParamInst<Int64Regs>;
def PseudoUseParamI32 : PseudoUseParamInst<Int32Regs>;
def PseudoUseParamI16 : PseudoUseParamInst<Int16Regs>;
def PseudoUseParamI8  : PseudoUseParamInst<Int8Regs>;
def PseudoUseParamF64 : PseudoUseParamInst<Float64Regs>;
def PseudoUseParamF32 : PseudoUseParamInst<Float32Regs>;


//
// Load / Store Handling
//
multiclass LD<NVPTXRegClass regclass> {
  def _avar : NVPTXInst<(outs regclass:$dst),
    (ins LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
      i32imm:$fromWidth, imem:$addr),
!strconcat("ld${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
           "$fromWidth \t$dst, [$addr];"), []>;
  def _areg : NVPTXInst<(outs regclass:$dst),
    (ins LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
      i32imm:$fromWidth, Int32Regs:$addr),
!strconcat("ld${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
           "$fromWidth \t$dst, [$addr];"), []>;
  def _areg_64 : NVPTXInst<(outs regclass:$dst),
    (ins LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
     i32imm:$fromWidth, Int64Regs:$addr),
     !strconcat("ld${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}$fromWidth",
                " \t$dst, [$addr];"), []>;
  def _ari : NVPTXInst<(outs regclass:$dst),
    (ins LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
      i32imm:$fromWidth, Int32Regs:$addr, i32imm:$offset),
!strconcat("ld${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
           "$fromWidth \t$dst, [$addr+$offset];"), []>;
  def _ari_64 : NVPTXInst<(outs regclass:$dst),
    (ins LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
     i32imm:$fromWidth, Int64Regs:$addr, i32imm:$offset),
    !strconcat("ld${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}$fromWidth",
               " \t$dst, [$addr+$offset];"), []>;
  def _asi : NVPTXInst<(outs regclass:$dst),
    (ins LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
      i32imm:$fromWidth, imem:$addr, i32imm:$offset),
!strconcat("ld${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
           "$fromWidth \t$dst, [$addr+$offset];"), []>;
}

let mayLoad=1, neverHasSideEffects=1 in {
defm LD_i8  : LD<Int8Regs>;
defm LD_i16 : LD<Int16Regs>;
defm LD_i32 : LD<Int32Regs>;
defm LD_i64 : LD<Int64Regs>;
defm LD_f32 : LD<Float32Regs>;
defm LD_f64 : LD<Float64Regs>;
}

multiclass ST<NVPTXRegClass regclass> {
  def _avar : NVPTXInst<(outs),
    (ins regclass:$src, LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec,
      LdStCode:$Sign, i32imm:$toWidth, imem:$addr),
!strconcat("st${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}$toWidth",
           " \t[$addr], $src;"), []>;
  def _areg : NVPTXInst<(outs),
    (ins regclass:$src, LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec,
      LdStCode:$Sign, i32imm:$toWidth, Int32Regs:$addr),
!strconcat("st${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}$toWidth",
           " \t[$addr], $src;"), []>;
  def _areg_64 : NVPTXInst<(outs),
    (ins regclass:$src, LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec,
     LdStCode:$Sign, i32imm:$toWidth, Int64Regs:$addr),
  !strconcat("st${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}$toWidth ",
               "\t[$addr], $src;"), []>;
  def _ari : NVPTXInst<(outs),
    (ins regclass:$src, LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec,
      LdStCode:$Sign, i32imm:$toWidth, Int32Regs:$addr, i32imm:$offset),
!strconcat("st${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}$toWidth",
           " \t[$addr+$offset], $src;"), []>;
  def _ari_64 : NVPTXInst<(outs),
    (ins regclass:$src, LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec,
     LdStCode:$Sign, i32imm:$toWidth, Int64Regs:$addr, i32imm:$offset),
  !strconcat("st${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}$toWidth ",
               "\t[$addr+$offset], $src;"), []>;
  def _asi : NVPTXInst<(outs),
    (ins regclass:$src, LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec,
      LdStCode:$Sign, i32imm:$toWidth, imem:$addr, i32imm:$offset),
!strconcat("st${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}$toWidth",
           " \t[$addr+$offset], $src;"), []>;
}

let mayStore=1, neverHasSideEffects=1 in {
defm ST_i8  : ST<Int8Regs>;
defm ST_i16 : ST<Int16Regs>;
defm ST_i32 : ST<Int32Regs>;
defm ST_i64 : ST<Int64Regs>;
defm ST_f32 : ST<Float32Regs>;
defm ST_f64 : ST<Float64Regs>;
}

// The following is used only in and after vector elementizations.
// Vector elementization happens at the machine instruction level, so the
// following instruction
// never appears in the DAG.
multiclass LD_VEC<NVPTXRegClass regclass> {
  def _v2_avar : NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
    (ins LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
      i32imm:$fromWidth, imem:$addr),
    !strconcat("ld${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t{{$dst1, $dst2}}, [$addr];"), []>;
  def _v2_areg : NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
    (ins LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
      i32imm:$fromWidth, Int32Regs:$addr),
    !strconcat("ld${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t{{$dst1, $dst2}}, [$addr];"), []>;
  def _v2_areg_64 : NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
    (ins LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
     i32imm:$fromWidth, Int64Regs:$addr),
    !strconcat("ld${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t{{$dst1, $dst2}}, [$addr];"), []>;
  def _v2_ari : NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
    (ins LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
      i32imm:$fromWidth, Int32Regs:$addr, i32imm:$offset),
    !strconcat("ld${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t{{$dst1, $dst2}}, [$addr+$offset];"), []>;
  def _v2_ari_64 : NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
    (ins LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
     i32imm:$fromWidth, Int64Regs:$addr, i32imm:$offset),
    !strconcat("ld${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t{{$dst1, $dst2}}, [$addr+$offset];"), []>;
  def _v2_asi : NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
    (ins LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
      i32imm:$fromWidth, imem:$addr, i32imm:$offset),
    !strconcat("ld${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t{{$dst1, $dst2}}, [$addr+$offset];"), []>;
  def _v4_avar : NVPTXInst<(outs regclass:$dst1, regclass:$dst2,
      regclass:$dst3, regclass:$dst4),
    (ins LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
      i32imm:$fromWidth, imem:$addr),
    !strconcat("ld${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t{{$dst1, $dst2, $dst3, $dst4}}, [$addr];"), []>;
  def _v4_areg : NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
      regclass:$dst4),
    (ins LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
      i32imm:$fromWidth, Int32Regs:$addr),
    !strconcat("ld${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t{{$dst1, $dst2, $dst3, $dst4}}, [$addr];"), []>;
  def _v4_areg_64 : NVPTXInst<(outs regclass:$dst1, regclass:$dst2,
                               regclass:$dst3, regclass:$dst4),
    (ins LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
     i32imm:$fromWidth, Int64Regs:$addr),
    !strconcat("ld${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t{{$dst1, $dst2, $dst3, $dst4}}, [$addr];"), []>;
  def _v4_ari : NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
      regclass:$dst4),
    (ins LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
      i32imm:$fromWidth, Int32Regs:$addr, i32imm:$offset),
    !strconcat("ld${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t{{$dst1, $dst2, $dst3, $dst4}}, [$addr+$offset];"),
                []>;
  def _v4_ari_64 : NVPTXInst<(outs regclass:$dst1, regclass:$dst2,
                              regclass:$dst3, regclass:$dst4),
    (ins LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
     i32imm:$fromWidth, Int64Regs:$addr, i32imm:$offset),
    !strconcat("ld${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t{{$dst1, $dst2, $dst3, $dst4}}, [$addr+$offset];"),
    []>;
  def _v4_asi : NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
      regclass:$dst4),
    (ins LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
      i32imm:$fromWidth, imem:$addr, i32imm:$offset),
    !strconcat("ld${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t{{$dst1, $dst2, $dst3, $dst4}}, [$addr+$offset];"),
                []>;
}
let mayLoad=1, neverHasSideEffects=1 in {
defm LDV_i8  : LD_VEC<Int8Regs>;
defm LDV_i16 : LD_VEC<Int16Regs>;
defm LDV_i32 : LD_VEC<Int32Regs>;
defm LDV_i64 : LD_VEC<Int64Regs>;
defm LDV_f32 : LD_VEC<Float32Regs>;
defm LDV_f64 : LD_VEC<Float64Regs>;
}

multiclass ST_VEC<NVPTXRegClass regclass> {
  def _v2_avar : NVPTXInst<(outs),
    (ins regclass:$src1, regclass:$src2, LdStCode:$isVol, LdStCode:$addsp,
      LdStCode:$Vec, LdStCode:$Sign, i32imm:$fromWidth, imem:$addr),
    !strconcat("st${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t[$addr], {{$src1, $src2}};"), []>;
  def _v2_areg : NVPTXInst<(outs),
    (ins regclass:$src1, regclass:$src2, LdStCode:$isVol, LdStCode:$addsp,
      LdStCode:$Vec, LdStCode:$Sign, i32imm:$fromWidth, Int32Regs:$addr),
    !strconcat("st${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t[$addr], {{$src1, $src2}};"), []>;
  def _v2_areg_64 : NVPTXInst<(outs),
    (ins regclass:$src1, regclass:$src2, LdStCode:$isVol, LdStCode:$addsp,
     LdStCode:$Vec, LdStCode:$Sign, i32imm:$fromWidth, Int64Regs:$addr),
    !strconcat("st${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t[$addr], {{$src1, $src2}};"), []>;
  def _v2_ari : NVPTXInst<(outs),
    (ins regclass:$src1, regclass:$src2, LdStCode:$isVol, LdStCode:$addsp,
      LdStCode:$Vec, LdStCode:$Sign, i32imm:$fromWidth, Int32Regs:$addr,
      i32imm:$offset),
    !strconcat("st${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t[$addr+$offset], {{$src1, $src2}};"), []>;
  def _v2_ari_64 : NVPTXInst<(outs),
    (ins regclass:$src1, regclass:$src2, LdStCode:$isVol, LdStCode:$addsp,
     LdStCode:$Vec, LdStCode:$Sign, i32imm:$fromWidth, Int64Regs:$addr,
     i32imm:$offset),
    !strconcat("st${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t[$addr+$offset], {{$src1, $src2}};"), []>;
  def _v2_asi : NVPTXInst<(outs),
    (ins regclass:$src1, regclass:$src2, LdStCode:$isVol, LdStCode:$addsp,
      LdStCode:$Vec, LdStCode:$Sign, i32imm:$fromWidth, imem:$addr,
      i32imm:$offset),
    !strconcat("st${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t[$addr+$offset], {{$src1, $src2}};"), []>;
  def _v4_avar : NVPTXInst<(outs),
    (ins regclass:$src1, regclass:$src2, regclass:$src3, regclass:$src4,
      LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
      i32imm:$fromWidth, imem:$addr),
    !strconcat("st${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t[$addr], {{$src1, $src2, $src3, $src4}};"), []>;
  def _v4_areg : NVPTXInst<(outs),
    (ins regclass:$src1, regclass:$src2, regclass:$src3, regclass:$src4,
      LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
      i32imm:$fromWidth, Int32Regs:$addr),
    !strconcat("st${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t[$addr], {{$src1, $src2, $src3, $src4}};"), []>;
  def _v4_areg_64 : NVPTXInst<(outs),
    (ins regclass:$src1, regclass:$src2, regclass:$src3, regclass:$src4,
     LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
     i32imm:$fromWidth, Int64Regs:$addr),
    !strconcat("st${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t[$addr], {{$src1, $src2, $src3, $src4}};"), []>;
  def _v4_ari : NVPTXInst<(outs),
    (ins regclass:$src1, regclass:$src2, regclass:$src3, regclass:$src4,
      LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
      i32imm:$fromWidth, Int32Regs:$addr, i32imm:$offset),
    !strconcat("st${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t[$addr+$offset], {{$src1, $src2, $src3, $src4}};"),
    []>;
  def _v4_ari_64 : NVPTXInst<(outs),
    (ins regclass:$src1, regclass:$src2, regclass:$src3, regclass:$src4,
     LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
     i32imm:$fromWidth, Int64Regs:$addr, i32imm:$offset),
    !strconcat("st${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t[$addr+$offset], {{$src1, $src2, $src3, $src4}};"),
     []>;
  def _v4_asi : NVPTXInst<(outs),
    (ins regclass:$src1, regclass:$src2, regclass:$src3, regclass:$src4,
      LdStCode:$isVol, LdStCode:$addsp, LdStCode:$Vec, LdStCode:$Sign,
      i32imm:$fromWidth, imem:$addr, i32imm:$offset),
    !strconcat("st${isVol:volatile}${addsp:addsp}${Vec:vec}.${Sign:sign}",
               "$fromWidth \t[$addr+$offset], {{$src1, $src2, $src3, $src4}};"),
    []>;
}
let mayStore=1, neverHasSideEffects=1 in {
defm STV_i8  : ST_VEC<Int8Regs>;
defm STV_i16 : ST_VEC<Int16Regs>;
defm STV_i32 : ST_VEC<Int32Regs>;
defm STV_i64 : ST_VEC<Int64Regs>;
defm STV_f32 : ST_VEC<Float32Regs>;
defm STV_f64 : ST_VEC<Float64Regs>;
}


//---- Conversion ----

multiclass CVT_INT_TO_FP <string OpStr, SDNode OpNode> {
// FIXME: need to add f16 support
//  def CVTf16i8 :
//    NVPTXInst<(outs Float16Regs:$d), (ins Int8Regs:$a),
//              !strconcat(!strconcat("cvt.rn.f16.", OpStr), "8 \t$d, $a;"),
//        [(set Float16Regs:$d, (OpNode Int8Regs:$a))]>;
//  def CVTf16i16 :
//    NVPTXInst<(outs Float16Regs:$d), (ins Int16Regs:$a),
//              !strconcat(!strconcat("cvt.rn.f16.", OpStr), "16 \t$d, $a;"),
//        [(set Float16Regs:$d, (OpNode Int16Regs:$a))]>;
//  def CVTf16i32 :
//    NVPTXInst<(outs Float16Regs:$d), (ins Int32Regs:$a),
//              !strconcat(!strconcat("cvt.rn.f16.", OpStr), "32 \t$d, $a;"),
//        [(set Float16Regs:$d, (OpNode Int32Regs:$a))]>;
//  def CVTf16i64:
//    NVPTXInst<(outs Float16Regs:$d), (ins Int64Regs:$a),
//          !strconcat(!strconcat("cvt.rn.f32.", OpStr), "64 \t$d, $a;"),
//            [(set Float32Regs:$d, (OpNode Int64Regs:$a))]>;

  def CVTf32i1 :
    NVPTXInst<(outs Float32Regs:$d), (ins Int1Regs:$a),
              "selp.f32 \t$d, 1.0, 0.0, $a;",
        [(set Float32Regs:$d, (OpNode Int1Regs:$a))]>;
  def CVTf32i8 :
    NVPTXInst<(outs Float32Regs:$d), (ins Int8Regs:$a),
              !strconcat(!strconcat("cvt.rn.f32.", OpStr), "8 \t$d, $a;"),
        [(set Float32Regs:$d, (OpNode Int8Regs:$a))]>;
  def CVTf32i16 :
    NVPTXInst<(outs Float32Regs:$d), (ins Int16Regs:$a),
              !strconcat(!strconcat("cvt.rn.f32.", OpStr), "16 \t$d, $a;"),
        [(set Float32Regs:$d, (OpNode Int16Regs:$a))]>;
  def CVTf32i32 :
    NVPTXInst<(outs Float32Regs:$d), (ins Int32Regs:$a),
              !strconcat(!strconcat("cvt.rn.f32.", OpStr), "32 \t$d, $a;"),
        [(set Float32Regs:$d, (OpNode Int32Regs:$a))]>;
  def CVTf32i64:
    NVPTXInst<(outs Float32Regs:$d), (ins Int64Regs:$a),
          !strconcat(!strconcat("cvt.rn.f32.", OpStr), "64 \t$d, $a;"),
            [(set Float32Regs:$d, (OpNode Int64Regs:$a))]>;

  def CVTf64i1 :
    NVPTXInst<(outs Float64Regs:$d), (ins Int1Regs:$a),
              "selp.f64 \t$d, 1.0, 0.0, $a;",
        [(set Float64Regs:$d, (OpNode Int1Regs:$a))]>;
  def CVTf64i8 :
    NVPTXInst<(outs Float64Regs:$d), (ins Int8Regs:$a),
              !strconcat(!strconcat("cvt.rn.f64.", OpStr), "8 \t$d, $a;"),
        [(set Float64Regs:$d, (OpNode Int8Regs:$a))]>;
  def CVTf64i16 :
    NVPTXInst<(outs Float64Regs:$d), (ins Int16Regs:$a),
              !strconcat(!strconcat("cvt.rn.f64.", OpStr), "16 \t$d, $a;"),
        [(set Float64Regs:$d, (OpNode Int16Regs:$a))]>;
  def CVTf64i32 :
    NVPTXInst<(outs Float64Regs:$d), (ins Int32Regs:$a),
              !strconcat(!strconcat("cvt.rn.f64.", OpStr), "32 \t$d, $a;"),
        [(set Float64Regs:$d, (OpNode Int32Regs:$a))]>;
  def CVTf64i64:
    NVPTXInst<(outs Float64Regs:$d), (ins Int64Regs:$a),
          !strconcat(!strconcat("cvt.rn.f64.", OpStr), "64 \t$d, $a;"),
            [(set Float64Regs:$d, (OpNode Int64Regs:$a))]>;
}

defm Sint_to_fp : CVT_INT_TO_FP <"s", sint_to_fp>;
defm Uint_to_fp : CVT_INT_TO_FP <"u", uint_to_fp>;

multiclass CVT_FP_TO_INT <string OpStr, SDNode OpNode> {
// FIXME: need to add f16 support
//  def CVTi8f16:
//    NVPTXInst<(outs Int8Regs:$d), (ins Float16Regs:$a),
//              !strconcat(!strconcat("cvt.rzi.", OpStr), "8.f16 $d, $a;"),
//        [(set Int8Regs:$d, (OpNode Float16Regs:$a))]>;
  def CVTi8f32_ftz:
    NVPTXInst<(outs Int8Regs:$d), (ins Float32Regs:$a),
              !strconcat(!strconcat("cvt.rzi.ftz.", OpStr), "16.f32 \t$d, $a;"),
        [(set Int8Regs:$d, (OpNode Float32Regs:$a))]>, Requires<[doF32FTZ]>;
  def CVTi8f32:
    NVPTXInst<(outs Int8Regs:$d), (ins Float32Regs:$a),
              !strconcat(!strconcat("cvt.rzi.", OpStr), "16.f32 \t$d, $a;"),
        [(set Int8Regs:$d, (OpNode Float32Regs:$a))]>;
  def CVTi8f64:
    NVPTXInst<(outs Int8Regs:$d), (ins Float64Regs:$a),
              !strconcat(!strconcat("cvt.rzi.", OpStr), "16.f64 \t$d, $a;"),
        [(set Int8Regs:$d, (OpNode Float64Regs:$a))]>;

// FIXME: need to add f16 support
//  def CVTi16f16:
//    NVPTXInst<(outs Int16Regs:$d), (ins Float16Regs:$a),
//              !strconcat(!strconcat("cvt.rzi.", OpStr), "16.f16 \t$d, $a;"),
//        [(set Int16Regs:$d, (OpNode Float16Regs:$a))]>;
  def CVTi16f32_ftz:
    NVPTXInst<(outs Int16Regs:$d), (ins Float32Regs:$a),
              !strconcat(!strconcat("cvt.rzi.ftz.", OpStr), "16.f32 \t$d, $a;"),
        [(set Int16Regs:$d, (OpNode Float32Regs:$a))]>, Requires<[doF32FTZ]>;
  def CVTi16f32:
    NVPTXInst<(outs Int16Regs:$d), (ins Float32Regs:$a),
              !strconcat(!strconcat("cvt.rzi.", OpStr), "16.f32 \t$d, $a;"),
        [(set Int16Regs:$d, (OpNode Float32Regs:$a))]>;
  def CVTi16f64:
    NVPTXInst<(outs Int16Regs:$d), (ins Float64Regs:$a),
              !strconcat(!strconcat("cvt.rzi.", OpStr), "16.f64 \t$d, $a;"),
        [(set Int16Regs:$d, (OpNode Float64Regs:$a))]>;

// FIXME: need to add f16 support
//  def CVTi32f16:  def CVTi32f16:
//    NVPTXInst<(outs Int32Regs:$d), (ins Float16Regs:$a),
//              !strconcat(!strconcat("cvt.rzi.", OpStr), "32.f16 \t$d, $a;"),
//        [(set Int32Regs:$d, (OpNode Float16Regs:$a))]>;
  def CVTi32f32_ftz:
    NVPTXInst<(outs Int32Regs:$d), (ins Float32Regs:$a),
              !strconcat(!strconcat("cvt.rzi.ftz.", OpStr), "32.f32 \t$d, $a;"),
        [(set Int32Regs:$d, (OpNode Float32Regs:$a))]>, Requires<[doF32FTZ]>;
  def CVTi32f32:
    NVPTXInst<(outs Int32Regs:$d), (ins Float32Regs:$a),
              !strconcat(!strconcat("cvt.rzi.", OpStr), "32.f32 \t$d, $a;"),
        [(set Int32Regs:$d, (OpNode Float32Regs:$a))]>;
  def CVTi32f64:
    NVPTXInst<(outs Int32Regs:$d), (ins Float64Regs:$a),
              !strconcat(!strconcat("cvt.rzi.", OpStr), "32.f64 \t$d, $a;"),
        [(set Int32Regs:$d, (OpNode Float64Regs:$a))]>;

// FIXME: need to add f16 support
//  def CVTi64f16:
//    NVPTXInst<(outs Int64Regs:$d), (ins Float16Regs:$a),
//              !strconcat(!strconcat("cvt.rzi.", OpStr), "64.f16 \t$d, $a;"),
//        [(set Int64Regs:$d, (OpNode Float16Regs:$a))]>;
  def CVTi64f32_ftz:
    NVPTXInst<(outs Int64Regs:$d), (ins Float32Regs:$a),
              !strconcat(!strconcat("cvt.rzi.ftz.", OpStr), "64.f32 \t$d, $a;"),
        [(set Int64Regs:$d, (OpNode Float32Regs:$a))]>, Requires<[doF32FTZ]>;
  def CVTi64f32:
    NVPTXInst<(outs Int64Regs:$d), (ins Float32Regs:$a),
              !strconcat(!strconcat("cvt.rzi.", OpStr), "64.f32 \t$d, $a;"),
        [(set Int64Regs:$d, (OpNode Float32Regs:$a))]>;
  def CVTi64f64:
    NVPTXInst<(outs Int64Regs:$d), (ins Float64Regs:$a),
              !strconcat(!strconcat("cvt.rzi.", OpStr), "64.f64 \t$d, $a;"),
        [(set Int64Regs:$d, (OpNode Float64Regs:$a))]>;
}

defm Fp_to_sint : CVT_FP_TO_INT <"s", fp_to_sint>;
defm Fp_to_uint : CVT_FP_TO_INT <"u", fp_to_uint>;

multiclass INT_EXTEND_UNSIGNED_1 <SDNode OpNode> {
  def ext1to8:
       NVPTXInst<(outs Int8Regs:$d), (ins Int1Regs:$a),
           "selp.u16 \t$d, 1, 0, $a;",
     [(set Int8Regs:$d, (OpNode Int1Regs:$a))]>;
  def ext1to16:
       NVPTXInst<(outs Int16Regs:$d), (ins Int1Regs:$a),
           "selp.u16 \t$d, 1, 0, $a;",
     [(set Int16Regs:$d, (OpNode Int1Regs:$a))]>;
  def ext1to32:
       NVPTXInst<(outs Int32Regs:$d), (ins Int1Regs:$a),
           "selp.u32 \t$d, 1, 0, $a;",
     [(set Int32Regs:$d, (OpNode Int1Regs:$a))]>;
  def ext1to64:
       NVPTXInst<(outs Int64Regs:$d), (ins Int1Regs:$a),
           "selp.u64 \t$d, 1, 0, $a;",
     [(set Int64Regs:$d, (OpNode Int1Regs:$a))]>;
}

multiclass INT_EXTEND_SIGNED_1 <SDNode OpNode> {
  def ext1to8:
       NVPTXInst<(outs Int8Regs:$d), (ins Int1Regs:$a),
           "selp.s16 \t$d, -1, 0, $a;",
     [(set Int8Regs:$d, (OpNode Int1Regs:$a))]>;
  def ext1to16:
       NVPTXInst<(outs Int16Regs:$d), (ins Int1Regs:$a),
           "selp.s16 \t$d, -1, 0, $a;",
     [(set Int16Regs:$d, (OpNode Int1Regs:$a))]>;
  def ext1to32:
       NVPTXInst<(outs Int32Regs:$d), (ins Int1Regs:$a),
           "selp.s32 \t$d, -1, 0, $a;",
     [(set Int32Regs:$d, (OpNode Int1Regs:$a))]>;
  def ext1to64:
       NVPTXInst<(outs Int64Regs:$d), (ins Int1Regs:$a),
           "selp.s64 \t$d, -1, 0, $a;",
     [(set Int64Regs:$d, (OpNode Int1Regs:$a))]>;
}

multiclass INT_EXTEND <string OpStr, SDNode OpNode> {
  // All Int8Regs are emiited as 16bit registers in ptx.
  // And there is no selp.u8 in ptx.
  def ext8to16:
       NVPTXInst<(outs Int16Regs:$d), (ins Int8Regs:$a),
           !strconcat("cvt.", !strconcat(OpStr, !strconcat("16.",
             !strconcat(OpStr, "8 \t$d, $a;")))),
     [(set Int16Regs:$d, (OpNode Int8Regs:$a))]>;
  def ext8to32:
       NVPTXInst<(outs Int32Regs:$d), (ins Int8Regs:$a),
           !strconcat("cvt.", !strconcat(OpStr, !strconcat("32.",
             !strconcat(OpStr, "8 \t$d, $a;")))),
     [(set Int32Regs:$d, (OpNode Int8Regs:$a))]>;
  def ext8to64:
       NVPTXInst<(outs Int64Regs:$d), (ins Int8Regs:$a),
           !strconcat("cvt.", !strconcat(OpStr, !strconcat("64.",
             !strconcat(OpStr, "8 \t$d, $a;")))),
     [(set Int64Regs:$d, (OpNode Int8Regs:$a))]>;
  def ext16to32:
       NVPTXInst<(outs Int32Regs:$d), (ins Int16Regs:$a),
           !strconcat("cvt.", !strconcat(OpStr, !strconcat("32.",
             !strconcat(OpStr, "16 \t$d, $a;")))),
     [(set Int32Regs:$d, (OpNode Int16Regs:$a))]>;
  def ext16to64:
       NVPTXInst<(outs Int64Regs:$d), (ins Int16Regs:$a),
           !strconcat("cvt.", !strconcat(OpStr, !strconcat("64.",
             !strconcat(OpStr, "16 \t$d, $a;")))),
     [(set Int64Regs:$d, (OpNode Int16Regs:$a))]>;
  def ext32to64:
       NVPTXInst<(outs Int64Regs:$d), (ins Int32Regs:$a),
           !strconcat("cvt.", !strconcat(OpStr, !strconcat("64.",
             !strconcat(OpStr, "32 \t$d, $a;")))),
     [(set Int64Regs:$d, (OpNode Int32Regs:$a))]>;
}

defm Sint_extend_1 : INT_EXTEND_SIGNED_1<sext>;
defm Zint_extend_1 : INT_EXTEND_UNSIGNED_1<zext>;
defm Aint_extend_1 : INT_EXTEND_UNSIGNED_1<anyext>;

defm Sint_extend : INT_EXTEND <"s", sext>;
defm Zint_extend : INT_EXTEND <"u", zext>;
defm Aint_extend : INT_EXTEND <"u", anyext>;

class TRUNC_to1_asm<string sz> {
  string s = !strconcat("{{\n\t",
             !strconcat(".reg ",
             !strconcat(sz,
             !strconcat(" temp;\n\t",
             !strconcat("and",
             !strconcat(sz,
             !strconcat("\t temp, $a, 1;\n\t",
             !strconcat("setp",
             !strconcat(sz, ".eq \t $d, temp, 1;\n\t}}")))))))));
}

def TRUNC_64to32 : NVPTXInst<(outs Int32Regs:$d), (ins Int64Regs:$a),
             "cvt.u32.u64 \t$d, $a;",
       [(set Int32Regs:$d, (trunc Int64Regs:$a))]>;
def TRUNC_64to16 : NVPTXInst<(outs Int16Regs:$d), (ins Int64Regs:$a),
             "cvt.u16.u64 \t$d, $a;",
       [(set Int16Regs:$d, (trunc Int64Regs:$a))]>;
def TRUNC_64to8 : NVPTXInst<(outs Int8Regs:$d), (ins Int64Regs:$a),
             "cvt.u8.u64 \t$d, $a;",
       [(set Int8Regs:$d, (trunc Int64Regs:$a))]>;
def TRUNC_32to16 : NVPTXInst<(outs Int16Regs:$d), (ins Int32Regs:$a),
             "cvt.u16.u32 \t$d, $a;",
       [(set Int16Regs:$d, (trunc Int32Regs:$a))]>;
def TRUNC_32to8 : NVPTXInst<(outs Int8Regs:$d), (ins Int32Regs:$a),
             "cvt.u8.u32 \t$d, $a;",
       [(set Int8Regs:$d, (trunc Int32Regs:$a))]>;
def TRUNC_16to8 : NVPTXInst<(outs Int8Regs:$d), (ins Int16Regs:$a),
             "cvt.u8.u16 \t$d, $a;",
       [(set Int8Regs:$d, (trunc Int16Regs:$a))]>;
def TRUNC_64to1 : NVPTXInst<(outs Int1Regs:$d), (ins Int64Regs:$a),
             TRUNC_to1_asm<".b64">.s,
             [(set Int1Regs:$d, (trunc Int64Regs:$a))]>;
def TRUNC_32to1 : NVPTXInst<(outs Int1Regs:$d), (ins Int32Regs:$a),
             TRUNC_to1_asm<".b32">.s,
             [(set Int1Regs:$d, (trunc Int32Regs:$a))]>;
def TRUNC_16to1 : NVPTXInst<(outs Int1Regs:$d), (ins Int16Regs:$a),
             TRUNC_to1_asm<".b16">.s,
             [(set Int1Regs:$d, (trunc Int16Regs:$a))]>;
def TRUNC_8to1 : NVPTXInst<(outs Int1Regs:$d), (ins Int8Regs:$a),
             TRUNC_to1_asm<".b16">.s,
             [(set Int1Regs:$d, (trunc Int8Regs:$a))]>;

// Select instructions
def : Pat<(select Int32Regs:$pred, Int8Regs:$a, Int8Regs:$b),
          (SELECTi8rr Int8Regs:$a, Int8Regs:$b, (TRUNC_32to1 Int32Regs:$pred))>;
def : Pat<(select Int32Regs:$pred, Int16Regs:$a, Int16Regs:$b),
          (SELECTi16rr Int16Regs:$a, Int16Regs:$b,
            (TRUNC_32to1 Int32Regs:$pred))>;
def : Pat<(select Int32Regs:$pred, Int32Regs:$a, Int32Regs:$b),
          (SELECTi32rr Int32Regs:$a, Int32Regs:$b,
            (TRUNC_32to1 Int32Regs:$pred))>;
def : Pat<(select Int32Regs:$pred, Int64Regs:$a, Int64Regs:$b),
          (SELECTi64rr Int64Regs:$a, Int64Regs:$b,
            (TRUNC_32to1 Int32Regs:$pred))>;
def : Pat<(select Int32Regs:$pred, Float32Regs:$a, Float32Regs:$b),
          (SELECTf32rr Float32Regs:$a, Float32Regs:$b,
            (TRUNC_32to1 Int32Regs:$pred))>;
def : Pat<(select Int32Regs:$pred, Float64Regs:$a, Float64Regs:$b),
          (SELECTf64rr Float64Regs:$a, Float64Regs:$b,
            (TRUNC_32to1 Int32Regs:$pred))>;

class F_BITCONVERT<string SzStr, NVPTXRegClass regclassIn,
  NVPTXRegClass regclassOut> :
           NVPTXInst<(outs regclassOut:$d), (ins regclassIn:$a),
           !strconcat("mov.b", !strconcat(SzStr, " \t $d, $a;")),
     [(set regclassOut:$d, (bitconvert regclassIn:$a))]>;

def BITCONVERT_32_I2F : F_BITCONVERT<"32", Int32Regs, Float32Regs>;
def BITCONVERT_32_F2I : F_BITCONVERT<"32", Float32Regs, Int32Regs>;
def BITCONVERT_64_I2F : F_BITCONVERT<"64", Int64Regs, Float64Regs>;
def BITCONVERT_64_F2I : F_BITCONVERT<"64", Float64Regs, Int64Regs>;

// pack a set of smaller int registers to a larger int register
def V4I8toI32 : NVPTXInst<(outs Int32Regs:$d),
                          (ins Int8Regs:$s1, Int8Regs:$s2,
                               Int8Regs:$s3, Int8Regs:$s4),
                          !strconcat("{{\n\t.reg .b8\t%t<4>;",
                          !strconcat("\n\tcvt.u8.u8\t%t0, $s1;",
                          !strconcat("\n\tcvt.u8.u8\t%t1, $s2;",
                          !strconcat("\n\tcvt.u8.u8\t%t2, $s3;",
                          !strconcat("\n\tcvt.u8.u8\t%t3, $s4;",
                           "\n\tmov.b32\t$d, {%t0, %t1, %t2, %t3};\n\t}}"))))),
                          []>;
def V4I16toI64 : NVPTXInst<(outs Int64Regs:$d),
                          (ins Int16Regs:$s1, Int16Regs:$s2,
                               Int16Regs:$s3, Int16Regs:$s4),
                          "mov.b64\t$d, {{$s1, $s2, $s3, $s4}};",
                          []>;
def V2I8toI16 : NVPTXInst<(outs Int16Regs:$d),
                          (ins Int8Regs:$s1, Int8Regs:$s2),
                          !strconcat("{{\n\t.reg .b8\t%t<2>;",
                          !strconcat("\n\tcvt.u8.u8\t%t0, $s1;",
                          !strconcat("\n\tcvt.u8.u8\t%t1, $s2;",
                                     "\n\tmov.b16\t$d, {%t0, %t1};\n\t}}"))),
                          []>;
def V2I16toI32 : NVPTXInst<(outs Int32Regs:$d),
                          (ins Int16Regs:$s1, Int16Regs:$s2),
                          "mov.b32\t$d, {{$s1, $s2}};",
                          []>;
def V2I32toI64 : NVPTXInst<(outs Int64Regs:$d),
                          (ins Int32Regs:$s1, Int32Regs:$s2),
                          "mov.b64\t$d, {{$s1, $s2}};",
                          []>;
def V2F32toF64 : NVPTXInst<(outs Float64Regs:$d),
                          (ins Float32Regs:$s1, Float32Regs:$s2),
                          "mov.b64\t$d, {{$s1, $s2}};",
                          []>;

// unpack a larger int register to a set of smaller int registers
def I32toV4I8 : NVPTXInst<(outs Int8Regs:$d1, Int8Regs:$d2,
                                Int8Regs:$d3, Int8Regs:$d4),
                          (ins Int32Regs:$s),
                          !strconcat("{{\n\t.reg .b8\t%t<4>;",
                          !strconcat("\n\tmov.b32\t{%t0, %t1, %t2, %t3}, $s;",
                          !strconcat("\n\tcvt.u8.u8\t$d1, %t0;",
                          !strconcat("\n\tcvt.u8.u8\t$d2, %t1;",
                          !strconcat("\n\tcvt.u8.u8\t$d3, %t2;",
                                     "\n\tcvt.u8.u8\t$d4, %t3;\n\t}}"))))),
                          []>;
def I64toV4I16 : NVPTXInst<(outs Int16Regs:$d1, Int16Regs:$d2,
                                 Int16Regs:$d3, Int16Regs:$d4),
                           (ins Int64Regs:$s),
                           "mov.b64\t{{$d1, $d2, $d3, $d4}}, $s;",
                          []>;
def I16toV2I8 : NVPTXInst<(outs Int8Regs:$d1, Int8Regs:$d2),
                          (ins Int16Regs:$s),
                          !strconcat("{{\n\t.reg .b8\t%t<2>;",
                          !strconcat("\n\tmov.b16\t{%t0, %t1}, $s;",
                          !strconcat("\n\tcvt.u8.u8\t$d1, %t0;",
                                     "\n\tcvt.u8.u8\t$d2, %t1;\n\t}}"))),
                          []>;
def I32toV2I16 : NVPTXInst<(outs Int16Regs:$d1, Int16Regs:$d2),
                           (ins Int32Regs:$s),
                           "mov.b32\t{{$d1, $d2}}, $s;",
                          []>;
def I64toV2I32 : NVPTXInst<(outs Int32Regs:$d1, Int32Regs:$d2),
                           (ins Int64Regs:$s),
                           "mov.b64\t{{$d1, $d2}}, $s;",
                          []>;
def F64toV2F32 : NVPTXInst<(outs Float32Regs:$d1, Float32Regs:$d2),
                           (ins Float64Regs:$s),
                           "mov.b64\t{{$d1, $d2}}, $s;",
                          []>;

def FPRound_ftz : NVPTXInst<(outs Float32Regs:$d), (ins Float64Regs:$a),
            "cvt.rn.ftz.f32.f64 \t$d, $a;",
      [(set Float32Regs:$d, (fround Float64Regs:$a))]>, Requires<[doF32FTZ]>;

def FPRound : NVPTXInst<(outs Float32Regs:$d), (ins Float64Regs:$a),
            "cvt.rn.f32.f64 \t$d, $a;",
      [(set Float32Regs:$d, (fround Float64Regs:$a))]>;

def FPExtend_ftz : NVPTXInst<(outs Float64Regs:$d), (ins Float32Regs:$a),
            "cvt.ftz.f64.f32 \t$d, $a;",
      [(set Float64Regs:$d, (fextend Float32Regs:$a))]>, Requires<[doF32FTZ]>;

def FPExtend : NVPTXInst<(outs Float64Regs:$d), (ins Float32Regs:$a),
            "cvt.f64.f32 \t$d, $a;",
      [(set Float64Regs:$d, (fextend Float32Regs:$a))]>;

def retflag       : SDNode<"NVPTXISD::RET_FLAG", SDTNone,
                           [SDNPHasChain, SDNPOptInGlue]>;

//-----------------------------------
// Control-flow
//-----------------------------------

let isTerminator=1 in {
   let isReturn=1, isBarrier=1 in
      def Return : NVPTXInst<(outs), (ins), "ret;", [(retflag)]>;

   let isBranch=1 in
      def CBranch : NVPTXInst<(outs), (ins Int1Regs:$a, brtarget:$target),
                          "@$a bra \t$target;",
                           [(brcond Int1Regs:$a, bb:$target)]>;
   let isBranch=1 in
      def CBranchOther : NVPTXInst<(outs), (ins Int1Regs:$a, brtarget:$target),
                          "@!$a bra \t$target;",
                           []>;

   let isBranch=1, isBarrier=1 in
      def GOTO : NVPTXInst<(outs), (ins brtarget:$target),
                        "bra.uni \t$target;",
                  [(br bb:$target)]>;
}

def : Pat<(brcond Int32Regs:$a, bb:$target), (CBranch
    (ISetUNEi32ri_p Int32Regs:$a, 0), bb:$target)>;

// SelectionDAGBuilder::visitSWitchCase() will invert the condition of a
// conditional branch if
// the target block is the next block so that the code can fall through to the
// target block.
// The invertion is done by 'xor condition, 1', which will be translated to
// (setne condition, -1).
// Since ptx supports '@!pred bra target', we should use it.
def : Pat<(brcond (i1 (setne Int1Regs:$a, -1)), bb:$target),
  (CBranchOther Int1Regs:$a, bb:$target)>;

// Call
def SDT_NVPTXCallSeqStart : SDCallSeqStart<[ SDTCisVT<0, i32> ]>;
def SDT_NVPTXCallSeqEnd   : SDCallSeqEnd<[ SDTCisVT<0, i32>,
                                        SDTCisVT<1, i32> ]>;

def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_NVPTXCallSeqStart,
                           [SDNPHasChain, SDNPOutGlue, SDNPSideEffect]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END",   SDT_NVPTXCallSeqEnd,
                           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                           SDNPSideEffect]>;

def SDT_NVPTXCall : SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>;
def call          : SDNode<"NVPTXISD::CALL", SDT_NVPTXCall,
                           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;
def calltarget : Operand<i32>;
let isCall=1 in {
   def CALL : NVPTXInst<(outs), (ins calltarget:$dst),
                  "call \t$dst, (1);", []>;
}

def : Pat<(call tglobaladdr:$dst),
          (CALL tglobaladdr:$dst)>;
def : Pat<(call texternalsym:$dst),
          (CALL texternalsym:$dst)>;

// Pseudo instructions.
class Pseudo<dag outs, dag ins, string asmstr, list<dag> pattern>
   : NVPTXInst<outs, ins, asmstr, pattern>;

// @TODO: We use some tricks here to emit curly braces.  Can we clean this up
// a bit without TableGen modifications?
def Callseq_Start : NVPTXInst<(outs), (ins i32imm:$amt),
  "// Callseq Start $amt\n\t{{\n\t.reg .b32 temp_param_reg;\n\t// <end>}}",
                               [(callseq_start timm:$amt)]>;
def Callseq_End : NVPTXInst<(outs), (ins i32imm:$amt1, i32imm:$amt2),
  "\n\t//{{\n\t}}// Callseq End $amt1",
                            [(callseq_end timm:$amt1, timm:$amt2)]>;

// trap instruction

def trapinst : NVPTXInst<(outs), (ins),
                         "trap;",
                         [(trap)]>;

include "NVPTXIntrinsics.td"


//-----------------------------------
// Notes
//-----------------------------------
// BSWAP is currently expanded. The following is a more efficient
// - for < sm_20, use vector scalar mov, as tesla support native 16-bit register
// - for sm_20, use pmpt (use vector scalar mov to get the pack and
//   unpack). sm_20 supports native 32-bit register, but not native 16-bit
// register.
