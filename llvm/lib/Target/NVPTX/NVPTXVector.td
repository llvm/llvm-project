//===- NVPTXVector.td - NVPTX Vector Specific Instruction defs -*- tblgen-*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//-----------------------------------
// Vector Specific
//-----------------------------------

//
// All vector instructions derive from NVPTXVecInst
//

class NVPTXVecInst<dag outs, dag ins, string asmstr, list<dag> pattern,
  NVPTXInst sInst=NOP>
  : NVPTXInst<outs, ins, asmstr, pattern> {
  NVPTXInst scalarInst=sInst;
}

let isAsCheapAsAMove=1, VecInstType=isVecExtract.Value in {
// Extract v2i16
def V2i16Extract : NVPTXVecInst<(outs Int16Regs:$dst),
  (ins V2I16Regs:$src, i8imm:$c),
                         "mov.u16 \t$dst, $src${c:vecelem};",
                         [(set Int16Regs:$dst, (vector_extract
                           (v2i16 V2I16Regs:$src), imm:$c))],
                         IMOV16rr>;

// Extract v4i16
def V4i16Extract : NVPTXVecInst<(outs Int16Regs:$dst),
  (ins V4I16Regs:$src, i8imm:$c),
                         "mov.u16 \t$dst, $src${c:vecelem};",
                         [(set Int16Regs:$dst, (vector_extract
                           (v4i16 V4I16Regs:$src), imm:$c))],
                         IMOV16rr>;

// Extract v2i8
def V2i8Extract : NVPTXVecInst<(outs Int8Regs:$dst),
  (ins V2I8Regs:$src, i8imm:$c),
                         "mov.u16 \t$dst, $src${c:vecelem};",
                         [(set Int8Regs:$dst, (vector_extract
                           (v2i8 V2I8Regs:$src), imm:$c))],
                         IMOV8rr>;

// Extract v4i8
def V4i8Extract : NVPTXVecInst<(outs Int8Regs:$dst),
  (ins V4I8Regs:$src, i8imm:$c),
                         "mov.u16 \t$dst, $src${c:vecelem};",
                         [(set Int8Regs:$dst, (vector_extract
                           (v4i8 V4I8Regs:$src), imm:$c))],
                         IMOV8rr>;

// Extract v2i32
def V2i32Extract : NVPTXVecInst<(outs Int32Regs:$dst),
  (ins V2I32Regs:$src, i8imm:$c),
                         "mov.u32 \t$dst, $src${c:vecelem};",
                         [(set Int32Regs:$dst, (vector_extract
                           (v2i32 V2I32Regs:$src), imm:$c))],
                         IMOV32rr>;

// Extract v2f32
def V2f32Extract : NVPTXVecInst<(outs Float32Regs:$dst),
  (ins V2F32Regs:$src, i8imm:$c),
                         "mov.f32 \t$dst, $src${c:vecelem};",
                         [(set Float32Regs:$dst, (vector_extract
                           (v2f32 V2F32Regs:$src), imm:$c))],
                         FMOV32rr>;

// Extract v2i64
def V2i64Extract : NVPTXVecInst<(outs Int64Regs:$dst),
  (ins V2I64Regs:$src, i8imm:$c),
                         "mov.u64 \t$dst, $src${c:vecelem};",
                         [(set Int64Regs:$dst, (vector_extract
                           (v2i64 V2I64Regs:$src), imm:$c))],
                         IMOV64rr>;

// Extract v2f64
def V2f64Extract : NVPTXVecInst<(outs Float64Regs:$dst),
  (ins V2F64Regs:$src, i8imm:$c),
                         "mov.f64 \t$dst, $src${c:vecelem};",
                         [(set Float64Regs:$dst, (vector_extract
                           (v2f64 V2F64Regs:$src), imm:$c))],
                         FMOV64rr>;

// Extract v4i32
def V4i32Extract : NVPTXVecInst<(outs Int32Regs:$dst),
  (ins V4I32Regs:$src, i8imm:$c),
                         "mov.u32 \t$dst, $src${c:vecelem};",
                         [(set Int32Regs:$dst, (vector_extract
                           (v4i32 V4I32Regs:$src), imm:$c))],
                         IMOV32rr>;

// Extract v4f32
def V4f32Extract : NVPTXVecInst<(outs Float32Regs:$dst),
  (ins V4F32Regs:$src, i8imm:$c),
                         "mov.f32 \t$dst, $src${c:vecelem};",
                         [(set Float32Regs:$dst, (vector_extract
                           (v4f32 V4F32Regs:$src), imm:$c))],
                         FMOV32rr>;
}

let isAsCheapAsAMove=1, VecInstType=isVecInsert.Value in {
// Insert v2i8
def V2i8Insert : NVPTXVecInst<(outs V2I8Regs:$dst),
  (ins V2I8Regs:$src, Int8Regs:$val, i8imm:$c),
        "mov.v2.u16 \t${dst:vecfull}, ${src:vecfull};"
        "\n\tmov.u16 \t$dst${c:vecelem}, $val;",
       [(set V2I8Regs:$dst,
         (vector_insert V2I8Regs:$src, Int8Regs:$val, imm:$c))],
                         IMOV8rr>;

// Insert v4i8
def V4i8Insert : NVPTXVecInst<(outs V4I8Regs:$dst),
  (ins V4I8Regs:$src, Int8Regs:$val, i8imm:$c),
                       "mov.v4.u16 \t${dst:vecfull}, ${src:vecfull};"
                       "\n\tmov.u16 \t$dst${c:vecelem}, $val;",
       [(set V4I8Regs:$dst,
         (vector_insert V4I8Regs:$src, Int8Regs:$val, imm:$c))],
                         IMOV8rr>;

// Insert v2i16
def V2i16Insert : NVPTXVecInst<(outs V2I16Regs:$dst),
  (ins V2I16Regs:$src, Int16Regs:$val, i8imm:$c),
                       "mov.v2.u16 \t${dst:vecfull}, ${src:vecfull};"
                       "\n\tmov.u16 \t$dst${c:vecelem}, $val;",
       [(set V2I16Regs:$dst,
         (vector_insert V2I16Regs:$src, Int16Regs:$val, imm:$c))],
                         IMOV16rr>;

// Insert v4i16
def V4i16Insert : NVPTXVecInst<(outs V4I16Regs:$dst),
  (ins V4I16Regs:$src, Int16Regs:$val, i8imm:$c),
                       "mov.v4.u16 \t${dst:vecfull}, ${src:vecfull};"
                       "\n\tmov.u16 \t$dst${c:vecelem}, $val;",
       [(set V4I16Regs:$dst,
         (vector_insert V4I16Regs:$src, Int16Regs:$val, imm:$c))],
                         IMOV16rr>;

// Insert v2i32
def V2i32Insert : NVPTXVecInst<(outs V2I32Regs:$dst),
  (ins V2I32Regs:$src, Int32Regs:$val, i8imm:$c),
                       "mov.v2.u32 \t${dst:vecfull}, ${src:vecfull};"
                       "\n\tmov.u32 \t$dst${c:vecelem}, $val;",
       [(set V2I32Regs:$dst,
         (vector_insert V2I32Regs:$src, Int32Regs:$val, imm:$c))],
                         IMOV32rr>;

// Insert v2f32
def V2f32Insert : NVPTXVecInst<(outs V2F32Regs:$dst),
  (ins V2F32Regs:$src, Float32Regs:$val, i8imm:$c),
                       "mov.v2.f32 \t${dst:vecfull}, ${src:vecfull};"
                       "\n\tmov.f32 \t$dst${c:vecelem}, $val;",
       [(set V2F32Regs:$dst,
         (vector_insert V2F32Regs:$src, Float32Regs:$val, imm:$c))],
                         FMOV32rr>;

// Insert v2i64
def V2i64Insert : NVPTXVecInst<(outs V2I64Regs:$dst),
  (ins V2I64Regs:$src, Int64Regs:$val, i8imm:$c),
                       "mov.v2.u64 \t${dst:vecfull}, ${src:vecfull};"
                       "\n\tmov.u64 \t$dst${c:vecelem}, $val;",
       [(set V2I64Regs:$dst,
         (vector_insert V2I64Regs:$src, Int64Regs:$val, imm:$c))],
                         IMOV64rr>;

// Insert v2f64
def V2f64Insert : NVPTXVecInst<(outs V2F64Regs:$dst),
  (ins V2F64Regs:$src, Float64Regs:$val, i8imm:$c),
                       "mov.v2.f64 \t${dst:vecfull}, ${src:vecfull};"
                       "\n\tmov.f64 \t$dst${c:vecelem}, $val;",
       [(set V2F64Regs:$dst,
         (vector_insert V2F64Regs:$src, Float64Regs:$val, imm:$c))],
                         FMOV64rr>;

// Insert v4i32
def V4i32Insert : NVPTXVecInst<(outs V4I32Regs:$dst),
  (ins V4I32Regs:$src, Int32Regs:$val, i8imm:$c),
                       "mov.v4.u32 \t${dst:vecfull}, ${src:vecfull};"
                       "\n\tmov.u32 \t$dst${c:vecelem}, $val;",
       [(set V4I32Regs:$dst,
         (vector_insert V4I32Regs:$src, Int32Regs:$val, imm:$c))],
                         IMOV32rr>;

// Insert v4f32
def V4f32Insert : NVPTXVecInst<(outs V4F32Regs:$dst),
  (ins V4F32Regs:$src, Float32Regs:$val, i8imm:$c),
                       "mov.v4.f32 \t${dst:vecfull}, ${src:vecfull};"
                       "\n\tmov.f32 \t$dst${c:vecelem}, $val;",
       [(set V4F32Regs:$dst,
         (vector_insert V4F32Regs:$src, Float32Regs:$val, imm:$c))],
                         FMOV32rr>;
}

class BinOpAsmString<string c> {
  string s = c;
}

class V4AsmStr<string opcode> : BinOpAsmString<
                          !strconcat(!strconcat(!strconcat(!strconcat(
                            !strconcat(!strconcat(!strconcat(
                          opcode,  " \t${dst}_0, ${a}_0, ${b}_0;\n\t"),
                          opcode), " \t${dst}_1, ${a}_1, ${b}_1;\n\t"),
                          opcode), " \t${dst}_2, ${a}_2, ${b}_2;\n\t"),
                          opcode), " \t${dst}_3, ${a}_3, ${b}_3;")>;

class V2AsmStr<string opcode> : BinOpAsmString<
                           !strconcat(!strconcat(!strconcat(
                           opcode,  " \t${dst}_0, ${a}_0, ${b}_0;\n\t"),
                           opcode), " \t${dst}_1, ${a}_1, ${b}_1;")>;

class V4MADStr<string opcode> : BinOpAsmString<
                          !strconcat(!strconcat(!strconcat(!strconcat(
                            !strconcat(!strconcat(!strconcat(
                          opcode,  " \t${dst}_0, ${a}_0, ${b}_0, ${c}_0;\n\t"),
                          opcode), " \t${dst}_1, ${a}_1, ${b}_1, ${c}_1;\n\t"),
                          opcode), " \t${dst}_2, ${a}_2, ${b}_2, ${c}_2;\n\t"),
                          opcode), " \t${dst}_3, ${a}_3, ${b}_3, ${c}_3;")>;

class V2MADStr<string opcode> : BinOpAsmString<
                           !strconcat(!strconcat(!strconcat(
                           opcode,  " \t${dst}_0, ${a}_0, ${b}_0, ${c}_0;\n\t"),
                           opcode), " \t${dst}_1, ${a}_1, ${b}_1, ${c}_1;")>;

class V4UnaryStr<string opcode> : BinOpAsmString<
                          !strconcat(!strconcat(!strconcat(!strconcat(
                            !strconcat(!strconcat(!strconcat(
                          opcode,  " \t${dst}_0, ${a}_0;\n\t"),
                          opcode), " \t${dst}_1, ${a}_1;\n\t"),
                          opcode), " \t${dst}_2, ${a}_2;\n\t"),
                          opcode), " \t${dst}_3, ${a}_3;")>;

class V2UnaryStr<string opcode> : BinOpAsmString<
                           !strconcat(!strconcat(!strconcat(
                           opcode,  " \t${dst}_0, ${a}_0;\n\t"),
                           opcode), " \t${dst}_1, ${a}_1;")>;

class VecBinaryOp<BinOpAsmString asmstr, SDNode OpNode, NVPTXRegClass regclass,
  NVPTXInst sInst=NOP> :
      NVPTXVecInst<(outs regclass:$dst), (ins regclass:$a, regclass:$b),
                 asmstr.s,
                 [(set regclass:$dst, (OpNode regclass:$a, regclass:$b))],
                 sInst>;

class VecShiftOp<BinOpAsmString asmstr, SDNode OpNode, NVPTXRegClass regclass1,
                 NVPTXRegClass regclass2, NVPTXInst sInst=NOP> :
      NVPTXVecInst<(outs regclass1:$dst), (ins regclass1:$a, regclass2:$b),
                 asmstr.s,
                 [(set regclass1:$dst, (OpNode regclass1:$a, regclass2:$b))],
                 sInst>;

class VecUnaryOp<BinOpAsmString asmstr, PatFrag OpNode, NVPTXRegClass regclass,
  NVPTXInst sInst=NOP> :
      NVPTXVecInst<(outs regclass:$dst), (ins regclass:$a),
                 asmstr.s,
                 [(set regclass:$dst, (OpNode regclass:$a))], sInst>;

multiclass IntBinVOp<string asmstr, SDNode OpNode,
                     NVPTXInst i64op=NOP, NVPTXInst i32op=NOP, NVPTXInst
                     i16op=NOP, NVPTXInst i8op=NOP> {
  def V2I64 : VecBinaryOp<V2AsmStr<!strconcat(asmstr, "64")>, OpNode, V2I64Regs,
    i64op>;
  def V4I32 : VecBinaryOp<V4AsmStr<!strconcat(asmstr, "32")>, OpNode, V4I32Regs,
    i32op>;
  def V2I32 : VecBinaryOp<V2AsmStr<!strconcat(asmstr, "32")>, OpNode, V2I32Regs,
    i32op>;
  def V4I16 : VecBinaryOp<V4AsmStr<!strconcat(asmstr, "16")>, OpNode, V4I16Regs,
    i16op>;
  def V2I16 : VecBinaryOp<V2AsmStr<!strconcat(asmstr, "16")>, OpNode, V2I16Regs,
    i16op>;
  def V4I8 : VecBinaryOp<V4AsmStr<!strconcat(asmstr, "16")>, OpNode, V4I8Regs,
    i8op>;
  def V2I8 : VecBinaryOp<V2AsmStr<!strconcat(asmstr, "16")>, OpNode, V2I8Regs,
    i8op>;
}

multiclass FloatBinVOp<string asmstr, SDNode OpNode,
                       NVPTXInst f64=NOP, NVPTXInst f32=NOP,
                       NVPTXInst f32_ftz=NOP> {
  def V2F64 : VecBinaryOp<V2AsmStr<!strconcat(asmstr, "f64")>, OpNode,
    V2F64Regs, f64>;
  def V4F32_ftz : VecBinaryOp<V4AsmStr<!strconcat(asmstr, "ftz.f32")>, OpNode,
    V4F32Regs, f32_ftz>, Requires<[doF32FTZ]>;
  def V2F32_ftz : VecBinaryOp<V2AsmStr<!strconcat(asmstr, "ftz.f32")>, OpNode,
    V2F32Regs, f32_ftz>, Requires<[doF32FTZ]>;
  def V4F32 : VecBinaryOp<V4AsmStr<!strconcat(asmstr, "f32")>, OpNode,
    V4F32Regs, f32>;
  def V2F32 : VecBinaryOp<V2AsmStr<!strconcat(asmstr, "f32")>, OpNode,
    V2F32Regs, f32>;
}

multiclass IntUnaryVOp<string asmstr, PatFrag OpNode,
                       NVPTXInst i64op=NOP, NVPTXInst i32op=NOP,
                       NVPTXInst i16op=NOP, NVPTXInst i8op=NOP> {
  def V2I64 : VecUnaryOp<V2UnaryStr<!strconcat(asmstr, "64")>, OpNode,
    V2I64Regs, i64op>;
  def V4I32 : VecUnaryOp<V4UnaryStr<!strconcat(asmstr, "32")>, OpNode,
    V4I32Regs, i32op>;
  def V2I32 : VecUnaryOp<V2UnaryStr<!strconcat(asmstr, "32")>, OpNode,
    V2I32Regs, i32op>;
  def V4I16 : VecUnaryOp<V4UnaryStr<!strconcat(asmstr, "16")>, OpNode,
    V4I16Regs, i16op>;
  def V2I16 : VecUnaryOp<V2UnaryStr<!strconcat(asmstr, "16")>, OpNode,
    V2I16Regs, i16op>;
  def V4I8  : VecUnaryOp<V4UnaryStr<!strconcat(asmstr, "16")>, OpNode,
    V4I8Regs,   i8op>;
  def V2I8  : VecUnaryOp<V2UnaryStr<!strconcat(asmstr, "16")>, OpNode,
    V2I8Regs,   i8op>;
}


// Integer Arithmetic
let VecInstType=isVecOther.Value in {
defm VAdd : IntBinVOp<"add.s", add, ADDi64rr, ADDi32rr, ADDi16rr, ADDi8rr>;
defm VSub : IntBinVOp<"sub.s", sub, SUBi64rr, SUBi32rr, SUBi16rr, SUBi8rr>;

def AddCCV4I32 : VecBinaryOp<V4AsmStr<"add.cc.s32">, addc, V4I32Regs,
  ADDCCi32rr>;
def AddCCV2I32 : VecBinaryOp<V2AsmStr<"add.cc.s32">, addc, V2I32Regs,
  ADDCCi32rr>;
def SubCCV4I32 : VecBinaryOp<V4AsmStr<"sub.cc.s32">, subc, V4I32Regs,
  SUBCCi32rr>;
def SubCCV2I32 : VecBinaryOp<V2AsmStr<"sub.cc.s32">, subc, V2I32Regs,
  SUBCCi32rr>;
def AddCCCV4I32 : VecBinaryOp<V4AsmStr<"addc.cc.s32">, adde, V4I32Regs,
  ADDCCCi32rr>;
def AddCCCV2I32 : VecBinaryOp<V2AsmStr<"addc.cc.s32">, adde, V2I32Regs,
  ADDCCCi32rr>;
def SubCCCV4I32 : VecBinaryOp<V4AsmStr<"subc.cc.s32">, sube, V4I32Regs,
  SUBCCCi32rr>;
def SubCCCV2I32 : VecBinaryOp<V2AsmStr<"subc.cc.s32">, sube, V2I32Regs,
  SUBCCCi32rr>;

def ShiftLV2I64 : VecShiftOp<V2AsmStr<"shl.b64">, shl, V2I64Regs, V2I32Regs,
  SHLi64rr>;
def ShiftLV2I32 : VecShiftOp<V2AsmStr<"shl.b32">, shl, V2I32Regs, V2I32Regs,
  SHLi32rr>;
def ShiftLV4I32 : VecShiftOp<V4AsmStr<"shl.b32">, shl, V4I32Regs, V4I32Regs,
  SHLi32rr>;
def ShiftLV2I16 : VecShiftOp<V2AsmStr<"shl.b16">, shl, V2I16Regs, V2I32Regs,
  SHLi16rr>;
def ShiftLV4I16 : VecShiftOp<V4AsmStr<"shl.b16">, shl, V4I16Regs, V4I32Regs,
  SHLi16rr>;
def ShiftLV2I8  : VecShiftOp<V2AsmStr<"shl.b16">, shl, V2I8Regs,  V2I32Regs,
  SHLi8rr>;
def ShiftLV4I8  : VecShiftOp<V4AsmStr<"shl.b16">, shl, V4I8Regs,  V4I32Regs,
  SHLi8rr>;
}

// cvt to v*i32, helpers for shift
class CVTtoVeci32<NVPTXRegClass inclass, NVPTXRegClass outclass, string asmstr,
  NVPTXInst sInst=NOP> :
      NVPTXVecInst<(outs outclass:$d), (ins inclass:$s), asmstr, [], sInst>;

class VecCVTStrHelper<string op, string dest, string src> {
  string s=!strconcat(op, !strconcat("\t",
           !strconcat(dest, !strconcat(", ", !strconcat(src, ";")))));
}

class Vec2CVTStr<string op> {
  string s=!strconcat(VecCVTStrHelper<op, "${d}_0", "${s}_0">.s,
           !strconcat("\n\t", VecCVTStrHelper<op, "${d}_1", "${s}_1">.s));
}

class Vec4CVTStr<string op> {
  string s=!strconcat(VecCVTStrHelper<op, "${d}_0", "${s}_0">.s,
           !strconcat("\n\t",
           !strconcat(VecCVTStrHelper<op, "${d}_1", "${s}_1">.s,
           !strconcat("\n\t",
           !strconcat(VecCVTStrHelper<op, "${d}_2", "${s}_2">.s,
           !strconcat("\n\t", VecCVTStrHelper<op, "${d}_3", "${s}_3">.s))))));
}

let VecInstType=isVecOther.Value in {
def CVTv2i8tov2i32 : CVTtoVeci32<V2I8Regs, V2I32Regs,
  Vec2CVTStr<"cvt.u32.u16">.s, Zint_extendext8to32>;
def CVTv2i16tov2i32 : CVTtoVeci32<V2I16Regs, V2I32Regs,
  Vec2CVTStr<"cvt.u32.u16">.s, Zint_extendext16to32>;
def CVTv4i8tov4i32 : CVTtoVeci32<V4I8Regs, V4I32Regs,
  Vec4CVTStr<"cvt.u32.u16">.s, Zint_extendext8to32>;
def CVTv4i16tov4i32 : CVTtoVeci32<V4I16Regs, V4I32Regs,
  Vec4CVTStr<"cvt.u32.u16">.s, Zint_extendext16to32>;
def CVTv2i64tov2i32 : CVTtoVeci32<V2I64Regs, V2I32Regs,
  Vec2CVTStr<"cvt.u32.u64">.s, TRUNC_64to32>;
}

def : Pat<(shl V2I16Regs:$src1, V2I16Regs:$src2),
          (ShiftLV2I16 V2I16Regs:$src1, (CVTv2i16tov2i32 V2I16Regs:$src2))>;
def : Pat<(shl V2I8Regs:$src1, V2I8Regs:$src2),
          (ShiftLV2I8 V2I8Regs:$src1, (CVTv2i8tov2i32 V2I8Regs:$src2))>;
def : Pat<(shl V2I64Regs:$src1, V2I64Regs:$src2),
          (ShiftLV2I64 V2I64Regs:$src1, (CVTv2i64tov2i32 V2I64Regs:$src2))>;

def : Pat<(shl V4I16Regs:$src1, V4I16Regs:$src2),
          (ShiftLV4I16 V4I16Regs:$src1, (CVTv4i16tov4i32 V4I16Regs:$src2))>;
def : Pat<(shl V4I8Regs:$src1, V4I8Regs:$src2),
          (ShiftLV4I8 V4I8Regs:$src1, (CVTv4i8tov4i32 V4I8Regs:$src2))>;

let VecInstType=isVecOther.Value in {
def ShiftRAV2I64 : VecShiftOp<V2AsmStr<"shr.s64">, sra, V2I64Regs, V2I32Regs,
  SRAi64rr>;
def ShiftRAV2I32 : VecShiftOp<V2AsmStr<"shr.s32">, sra, V2I32Regs, V2I32Regs,
  SRAi32rr>;
def ShiftRAV4I32 : VecShiftOp<V4AsmStr<"shr.s32">, sra, V4I32Regs, V4I32Regs,
  SRAi32rr>;
def ShiftRAV2I16 : VecShiftOp<V2AsmStr<"shr.s16">, sra, V2I16Regs, V2I32Regs,
  SRAi16rr>;
def ShiftRAV4I16 : VecShiftOp<V4AsmStr<"shr.s16">, sra, V4I16Regs, V4I32Regs,
  SRAi16rr>;
def ShiftRAV2I8  : VecShiftOp<V2AsmStr<"shr.s16">, sra, V2I8Regs,  V2I32Regs,
  SRAi8rr>;
def ShiftRAV4I8  : VecShiftOp<V4AsmStr<"shr.s16">, sra, V4I8Regs,  V4I32Regs,
  SRAi8rr>;

def ShiftRLV2I64 : VecShiftOp<V2AsmStr<"shr.u64">, srl, V2I64Regs, V2I32Regs,
  SRLi64rr>;
def ShiftRLV2I32 : VecShiftOp<V2AsmStr<"shr.u32">, srl, V2I32Regs, V2I32Regs,
  SRLi32rr>;
def ShiftRLV4I32 : VecShiftOp<V4AsmStr<"shr.u32">, srl, V4I32Regs, V4I32Regs,
  SRLi32rr>;
def ShiftRLV2I16 : VecShiftOp<V2AsmStr<"shr.u16">, srl, V2I16Regs, V2I32Regs,
  SRLi16rr>;
def ShiftRLV4I16 : VecShiftOp<V4AsmStr<"shr.u16">, srl, V4I16Regs, V4I32Regs,
  SRLi16rr>;
def ShiftRLV2I8  : VecShiftOp<V2AsmStr<"shr.u16">, srl, V2I8Regs,  V2I32Regs,
  SRLi8rr>;
def ShiftRLV4I8  : VecShiftOp<V4AsmStr<"shr.u16">, srl, V4I8Regs,  V4I32Regs,
  SRLi8rr>;

defm VMult   : IntBinVOp<"mul.lo.s", mul, MULTi64rr, MULTi32rr, MULTi16rr,
  MULTi8rr>;
defm VMultHS : IntBinVOp<"mul.hi.s", mulhs, MULTHSi64rr, MULTHSi32rr,
  MULTHSi16rr,
  MULTHSi8rr>;
defm VMultHU : IntBinVOp<"mul.hi.u", mulhu, MULTHUi64rr, MULTHUi32rr,
  MULTHUi16rr,
  MULTHUi8rr>;
defm VSDiv   : IntBinVOp<"div.s", sdiv, SDIVi64rr, SDIVi32rr, SDIVi16rr,
  SDIVi8rr>;
defm VUDiv   : IntBinVOp<"div.u", udiv, UDIVi64rr, UDIVi32rr, UDIVi16rr,
  UDIVi8rr>;
defm VSRem   : IntBinVOp<"rem.s", srem, SREMi64rr, SREMi32rr, SREMi16rr,
  SREMi8rr>;
defm VURem   : IntBinVOp<"rem.u", urem, UREMi64rr, UREMi32rr, UREMi16rr,
  UREMi8rr>;
}

def : Pat<(sra V2I16Regs:$src1, V2I16Regs:$src2),
          (ShiftRAV2I16 V2I16Regs:$src1, (CVTv2i16tov2i32 V2I16Regs:$src2))>;
def : Pat<(sra V2I8Regs:$src1, V2I8Regs:$src2),
          (ShiftRAV2I8 V2I8Regs:$src1, (CVTv2i8tov2i32 V2I8Regs:$src2))>;
def : Pat<(sra V2I64Regs:$src1, V2I64Regs:$src2),
          (ShiftRAV2I64 V2I64Regs:$src1, (CVTv2i64tov2i32 V2I64Regs:$src2))>;

def : Pat<(sra V4I16Regs:$src1, V4I16Regs:$src2),
          (ShiftRAV4I16 V4I16Regs:$src1, (CVTv4i16tov4i32 V4I16Regs:$src2))>;
def : Pat<(sra V4I8Regs:$src1, V4I8Regs:$src2),
          (ShiftRAV4I8 V4I8Regs:$src1, (CVTv4i8tov4i32 V4I8Regs:$src2))>;

def : Pat<(srl V2I16Regs:$src1, V2I16Regs:$src2),
          (ShiftRLV2I16 V2I16Regs:$src1, (CVTv2i16tov2i32 V2I16Regs:$src2))>;
def : Pat<(srl V2I8Regs:$src1, V2I8Regs:$src2),
          (ShiftRLV2I8 V2I8Regs:$src1, (CVTv2i8tov2i32 V2I8Regs:$src2))>;
def : Pat<(srl V2I64Regs:$src1, V2I64Regs:$src2),
          (ShiftRLV2I64 V2I64Regs:$src1, (CVTv2i64tov2i32 V2I64Regs:$src2))>;

def : Pat<(srl V4I16Regs:$src1, V4I16Regs:$src2),
          (ShiftRLV4I16 V4I16Regs:$src1, (CVTv4i16tov4i32 V4I16Regs:$src2))>;
def : Pat<(srl V4I8Regs:$src1, V4I8Regs:$src2),
          (ShiftRLV4I8 V4I8Regs:$src1, (CVTv4i8tov4i32 V4I8Regs:$src2))>;

multiclass VMAD<string asmstr, NVPTXRegClass regclassv4,
  NVPTXRegClass regclassv2,
                SDNode an=add, SDNode mn=mul, NVPTXInst sop=NOP,
                Predicate Pred> {
  def V4 : NVPTXVecInst<(outs regclassv4:$dst),
    (ins regclassv4:$a, regclassv4:$b, regclassv4:$c),
                      V4MADStr<asmstr>.s,
                      [(set regclassv4:$dst,
                        (an (mn regclassv4:$a, regclassv4:$b), regclassv4:$c))],
                      sop>,
           Requires<[Pred]>;
  def V2 : NVPTXVecInst<(outs regclassv2:$dst),
    (ins regclassv2:$a, regclassv2:$b, regclassv2:$c),
                      V2MADStr<asmstr>.s,
                      [(set regclassv2:$dst,
                        (an (mn regclassv2:$a, regclassv2:$b), regclassv2:$c))],
                      sop>,
           Requires<[Pred]>;
}

multiclass VMADV2Only<string asmstr, NVPTXRegClass regclass, NVPTXInst sop=NOP,
  Predicate Pred> {
  def V2 : NVPTXVecInst<(outs regclass:$dst),
    (ins regclass:$a, regclass:$b, regclass:$c),
                      V2MADStr<asmstr>.s,
                      [(set regclass:$dst, (add
                        (mul regclass:$a, regclass:$b), regclass:$c))], sop>,
           Requires<[Pred]>;
}
multiclass VFMADV2Only<string asmstr, NVPTXRegClass regclass, NVPTXInst sop=NOP,
  Predicate Pred> {
  def V2 : NVPTXVecInst<(outs regclass:$dst),
    (ins regclass:$a, regclass:$b, regclass:$c),
                      V2MADStr<asmstr>.s,
                      [(set regclass:$dst, (fadd
                        (fmul regclass:$a, regclass:$b), regclass:$c))], sop>,
           Requires<[Pred]>;
}

let VecInstType=isVecOther.Value in {
defm I8MAD  : VMAD<"mad.lo.s16", V4I8Regs, V2I8Regs, add, mul, MAD8rrr, true>;
defm I16MAD : VMAD<"mad.lo.s16", V4I16Regs, V2I16Regs, add, mul, MAD16rrr,
  true>;
defm I32MAD : VMAD<"mad.lo.s32", V4I32Regs, V2I32Regs, add, mul, MAD32rrr,
  true>;
defm I64MAD : VMADV2Only<"mad.lo.s64", V2I64Regs, MAD64rrr, true>;

defm VNeg : IntUnaryVOp<"neg.s", ineg, INEG64, INEG32, INEG16, INEG8>;

defm VAddf : FloatBinVOp<"add.", fadd, FADDf64rr, FADDf32rr, FADDf32rr_ftz>;
defm VSubf : FloatBinVOp<"sub.", fsub, FSUBf64rr, FSUBf32rr, FSUBf32rr_ftz>;
defm VMulf : FloatBinVOp<"mul.", fmul, FMULf64rr, FMULf32rr, FMULf32rr_ftz>;

defm F32MAD_ftz : VMAD<"mad.ftz.f32", V4F32Regs, V2F32Regs, fadd, fmul,
  FMAD32_ftzrrr, doFMADF32_ftz>;
defm F32FMA_ftz : VMAD<"fma.rn.ftz.f32", V4F32Regs, V2F32Regs, fadd, fmul,
  FMA32_ftzrrr, doFMAF32_ftz>;
defm F32MAD : VMAD<"mad.f32", V4F32Regs, V2F32Regs, fadd, fmul, FMAD32rrr,
  doFMADF32>;
defm F32FMA : VMAD<"fma.rn.f32", V4F32Regs, V2F32Regs, fadd, fmul, FMA32rrr,
  doFMAF32>;
defm F64FMA : VFMADV2Only<"fma.rn.f64", V2F64Regs, FMA64rrr, doFMAF64>;
}

let VecInstType=isVecOther.Value in {
def V4F32Div_prec_ftz : VecBinaryOp<V4AsmStr<"div.rn.ftz.f32">, fdiv, V4F32Regs,
  FDIV32rr_prec_ftz>, Requires<[doF32FTZ, reqPTX20]>;
def V2F32Div_prec_ftz : VecBinaryOp<V2AsmStr<"div.rn.ftz.f32">, fdiv, V2F32Regs,
  FDIV32rr_prec_ftz>, Requires<[doF32FTZ, reqPTX20]>;
def V4F32Div_prec : VecBinaryOp<V4AsmStr<"div.rn.f32">, fdiv, V4F32Regs,
  FDIV32rr_prec>, Requires<[reqPTX20]>;
def V2F32Div_prec : VecBinaryOp<V2AsmStr<"div.rn.f32">, fdiv, V2F32Regs,
  FDIV32rr_prec>, Requires<[reqPTX20]>;
def V2F32Div_ftz : VecBinaryOp<V2AsmStr<"div.full.ftz.f32">, fdiv, V2F32Regs,
  FDIV32rr_ftz>, Requires<[doF32FTZ]>;
def V4F32Div_ftz : VecBinaryOp<V4AsmStr<"div.full.ftz.f32">, fdiv, V4F32Regs,
  FDIV32rr_ftz>, Requires<[doF32FTZ]>;
def V2F32Div : VecBinaryOp<V2AsmStr<"div.full.f32">, fdiv, V2F32Regs, FDIV32rr>;
def V4F32Div : VecBinaryOp<V4AsmStr<"div.full.f32">, fdiv, V4F32Regs, FDIV32rr>;
def V2F64Div : VecBinaryOp<V2AsmStr<"div.rn.f64">, fdiv, V2F64Regs, FDIV64rr>;
}

def fnegpat : PatFrag<(ops node:$in), (fneg node:$in)>;

let VecInstType=isVecOther.Value in {
def VNegv2f32_ftz : VecUnaryOp<V2UnaryStr<"neg.ftz.f32">, fnegpat, V2F32Regs,
  FNEGf32_ftz>, Requires<[doF32FTZ]>;
def VNegv4f32_ftz : VecUnaryOp<V4UnaryStr<"neg.ftz.f32">, fnegpat, V4F32Regs,
  FNEGf32_ftz>, Requires<[doF32FTZ]>;
def VNegv2f32 : VecUnaryOp<V2UnaryStr<"neg.f32">, fnegpat, V2F32Regs, FNEGf32>;
def VNegv4f32 : VecUnaryOp<V4UnaryStr<"neg.f32">, fnegpat, V4F32Regs, FNEGf32>;
def VNegv2f64 : VecUnaryOp<V2UnaryStr<"neg.f64">, fnegpat, V2F64Regs, FNEGf64>;

// Logical Arithmetic
defm VAnd : IntBinVOp<"and.b", and, ANDb64rr, ANDb32rr, ANDb16rr, ANDb8rr>;
defm VOr  : IntBinVOp<"or.b", or, ORb64rr, ORb32rr, ORb16rr, ORb8rr>;
defm VXor : IntBinVOp<"xor.b", xor, XORb64rr, XORb32rr, XORb16rr, XORb8rr>;

defm VNot : IntUnaryVOp<"not.b", not, NOT64, NOT32, NOT16, NOT8>;
}


multiclass V2FPCONTRACT32_SUB_PAT<NVPTXInst Inst, Predicate Pred> {
  def : Pat<(fsub V2F32Regs:$a, (fmul V2F32Regs:$b, V2F32Regs:$c)),
          (Inst (VNegv2f32 V2F32Regs:$b), V2F32Regs:$c,  V2F32Regs:$a)>,
          Requires<[Pred]>;

  def : Pat<(fsub (fmul V2F32Regs:$a, V2F32Regs:$b), V2F32Regs:$c),
          (Inst V2F32Regs:$a, V2F32Regs:$b, (VNegv2f32 V2F32Regs:$c))>,
          Requires<[Pred]>;
}

defm V2FMAF32ext_ftz  : V2FPCONTRACT32_SUB_PAT<F32FMA_ftzV2, doFMAF32AGG_ftz>;
defm V2FMADF32ext_ftz : V2FPCONTRACT32_SUB_PAT<F32MAD_ftzV2, doFMADF32_ftz>;
defm V2FMAF32ext  : V2FPCONTRACT32_SUB_PAT<F32FMAV2, doFMAF32AGG>;
defm V2FMADF32ext : V2FPCONTRACT32_SUB_PAT<F32MADV2, doFMADF32>;

multiclass V4FPCONTRACT32_SUB_PAT<NVPTXInst Inst, Predicate Pred> {
  def : Pat<(fsub V4F32Regs:$a, (fmul V4F32Regs:$b, V4F32Regs:$c)),
          (Inst (VNegv4f32 V4F32Regs:$b), V4F32Regs:$c,  V4F32Regs:$a)>,
          Requires<[Pred]>;

  def : Pat<(fsub (fmul V4F32Regs:$a, V4F32Regs:$b), V4F32Regs:$c),
          (Inst V4F32Regs:$a, V4F32Regs:$b, (VNegv4f32 V4F32Regs:$c))>,
          Requires<[Pred]>;
}

defm V4FMAF32ext_ftz  : V4FPCONTRACT32_SUB_PAT<F32FMA_ftzV4, doFMAF32AGG_ftz>;
defm V4FMADF32ext_ftz : V4FPCONTRACT32_SUB_PAT<F32MAD_ftzV4, doFMADF32_ftz>;
defm V4FMAF32ext  : V4FPCONTRACT32_SUB_PAT<F32FMAV4, doFMAF32AGG>;
defm V4FMADF32ext : V4FPCONTRACT32_SUB_PAT<F32MADV4, doFMADF32>;

multiclass V2FPCONTRACT64_SUB_PAT<NVPTXInst Inst, Predicate Pred> {
  def : Pat<(fsub V2F64Regs:$a, (fmul V2F64Regs:$b, V2F64Regs:$c)),
          (Inst (VNegv2f64 V2F64Regs:$b), V2F64Regs:$c, V2F64Regs:$a)>,
          Requires<[Pred]>;

  def : Pat<(fsub (fmul V2F64Regs:$a, V2F64Regs:$b), V2F64Regs:$c),
          (Inst V2F64Regs:$a, V2F64Regs:$b, (VNegv2f64 V2F64Regs:$c))>,
          Requires<[Pred]>;
}

defm V2FMAF64ext : V2FPCONTRACT64_SUB_PAT<F64FMAV2, doFMAF64AGG>;

class VecModStr<string vecsize, string elem, string extra, string l="">
{
  string t1 = !strconcat("${c", elem);
  string t2 = !strconcat(t1, ":vecv");
  string t3 = !strconcat(t2, vecsize);
  string t4 = !strconcat(t3, extra);
  string t5 = !strconcat(t4, l);
  string s =  !strconcat(t5, "}");
}
class ShuffleOneLine<string vecsize, string elem, string type>
{
  string t1 = VecModStr<vecsize, elem, "comm", "1">.s;
  string t2 = !strconcat(t1, "mov.");
  string t3 = !strconcat(t2, type);
  string t4 = !strconcat(t3, " \t${dst}_");
  string t5 = !strconcat(t4, elem);
  string t6 = !strconcat(t5, ", $src1");
  string t7 = !strconcat(t6, VecModStr<vecsize, elem, "pos">.s);
  string t8 = !strconcat(t7, ";\n\t");
  string t9 = !strconcat(t8, VecModStr<vecsize, elem, "comm", "2">.s);
  string t10 = !strconcat(t9, "mov.");
  string t11 = !strconcat(t10, type);
  string t12 = !strconcat(t11, " \t${dst}_");
  string t13 = !strconcat(t12, elem);
  string t14 = !strconcat(t13, ", $src2");
  string t15 = !strconcat(t14, VecModStr<vecsize, elem, "pos">.s);
  string s =   !strconcat(t15, ";");
}
class ShuffleAsmStr2<string type>
{
  string t1 = ShuffleOneLine<"2", "0", type>.s;
  string t2 = !strconcat(t1, "\n\t");
  string s  = !strconcat(t2, ShuffleOneLine<"2", "1", type>.s);
}
class ShuffleAsmStr4<string type>
{
  string t1 = ShuffleOneLine<"4", "0", type>.s;
  string t2 = !strconcat(t1, "\n\t");
  string t3 = !strconcat(t2, ShuffleOneLine<"4", "1", type>.s);
  string t4 = !strconcat(t3, "\n\t");
  string t5 = !strconcat(t4, ShuffleOneLine<"4", "2", type>.s);
  string t6 = !strconcat(t5, "\n\t");
  string s  = !strconcat(t6, ShuffleOneLine<"4", "3", type>.s);
}

let neverHasSideEffects=1, VecInstType=isVecShuffle.Value in {
def VecShuffle_v4f32 : NVPTXVecInst<(outs V4F32Regs:$dst),
                       (ins  V4F32Regs:$src1, V4F32Regs:$src2,
                             i8imm:$c0, i8imm:$c1, i8imm:$c2, i8imm:$c3),
                 !strconcat("//Mov $dst, $src1, $src2, $c0, $c1, $c2, $c3;\n\t",
                                 ShuffleAsmStr4<"f32">.s),
                       [], FMOV32rr>;

def VecShuffle_v4i32 : NVPTXVecInst<(outs V4I32Regs:$dst),
                       (ins  V4I32Regs:$src1, V4I32Regs:$src2,
                             i8imm:$c0, i8imm:$c1, i8imm:$c2, i8imm:$c3),
                 !strconcat("//Mov $dst, $src1, $src2, $c0, $c1, $c2, $c3;\n\t",
                                 ShuffleAsmStr4<"u32">.s),
                       [], IMOV32rr>;

def VecShuffle_v4i16 : NVPTXVecInst<(outs V4I16Regs:$dst),
                       (ins  V4I16Regs:$src1, V4I16Regs:$src2,
                             i8imm:$c0, i8imm:$c1, i8imm:$c2, i8imm:$c3),
                 !strconcat("//Mov $dst, $src1, $src2, $c0, $c1, $c2, $c3;\n\t",
                                 ShuffleAsmStr4<"u16">.s),
                       [], IMOV16rr>;

def VecShuffle_v4i8 : NVPTXVecInst<(outs V4I8Regs:$dst),
                       (ins  V4I8Regs:$src1, V4I8Regs:$src2,
                             i8imm:$c0, i8imm:$c1, i8imm:$c2, i8imm:$c3),
                 !strconcat("//Mov $dst, $src1, $src2, $c0, $c1, $c2, $c3;\n\t",
                                 ShuffleAsmStr4<"u16">.s),
                       [], IMOV8rr>;

def VecShuffle_v2f32 : NVPTXVecInst<(outs V2F32Regs:$dst),
                       (ins  V2F32Regs:$src1, V2F32Regs:$src2,
                             i8imm:$c0, i8imm:$c1),
                       !strconcat("//Mov $dst, $src1, $src2, $c0, $c1;\n\t",
                                 ShuffleAsmStr2<"f32">.s),
                       [], FMOV32rr>;

def VecShuffle_v2i32 : NVPTXVecInst<(outs V2I32Regs:$dst),
                       (ins  V2I32Regs:$src1, V2I32Regs:$src2,
                             i8imm:$c0, i8imm:$c1),
                       !strconcat("//Mov $dst, $src1, $src2, $c0, $c1;\n\t",
                                 ShuffleAsmStr2<"u32">.s),
                       [], IMOV32rr>;

def VecShuffle_v2i8 : NVPTXVecInst<(outs V2I8Regs:$dst),
                       (ins  V2I8Regs:$src1, V2I8Regs:$src2,
                             i8imm:$c0, i8imm:$c1),
                       !strconcat("//Mov $dst, $src1, $src2, $c0, $c1;\n\t",
                                 ShuffleAsmStr2<"u16">.s),
                       [], IMOV8rr>;

def VecShuffle_v2i16 : NVPTXVecInst<(outs V2I16Regs:$dst),
                       (ins  V2I16Regs:$src1, V2I16Regs:$src2,
                             i8imm:$c0, i8imm:$c1),
                       !strconcat("//Mov $dst, $src1, $src2, $c0, $c1;\n\t",
                                 ShuffleAsmStr2<"u16">.s),
                       [], IMOV16rr>;

def VecShuffle_v2f64 : NVPTXVecInst<(outs V2F64Regs:$dst),
                       (ins  V2F64Regs:$src1, V2F64Regs:$src2,
                             i8imm:$c0, i8imm:$c1),
                       !strconcat("//Mov $dst, $src1, $src2, $c0, $c1;\n\t",
                                 ShuffleAsmStr2<"f64">.s),
                       [], FMOV64rr>;

def VecShuffle_v2i64 : NVPTXVecInst<(outs V2I64Regs:$dst),
                       (ins  V2I64Regs:$src1, V2I64Regs:$src2,
                             i8imm:$c0, i8imm:$c1),
                       !strconcat("//Mov $dst, $src1, $src2, $c0, $c1;\n\t",
                                 ShuffleAsmStr2<"u64">.s),
                       [], IMOV64rr>;
}

def ShuffleMask0 : SDNodeXForm<vector_shuffle, [{
  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(N);
  return CurDAG->getTargetConstant(SVOp->getMaskElt(0), MVT::i32);
}]>;
def ShuffleMask1 : SDNodeXForm<vector_shuffle, [{
  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(N);
  return CurDAG->getTargetConstant(SVOp->getMaskElt(1), MVT::i32);
}]>;
def ShuffleMask2 : SDNodeXForm<vector_shuffle, [{
  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(N);
  return CurDAG->getTargetConstant(SVOp->getMaskElt(2), MVT::i32);
}]>;
def ShuffleMask3 : SDNodeXForm<vector_shuffle, [{
  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(N);
  return CurDAG->getTargetConstant(SVOp->getMaskElt(3), MVT::i32);
}]>;

// The spurious call is here to silence a compiler warning about N being
// unused.
def vec_shuf : PatFrag<(ops node:$lhs, node:$rhs),
                       (vector_shuffle node:$lhs, node:$rhs),
                       [{ N->getGluedNode(); return true; }]>;

def : Pat<(v2f64 (vec_shuf:$op V2F64Regs:$src1, V2F64Regs:$src2)),
          (VecShuffle_v2f64 V2F64Regs:$src1, V2F64Regs:$src2,
                            (ShuffleMask0 node:$op), (ShuffleMask1 node:$op))>;

def : Pat<(v4f32 (vec_shuf:$op V4F32Regs:$src1, V4F32Regs:$src2)),
          (VecShuffle_v4f32 V4F32Regs:$src1, V4F32Regs:$src2,
                            (ShuffleMask0 node:$op), (ShuffleMask1 node:$op),
                            (ShuffleMask2 node:$op), (ShuffleMask3 node:$op))>;

def : Pat<(v2f32 (vec_shuf:$op V2F32Regs:$src1, V2F32Regs:$src2)),
          (VecShuffle_v2f32 V2F32Regs:$src1, V2F32Regs:$src2,
                            (ShuffleMask0 node:$op), (ShuffleMask1 node:$op))>;

def : Pat<(v2i64 (vec_shuf:$op V2I64Regs:$src1, V2I64Regs:$src2)),
          (VecShuffle_v2i64 V2I64Regs:$src1, V2I64Regs:$src2,
                            (ShuffleMask0 node:$op), (ShuffleMask1 node:$op))>;

def : Pat<(v4i32 (vec_shuf:$op V4I32Regs:$src1, V4I32Regs:$src2)),
          (VecShuffle_v4i32 V4I32Regs:$src1, V4I32Regs:$src2,
                            (ShuffleMask0 node:$op), (ShuffleMask1 node:$op),
                            (ShuffleMask2 node:$op), (ShuffleMask3 node:$op))>;

def : Pat<(v2i32 (vec_shuf:$op V2I32Regs:$src1, V2I32Regs:$src2)),
          (VecShuffle_v2i32 V2I32Regs:$src1, V2I32Regs:$src2,
                            (ShuffleMask0 node:$op), (ShuffleMask1 node:$op))>;

def : Pat<(v4i16 (vec_shuf:$op V4I16Regs:$src1, V4I16Regs:$src2)),
          (VecShuffle_v4i16 V4I16Regs:$src1, V4I16Regs:$src2,
                            (ShuffleMask0 node:$op), (ShuffleMask1 node:$op),
                            (ShuffleMask2 node:$op), (ShuffleMask3 node:$op))>;

def : Pat<(v2i16 (vec_shuf:$op V2I16Regs:$src1, V2I16Regs:$src2)),
          (VecShuffle_v2i16 V2I16Regs:$src1, V2I16Regs:$src2,
                            (ShuffleMask0 node:$op), (ShuffleMask1 node:$op))>;

def : Pat<(v4i8 (vec_shuf:$op V4I8Regs:$src1, V4I8Regs:$src2)),
          (VecShuffle_v4i8 V4I8Regs:$src1, V4I8Regs:$src2,
                            (ShuffleMask0 node:$op), (ShuffleMask1 node:$op),
                            (ShuffleMask2 node:$op), (ShuffleMask3 node:$op))>;

def : Pat<(v2i8 (vec_shuf:$op V2I8Regs:$src1, V2I8Regs:$src2)),
          (VecShuffle_v2i8 V2I8Regs:$src1, V2I8Regs:$src2,
                            (ShuffleMask0 node:$op), (ShuffleMask1 node:$op))>;

class Build_Vector2<string asmstr, NVPTXRegClass vclass, NVPTXRegClass sclass,
  NVPTXInst si>
                   : NVPTXVecInst<(outs vclass:$dst),
                   (ins  sclass:$a1, sclass:$a2),
                   !strconcat(asmstr, "\t${dst:vecfull}, {{$a1, $a2}};"),
                   [(set vclass:$dst, (build_vector sclass:$a1, sclass:$a2))],
                   si>;
class Build_Vector4<string asmstr, NVPTXRegClass vclass, NVPTXRegClass sclass,
  NVPTXInst si>
                   : NVPTXVecInst<(outs vclass:$dst),
                   (ins  sclass:$a1, sclass:$a2, sclass:$a3, sclass:$a4),
               !strconcat(asmstr, "\t${dst:vecfull}, {{$a1, $a2, $a3, $a4}};"),
                   [(set vclass:$dst,
                     (build_vector sclass:$a1, sclass:$a2,
                       sclass:$a3, sclass:$a4))], si>;

let isAsCheapAsAMove=1, VecInstType=isVecBuild.Value in {
def Build_Vector2_f32 : Build_Vector2<"mov.v2.f32", V2F32Regs, Float32Regs,
  FMOV32rr>;
def Build_Vector2_f64 : Build_Vector2<"mov.v2.f64", V2F64Regs, Float64Regs,
  FMOV64rr>;

def Build_Vector2_i32 : Build_Vector2<"mov.v2.u32", V2I32Regs, Int32Regs,
  IMOV32rr>;
def Build_Vector2_i64 : Build_Vector2<"mov.v2.u64", V2I64Regs, Int64Regs,
  IMOV64rr>;
def Build_Vector2_i16 : Build_Vector2<"mov.v2.u16", V2I16Regs, Int16Regs,
  IMOV16rr>;
def Build_Vector2_i8  : Build_Vector2<"mov.v2.u16",  V2I8Regs,  Int8Regs,
  IMOV8rr>;

def Build_Vector4_f32 : Build_Vector4<"mov.v4.f32", V4F32Regs, Float32Regs,
  FMOV32rr>;

def Build_Vector4_i32 : Build_Vector4<"mov.v4.u32", V4I32Regs, Int32Regs,
  IMOV32rr>;
def Build_Vector4_i16 : Build_Vector4<"mov.v4.u16", V4I16Regs, Int16Regs,
  IMOV16rr>;
def Build_Vector4_i8  : Build_Vector4<"mov.v4.u16", V4I8Regs, Int8Regs,
  IMOV8rr>;
}

class Vec_Move<string asmstr, NVPTXRegClass vclass, NVPTXInst sop=NOP>
                 : NVPTXVecInst<(outs vclass:$dst), (ins vclass:$src),
                   !strconcat(asmstr, "\t${dst:vecfull}, ${src:vecfull};"),
                   [], sop>;

let isAsCheapAsAMove=1, neverHasSideEffects=1, IsSimpleMove=1,
  VecInstType=isVecOther.Value in {
def V4f32Mov : Vec_Move<"mov.v4.f32", V4F32Regs, FMOV32rr>;
def V2f32Mov : Vec_Move<"mov.v2.f32", V2F32Regs, FMOV32rr>;

def V4i32Mov : Vec_Move<"mov.v4.u32", V4I32Regs, IMOV32rr>;
def V2i32Mov : Vec_Move<"mov.v2.u32", V2I32Regs, IMOV32rr>;

def V4i16Mov : Vec_Move<"mov.v4.u16", V4I16Regs, IMOV16rr>;
def V2i16Mov : Vec_Move<"mov.v2.u16", V2I16Regs, IMOV16rr>;

def V4i8Mov : Vec_Move<"mov.v4.u16", V4I8Regs, IMOV8rr>;
def V2i8Mov : Vec_Move<"mov.v2.u16", V2I8Regs, IMOV8rr>;

def V2f64Mov : Vec_Move<"mov.v2.f64", V2F64Regs, FMOV64rr>;
def V2i64Mov : Vec_Move<"mov.v2.u64", V2I64Regs, IMOV64rr>;
}

// extract subvector patterns
def extract_subvec : SDNode<"ISD::EXTRACT_SUBVECTOR",
                        SDTypeProfile<1, 2, [SDTCisPtrTy<2>]>>;

def : Pat<(v2f32 (extract_subvec V4F32Regs:$src, 0)),
                 (Build_Vector2_f32 (V4f32Extract V4F32Regs:$src, 0),
                                    (V4f32Extract V4F32Regs:$src, 1))>;
def : Pat<(v2f32 (extract_subvec V4F32Regs:$src, 2)),
                 (Build_Vector2_f32 (V4f32Extract V4F32Regs:$src, 2),
                                    (V4f32Extract V4F32Regs:$src, 3))>;
def : Pat<(v2i32 (extract_subvec V4I32Regs:$src, 0)),
                 (Build_Vector2_i32 (V4i32Extract V4I32Regs:$src, 0),
                                    (V4i32Extract V4I32Regs:$src, 1))>;
def : Pat<(v2i32 (extract_subvec V4I32Regs:$src, 2)),
                 (Build_Vector2_i32 (V4i32Extract V4I32Regs:$src, 2),
                                    (V4i32Extract V4I32Regs:$src, 3))>;
def : Pat<(v2i16 (extract_subvec V4I16Regs:$src, 0)),
                 (Build_Vector2_i16 (V4i16Extract V4I16Regs:$src, 0),
                                    (V4i16Extract V4I16Regs:$src, 1))>;
def : Pat<(v2i16 (extract_subvec V4I16Regs:$src, 2)),
                 (Build_Vector2_i16 (V4i16Extract V4I16Regs:$src, 2),
                                    (V4i16Extract V4I16Regs:$src, 3))>;
def : Pat<(v2i8 (extract_subvec V4I8Regs:$src, 0)),
                 (Build_Vector2_i8 (V4i8Extract V4I8Regs:$src, 0),
                                    (V4i8Extract V4I8Regs:$src, 1))>;
def : Pat<(v2i8 (extract_subvec V4I8Regs:$src, 2)),
                 (Build_Vector2_i8 (V4i8Extract V4I8Regs:$src, 2),
                                    (V4i8Extract V4I8Regs:$src, 3))>;

// Select instructions
class Select_OneLine<string type, string pos> {
  string t1 = !strconcat("selp.", type);
  string t2 = !strconcat(t1, " \t${dst}_");
  string t3 = !strconcat(t2, pos);
  string t4 = !strconcat(t3, ", ${src1}_");
  string t5 = !strconcat(t4, pos);
  string t6 = !strconcat(t5, ", ${src2}_");
  string t7 = !strconcat(t6, pos);
  string s  = !strconcat(t7, ", $p;");
}

class Select_Str2<string type> {
  string t1 = Select_OneLine<type, "0">.s;
  string t2 = !strconcat(t1, "\n\t");
  string s  = !strconcat(t2, Select_OneLine<type, "1">.s);
}

class Select_Str4<string type> {
  string t1 = Select_OneLine<type, "0">.s;
  string t2 = !strconcat(t1, "\n\t");
  string t3 = !strconcat(t2, Select_OneLine<type, "1">.s);
  string t4 = !strconcat(t3, "\n\t");
  string t5 = !strconcat(t4, Select_OneLine<type, "2">.s);
  string t6 = !strconcat(t5, "\n\t");
  string s  = !strconcat(t6, Select_OneLine<type, "3">.s);

}

class Vec_Select<NVPTXRegClass vclass, string asmstr, NVPTXInst sop>
      : NVPTXVecInst<(outs vclass:$dst),
                     (ins  vclass:$src1, vclass:$src2, Int1Regs:$p),
                     asmstr,
                     [(set vclass:$dst, (select Int1Regs:$p, vclass:$src1,
                       vclass:$src2))],
                     sop>;

let VecInstType=isVecOther.Value in {
def V2I64_Select : Vec_Select<V2I64Regs, Select_Str2<"b64">.s, SELECTi64rr>;
def V4I32_Select : Vec_Select<V4I32Regs, Select_Str4<"b32">.s, SELECTi32rr>;
def V2I32_Select : Vec_Select<V2I32Regs, Select_Str2<"b32">.s, SELECTi32rr>;
def V4I16_Select : Vec_Select<V4I16Regs, Select_Str4<"b16">.s, SELECTi16rr>;
def V2I16_Select : Vec_Select<V2I16Regs, Select_Str2<"b16">.s, SELECTi16rr>;
def V4I8_Select  : Vec_Select<V4I8Regs,  Select_Str4<"b16">.s, SELECTi8rr>;
def V2I8_Select  : Vec_Select<V2I8Regs,  Select_Str2<"b16">.s, SELECTi8rr>;

def V2F64_Select : Vec_Select<V2F64Regs, Select_Str2<"f64">.s, SELECTf64rr>;
def V4F32_Select : Vec_Select<V4F32Regs, Select_Str4<"f32">.s, SELECTf32rr>;
def V2F32_Select : Vec_Select<V2F32Regs, Select_Str2<"f32">.s, SELECTf32rr>;
}

// Comparison instructions

// setcc convenience fragments.
def vsetoeq : PatFrag<(ops node:$lhs, node:$rhs),
                      (setcc node:$lhs, node:$rhs, SETOEQ)>;
def vsetogt : PatFrag<(ops node:$lhs, node:$rhs),
                      (setcc node:$lhs, node:$rhs, SETOGT)>;
def vsetoge : PatFrag<(ops node:$lhs, node:$rhs),
                      (setcc node:$lhs, node:$rhs, SETOGE)>;
def vsetolt : PatFrag<(ops node:$lhs, node:$rhs),
                      (setcc node:$lhs, node:$rhs, SETOLT)>;
def vsetole : PatFrag<(ops node:$lhs, node:$rhs),
                      (setcc node:$lhs, node:$rhs, SETOLE)>;
def vsetone : PatFrag<(ops node:$lhs, node:$rhs),
                      (setcc node:$lhs, node:$rhs, SETONE)>;
def vseto   : PatFrag<(ops node:$lhs, node:$rhs),
                      (setcc node:$lhs, node:$rhs, SETO)>;
def vsetuo  : PatFrag<(ops node:$lhs, node:$rhs),
                      (setcc node:$lhs, node:$rhs, SETUO)>;
def vsetueq : PatFrag<(ops node:$lhs, node:$rhs),
                      (setcc node:$lhs, node:$rhs, SETUEQ)>;
def vsetugt : PatFrag<(ops node:$lhs, node:$rhs),
                      (setcc node:$lhs, node:$rhs, SETUGT)>;
def vsetuge : PatFrag<(ops node:$lhs, node:$rhs),
                      (setcc node:$lhs, node:$rhs, SETUGE)>;
def vsetult : PatFrag<(ops node:$lhs, node:$rhs),
                      (setcc node:$lhs, node:$rhs, SETULT)>;
def vsetule : PatFrag<(ops node:$lhs, node:$rhs),
                      (setcc node:$lhs, node:$rhs, SETULE)>;
def vsetune : PatFrag<(ops node:$lhs, node:$rhs),
                      (setcc node:$lhs, node:$rhs, SETUNE)>;
def vseteq  : PatFrag<(ops node:$lhs, node:$rhs),
                      (setcc node:$lhs, node:$rhs, SETEQ)>;
def vsetgt  : PatFrag<(ops node:$lhs, node:$rhs),
                      (setcc node:$lhs, node:$rhs, SETGT)>;
def vsetge  : PatFrag<(ops node:$lhs, node:$rhs),
                      (setcc node:$lhs, node:$rhs, SETGE)>;
def vsetlt  : PatFrag<(ops node:$lhs, node:$rhs),
                      (setcc node:$lhs, node:$rhs, SETLT)>;
def vsetle  : PatFrag<(ops node:$lhs, node:$rhs),
                      (setcc node:$lhs, node:$rhs, SETLE)>;
def vsetne  : PatFrag<(ops node:$lhs, node:$rhs),
                      (setcc node:$lhs, node:$rhs, SETNE)>;

class Vec_Compare<PatFrag op, NVPTXRegClass outrclass, NVPTXRegClass inrclass,
  NVPTXInst sop>
    : NVPTXVecInst<(outs outrclass:$dst),
                   (ins  inrclass:$a, inrclass:$b),
                   "Unsupported",
                   [(set outrclass:$dst, (op inrclass:$a, inrclass:$b))],
                   sop>;

multiclass Vec_Compare_All<PatFrag op,
                           NVPTXInst inst8,
                           NVPTXInst inst16,
                           NVPTXInst inst32,
                           NVPTXInst inst64>
{
  def  V2I8 : Vec_Compare<op, V2I8Regs,  V2I8Regs,  inst8>;
  def  V4I8 : Vec_Compare<op, V4I8Regs,  V4I8Regs,  inst8>;
  def V2I16 : Vec_Compare<op, V2I16Regs, V2I16Regs, inst16>;
  def V4I16 : Vec_Compare<op, V4I16Regs, V4I16Regs, inst16>;
  def V2I32 : Vec_Compare<op, V2I32Regs, V2I32Regs, inst32>;
  def V4I32 : Vec_Compare<op, V4I32Regs, V4I32Regs, inst32>;
  def V2I64 : Vec_Compare<op, V2I64Regs, V2I64Regs, inst64>;
}

let VecInstType=isVecOther.Value in {
  defm VecSGT : Vec_Compare_All<vsetgt,  ISetSGTi8rr_toi8, ISetSGTi16rr_toi16,
    ISetSGTi32rr_toi32, ISetSGTi64rr_toi64>;
  defm VecUGT : Vec_Compare_All<vsetugt, ISetUGTi8rr_toi8, ISetUGTi16rr_toi16,
    ISetUGTi32rr_toi32, ISetUGTi64rr_toi64>;
  defm VecSLT : Vec_Compare_All<vsetlt,  ISetSLTi8rr_toi8, ISetSLTi16rr_toi16,
    ISetSLTi32rr_toi32, ISetSLTi64rr_toi64>;
  defm VecULT : Vec_Compare_All<vsetult, ISetULTi8rr_toi8, ISetULTi16rr_toi16,
    ISetULTi32rr_toi32, ISetULTi64rr_toi64>;
  defm VecSGE : Vec_Compare_All<vsetge,  ISetSGEi8rr_toi8, ISetSGEi16rr_toi16,
    ISetSGEi32rr_toi32, ISetSGEi64rr_toi64>;
  defm VecUGE : Vec_Compare_All<vsetuge, ISetUGEi8rr_toi8, ISetUGEi16rr_toi16,
    ISetUGEi32rr_toi32, ISetUGEi64rr_toi64>;
  defm VecSLE : Vec_Compare_All<vsetle,  ISetSLEi8rr_toi8, ISetSLEi16rr_toi16,
    ISetSLEi32rr_toi32, ISetSLEi64rr_toi64>;
  defm VecULE : Vec_Compare_All<vsetule, ISetULEi8rr_toi8, ISetULEi16rr_toi16,
    ISetULEi32rr_toi32, ISetULEi64rr_toi64>;
  defm VecSEQ : Vec_Compare_All<vseteq,  ISetSEQi8rr_toi8, ISetSEQi16rr_toi16,
    ISetSEQi32rr_toi32, ISetSEQi64rr_toi64>;
  defm VecUEQ : Vec_Compare_All<vsetueq, ISetUEQi8rr_toi8, ISetUEQi16rr_toi16,
    ISetUEQi32rr_toi32, ISetUEQi64rr_toi64>;
  defm VecSNE : Vec_Compare_All<vsetne,  ISetSNEi8rr_toi8, ISetSNEi16rr_toi16,
    ISetSNEi32rr_toi32, ISetSNEi64rr_toi64>;
  defm VecUNE : Vec_Compare_All<vsetune, ISetUNEi8rr_toi8, ISetUNEi16rr_toi16,
    ISetUNEi32rr_toi32, ISetUNEi64rr_toi64>;
}

multiclass FVec_Compare_All<PatFrag op,
                            NVPTXInst instf32,
                            NVPTXInst instf64>
{
  def V2F32 : Vec_Compare<op, V2I32Regs, V2F32Regs, instf32>;
  def V4F32 : Vec_Compare<op, V4I32Regs, V4F32Regs, instf32>;
  def V2F64 : Vec_Compare<op, V2I64Regs, V2F64Regs, instf64>;
}

let VecInstType=isVecOther.Value in {
  defm FVecGT :  FVec_Compare_All<vsetogt, FSetGTf32rr_toi32,
    FSetGTf64rr_toi64>;
  defm FVecLT :  FVec_Compare_All<vsetolt, FSetLTf32rr_toi32,
    FSetLTf64rr_toi64>;
  defm FVecGE :  FVec_Compare_All<vsetoge, FSetGEf32rr_toi32,
    FSetGEf64rr_toi64>;
  defm FVecLE :  FVec_Compare_All<vsetole, FSetLEf32rr_toi32,
    FSetLEf64rr_toi64>;
  defm FVecEQ :  FVec_Compare_All<vsetoeq, FSetEQf32rr_toi32,
    FSetEQf64rr_toi64>;
  defm FVecNE :  FVec_Compare_All<vsetone, FSetNEf32rr_toi32,
    FSetNEf64rr_toi64>;

  defm FVecUGT :  FVec_Compare_All<vsetugt, FSetUGTf32rr_toi32,
    FSetUGTf64rr_toi64>;
  defm FVecULT :  FVec_Compare_All<vsetult, FSetULTf32rr_toi32,
    FSetULTf64rr_toi64>;
  defm FVecUGE :  FVec_Compare_All<vsetuge, FSetUGEf32rr_toi32,
    FSetUGEf64rr_toi64>;
  defm FVecULE :  FVec_Compare_All<vsetule, FSetULEf32rr_toi32,
    FSetULEf64rr_toi64>;
  defm FVecUEQ :  FVec_Compare_All<vsetueq, FSetUEQf32rr_toi32,
    FSetUEQf64rr_toi64>;
  defm FVecUNE :  FVec_Compare_All<vsetune, FSetUNEf32rr_toi32,
    FSetUNEf64rr_toi64>;

  defm FVecNUM :  FVec_Compare_All<vseto,  FSetNUMf32rr_toi32,
    FSetNUMf64rr_toi64>;
  defm FVecNAN :  FVec_Compare_All<vsetuo, FSetNANf32rr_toi32,
    FSetNANf64rr_toi64>;
}

class LoadParamScalar4Inst<NVPTXRegClass regclass, string opstr> :
      NVPTXInst<(outs regclass:$d1, regclass:$d2, regclass:$d3, regclass:$d4),
                (ins i32imm:$a, i32imm:$b),
                !strconcat(!strconcat("ld.param", opstr),
                  "\t{{$d1, $d2, $d3, $d4}}, [retval0+$b];"), []>;

class LoadParamScalar2Inst<NVPTXRegClass regclass, string opstr> :
      NVPTXInst<(outs regclass:$d1, regclass:$d2),
                (ins i32imm:$a, i32imm:$b),
                !strconcat(!strconcat("ld.param", opstr),
                  "\t{{$d1, $d2}}, [retval0+$b];"), []>;


class StoreParamScalar4Inst<NVPTXRegClass regclass, string opstr> :
      NVPTXInst<(outs),
                (ins regclass:$s1, regclass:$s2, regclass:$s3, regclass:$s4,
                  i32imm:$a, i32imm:$b),
                !strconcat(!strconcat("st.param", opstr),
                  "\t[param$a+$b], {{$s1, $s2, $s3, $s4}};"), []>;

class StoreParamScalar2Inst<NVPTXRegClass regclass, string opstr> :
      NVPTXInst<(outs),
                (ins regclass:$s1, regclass:$s2, i32imm:$a, i32imm:$b),
                !strconcat(!strconcat("st.param", opstr),
                  "\t[param$a+$b], {{$s1, $s2}};"), []>;

class StoreRetvalScalar4Inst<NVPTXRegClass regclass, string opstr> :
      NVPTXInst<(outs),
                (ins regclass:$s1, regclass:$s2, regclass:$s3, regclass:$s4,
                  i32imm:$a),
                !strconcat(!strconcat("st.param", opstr),
                  "\t[func_retval+$a], {{$s1, $s2, $s3, $s4}};"), []>;

class StoreRetvalScalar2Inst<NVPTXRegClass regclass, string opstr> :
      NVPTXInst<(outs),
                (ins regclass:$s1, regclass:$s2, i32imm:$a),
                !strconcat(!strconcat("st.param", opstr),
                  "\t[func_retval+$a], {{$s1, $s2}};"), []>;

def LoadParamScalar4I32 : LoadParamScalar4Inst<Int32Regs, ".v4.b32">;
def LoadParamScalar4I16 : LoadParamScalar4Inst<Int16Regs, ".v4.b16">;
def LoadParamScalar4I8  : LoadParamScalar4Inst<Int8Regs, ".v4.b8">;

def LoadParamScalar2I64 : LoadParamScalar2Inst<Int32Regs, ".v2.b64">;
def LoadParamScalar2I32 : LoadParamScalar2Inst<Int32Regs, ".v2.b32">;
def LoadParamScalar2I16 : LoadParamScalar2Inst<Int32Regs, ".v2.b16">;
def LoadParamScalar2I8  : LoadParamScalar2Inst<Int32Regs, ".v2.b8">;

def LoadParamScalar4F32 : LoadParamScalar4Inst<Float32Regs, ".v4.f32">;
def LoadParamScalar2F32 : LoadParamScalar2Inst<Float32Regs, ".v2.f32">;
def LoadParamScalar2F64 : LoadParamScalar2Inst<Float64Regs, ".v2.f64">;

def StoreParamScalar4I32 : StoreParamScalar4Inst<Int32Regs, ".v4.b32">;
def StoreParamScalar4I16 : StoreParamScalar4Inst<Int16Regs, ".v4.b16">;
def StoreParamScalar4I8  : StoreParamScalar4Inst<Int8Regs, ".v4.b8">;

def StoreParamScalar2I64 : StoreParamScalar2Inst<Int64Regs, ".v2.b64">;
def StoreParamScalar2I32 : StoreParamScalar2Inst<Int32Regs, ".v2.b32">;
def StoreParamScalar2I16 : StoreParamScalar2Inst<Int16Regs, ".v2.b16">;
def StoreParamScalar2I8  : StoreParamScalar2Inst<Int8Regs, ".v2.b8">;

def StoreParamScalar4F32 : StoreParamScalar4Inst<Float32Regs, ".v4.f32">;
def StoreParamScalar2F32 : StoreParamScalar2Inst<Float32Regs, ".v2.f32">;
def StoreParamScalar2F64 : StoreParamScalar2Inst<Float64Regs, ".v2.f64">;

def StoreRetvalScalar4I32 : StoreRetvalScalar4Inst<Int32Regs, ".v4.b32">;
def StoreRetvalScalar4I16 : StoreRetvalScalar4Inst<Int16Regs, ".v4.b16">;
def StoreRetvalScalar4I8  : StoreRetvalScalar4Inst<Int8Regs, ".v4.b8">;

def StoreRetvalScalar2I64 : StoreRetvalScalar2Inst<Int64Regs, ".v2.b64">;
def StoreRetvalScalar2I32 : StoreRetvalScalar2Inst<Int32Regs, ".v2.b32">;
def StoreRetvalScalar2I16 : StoreRetvalScalar2Inst<Int16Regs, ".v2.b16">;
def StoreRetvalScalar2I8  : StoreRetvalScalar2Inst<Int8Regs, ".v2.b8">;

def StoreRetvalScalar4F32 : StoreRetvalScalar4Inst<Float32Regs, ".v4.f32">;
def StoreRetvalScalar2F32 : StoreRetvalScalar2Inst<Float32Regs, ".v2.f32">;
def StoreRetvalScalar2F64 : StoreRetvalScalar2Inst<Float64Regs, ".v2.f64">;

class LoadParamVecInst<NVPTXRegClass regclass, string opstr, NVPTXInst sop=NOP>:
      NVPTXVecInst<(outs regclass:$dst), (ins i32imm:$a, i32imm:$b),
                "loadparam : $dst <- [$a, $b]",
                [(set regclass:$dst, (LoadParam (i32 imm:$a), (i32 imm:$b)))],
                sop>;

class StoreParamVecInst<NVPTXRegClass regclass, string opstr, NVPTXInst sop=NOP>
      : NVPTXVecInst<(outs), (ins regclass:$val, i32imm:$a, i32imm:$b),
                "storeparam : [$a, $b] <- $val",
                [(StoreParam (i32 imm:$a), (i32 imm:$b), regclass:$val)], sop>;

class StoreRetvalVecInst<NVPTXRegClass regclass, string opstr,
  NVPTXInst sop=NOP>
      : NVPTXVecInst<(outs), (ins regclass:$val, i32imm:$a),
                "storeretval : retval[$a] <- $val",
                [(StoreRetval (i32 imm:$a), regclass:$val)], sop>;

let VecInstType=isVecLD.Value in {
def LoadParamV4I32  : LoadParamVecInst<V4I32Regs, ".v4.b32",
  LoadParamScalar4I32>;
def LoadParamV4I16  : LoadParamVecInst<V4I16Regs, ".v4.b16",
  LoadParamScalar4I16>;
def LoadParamV4I8   : LoadParamVecInst<V4I8Regs, ".v4.b8",
  LoadParamScalar4I8>;

def LoadParamV2I64  : LoadParamVecInst<V2I64Regs, ".v2.b64",
  LoadParamScalar2I64>;
def LoadParamV2I32  : LoadParamVecInst<V2I32Regs, ".v2.b32",
  LoadParamScalar2I32>;
def LoadParamV2I16  : LoadParamVecInst<V2I16Regs, ".v2.b16",
  LoadParamScalar2I16>;
def LoadParamV2I8   : LoadParamVecInst<V2I8Regs, ".v2.b8",
  LoadParamScalar2I8>;

def LoadParamV4F32  : LoadParamVecInst<V4F32Regs, ".v4.f32",
  LoadParamScalar4F32>;
def LoadParamV2F32  : LoadParamVecInst<V2F32Regs, ".v2.f32",
  LoadParamScalar2F32>;
def LoadParamV2F64  : LoadParamVecInst<V2F64Regs, ".v2.f64",
  LoadParamScalar2F64>;
}

let VecInstType=isVecST.Value in {
def StoreParamV4I32  : StoreParamVecInst<V4I32Regs, ".v4.b32",
  StoreParamScalar4I32>;
def StoreParamV4I16  : StoreParamVecInst<V4I16Regs, ".v4.b16",
  StoreParamScalar4I16>;
def StoreParamV4I8   : StoreParamVecInst<V4I8Regs, ".v4.b8",
  StoreParamScalar4I8>;

def StoreParamV2I64  : StoreParamVecInst<V2I64Regs, ".v2.b64",
  StoreParamScalar2I64>;
def StoreParamV2I32  : StoreParamVecInst<V2I32Regs, ".v2.b32",
  StoreParamScalar2I32>;
def StoreParamV2I16  : StoreParamVecInst<V2I16Regs, ".v2.b16",
  StoreParamScalar2I16>;
def StoreParamV2I8   : StoreParamVecInst<V2I8Regs, ".v2.b8",
  StoreParamScalar2I8>;

def StoreParamV4F32  : StoreParamVecInst<V4F32Regs, ".v4.f32",
  StoreParamScalar4F32>;
def StoreParamV2F32  : StoreParamVecInst<V2F32Regs, ".v2.f32",
  StoreParamScalar2F32>;
def StoreParamV2F64  : StoreParamVecInst<V2F64Regs, ".v2.f64",
  StoreParamScalar2F64>;

def StoreRetvalV4I32  : StoreRetvalVecInst<V4I32Regs, ".v4.b32",
  StoreRetvalScalar4I32>;
def StoreRetvalV4I16  : StoreRetvalVecInst<V4I16Regs, ".v4.b16",
  StoreRetvalScalar4I16>;
def StoreRetvalV4I8   : StoreRetvalVecInst<V4I8Regs,  ".v4.b8",
  StoreRetvalScalar4I8>;

def StoreRetvalV2I64  : StoreRetvalVecInst<V2I64Regs, ".v2.b64",
  StoreRetvalScalar2I64>;
def StoreRetvalV2I32  : StoreRetvalVecInst<V2I32Regs, ".v2.b32",
  StoreRetvalScalar2I32>;
def StoreRetvalV2I16  : StoreRetvalVecInst<V2I16Regs, ".v2.b16",
  StoreRetvalScalar2I16>;
def StoreRetvalV2I8   : StoreRetvalVecInst<V2I8Regs,  ".v2.b8",
  StoreRetvalScalar2I8>;

def StoreRetvalV4F32  : StoreRetvalVecInst<V4F32Regs, ".v4.f32",
  StoreRetvalScalar4F32>;
def StoreRetvalV2F32  : StoreRetvalVecInst<V2F32Regs, ".v2.f32",
  StoreRetvalScalar2F32>;
def StoreRetvalV2F64  : StoreRetvalVecInst<V2F64Regs, ".v2.f64",
  StoreRetvalScalar2F64>;

}


// Int vector to int scalar bit convert
// v4i8 -> i32
def : Pat<(i32 (bitconvert V4I8Regs:$s)),
          (V4I8toI32 (V4i8Extract V4I8Regs:$s,0), (V4i8Extract V4I8Regs:$s,1),
                     (V4i8Extract V4I8Regs:$s,2), (V4i8Extract V4I8Regs:$s,3))>;
// v4i16 -> i64
def : Pat<(i64 (bitconvert V4I16Regs:$s)),
          (V4I16toI64 (V4i16Extract V4I16Regs:$s,0),
            (V4i16Extract V4I16Regs:$s,1),
                     (V4i16Extract V4I16Regs:$s,2),
                     (V4i16Extract V4I16Regs:$s,3))>;
// v2i8 -> i16
def : Pat<(i16 (bitconvert V2I8Regs:$s)),
          (V2I8toI16 (V2i8Extract V2I8Regs:$s,0), (V2i8Extract V2I8Regs:$s,1))>;
// v2i16 -> i32
def : Pat<(i32 (bitconvert V2I16Regs:$s)),
          (V2I16toI32 (V2i16Extract V2I16Regs:$s,0),
            (V2i16Extract V2I16Regs:$s,1))>;
// v2i32 -> i64
def : Pat<(i64 (bitconvert V2I32Regs:$s)),
          (V2I32toI64 (V2i32Extract V2I32Regs:$s,0),
            (V2i32Extract V2I32Regs:$s,1))>;

// Int scalar to int vector bit convert
let VecInstType=isVecDest.Value in {
// i32 -> v4i8
def VecI32toV4I8 : NVPTXVecInst<(outs V4I8Regs:$d), (ins Int32Regs:$s),
                                "Error!",
                                [(set V4I8Regs:$d, (bitconvert Int32Regs:$s))],
                                I32toV4I8>;
// i64 -> v4i16
def VecI64toV4I16 : NVPTXVecInst<(outs V4I16Regs:$d), (ins Int64Regs:$s),
                                 "Error!",
                                [(set V4I16Regs:$d, (bitconvert Int64Regs:$s))],
                                 I64toV4I16>;
// i16 -> v2i8
def VecI16toV2I8 : NVPTXVecInst<(outs V2I8Regs:$d), (ins Int16Regs:$s),
                                "Error!",
                               [(set V2I8Regs:$d, (bitconvert Int16Regs:$s))],
                                I16toV2I8>;
// i32 -> v2i16
def VecI32toV2I16 : NVPTXVecInst<(outs V2I16Regs:$d), (ins Int32Regs:$s),
                                 "Error!",
                                [(set V2I16Regs:$d, (bitconvert Int32Regs:$s))],
                                 I32toV2I16>;
// i64 -> v2i32
def VecI64toV2I32 : NVPTXVecInst<(outs V2I32Regs:$d), (ins Int64Regs:$s),
                                  "Error!",
                                [(set V2I32Regs:$d, (bitconvert Int64Regs:$s))],
                                  I64toV2I32>;
}

// Int vector to int vector bit convert
// v4i8 -> v2i16
def : Pat<(v2i16 (bitconvert V4I8Regs:$s)),
          (VecI32toV2I16
          (V4I8toI32 (V4i8Extract V4I8Regs:$s,0), (V4i8Extract V4I8Regs:$s,1),
                    (V4i8Extract V4I8Regs:$s,2), (V4i8Extract V4I8Regs:$s,3)))>;
// v4i16 -> v2i32
def : Pat<(v2i32 (bitconvert V4I16Regs:$s)),
          (VecI64toV2I32
       (V4I16toI64 (V4i16Extract V4I16Regs:$s,0), (V4i16Extract V4I16Regs:$s,1),
                (V4i16Extract V4I16Regs:$s,2), (V4i16Extract V4I16Regs:$s,3)))>;
// v2i16 -> v4i8
def : Pat<(v4i8 (bitconvert V2I16Regs:$s)),
          (VecI32toV4I8
    (V2I16toI32 (V2i16Extract V2I16Regs:$s,0), (V2i16Extract V2I16Regs:$s,1)))>;
// v2i32 -> v4i16
def : Pat<(v4i16 (bitconvert V2I32Regs:$s)),
          (VecI64toV4I16
    (V2I32toI64 (V2i32Extract V2I32Regs:$s,0), (V2i32Extract V2I32Regs:$s,1)))>;
// v2i64 -> v4i32
def : Pat<(v4i32 (bitconvert V2I64Regs:$s)),
          (Build_Vector4_i32
            (V2i32Extract (VecI64toV2I32 (V2i64Extract V2I64Regs:$s, 0)), 0),
            (V2i32Extract (VecI64toV2I32 (V2i64Extract V2I64Regs:$s, 0)), 1),
            (V2i32Extract (VecI64toV2I32 (V2i64Extract V2I64Regs:$s, 1)), 0),
            (V2i32Extract (VecI64toV2I32 (V2i64Extract V2I64Regs:$s, 1)), 1))>;
// v4i32 -> v2i64
def : Pat<(v2i64 (bitconvert V4I32Regs:$s)),
          (Build_Vector2_i64
      (V2I32toI64 (V4i32Extract V4I32Regs:$s,0), (V4i32Extract V4I32Regs:$s,1)),
    (V2I32toI64 (V4i32Extract V4I32Regs:$s,2), (V4i32Extract V4I32Regs:$s,3)))>;

// Fp scalar to fp vector convert
// f64 -> v2f32
let VecInstType=isVecDest.Value in {
def VecF64toV2F32 : NVPTXVecInst<(outs V2F32Regs:$d), (ins Float64Regs:$s),
                                  "Error!",
                              [(set V2F32Regs:$d, (bitconvert Float64Regs:$s))],
                                  F64toV2F32>;
}

// Fp vector to fp scalar convert
// v2f32 -> f64
def : Pat<(f64 (bitconvert V2F32Regs:$s)),
     (V2F32toF64 (V2f32Extract V2F32Regs:$s,0), (V2f32Extract V2F32Regs:$s,1))>;

// Fp scalar to int vector convert
// f32 -> v4i8
def : Pat<(v4i8 (bitconvert Float32Regs:$s)),
          (VecI32toV4I8 (BITCONVERT_32_F2I Float32Regs:$s))>;
// f32 -> v2i16
def : Pat<(v2i16 (bitconvert Float32Regs:$s)),
          (VecI32toV2I16 (BITCONVERT_32_F2I Float32Regs:$s))>;
// f64 -> v4i16
def : Pat<(v4i16 (bitconvert Float64Regs:$s)),
          (VecI64toV4I16 (BITCONVERT_64_F2I Float64Regs:$s))>;
// f64 -> v2i32
def : Pat<(v2i32 (bitconvert Float64Regs:$s)),
          (VecI64toV2I32 (BITCONVERT_64_F2I Float64Regs:$s))>;

// Int vector to fp scalar convert
// v4i8 -> f32
def : Pat<(f32 (bitconvert V4I8Regs:$s)),
          (BITCONVERT_32_I2F
          (V4I8toI32 (V4i8Extract V4I8Regs:$s,0), (V4i8Extract V4I8Regs:$s,1),
                    (V4i8Extract V4I8Regs:$s,2), (V4i8Extract V4I8Regs:$s,3)))>;
// v4i16 -> f64
def : Pat<(f64 (bitconvert V4I16Regs:$s)),
          (BITCONVERT_64_I2F
       (V4I16toI64 (V4i16Extract V4I16Regs:$s,0), (V4i16Extract V4I16Regs:$s,1),
                (V4i16Extract V4I16Regs:$s,2), (V4i16Extract V4I16Regs:$s,3)))>;
// v2i16 -> f32
def : Pat<(f32 (bitconvert V2I16Regs:$s)),
          (BITCONVERT_32_I2F
    (V2I16toI32 (V2i16Extract V2I16Regs:$s,0), (V2i16Extract V2I16Regs:$s,1)))>;
// v2i32 -> f64
def : Pat<(f64 (bitconvert V2I32Regs:$s)),
          (BITCONVERT_64_I2F
    (V2I32toI64 (V2i32Extract V2I32Regs:$s,0), (V2i32Extract V2I32Regs:$s,1)))>;

// Int scalar to fp vector convert
// i64 -> v2f32
def : Pat<(v2f32 (bitconvert Int64Regs:$s)),
          (VecF64toV2F32 (BITCONVERT_64_I2F Int64Regs:$s))>;

// Fp vector to int scalar convert
// v2f32 -> i64
def : Pat<(i64 (bitconvert V2F32Regs:$s)),
          (BITCONVERT_64_F2I
    (V2F32toF64 (V2f32Extract V2F32Regs:$s,0), (V2f32Extract V2F32Regs:$s,1)))>;

// Int vector to fp vector convert
// v2i64 -> v4f32
def : Pat<(v4f32 (bitconvert V2I64Regs:$s)),
          (Build_Vector4_f32
            (BITCONVERT_32_I2F (V2i32Extract (VecI64toV2I32
              (V2i64Extract V2I64Regs:$s, 0)), 0)),
            (BITCONVERT_32_I2F (V2i32Extract (VecI64toV2I32
              (V2i64Extract V2I64Regs:$s, 0)), 1)),
            (BITCONVERT_32_I2F (V2i32Extract (VecI64toV2I32
              (V2i64Extract V2I64Regs:$s, 1)), 0)),
            (BITCONVERT_32_I2F (V2i32Extract (VecI64toV2I32
              (V2i64Extract V2I64Regs:$s, 1)), 1)))>;
// v2i64 -> v2f64
def : Pat<(v2f64 (bitconvert V2I64Regs:$s)),
    (Build_Vector2_f64
            (BITCONVERT_64_I2F (V2i64Extract V2I64Regs:$s,0)),
            (BITCONVERT_64_I2F (V2i64Extract V2I64Regs:$s,1)))>;
// v2i32 -> v2f32
def : Pat<(v2f32 (bitconvert V2I32Regs:$s)),
    (Build_Vector2_f32
            (BITCONVERT_32_I2F (V2i32Extract V2I32Regs:$s,0)),
            (BITCONVERT_32_I2F (V2i32Extract V2I32Regs:$s,1)))>;
// v4i32 -> v2f64
def : Pat<(v2f64 (bitconvert V4I32Regs:$s)),
          (Build_Vector2_f64
           (BITCONVERT_64_I2F (V2I32toI64 (V4i32Extract V4I32Regs:$s,0),
             (V4i32Extract V4I32Regs:$s,1))),
           (BITCONVERT_64_I2F (V2I32toI64 (V4i32Extract V4I32Regs:$s,2),
             (V4i32Extract V4I32Regs:$s,3))))>;
// v4i32 -> v4f32
def : Pat<(v4f32 (bitconvert V4I32Regs:$s)),
    (Build_Vector4_f32
            (BITCONVERT_32_I2F (V4i32Extract V4I32Regs:$s,0)),
            (BITCONVERT_32_I2F (V4i32Extract V4I32Regs:$s,1)),
            (BITCONVERT_32_I2F (V4i32Extract V4I32Regs:$s,2)),
            (BITCONVERT_32_I2F (V4i32Extract V4I32Regs:$s,3)))>;
// v4i16 -> v2f32
def : Pat<(v2f32 (bitconvert V4I16Regs:$s)),
          (VecF64toV2F32 (BITCONVERT_64_I2F
          (V4I16toI64 (V4i16Extract V4I16Regs:$s,0),
            (V4i16Extract V4I16Regs:$s,1),
                      (V4i16Extract V4I16Regs:$s,2),
                      (V4i16Extract V4I16Regs:$s,3))))>;

// Fp vector to int vector convert
// v2i64 <- v4f32
def : Pat<(v2i64 (bitconvert V4F32Regs:$s)),
          (Build_Vector2_i64
           (BITCONVERT_64_F2I (V2F32toF64 (V4f32Extract V4F32Regs:$s,0),
             (V4f32Extract V4F32Regs:$s,1))),
           (BITCONVERT_64_F2I (V2F32toF64 (V4f32Extract V4F32Regs:$s,2),
             (V4f32Extract V4F32Regs:$s,3))))>;
// v2i64 <- v2f64
def : Pat<(v2i64 (bitconvert V2F64Regs:$s)),
    (Build_Vector2_i64
            (BITCONVERT_64_F2I (V2f64Extract V2F64Regs:$s,0)),
            (BITCONVERT_64_F2I (V2f64Extract V2F64Regs:$s,1)))>;
// v2i32 <- v2f32
def : Pat<(v2i32 (bitconvert V2F32Regs:$s)),
    (Build_Vector2_i32
            (BITCONVERT_32_F2I (V2f32Extract V2F32Regs:$s,0)),
            (BITCONVERT_32_F2I (V2f32Extract V2F32Regs:$s,1)))>;
// v4i32 <- v2f64
def : Pat<(v4i32 (bitconvert V2F64Regs:$s)),
          (Build_Vector4_i32
            (BITCONVERT_32_F2I (V2f32Extract (VecF64toV2F32
              (V2f64Extract V2F64Regs:$s, 0)), 0)),
            (BITCONVERT_32_F2I (V2f32Extract (VecF64toV2F32
              (V2f64Extract V2F64Regs:$s, 0)), 1)),
            (BITCONVERT_32_F2I (V2f32Extract (VecF64toV2F32
              (V2f64Extract V2F64Regs:$s, 1)), 0)),
            (BITCONVERT_32_F2I (V2f32Extract (VecF64toV2F32
              (V2f64Extract V2F64Regs:$s, 1)), 1)))>;
// v4i32 <- v4f32
def : Pat<(v4i32 (bitconvert V4F32Regs:$s)),
          (Build_Vector4_i32
            (BITCONVERT_32_F2I (V4f32Extract V4F32Regs:$s,0)),
            (BITCONVERT_32_F2I (V4f32Extract V4F32Regs:$s,1)),
            (BITCONVERT_32_F2I (V4f32Extract V4F32Regs:$s,2)),
            (BITCONVERT_32_F2I (V4f32Extract V4F32Regs:$s,3)))>;
// v4i16 <- v2f32
def : Pat<(v4i16 (bitconvert V2F32Regs:$s)),
          (VecI64toV4I16 (BITCONVERT_64_F2I
          (V2F32toF64 (V2f32Extract V2F32Regs:$s,0),
            (V2f32Extract V2F32Regs:$s,1))))>;
