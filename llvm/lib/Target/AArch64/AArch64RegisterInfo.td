//===- AArch64RegisterInfo.td - ARM Register defs ----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file contains declarations that describe the AArch64 register file
//
//===----------------------------------------------------------------------===//

let Namespace = "AArch64" in {
def sub_128 : SubRegIndex<128>;
def sub_64 : SubRegIndex<64>;
def sub_32 : SubRegIndex<32>;
def sub_16 : SubRegIndex<16>;
def sub_8  : SubRegIndex<8>;

// Note: Code depends on these having consecutive numbers.
def qqsub : SubRegIndex<256, 256>;

def qsub_0 : SubRegIndex<128>;
def qsub_1 : SubRegIndex<128, 128>;
def qsub_2 : ComposedSubRegIndex<qqsub, qsub_0>;
def qsub_3 : ComposedSubRegIndex<qqsub, qsub_1>;

def dsub_0 : SubRegIndex<64>;
def dsub_1 : SubRegIndex<64, 64>;
def dsub_2 : ComposedSubRegIndex<qsub_1, dsub_0>;
def dsub_3 : ComposedSubRegIndex<qsub_1, dsub_1>;
def dsub_4 : ComposedSubRegIndex<qsub_2, dsub_0>;
}

// Registers are identified with 5-bit ID numbers.
class AArch64Reg<bits<16> enc, string n> : Register<n> {
  let HWEncoding = enc;
  let Namespace = "AArch64";
}

class AArch64RegWithSubs<bits<16> enc, string n, list<Register> subregs = [],
                         list<SubRegIndex> inds = []>
      : AArch64Reg<enc, n> {
  let SubRegs = subregs;
  let SubRegIndices = inds;
}

//===----------------------------------------------------------------------===//
//  Integer registers: w0-w30, wzr, wsp, x0-x30, xzr, sp
//===----------------------------------------------------------------------===//

foreach Index = 0-30 in {
  def W#Index : AArch64Reg< Index, "w"#Index>, DwarfRegNum<[Index]>;
}

def WSP : AArch64Reg<31, "wsp">, DwarfRegNum<[31]>;
def WZR : AArch64Reg<31, "wzr">;

// Could be combined with previous loop, but this way leaves w and x registers
// consecutive as LLVM register numbers, which makes for easier debugging.
foreach Index = 0-30 in {
  def X#Index : AArch64RegWithSubs<Index, "x"#Index,
                                   [!cast<Register>("W"#Index)], [sub_32]>,
                DwarfRegNum<[Index]>;
}

def XSP : AArch64RegWithSubs<31, "sp", [WSP], [sub_32]>, DwarfRegNum<[31]>;
def XZR : AArch64RegWithSubs<31, "xzr", [WZR], [sub_32]>;

// Most instructions treat register 31 as zero for reads and a black-hole for
// writes.

// Note that the order of registers is important for the Disassembler here:
// tablegen uses it to form MCRegisterClass::getRegister, which we assume can
// take an encoding value.
def GPR32 : RegisterClass<"AArch64", [i32], 32,
                          (add (sequence "W%u", 0, 30), WZR)> {
}

def GPR64 : RegisterClass<"AArch64", [i64], 64,
                          (add (sequence "X%u", 0, 30), XZR)> {
}

def GPR32nowzr : RegisterClass<"AArch64", [i32], 32,
                               (sequence "W%u", 0, 30)> {
}

def GPR64noxzr : RegisterClass<"AArch64", [i64], 64,
                               (sequence "X%u", 0, 30)> {
}

// For tail calls, we can't use callee-saved registers or the structure-return
// register, as they are supposed to be live across function calls and may be
// clobbered by the epilogue.
def tcGPR64 : RegisterClass<"AArch64", [i64], 64,
                            (add (sequence "X%u", 0, 7),
                                 (sequence "X%u", 9, 18))> {
}


// Certain addressing-useful instructions accept sp directly. Again the order of
// registers is important to the Disassembler.
def GPR32wsp : RegisterClass<"AArch64", [i32], 32,
                             (add (sequence "W%u", 0, 30), WSP)> {
}

def GPR64xsp : RegisterClass<"AArch64", [i64], 64,
                             (add (sequence "X%u", 0, 30), XSP)> {
}

// Some aliases *only* apply to SP (e.g. MOV uses different encoding for SP and
// non-SP variants). We can't use a bare register in those patterns because
// TableGen doesn't like it, so we need a class containing just stack registers
def Rxsp : RegisterClass<"AArch64", [i64], 64,
                         (add XSP)> {
}

def Rwsp : RegisterClass<"AArch64", [i32], 32,
                         (add WSP)> {
}

//===----------------------------------------------------------------------===//
//  Scalar registers in the vector unit:
//  b0-b31, h0-h31, s0-s31, d0-d31, q0-q31
//===----------------------------------------------------------------------===//

foreach Index = 0-31 in {
  def B # Index : AArch64Reg< Index, "b" # Index>,
                  DwarfRegNum<[!add(Index, 64)]>;

  def H # Index : AArch64RegWithSubs<Index, "h" # Index,
                                     [!cast<Register>("B" # Index)], [sub_8]>,
                  DwarfRegNum<[!add(Index, 64)]>;

  def S # Index : AArch64RegWithSubs<Index, "s" # Index,
                                     [!cast<Register>("H" # Index)], [sub_16]>,
                  DwarfRegNum<[!add(Index, 64)]>;

  def D # Index : AArch64RegWithSubs<Index, "d" # Index,
                                     [!cast<Register>("S" # Index)], [sub_32]>,
                  DwarfRegNum<[!add(Index, 64)]>;

  def Q # Index : AArch64RegWithSubs<Index, "q" # Index,
                                     [!cast<Register>("D" # Index)], [sub_64]>,
                  DwarfRegNum<[!add(Index, 64)]>;
}


def FPR8 : RegisterClass<"AArch64", [i8, v1i8], 8,
                          (sequence "B%u", 0, 31)> {
}

def FPR16 : RegisterClass<"AArch64", [f16, v1i16], 16,
                          (sequence "H%u", 0, 31)> {
}

def FPR32 : RegisterClass<"AArch64", [f32, v1i32, v1f32], 32,
                          (sequence "S%u", 0, 31)> {
}

def FPR64 : RegisterClass<"AArch64",
                          [f64, v2f32, v2i32, v4i16, v8i8, v1i64, v1f64],
                          64, (sequence "D%u", 0, 31)>;

def FPR128 : RegisterClass<"AArch64",
                           [f128, v2f64, v2i64, v4f32, v4i32, v8i16, v16i8],
                           128, (sequence "Q%u", 0, 31)>;

def FPR64Lo : RegisterClass<"AArch64",
                            [f64, v2f32, v2i32, v4i16, v8i8, v1i64, v1f64],
                            64, (sequence "D%u", 0, 15)>;

def FPR128Lo : RegisterClass<"AArch64",
                             [f128, v2f64, v2i64, v4f32, v4i32, v8i16, v16i8],
                             128, (sequence "Q%u", 0, 15)>;

//===----------------------------------------------------------------------===//
//  Vector registers:
//===----------------------------------------------------------------------===//

def VPR64AsmOperand : AsmOperandClass {
  let Name = "VPR";
  let PredicateMethod = "isReg";
  let RenderMethod = "addRegOperands";
}

def VPR64 : RegisterOperand<FPR64, "printVPRRegister">;

def VPR128 : RegisterOperand<FPR128, "printVPRRegister">;

def VPR64Lo : RegisterOperand<FPR64Lo, "printVPRRegister">;

def VPR128Lo : RegisterOperand<FPR128Lo, "printVPRRegister">;

// Flags register
def NZCV : Register<"nzcv"> {
  let Namespace = "AArch64";
}

def FlagClass : RegisterClass<"AArch64", [i32], 32, (add NZCV)> {
  let CopyCost = -1;
  let isAllocatable = 0;
}

//===----------------------------------------------------------------------===//
//  Consecutive vector registers
//===----------------------------------------------------------------------===//
// 2 Consecutive 64-bit registers: D0_D1, D1_D2, ..., D30_D31
def Tuples2D : RegisterTuples<[dsub_0, dsub_1],
                              [(rotl FPR64, 0), (rotl FPR64, 1)]>;
                              
// 3 Consecutive 64-bit registers: D0_D1_D2, ..., D31_D0_D1
def Tuples3D : RegisterTuples<[dsub_0, dsub_1, dsub_2],
                              [(rotl FPR64, 0), (rotl FPR64, 1),
                               (rotl FPR64, 2)]>;
                               
// 4 Consecutive 64-bit registers: D0_D1_D2_D3, ..., D31_D0_D1_D2
def Tuples4D : RegisterTuples<[dsub_0, dsub_1, dsub_2, dsub_3],
                              [(rotl FPR64, 0), (rotl FPR64, 1),
                               (rotl FPR64, 2), (rotl FPR64, 3)]>;

// 2 Consecutive 128-bit registers: Q0_Q1, Q1_Q2, ..., Q30_Q31
def Tuples2Q : RegisterTuples<[qsub_0, qsub_1],
                              [(rotl FPR128, 0), (rotl FPR128, 1)]>;

// 3 Consecutive 128-bit registers: Q0_Q1_Q2, ..., Q31_Q0_Q1
def Tuples3Q : RegisterTuples<[qsub_0, qsub_1, qsub_2],
                              [(rotl FPR128, 0), (rotl FPR128, 1),
                               (rotl FPR128, 2)]>;
                               
// 4 Consecutive 128-bit registers: Q0_Q1_Q2_Q3, ..., Q31_Q0_Q1_Q2
def Tuples4Q : RegisterTuples<[qsub_0, qsub_1, qsub_2, qsub_3],
                              [(rotl FPR128, 0), (rotl FPR128, 1),
                               (rotl FPR128, 2), (rotl FPR128, 3)]>;

// The followings are super register classes to model 2/3/4 consecutive
// 64-bit/128-bit registers.

def DPair : RegisterClass<"AArch64", [v2i64], 64, (add Tuples2D)>;

def DTriple : RegisterClass<"AArch64", [untyped], 64, (add Tuples3D)> {
  let Size = 192; // 3 x 64 bits, we have no predefined type of that size.
}

def DQuad : RegisterClass<"AArch64", [v4i64], 64, (add Tuples4D)>;

def QPair : RegisterClass<"AArch64", [v4i64], 128, (add Tuples2Q)>;

def QTriple : RegisterClass<"AArch64", [untyped], 128, (add Tuples3Q)> {
  let Size = 384; // 3 x 128 bits, we have no predefined type of that size.
}

def QQuad : RegisterClass<"AArch64", [v8i64], 128, (add Tuples4Q)>;


// The followings are vector list operands
multiclass VectorList_operands<string PREFIX, string LAYOUT, int Count,
                               RegisterClass RegList> {
  def _asmoperand : AsmOperandClass {
    let Name = PREFIX # LAYOUT # Count;
    let RenderMethod = "addVectorListOperands";
    let PredicateMethod = 
        "isVectorList<A64Layout::VL_" # LAYOUT # ", " # Count # ">";
    let ParserMethod = "ParseVectorList";
  }

  def _operand : RegisterOperand<RegList,
        "printVectorList<A64Layout::VL_" # LAYOUT # ", " # Count # ">"> {
    let ParserMatchClass =
      !cast<AsmOperandClass>(PREFIX # LAYOUT # "_asmoperand");
  }
}

multiclass VectorList_BHSD<string PREFIX, int Count, RegisterClass DRegList,
                           RegisterClass QRegList> {
  defm 8B : VectorList_operands<PREFIX, "8B", Count, DRegList>;
  defm 4H : VectorList_operands<PREFIX, "4H", Count, DRegList>;
  defm 2S : VectorList_operands<PREFIX, "2S", Count, DRegList>;
  defm 1D : VectorList_operands<PREFIX, "1D", Count, DRegList>;
  defm 16B : VectorList_operands<PREFIX, "16B", Count, QRegList>;
  defm 8H : VectorList_operands<PREFIX, "8H", Count, QRegList>;
  defm 4S : VectorList_operands<PREFIX, "4S", Count, QRegList>;
  defm 2D : VectorList_operands<PREFIX, "2D", Count, QRegList>;
}

// Vector list operand with 1/2/3/4 registers: VOne8B_operand,..., VQuad2D_operand
defm VOne : VectorList_BHSD<"VOne", 1, FPR64, FPR128>;
defm VPair : VectorList_BHSD<"VPair", 2, DPair, QPair>;
defm VTriple : VectorList_BHSD<"VTriple", 3, DTriple, QTriple>;
defm VQuad : VectorList_BHSD<"VQuad", 4, DQuad, QQuad>;