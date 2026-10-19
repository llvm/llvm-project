//===-- G13Instructions.td - AGX2 encoding family ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Real instructions of the G13/G14 encoding family. Bit numbers count from
// bit 0 of the first byte; instructions are little endian. The layout of each
// instruction follows dougallj/applegpu and the Mesa asahi packer.
//
// In ALU instructions bits 6:0 hold the opcode (bit 6 is the saturate flag
// where saturation exists), bits 14:7 the low destination bits and bit 15 the
// length bit. Source N starts at bit 16 + 12 * N. The two high bits of each
// register field are stored in the last 16 bits of the long form, in reverse
// order: destination at bits 12-13 of that word, source N at 10 - 2 * N.
//
//===----------------------------------------------------------------------===//

class G13ALU<string mn, dag outs, dag ins, string ops, int short,
             int long = short>
    : AGXReal<mn, outs, ins, ops, short, long> {
  let IsALU = 1;
}

//===----------------------------------------------------------------------===//
// Moves
//===----------------------------------------------------------------------===//

def MOV_IMM16_g13 : G13ALU<"mov_imm", (outs AGXDst16:$dst), (ins AGXImm:$imm),
                           "$dst, $imm", 4, 6> {
  bits<10> dst;
  bits<16> imm;
  let Inst{6-0} = 0b1100010;
  let Inst{7} = dst{0};
  let Inst{8} = 0;
  let Inst{14-9} = dst{7-2};
  let Inst{31-16} = imm;
  let Inst{45-44} = dst{9-8};
}

def MOV_IMM32_g13 : G13ALU<"mov_imm", (outs AGXDst32:$dst), (ins AGXImm:$imm),
                           "$dst, $imm", 6, 8> {
  bits<10> dst;
  bits<32> imm;
  let Inst{6-0} = 0b1100010;
  let Inst{7} = dst{0};
  let Inst{8} = 1;
  let Inst{14-9} = dst{7-2};
  let Inst{47-16} = imm;
  let Inst{61-60} = dst{9-8};
  let Inst{62} = 0;
}

def GET_SR_g13 : G13ALU<"get_sr", (outs AGXDst:$dst), (ins AGXSpecialReg:$sr),
                        "$dst, $sr", 4> {
  bits<10> dst;
  bits<8> sr;
  let Inst{6-0} = 0b1110010;
  let Inst{14-7} = dst{7-0};
  let Inst{15} = 0;
  let Inst{21-16} = sr{5-0};
  let Inst{27-26} = sr{7-6};
  let Inst{29-28} = dst{9-8};
}

//===----------------------------------------------------------------------===//
// Integer arithmetic
//===----------------------------------------------------------------------===//

class G13IAdd<string mn, bit neg, bit sat>
    : G13ALU<mn#!if(sat, ".sat", ""), (outs AGXDst64:$dst),
             (ins AGXSrcSX64:$src0, AGXSrcSX64:$src1, AGXShift:$shift),
             "$dst, $src0, $src1$shift", 8> {
  bits<10> dst;
  bits<13> src0;
  bits<13> src1;
  bits<3> shift;
  let Inst{5-0} = 0b001110;
  let Inst{6} = sat;
  let Inst{14-7} = dst{7-0};
  let Inst{15} = 0;
  let Inst{26-16} = src0{10-0};
  let Inst{27} = neg;
  let Inst{38-28} = src1{10-0};
  let Inst{39} = shift{0};
  let Inst{41-40} = src1{12-11};
  let Inst{43-42} = src0{12-11};
  let Inst{45-44} = dst{9-8};
  let Inst{53-52} = shift{2-1};
}

def IADD_g13 : G13IAdd<"iadd", 0, 0>;
def IADD_SAT_g13 : G13IAdd<"iadd", 0, 1>;
def ISUB_g13 : G13IAdd<"isub", 1, 0>;
def ISUB_SAT_g13 : G13IAdd<"isub", 1, 1>;

class G13IMAdd<string mn, bit neg, bit sat>
    : G13ALU<mn#!if(sat, ".sat", ""), (outs AGXDst64:$dst),
             (ins AGXSrcSX:$src0, AGXSrcSX:$src1, AGXSrcSX64:$src2,
                  AGXShift:$shift),
             "$dst, $src0, $src1, $src2$shift", 8> {
  bits<10> dst;
  bits<13> src0;
  bits<13> src1;
  bits<13> src2;
  bits<3> shift;
  let Inst{5-0} = 0b011110;
  let Inst{6} = sat;
  let Inst{14-7} = dst{7-0};
  let Inst{15} = 0;
  let Inst{26-16} = src0{10-0};
  let Inst{27} = neg;
  let Inst{38-28} = src1{10-0};
  let Inst{39} = shift{0};
  let Inst{50-40} = src2{10-0};
  let Inst{53-52} = shift{2-1};
  let Inst{55-54} = src2{12-11};
  let Inst{57-56} = src1{12-11};
  let Inst{59-58} = src0{12-11};
  let Inst{61-60} = dst{9-8};
}

def IMADD_g13 : G13IMAdd<"imadd", 0, 0>;
def IMADD_SAT_g13 : G13IMAdd<"imadd", 0, 1>;
def IMSUB_g13 : G13IMAdd<"imsub", 1, 0>;
def IMSUB_SAT_g13 : G13IMAdd<"imsub", 1, 1>;

//===----------------------------------------------------------------------===//
// Shifts and bit fields
//===----------------------------------------------------------------------===//

class G13Bitfield<string mn, bit sel, bits<2> sub>
    : G13ALU<mn, (outs AGXDst:$dst),
             (ins AGXSrc:$src0, AGXSrc:$src1, AGXSrc:$src2,
                  AGXBitfieldMask:$mask),
             "$dst, $src0, $src1, $src2$mask", 8> {
  bits<10> dst;
  bits<12> src0;
  bits<12> src1;
  bits<12> src2;
  bits<5> mask;
  let Inst{6-0} = 0b0101110;
  let Inst{14-7} = dst{7-0};
  let Inst{15} = sel;
  let Inst{25-16} = src0{9-0};
  let Inst{27-26} = sub;
  let Inst{37-28} = src1{9-0};
  let Inst{39-38} = mask{1-0};
  let Inst{49-40} = src2{9-0};
  let Inst{51-50} = mask{3-2};
  let Inst{55-54} = src2{11-10};
  let Inst{57-56} = src1{11-10};
  let Inst{59-58} = src0{11-10};
  let Inst{61-60} = dst{9-8};
  let Inst{63} = mask{4};
}

def BFI_g13 : G13Bitfield<"bfi", 0, 0>;
def BFEIL_g13 : G13Bitfield<"bfeil", 1, 0>;
def EXTR_g13 : G13Bitfield<"extr", 0, 1>;
def SHLHI_g13 : G13Bitfield<"shlhi", 0, 2>;
def SHRHI_g13 : G13Bitfield<"shrhi", 1, 2>;

class G13Shift<string mn, bits<2> sub>
    : G13ALU<mn, (outs AGXDst:$dst), (ins AGXSrc:$src0, AGXSrc:$src1),
             "$dst, $src0, $src1", 8> {
  bits<10> dst;
  bits<12> src0;
  bits<12> src1;
  let Inst{6-0} = 0b0101110;
  let Inst{14-7} = dst{7-0};
  let Inst{15} = 1;
  let Inst{25-16} = src0{9-0};
  let Inst{27-26} = sub;
  let Inst{37-28} = src1{9-0};
  let Inst{57-56} = src1{11-10};
  let Inst{59-58} = src0{11-10};
  let Inst{61-60} = dst{9-8};
}

def ASR_g13 : G13Shift<"asr", 1>;
def ASRH_g13 : G13Shift<"asrh", 3>;

//===----------------------------------------------------------------------===//
// Bitwise operations
//===----------------------------------------------------------------------===//

// D = truth_table(A, B). The truth table bit i gives the result for the
// operand bits (a, b) = (i & 1, i >> 1).
def BITOP_g13 : G13ALU<"bitop", (outs AGXDst:$dst),
                       (ins AGXTruthTable:$tt, AGXSrc:$src0, AGXSrc:$src1),
                       "$tt, $dst, $src0, $src1", 6> {
  bits<10> dst;
  bits<4> tt;
  bits<12> src0;
  bits<12> src1;
  let Inst{6-0} = 0b1111110;
  let Inst{14-7} = dst{7-0};
  let Inst{15} = 0;
  let Inst{25-16} = src0{9-0};
  let Inst{27-26} = tt{1-0};
  let Inst{37-28} = src1{9-0};
  let Inst{39-38} = tt{3-2};
  let Inst{41-40} = src1{11-10};
  let Inst{43-42} = src0{11-10};
  let Inst{45-44} = dst{9-8};
}

// Truth table aliases, in the applegpu spelling.
class AGXBitopAlias<string mn, int tt> {
  string Mnemonic = mn;
  int TT = tt;
}

foreach Alias = [AGXBitopAlias<"and", 0b1000>, AGXBitopAlias<"or", 0b1110>,
                 AGXBitopAlias<"xor", 0b0110>, AGXBitopAlias<"nand", 0b0111>,
                 AGXBitopAlias<"nor", 0b0001>, AGXBitopAlias<"xnor", 0b1001>] in
def : InstAlias<Alias.Mnemonic#" $dst, $src0, $src1",
                (BITOP_g13 AGXDst:$dst, Alias.TT, AGXSrc:$src0,
                           AGXSrc:$src1)>;

class G13IUnary<string mn, bits<2> op>
    : G13ALU<mn, (outs AGXDst:$dst), (ins AGXSrc:$src0), "$dst, $src0", 6> {
  bits<10> dst;
  bits<12> src0;
  let Inst{6-0} = 0b0111110;
  let Inst{14-7} = dst{7-0};
  let Inst{15} = 0;
  let Inst{25-16} = src0{9-0};
  let Inst{27-26} = op;
  let Inst{37-28} = 0;
  let Inst{39-38} = 0;
  let Inst{41-40} = 0;
  let Inst{43-42} = src0{11-10};
  let Inst{45-44} = dst{9-8};
}

def BITREV_g13 : G13IUnary<"bitrev", 1>;
def POPCOUNT_g13 : G13IUnary<"popcount", 2>;
def FFS_g13 : G13IUnary<"ffs", 3>;

def INTL_g13 : G13ALU<"intl", (outs AGXDst:$dst),
                      (ins AGXSrc:$src0, AGXSrc:$src1),
                      "$dst, $src0, $src1", 6> {
  bits<10> dst;
  bits<12> src0;
  bits<12> src1;
  let Inst{6-0} = 0b0111110;
  let Inst{14-7} = dst{7-0};
  let Inst{15} = 0;
  let Inst{25-16} = src0{9-0};
  let Inst{27-26} = 0;
  let Inst{37-28} = src1{9-0};
  let Inst{39-38} = 0;
  let Inst{41-40} = src1{11-10};
  let Inst{43-42} = src0{11-10};
  let Inst{45-44} = dst{9-8};
}

def CONVERT_g13 : G13ALU<"convert", (outs AGXDst:$dst),
                         (ins AGXConvertMode:$mode, AGXSrc:$src0,
                              AGXRoundMode:$round),
                         "$mode, $dst, $src0, $round", 6> {
  bits<10> dst;
  bits<6> mode;
  bits<12> src0;
  bits<2> round;
  let Inst{6-0} = 0b0111110;
  let Inst{14-7} = dst{7-0};
  let Inst{15} = 1;
  let Inst{21-16} = mode;
  let Inst{25-22} = 0;
  let Inst{27-26} = round;
  let Inst{37-28} = src0{9-0};
  let Inst{39-38} = 0;
  let Inst{41-40} = src0{11-10};
  let Inst{43-42} = 0;
  let Inst{45-44} = dst{9-8};
}

//===----------------------------------------------------------------------===//
// Floating point arithmetic
//===----------------------------------------------------------------------===//

class G13FBin32<string mn, bits<6> op, bit sat>
    : G13ALU<mn#!if(sat, ".sat", ""), (outs AGXDst:$dst),
             (ins AGXSrcF:$src0, AGXSrcF:$src1), "$dst, $src0, $src1", 4, 6> {
  bits<10> dst;
  bits<14> src0;
  bits<14> src1;
  let Inst{5-0} = op;
  let Inst{6} = sat;
  let Inst{14-7} = dst{7-0};
  let Inst{27-16} = src0{11-0};
  let Inst{39-28} = src1{11-0};
  let Inst{41-40} = src1{13-12};
  let Inst{43-42} = src0{13-12};
  let Inst{45-44} = dst{9-8};
}

class G13FBin16<string mn, bits<6> op, bit sat>
    : G13ALU<mn#!if(sat, ".sat", ""), (outs AGXDst16:$dst),
             (ins AGXSrcF16:$src0, AGXSrcF16:$src1), "$dst, $src0, $src1",
             4, 6> {
  bits<10> dst;
  bits<13> src0;
  bits<13> src1;
  let Inst{5-0} = op;
  let Inst{6} = sat;
  let Inst{7} = dst{0};
  let Inst{8} = 0;
  let Inst{14-9} = dst{7-2};
  let Inst{26-16} = src0{10-0};
  let Inst{38-28} = src1{10-0};
  let Inst{41-40} = src1{12-11};
  let Inst{43-42} = src0{12-11};
  let Inst{45-44} = dst{9-8};
}

foreach Sat = [0, 1] in {
  defvar S = !if(Sat, "_SAT", "");
  def FADD32#S#_g13 : G13FBin32<"fadd32", 0b101010, Sat>;
  def FMUL32#S#_g13 : G13FBin32<"fmul32", 0b011010, Sat>;
  def FADD16#S#_g13 : G13FBin16<"fadd16", 0b100110, Sat>;
  def FMUL16#S#_g13 : G13FBin16<"fmul16", 0b010110, Sat>;
}

class G13FMA32<bit sat>
    : G13ALU<"fmadd32"#!if(sat, ".sat", ""), (outs AGXDst:$dst),
             (ins AGXSrcF:$src0, AGXSrcF:$src1, AGXSrcF:$src2),
             "$dst, $src0, $src1, $src2", 6, 8> {
  bits<10> dst;
  bits<14> src0;
  bits<14> src1;
  bits<14> src2;
  let Inst{5-0} = 0b111010;
  let Inst{6} = sat;
  let Inst{14-7} = dst{7-0};
  let Inst{27-16} = src0{11-0};
  let Inst{39-28} = src1{11-0};
  let Inst{51-40} = src2{11-0};
  let Inst{55-54} = src2{13-12};
  let Inst{57-56} = src1{13-12};
  let Inst{59-58} = src0{13-12};
  let Inst{61-60} = dst{9-8};
}

class G13FMA16<bit sat>
    : G13ALU<"fmadd16"#!if(sat, ".sat", ""), (outs AGXDst16:$dst),
             (ins AGXSrcF16:$src0, AGXSrcF16:$src1, AGXSrcF16:$src2),
             "$dst, $src0, $src1, $src2", 6, 8> {
  bits<10> dst;
  bits<13> src0;
  bits<13> src1;
  bits<13> src2;
  let Inst{5-0} = 0b110110;
  let Inst{6} = sat;
  let Inst{7} = dst{0};
  let Inst{8} = 0;
  let Inst{14-9} = dst{7-2};
  let Inst{26-16} = src0{10-0};
  let Inst{38-28} = src1{10-0};
  let Inst{50-40} = src2{10-0};
  let Inst{55-54} = src2{12-11};
  let Inst{57-56} = src1{12-11};
  let Inst{59-58} = src0{12-11};
  let Inst{61-60} = dst{9-8};
}

def FMADD32_g13 : G13FMA32<0>;
def FMADD32_SAT_g13 : G13FMA32<1>;
def FMADD16_g13 : G13FMA16<0>;
def FMADD16_SAT_g13 : G13FMA16<1>;

class G13FUnary<string mn, bits<6> op, bit sat>
    : G13ALU<mn#!if(sat, ".sat", ""), (outs AGXDst:$dst), (ins AGXSrcF:$src0),
             "$dst, $src0", 4, 6> {
  bits<10> dst;
  bits<14> src0;
  let Inst{5-0} = 0b001010;
  let Inst{6} = sat;
  let Inst{14-7} = dst{7-0};
  let Inst{27-16} = src0{11-0};
  let Inst{33-28} = op;
  let Inst{41-34} = 0;
  let Inst{43-42} = src0{13-12};
  let Inst{45-44} = dst{9-8};
}

foreach Sat = [0, 1] in {
  defvar S = !if(Sat, "_SAT", "");
  def FLOOR#S#_g13 : G13FUnary<"floor", 0b000000, Sat>;
  def RSQRT_SPECIAL#S#_g13 : G13FUnary<"rsqrt_special", 0b000001, Sat>;
  def DFDX#S#_g13 : G13FUnary<"dfdx", 0b000100, Sat>;
  def DFDY#S#_g13 : G13FUnary<"dfdy", 0b000110, Sat>;
  def RCP#S#_g13 : G13FUnary<"rcp", 0b001000, Sat>;
  def RSQRT#S#_g13 : G13FUnary<"rsqrt", 0b001001, Sat>;
  def SIN_PT_1#S#_g13 : G13FUnary<"sin_pt_1", 0b001010, Sat>;
  def LOG2#S#_g13 : G13FUnary<"log2", 0b001100, Sat>;
  def EXP2#S#_g13 : G13FUnary<"exp2", 0b001101, Sat>;
  def SIN_PT_2#S#_g13 : G13FUnary<"sin_pt_2", 0b001110, Sat>;
  def CEIL#S#_g13 : G13FUnary<"ceil", 0b010000, Sat>;
  def TRUNC#S#_g13 : G13FUnary<"trunc", 0b100000, Sat>;
  def RINT#S#_g13 : G13FUnary<"rint", 0b110000, Sat>;
}

//===----------------------------------------------------------------------===//
// Compare and select
//===----------------------------------------------------------------------===//

def ICMPSEL_g13 : G13ALU<"icmpsel", (outs AGXDst:$dst),
                         (ins AGXICond3:$cc, AGXSrc:$src0, AGXSrc:$src1,
                              AGXCmpselSrc:$src2, AGXCmpselSrc:$src3),
                         "$cc, $dst, $src0, $src1, $src2, $src3", 8, 10> {
  bits<10> dst;
  bits<3> cc;
  bits<12> src0;
  bits<12> src1;
  bits<11> src2;
  bits<11> src3;
  let Inst{6-0} = 0b0010010;
  let Inst{14-7} = dst{7-0};
  let Inst{25-16} = src0{9-0};
  let Inst{37-28} = src1{9-0};
  let Inst{48-40} = src2{8-0};
  let Inst{60-52} = src3{8-0};
  let Inst{63-61} = cc;
  let Inst{69-68} = src3{10-9};
  let Inst{71-70} = src2{10-9};
  let Inst{73-72} = src1{11-10};
  let Inst{75-74} = src0{11-10};
  let Inst{77-76} = dst{9-8};
}

def FCMPSEL_g13 : G13ALU<"fcmpsel", (outs AGXDst:$dst),
                         (ins AGXFCond3:$cc, AGXSrcF:$src0, AGXSrcF:$src1,
                              AGXCmpselSrc:$src2, AGXCmpselSrc:$src3),
                         "$cc, $dst, $src0, $src1, $src2, $src3", 8, 10> {
  bits<10> dst;
  bits<3> cc;
  bits<14> src0;
  bits<14> src1;
  bits<11> src2;
  bits<11> src3;
  let Inst{6-0} = 0b0000010;
  let Inst{14-7} = dst{7-0};
  let Inst{27-16} = src0{11-0};
  let Inst{39-28} = src1{11-0};
  let Inst{48-40} = src2{8-0};
  let Inst{60-52} = src3{8-0};
  let Inst{63-61} = cc;
  let Inst{69-68} = src3{10-9};
  let Inst{71-70} = src2{10-9};
  let Inst{73-72} = src1{13-12};
  let Inst{75-74} = src0{13-12};
  let Inst{77-76} = dst{9-8};
}

//===----------------------------------------------------------------------===//
// Ballots
//===----------------------------------------------------------------------===//

class G13ICmpBallot<string mn, bit simd>
    : G13ALU<mn, (outs AGXDst:$dst),
             (ins AGXICond:$cc, AGXSrc:$src0, AGXSrc:$src1),
             "$dst, $cc, $src0, $src1", 8> {
  bits<10> dst;
  bits<4> cc;
  bits<12> src0;
  bits<12> src1;
  let Inst{6-0} = 0b0110010;
  let Inst{14-7} = dst{7-0};
  let Inst{15} = 0;
  let Inst{25-16} = src0{9-0};
  let Inst{27-26} = 0;
  let Inst{37-28} = src1{9-0};
  let Inst{39-38} = 0;
  let Inst{41-40} = src1{11-10};
  let Inst{43-42} = src0{11-10};
  let Inst{45-44} = dst{9-8};
  let Inst{47} = cc{3};
  let Inst{48} = simd;
  let Inst{60-49} = 0;
  let Inst{63-61} = cc{2-0};
}

class G13FCmpBallot<string mn, bit simd>
    : G13ALU<mn, (outs AGXDst:$dst),
             (ins AGXFCond:$cc, AGXSrcF:$src0, AGXSrcF:$src1),
             "$dst, $cc, $src0, $src1", 8> {
  bits<10> dst;
  bits<4> cc;
  bits<14> src0;
  bits<14> src1;
  let Inst{6-0} = 0b0100010;
  let Inst{14-7} = dst{7-0};
  let Inst{15} = 0;
  let Inst{27-16} = src0{11-0};
  let Inst{39-28} = src1{11-0};
  let Inst{41-40} = src1{13-12};
  let Inst{43-42} = src0{13-12};
  let Inst{45-44} = dst{9-8};
  let Inst{47} = cc{3};
  let Inst{48} = simd;
  let Inst{60-49} = 0;
  let Inst{63-61} = cc{2-0};
}

def ICMP_BALLOT_g13 : G13ICmpBallot<"icmp_ballot", 1>;
def ICMP_QUAD_BALLOT_g13 : G13ICmpBallot<"icmp_quad_ballot", 0>;
def FCMP_BALLOT_g13 : G13FCmpBallot<"fcmp_ballot", 1>;
def FCMP_QUAD_BALLOT_g13 : G13FCmpBallot<"fcmp_quad_ballot", 0>;

//===----------------------------------------------------------------------===//
// SIMD-group and quad-group shuffles and reductions
//===----------------------------------------------------------------------===//

// SIMD-group shuffles take a 16-bit lane index; quad-group shuffles accept
// up to 64-bit sources for both operands.
class G13Shuffle<string mn, bit simd, bits<2> kind, bit rotate>
    : G13ALU<mn, (outs AGXDst:$dst),
             !if(simd, (ins AGXSrc:$src0, AGXSrc16:$src1),
                       (ins AGXSrc64:$src0, AGXSrc64:$src1)),
             "$dst, $src0, $src1", 6> {
  bits<10> dst;
  bits<12> src0;
  bits<12> src1;
  let Inst{6-0} = 0b1101111;
  let Inst{14-7} = dst{7-0};
  let Inst{15} = 0;
  let Inst{25-16} = src0{9-0};
  let Inst{27-26} = {0, simd};
  let Inst{37-28} = src1{9-0};
  let Inst{39-38} = kind;
  let Inst{41-40} = src1{11-10};
  let Inst{43-42} = src0{11-10};
  let Inst{45-44} = dst{9-8};
  let Inst{47} = rotate;
}

class AGXGroupInfo<string name, bit simd> {
  string Name = name;
  string Upper = !toupper(name);
  bit Simd = simd;
}

foreach Group = [AGXGroupInfo<"simd", 1>, AGXGroupInfo<"quad", 0>] in {
  defvar G = Group.Name;
  defvar U = Group.Upper;
  def U#_SHUFFLE_g13 : G13Shuffle<G#"_shuffle", Group.Simd, 0, 0>;
  def U#_SHUFFLE_XOR_g13 : G13Shuffle<G#"_shuffle_xor", Group.Simd, 1, 0>;
  def U#_SHUFFLE_UP_g13 : G13Shuffle<G#"_shuffle_up", Group.Simd, 2, 0>;
  def U#_SHUFFLE_DOWN_g13 : G13Shuffle<G#"_shuffle_down", Group.Simd, 3, 0>;
  def U#_SHUFFLE_ROTATE_UP_g13 : G13Shuffle<G#"_shuffle_rotate_up",
                                            Group.Simd, 2, 1>;
}

// Reductions and exclusive scans. opc: 0 quad reduce, 1 simd reduce,
// 2 quad prefix, 3 simd prefix. op is the AGX::SIMDOp number.
class G13Reduce<string mn, bits<2> opc, bits<5> op>
    : G13ALU<mn, (outs AGXDst:$dst), (ins AGXSrc64:$src0), "$dst, $src0", 6> {
  bits<10> dst;
  bits<12> src0;
  let Inst{6-0} = 0b1101111;
  let Inst{14-7} = dst{7-0};
  let Inst{15} = 1;
  let Inst{25-16} = src0{9-0};
  let Inst{27-26} = 0;
  let Inst{28} = op{0};
  let Inst{30-29} = opc;
  let Inst{37-31} = 0;
  let Inst{40-38} = op{3-1};
  let Inst{41} = 0;
  let Inst{43-42} = src0{11-10};
  let Inst{45-44} = dst{9-8};
  let Inst{47} = op{4};
}

class AGXSIMDOpInfo<string name, string mn, int op> {
  string Name = name;
  string Mnemonic = mn;
  int Op = op;
}

defvar AGXSIMDOps = [AGXSIMDOpInfo<"AND", "and", 0>,
                     AGXSIMDOpInfo<"FADD", "fadd", 1>,
                     AGXSIMDOpInfo<"OR", "or", 2>,
                     AGXSIMDOpInfo<"FMUL", "fmul", 3>,
                     AGXSIMDOpInfo<"XOR", "xor", 4>,
                     AGXSIMDOpInfo<"FMIN", "min.f", 5>,
                     AGXSIMDOpInfo<"FMAX", "max.f", 7>,
                     AGXSIMDOpInfo<"IADD", "iadd", 16>,
                     AGXSIMDOpInfo<"SMIN", "min.s", 20>,
                     AGXSIMDOpInfo<"SMAX", "max.s", 22>,
                     AGXSIMDOpInfo<"UMIN", "min.u", 28>,
                     AGXSIMDOpInfo<"UMAX", "max.u", 30>];

class AGXReduceKind<string name, string prefix, int opc> {
  string Name = name;
  string Prefix = prefix;
  int Opc = opc;
}

defvar AGXReduceKinds = [AGXReduceKind<"QUAD_REDUCE", "quad_", 0>,
                         AGXReduceKind<"SIMD_REDUCE", "simd_", 1>,
                         AGXReduceKind<"QUAD_PREFIX", "quad_prefix_", 2>,
                         AGXReduceKind<"SIMD_PREFIX", "simd_prefix_", 3>];

foreach Kind = AGXReduceKinds in
foreach Op = AGXSIMDOps in
def Kind.Name#_#Op.Name#_g13
    : G13Reduce<Kind.Prefix#Op.Mnemonic, Kind.Opc, Op.Op>;

def SIMD_MATRIX_FMADD32_g13
    : G13ALU<"simd_matrix_fmadd32", (outs AGXDst:$dst),
             (ins AGXSrcF:$src0, AGXSrcF:$src1, AGXSrcF:$src2),
             "$dst, $src0, $src1, $src2", 8> {
  bits<10> dst;
  bits<14> src0;
  bits<14> src1;
  bits<14> src2;
  let Inst{6-0} = 0b1101111;
  let Inst{14-7} = dst{7-0};
  let Inst{15} = 0;
  let Inst{25-16} = src0{9-0};
  let Inst{26} = 1;
  let Inst{27} = 1;
  let Inst{37-28} = src1{9-0};
  let Inst{39-38} = src1{11-10};
  let Inst{49-40} = src2{9-0};
  let Inst{51-50} = src2{11-10};
  let Inst{53-52} = src0{11-10};
  let Inst{55-54} = src2{13-12};
  let Inst{57-56} = src1{13-12};
  let Inst{59-58} = src0{13-12};
  let Inst{61-60} = dst{9-8};
  let Inst{63} = 1;
}

//===----------------------------------------------------------------------===//
// Execution mask control flow
//===----------------------------------------------------------------------===//

// cf: 0 = if, 1 = else, 2 = while. The nesting counter r0l is implicit.
class G13ICmpCF<string mn, bits<2> cf>
    : AGXReal<mn, (outs),
              (ins AGXNestCounter:$nest, AGXICond:$cc, AGXSrc:$src0,
                   AGXSrc:$src1, AGXImm:$n),
              "$nest, $cc, $src0, $src1, $n", 6> {
  bits<1> nest;
  bits<4> cc;
  bits<12> src0;
  bits<12> src1;
  bits<2> n;
  let IsExecMaskOp = 1;
  let Inst{6-0} = 0b1010010;
  let Inst{7} = nest;
  let Inst{8} = cc{3};
  let Inst{10-9} = cf;
  let Inst{12-11} = n;
  let Inst{15-13} = cc{2-0};
  let Inst{25-16} = src0{9-0};
  let Inst{27-26} = 0;
  let Inst{37-28} = src1{9-0};
  let Inst{39-38} = 0;
  let Inst{41-40} = src1{11-10};
  let Inst{43-42} = src0{11-10};
  let Inst{45-44} = 0;
}

class G13FCmpCF<string mn, bits<2> cf>
    : AGXReal<mn, (outs),
              (ins AGXNestCounter:$nest, AGXFCond:$cc, AGXSrcF:$src0,
                   AGXSrcF:$src1, AGXImm:$n),
              "$nest, $cc, $src0, $src1, $n", 6> {
  bits<1> nest;
  bits<4> cc;
  bits<14> src0;
  bits<14> src1;
  bits<2> n;
  let IsExecMaskOp = 1;
  let Inst{6-0} = 0b1000010;
  let Inst{7} = nest;
  let Inst{8} = cc{3};
  let Inst{10-9} = cf;
  let Inst{12-11} = n;
  let Inst{15-13} = cc{2-0};
  let Inst{27-16} = src0{11-0};
  let Inst{39-28} = src1{11-0};
  let Inst{41-40} = src1{13-12};
  let Inst{43-42} = src0{13-12};
  let Inst{45-44} = 0;
}

def IF_ICMP_g13 : G13ICmpCF<"if_icmp", 0>;
def ELSE_ICMP_g13 : G13ICmpCF<"else_icmp", 1>;
def WHILE_ICMP_g13 : G13ICmpCF<"while_icmp", 2>;
def IF_FCMP_g13 : G13FCmpCF<"if_fcmp", 0>;
def ELSE_FCMP_g13 : G13FCmpCF<"else_fcmp", 1>;
def WHILE_FCMP_g13 : G13FCmpCF<"while_fcmp", 2>;

def POP_EXEC_g13 : AGXReal<"pop_exec", (outs),
                           (ins AGXNestCounter:$nest, AGXImm:$n),
                           "$nest, $n", 6> {
  bits<1> nest;
  bits<2> n;
  let IsExecMaskOp = 1;
  let Inst{6-0} = 0b1010010;
  let Inst{7} = nest;
  let Inst{8} = 0;
  let Inst{10-9} = 3;
  let Inst{12-11} = n;
  let Inst{47-13} = 0;
}

let isBranch = 1, isTerminator = 1 in {
def JMP_EXEC_ANY_g13 : AGXReal<"jmp_exec_any", (outs),
                               (ins AGXBrTarget:$target), "$target", 6> {
  bits<32> target;
  let Inst{15-0} = 0xC000;
  let Inst{47-16} = target;
}

def JMP_EXEC_NONE_g13 : AGXReal<"jmp_exec_none", (outs),
                                (ins AGXBrTarget:$target), "$target", 6> {
  bits<32> target;
  let Inst{15-0} = 0xC020;
  let Inst{47-16} = target;
}
}

let isTerminator = 1, isReturn = 1, isBarrier = 1 in
def STOP_g13 : AGXReal<"stop", (outs), (ins), "", 2> {
  let Inst{15-0} = 0x0088;
}

let isTrap = 1, isTerminator = 1, isBarrier = 1 in
def TRAP_g13 : AGXReal<"trap", (outs), (ins), "", 2> {
  let Inst{15-0} = 0x0008;
}

def WAIT_g13 : AGXReal<"wait", (outs), (ins AGXImm:$slot), "$slot", 2> {
  bits<1> slot;
  let Inst{7-0} = 0x38;
  let Inst{8} = slot;
}

def THREADGROUP_BARRIER_g13 : AGXReal<"threadgroup_barrier", (outs), (ins),
                                      "", 2> {
  let Inst{15-0} = 0x0068;
}

def MEMORY_BARRIER_g13 : AGXReal<"memory_barrier", (outs),
                                 (ins AGXImm:$a, AGXImm:$b, AGXImm:$c),
                                 "$a, $b, $c", 2> {
  bits<2> a;
  bits<2> b;
  bits<4> c;
  let Inst{7-0} = 0xF5;
  let Inst{9-8} = b;
  let Inst{11-10} = a;
  let Inst{15-12} = c;
}

//===----------------------------------------------------------------------===//
// Device memory
//===----------------------------------------------------------------------===//

// Fields shared by device memory, uniform store and stack instructions:
// R register 15:10 + 41:40 with 32-bit flag 49; index O 23:20 + 35:32 + 63:56
// with immediate flag 24; scoreboard slot 30; length bit 47 (always set).
class G13DeviceMem<string mn, bits<7> op, dag outs, dag ins, string ops>
    : AGXReal<mn, outs, ins, ops, 8> {
  bits<1> g;
  bits<4> fmt;
  bits<4> mask;
  bits<9> R;
  bits<9> A;
  bits<17> O;
  bits<1> sgn;
  bits<2> shift;
  let IsMemory = 1;
  let Inst{6-0} = op;
  let Inst{9-7} = fmt{2-0};
  let Inst{15-10} = R{5-0};
  let Inst{19-16} = A{3-0};
  let Inst{23-20} = O{3-0};
  let Inst{24} = O{16};
  let Inst{25} = sgn;
  let Inst{26} = 1;
  let Inst{27} = A{8};
  let Inst{29-28} = 0;
  let Inst{30} = g;
  let Inst{35-32} = O{7-4};
  let Inst{39-36} = A{7-4};
  let Inst{41-40} = R{7-6};
  let Inst{43-42} = shift;
  let Inst{46-44} = 4;
  let Inst{47} = 1;
  let Inst{48} = fmt{3};
  let Inst{49} = R{8};
  let Inst{51-50} = 0;
  let Inst{55-52} = mask;
  let Inst{63-56} = O{15-8};
}

let mayLoad = 1, IsAsync = 1 in
def DEVICE_LOAD_g13
    : G13DeviceMem<"device_load", 0b0000101, (outs AGXMemReg:$R),
                   (ins AGXImm:$g, AGXMemFormat:$fmt, AGXMemMask:$mask,
                        AGXMemBase:$A, AGXMemIndex:$O, AGXSignedness:$sgn,
                        AGXShift:$shift),
                   "$g, $fmt, $mask, $R, $A, $O, $sgn$shift">;

let mayStore = 1 in
def DEVICE_STORE_g13
    : G13DeviceMem<"device_store", 0b1000101, (outs),
                   (ins AGXImm:$g, AGXMemFormat:$fmt, AGXMemMask:$mask,
                        AGXMemReg:$R, AGXMemBase:$A, AGXMemIndex:$O,
                        AGXSignedness:$sgn, AGXShift:$shift),
                   "$g, $fmt, $mask, $R, $A, $O, $sgn$shift">;

// Initialise uniform registers from general purpose registers. The index is
// the first 16-bit uniform register written.
let mayStore = 1, hasSideEffects = 1 in
def UNIFORM_STORE_g13
    : AGXReal<"uniform_store", (outs),
              (ins AGXImm:$g, AGXMemMask:$mask, AGXMemReg:$R,
                   AGXMemIndex:$O),
              "$g, $mask, $R, $O", 8> {
  bits<1> g;
  bits<4> mask;
  bits<9> R;
  bits<17> O;
  let IsMemory = 1;
  let Inst{6-0} = 0b1000101;
  let Inst{9-7} = 1;
  let Inst{15-10} = R{5-0};
  let Inst{19-16} = 0;
  let Inst{23-20} = O{3-0};
  let Inst{24} = O{16};
  let Inst{26-25} = 0b10;
  let Inst{29-27} = 0b111;
  let Inst{30} = g;
  let Inst{35-32} = O{7-4};
  let Inst{39-36} = 0;
  let Inst{41-40} = R{7-6};
  let Inst{43-42} = 0;
  let Inst{46-44} = 0;
  let Inst{47} = 1;
  let Inst{48} = 0;
  let Inst{49} = R{8};
  let Inst{51-50} = 0;
  let Inst{55-52} = mask;
  let Inst{63-56} = O{15-8};
}

//===----------------------------------------------------------------------===//
// Stack (scratch) memory
//===----------------------------------------------------------------------===//

class G13StackMem<string mn, bits<8> op, dag outs, dag ins>
    : AGXReal<mn, outs, ins, "$g, $fmt, $mask, $R, $O", 8> {
  bits<1> g;
  bits<4> fmt;
  bits<4> mask;
  bits<9> R;
  bits<17> O;
  let IsMemory = 1;
  let Inst{7-0} = op;
  let Inst{9-8} = fmt{1-0};
  let Inst{15-10} = R{5-0};
  let Inst{19-16} = 0;
  let Inst{23-20} = O{3-0};
  let Inst{24} = O{16};
  let Inst{25} = 0;
  let Inst{26} = 1;
  let Inst{29-27} = 0;
  let Inst{30} = g;
  let Inst{35-32} = O{7-4};
  let Inst{38-36} = 0;
  let Inst{41-40} = R{7-6};
  let Inst{46-44} = 4;
  let Inst{47} = 1;
  let Inst{49} = R{8};
  let Inst{51-50} = fmt{3-2};
  let Inst{55-52} = mask;
  let Inst{63-56} = O{15-8};
}

let mayLoad = 1, IsAsync = 1 in
def STACK_LOAD_g13
    : G13StackMem<"stack_load", 0x35, (outs AGXMemReg:$R),
                  (ins AGXImm:$g, AGXMemFormat:$fmt, AGXMemMask:$mask,
                       AGXMemIndex:$O)>;

let mayStore = 1 in
def STACK_STORE_g13
    : G13StackMem<"stack_store", 0xB5, (outs),
                  (ins AGXImm:$g, AGXMemFormat:$fmt, AGXMemMask:$mask,
                       AGXMemReg:$R, AGXMemIndex:$O)>;

let hasSideEffects = 1 in
def STACK_ADJUST_g13 : AGXReal<"stack_adjust", (outs), (ins AGXImm:$size),
                               "$size", 8> {
  bits<16> size;
  let Inst{7-0} = 0xB5;
  let Inst{19-16} = 1;
  let Inst{23-20} = size{3-0};
  let Inst{24} = 1;
  let Inst{25} = 0;
  let Inst{26} = 1;
  let Inst{35-32} = size{7-4};
  let Inst{38-36} = 2;
  let Inst{46-44} = 0;
  let Inst{47} = 1;
  let Inst{55-50} = 0;
  let Inst{63-56} = size{15-8};
}

//===----------------------------------------------------------------------===//
// Threadgroup memory
//===----------------------------------------------------------------------===//

class G13LocalMem<string mn, bits<2> op, dag outs, dag ins>
    : AGXReal<mn, outs, ins, "$fmt, $mask, $R, $A, $O", 8> {
  bits<4> fmt;
  bits<4> mask;
  bits<9> R;
  bits<10> A;
  bits<17> O;
  let IsMemory = 1;
  let Inst{3-0} = 0b1001;
  let Inst{4} = 0;
  let Inst{6-5} = op;
  let Inst{7} = 0;
  let Inst{8} = R{8};
  let Inst{14-9} = R{5-0};
  let Inst{15} = 1;
  let Inst{21-16} = A{5-0};
  let Inst{23-22} = A{9-8};
  let Inst{27-24} = fmt;
  let Inst{33-28} = O{5-0};
  let Inst{34} = O{16};
  let Inst{35} = 0;
  let Inst{39-36} = mask;
  let Inst{47-40} = 0;
  let Inst{57-48} = O{15-6};
  let Inst{59-58} = A{7-6};
  let Inst{61-60} = R{7-6};
  let Inst{63-62} = 0;
}

let mayLoad = 1 in
def THREADGROUP_LOAD_g13
    : G13LocalMem<"threadgroup_load", 0b11, (outs AGXMemReg:$R),
                  (ins AGXMemFormat:$fmt, AGXMemMask:$mask, AGXLocalBase:$A,
                       AGXLocalIndex:$O)>;

let mayStore = 1 in
def THREADGROUP_STORE_g13
    : G13LocalMem<"threadgroup_store", 0b01, (outs),
                  (ins AGXMemFormat:$fmt, AGXMemMask:$mask, AGXMemReg:$R,
                       AGXLocalBase:$A, AGXLocalIndex:$O)>;

//===----------------------------------------------------------------------===//
// Atomics
//===----------------------------------------------------------------------===//

let mayLoad = 1, mayStore = 1, hasSideEffects = 1, IsAsync = 1 in
def ATOMIC_g13 : AGXReal<"atomic", (outs AGXAtomicDst:$R),
                         (ins AGXAtomicOp:$op, AGXImm:$g, AGXMemBase:$A,
                              AGXMemIndex:$O, AGXSignedness:$sgn,
                              AGXAtomicSrc:$S, AGXImm:$coherent),
                         "$op, $g, $R, $A, $O, $sgn, $S, $coherent", 8> {
  bits<4> op;
  bits<1> g;
  bits<9> R;
  bits<9> A;
  bits<17> O;
  bits<1> sgn;
  bits<8> S;
  bits<1> coherent;
  let IsMemory = 1;
  let Inst{5-0} = 0b010101;
  let Inst{9-6} = op;
  let Inst{15-10} = R{5-0};
  let Inst{19-16} = A{3-0};
  let Inst{23-20} = O{3-0};
  let Inst{24} = O{16};
  let Inst{25} = sgn;
  let Inst{26} = 1;
  let Inst{27} = A{8};
  let Inst{29-28} = 0;
  let Inst{30} = g;
  let Inst{31} = 1;
  let Inst{35-32} = O{7-4};
  let Inst{39-36} = A{7-4};
  let Inst{41-40} = R{7-6};
  let Inst{43-42} = 0;
  let Inst{44} = 1;
  let Inst{45} = coherent;
  let Inst{46} = 1;
  let Inst{47} = R{8};
  let Inst{55-48} = S;
  let Inst{63-56} = O{15-8};
}

let mayLoad = 1, mayStore = 1, hasSideEffects = 1 in
def THREADGROUP_ATOMIC_g13
    : AGXReal<"threadgroup_atomic", (outs AGXAtomicDst:$R),
              (ins AGXAtomicOp:$op, AGXLocalBase:$A, AGXLocalIndex:$O,
                   AGXAtomicSrc:$S),
              "$op, $R, $A, $O, $S", 10> {
  bits<4> op;
  bits<9> R;
  bits<10> A;
  bits<17> O;
  bits<8> S;
  let IsMemory = 1;
  let Inst{5-0} = 0b011001;
  let Inst{7-6} = 0;
  let Inst{8} = R{8};
  let Inst{14-9} = R{5-0};
  let Inst{15} = 1;
  let Inst{21-16} = A{5-0};
  let Inst{23-22} = A{9-8};
  let Inst{27-24} = op;
  let Inst{33-28} = O{5-0};
  let Inst{34} = O{16};
  let Inst{37-35} = 0b010;
  let Inst{38} = R{8};
  let Inst{46-39} = 0;
  let Inst{47} = 1;
  let Inst{57-48} = O{15-6};
  let Inst{59-58} = A{7-6};
  let Inst{61-60} = R{7-6};
  let Inst{63-62} = 0;
  let Inst{71-64} = S;
  let Inst{79-72} = 0;
}
