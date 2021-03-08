//=== M68kInstrFormats.td - M68k Instruction Formats ---*- tablegen -*-===//
//                     The LLVM Compiler Infrastructure
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains M68k instruction formats.
///
/// Since M68k has quite a lot memory addressing modes there are more
/// instruction prefixes than just i, r and m:
/// TSF  Since     Form                     Letter  Description
///  00   M68000    Dn or An                 r       any register
///  01   M68000    Dn                       d       data register direct
///  02   M68000    An                       a       address register direct
///  03   M68000    (An)                     j       address register indirect
///  04   M68000    (An)+                    o       address register indirect with postincrement
///  05   M68000    -(An)                    e       address register indirect with predecrement
///  06   M68000    (i,An)                   p       address register indirect with displacement
///  10   M68000    (i,An,Xn.L)              f       address register indirect with index and scale = 1
///  07   M68000    (i,An,Xn.W)              F       address register indirect with index and scale = 1
///  12   M68020    (i,An,Xn.L,SCALE)        g       address register indirect with index
///  11   M68020    (i,An,Xn.W,SCALE)        G       address register indirect with index
///  14   M68020    ([bd,An],Xn.L,SCALE,od)  u       memory indirect postindexed mode
///  13   M68020    ([bd,An],Xn.W,SCALE,od)  U       memory indirect postindexed mode
///  16   M68020    ([bd,An,Xn.L,SCALE],od)  v       memory indirect preindexed mode
///  15   M68020    ([bd,An,Xn.W,SCALE],od)  V       memory indirect preindexed mode
///  20   M68000    abs.L                    b       absolute long address
///  17   M68000    abs.W                    B       absolute short address
///  21   M68000    (i,PC)                   q       program counter with displacement
///  23   M68000    (i,PC,Xn.L)              k       program counter with index and scale = 1
///  22   M68000    (i,PC,Xn.W)              K       program counter with index and scale = 1
///  25   M68020    (i,PC,Xn.L,SCALE)        l       program counter with index
///  24   M68020    (i,PC,Xn.W,SCALE)        L       program counter with index
///  27   M68020    ([bd,PC],Xn.L,SCALE,od)  x       program counter memory indirect postindexed mode
///  26   M68020    ([bd,PC],Xn.W,SCALE,od)  X       program counter memory indirect postindexed mode
///  31   M68020    ([bd,PC,Xn.L,SCALE],od)  y       program counter memory indirect preindexed mode
///  30   M68020    ([bd,PC,Xn.W,SCALE],od)  Y       program counter memory indirect preindexed mode
///  32   M68000    #immediate               i       immediate data
///
/// NOTE that long form is always lowercase, word variants are capitalized
///
/// Operand can be qualified with size where appropriate to force a particular
/// instruction encoding, e.g.:
///    (i8,An,Xn.W)             f8      1 extension word
///    (i16,An,Xn.W)            f16     2 extension words
///    (i32,An,Xn.W)            f32     3 extension words
///
/// Form without size qualifier will adapt to operand size automatically, e.g.:
///    (i,An,Xn.W)              f       1, 2 or 3 extension words
///
/// Some forms already imply a particular size of their operands, e.g.:
///    (i,An)                   p       1 extension word and i is 16bit
///
/// Operand order follows x86 Intel order(destination before source), e.g.:
///    MOV8df                   MOVE (4,A0,D0), D1
///
/// Number after instruction mnemonics determines the size of the data
///
//===----------------------------------------------------------------------===//

/// ??? Is it possible to use this stuff for disassembling?
/// NOTE 1: In case of conditional beads(DA, DAReg), cond part is able to
/// consume any bit, though a more general instructions must be chosen, e.g.
/// d -> r, a -> r

//===----------------------------------------------------------------------===//
// Encoding primitives
//===----------------------------------------------------------------------===//

class MxBead<bits<4> type, bit b4 = 0, bit b5 = 0, bit b6 = 0, bit b7 = 0> {
  bits<8> Value = 0b00000000;
  let Value{3-0} = type;
  let Value{4} = b4;
  let Value{5} = b5;
  let Value{6} = b6;
  let Value{7} = b7;
}

/// System beads, allow to control beading flow
def   MxBeadTerm   : MxBead<0x0, 0, 0, 0, 0>;
def   MxBeadIgnore : MxBead<0x0, 1, 0, 0, 0>;

/// Add plain bit to the instruction
class MxBead1Bit  <bits<1> b> : MxBead<0x1, b>;
class MxBead2Bits <bits<2> b> : MxBead<0x2, b{0}, b{1}>;
class MxBead3Bits <bits<3> b> : MxBead<0x3, b{0}, b{1}, b{2}>;
class MxBead4Bits <bits<4> b> : MxBead<0x4, b{0}, b{1}, b{2}, b{3}>;

/// bits<3> o - operand number
/// bit a     - use alternative, used to select index register or
///             outer displacement/immediate
/// suffix NP means non-padded
class MxBeadDAReg  <bits<3> o, bit a = 0> : MxBead<0x5, o{0}, o{1}, o{2}, a>;
class MxBeadDA     <bits<3> o, bit a = 0> : MxBead<0x6, o{0}, o{1}, o{2}, a>;
class MxBeadReg    <bits<3> o, bit a = 0> : MxBead<0x7, o{0}, o{1}, o{2}, a>;
class MxBead8Disp  <bits<3> o, bit a = 0> : MxBead<0x8, o{0}, o{1}, o{2}, a>;

/// Add Immediate to the instruction. 8-bit version is padded with zeros to fit
/// the word.
class MxBead8Imm   <bits<3> o, bit a = 0> : MxBead<0x9, o{0}, o{1}, o{2}, a>;
class MxBead16Imm  <bits<3> o, bit a = 0> : MxBead<0xA, o{0}, o{1}, o{2}, a>;
class MxBead32Imm  <bits<3> o, bit a = 0> : MxBead<0xB, o{0}, o{1}, o{2}, a>;

/// Encodes an immediate 0-7(alt. 1-8) into 3 bit field
class MxBead3Imm   <bits<3> o, bit a = 0> : MxBead<0xC, o{0}, o{1}, o{2}, a>;


class MxEncoding<MxBead n0  = MxBeadTerm, MxBead n1  = MxBeadTerm,
                 MxBead n2  = MxBeadTerm, MxBead n3  = MxBeadTerm,
                 MxBead n4  = MxBeadTerm, MxBead n5  = MxBeadTerm,
                 MxBead n6  = MxBeadTerm, MxBead n7  = MxBeadTerm,
                 MxBead n8  = MxBeadTerm, MxBead n9  = MxBeadTerm,
                 MxBead n10 = MxBeadTerm, MxBead n11 = MxBeadTerm,
                 MxBead n12 = MxBeadTerm, MxBead n13 = MxBeadTerm,
                 MxBead n14 = MxBeadTerm, MxBead n15 = MxBeadTerm,
                 MxBead n16 = MxBeadTerm, MxBead n17 = MxBeadTerm,
                 MxBead n18 = MxBeadTerm, MxBead n19 = MxBeadTerm,
                 MxBead n20 = MxBeadTerm, MxBead n21 = MxBeadTerm,
                 MxBead n22 = MxBeadTerm, MxBead n23 = MxBeadTerm> {
  bits <192> Value;
  let Value{7-0}     = n0.Value;
  let Value{15-8}    = n1.Value;
  let Value{23-16}   = n2.Value;
  let Value{31-24}   = n3.Value;
  let Value{39-32}   = n4.Value;
  let Value{47-40}   = n5.Value;
  let Value{55-48}   = n6.Value;
  let Value{63-56}   = n7.Value;
  let Value{71-64}   = n8.Value;
  let Value{79-72}   = n9.Value;
  let Value{87-80}   = n10.Value;
  let Value{95-88}   = n11.Value;
  let Value{103-96}  = n12.Value;
  let Value{111-104} = n13.Value;
  let Value{119-112} = n14.Value;
  let Value{127-120} = n15.Value;
  let Value{135-128} = n16.Value;
  let Value{143-136} = n17.Value;
  let Value{151-144} = n18.Value;
  let Value{159-152} = n19.Value;
  let Value{167-160} = n20.Value;
  let Value{175-168} = n21.Value;
  let Value{183-176} = n22.Value;
  let Value{191-184} = n23.Value;
}

class MxEncFixed<bits<16> value> : MxEncoding {
  let Value{7-0}   = MxBead4Bits<value{3-0}>.Value;
  let Value{15-8}  = MxBead4Bits<value{7-4}>.Value;
  let Value{23-16} = MxBead4Bits<value{11-8}>.Value;
  let Value{31-24} = MxBead4Bits<value{15-12}>.Value;
}

//===----------------------------------------------------------------------===//
// Encoding composites
//
// These must be lowered to MxEncoding by instr specific wrappers
//
// HERE BE DRAGONS...
//===----------------------------------------------------------------------===//

class MxEncByte<bits<8> value> : MxEncoding {
  MxBead4Bits LO = MxBead4Bits<value{3-0}>;
  MxBead4Bits HI = MxBead4Bits<value{7-4}>;
}

def MxEncEmpty : MxEncoding;


/// M68k Standard Effective Address layout:
///
/// :-------------------:
/// | 5  4  3 | 2  1  0 |
/// |   mode  |   reg   |
/// :-------------------:
///
/// If the EA is a direct register mode, bits 4 and 5 are 0, and the register
/// number will be encoded in bit 0 - 3. Since the first address register's
/// (A0) register number is 8, we can easily tell data registers from
/// address registers by only inspecting bit 3 (i.e. if bit 3 is set, it's an
/// address register).
///
///
/// But MOVE instruction uses reversed layout for destination EA:
///
/// :-------------------:
/// | 5  4  3 | 2  1  0 |
/// |   reg   |  mode   |
/// :-------------------:
///
/// And this complicates things a bit because the DA bit is now separated from
/// the register and we have to encode those separately using MxBeadDA<opN>
///
class MxEncEA<MxBead reg, MxBead mode, MxBead da = MxBeadIgnore> {
  MxBead Reg = reg;
  MxBead Mode = mode;
  MxBead DA = da;
}

// FIXME: Is there a way to factorize the addressing mode suffix (i.e.
// 'r', 'd', 'a' etc.) and use something like multiclass to replace?
def MxEncEAr_0: MxEncEA<MxBeadDAReg<0>, MxBead2Bits<0b00>>;
def MxEncEAd_0: MxEncEA<MxBeadReg<0>, MxBead2Bits<0b00>, MxBead1Bit<0>>;
def MxEncEAa_0: MxEncEA<MxBeadReg<0>, MxBead2Bits<0b00>, MxBead1Bit<1>>;
def MxEncEAj_0: MxEncEA<MxBeadReg<0>, MxBead2Bits<0b01>, MxBead1Bit<0>>;
def MxEncEAo_0: MxEncEA<MxBeadReg<0>, MxBead2Bits<0b01>, MxBead1Bit<1>>;
def MxEncEAe_0: MxEncEA<MxBeadReg<0>, MxBead2Bits<0b10>, MxBead1Bit<0>>;
def MxEncEAp_0: MxEncEA<MxBeadReg<0>, MxBead2Bits<0b10>, MxBead1Bit<1>>;
def MxEncEAf_0: MxEncEA<MxBeadReg<0>, MxBead2Bits<0b11>, MxBead1Bit<0>>;

def MxEncEAa_0_reflected : MxEncEA<MxBeadReg<0>, MxBead3Bits<0b001>>;
def MxEncEAr_0_reflected : MxEncEA<MxBeadReg<0>, MxBead2Bits<0b00>, MxBeadDA<0>>;

def MxEncEAr_1: MxEncEA<MxBeadDAReg<1>, MxBead2Bits<0b00>>;
def MxEncEAd_1: MxEncEA<MxBeadReg<1>, MxBead2Bits<0b00>, MxBead1Bit<0>>;
def MxEncEAa_1: MxEncEA<MxBeadReg<1>, MxBead2Bits<0b00>, MxBead1Bit<1>>;
def MxEncEAj_1: MxEncEA<MxBeadReg<1>, MxBead2Bits<0b01>, MxBead1Bit<0>>;
def MxEncEAo_1: MxEncEA<MxBeadReg<1>, MxBead2Bits<0b01>, MxBead1Bit<1>>;
def MxEncEAe_1: MxEncEA<MxBeadReg<1>, MxBead2Bits<0b10>, MxBead1Bit<0>>;
def MxEncEAp_1: MxEncEA<MxBeadReg<1>, MxBead2Bits<0b10>, MxBead1Bit<1>>;
def MxEncEAf_1: MxEncEA<MxBeadReg<1>, MxBead2Bits<0b11>, MxBead1Bit<0>>;

def MxEncEAr_2: MxEncEA<MxBeadDAReg<2>, MxBead2Bits<0b00>>;
def MxEncEAd_2: MxEncEA<MxBeadReg<2>, MxBead2Bits<0b00>, MxBead1Bit<0>>;
def MxEncEAa_2: MxEncEA<MxBeadReg<2>, MxBead2Bits<0b00>, MxBead1Bit<1>>;
def MxEncEAj_2: MxEncEA<MxBeadReg<2>, MxBead2Bits<0b01>, MxBead1Bit<0>>;
def MxEncEAo_2: MxEncEA<MxBeadReg<2>, MxBead2Bits<0b01>, MxBead1Bit<1>>;
def MxEncEAe_2: MxEncEA<MxBeadReg<2>, MxBead2Bits<0b10>, MxBead1Bit<0>>;
def MxEncEAp_2: MxEncEA<MxBeadReg<2>, MxBead2Bits<0b10>, MxBead1Bit<1>>;
def MxEncEAf_2: MxEncEA<MxBeadReg<2>, MxBead2Bits<0b11>, MxBead1Bit<0>>;

def MxEncEAb : MxEncEA<MxBead3Bits<0b001>, MxBead2Bits<0b11>, MxBead1Bit<1>>;
def MxEncEAq : MxEncEA<MxBead3Bits<0b010>, MxBead2Bits<0b11>, MxBead1Bit<1>>;
def MxEncEAk : MxEncEA<MxBead3Bits<0b011>, MxBead2Bits<0b11>, MxBead1Bit<1>>;
def MxEncEAi : MxEncEA<MxBead3Bits<0b100>, MxBead2Bits<0b11>, MxBead1Bit<1>>;

// Allows you to specify each bit of opcode
class MxEncOpMode<MxBead b0, MxBead b1 = MxBeadIgnore, MxBead b2 = MxBeadIgnore> {
  MxBead B0 = b0;
  MxBead B1 = b1;
  MxBead B2 = b2;
}

// op EA, Dn
def MxOpMode8dEA  : MxEncOpMode<MxBead3Bits<0b000>>;
def MxOpMode16dEA : MxEncOpMode<MxBead3Bits<0b001>>;
def MxOpMode32dEA : MxEncOpMode<MxBead3Bits<0b010>>;

// op EA, An
def MxOpMode16aEA : MxEncOpMode<MxBead3Bits<0b110>>;
def MxOpMode32aEA : MxEncOpMode<MxBead3Bits<0b111>>;

// op EA, Rn
// As you might noticed this guy is special... Since M68k differentiates
// between Data and Address registers we required to use different OPMODE codes
// for Address registers DST operands. One way of dealing with it is to use
// separate tablegen instructions, but in this case it would force Register
// Allocator to use specific Register Classes and eventually will lead to
// superfluous moves. Another approach is to use reg-variadic encoding which will
// change OPMODE base on Register Class used. Luckily, all the bits that differ go
// from 0 to 1 and can be encoded with MxBeadDA.
// Basically, if the register used is of Data type these encodings will be
// the same as MxOpMode{16,32}dEA above and used with regular instructions(e.g. ADD,
// SUB), but if the register is of Address type the appropriate bits will flip and
// the instructions become of *A type(e.g ADDA, SUBA).
def MxOpMode16rEA : MxEncOpMode<MxBead1Bit<1>, MxBeadDA<0>, MxBead1Bit<0>>;
def MxOpMode32rEA : MxEncOpMode<MxBeadDA<0>, MxBead1Bit<1>, MxBeadDA<0>>;

// op Dn, EA
def MxOpMode8EAd : MxEncOpMode<MxBead3Bits<0b100>>;
def MxOpMode16EAd : MxEncOpMode<MxBead3Bits<0b101>>;
def MxOpMode32EAd : MxEncOpMode<MxBead3Bits<0b110>>;


// Represents two types of extension word:
//   - Imm extension word
//   - Brief extension word
class MxEncExt<MxBead imm   = MxBeadIgnore,   MxBead b8 = MxBeadIgnore,
               MxBead scale = MxBeadIgnore, MxBead wl = MxBeadIgnore,
               MxBead daReg = MxBeadIgnore> {
  MxBead Imm = imm;
  MxBead B8 = b8;
  MxBead Scale = scale;
  MxBead WL = wl;
  MxBead DAReg = daReg;
}

def MxExtEmpty : MxEncExt;

// These handle encoding of displacement fields, absolute addresses and
// immediate values, since encoding for these categories is mainly the same,
// with exception of some weird immediates.
def  MxExtI8_0 : MxEncExt<MxBead8Imm<0>>;
def MxExtI16_0 : MxEncExt<MxBead16Imm<0>>;
def MxExtI32_0 : MxEncExt<MxBead32Imm<0>>;

def  MxExtI8_1 : MxEncExt<MxBead8Imm<1>>;
def MxExtI16_1 : MxEncExt<MxBead16Imm<1>>;
def MxExtI32_1 : MxEncExt<MxBead32Imm<1>>;

def  MxExtI8_2 : MxEncExt<MxBead8Imm<2>>;
def MxExtI16_2 : MxEncExt<MxBead16Imm<2>>;
def MxExtI32_2 : MxEncExt<MxBead32Imm<2>>;

// NOTE They are all using Long Xn
def MxExtBrief_0 : MxEncExt<MxBead8Disp<0>, MxBead1Bit<0b0>,
                            MxBead2Bits<0b00>, MxBead1Bit<1>,
                            MxBeadDAReg<0, 1>>;

def MxExtBrief_1 : MxEncExt<MxBead8Disp<1>, MxBead1Bit<0b0>,
                            MxBead2Bits<0b00>, MxBead1Bit<1>,
                            MxBeadDAReg<1, 1>>;

def MxExtBrief_2 : MxEncExt<MxBead8Disp<2>, MxBead1Bit<0b0>,
                            MxBead2Bits<0b00>, MxBead1Bit<1>,
                            MxBeadDAReg<2, 1>>;

def MxExtBrief_3 : MxEncExt<MxBead8Disp<3>, MxBead1Bit<0b0>,
                            MxBead2Bits<0b00>, MxBead1Bit<1>,
                            MxBeadDAReg<3, 1>>;

def MxExtBrief_4 : MxEncExt<MxBead8Disp<4>, MxBead1Bit<0b0>,
                            MxBead2Bits<0b00>, MxBead1Bit<1>,
                            MxBeadDAReg<4, 1>>;

class MxEncSize<bits<2> value> : MxBead2Bits<value>;
def MxEncSize8  : MxEncSize<0b00>;
def MxEncSize16 : MxEncSize<0b01>;
def MxEncSize32 : MxEncSize<0b10>;
def MxEncSize64 : MxEncSize<0b11>;

// M68k INSTRUCTION. Most instructions specify the location of an operand by
// using the effective address field in the operation word. The effective address
// is composed of two 3-bit fields: the mode field and the register field. The
// value in the mode field selects the different address modes. The register
// field contains the number of a register.  The effective address field may
// require additional information to fully specify the operand. This additional
// information, called the effective address extension, is contained in the
// following word or words and is considered part of the instruction. The
// effective address modes are grouped into three categories: register direct,
// memory addressing, and special.
class MxInst<dag outs, dag ins,
             string asmStr = "",
             list<dag> pattern = [],
             MxEncoding beads = MxEncEmpty,
             InstrItinClass itin = NoItinerary>
    : Instruction {
  let Namespace      = "M68k";
  let OutOperandList = outs;
  let InOperandList  = ins;
  let AsmString      = asmStr;
  let Pattern        = pattern;
  let Itinerary      = itin;

  // Byte stream
  field bits<192> Beads = beads.Value;

  // Number of bytes
  let Size = 0;

  let UseLogicalOperandMappings = 1;
}

// M68k PSEUDO INSTRUCTION
class MxPseudo<dag outs, dag ins, string asmStr = "", list<dag> pattern = []>
    : MxInst<outs, ins, asmStr, pattern> {
  let isPseudo = 1;
}
