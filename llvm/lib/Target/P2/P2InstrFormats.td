//===-- P2InstrFormats.td - P2 Instruction Formats -----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// instruction formats in P2
//
// Each instruction breaks down as one of the following:
//   COND | OOOOOOO | CZI | DDDDDDDDD | SSSSSSSSS || most math and logic commands. Single operand instructions have specific S values and I = 0. only S can be immediate
//   COND | OOOOOOO | CZL | DDDDDDDDD | SSSSSSSSS || single operand where D can be immediate, and can write both C and Z
//   COND | OOOOOOO | CLI | DDDDDDDDD | SSSSSSSSS || (notice that L can move!) instructions where both D and S can be immediates
//   COND | OOOOOON | NNI | DDDDDDDDD | SSSSSSSSS || special instructions that sets word or nibbles defined by N
//   COND | OOOOOOO | RAA | AAAAAAAAA | AAAAAAAAA || call/jump instructions that modify the entire PC (for hub exec modes)
//   COND | OOOOOnn | nnn | nnnnnnnnn | nnnnnnnnn || AUGS/AUGD
//
// COND = Conditions stuck on the instruction to determine if it should be executed based on status word flags
// O = op code. typically 7 bits, sometimes fewer and the extra bits are used as operands
// C = write C. Will write C status flag if 1
// Z = write Z. Will write Z status flag if 1
// I = S is immediate value (and not register) if 1
// L = D is immediate value (and not register) if 1
// N = nib, byte, or short for special instructions that operate on them
// R = relative for hub exec call/jumps
// n = immediate to be glued to the next #D or #S immediate to turn it into a 32 bit immediate value.
//
// We'll split this up into the following classes:
//
//      0:  P2Inst:         Generic instruction class that should be subclasses for all the instructions.
//      1:  P2InstCZIDS:    7 bit opcode, CZI flags, and two 9-bit operands (D/S). S can be immediate
//      2:  P2Inst3NIDS:    6 bit opcode, a 3-bit N value for nib-based instructions, I flag, and two operands (D/S). S can be imm
//      3:  P2Inst2NIDS:    7 bit opcode, a 2-bit N value for byte-based instructios, I flag, and two operands (D/S). S can be imm
//      4:  P2Inst1NIDS:    8 bit opcode, a 1-bit N value for short (2-byte) based instructions, I flag, and two operands (D/S). S can be imm
//      5:  P2InstIDS:      9 bit opcode, I flag, and two operands (D/S). S can be imm.
//      6:  P2InstZIDS:     8 bit opcode, ZI flags, and two operands (D/S). S can be imm.
//      7:  P2InstCIDS:     8 bit (*) opcode, CI flags, and two operands (D/S). S can be imm.
//      8:  P2InstLIDS:     8 bit opcode, LI flags, and two operands (D/S). D and S can be imm.
//      9:  P2InstIS:       9 bit opcode, I flags, and one operand (S). S can be imm. D field serves as a sub-opcode.
//      10: P2InstCLIDS:    7 bit opcode, CLI flags, and two operands (D/S). D and S can be imm.
//      11: P2InstLD:       7 bit opcode, L flag, and one operand (D). D can be imm. S field serves as a sub-opcode.
//      12: P2InstCLD:      7 bit opcode, CL flags, and one operand (D). D can be imm. S field serves as a sub-opcode.
//      13: P2InstCZD:      7 bit opcode, CZ flags, and one operand (D). S field serves as a sub-opcode.
//      14: P2InstCZ:       8 bit (**) opcode, CZ flags and no operands. D and S fields are used as a sub-opcode
//      15: P2InstCZLD:     7 bit opcode, CZL flags, and one operand (D). D can be imm. S field serves as a sub-opcode.
//      16: P2InstD:        10 bit opcode, (7 bits + 000) and one D operand. S field serves as a sub-opcode.
//      17: P2InstRA:       7 bit opcode, R flag, and 20-bit A field
//      18: P2InstWRA:      5 bit opcode, 2-bit W flag, and 20-bit A field.
//      19: P2InstN:        5 bit opcode, 23-bit bit n field.
//
//  * the lowest bit of the opcode is between the C and I flags
//  ** the lowest bit of the opcode is between Z flag and D field.
//
// One instruction not captured here is MODCZ, cause I don't yet get how it works.
// Also aliases are not captured here, but I don't think they are needed, as the AsmParser should convert aliases to formal instructions
//
// Condition and effect bits (COND and C/Z) will be treated as extra immediate operands. To make this work, we can no longer set the Pattern
// field of the instruction and instead need to define explicit Pat instances in InstrInfo. Otherwise it will complain about not having
// a matching operand in the pattern. Mildly inconvenient but otherwise okay.
//
// we'll put cc before cz so that we always append a condition code operand in the asm parser, and then optionally append a cz operand.
// that way, if no cz is specified and no operand is appeneded, then the cz operand will just be 0, which is the same as not including it.
//
//===----------------------------------------------------------------------===//

def P2Cond : Operand<i8> {
    //let EncoderMethod = "encodeCondition";
    let PrintMethod = "printCondition";
}

def P2EffectOperand : AsmOperandClass {
    let Name = "Effect";
    let RenderMethod = "addImmOperands";
    let PredicateMethod = "isImm";
    let IsOptional = 1;
}

def P2Effect : Operand<i8> {
    //let EncoderMethod = "encodeEffect";
    let PrintMethod = "printEffect";
    let ParserMatchClass = P2EffectOperand;
}

defvar _ret_ = 0x0;
defvar if_nc_and_nz = 0x1;
defvar if_nc_and_z = 0x2;
defvar if_nc = 0x3;
defvar if_c_and_nz = 0x4;
defvar if_nz = 0x5;
defvar if_c_ne_z = 0x6;
defvar if_nc_or_nz = 0x7;
defvar if_c_and_z = 0x8;
defvar if_c_eq_z = 0x9;
defvar if_z = 0xa;
defvar if_nc_or_z = 0xb;
defvar if_c = 0xc;
defvar if_c_or_nz = 0xd;
defvar if_c_or_z = 0xe;
defvar always = 0xf;

// write C/Z flags, as well as W flag for addresses
defvar noeff = 0b00;
defvar wz = 0b01;
defvar wc = 0b10;
defvar wcz = 0b11;
defvar wpa = 0b00;
defvar wpb = 0b01;
defvar wptra = 0b10;
defvar wptrb = 0b11;

// generic instructions class
class P2Inst<bits<5> type, dag outs, dag inputs, string asmstr>: Instruction {
    // Inst and Size: for tablegen(... -gen-emitter) and
    // tablegen(... -gen-disassembler) in CMakeLists.txt
    bits<32> Inst;

    let Namespace = "P2";
    let Size = 4;

    // Top 4 bits are the 'effect' field for every instruction
    // the remaining bits will be defined by one of the instruction subclasses.

    let OutOperandList = outs;
    let InOperandList = inputs; //!con(inputs, (ins )); // concat the condition for all instructions

    let AsmString = asmstr;

    // TSFlags layout should be kept in sync with P2InstrInfo.h.
    let TSFlags{8-4} = type; // the style of instruction (i.e subclass below) for the code emitter to know the operand positions and such

    let DecoderNamespace = "P2";

    bits<32> SoftFail = 0;
}

// 7 bit opcode, CZI flags, and two 9-bit operands (D/S). S can be immediate
class P2InstCZIDS<bits<7> op, bits<1> i, dag outs, dag inputs, string asmstr> :
    P2Inst<1, outs, !con(inputs, (ins P2Cond:$cc, P2Effect:$cz)), !strconcat("$cc ", asmstr, "\t$cz")> {

    bits<9> d;
    bits<9> s;
    bits<4> cc;
    bits<2> cz;

    let Inst{31-28} = cc;
    let Inst{27-21} = op;
    let Inst{20-19} = cz;
    let Inst{18} = i;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 6 bit opcode, a 3-bit N value for nib-based instructions, I flag, and two operands (D/S). S can be imm
class P2Inst3NIDS<bits<6> op, bits<1> i, dag outs, dag inputs, string asmstr> :
    P2Inst<2, outs, !con(inputs, (ins P2Cond:$cc)), !strconcat("$cc ", asmstr)> {

    bits<9> d;
    bits<9> s;
    bits<3> n;
    bits<4> cc;

    let Inst{31-28} = cc;
    let Inst{27-22} = op;
    let Inst{21-19} = n;
    let Inst{18} = i;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 7 bit opcode, a 2-bit N value for byte-based instructios, I flag, and two operands (D/S). S can be imm
class P2Inst2NIDS<bits<7> op, bits<1> i, dag outs, dag inputs, string asmstr> :
    P2Inst<3, outs, !con(inputs, (ins P2Cond:$cc)), !strconcat("$cc ", asmstr)> {

    bits<9> d;
    bits<9> s;
    bits<2> n;
    bits<4> cc;

    let Inst{31-28} = cc;
    let Inst{27-21} = op;
    let Inst{20-19} = n;
    let Inst{18} = i;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 8 bit opcode, a 1-bit N value for short (2-byte) based, instructions, I flag, and two operands (D/S). S can be imm
class P2Inst1NIDS<bits<8> op, bits<1> i, dag outs, dag inputs, string asmstr> :
    P2Inst<4, outs, !con(inputs, (ins P2Cond:$cc)), !strconcat("$cc ", asmstr)> {

    bits<9> d;
    bits<9> s;
    bits<1> n;
    bits<4> cc;

    let Inst{31-28} = cc;
    let Inst{27-20} = op;
    let Inst{19} = n;
    let Inst{18} = i;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 9 bit opcode, I flag, and two operands (D/S). S can be imm.
class P2InstIDS<bits<9> op, bits<1> i, dag outs, dag inputs, string asmstr> :
    P2Inst<5, outs, !con(inputs, (ins P2Cond:$cc)), !strconcat("$cc ", asmstr)> {

    bits<9> d;
    bits<9> s;
    bits<4> cc;

    let Inst{31-28} = cc;
    let Inst{27-19} = op;
    let Inst{18} = i;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 8 bit opcode, ZI flags, and two operands (D/S). S can be imm.
class P2InstZIDS<bits<8> op, bits<1> i, dag outs, dag inputs, string asmstr> :
    P2Inst<6, outs, !con(inputs, (ins P2Cond:$cc, P2Effect:$z)), !strconcat("$cc ", asmstr, "\t$z")> {

    bits<9> d;
    bits<9> s;
    bits<4> cc;
    bits<1> z;

    let Inst{31-28} = cc;
    let Inst{27-20} = op;
    let Inst{19} = z;
    let Inst{18} = i;
    let Inst{17-9} = d;
    let Inst{8-0} = s;

}

// 8 bit opcode, CI flags, and two operands (D/S). S can be imm.
// the lowest bit of the opcode is between the C and I flags
class P2InstCIDS<bits<8> op, bits<1> i, dag outs, dag inputs, string asmstr> :
    P2Inst<7, outs, !con(inputs, (ins P2Cond:$cc, P2Effect:$c)), !strconcat("$cc ", asmstr, "\t$c")> {

    bits<9> d;
    bits<9> s;
    bits<4> cc;
    bits<2> c; // treat as 2 bits as if it's a cz flag, so that wc (which == 2), just sets this bit
               // preventing needing to determine the type of instruction for the flag.

    let Inst{31-28} = cc;
    let Inst{27-21} = op{7-1};
    let Inst{20} = c{1};
    let Inst{19} = op{0};
    let Inst{18} = i;
    let Inst{17-9} = d;
    let Inst{8-0} = s;

}

// 8 bit opcode, LI flags, and two operands (D/S). D and S can be imm.
class P2InstLIDS<bits<8> op, bits<2> li, dag outs, dag inputs, string asmstr> :
    P2Inst<8, outs, !con(inputs, (ins P2Cond:$cc)), !strconcat("$cc ", asmstr)> {

    bits<9> d;
    bits<9> s;
    bits<4> cc;

    let Inst{31-28} = cc;
    let Inst{27-20} = op;
    let Inst{19-18} = li;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 9 bit opcode, I flags, and one operand (S). S can be imm. D field serves as a sub-opcode.
class P2InstIS<bits<9> op, bits<9> d, bits<1> i, dag outs, dag inputs, string asmstr> :
    P2Inst<9, outs, !con(inputs, (ins P2Cond:$cc)), !strconcat("$cc ", asmstr)> {

    bits<9> s;
    bits<4> cc;

    let Inst{31-28} = cc;
    let Inst{27-19} = op;
    let Inst{18} = i;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 7 bit opcode, CLI flags, and two operands (D/S). D and S can be imm.
class P2InstCLIDS<bits<7> op, bits<2> li, dag outs, dag inputs, string asmstr> :
    P2Inst<10, outs, !con(inputs, (ins P2Cond:$cc, P2Effect:$c)), !strconcat("$cc ", asmstr, "\t$c")> {

    bits<9> d;
    bits<9> s;
    bits<4> cc;
    bits<2> c;

    let Inst{31-28} = cc;
    let Inst{27-21} = op;
    let Inst{20} = c{1};
    let Inst{19-18} = li;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 7 bit opcode, L flag, and one operand (D). D can be imm. S field serves as a sub-opcode.
class P2InstLD<bits<7> op, bits<9> s, bits<1> l, dag outs, dag inputs, string asmstr> :
    P2Inst<11, outs, !con(inputs, (ins P2Cond:$cc)), !strconcat("$cc ", asmstr)> {

    bits<9> d;
    bits<4> cc;

    let Inst{31-28} = cc;
    let Inst{27-21} = op;
    let Inst{20-19} = 0b00;
    let Inst{18} = l;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 7 bit opcode, CL flags, and one operand (D). D can be imm. S field serves as a sub-opcode.
class P2InstCLD<bits<7> op, bits<9> s, bits<1> l, dag outs, dag inputs, string asmstr> :
    P2Inst<12, outs, !con(inputs, (ins P2Cond:$cc, P2Effect:$c)), !strconcat("$cc ", asmstr, "\t$c")> {

    bits<9> d;
    bits<4> cc;
    bits<2> c;

    let Inst{31-28} = cc;
    let Inst{27-21} = op;
    let Inst{20} = c{1};
    let Inst{19} = 0b0;
    let Inst{18} = l;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 7 bit opcode, CZ flags, and one operand (D). S field serves as a sub-opcode.
class P2InstCZD<bits<7> op, bits<9> s, dag outs, dag inputs, string asmstr> :
    P2Inst<13, outs, !con(inputs, (ins P2Cond:$cc, P2Effect:$cz)), !strconcat("$cc ", asmstr, "\t$cz")> {

    bits<9> d;
    bits<4> cc;
    bits<2> cz;

    let Inst{31-28} = cc;
    let Inst{27-21} = op;
    let Inst{20-19} = cz;
    let Inst{18} = 0b0;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 8 bit opcode, CZ flags and no operands. D and S fields are used as a sub-opcode
// the lowest bit of the opcode is between Z flag and D field.
class P2InstCZ<bits<8> op, bits<9> d, bits<9> s, dag outs, dag inputs, string asmstr> :
    P2Inst<14, outs, !con(inputs, (ins P2Cond:$cc, P2Effect:$cz)), !strconcat("$cc ", asmstr, "\t$cz")> {

    bits<4> cc;
    bits<2> cz;

    let Inst{31-28} = cc;
    let Inst{27-21} = op{7-1};
    let Inst{20-19} = cz;
    let Inst{18} = op{0};
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 7 bit opcode, CZL flags, and one operand (D). D can be imm. S field serves as a sub-opcode.
class P2InstCZLD<bits<7> op, bits<9> s, bits<1> l, dag outs, dag inputs, string asmstr> :
    P2Inst<15, outs, !con(inputs, (ins P2Cond:$cc, P2Effect:$cz)), !strconcat("$cc ", asmstr, "\t$cz")> {

    bits<9> d;
    bits<4> cc;
    bits<2> cz;

    let Inst{31-28} = cc;
    let Inst{27-21} = op;
    let Inst{20-19} = cz;
    let Inst{18} = l;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 10 bit opcode, (7 bits + 000) and one D operand. S field serves as a sub-opcode.
class P2InstD<bits<7> op, bits<9> s, dag outs, dag inputs, string asmstr> :
    P2Inst<16, outs, !con(inputs, (ins P2Cond:$cc)), !strconcat("$cc ", asmstr)> {

    bits<9> d;
    bits<4> cc;

    let Inst{31-28} = cc;
    let Inst{27-21} = op;
    let Inst{20-18} = 0b000;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 7 bit opcode, R flag, and 20-bit A field
class P2InstRA<bits<7> op, bits<1> r, dag outs, dag inputs, string asmstr> :
    P2Inst<17, outs, !con(inputs, (ins P2Cond:$cc)), !strconcat("$cc ", asmstr)> {

    bits<20> a;
    bits<4> cc;

    let Inst{31-28} = cc;
    let Inst{27-21} = op;
    let Inst{20} = r;
    let Inst{19-0} = a;
}

// 5 bit opcode, 2-bit W flag, and 20-bit A field.
class P2InstWRA<bits<5> op, bits<1> r, dag outs, dag inputs, string asmstr> :
    P2Inst<18, outs, !con(inputs, (ins P2Cond:$cc, P2Effect:$w)), !strconcat("$cc ", asmstr)> {

    bits<20> a;
    bits<4> cc;
    bits<2> w;

    let Inst{31-28} = cc;
    let Inst{27-23} = op;
    let Inst{22-21} = w;
    let Inst{20} = r;
    let Inst{19-0} = a;
}

// 5 bit opcode, 23-bit bit n field.
class P2InstN<bits<5> op, dag outs, dag inputs, string asmstr> :
    P2Inst<19, outs, !con(inputs, (ins P2Cond:$cc)), !strconcat("$cc ", asmstr)> {

    bits<23> n;
    bits<4> cc;

    let Inst{31-28} = cc;
    let Inst{27-23} = op;
    let Inst{22-0} = n;
}

class P2InstNOP : P2Inst<0, (outs), (ins), "nop"> {
    let Inst{31-0} = 0b00000000000000000000000000000000;
}

class Pseudo<dag outs, dag ins, string asmstr> : P2Inst<0, outs, ins, !strconcat("PSEUDO ", asmstr)> {
    let isPseudo = 1;
    let isCodeGenOnly = 1;
}
