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
//   EEEE | OOOOOOO | CZI | DDDDDDDDD | SSSSSSSSS || most math and logic commands. Single operand instructions have specific S values and I = 0. only S can be immediate
//   EEEE | OOOOOOO | CZL | DDDDDDDDD | SSSSSSSSS || single operand where D can be immediate, and can write both C and Z
//   EEEE | OOOOOOO | CLI | DDDDDDDDD | SSSSSSSSS || (notice that L can move!) instructions where both D and S can be immediates
//   EEEE | OOOOOON | NNI | DDDDDDDDD | SSSSSSSSS || special instructions that sets word or nibbles defined by N
//   EEEE | OOOOOOO | RAA | AAAAAAAAA | AAAAAAAAA || call/jump instructions that modify the entire PC (for hub exec modes)
//   EEEE | OOOOOnn | nnn | nnnnnnnnn | nnnnnnnnn || AUGS/AUGD
//
// E = Effects. conditions stuck on the instruction to determine if it should be executed based on status word flags
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
//===----------------------------------------------------------------------===//

// effect string/encoding pairing
class P2Cond<string str, bits<4> cond> {
    field bits<4> condition = cond;
    field string condition_str = str;
}

// keep this in sync with the Instruction printer
def _ret_           : P2Cond<"_ret_", 0b0000>;
def if_nc_and_nz    : P2Cond<"if_nc_and_nz", 0b0001>;
def if_nc_and_z     : P2Cond<"if_nc_and_z", 0b0010>;
def if_nc           : P2Cond<"if_nc", 0b0011>;
def if_c_and_nz     : P2Cond<"if_c_and_nz", 0b0100>;
def if_nz           : P2Cond<"if_nz", 0b0101>;
def if_c_ne_z       : P2Cond<"if_c_ne_z", 0b0110>;
def if_nc_or_nz     : P2Cond<"if_nc_or_nz", 0b0111>;
def if_c_and_z      : P2Cond<"if_c_and_z", 0b1000>;
def if_c_eq_z       : P2Cond<"if_c_eq_z", 0b1001>;
def if_z            : P2Cond<"if_z", 0b1010>;
def if_nc_or_z      : P2Cond<"if_nc_or_z", 0b1011>;
def if_c            : P2Cond<"if_c", 0b1100>;
def if_c_or_nz      : P2Cond<"if_c_or_nz", 0b1101>;
def if_c_or_z       : P2Cond<"if_c_or_z", 0b1110>;
def always          : P2Cond<"", 0b1111>;

// generic instructions class
class P2Inst<P2Cond cond, bits<5> type, dag outs, dag ins, string asmstr, list<dag> pattern>: Instruction {
    // Inst and Size: for tablegen(... -gen-emitter) and
    // tablegen(... -gen-disassembler) in CMakeLists.txt
    field bits<32> Inst;

    let Namespace = "P2";
    let Size = 4;

    bits<4> condition; // this is present in all instructions

    let condition = cond.condition;

    // Top 4 bits are the 'effect' field for every instruction
    // the remaining bits will be defined by one of the instruction subclasses.
    let Inst{31-28} = condition;

    let OutOperandList = outs;
    let InOperandList = ins;

    let AsmString = !strconcat(cond.condition_str, "\t", asmstr);
    let Pattern = pattern;

    // TSFlags layout should be kept in sync with P2InstrInfo.h.
    let TSFlags{3-0} = condition; // put the effect bits into the TSFlags field for use by the instruction printer
    let TSFlags{8-4} = type; // the style of instruction (i.e subclass below) for the code emitter to know the operand positions and such

    let DecoderNamespace = "P2";

    field bits<32> SoftFail = 0;
}

// 7 bit opcode, CZI flags, and two 9-bit operands (D/S). S can be immediate
class P2InstCZIDS<bits<7> op, bits<3> czi, P2Cond cond, dag outs, dag ins, string asmstr, list<dag> pattern> :
    P2Inst<cond, 1, outs, ins, asmstr, pattern> {
    bits<9> d;
    bits<9> s;

    let Inst{27-21} = op;
    let Inst{20-18} = czi;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 6 bit opcode, a 3-bit N value for nib-based instructions, I flag, and two operands (D/S). S can be imm
class P2Inst3NIDS<bits<6> op, bits<1> i, P2Cond cond, dag outs, dag ins, string asmstr, list<dag> pattern> :
    P2Inst<cond, 2, outs, ins, asmstr, pattern> {

    bits<9> d;
    bits<9> s;
    bits<3> n;

    let Inst{27-22} = op;
    let Inst{21-19} = n;
    let Inst{18} = i;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 7 bit opcode, a 2-bit N value for byte-based instructios, I flag, and two operands (D/S). S can be imm
class P2Inst2NIDS<bits<7> op, bits<1> i, P2Cond cond, dag outs, dag ins, string asmstr, list<dag> pattern> :
    P2Inst<cond, 3, outs, ins, asmstr, pattern> {

    bits<9> d;
    bits<9> s;
    bits<2> n;

    let Inst{27-21} = op;
    let Inst{20-19} = n;
    let Inst{18} = i;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 8 bit opcode, a 1-bit N value for short (2-byte) based, instructions, I flag, and two operands (D/S). S can be imm
class P2Inst1NIDS<bits<8> op, bits<1> i, P2Cond cond, dag outs, dag ins, string asmstr, list<dag> pattern> :
    P2Inst<cond, 4, outs, ins, asmstr, pattern> {

    bits<9> d;
    bits<9> s;
    bits<1> n;

    let Inst{27-20} = op;
    let Inst{19} = n;
    let Inst{18} = i;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 9 bit opcode, I flag, and two operands (D/S). S can be imm.
class P2InstIDS<bits<9> op, bits<1> i, P2Cond cond, dag outs, dag ins, string asmstr, list<dag> pattern> :
    P2Inst<cond, 5, outs, ins, asmstr, pattern> {

    bits<9> d;
    bits<9> s;

    let Inst{27-19} = op;
    let Inst{18} = i;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 8 bit opcode, ZI flags, and two operands (D/S). S can be imm.
class P2InstZIDS<bits<8> op, bits<2> zi, P2Cond cond, dag outs, dag ins, string asmstr, list<dag> pattern> :
    P2Inst<cond, 6, outs, ins, asmstr, pattern> {

    bits<9> d;
    bits<9> s;

    let Inst{27-20} = op;
    let Inst{19-18} = zi;
    let Inst{17-9} = d;
    let Inst{8-0} = s;

}

// 8 bit opcode, CI flags, and two operands (D/S). S can be imm.
// the lowest bit of the opcode is between the C and I flags
class P2InstCIDS<bits<8> op, bits<2> ci, P2Cond cond, dag outs, dag ins, string asmstr, list<dag> pattern> :
    P2Inst<cond, 7, outs, ins, asmstr, pattern> {

    bits<9> d;
    bits<9> s;

    let Inst{27-21} = op{7-1};
    let Inst{20} = ci{1};
    let Inst{19} = op{0};
    let Inst{18} = ci{0};
    let Inst{17-9} = d;
    let Inst{8-0} = s;

}

// 8 bit opcode, LI flags, and two operands (D/S). D and S can be imm.
class P2InstLIDS<bits<8> op, bits<2> li, P2Cond cond, dag outs, dag ins, string asmstr, list<dag> pattern> :
    P2Inst<cond, 8, outs, ins, asmstr, pattern> {

    bits<9> d;
    bits<9> s;

    let Inst{27-20} = op;
    let Inst{19-18} = li;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 9 bit opcode, I flags, and one operand (S). S can be imm. D field serves as a sub-opcode.
class P2InstIS<bits<9> op, bits<9> d, bits<1> i, P2Cond cond, dag outs, dag ins, string asmstr, list<dag> pattern> :
    P2Inst<cond, 9, outs, ins, asmstr, pattern> {

    bits<9> s;

    let Inst{27-19} = op;
    let Inst{18} = i;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 7 bit opcode, CLI flags, and two operands (D/S). D and S can be imm.
class P2InstCLIDS<bits<7> op, bits<3> cli, P2Cond cond, dag outs, dag ins, string asmstr, list<dag> pattern> :
    P2Inst<cond, 10, outs, ins, asmstr, pattern> {

    bits<9> d;
    bits<9> s;

    let Inst{27-21} = op;
    let Inst{20-18} = cli;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 7 bit opcode, L flag, and one operand (D). D can be imm. S field serves as a sub-opcode.
class P2InstLD<bits<7> op, bits<9> s, bits<1> l, P2Cond cond, dag outs, dag ins, string asmstr, list<dag> pattern> :
    P2Inst<cond, 11, outs, ins, asmstr, pattern> {

    bits<9> d;

    let Inst{27-21} = op;
    let Inst{18} = l;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 7 bit opcode, CL flags, and one operand (D). D can be imm. S field serves as a sub-opcode.
class P2InstCLD<bits<7> op, bits<9> s, bits<2> cl, P2Cond cond, dag outs, dag ins, string asmstr, list<dag> pattern> :
    P2Inst<cond, 12, outs, ins, asmstr, pattern> {

    bits<9> d;

    let Inst{27-21} = op;
    let Inst{20} = cl{0};
    let Inst{19} = 0b0;
    let Inst{18} = cl{1};
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 7 bit opcode, CZ flags, and one operand (D). S field serves as a sub-opcode.
class P2InstCZD<bits<7> op, bits<9> s, bits<2> cz, P2Cond cond, dag outs, dag ins, string asmstr, list<dag> pattern> :
    P2Inst<cond, 13, outs, ins, asmstr, pattern> {

    bits<9> d;

    let Inst{27-21} = op;
    let Inst{20-19} = cz;
    let Inst{18} = 0b0;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 8 bit opcode, CZ flags and no operands. D and S fields are used as a sub-opcode
// the lowest bit of the opcode is between Z flag and D field.
class P2InstCZ<bits<8> op, bits<9> d, bits<9> s, bits<2> cz, P2Cond cond, dag outs, dag ins, string asmstr, list<dag> pattern> :
    P2Inst<cond, 14, outs, ins, asmstr, pattern> {

    let Inst{27-21} = op{7-1};
    let Inst{20-19} = cz;
    let Inst{18} = op{0};
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 7 bit opcode, CZL flags, and one operand (D). D can be imm. S field serves as a sub-opcode.
class P2InstCZLD<bits<7> op, bits<9> s, bits<3> czl, P2Cond cond, dag outs, dag ins, string asmstr, list<dag> pattern> :
    P2Inst<cond, 15, outs, ins, asmstr, pattern> {

    bits<9> d;

    let Inst{27-21} = op;
    let Inst{20-18} = czl;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 10 bit opcode, (7 bits + 000) and one D operand. S field serves as a sub-opcode.
class P2InstD<bits<7> op, bits<9> s, P2Cond cond, dag outs, dag ins, string asmstr, list<dag> pattern> :
    P2Inst<cond, 16, outs, ins, asmstr, pattern> {

    bits<9> d;

    let Inst{27-21} = op;
    let Inst{20-18} = 0b000;
    let Inst{17-9} = d;
    let Inst{8-0} = s;
}

// 7 bit opcode, R flag, and 20-bit A field
class P2InstRA<bits<7> op, bits<1> r, P2Cond cond, dag outs, dag ins, string asmstr, list<dag> pattern> :
    P2Inst<cond, 17, outs, ins, asmstr, pattern> {

    bits<20> a;

    let Inst{27-21} = op;
    let Inst{20} = r;
    let Inst{19-0} = a;
}

// 5 bit opcode, 2-bit W flag, and 20-bit A field.
class P2InstWRA<bits<5> op, bits<3> wr, P2Cond cond, dag outs, dag ins, string asmstr, list<dag> pattern> :
    P2Inst<cond, 18, outs, ins, asmstr, pattern> {

    bits<20> a;

    let Inst{27-23} = op;
    let Inst{22-20} = wr;
    let Inst{19-0} = a;
}

// 5 bit opcode, 23-bit bit n field.
class P2InstN<bits<5> op, P2Cond cond, dag outs, dag ins, string asmstr, list<dag> pattern> :
    P2Inst<cond, 19, outs, ins, asmstr, pattern> {

    bits<23> n;

    let Inst{27-23} = op;
    let Inst{22-0} = n;
}

// 7 bit opcode, CZI flags, and two 9-bit operands (D/S). S can be immediate
class P2InstNOP : P2Inst<_ret_, 0, (outs), (ins), "nop ", []> {
    let Inst{27-21} = 0b0000000;
    let Inst{20-18} = 0b000;
    let Inst{17-9} = 0b000000000;
    let Inst{8-0} = 0b000000000;
}

// branch instruction that should execute based on the effects
// ins must have a "$target" operand and it should be the first one in the list
class P2BranchInst<dag outs, dag ins, P2Cond cond, list<dag> pattern> :
    P2InstRA<0b1101100, 0b1, cond, outs, ins, "jmp #$target", pattern> {

    let isBarrier = 1;
    let isBranch = 1;
    let isTerminator = 1;
}

class Pseudo<dag outs, dag ins, string asmstr, list<dag> pattern> : P2Inst<always, 0, outs, ins, !strconcat(asmstr, " PSEUDO"), pattern> {
    let Pattern = pattern;
    let isPseudo = 1;
    let isCodeGenOnly = 1;
}
