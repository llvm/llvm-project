//===- ARCInstrFormats.td - ARC Instruction Formats --------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//

class Encoding64 {
  field bits<64> Inst;
  field bits<64> SoftFail = 0;
}

// Address operands
def immU6 : Operand<i32>, PatLeaf<(imm), [{
    return isUInt<6>(N->getSExtValue()); }]> {
}

def immS12 : Operand<i32>, PatLeaf<(imm), [{
    return isInt<12>(N->getSExtValue()); }]> {
  let DecoderMethod = "DecodeS12Operand";
}

def immS9 : Operand<i32>, PatLeaf<(imm), [{
    return isInt<9>(N->getSExtValue()); }]> {
  let DecoderMethod = "DecodeS9Operand";
}

def MEMii : Operand<i32> {
  let MIOperandInfo = (ops i32imm, i32imm);
}

def MEMrs9 : Operand<iAny> {
  let MIOperandInfo = (ops GPR32:$B, immS9:$S9);
  let PrintMethod = "printMemOperandRI";
  let DecoderMethod = "DecodeMEMrs9";
}

def MEMrlimm : Operand<iAny> {
  let MIOperandInfo = (ops GPR32:$B, i32imm:$LImm);
  let PrintMethod = "printMemOperandRI";
  let DecoderMethod = "DecodeMEMrlimm";
}

class InstARC<int sz, dag outs, dag ins, string asmstr, list<dag> pattern>
    : Instruction, Encoding64 {

  let Namespace = "ARC";
  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString   = asmstr;
  let Pattern = pattern;
  let Size = sz;
}

// ARC pseudo instructions format
class PseudoInstARC<dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstARC<0, outs, ins, asmstr, pattern> {
  let isPseudo = 1;
}

//===----------------------------------------------------------------------===//
// Instruction formats
//===----------------------------------------------------------------------===//

// All 32-bit ARC instructions have a 5-bit "major" opcode class designator
// in bits 27-31.   
// 
// Some general naming conventions:
// N  - Delay Slot bit.  ARC v2 branch instructions have an optional delay slot
//      which is encoded with this bit.  When set, a delay slot exists.
// cc - Condition code.
// SX - Signed X-bit immediate.
// UX - Unsigned X-bit immediate.
// 
// [ABC] - 32-bit register operand.  These are 6-bit fields.  This encodes the 
//         standard 32 general purpose registers, and allows use of additional
//         (extension) registers.  This also encodes an instruction that uses
//         a 32-bit Long Immediate (LImm), using 0x3e==62 as the field value.
//         This makes 32-bit format instructions with Long Immediates
//         64-bit instructions, with the Long Immediate in bits 32-63.
// A - Inst[5-0] = A[5-0], when the format has A.  A is always a register.
// B - Inst[14-12] = B[5-3], Inst[26-24] = B[2-0], when the format has B.
//     B is always a register.
// C - Inst[11-6] = C[5-0], when the format has C.  C can either be a register,
//     or a 6-bit unsigned immediate (immU6), depending on the format.
// F - Many instructions specify a flag bit. When set, the result of these
//     instructions will set the ZNCV flags of the STATUS32 register
//     (Zero/Negative/Carry/oVerflow).

// Branch Instructions.
class F32_BR<bits<5> major, dag outs, dag ins, bit b16, string asmstr,
             list<dag> pattern> :
  InstARC<4, outs, ins, asmstr, pattern> {
  bit N;

  let Inst{31-27} = major;
  let Inst{16} = b16;
  let Inst{5} = N;
}

class F32_BR_COND<bits<5> major, dag outs, dag ins, bit b16, string asmstr,
                  list<dag> pattern> :
  F32_BR<major, outs, ins, b16, asmstr, pattern> {
  bits<21> S21; // 2-byte aligned 21-bit byte-offset.
  bits<5> cc; 
  let Inst{26-18} = S21{10-2};
  let Inst{15-6} = S21{20-11};
  let Inst{4-0} = cc;
}

class F32_BR_UCOND_FAR<bits<5> major, dag outs, dag ins, bit b16, string asmstr,
                       list<dag> pattern> :
  F32_BR<major, outs, ins, b16, asmstr, pattern> {
  bits<25> S25; // 2-byte aligned 25-bit byte-offset.
  let Inst{26-18} = S25{10-2};
  let Inst{15-6} = S25{20-11};
  let Inst{4} = 0;
  let Inst{3-0} = S25{24-21};
}

class F32_BR0_COND<dag outs, dag ins, string asmstr, list<dag> pat> :
  F32_BR_COND<0b00000, outs, ins, 0, asmstr, pat> {
  let Inst{17} = S21{1};
}

// Branch targets are 2-byte aligned, so S25[0] is implied 0.
// |26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|9|8|7|6|5|4|3|2|1|0   |
// |S25[10-1]                    | 1|S25[20-11]               |N|0|S25[24-21]|
class F32_BR0_UCOND_FAR<dag outs, dag ins, string asmstr, list<dag> pat> :
  F32_BR_UCOND_FAR<0b00000, outs, ins, 1, asmstr, pat> {
  let Inst{17} = S25{1};
}

// BL targets (functions) are 4-byte aligned, so S25[1-0] = 0b00
// |26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|9|8|7|6|5|4|3|2|1|0   |
// |S25[10-2]                 | 1| 0|S25[20-11]               |N|0|S25[24-21]|
class F32_BR1_BL_UCOND_FAR<dag outs, dag ins, string asmstr, list<dag> pat> :
  F32_BR_UCOND_FAR<0b00001, outs, ins, 0, asmstr, pat> {
  let Inst{17} = 1;
}

// BLcc targets have 21 bit range, and are 4-byte aligned.
// |26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|9|8|7|6|5|4|3|2|1|0|
// |S25[10-2]                 | 0| 0|S25[20-11]               |N|0|cc     |
class F32_BR1_BL_COND<dag outs, dag ins, string asmstr, list<dag> pat> :
  F32_BR_COND<0b00001, outs, ins, 0, asmstr, pat> {
  let Inst{17} = 0;
}


// BRcc targets have limited 9-bit range.  These are for compare and branch
// in single instruction.  Their targets are 2-byte aligned.  They also use
// a different (3-bit) set of condition codes.
// |26|25|24|23|22|21|20|19|18|17|16|15   |14|13|12|11|10|9|8|7|6|5|4|3|2|1|0|
// |B[2-0]  |S9[7-1]             | 1|S9[8]|B[5-3]  |C            |N|u|0|cc   |
class F32_BR1_BCC<dag outs, dag ins, string asmstr, bit IsU6,
                  list<dag> pattern> :
  InstARC<4, outs, ins, asmstr, pattern> {

  bits<3> cc;
  bits<6> B;
  bits<6> C;
  bit N;
  bits<9> S9; // 2-byte aligned 9-bit byte-offset.

  let Inst{31-27} = 0b00001;
  let Inst{26-24} = B{2-0};
  let Inst{23-17} = S9{7-1};
  let Inst{16} = 1;
  let Inst{15} = S9{8};
  let Inst{14-12} = B{5-3};
  let Inst{11-6} = C;
  let Inst{5} = N;
  let Inst{4} = IsU6;
  let Inst{3} = 0;
  let Inst{2-0} = cc;
}

// General operations instructions.
// Single Operand Instructions.  Inst[5-0] specifies the specific operation
// for this format.
// |26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|9|8|7|6|5|4|3|2|1|0|
// |B[2-0]  | 0| 0| 1| 0| 1| 1| 1| 1| F|B[5-3]  |C            |subop      |
class F32_SOP_RR<bits<5> major, bits<6> subop, bit F, dag outs, dag ins,
                 string asmstr, list<dag> pattern> :
  InstARC<4, outs, ins, asmstr, pattern> {

  bits<6> C;
  bits<6> B;

  let Inst{31-27} = major;
  let Inst{26-24} = B{2-0};
  let Inst{23-22} = 0b00;
  let Inst{21-16} = 0b101111;
  let Inst{15} = F;
  let Inst{14-12} = B{5-3};
  let Inst{11-6} = C;
  let Inst{5-0} = subop;
}

// Dual Operand Instructions.  Inst[21-16] specifies the specific operation
// for this format.

// 3-register Dual Operand instruction.
// |26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|9|8|7|6|5|4|3|2|1|0|
// |B[2-0]  | 0| 0|            subop| F|B[5-3]  |C            |A          |
class F32_DOP_RR<bits<5> major, bits<6> subop, bit F, dag outs, dag ins,
                 string asmstr, list<dag> pattern> :
  InstARC<4, outs, ins, asmstr, pattern> {
  bits<6> C;
  bits<6> B;
  bits<6> A;

  let Inst{31-27} = major;
  let Inst{26-24} = B{2-0};
  let Inst{23-22} = 0b00;
  let Inst{21-16} = subop;
  let Inst{15} = F;
  let Inst{14-12} = B{5-3};
  let Inst{11-6} = C;
  let Inst{5-0} = A;
}

// Conditional Dual Operand instruction.  This instruction uses B as the
// first 2 operands (i.e, add.cc B, B, C).
// |26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|9|8|7|6|5|4|3|2|1|0|
// |B[2-0]  | 1| 1|            subop| F|B[5-3]  |C            |A          |
class F32_DOP_CC_RR<bits<5> major, bits<6> subop, bit F, dag outs, dag ins,
                    string asmstr, list<dag> pattern> :
  InstARC<4, outs, ins, asmstr, pattern> {
  bits<5> cc;
  bits<6> C;
  bits<6> B;

  let Inst{31-27} = major;
  let Inst{26-24} = B{2-0};
  let Inst{23-22} = 0b11;
  let Inst{21-16} = subop;
  let Inst{15} = F;
  let Inst{14-12} = B{5-3};
  let Inst{11-6} = C;
  let Inst{5} = 0;
  let Inst{4-0} = cc;
}


// 2-register, unsigned 6-bit immediate Dual Operand instruction.
// |26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|9|8|7|6|5|4|3|2|1|0|
// |B[2-0]  | 0| 1|            subop| F|B[5-3]  |U6           |A          |
class F32_DOP_RU6<bits<5> major, bits<6> subop, bit F, dag outs, dag ins,
                  string asmstr, list<dag> pattern> :
  InstARC<4, outs, ins, asmstr, pattern> {
  bits<6> U6;
  bits<6> B;
  bits<6> A;

  let Inst{31-27} = major;
  let Inst{26-24} = B{2-0};
  let Inst{23-22} = 0b01;
  let Inst{21-16} = subop;
  let Inst{15} = F;
  let Inst{14-12} = B{5-3};
  let Inst{11-6} = U6;
  let Inst{5-0} = A;
}

// 2-register, signed 12-bit immediate Dual Operand instruction.
// This instruction uses B as the first 2 operands (i.e., add B, B, -128). 
// |26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|9|8|7|6|5|4|3|2|1|0|
// |B[2-0]  | 1| 0|            subop| F|B[5-3]  |S12[5-0]     |S12[11-6]  |
class F32_DOP_RS12<bits<5> major, bits<6> subop, bit F, dag outs, dag ins,
                   string asmstr, list<dag> pattern> :
  InstARC<4, outs, ins, asmstr, pattern> {
  bits<6> B;
  bits<12> S12;
  
  let Inst{31-27} = major;
  let Inst{26-24} = B{2-0};
  let Inst{23-22} = 0b10;
  let Inst{21-16} = subop;
  let Inst{15} = F;
  let Inst{14-12} = B{5-3};
  let Inst{11-6} = S12{5-0};
  let Inst{5-0} = S12{11-6};
}

// 2-register, 32-bit immediate (LImm) Dual Operand instruction.
// This instruction has the 32-bit immediate in bits 32-63, and
// 62 in the C register operand slot, but is otherwise F32_DOP_RR.
class F32_DOP_RLIMM<bits<5> major, bits<6> subop, bit F, dag outs, dag ins,
                    string asmstr, list<dag> pattern> :
  InstARC<8, outs, ins, asmstr, pattern> {
  bits<6> B;
  bits<6> A;
  bits<32> LImm;

  let Inst{63-32} = LImm;
  let Inst{31-27} = major;
  let Inst{26-24} = B{2-0};
  let Inst{23-22} = 0b00;
  let Inst{21-16} = subop;
  let Inst{15} = F;
  let Inst{14-12} = B{5-3};
  let Inst{11-6} = 0b111110;
  let Inst{5-0} = A;
}


// Load and store instructions.
// In addition to the previous naming conventions, load and store instructions
// have:
// di - Uncached bit.  When set, loads/stores bypass the cache and access
//      memory directly.
// aa - Incrementing mode.  Loads and stores can write-back address pre- or
//      post- memory operation.
// zz - Memory size (can be 8/16/32 bit load/store).
//  x - Sign-extending.  When set, short loads can be sign-extended to 32-bits.
// Loads and Stores support different memory addressing modes:
// Base Register + Signed 9-bit Immediate: Both Load/Store.
// LImm: Both Load/Store (Load/Store from a fixed 32-bit address).
// Register + Register: Load Only.
// Register + LImm: Load Only.

// Register + S9 Load. (B + S9)
// |26|25|24|23|22|21|20|19|18|17|16|15   |14|13|12|11|10|9|8|7|6|5|4|3|2|1|0|
// |B[2-0]  |S9[7-0]                |S9[8]|B[5-3]  |di|aa  |zz |x|A          |
class F32_LD_RS9<bit x, bits<2> aa, bit di, bits<2> zz, dag outs, dag ins,
                 string asmstr, list<dag> pattern> :
  InstARC<4, outs, ins, asmstr, pattern> {
  bits<6> B;
  bits<6> A;
  bits<9> S9;

  let Inst{31-27} = 0b00010;
  let Inst{26-24} = B{2-0};
  let Inst{23-16} = S9{7-0};
  let Inst{15} = S9{8};
  let Inst{14-12} = B{5-3};
  let Inst{11} = di;
  let Inst{10-9} = aa;
  let Inst{8-7} = zz;
  let Inst{6} = x;
  let Inst{5-0} = A;
}

class F32_LD_ADDR<bit x, bits<2> aa, bit di, bits<2> zz, dag outs, dag ins,
                  string asmstr, list<dag> pattern> :
  F32_LD_RS9<x, aa, di, zz, outs, ins, asmstr, pattern> {
  bits<15> addr;

  let B = addr{14-9};
  let S9 = addr{8-0};
}


// LImm Load.  The 32-bit immediate address is in Inst[63-32].
// |26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|9|8|7|6|5|4|3|2|1|0|
// | 1| 1| 0| 0                        | 1| 1| 1|di| 0|0|zz |x|A          |
class F32_LD_LIMM<bit x, bit di, bits<2> zz, dag outs, dag ins,
                  string asmstr, list<dag> pattern> :
  InstARC<8, outs, ins, asmstr, pattern> {
  bits<6> LImmReg = 0b111110;
  bits<6> A;
  bits<32> LImm;

  let Inst{63-32} = LImm;
  let Inst{31-27} = 0b00010;
  let Inst{26-24} = LImmReg{2-0};
  let Inst{23-15} = 0;
  let Inst{14-12} = LImmReg{5-3};
  let Inst{11} = di;
  let Inst{10-9} = 0;
  let Inst{8-7} = zz;
  let Inst{6} = x;
  let Inst{5-0} = A;
  let DecoderMethod = "DecodeLdLImmInstruction";
}

// Register + LImm load.  The 32-bit immediate address is in Inst[63-32].
// |26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|9|8|7|6|5|4|3|2|1|0|
// |B[2-0]  |aa   | 1| 1| 0|zz   | x|di|B[5-3]  | 1| 1|1|1|1|0|A          |
class F32_LD_RLIMM<bit x, bits<2> aa, bit di, bits<2> zz, dag outs, dag ins,
                   string asmstr, list<dag> pattern> :
  InstARC<8, outs, ins, asmstr, pattern> {
  bits<6> LImmReg = 0b111110;
  bits<32> LImm;
  bits<6> B;
  bits<6> A;
  bits<38> addr;
  let B = addr{37-32};
  let LImm = addr{31-0};

  let Inst{63-32} = LImm;
  let Inst{31-27} = 0b00100;
  let Inst{26-24} = B{2-0};
  let Inst{23-22} = aa;
  let Inst{21-19} = 0b110;
  let Inst{18-17} = zz;
  let Inst{16} = x;
  let Inst{15} = di;
  let Inst{14-12} = B{5-3};
  let Inst{11-6} = LImmReg;
  let Inst{5-0} = A;
  let DecoderMethod = "DecodeLdRLImmInstruction";
}

// Register + S9 Store. (B + S9)
// |26|25|24|23|22|21|20|19|18|17|16|15   |14|13|12|11|10|9|8|7|6|5 |4|3|2|1|0|
// |B[2-0]  |S9[7-0]                |S9[8]|B[5-3]  |C            |di|aa |zz |0|
class F32_ST_RS9<bits<2> aa, bit di, bits<2> zz, dag outs, dag ins,
                 string asmstr, list<dag> pattern> :
  InstARC<4, outs, ins, asmstr, pattern> {
  bits<6> B;
  bits<6> C;
  bits<9> S9;

  let Inst{31-27} = 0b00011;
  let Inst{26-24} = B{2-0};
  let Inst{23-16} = S9{7-0};
  let Inst{15} = S9{8};
  let Inst{14-12} = B{5-3};
  let Inst{11-6} = C;
  let Inst{5} = di;
  let Inst{4-3} = aa;
  let Inst{2-1} = zz;
  let Inst{0} = 0;
}

class F32_ST_ADDR<bits<2> aa, bit di, bits<2> zz, dag outs, dag ins,
                  string asmstr, list<dag> pattern> :
  F32_ST_RS9<aa, di, zz, outs, ins, asmstr, pattern> {
  bits<15> addr;

  let B = addr{14-9};
  let S9 = addr{8-0};
}

// LImm Store.
// |26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|9|8|7|6|5 |4|3|2|1|0|
// | 1| 1| 0| 0                        | 1| 1| 1|C            |di|0|0|zz |0|
class F32_ST_LIMM<bit di, bits<2> zz, dag outs, dag ins,
                  string asmstr, list<dag> pattern> :
  InstARC<8, outs, ins, asmstr, pattern> {
  bits<6> LImmReg = 0b111110;
  bits<6> C;
  bits<32> LImm;

  let Inst{63-32} = LImm;
  let Inst{31-27} = 0b00011;
  let Inst{26-24} = LImmReg{2-0};
  let Inst{23-15} = 0;
  let Inst{14-12} = LImmReg{5-3};
  let Inst{11-6} = C;
  let Inst{5} = di;
  let Inst{4-3} = 0;
  let Inst{2-1} = zz;
  let Inst{0} = 0;
  let DecoderMethod = "DecodeStLImmInstruction";
}

// Special types for different instruction operands.
def cmovpred : Operand<i32>, PredicateOp,
               ComplexPattern<i32, 2, "SelectCMOVPred"> {
  let MIOperandInfo = (ops i32imm, i32imm);
  let PrintMethod = "printPredicateOperand";
}

def ccond : Operand<i32> {
  let MIOperandInfo = (ops i32imm);
  let PrintMethod = "printPredicateOperand";
}

def brccond : Operand<i32> {
  let MIOperandInfo = (ops i32imm);
  let PrintMethod = "printBRCCPredicateOperand";
}

// Branch targets of different offset sizes.
def btarget : Operand<OtherVT> {
  let OperandType = "OPERAND_PCREL";
}

def btargetS9 : Operand<OtherVT> {
  let OperandType = "OPERAND_PCREL";
  let DecoderMethod = "DecodeBranchTargetS9";
}

def btargetS21 : Operand<OtherVT> {
  let OperandType = "OPERAND_PCREL";
  let DecoderMethod = "DecodeBranchTargetS21";
}

def btargetS25 : Operand<OtherVT> {
  let OperandType = "OPERAND_PCREL";
  let DecoderMethod = "DecodeBranchTargetS25";
}

def calltargetS25: Operand<i32> {
  let OperandType = "OPERAND_PCREL";
  let DecoderMethod = "DecodeBranchTargetS25";
}

