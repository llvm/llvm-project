//===-- EZHInstrFormats.td - EZH Instruction Formats ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

class EZHInst<dag outs, dag ins, string asmstr, list<dag> pattern>
    : Instruction {
  field bits<32> Inst;

  let Namespace = "EZH";
  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;

  let Size = 4;

  // Basic fields
  bits<5> Opcode;
  bits<4> Cond = 0;
  bits<1> S = 0; // Set flags

  let Inst{4-0} = Opcode;
  let Inst{8-5} = Cond;
  let Inst{9}   = S;

  let Predicates = [];
}

// Format for ALU instructions (Register-Register)
class EZHInstALU<bits<5> op, bits<1> invert, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = op;
  bits<4> Rd;
  bits<4> Rs1;
  bits<4> Rs2;
  let hasSideEffects = 1;

  let Inst{13-10} = Rd;
  let Inst{17-14} = Rs1;
  let Inst{18}    = 0; // Register mode
  let Inst{19}    = invert;
  let Inst{23-20} = Rs2;
}

// Format for ALU instructions (Register-Immediate)

// Format for ALU instructions with shifted register
class EZHInstALUShifted<bits<5> op, bits<3> sh_type, bits<1> invert, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = op;
  bits<4> Rd;
  bits<4> Rs1;
  bits<4> Rs2;
  bits<5> ShAmt;
  let hasSideEffects = 1;

  let Inst{13-10} = Rd;
  let Inst{17-14} = Rs1;
  let Inst{23-20} = Rs2;
  let Inst{28-24} = ShAmt;
  let Inst{31-29} = sh_type;
  let Inst{19}    = invert;
  let Inst{18}    = 0;
}
class EZHInstALUI<bits<5> op, bits<1> invert, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = op;
  bits<4> Rd;
  bits<4> Rs1;
  bits<11> Imm;
  let hasSideEffects = 1;

  let Inst{13-10} = Rd;
  let Inst{17-14} = Rs1;
  let Inst{18}    = 1; // Immediate mode
  let Inst{19}    = 0; // Fix disassembly collision with EZHInstALUIShifted
  let Inst{31}    = invert;
  let Inst{30-20} = Imm;
}

// Format for Flip ALU instructions
class EZHInstFlipALU<bits<3> alu_op, bits<1> is_asr, bits<1> is_fbit, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = 0x11;
  bits<4> Rd;
  bits<4> Rs1;
  bits<4> Rs2;
  bits<5> ShAmt;
  let hasSideEffects = 1;

  let Inst{13-10} = Rd;
  let Inst{17-14} = Rs1;
  let Inst{23-20} = Rs2;
  let Inst{28-24} = ShAmt;
  let Inst{31-29} = alu_op;
  let Inst{19}    = is_asr;
  let Inst{18}    = is_fbit;
}

class EZHInstALUI12<bits<5> op, bits<1> invert, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = op;
  bits<4> Rd;
  bits<4> Rs1;
  bits<12> Imm;
  let hasSideEffects = 1;

  let Inst{13-10} = Rd;
  let Inst{17-14} = Rs1;
  let Inst{18}    = 1; // Immediate mode
  let Inst{19}    = invert;
  let Inst{31-20} = Imm;
}

class EZHInstALUIShifted<bits<5> op, bits<1> invert, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = op;
  bits<4> Rd;
  bits<5> ShAmt;
  bits<11> Imm;

  let Inst{13-10} = Rd;
  let Inst{17-14} = ShAmt{3-0};
  let Inst{18}    = 1; // Immediate mode
  let Inst{19}    = ShAmt{4};
  let Inst{30-20} = Imm;
  let Inst{31}    = invert;
}

// Load/Store format
class EZHInstMem<bits<5> op, bits<1> isStore, bits<1> wordSize, bits<1> signedAccess, bits<1> prePost, bits<1> update, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = op;
  bits<4> Rd;
  bits<4> Rn;
  bits<8> Offset;
  bits<1> PrePost = prePost;
  bits<1> Update = update;

  let Inst{13-10} = !if(isStore, {0, 0, 0, Update}, Rd);
  let Inst{17-14} = Rn;
  let Inst{18}    = wordSize;
  let Inst{19}    = PrePost;
  let Inst{23-20} = !if(isStore, Rd, {0, 0, signedAccess, Update});
  let Inst{31-24} = Offset;
}

// Format for ANDOR
class EZHInstAndOr<dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = 0x16;
  bits<4> Rd;
  bits<4> Rs1;
  bits<4> Rs2;
  bits<4> Rs3;

  let Inst{13-10} = Rd;
  let Inst{17-14} = Rs1;
  let Inst{23-20} = Rs2;
  let Inst{27-24} = Rs3;
}

// Format for Bit Manipulation
class EZHInstBit<bits<2> op_type, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = 0x18;
  bits<4> Rd;
  bits<4> Rs;
  bits<5> Bit5;
  bits<1> S_bit;

  let Inst{13-10} = Rd;
  let Inst{17-14} = Rs;
  let Inst{18}    = 0;
  let Inst{23-19} = 0;
  let Inst{28-24} = Bit5;
  let Inst{30-29} = op_type;
  let Inst{31}    = S_bit; // 1 for BTST, BSET, BTOG. 0 for BCLR.
}

class EZHInstBitReg<bits<2> op_type, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = 0x18;
  bits<4> Rd;
  bits<4> Rs;
  bits<4> Rbit;
  bits<1> S_bit;

  let Inst{13-10} = Rd;
  let Inst{17-14} = Rs;
  let Inst{23-20} = Rbit;
  let Inst{18}    = 1;
  let Inst{19}    = 0;
  let Inst{28-24} = 0;
  let Inst{30-29} = op_type;
  let Inst{31}    = S_bit; // 1 for BTST, BSET, BTOG. 0 for BCLR.
}

// Format for Flip Endian / Flip Bitwise
class EZHInstFEND<bits<2> type_bits, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = 0x11;
  bits<4> Rd;
  bits<4> Rs;
  bits<5> ShAmt;

  let Inst{13-10} = Rd;
  let Inst{23-20} = Rs;
  let Inst{19-18} = type_bits;
  let Inst{28-24} = ShAmt;
}

// Format for Register Shift (RLSL, RLSR, RASR, RROR)
class EZHInstRegShift<bits<2> sh_type, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = 0x19;
  bits<4> Rd;
  bits<4> Rs;
  bits<5> Rshift; // Only using 4 bits for register, but macro takes 5

  let Inst{13-10} = Rd;
  let Inst{23-20} = Rs;
  let Inst{19-18} = sh_type;
  let Inst{28-24} = Rshift;
}

// Format for Tight Loop
class EZHInstTightLoop<dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = 0x1A;
  bits<4> Rend;
  bits<4> Rcount;

  let Inst{17-14} = Rend;
  let Inst{23-20} = Rcount;
}

// Format for Modify GPO Byte
class EZHInstModifyGPOByte<dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = 0x1E;
  bits<8> AndMask;
  bits<8> OrMask;
  bits<8> XorMask;

  let Inst{15-8}  = AndMask;
  let Inst{23-16} = OrMask;
  let Inst{31-24} = XorMask;
}
class EZHInstSingle<bits<5> op, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = op;
}

// Reg Offset Load/Store
class EZHInstMemReg<bits<5> op, bits<1> isStore, bits<1> wordSize, bits<1> signedAccess, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = op;
  bits<4> Rd;
  bits<4> Rn;
  bits<4> Rm;

  let Inst{13-10} = !if(isStore, 0, Rd);
  let Inst{17-14} = Rn; // source/raddr
  let Inst{23-20} = !if(isStore, Rd, {0, 0, signedAccess, 0}); // rdata for store
  let Inst{27-24} = Rm; // roffset
  let Inst{18}    = wordSize;
  let Inst{30}    = isStore;
}

// Format for CALL
class EZHInstCall<bits<5> op, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = op;
  bits<30> addr;
  let Inst{31-2} = addr;
  let Inst{1-0} = 3;
}

// Goto format
class EZHInstGoto<bits<1> isL, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = 0x15;
  bits<21> addr;
  let Inst{8-5} = Cond;
  let Inst{9} = 1;
  let Inst{10} = isL;
  let Inst{31-11} = addr;
}

class EZHInstGotoReg<bits<1> isL, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = 0x15;
  bits<4> raddr;
  let Inst{8-5} = Cond;
  let Inst{9} = 0;
  let Inst{10} = isL;
  let Inst{17-14} = raddr;
  let Inst{31-18} = 0;
  let Inst{13-11} = 0;
}

// Shift format
class EZHInstShift<bits<2> sh_type, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = 0x10;
  bits<4> Rd;
  bits<4> Rs;
  bits<5> ShAmt;

  let Inst{13-10} = Rd;
  let Inst{23-20} = Rs;
  let Inst{19-18} = sh_type;
  let Inst{28-24} = ShAmt;
  let Inst{31-29} = 0;
  let Inst{17-14} = 0;
}

// Per Read format
class EZHInstPerRead<bits<5> op, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = op;
  bits<4> Rd;
  bits<20> addr;
  let Inst{13-10} = Rd;
  let Inst{31-14} = addr{19-2};
}

// Per Write format
class EZHInstPerWrite<bits<5> op, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = op;
  bits<4> Rs;
  bits<20> addr;
  let Inst{23-20} = Rs;
  let Inst{31-24} = addr{19-12};
  let Inst{19-10} = addr{11-2};
}


class EZHInstShiftALU<bits<3> alu_op, bits<2> sh_type, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = 0x10;
  bits<4> Rd;
  bits<4> Rs1;
  bits<4> Rs2;
  bits<5> ShAmt;

  let Inst{13-10} = Rd;
  let Inst{17-14} = Rs1;
  let Inst{23-20} = Rs2;
  let Inst{28-24} = ShAmt;
  let Inst{19}    = sh_type{0}; // LSR/ROR is 1, LSL/ASR is 0
  let Inst{18}    = sh_type{1}; // ASR/ROR is 1, LSL/LSR is 0
  let Inst{31-29} = alu_op;
}

class EZHInstRegShiftALU<bits<3> alu_op, bits<2> sh_type, bits<1> is_post, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = 0x19;
  bits<4> Rd;
  bits<4> Rs1;
  bits<4> Rs2;
  bits<5> Rshift;

  let Inst{13-10} = Rd;
  let Inst{17-14} = Rs1;
  let Inst{23-20} = Rs2;
  let Inst{26-24} = Rshift{2-0};
  let Inst{27}    = Rshift{3};
  let Inst{28}    = is_post;
  let Inst{19}    = sh_type{0}; // LSR/ROR is 1, LSL/ASR is 0
  let Inst{18}    = sh_type{1}; // ASR/ROR is 1, LSL/LSR is 0
  let Inst{31-29} = alu_op;
}

class EZHInstVectoredHold<bits<1> is_nra, bits<1> is_lv, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = 0x1C;
  bits<4> table;
  let Inst{13-10} = table;
  let Inst{23-20} = table;
  let Inst{15} = 0;
  let Inst{18} = is_nra;
  let Inst{19} = is_lv;
  let Inst{17-16} = 0;
  let Inst{14} = 0;
  let Inst{31-24} = 0;
}

class EZHInstAccVectoredHold<bits<1> is_nra, bits<1> is_lv, dag outs, dag ins, string asmstr, list<dag> pattern>
    : EZHInst<outs, ins, asmstr, pattern> {
  let Opcode = 0x1C;
  bits<4> table;
  bits<8> vectors;
  let Inst{13-10} = table;
  let Inst{23-20} = table;
  let Inst{31-24} = vectors;
  let Inst{15} = 0;
  let Inst{18} = is_nra;
  let Inst{19} = is_lv;
  let Inst{17-16} = 0;
  let Inst{14} = 0;
  let Inst{9} = 1;
}
