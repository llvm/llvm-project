//=-   MicroMips64r6InstrFormats.td - Instruction Formats  -*- tablegen -*  -=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes microMIPS64r6 instruction formats.
//
//===----------------------------------------------------------------------===//

class DAUI_FM_MMR6 {
  bits<5> rt;
  bits<5> rs;
  bits<16> imm;

  bits<32> Inst;

  let Inst{31-26} = 0b111100;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-0}  = imm;
}

class POOL32I_ADD_IMM_FM_MMR6<bits<5> funct> {
  bits<5> rs;
  bits<16> imm;

  bits<32> Inst;

  let Inst{31-26} = 0b010000;
  let Inst{25-21} = funct;
  let Inst{20-16} = rs;
  let Inst{15-0} = imm;
}

class POOL32S_EXTBITS_FM_MMR6<bits<6> funct> {
  bits<5> rt;
  bits<5> rs;
  bits<5> size;
  bits<5> pos;

  bits<32> Inst;

  let Inst{31-26} = 0b010110;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-11} = size;
  let Inst{10-6}  = pos;
  let Inst{5-0}   = funct;
}

class POOL32S_DALIGN_FM_MMR6 {
  bits<5> rs;
  bits<5> rt;
  bits<5> rd;
  bits<3> bp;

  bits<32> Inst;

  let Inst{31-26} = 0b010110;
  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-11} = rd;
  let Inst{10-8}  = bp;
  let Inst{7-6}   = 0b00;
  let Inst{5-0}   = 0b011100;
}

class POOL32A_DIVMOD_FM_MMR6<string instr_asm, bits<9> funct>
    : MMR6Arch<instr_asm> {
  bits<5> rt;
  bits<5> rs;
  bits<5> rd;

  bits<32> Inst;

  let Inst{31-26} = 0b010110;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-11} = rd;
  let Inst{10-9}  = 0b00;
  let Inst{8-0}  = funct;
}

class POOL32S_DMFTC0_FM_MMR6<string instr_asm, bits<5> funct>
    : MMR6Arch<instr_asm>, MipsR6Inst {
  bits<5> rt;
  bits<5> rs;
  bits<3> sel;

  bits<32> Inst;

  let Inst{31-26} = 0b010110;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-14} = 0;
  let Inst{13-11} = sel;
  let Inst{10-6}  = funct;
  let Inst{5-0}   = 0b111100;
}

class POOL32S_ARITH_FM_MMR6<string opstr, bits<9> funct>
    : MMR6Arch<opstr> {
  bits<5> rt;
  bits<5> rs;
  bits<5> rd;

  bits<32> Inst;

  let Inst{31-26} = 0b010110;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-11} = rd;
  let Inst{10-9}  = 0b00;
  let Inst{8-0}   = funct;
}

class DADDIU_FM_MMR6<string opstr> : MMR6Arch<opstr> {
  bits<5> rt;
  bits<5> rs;
  bits<16> imm16;

  bits<32> Inst;

  let Inst{31-26} = 0b010111;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-0}  = imm16;
}

class PCREL18_FM_MMR6<bits<3> funct> : MipsR6Inst {
  bits<5> rt;
  bits<18> imm;

  bits<32> Inst;

  let Inst{31-26} = 0b011110;
  let Inst{25-21} = rt;
  let Inst{20-18} = funct;
  let Inst{17-0} = imm;
}

class POOL32S_2R_FM_MMR6<string instr_asm, bits<10> funct>
    : MMR6Arch<instr_asm>, MipsR6Inst {
  bits<5> rt;
  bits<5> rs;

  bits<32> Inst;

  let Inst{31-26} = 0b010110;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-6} = funct;
  let Inst{5-0} = 0b111100;
}

class POOL32S_2RSA5B0_FM_MMR6<string instr_asm, bits<9> funct>
    : MMR6Arch<instr_asm>, MipsR6Inst {
  bits<5> rt;
  bits<5> rs;
  bits<5> sa;

  bits<32> Inst;

  let Inst{31-26} = 0b010110;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-11} = sa;
  let Inst{10-9} = 0b00;
  let Inst{8-0} = funct;
}

class LD_SD_32_2R_OFFSET16_FM_MMR6<string instr_asm, bits<6> op>
    : MMR6Arch<instr_asm>, MipsR6Inst {
  bits<5> rt;
  bits<21> addr;
  bits<5> base = addr{20-16};
  bits<16> offset = addr{15-0};

  bits<32> Inst;

  let Inst{31-26} = op;
  let Inst{25-21} = rt;
  let Inst{20-16} = base;
  let Inst{15-0}  = offset;
}

class POOL32C_2R_OFFSET12_FM_MMR6<string instr_asm, bits<4> funct>
    : MMR6Arch<instr_asm>, MipsR6Inst {
  bits<5> rt;
  bits<21> addr;
  bits<5> base = addr{20-16};
  bits<12> offset = addr{11-0};

  bits<32> Inst;

  let Inst{31-26} = 0b011000;
  let Inst{25-21} = rt;
  let Inst{20-16} = base;
  let Inst{15-12} = funct;
  let Inst{11-0}  = offset;
}

class POOL32S_3R_FM_MMR6<string instr_asm, bits<9> funct>
    : MMR6Arch<instr_asm>, MipsR6Inst {
  bits<5> rt;
  bits<5> rs;
  bits<5> rd;

  bits<32> Inst;

  let Inst{31-26} = 0b010110;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-11} = rd;
  let Inst{10-9}  = 0b00;
  let Inst{8-0}   = funct;
}
