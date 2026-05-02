//===-- RISCVInstrFormatXAIF.td ----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Instruction formats for the AI Foundry ET extensions, formerly known as the
// ET-SoC-1 Minion extensions by Esperanto Technologies.
//
// Full documentation for these extensions is publicly available at the
// following URL:
//
//   https://github.com/aifoundry-org/et-man/blob/main/ET%20Programmer's%20Reference%20Manual.pdf
//
//===----------------------------------------------------------------------===//

// Opcodes used by the ET extensions (referencing RISCVOpcode records defined in
// RISCVInstrFormats.td)
defvar OPC_ET_MEM_PS   = OPC_CUSTOM_0;
defvar OPC_ET_IMM_PS   = OPC_0011111;
defvar OPC_ET_IMM10_PI = OPC_0111111;
defvar OPC_ET_OP3_PS   = OPC_CUSTOM_2;
defvar OPC_ET_IMM_PI   = OPC_1011111;
defvar OPC_ET_CVT_PS   = OPC_OP_VE;
defvar OPC_ET_OP_PS    = OPC_CUSTOM_3;

// Example: AIF.FSWIZZ.PS
class RVInstET1<bits<7> funct7, RISCVOpcode opcode, dag outs, dag ins,
                string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<5> rd;
  bits<5> rs1;
  bits<8> imm;

  let Inst{31-25} = funct7;
  let Inst{24-20} = imm{7-3};
  let Inst{19-15} = rs1;
  let Inst{14-12} = imm{2-0};
  let Inst{11-7} = rd;
  let Inst{6-0} = opcode.Value;
}

// Example: AIF.FADDI.PS
class RVInstET2<bits<3> funct3, RISCVOpcode opcode, dag outs, dag ins,
                string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<10> imm;
  bits<5> rs1;
  bits<5> rd;

  let Inst{31-27} = imm{9-5};
  let Inst{26-25} = 0b10;
  let Inst{24-20} = imm{4-0};
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = rd;
  let Inst{6-0} = opcode.Value;
}

// Example: AIF.FADD.PS
class RVInstET3<bits<7> funct7, RISCVOpcode opcode, dag outs, dag ins,
                string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<5> rd;
  bits<5> rs1;
  bits<5> rs2;
  bits<3> rm;

  let Inst{31-25} = funct7;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = rm;
  let Inst{11-7} = rd;
  let Inst{6-0} = opcode.Value;
}

// Example: AIF.FMADD.PS
class RVInstET4<bits<2> funct2, RISCVOpcode opcode, dag outs, dag ins,
                string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<5> rd;
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rs3;
  bits<3> rm;

  let Inst{31-27} = rs3;
  let Inst{26-25} = funct2;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = rm;
  let Inst{11-7} = rd;
  let Inst{6-0} = opcode.Value;
}

// Example: AIF.FMVZ.X.PS
class RVInstET5<bits<7> funct7, bits<3> funct3,
                RISCVOpcode opcode, dag outs, dag ins,
                string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<5> rd;
  bits<5> rs1;
  bits<3> idx;

  let Inst{31-25} = funct7;
  let Inst{24-23} = 0;
  let Inst{22-20} = idx;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = rd;
  let Inst{6-0} = opcode.Value;
}

// Example: AIF.FSLLI.PS
class RVInstET6<bits<3> funct3, RISCVOpcode opcode, dag outs, dag ins,
                string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<5> imm;
  bits<5> rs1;
  bits<5> rd;

  let Inst{31-25} = 0b0100111;
  let Inst{24-20} = imm;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = rd;
  let Inst{6-0} = opcode.Value;
}

// Example: AIF.FROUND.P
class RVInstET7<RISCVOpcode opcode, dag outs, dag ins,
                string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<5> rs1;
  bits<3> rm;
  bits<5> rd;

  let Inst{31-20} = 0b010110000001;
  let Inst{19-15} = rs1;
  let Inst{14-12} = rm;
  let Inst{11-7} = rd;
  let Inst{6-0} = opcode.Value;
}

// Example: AIF.MOV.M.X
class RVInstET8<RISCVOpcode opcode, dag outs, dag ins,
                string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<8> imm;
  bits<5> rs1;
  bits<3> rd;

  let Inst{31-25} = 0b0101011;
  let Inst{24-20} = imm{7-3};
  let Inst{19-15} = rs1;
  let Inst{14-12} = imm{2-0};
  let Inst{11-10} = 0b00;
  let Inst{9-7} = rd;
  let Inst{6-0} = opcode.Value;
}

// Example: AIF.MASKAND
class RVInstET9<bits<3> funct3, RISCVOpcode opcode, dag outs, dag ins,
                string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<3> rs2;
  bits<3> rs1;
  bits<3> rd;

  let Inst{31-25} = 0b0110011;
  let Inst{24-23} = 0b00;
  let Inst{22-20} = rs2;
  let Inst{19-18} = 0b00;
  let Inst{17-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-10} = 0b00;
  let Inst{9-7} = rd;
  let Inst{6-0} = opcode.Value;
}

// Example: AIF.MASKPOPC
class RVInstET10<bits<2> funct2, RISCVOpcode opcode, dag outs, dag ins,
                 string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<3> rs1;
  bits<5> rd;

  let Inst{31-27} = 0b01010;
  let Inst{26-25} = funct2;
  let Inst{24-18} = 0b0000000;
  let Inst{17-15} = rs1;
  let Inst{14-12} = 0b000;
  let Inst{11-7} = rd;
  let Inst{6-0} = opcode.Value;
}

// Example: AIF.MASKPOPC.RAST
class RVInstET11<RISCVOpcode opcode, dag outs, dag ins,
                 string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<4> imm;
  bits<3> rs2;
  bits<3> rs1;
  bits<5> rd;

  let Inst{31-25} = 0b0101111;
  let Inst{24-23} = imm{3-2};
  let Inst{22-20} = rs2;
  let Inst{19-18} = imm{1-0};
  let Inst{17-15} = rs1;
  let Inst{14-12} = 0b000;
  let Inst{11-7} = rd;
  let Inst{6-0} = opcode.Value;
}

// Example: AIF.FSCW.PS
class RVInstET12<bits<7> funct7, bits<3> funct3,
                 RISCVOpcode opcode, dag outs, dag ins,
                 string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rs3;

  let Inst{31-25} = funct7;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = rs3;
  let Inst{6-0} = opcode.Value;
}

// Example: AIF.SBG
class RVInstET13<bits<7> funct7, bits<3> funct3,
                 RISCVOpcode opcode, dag outs, dag ins,
                 string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<5> rs1;
  bits<5> rs2;

  let Inst{31-25} = funct7;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = 0b00000;
  let Inst{6-0} = opcode.Value;
}

// Example: AIF.FSWL.PS
class RVInstET14<bits<7> funct7, bits<3> funct3, RISCVOpcode opcode, dag outs, dag ins,
                 string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<5> fs3;
  bits<5> rs1;

  let Inst{31-25} = funct7;
  let Inst{24-20} = 0b00000;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = fs3;
  let Inst{6-0} = opcode.Value;
}

