//===-- RISCVInstrFormatsV.td - RISCV V Instruction Formats --*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file describes the RISC-V V extension instruction formats.
//
//===----------------------------------------------------------------------===//

class RISCVVFormat<bits<3> val> {
  bits<3> Value = val;
}
def OPIVV : RISCVVFormat<0b000>;
def OPFVV : RISCVVFormat<0b001>;
def OPMVV : RISCVVFormat<0b010>;
def OPIVI : RISCVVFormat<0b011>;
def OPIVX : RISCVVFormat<0b100>;
def OPFVF : RISCVVFormat<0b101>;
def OPMVX : RISCVVFormat<0b110>;

class RISCVMOP<bits<3> val> {
  bits<3> Value = val;
}
def MOPLDUnitStrideU : RISCVMOP<0b000>;
def MOPLDStridedU    : RISCVMOP<0b010>;
def MOPLDIndexedU    : RISCVMOP<0b011>;
def MOPLDUnitStrideS : RISCVMOP<0b100>;
def MOPLDStridedS    : RISCVMOP<0b110>;
def MOPLDIndexedS    : RISCVMOP<0b111>;

def MOPSTUnitStride  : RISCVMOP<0b000>;
def MOPSTStrided     : RISCVMOP<0b010>;
def MOPSTIndexedOrder: RISCVMOP<0b011>;
def MOPSTIndexedUnOrd: RISCVMOP<0b111>;

class RISCVLSUMOP<bits<5> val> {
  bits<5> Value = val;
}
def LUMOPUnitStride  : RISCVLSUMOP<0b00000>;
def LUMOPUnitStrideWholeReg : RISCVLSUMOP<0b01000>;
def LUMOPUnitStrideFF: RISCVLSUMOP<0b10000>;
def SUMOPUnitStride  : RISCVLSUMOP<0b00000>;
def SUMOPUnitStrideWholeReg : RISCVLSUMOP<0b01000>;

class RISCVWidth<bits<3> val> {
  bits<3> Value = val;
}
def LSWidthVByte : RISCVWidth<0b000>;
def LSWidthVHalf : RISCVWidth<0b101>;
def LSWidthVWord : RISCVWidth<0b110>;
def LSWidthVSEW  : RISCVWidth<0b111>;

class RVInstSetVLi<dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatI> {
  bits<5> rs1;
  bits<5> rd;
  bits<11> vtypei;

  let Inst{31} = 0;
  let Inst{30-20} = vtypei;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b111;
  let Inst{11-7} = rd;
  let Opcode = OPC_OP_V.Value;

  let Defs = [VTYPE, VL];
}

class RVInstSetVL<dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> rd;

  let Inst{31} = 1;
  let Inst{30-25} = 0b000000;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b111;
  let Inst{11-7} = rd;
  let Opcode = OPC_OP_V.Value;

  let Defs = [VTYPE, VL];
}

class RVInstVV<bits<6> funct6, RISCVVFormat opv, dag outs, dag ins,
               string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> vs1;
  bits<5> vd;
  bit vm;

  let Inst{31-26} = funct6;
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = vs1;
  let Inst{14-12} = opv.Value;
  let Inst{11-7} = vd;
  let Opcode = OPC_OP_V.Value;

  let Uses = [VTYPE, VL];
}

class RVInstVX<bits<6> funct6, RISCVVFormat opv, dag outs, dag ins,
                string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> rs1;
  bits<5> vd;
  bit vm;

  let Inst{31-26} = funct6;
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = opv.Value;
  let Inst{11-7} = vd;
  let Opcode = OPC_OP_V.Value;

  let Uses = [VTYPE, VL];
}

class RVInstV2<bits<6> funct6, bits<5> vs2, RISCVVFormat opv, dag outs, dag ins,
                string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> rs1;
  bits<5> vd;
  bit vm;

  let Inst{31-26} = funct6;
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = opv.Value;
  let Inst{11-7} = vd;
  let Opcode = OPC_OP_V.Value;

  let Uses = [VTYPE, VL];
}

class RVInstIVI<bits<6> funct6, dag outs, dag ins, string opcodestr,
                string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> imm;
  bits<5> vd;
  bit vm;

  let Inst{31-26} = funct6;
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = imm;
  let Inst{14-12} = 0b011;
  let Inst{11-7} = vd;
  let Opcode = OPC_OP_V.Value;

  let Uses = [VTYPE, VL];
}

class RVInstV<bits<6> funct6, bits<5> vs1, RISCVVFormat opv, dag outs,
              dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> vd;
  bit vm;

  let Inst{31-26} = funct6;
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = vs1;
  let Inst{14-12} = opv.Value;
  let Inst{11-7} = vd;
  let Opcode = OPC_OP_V.Value;

  let Uses = [VTYPE, VL];
}

class RVInstVLU<bits<3> nf, RISCVMOP mop, RISCVLSUMOP lumop,
                RISCVWidth width, dag outs, dag ins, string opcodestr,
                string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> rs1;
  bits<5> vd;
  bit vm;

  let Inst{31-29} = nf;
  let Inst{28-26} = mop.Value;
  let Inst{25} = vm;
  let Inst{24-20} = lumop.Value;
  let Inst{19-15} = rs1;
  let Inst{14-12} = width.Value;
  let Inst{11-7} = vd;
  let Opcode = OPC_LOAD_FP.Value;

  let Uses = [VTYPE, VL];
}

class RVInstVLS<bits<3> nf, RISCVMOP mop, RISCVWidth width,
                dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> vd;
  bit vm;

  let Inst{31-29} = nf;
  let Inst{28-26} = mop.Value;
  let Inst{25} = vm;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = width.Value;
  let Inst{11-7} = vd;
  let Opcode = OPC_LOAD_FP.Value;

  let Uses = [VTYPE, VL];
}

class RVInstVLX<bits<3> nf, RISCVMOP mop, RISCVWidth width,
                dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> rs1;
  bits<5> vd;
  bit vm;

  let Inst{31-29} = nf;
  let Inst{28-26} = mop.Value;
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = width.Value;
  let Inst{11-7} = vd;
  let Opcode = OPC_LOAD_FP.Value;

  let Uses = [VTYPE, VL];
}

class RVInstVSU<bits<3> nf, RISCVMOP mop, RISCVLSUMOP sumop,
                RISCVWidth width, dag outs, dag ins, string opcodestr,
                string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> rs1;
  bits<5> vs3;
  bit vm;

  let Inst{31-29} = nf;
  let Inst{28-26} = mop.Value;
  let Inst{25} = vm;
  let Inst{24-20} = sumop.Value;
  let Inst{19-15} = rs1;
  let Inst{14-12} = width.Value;
  let Inst{11-7} = vs3;
  let Opcode = OPC_STORE_FP.Value;

  let Uses = [VTYPE, VL];
}

class RVInstVSS<bits<3> nf, RISCVMOP mop, RISCVWidth width,
                dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> vs3;
  bit vm;

  let Inst{31-29} = nf;
  let Inst{28-26} = mop.Value;
  let Inst{25} = vm;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = width.Value;
  let Inst{11-7} = vs3;
  let Opcode = OPC_STORE_FP.Value;

  let Uses = [VTYPE, VL];
}

class RVInstVSX<bits<3> nf, RISCVMOP mop, RISCVWidth width,
                dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> rs1;
  bits<5> vs3;
  bit vm;

  let Inst{31-29} = nf;
  let Inst{28-26} = mop.Value;
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = width.Value;
  let Inst{11-7} = vs3;
  let Opcode = OPC_STORE_FP.Value;

  let Uses = [VTYPE, VL];
}
