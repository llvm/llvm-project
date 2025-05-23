//=== ParasolInstrFormats.td - Parasol Instruction Formats -----*- tablegen -*-===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Instructions
//===----------------------------------------------------------------------===//

class ParasolOpcode<bits<8> val> {
  bits<8> Value = val;
}

// 0x01 - 0x0E, load and store
def OPC_STORE         : ParasolOpcode<0x01>;
def OPC_LOAD          : ParasolOpcode<0x09>;
def OPC_LOADI         : ParasolOpcode<0x0A>;

// 0x11 - 0x1E, resize
def OPC_TRUNC         : ParasolOpcode<0x11>;
def OPC_ZEXT          : ParasolOpcode<0x15>;
def OPC_SEXT          : ParasolOpcode<0x16>;

// 0x21 - 0x2E, move
def OPC_MOV           : ParasolOpcode<0x21>;

// 0x31 - 0x3E, bitwise
def OPC_NOT           : ParasolOpcode<0x31>;
def OPC_AND           : ParasolOpcode<0x32>;
def OPC_OR            : ParasolOpcode<0x33>;
def OPC_XOR           : ParasolOpcode<0x34>;

// 0x41 - 0x4E, arithmetic (add / sub)
def OPC_ADD           : ParasolOpcode<0x41>;
def OPC_ADDC          : ParasolOpcode<0x42>;
def OPC_SUB           : ParasolOpcode<0x45>;
def OPC_SUBB          : ParasolOpcode<0x46>;
def OPC_NEG           : ParasolOpcode<0x49>;

// 0x51 - 0x5E, arithmetic (mul / div)
def OPC_MUL           : ParasolOpcode<0x51>;

// 0x61 - 0x6E, reserved

// 0x71 - 0x7E, reserved

// 0x81 - 0x8E, shift
def OPC_ROTL          : ParasolOpcode<0x81>;
def OPC_ROTR          : ParasolOpcode<0x82>;
def OPC_SHL           : ParasolOpcode<0x85>;
def OPC_SHR           : ParasolOpcode<0x86>;
def OPC_SHRA          : ParasolOpcode<0x87>;

// 0x91 - 0x9E, compare
def OPC_EQ            : ParasolOpcode<0x91>;
def OPC_GT            : ParasolOpcode<0x95>;
def OPC_GTS           : ParasolOpcode<0x96>;
def OPC_GE            : ParasolOpcode<0x97>;
def OPC_GES           : ParasolOpcode<0x98>;
def OPC_LT            : ParasolOpcode<0x99>;
def OPC_LTS           : ParasolOpcode<0x9A>;
def OPC_LE            : ParasolOpcode<0x9B>;
def OPC_LES           : ParasolOpcode<0x9C>;

// 0xA1 - 0xAE, reserved

// 0xB1 - 0xBE, control flow
def OPC_BRNZ          : ParasolOpcode<0xB1>;
def OPC_BRZ           : ParasolOpcode<0xB2>;
def OPC_BR            : ParasolOpcode<0xB5>;
def OPC_JALR          : ParasolOpcode<0xBA>;

// 0xC1 - 0xCE, target specific
def OPC_CMUX          : ParasolOpcode<0xC1>;

// 0xD1 - 0xDE, reserved

// 0xE1 - 0xEE, reserved

// 0xF1 - 0xFE, reserved


def ParasolJmptarget : Operand<OtherVT> {
  //let PrintMethod = "printPCRelImmOperand";
  //let EncoderMethod = "getPCRelImmOpValue";
}

def Parasolimm32 : Operand<i32>, ImmLeaf<i32, [{
    return true;
}]>;

def Parasolimm16 : Operand<i16>, ImmLeaf<i16, [{
    return true;
}]>;

def Parasolimm8 : Operand<i8>, ImmLeaf<i8, [{
    return true;
}]>;

def Parasolimm1 : Operand<i1>, ImmLeaf<i1, [{
    return true;
}]>;

class ParasolInst<dag outs, dag ins, string opcodestr, string argstr,
            list<dag> pattern>
    : Instruction {
  field bits<64> Inst;

  // SoftFail is a field the disassembler can use to provide a way for
  // instructions to not match without killing the whole decode process. It is
  // mainly used for ARM, but Tablegen expects this field to exist or it fails
  // to build the decode table.
  field bits<64> SoftFail = 0;
  let Size = 8;

  bits<8> Opcode = 0;

  let Inst{7-0} = Opcode;

  let Namespace = "Parasol";

  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = opcodestr # "\t" # argstr;
  let Pattern = pattern;
}

// Pseudo instructions
class Pseudo<dag outs, dag ins, list<dag> pattern, string opcodestr = "",
             string argstr = "">
    : ParasolInst<outs, ins, opcodestr, argstr, pattern>,
      Sched<[]> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

class ParasolInstI<ParasolOpcode opcode, dag outs, dag ins,
              string opcodestr, string argstr>
    : ParasolInst<outs, ins, opcodestr, argstr, []> {
  bits<32> imm32;
  bits<6> rs1;
  bits<6> rd;

  let Inst{51-20} = imm32;
  let Inst{19-14} = rs1;
  let Inst{13-8} = rd;
  let Opcode = opcode.Value;
}

class ParasolUnaryALUInst<ParasolOpcode opcode, string opcodestr>
: ParasolInst<(outs IR:$rd), (ins IR:$rs1), opcodestr, "$rd, $rs1", []> {
  bits<6> rs1;
  bits<6> rd;

  let Inst{19-14} = rs1;
  let Inst{13-8} = rd;
  let Opcode = opcode.Value;
}

class ParasolBinaryALUInst<ParasolOpcode opcode, string opcodestr>
: ParasolInst<(outs IR:$rd), (ins IR:$rs1, IR:$rs2), opcodestr, "$rd, $rs1, $rs2", []> {
  bits<6> rs2;
  bits<6> rs1;
  bits<6> rd;

  let Inst{25-20} = rs2;
  let Inst{19-14} = rs1;
  let Inst{13-8} = rd;
  let Opcode = opcode.Value;
}

class ParasolTernaryInst<ParasolOpcode opcode, string opcodestr>
: ParasolInst<(outs IR:$rd), (ins IR:$rs1, IR:$rs2, IR:$rs3), opcodestr, "$rd, $rs1, $rs2, $rs3", []> {
  bits<6> rs3;
  bits<6> rs2;
  bits<6> rs1;
  bits<6> rd;

  let Inst{31-26} = rs3;
  let Inst{25-20} = rs2;
  let Inst{19-14} = rs1;
  let Inst{13-8} = rd;
  let Opcode = opcode.Value;
}

class ParasolLoadImm<Operand op, ValueType vt> :
ParasolInst<(outs IR:$rd), (ins op:$imm), "ldi", "$rd, $imm, " # vt.Size, [(set vt:$rd, op:$imm)]> {
  bits<6> rd;
  bits<32> imm;

  let Inst{52-46} = vt.Size;
  let Inst{45-14} = imm;
  let Inst{13-8} = rd;
  let Opcode = OPC_LOADI.Value;
}

class ParasolCondBranchInstruction<ParasolOpcode opcode, string opcodestr>
  : ParasolInst<(outs), (ins IR:$cnd, ParasolJmptarget:$loc), opcodestr, "$cnd, $loc", []>
{
  bits<6> cnd;
  bits<32> loc;

  let isTerminator = 1;
  let isBranch = 1;
  let isBarrier = 1;
  let hasSideEffects = 0;
  let mayStore = 0;
  let mayLoad = 0;

  let Inst{45-14} = loc;
  let Inst{13-8} = cnd;
  let Opcode = opcode.Value;
}

class ParasolBranchInstruction
   : ParasolInst<(outs), (ins ParasolJmptarget:$loc), "br", "$loc", [(br bb:$loc)]>
{
  bits<32> loc;

  let isTerminator = 1;
  let isBranch = 1;
  let isBarrier = 1;
  let hasSideEffects = 0;
  let mayStore = 0;
  let mayLoad = 0;

  let Inst{39-8} = loc;
  let Opcode = OPC_BR.Value;
}