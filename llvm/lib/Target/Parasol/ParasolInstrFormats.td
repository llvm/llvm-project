//=== ParasolInstrFormats.td - Parasol Instruction Formats -----*- tablegen -*-===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  General functions
//===----------------------------------------------------------------------===//

class log2ValueType<ValueType vt> {
    int value = !cond(!eq(vt, i1) : 0,
                      !eq(vt, i8) : 3,
                      !eq(vt, i16) : 4,
                      !eq(vt, i32) : 5,
                      true: 0);
}

//===----------------------------------------------------------------------===//
//  Instructions
//===----------------------------------------------------------------------===//

class ParasolOpcode<bits<8> val> {
  bits<8> Value = val;
}

// Types and loading
def OPC_BINDREADONLY  : ParasolOpcode<0x00>;
def OPC_BINDREADWRITE : ParasolOpcode<0x01>;
def OPC_LOAD          : ParasolOpcode<0x02>;
def OPC_LOADI         : ParasolOpcode<0x03>;
def OPC_STORE         : ParasolOpcode<0x04>;
def OPC_ZEXT          : ParasolOpcode<0x05>;
def OPC_TRUNC         : ParasolOpcode<0x06>;
def OPC_MOV           : ParasolOpcode<0x07>;

// Arithmetic
def OPC_ADD           : ParasolOpcode<0x10>;
def OPC_ADDC          : ParasolOpcode<0x11>;
def OPC_SUB           : ParasolOpcode<0x12>;
def OPC_SUBB          : ParasolOpcode<0x13>;
def OPC_MUL           : ParasolOpcode<0x14>;

// Shifts
def OPC_SHL           : ParasolOpcode<0x20>;
def OPC_ROTL          : ParasolOpcode<0x21>;
def OPC_SHR           : ParasolOpcode<0x22>;
def OPC_ROTR          : ParasolOpcode<0x23>;

// Logic
def OPC_AND           : ParasolOpcode<0x30>;
def OPC_OR            : ParasolOpcode<0x31>;
def OPC_XOR           : ParasolOpcode<0x32>;
def OPC_NOT           : ParasolOpcode<0x33>;
def OPC_NEG           : ParasolOpcode<0x34>;

// Comparisons
def OPC_GT            : ParasolOpcode<0x40>;
def OPC_GE            : ParasolOpcode<0x41>;
def OPC_LT            : ParasolOpcode<0x42>;
def OPC_LE            : ParasolOpcode<0x43>;
def OPC_EQ            : ParasolOpcode<0x44>;
def OPC_CMUX          : ParasolOpcode<0x45>;
def OPC_SETCC         : ParasolOpcode<0x46>;

// Control flow
def OPC_BRANCH    : ParasolOpcode<0xFD>;
def OPC_JALR      : ParasolOpcode<0xFE>;
def OPC_OP            : ParasolOpcode<0xFF>;

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

class ParasolBindInst<ParasolOpcode opcode, string opcodestr, bits<1> encrypted, SDPatternOperator op>
: ParasolInst<(outs PR:$rd), (ins PR:$rs1, Parasolimm32:$buffer_id), opcodestr, "$rd, $buffer_id", [(set PR:$rd, (op PR:$rs1, Parasolimm32:$buffer_id))]> {
  bits<6> rd;
  bits<10> buffer_id;

  let Constraints = "$rd = $rs1";

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;

  let Inst{24-24} = encrypted;
  let Inst{23-14} = buffer_id;
  let Inst{13-8} = rd;
  let Opcode = opcode.Value;
}

class ParasolLoadImm<Operand op, ValueType vt> : 
ParasolInst<(outs IR:$rd), (ins op:$imm), "ldi", "$rd, $imm, " # log2ValueType<vt>.value, [(set vt:$rd, op:$imm)]> {
  bits<6> rd;
  bits<32> imm;

  let Inst{48-17} = imm;
    // 2 ^ size bits
  let Inst{16-14} = log2ValueType<vt>.value;
  let Inst{13-8} = rd;
  let Opcode = OPC_LOADI.Value;
}
