// RUN: llvm-tblgen -gen-emitter -I %p/../../include %s | FileCheck %s

// Check if VarLenCodeEmitterGen works correctly.

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo { }

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
}

def Reg : Register<"reg">;

def RegClass : RegisterClass<"foo", [i64], 0, (add Reg)>;

def GR64 : RegisterOperand<RegClass>;

class MyMemOperand<dag sub_ops> : Operand<iPTR> {
  let MIOperandInfo = sub_ops;
  dag Base;
  dag Extension;
}

class MyVarInst<MyMemOperand memory_op> : Instruction {
  dag Inst;

  let OutOperandList = (outs GR64:$dst);
  let InOperandList  = (ins memory_op:$src);

  // Testing `ascend` and `descend`
  let Inst = (ascend
    (descend 0b10110111, memory_op.Base),
    memory_op.Extension,
    // Testing operand referencing.
    (operand "$dst", 4),
    // Testing operand referencing with a certain bit range.
    (slice "$dst", 3, 1)
  );
}

class MemOp16<string op_name> : MyMemOperand<(ops GR64:$reg, i16imm:$offset)> {
  // Testing sub-operand referencing.
  let Base = (operand "$"#op_name#".reg", 8);
  let Extension = (operand "$"#op_name#".offset", 16);
}

class MemOp32<string op_name> : MyMemOperand<(ops GR64:$reg, i32imm:$offset)> {
  let Base = (operand "$"#op_name#".reg", 8);
  // Testing variable-length instruction encoding.
  let Extension = (operand "$"#op_name#".offset", 32);
}

def FOO16 : MyVarInst<MemOp16<"src">>;
def FOO32 : MyVarInst<MemOp32<"src">>;

// The fixed bits part
// CHECK: {/*NumBits*/39,
// CHECK-SAME: // FOO16
// CHECK: {/*NumBits*/55,
// CHECK-SAME: // FOO32
// CHECK: UINT64_C(46848), // FOO16
// CHECK: UINT64_C(46848), // FOO32

// CHECK-LABEL: case ::FOO16: {
// CHECK: Scratch = Scratch.zextOrSelf(39);
// src.reg
// CHECK: getMachineOpValue(MI, MI.getOperand(1), Scratch, Fixups, STI);
// CHECK: Inst.insertBits(Scratch.extractBits(8, 0), 0);
// src.offset
// CHECK: getMachineOpValue(MI, MI.getOperand(2), Scratch, Fixups, STI);
// CHECK: Inst.insertBits(Scratch.extractBits(16, 0), 16);
// 1st dst
// CHECK: getMachineOpValue(MI, MI.getOperand(0), Scratch, Fixups, STI);
// CHECK: Inst.insertBits(Scratch.extractBits(4, 0), 32);
// 2nd dst
// CHECK: getMachineOpValue(MI, MI.getOperand(0), Scratch, Fixups, STI);
// CHECK: Inst.insertBits(Scratch.extractBits(3, 1), 36);

// CHECK-LABEL: case ::FOO32: {
// CHECK: Scratch = Scratch.zextOrSelf(55);
// src.reg
// CHECK: getMachineOpValue(MI, MI.getOperand(1), Scratch, Fixups, STI);
// CHECK: Inst.insertBits(Scratch.extractBits(8, 0), 0);
// src.offset
// CHECK: getMachineOpValue(MI, MI.getOperand(2), Scratch, Fixups, STI);
// CHECK: Inst.insertBits(Scratch.extractBits(32, 0), 16);
// 1st dst
// CHECK: getMachineOpValue(MI, MI.getOperand(0), Scratch, Fixups, STI);
// CHECK: Inst.insertBits(Scratch.extractBits(4, 0), 48);
// 2nd dst
// CHECK: getMachineOpValue(MI, MI.getOperand(0), Scratch, Fixups, STI);
// CHECK: Inst.insertBits(Scratch.extractBits(3, 1), 52);
