// RUN: llvm-tblgen -gen-emitter -I %p/../../include %s | FileCheck %s

// Check that EncoderMethod for RegisterOperand is working correctly

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo { }

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
}

def Reg : Register<"reg">;

def RegClass : RegisterClass<"foo", [i32], 0, (add Reg)>;

def RegOperand : RegisterOperand<RegClass> {
  let EncoderMethod = "barEncoder";
}

def foo1 : Instruction {
  let Size = 1;

  let OutOperandList = (outs);
  let InOperandList = (ins RegOperand:$bar);

  bits<8> bar;
  bits<8> Inst = bar;
}

// CHECK: case ::foo1: {
// CHECK:   op = barEncoder
// CHECK:   op &= UINT64_C(255);
// CHECK:   Value |= op;
// CHECK:   break;
// CHECK: }


// Also check that it works from a complex operand.

def RegPair : Operand<i32> {
  let MIOperandInfo = (ops RegOperand, RegOperand);
}

def foo2 : Instruction {
  let Size = 1;

  let OutOperandList = (outs);
  let InOperandList = (ins (RegPair $r1, $r2):$r12);

  bits<4> r1;
  bits<4> r2;
  bits<8> Inst;
  let Inst{3-0} = r1;
  let Inst{7-4} = r2;
}

// CHECK: case ::foo2: {
// CHECK:   op = barEncoder
// CHECK:   op &= UINT64_C(15);
// CHECK:   Value |= op;
// CHECK:   op = barEncoder
// CHECK:   op &= UINT64_C(15);
// CHECK:   Value |= op;
// CHECK:   break;
// CHECK: }
