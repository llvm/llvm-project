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

def foo : Instruction {
  let Size = 1;

  let OutOperandList = (outs);
  let InOperandList = (ins RegOperand:$bar);

  bits<8> bar;
  bits<8> Inst = bar;
}

// CHECK: case ::foo: {
// CHECK:   op = barEncoder
// CHECK:   op &= UINT64_C(255);
// CHECK:   Value |= op;
// CHECK:   break;
// CHECK: }
