// RUN: llvm-tblgen -gen-asm-writer -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo { }

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
}

def R0 : Register<"r0">;
def Reg : RegisterClass<"Reg", [i32], 0, (add R0)>;

def IntOperand: Operand<i32>;

def PCRelOperand : Operand<i32> {
  let OperandType = "OPERAND_PCREL";
}

def foo : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins Reg:$reg, IntOperand:$imm);
  let AsmString = "foo $reg, $imm";
}

def bar : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins Reg:$reg, PCRelOperand:$imm);
  let AsmString = "bar $reg, $imm";
}

// CHECK:      ArchInstPrinter::printInstruction(
// CHECK:      // bar, foo
// CHECK-NEXT: printOperand(MI, 0, O);
// CHECK:      // foo
// CHECK-NEXT: printOperand(MI, 1, O);
// CHECK:      // bar
// CHECK-NEXT: printOperand(MI, Address, 1, O);
