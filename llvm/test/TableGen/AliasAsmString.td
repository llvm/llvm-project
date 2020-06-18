// RUN: llvm-tblgen -gen-asm-writer -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo { }

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
}

def Reg : Register<"reg">;

def RegClass : RegisterClass<"foo", [i32], 0, (add Reg)>;

def IntOperand: Operand<i32>;

def foo : Instruction {
  let Size = 2;
  let OutOperandList = (outs);
  let InOperandList = (ins IntOperand:$imm);
  let AsmString = "foo $imm";
  let Namespace = "Arch";
}

def FooBraces : InstAlias<"foo \\{$imm\\}", (foo IntOperand:$imm)>;

// CHECK: static const char AsmStrings[] =
// CHECK-NEXT: /* 0 */ "foo {$\x01}\0"
