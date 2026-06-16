// RUN: llvm-tblgen -gen-instr-info -I %p/../../include %s -o - | FileCheck %s

include "llvm/Target/Target.td"

class TestRegister<string Name> : Register<Name> {
  let Namespace = "Test";
}

def R0 : TestRegister<"r0">;
def R1 : TestRegister<"r1">;
def TestRegisters : RegisterClass<"Test", [i32], 32, (add R0, R1)>;

class TestInstruction : Instruction {
  let Namespace = "Test";
  let OutOperandList = (outs);
  let InOperandList = (ins);
  let AsmString = "";
}

def Empty : TestInstruction;
def UsesR0DefsR1 : TestInstruction {
  let Uses = [R0];
  let Defs = [R1];
}
def UsesR0R1 : TestInstruction {
  let Uses = [R0, R1];
}
def DuplicateUsesR0DefsR1 : TestInstruction {
  let Uses = [R0];
  let Defs = [R1];
}

def TestInstrInfo : InstrInfo;
def TestTarget : Target {
  let InstructionSet = TestInstrInfo;
}

defm remaps : RemapAllTargetPseudoPointerOperands<TestRegisters>;

// Offset zero is the empty-list sentinel. The two lists containing R0 and R1
// have distinct headers because their use/definition boundaries differ. The
// duplicate list is emitted only once.
// CHECK: MCPhysReg ImplicitOps[7];
// CHECK: /* 0 */ 0,
// CHECK-NEXT: /* 1 */ 65, Test::R0, Test::R1,
// CHECK-NEXT: /* 4 */ 2, Test::R0, Test::R1,
// CHECK-NOT: Test::R0, Test::R1,
