// RUN: llvm-tblgen -gen-asm-matcher -I %p/../../include %s -o - | FileCheck %s

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo;

def ArchAsmParser : AsmParser {
  let ReportMultipleNearMisses = true;
  let PrioritizeFeatureInMultiMismatchFallback = true;
}

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
  let AssemblyParsers = [ArchAsmParser];
}

def Reg : Register<"reg">;
def RegClass : RegisterClass<"foo", [i32], 0, (add Reg)>;

class ArchInstruction<string AsmStr, dag OutOperands, dag InOperands>
    : Instruction {
  let OutOperandList = OutOperands;
  let InOperandList = InOperands;
  let AsmString = AsmStr;
}

def Instr : ArchInstruction<"instr $foo", (outs RegClass:$foo), (ins)> {
  bits<1> foo;
}

// Check that a feature near-miss is preferred over an operand near-miss
// in the MultiMismatchFallback path.
// CHECK: if (FeaturesNearMiss)
// CHECK-NEXT: MultiMismatchFallback.push_back(FeaturesNearMiss);
// CHECK-NEXT: else if (OperandNearMiss)
// CHECK-NEXT: MultiMismatchFallback.push_back(OperandNearMiss);
