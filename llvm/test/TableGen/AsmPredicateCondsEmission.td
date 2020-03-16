// RUN: llvm-tblgen -gen-disassembler -I %p/../../include %s | FileCheck %s

// Check that we don't generate invalid code of the form "( && Cond2)" when
// emitting AssemblerPredicate conditions. In the example below, the invalid
// code would be: "return ( && (Bits & arch::AssemblerCondition2));".

include "llvm/Target/Target.td"

def archInstrInfo : InstrInfo { }

def arch : Target {
  let InstructionSet = archInstrInfo;
}

def AssemblerCondition2 : SubtargetFeature<"cond2", "cond2", "true", "">;
def Pred1 : Predicate<"Condition1">;
def Pred2 : Predicate<"Condition2">,
            AssemblerPredicate<(all_of AssemblerCondition2)>;

def foo : Instruction {
  let Size = 2;
  let OutOperandList = (outs);
  let InOperandList = (ins);
  field bits<16> Inst;
  let Inst = 0xAAAA;
  let AsmString = "foo";
  field bits<16> SoftFail = 0;
  // This is the important bit:
  let Predicates = [Pred1, Pred2];
}

// CHECK: return (Bits[arch::AssemblerCondition2]);
