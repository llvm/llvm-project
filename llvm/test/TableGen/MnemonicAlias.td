// RUN: llvm-tblgen -gen-asm-matcher -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo { }

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
}

def Reg : Register<"reg">;
def RegClass : RegisterClass<"foo", [i32], 0, (add Reg)>;

def AsmCond1 : SubtargetFeature<"cond1", "cond1", "true", "">;
def AsmCond2 : SubtargetFeature<"cond2", "cond2", "true", "">;

def Subtarget1 : Predicate<"Pred1">, AssemblerPredicate<(all_of AsmCond1)>;
def Subtarget2 : Predicate<"Pred2">, AssemblerPredicate<(all_of AsmCond2)>;

multiclass DefInstruction<string name, dag outs, dag ins, Predicate pred> {
  def "" : Instruction {
    let Size = 2;
    let OutOperandList = outs;
    let InOperandList = ins;
    let AsmString = name;
    let Predicates = [pred];
  }
  def : MnemonicAlias<name # "_alias", name>;
}

defm FooInst1 : DefInstruction<"foo", (outs), (ins), Subtarget1>;

defm FooInst2 : DefInstruction<"foo", (outs), (ins), Subtarget2>;

// Check that applyMnemonicAliases maps "foo_alias" to "foo" once only and
// without checking any predicates.

// CHECK:      if (memcmp(Mnemonic.data()+0, "foo_alias", 9) != 0)
// CHECK-NEXT:   break;
// CHECK-NEXT: Mnemonic = "foo";        // "foo_alias"
// CHECK-NEXT: return;
