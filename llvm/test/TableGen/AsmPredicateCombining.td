// RUN: llvm-tblgen -gen-disassembler -ignore-non-decodable-operands -I %p/../../include %s | \
// RUN:     FileCheck --check-prefix=DISASS %s
// RUN: llvm-tblgen -gen-asm-matcher -I %p/../../include %s | \
// RUN:     FileCheck --check-prefix=MATCHER %s
// RUN: llvm-tblgen -gen-asm-writer -I %p/../../include %s | \
// RUN:     FileCheck --check-prefix=WRITER %s

// Check that combining conditions in AssemblerPredicate generates the correct
// output when using both the (all_of) AND operator, and the (any_of) OR
// operator.

include "llvm/Target/Target.td"

def archInstrInfo : InstrInfo { }
def archAsmWriter : AsmWriter {
  int PassSubtarget = 1;
}

def arch : Target {
  let InstructionSet = archInstrInfo;
  let AssemblyWriters = [archAsmWriter];
}

let Namespace = "arch" in {
  def R0 : Register<"r0">;
  def R1 : Register<"r1">;
  def R2 : Register<"r2">;
  def R3 : Register<"r3">;
  def R4 : Register<"r4">;
}
def Regs : RegisterClass<"Regs", [i32], 32, (add R0, R1, R2, R3, R4)>;

class TestInsn<int Opc, list<Predicate> Preds> : Instruction {
  let Size = 2;
  let OutOperandList = (outs);
  let InOperandList = (ins Regs:$r);
  field bits<16> Inst;
  let Inst = Opc;
  let AsmString = NAME # " $r";
  field bits<16> SoftFail = 0;
  let Predicates = Preds;
}


def AsmCond1 : SubtargetFeature<"cond1", "cond1", "true", "">;
def AsmCond2a: SubtargetFeature<"cond2a", "cond2a", "true", "">;
def AsmCond2b: SubtargetFeature<"cond2b", "cond2b", "true", "">;
def AsmCond3a: SubtargetFeature<"cond3a", "cond3a", "true", "">;
def AsmCond3b: SubtargetFeature<"cond3b", "cond3b", "true", "">;
def AsmCond4 : SubtargetFeature<"cond4", "cond4", "true", "">;

def AsmPred1 : Predicate<"Pred1">, AssemblerPredicate<(all_of AsmCond1)>;
def AsmPred2 : Predicate<"Pred2">, AssemblerPredicate<(all_of AsmCond2a, AsmCond2b)>;
def AsmPred3 : Predicate<"Pred3">, AssemblerPredicate<(any_of AsmCond3a, AsmCond3b)>;
def AsmPred4 : Predicate<"Pred4">, AssemblerPredicate<(all_of AsmCond4, (not (any_of AsmCond3a, AsmCond3b)))>;
// MATCHER:      if (FB[arch::AsmCond1])
// MATCHER-NEXT:   Features.set(Feature_AsmPred1Bit);
// MATCHER-NEXT: if (FB[arch::AsmCond2a] && FB[arch::AsmCond2b])
// MATCHER-NEXT:   Features.set(Feature_AsmPred2Bit);
// MATCHER-NEXT: if (FB[arch::AsmCond3a] || FB[arch::AsmCond3b])
// MATCHER-NEXT:   Features.set(Feature_AsmPred3Bit);
// MATCHER-NEXT: if (FB[arch::AsmCond4] && !(FB[arch::AsmCond3a] || FB[arch::AsmCond3b]))
// MATCHER-NEXT:   Features.set(Feature_AsmPred4Bit);

def insn1 : TestInsn<1, [AsmPred1]>;
// DISASS: return (Bits[arch::AsmCond1]);

def insn2 : TestInsn<2, [AsmPred2]>;
// DISASS: return (Bits[arch::AsmCond2a] && Bits[arch::AsmCond2b])

def insn3 : TestInsn<3, [AsmPred3]>;
// DISASS: return (Bits[arch::AsmCond3a] || Bits[arch::AsmCond3b])

def insn4 : TestInsn<4, [AsmPred1, AsmPred2]>;
// DISASS: return (Bits[arch::AsmCond1] && (Bits[arch::AsmCond2a] && Bits[arch::AsmCond2b]))

def insn5 : TestInsn<5, [AsmPred1, AsmPred3]>;
// DISASS: return (Bits[arch::AsmCond1] && (Bits[arch::AsmCond3a] || Bits[arch::AsmCond3b]))

def insn6 : TestInsn<6, []>;
def : InstAlias<"alias1", (insn6 R0)> { let Predicates = [AsmPred1]; }
// WRITER:      // (insn6 R0)
// WRITER-NEXT: {AliasPatternCond::K_Reg, arch::R0},
// WRITER-NEXT: {AliasPatternCond::K_Feature, arch::AsmCond1},
def : InstAlias<"alias2", (insn6 R1)> { let Predicates = [AsmPred2]; }
// WRITER:      // (insn6 R1)
// WRITER-NEXT: {AliasPatternCond::K_Reg, arch::R1},
// WRITER-NEXT: {AliasPatternCond::K_Feature, arch::AsmCond2a},
// WRITER-NEXT: {AliasPatternCond::K_Feature, arch::AsmCond2b},
def : InstAlias<"alias3", (insn6 R2)> { let Predicates = [AsmPred3]; }
// WRITER:      // (insn6 R2)
// WRITER-NEXT: {AliasPatternCond::K_Reg, arch::R2},
// WRITER-NEXT: {AliasPatternCond::K_OrFeature, arch::AsmCond3a},
// WRITER-NEXT: {AliasPatternCond::K_OrFeature, arch::AsmCond3b},
// WRITER-NEXT: {AliasPatternCond::K_EndOrFeatures, 0},
def : InstAlias<"alias4", (insn6 R3)> { let Predicates = [AsmPred1, AsmPred2]; }
// WRITER:      // (insn6 R3)
// WRITER-NEXT: {AliasPatternCond::K_Reg, arch::R3},
// WRITER-NEXT: {AliasPatternCond::K_Feature, arch::AsmCond1},
// WRITER-NEXT: {AliasPatternCond::K_Feature, arch::AsmCond2a},
// WRITER-NEXT: {AliasPatternCond::K_Feature, arch::AsmCond2b},
def : InstAlias<"alias5", (insn6 R4)> { let Predicates = [AsmPred1, AsmPred3]; }
// WRITER:      // (insn6 R4)
// WRITER-NEXT: {AliasPatternCond::K_Reg, arch::R4},
// WRITER-NEXT: {AliasPatternCond::K_Feature, arch::AsmCond1},
// WRITER-NEXT: {AliasPatternCond::K_OrFeature, arch::AsmCond3a},
// WRITER-NEXT: {AliasPatternCond::K_OrFeature, arch::AsmCond3b},
// WRITER-NEXT: {AliasPatternCond::K_EndOrFeatures, 0},
