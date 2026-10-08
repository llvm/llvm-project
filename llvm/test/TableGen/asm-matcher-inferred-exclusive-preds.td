// RUN: not llvm-tblgen -gen-asm-matcher -I %p/../../include %s 2>&1 | \
// RUN:     FileCheck %s
// RUN: not llvm-tblgen -gen-asm-matcher -I %p/../../include -DNO_IMPLIES %s \
// RUN:     2>&1 | FileCheck --check-prefix=NOIMP %s

// Check that mutually exclusive assembler predicates are inferred from the
// feature requirements, without having to be declared, and that the subtarget
// feature implications are what makes the inference reach.

include "llvm/Target/Target.td"

def archInstrInfo : InstrInfo {}

def ArchAsmParser : AsmParser {
  let ErrorOnAmbiguousMatchables = 1;
}

def arch : Target {
  let InstructionSet = archInstrInfo;
  let AssemblyParsers = [ArchAsmParser];
}

// A generation ladder: each generation has everything the one before it had.
def FeatureGen1 : SubtargetFeature<"gen1", "Gen1", "true", "">;
#ifdef NO_IMPLIES
def FeatureGen2 : SubtargetFeature<"gen2", "Gen2", "true", "">;
def FeatureGen3 : SubtargetFeature<"gen3", "Gen3", "true", "">;
#else
def FeatureGen2 : SubtargetFeature<"gen2", "Gen2", "true", "",
                                   [FeatureGen1]>;
def FeatureGen3 : SubtargetFeature<"gen3", "Gen3", "true", "",
                                   [FeatureGen2]>;
#endif

// Two features that have nothing to do with each other.
def FeatureX : SubtargetFeature<"x", "HasX", "true", "">;
def FeatureY : SubtargetFeature<"y", "HasY", "true", "">;

// Gen1 but not Gen2, so not Gen3 either, though only transitively.
def isGen1Only : Predicate<"isGen1Only">,
                 AssemblerPredicate<(all_of FeatureGen1,
                                            (not FeatureGen2))>;
def isGen3Plus : Predicate<"isGen3Plus">,
                 AssemblerPredicate<(all_of FeatureGen3)>;

def HasX : Predicate<"HasX">, AssemblerPredicate<(all_of FeatureX)>;
def HasY : Predicate<"HasY">, AssemblerPredicate<(all_of FeatureY)>;

let Namespace = "arch" in
def R0 : Register<"r0">;
def Regs : RegisterClass<"Regs", [i32], 32, (add R0)>;

class TestInsn<int Opc, string Asm, list<Predicate> Preds> : Instruction {
  let Size = 2;
  let OutOperandList = (outs);
  let InOperandList = (ins);
  field bits<16> Inst = Opc;
  field bits<16> SoftFail = 0;
  let AsmString = Asm;
  let Predicates = Preds;
}

// No subtarget has Gen1 without Gen2 and also has Gen3, so these two never
// compete. Proving it needs the implication Gen3 -> Gen2.
def InsnGen1 : TestInsn<0, "ladder", [isGen1Only]>;
def InsnGen3 : TestInsn<1, "ladder", [isGen3Plus]>;

// Nothing stops a subtarget having both X and Y, so these really are
// ambiguous and must still be reported.
def InsnX : TestInsn<2, "indep", [HasX]>;
def InsnY : TestInsn<3, "indep", [HasY]>;

// CHECK-NOT: ladder
// CHECK: error: matchable is indistinguishable to the parser
// CHECK: indep
// CHECK: error: 1 ambiguous matchables found

// Without the implication, Gen1 && !Gen2 && Gen3 is satisfiable as far as the
// checker can tell, so the ladder pair is reported too.
// NOIMP: error: 2 ambiguous matchables found
