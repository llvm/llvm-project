// RUN: not llvm-tblgen -gen-asm-matcher -I %p/../../include %s 2>&1 | \
// RUN:     FileCheck --check-prefix=ERR %s
// RUN: llvm-tblgen -gen-asm-matcher -I %p/../../include \
// RUN:     -DEXCLUSIVE %s 2>/dev/null | FileCheck %s

// Check that ErrorOnAmbiguousMatchables catches genuinely ambiguous matchables,
// and that MutuallyExclusiveAssemblerPredicates suppresses false positives.

include "llvm/Target/Target.td"

def archInstrInfo : InstrInfo {}

def ArchAsmParser : AsmParser {
  let ErrorOnAmbiguousMatchables = 1;
}

def arch : Target {
  let InstructionSet = archInstrInfo;
  let AssemblyParsers = [ArchAsmParser];
}

def FeatureA : SubtargetFeature<"featureA", "featureA", "true", "">;
def FeatureB : SubtargetFeature<"featureB", "featureB", "true", "">;

def PredA : Predicate<"PredA">, AssemblerPredicate<(all_of FeatureA)>;
def PredB : Predicate<"PredB">, AssemblerPredicate<(all_of FeatureB)>;

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

def InsnA : TestInsn<0, "insn", [PredA]>;
def InsnB : TestInsn<1, "insn", [PredB]>;

#ifdef EXCLUSIVE
def : MutuallyExclusiveAssemblerPredicates<PredA, [PredB]>;
#endif

// Without EXCLUSIVE: InsnA and InsnB have the same asm string with no declared
// mutual exclusion, so the clash check fires.
// ERR: error: matchable is indistinguishable to the parser
// ERR: error: 1 ambiguous matchables found

// With EXCLUSIVE: PredA and PredB are declared mutually exclusive, so no clash
// is reported and the output is generated normally.
// CHECK: MatchTable
