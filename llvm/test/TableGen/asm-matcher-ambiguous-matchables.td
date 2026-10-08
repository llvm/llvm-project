// RUN: not llvm-tblgen -gen-asm-matcher -I %p/../../include %s 2>&1 | \
// RUN:     FileCheck --check-prefix=ERR %s
// RUN: llvm-tblgen -gen-asm-matcher -I %p/../../include \
// RUN:     -DEXCLUSIVE %s 2>/dev/null | FileCheck %s

// Check that ErrorOnAmbiguousMatchables catches genuinely ambiguous matchables,
// and that predicates no subtarget can satisfy at once suppress false
// positives.

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
#ifdef EXCLUSIVE
def PredB : Predicate<"PredB">,
            AssemblerPredicate<(all_of FeatureB, (not FeatureA))>;
#else
def PredB : Predicate<"PredB">, AssemblerPredicate<(all_of FeatureB)>;
#endif

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

// Without EXCLUSIVE: a subtarget can have both FeatureA and FeatureB, so
// InsnA and InsnB really do compete and the clash check fires.
// ERR: error: matchable is indistinguishable to the parser
// ERR: error: 1 ambiguous matchables found

// With EXCLUSIVE: PredB requires FeatureA to be off, so no subtarget can make
// both available, no clash is reported and the output is generated normally.
// CHECK: MatchTable
