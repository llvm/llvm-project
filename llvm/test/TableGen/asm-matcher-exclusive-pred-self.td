// RUN: not llvm-tblgen -gen-asm-matcher -I %p/../../include %s 2>&1 | \
// RUN:     FileCheck %s

// Check that declaring a predicate mutually exclusive with itself is an error,
// regardless of whether ErrorOnAmbiguousMatchables is set.

include "llvm/Target/Target.td"

def archInstrInfo : InstrInfo {}

def arch : Target {
  let InstructionSet = archInstrInfo;
}

def FeatureA : SubtargetFeature<"featureA", "featureA", "true", "">;
def PredA : Predicate<"PredA">, AssemblerPredicate<(all_of FeatureA)>;

let Namespace = "arch" in
def R0 : Register<"r0">;
def Regs : RegisterClass<"Regs", [i32], 32, (add R0)>;

def : MutuallyExclusiveAssemblerPredicates<PredA, [PredA]>;

// CHECK: error: predicate 'PredA' declared mutually exclusive with itself
