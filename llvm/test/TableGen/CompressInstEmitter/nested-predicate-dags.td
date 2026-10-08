// RUN: llvm-tblgen -gen-compress-inst-emitter -I %p/../../../include %s 2>&1 | \
// RUN:     FileCheck %s
// RUN: not llvm-tblgen -gen-compress-inst-emitter -I %p/../../../include \
// RUN:     -DBAD_NOT %s 2>&1 | FileCheck --check-prefix=ERR %s
// RUN: not llvm-tblgen -gen-compress-inst-emitter -I %p/../../../include \
// RUN:     -DBAD_LEAF %s 2>&1 | FileCheck --check-prefix=ERR %s
// RUN: not llvm-tblgen -gen-compress-inst-emitter -I %p/../../../include \
// RUN:     -DBAD_EMPTY %s 2>&1 | FileCheck --check-prefix=ERR %s

// Check that an AssemblerCondDag made of arbitrarily nested all_of/any_of/not
// is lowered to conjunctive normal form, rather than rejected as invalid, and
// that dags which are malformed rather than merely nested are still rejected.

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo {}
def ArchAsmWriter : AsmWriter { int PassSubtarget = 1; }

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
  let AssemblyWriters = [ArchAsmWriter];
}

def Reg0 : Register<"reg0">;
def Regs : RegisterClass<"Arch", [i32], 32, (add Reg0)>;

def FeatureA : SubtargetFeature<"a", "HasA", "true", "">;
def FeatureB : SubtargetFeature<"b", "HasB", "true", "">;
def FeatureC : SubtargetFeature<"c", "HasC", "true", "">;

// !(A && B), which is (!A || !B).
def PredNotAll : Predicate<"PredNotAll">,
                 AssemblerPredicate<(not (all_of FeatureA, FeatureB))>;

// A || (B && C), which is (A || B) && (A || C).
def PredAnyOverAll : Predicate<"PredAnyOverAll">,
                     AssemblerPredicate<(any_of FeatureA,
                                                (all_of FeatureB, FeatureC))>;

// A && (B || !C), which is already in normal form.
def PredAllOverAny : Predicate<"PredAllOverAny">,
                     AssemblerPredicate<(all_of FeatureA,
                                                (any_of FeatureB,
                                                        (not FeatureC)))>;

// !!A, which is just A.
def PredDoubleNot : Predicate<"PredDoubleNot">,
                    AssemblerPredicate<(not (not (all_of FeatureA)))>;

// 'not' takes exactly one operand.
#ifdef BAD_NOT
def PredBad : Predicate<"PredBad">,
              AssemblerPredicate<(all_of (not FeatureA, FeatureB))>;
#endif

// Leaves must be subtarget features.
#ifdef BAD_LEAF
def NotAFeature : Predicate<"NotAFeature">;
def PredBad : Predicate<"PredBad">,
              AssemblerPredicate<(all_of (any_of NotAFeature))>;
#endif

// A nested operator needs at least one operand.
#ifdef BAD_EMPTY
def PredBad : Predicate<"PredBad">,
              AssemblerPredicate<(all_of FeatureA, (any_of))>;
#endif

class Big<string Asm> : Instruction {
  let Namespace = "Arch";
  let OutOperandList = (outs Regs:$dst);
  let InOperandList = (ins Regs:$src);
  let Size = 4;
  let AsmString = Asm # " $dst, $src";
}

class Small<string Asm> : Instruction {
  let Namespace = "Arch";
  let OutOperandList = (outs Regs:$dst);
  let InOperandList = (ins Regs:$src);
  let Size = 2;
  let AsmString = Asm # " $dst, $src";
}

def BigA : Big<"biga">;   def SmallA : Small<"smalla">;
def BigB : Big<"bigb">;   def SmallB : Small<"smallb">;
def BigC : Big<"bigc">;   def SmallC : Small<"smallc">;
def BigD : Big<"bigd">;   def SmallD : Small<"smalld">;

def : CompressPat<(BigA Regs:$dst, Regs:$src), (SmallA Regs:$dst, Regs:$src),
                  [PredNotAll]>;
def : CompressPat<(BigB Regs:$dst, Regs:$src), (SmallB Regs:$dst, Regs:$src),
                  [PredAnyOverAll]>;
def : CompressPat<(BigC Regs:$dst, Regs:$src), (SmallC Regs:$dst, Regs:$src),
                  [PredAllOverAny]>;
def : CompressPat<(BigD Regs:$dst, Regs:$src), (SmallD Regs:$dst, Regs:$src),
                  [PredDoubleNot]>;

// A malformed dag is only looked at once some pattern requires it.
def BigE : Big<"bige">;   def SmallE : Small<"smalle">;

// ERR: error: Invalid AssemblerCondDag!
#ifdef BAD_NOT
def : CompressPat<(BigE Regs:$dst, Regs:$src), (SmallE Regs:$dst, Regs:$src),
                  [PredBad]>;
#endif
#ifdef BAD_LEAF
def : CompressPat<(BigE Regs:$dst, Regs:$src), (SmallE Regs:$dst, Regs:$src),
                  [PredBad]>;
#endif
#ifdef BAD_EMPTY
def : CompressPat<(BigE Regs:$dst, Regs:$src), (SmallE Regs:$dst, Regs:$src),
                  [PredBad]>;
#endif

// A single clause of two negated literals stays one group.
// CHECK:      case Arch::BigA:
// CHECK-NEXT: if ((!STI.getFeatureBits()[Arch::FeatureA] || !STI.getFeatureBits()[Arch::FeatureB]) &&

// Distributing any_of over all_of yields two groups.
// CHECK:      case Arch::BigB:
// CHECK-NEXT: if ((STI.getFeatureBits()[Arch::FeatureA] || STI.getFeatureBits()[Arch::FeatureB]) &&
// CHECK-NEXT: (STI.getFeatureBits()[Arch::FeatureA] || STI.getFeatureBits()[Arch::FeatureC]) &&

// A conjunct of one literal is a required feature, not a group.
// CHECK:      case Arch::BigC:
// CHECK-NEXT: if (STI.getFeatureBits()[Arch::FeatureA] &&
// CHECK-NEXT: (STI.getFeatureBits()[Arch::FeatureB] || !STI.getFeatureBits()[Arch::FeatureC]) &&

// Double negation cancels, leaving a plain required feature.
// CHECK:      case Arch::BigD:
// CHECK-NEXT: if (STI.getFeatureBits()[Arch::FeatureA] &&
// CHECK-NEXT: MI.getOperand(0).isReg() &&
