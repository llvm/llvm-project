// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

// Verify that InstructionMatcher::isHigherPriorityThan defines a valid
// strict-weak ordering when comparing rules with different numbers of
// instruction predicates (1 vs 2) and operand matchers.
//
// RuleA: 2 instruction predicates (Opcode, GenericPredicate), Op2 has 2 ops
// RuleB: 2 instruction predicates (Opcode, MIFlags), Op1 & Op2 have 2 ops
// RuleC: 1 instruction predicate  (Opcode), Op2 has 3 ops
//
// Priority order must be: RuleA, RuleB, RuleC.

def RuleA : GICombineRule<
  (defs root:$dst),
  (match (G_SEXT $tmp, $x),
         (G_ADD $dst, $a, $tmp),
         [{ return true; }]),
  (apply (G_ADD $dst, $a, $x))>;

def RuleB : GICombineRule<
  (defs root:$dst),
  (match (G_SEXT $tmp1, $x),
         (G_SEXT $tmp2, $y),
         (G_ADD $dst, $tmp1, $tmp2, (MIFlags NoSWrap))),
  (apply (G_ADD $dst, $x, $y))>;

def RuleC : GICombineRule<
  (defs root:$dst),
  (match (G_SUB $tmp, $x, $y),
         (G_ADD $dst, $a, $tmp)),
  (apply (G_ADD $dst, $a, $x))>;

def MyCombiner : GICombiner<"GenMyCombiner", [RuleA, RuleC, RuleB]>;

// CHECK:      // Combiner Rule #0: RuleA
// CHECK:      // Combiner Rule #2: RuleB
// CHECK:      // Combiner Rule #1: RuleC
