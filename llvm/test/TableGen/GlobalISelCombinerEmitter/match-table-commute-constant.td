// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=WithCanon %s | FileCheck %s --check-prefix=CANON
// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=NoCanon %s | FileCheck %s --check-prefix=NOCANON

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

// Keep the swapped version even when a canonicalizer moves constants to the
// RHS: it does not commute two constants, and constant folding does not handle
// every constant expression or vector splat recognized by immediate matching.
def MulOne : GICombineRule<
  (defs root:$dst),
  (match (G_MUL $dst, $x, 1)),
  (apply (GIReplaceReg $dst, $x))>;

// The LHS operand is defined by a G_CONSTANT, so both operands are constants
// and the constant is not moved. The swapped version is always emitted.
def MulConstOne : GICombineRule<
  (defs root:$dst),
  (match (G_CONSTANT $x, 5),
         (G_MUL $dst, $x, 1)),
  (apply (GIReplaceReg $dst, $x))>;

// G_ADD has no constant canonicalizer in either combiner.
def AddOne : GICombineRule<
  (defs root:$dst),
  (match (G_ADD $dst, $x, 1)),
  (apply (GIReplaceReg $dst, $x))>;

def WithCanon: GICombiner<"GenWithCanon", [
  G_MUL_commute_constant_to_rhs,
  MulOne,
  MulConstOne,
  AddOne
]>;

def NoCanon: GICombiner<"GenNoCanon", [
  MulOne,
  MulConstOne,
  AddOne
]>;

// CANON-DAG: // Combiner Rule #1: MulOne
// CANON-DAG: // Combiner Rule #1: MulOne @ [__MulOne_match_0[commuted]]
// CANON-DAG: // Combiner Rule #2: MulConstOne
// CANON-DAG: // Combiner Rule #2: MulConstOne @ [__MulConstOne_match_1[commuted]]
// CANON-DAG: // Combiner Rule #3: AddOne
// CANON-DAG: // Combiner Rule #3: AddOne @ [__AddOne_match_0[commuted]]

// NOCANON-DAG: // Combiner Rule #0: MulOne
// NOCANON-DAG: // Combiner Rule #0: MulOne @ [__MulOne_match_0[commuted]]
// NOCANON-DAG: // Combiner Rule #1: MulConstOne
// NOCANON-DAG: // Combiner Rule #1: MulConstOne @ [__MulConstOne_match_1[commuted]]
// NOCANON-DAG: // Combiner Rule #2: AddOne
// NOCANON-DAG: // Combiner Rule #2: AddOne @ [__AddOne_match_0[commuted]]
