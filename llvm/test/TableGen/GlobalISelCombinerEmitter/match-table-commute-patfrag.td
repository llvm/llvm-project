// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | FileCheck %s
// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=UndefCombiner %s -o %t
// RUN: FileCheck %s --check-prefix=UNDEF-ZERO < %t
// RUN: FileCheck %s --check-prefix=UNDEF-ONES < %t
// RUN: FileCheck %s --check-prefix=UNDEF-PROPAGATE < %t

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def XorFrag : GICombinePatFrag<
  (outs root:$dst), (ins gi_mo:$a),
  [(pattern (G_XOR $dst, $a, $b))]>;

// An unbound parameter is not a constraint. The single permutation must fit
// within MaxPermutations, just as it would without the PatFrag.
def FreeOperands : GICombineRule<
  (defs root:$dst),
  (match (XorFrag $dst, $p):$f),
  (apply (G_CONSTANT $dst, (i32 0)))> {
  let MaxPermutations = 1;
}

// Constraints on the caller's argument must still commute the fragment.
def CallerCXX : GICombineRule<
  (defs root:$dst),
  (match (XorFrag $dst, $p):$f, "return isFoo(${p});"),
  (apply (G_CONSTANT $dst, (i32 0)))>;

// A local constraint also requires commutation when the caller's argument is
// otherwise unconstrained.
def ConstrainedXorFrag : GICombinePatFrag<
  (outs root:$dst), (ins gi_mo:$a),
  [(pattern (G_XOR $dst, $a, $b), "return isFoo(${a});")]>;

def LocalCXX : GICombineRule<
  (defs root:$dst),
  (match (ConstrainedXorFrag $dst, $p):$f),
  (apply (G_CONSTANT $dst, (i32 0)))>;

def AddFrag : GICombinePatFrag<
  (outs root:$dst), (ins gi_mo:$a, gi_mo:$b),
  [(pattern (G_ADD $dst, $a, $b))]>;

// Distinct parameters bound to the same caller operand need no commutation.
def SameArgument : GICombineRule<
  (defs root:$dst),
  (match (AddFrag $dst, $x, $x):$f),
  (apply (G_SHL $dst, $x, (i32 1)))> {
  let MaxPermutations = 1;
}

def MyCombiner : GICombiner<"GenMyCombiner", [
  FreeOperands, CallerCXX, LocalCXX, SameArgument
]>;

// CHECK-DAG: return isFoo(State.MIs[0]->getOperand(1));
// CHECK-DAG: return isFoo(State.MIs[0]->getOperand(2));
// CHECK-DAG: // Combiner Rule #0: FreeOperands @ [f[0]]
// CHECK-DAG: // Combiner Rule #1: CallerCXX @ [f[0]]
// CHECK-DAG: // Combiner Rule #1: CallerCXX @ [f.__XorFrag_alt0_pattern_0[commuted], f[0]]
// CHECK-DAG: // Combiner Rule #2: LocalCXX @ [f[0]]
// CHECK-DAG: // Combiner Rule #2: LocalCXX @ [f.__ConstrainedXorFrag_alt0_pattern_0[commuted], f[0]]
// CHECK-DAG: // Combiner Rule #3: SameArgument @ [f[0]]

// Each opcode needs one match per undef position. G_SUB still requires both
// explicit alternatives because it is not commutable.
def UndefCombiner : GICombiner<"GenUndefCombiner", [
  undef_to_int_zero, undef_to_negative_one, propagate_undef_any_op
]>;

// UNDEF-ZERO-COUNT-4: // Combiner Rule #0: undef_to_int_zero
// UNDEF-ZERO-NOT: // Combiner Rule #0: undef_to_int_zero
// UNDEF-ONES-COUNT-2: // Combiner Rule #1: undef_to_negative_one
// UNDEF-ONES-NOT: // Combiner Rule #1: undef_to_negative_one
// UNDEF-PROPAGATE-COUNT-6: // Combiner Rule #2: propagate_undef_any_op
// UNDEF-PROPAGATE-NOT: // Combiner Rule #2: propagate_undef_any_op
