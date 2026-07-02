// RUN: llvm-tblgen -gen-register-info -I %p/../../include %s -o - | FileCheck %s
// Companion to subreg-index-overflow.td: patterns that look like an overflow
// but are legitimate and must NOT be diagnosed by the size check.
include "llvm/Target/Target.td"

class MyReg<string n, list<Register> subs = []> : Register<n> {
  let Namespace = "Test";
  let SubRegs = subs;
  let CoveredBySubRegs = 1;
}
class MyClass<int sz, list<ValueType> types, dag regs>
  : RegisterClass<"Test", types, sz, regs> { let Size = sz; }

def lo : SubRegIndex<32, 0>;
def hi_gap : SubRegIndex<32, 64>; // leaves a [32,64) gap -> non-contiguous
def lo2 : SubRegIndex<32, 0>;
def hi2 : SubRegIndex<32, 32>;

foreach i = 0-7 in { def R#i : MyReg<"r"#i>; }
def GPR : MyClass<32, [i32], (sequence "R%u", 0, 7)>;

// (1) Strided/spaced tuple: sub-registers at [0,32) and [64,96) extend past the
// 64-bit class size, but are non-contiguous, so this is allowed.
def StridedPair : RegisterTuples<[lo, hi_gap], [(add R0, R2), (add R1, R3)]>;
def StridedRC : MyClass<64, [untyped], (add StridedPair)>;

// (2) Untyped class: its size is a placeholder, not a bit width, so a 64-bit
// pair living in it must not be diagnosed as overflowing.
def DensePair : RegisterTuples<[lo2, hi2], [(add R4, R6), (add R5, R7)]>;
def UntypedRC : MyClass<8, [untyped], (add DensePair)>;

def TestTarget : Target;

// Both synthesized tuples must be emitted, i.e. neither was rejected by the
// sub-register overflow check.
// CHECK-DAG: R0_R1 =
// CHECK-DAG: R4_R5 =
