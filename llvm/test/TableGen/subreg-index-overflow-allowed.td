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

// (3) Exact fit: a typed 64-bit pair whose two 32-bit halves tile exactly
// [0,64). offset+size equals the register size, which is the boundary case;
// the check rejects sub-registers that cover strictly more, so this is fine.
def elo : SubRegIndex<32, 0>;
def ehi : SubRegIndex<32, 32>;
def E0 : MyReg<"e0">;
def E1 : MyReg<"e1">;
def EGPR : MyClass<32, [i32], (add E0, E1)>;
let SubRegIndices = [elo, ehi] in
  def EP0 : MyReg<"ep0", [E0, E1]>;
def ExactRC : MyClass<64, [i64], (add EP0)>;

// (4) Not covered by sub-registers: a register that does not set
// CoveredBySubRegs makes no full-tiling promise, so an explicit SubRegIndex
// running past its size ([0,32)+[32,96) = 96 > 64) is that register's own
// business and must not be diagnosed.
def nlo : SubRegIndex<32, 0>;
def nhi : SubRegIndex<64, 32>;
def N0 : MyReg<"n0">;
def N1 : MyReg<"n1">;
def NGPR : MyClass<32, [i32], (add N0, N1)>;
let SubRegIndices = [nlo, nhi] in
  def NC0 : Register<"nc0"> { let Namespace = "Test"; let SubRegs = [N0, N1]; }
def NotCoveredRC : MyClass<64, [i64], (add NC0)>;

def TestTarget : Target;

// Every synthesized/explicit register below must be emitted, i.e. none was
// rejected by the sub-register overflow check.
// CHECK-DAG: R0_R1 =
// CHECK-DAG: R4_R5 =
// CHECK-DAG: EP0 =
// CHECK-DAG: NC0 =
