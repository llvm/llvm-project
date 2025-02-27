// RUN: llvm-tblgen -gen-register-info -register-info-debug -I %p/../../include %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK
include "llvm/Target/Target.td"

// This file tests that when using `isArtificial` for subregisters in
// combination with `CoveredBySubRegs`, that TableGen infers the
// correct register classes, subregisters and lane masks, especially
// when the registers (that consist partially from artificial subregs)
// are used in tuples.
//
// The register hierarchy that this test implements is:
//
//                  ssub_hi   ssub
//                      \    /
//                       dsub
//             dsub_hi     /
//                 \      /
//                   qsub
//
// Where the _hi parts are artificial and where subregs ssub, dsub and qsub
// are all addressable as real registers.
//
// These are then used in {S0, S1, S2}, {D0, D1, D2} and {Q0, Q1, Q2},
// from which tuples are created.

class MyReg<string n, list<Register> subregs = []>
  : Register<n> {
  let Namespace = "Test";
  let SubRegs = subregs;
}

class MyClass<int size, list<ValueType> types, dag registers>
  : RegisterClass<"Test", types, size, registers> {
  let Size = size;
}

def ssub    : SubRegIndex< 32,   0>;
def ssub_hi : SubRegIndex< 32,  32>;
def dsub    : SubRegIndex< 64,   0>;
def dsub_hi : SubRegIndex< 64,  64>;
def qsub    : SubRegIndex<128,   0>;

def S0    : MyReg<"s0">;
def S1    : MyReg<"s1">;
def S2    : MyReg<"s2">;

let isArtificial = 1 in {
def S0_HI : MyReg<"s0_hi">;
def S1_HI : MyReg<"s1_hi">;
def S2_HI : MyReg<"s2_hi">;

def D0_HI : MyReg<"D0_hi">;
def D1_HI : MyReg<"D1_hi">;
def D2_HI : MyReg<"D2_hi">;
}

let SubRegIndices = [ssub, ssub_hi], CoveredBySubRegs = 1 in {
def D0    : MyReg<"d0", [S0, S0_HI]>;
def D1    : MyReg<"d1", [S1, S1_HI]>;
def D2    : MyReg<"d2", [S2, S2_HI]>;
}

let SubRegIndices = [dsub, dsub_hi], CoveredBySubRegs = 1 in {
def Q0    : MyReg<"q0", [D0, D0_HI]>;
def Q1    : MyReg<"q1", [D1, D1_HI]>;
def Q2    : MyReg<"q2", [D2, D2_HI]>;
}

def SRegs : MyClass<32, [i32], (sequence "S%u", 0, 2)>;
def DRegs : MyClass<64, [i64], (sequence "D%u", 0, 2)>;
def QRegs : MyClass<128, [i128], (sequence "Q%u", 0, 2)>;

def dsub0 : SubRegIndex<64>;
def dsub1 : SubRegIndex<64>;
def dsub2 : SubRegIndex<64>;

def ssub0 : SubRegIndex<32>;
def ssub1 : ComposedSubRegIndex<dsub1, ssub>;
def ssub2 : ComposedSubRegIndex<dsub2, ssub>;

def STuples2 : RegisterTuples<[ssub0, ssub1],
                             [(shl SRegs, 0), (shl SRegs, 1)]>;
def STuplesRC2 : MyClass<64, [untyped], (add STuples2)>;

def DTuples2 : RegisterTuples<[dsub0, dsub1],
                             [(shl DRegs, 0), (shl DRegs, 1)]>;
def DTuplesRC2 : MyClass<128, [untyped], (add DTuples2)>;

def STuples3 : RegisterTuples<[ssub0, ssub1, ssub2],
                             [(shl SRegs, 0), (shl SRegs, 1), (shl SRegs, 2)]>;
def STuplesRC3 : MyClass<96, [untyped], (add STuples3)>;

def DTuples3 : RegisterTuples<[dsub0, dsub1, dsub2],
                             [(shl DRegs, 0), (shl DRegs, 1), (shl DRegs, 2)]>;
def DTuplesRC3 : MyClass<192, [untyped], (add DTuples3)>;

def TestTarget : Target;

// CHECK:      RegisterClass SRegs:
// CHECK:      	LaneMask: 0000000000000001
// CHECK:      	HasDisjunctSubRegs: 0
// CHECK:      	CoveredBySubRegs: 0
// CHECK:      	Regs: S0 S1 S2
// CHECK:      	SubClasses: SRegs
// CHECK:      	SuperClasses:
//
// CHECK:      RegisterClass DRegs:
// CHECK:      	LaneMask: 0000000000000044
// CHECK:      	HasDisjunctSubRegs: 1
// CHECK:      	CoveredBySubRegs: 1
// CHECK:      	Regs: D0 D1 D2
// CHECK:      	SubClasses: DRegs
// CHECK:      	SuperClasses:
//
// CHECK:      RegisterClass QRegs:
// CHECK:      	LaneMask: 0000000000000045
// CHECK:      	HasDisjunctSubRegs: 1
// CHECK:      	CoveredBySubRegs: 1
// CHECK:      	Regs: Q0 Q1 Q2
// CHECK:      	SubClasses: QRegs
// CHECK:      	SuperClasses:
//
// CHECK:      SubRegIndex dsub:
// CHECK-NEXT: 	LaneMask: 0000000000000044
// CHECK:      SubRegIndex dsub0:
// CHECK-NEXT: 	LaneMask: 0000000000000044
// CHECK:      SubRegIndex dsub1:
// CHECK-NEXT: 	LaneMask: 0000000000000090
// CHECK:      SubRegIndex dsub2:
// CHECK-NEXT: 	LaneMask: 0000000000000120
// CHECK:      SubRegIndex dsub_hi:
// CHECK-NEXT: 	LaneMask: 0000000000000001
// CHECK:      SubRegIndex ssub:
// CHECK-NEXT: 	LaneMask: 0000000000000004
// CHECK:      SubRegIndex ssub0:
// CHECK-NEXT: 	LaneMask: 0000000000000008
// CHECK:      SubRegIndex ssub1:
// CHECK-NEXT: 	LaneMask: 0000000000000010
// CHECK:      SubRegIndex ssub2:
// CHECK-NEXT: 	LaneMask: 0000000000000020
// CHECK:      SubRegIndex ssub_hi:
// CHECK-NEXT: 	LaneMask: 0000000000000040
// CHECK:      SubRegIndex dsub1_then_ssub_hi:
// CHECK-NEXT: 	LaneMask: 0000000000000080
// CHECK:      SubRegIndex dsub2_then_ssub_hi:
// CHECK-NEXT: 	LaneMask: 0000000000000100
// CHECK:      SubRegIndex ssub_ssub1:
// CHECK-NEXT: 	LaneMask: 0000000000000014
// CHECK:      SubRegIndex dsub0_dsub1:
// CHECK-NEXT: 	LaneMask: 00000000000000D4
// CHECK:      SubRegIndex dsub1_dsub2:
// CHECK-NEXT: 	LaneMask: 00000000000001B0
// CHECK:      SubRegIndex ssub_ssub1_ssub2:
// CHECK-NEXT: 	LaneMask: 0000000000000034
// CHECK:      SubRegIndex ssub1_ssub2:
// CHECK-NEXT: 	LaneMask: 0000000000000030
// CHECK:      SubRegIndex ssub0_ssub1:
// CHECK-NEXT: 	LaneMask: 0000000000000018

// CHECK: Register D0:
// CHECK: 	CoveredBySubregs: 1
// CHECK: 	HasDisjunctSubRegs: 1
// CHECK: 	SubReg ssub = S0
// CHECK: 	SubReg ssub_hi = S0_HI
//
// CHECK: Register Q0:
// CHECK: 	CoveredBySubregs: 1
// CHECK: 	HasDisjunctSubRegs: 1
// CHECK: 	SubReg dsub = D0
// CHECK: 	SubReg dsub_hi = D0_HI
// CHECK: 	SubReg ssub = S0
// CHECK: 	SubReg ssub_hi = S0_HI
//
// CHECK: Register S0:
// CHECK: 	CoveredBySubregs: 0
// CHECK: 	HasDisjunctSubRegs: 0
//
// CHECK: Register D0_D1:
// CHECK: 	CoveredBySubregs: 1
// CHECK: 	HasDisjunctSubRegs: 1
// CHECK: 	SubReg dsub0 = D0
// CHECK: 	SubReg dsub1 = D1
// CHECK: 	SubReg ssub = S0
// CHECK: 	SubReg ssub1 = S1
// CHECK: 	SubReg ssub_hi = S0_HI
// CHECK: 	SubReg dsub1_then_ssub_hi = S1_HI
// CHECK: 	SubReg ssub_ssub1 = S0_S1
//
// CHECK: Register D0_D1_D2:
// CHECK: 	CoveredBySubregs: 1
// CHECK: 	HasDisjunctSubRegs: 1
// CHECK: 	SubReg dsub0 = D0
// CHECK: 	SubReg dsub1 = D1
// CHECK: 	SubReg dsub2 = D2
// CHECK: 	SubReg ssub = S0
// CHECK: 	SubReg ssub1 = S1
// CHECK: 	SubReg ssub2 = S2
// CHECK: 	SubReg ssub_hi = S0_HI
// CHECK: 	SubReg dsub1_then_ssub_hi = S1_HI
// CHECK: 	SubReg dsub2_then_ssub_hi = S2_HI
// CHECK: 	SubReg ssub_ssub1 = S0_S1
// CHECK: 	SubReg dsub0_dsub1 = D0_D1
// CHECK: 	SubReg dsub1_dsub2 = D1_D2
// CHECK: 	SubReg ssub_ssub1_ssub2 = S0_S1_S2
// CHECK: 	SubReg ssub1_ssub2 = S1_S2
//
// CHECK: Register S0_S1:
// CHECK: 	CoveredBySubregs: 1
// CHECK: 	HasDisjunctSubRegs: 1
// CHECK: 	SubReg ssub0 = S0
// CHECK: 	SubReg ssub1 = S1
//
// CHECK: Register S0_S1_S2:
// CHECK: 	CoveredBySubregs: 1
// CHECK: 	HasDisjunctSubRegs: 1
// CHECK: 	SubReg ssub0 = S0
// CHECK: 	SubReg ssub1 = S1
// CHECK: 	SubReg ssub2 = S2
// CHECK: 	SubReg ssub1_ssub2 = S1_S2
// CHECK: 	SubReg ssub0_ssub1 = S0_S1
