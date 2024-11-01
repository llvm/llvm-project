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
def qsub_hi : SubRegIndex<128, 128>;

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
// CHECK-NEXT: 	SpillSize: { Default:32 }
// CHECK-NEXT: 	SpillAlignment: { Default:32 }
// CHECK-NEXT: 	NumRegs: 3
// CHECK-NEXT: 	LaneMask: 0000000000000001
// CHECK-NEXT: 	HasDisjunctSubRegs: 0
// CHECK-NEXT: 	CoveredBySubRegs: 0
// CHECK-NEXT: 	Allocatable: 1
// CHECK-NEXT: 	AllocationPriority: 0
// CHECK-NEXT: 	BaseClassOrder: None
// CHECK-NEXT: 	Regs: S0 S1 S2
// CHECK-NEXT: 	SubClasses: SRegs
// CHECK-NEXT: 	SuperClasses:
// CHECK-NEXT: RegisterClass DRegs:
// CHECK-NEXT: 	SpillSize: { Default:64 }
// CHECK-NEXT: 	SpillAlignment: { Default:64 }
// CHECK-NEXT: 	NumRegs: 3
// CHECK-NEXT: 	LaneMask: 0000000000000008
// CHECK-NEXT: 	HasDisjunctSubRegs: 1
// CHECK-NEXT: 	CoveredBySubRegs: 1
// CHECK-NEXT: 	Allocatable: 1
// CHECK-NEXT: 	AllocationPriority: 0
// CHECK-NEXT: 	BaseClassOrder: None
// CHECK-NEXT: 	Regs: D0 D1 D2
// CHECK-NEXT: 	SubClasses: DRegs
// CHECK-NEXT: 	SuperClasses:
// CHECK-NEXT: RegisterClass STuplesRC2:
// CHECK-NEXT: 	SpillSize: { Default:64 }
// CHECK-NEXT: 	SpillAlignment: { Default:64 }
// CHECK-NEXT: 	NumRegs: 2
// CHECK-NEXT: 	LaneMask: 0000000000000030
// CHECK-NEXT: 	HasDisjunctSubRegs: 1
// CHECK-NEXT: 	CoveredBySubRegs: 1
// CHECK-NEXT: 	Allocatable: 1
// CHECK-NEXT: 	AllocationPriority: 0
// CHECK-NEXT: 	BaseClassOrder: None
// CHECK-NEXT: 	Regs: S0_S1 S1_S2
// CHECK-NEXT: 	SubClasses: STuplesRC2
// CHECK-NEXT: 	SuperClasses:
// CHECK-NEXT: RegisterClass STuplesRC3:
// CHECK-NEXT: 	SpillSize: { Default:96 }
// CHECK-NEXT: 	SpillAlignment: { Default:96 }
// CHECK-NEXT: 	NumRegs: 1
// CHECK-NEXT: 	LaneMask: 0000000000000070
// CHECK-NEXT: 	HasDisjunctSubRegs: 1
// CHECK-NEXT: 	CoveredBySubRegs: 1
// CHECK-NEXT: 	Allocatable: 1
// CHECK-NEXT: 	AllocationPriority: 0
// CHECK-NEXT: 	BaseClassOrder: None
// CHECK-NEXT: 	Regs: S0_S1_S2
// CHECK-NEXT: 	SubClasses: STuplesRC3
// CHECK-NEXT: 	SuperClasses:
// CHECK-NEXT: RegisterClass QRegs:
// CHECK-NEXT: 	SpillSize: { Default:128 }
// CHECK-NEXT: 	SpillAlignment: { Default:128 }
// CHECK-NEXT: 	NumRegs: 3
// CHECK-NEXT: 	LaneMask: 0000000000000088
// CHECK-NEXT: 	HasDisjunctSubRegs: 1
// CHECK-NEXT: 	CoveredBySubRegs: 1
// CHECK-NEXT: 	Allocatable: 1
// CHECK-NEXT: 	AllocationPriority: 0
// CHECK-NEXT: 	BaseClassOrder: None
// CHECK-NEXT: 	Regs: Q0 Q1 Q2
// CHECK-NEXT: 	SubClasses: QRegs
// CHECK-NEXT: 	SuperClasses:
// CHECK-NEXT: RegisterClass DTuplesRC2:
// CHECK-NEXT: 	SpillSize: { Default:128 }
// CHECK-NEXT: 	SpillAlignment: { Default:128 }
// CHECK-NEXT: 	NumRegs: 2
// CHECK-NEXT: 	LaneMask: 00000000000001A8
// CHECK-NEXT: 	HasDisjunctSubRegs: 1
// CHECK-NEXT: 	CoveredBySubRegs: 1
// CHECK-NEXT: 	Allocatable: 1
// CHECK-NEXT: 	AllocationPriority: 0
// CHECK-NEXT: 	BaseClassOrder: None
// CHECK-NEXT: 	Regs: D0_D1 D1_D2
// CHECK-NEXT: 	SubClasses: DTuplesRC2
// CHECK-NEXT: 	SuperClasses:
// CHECK-NEXT: RegisterClass DTuplesRC3:
// CHECK-NEXT: 	SpillSize: { Default:192 }
// CHECK-NEXT: 	SpillAlignment: { Default:192 }
// CHECK-NEXT: 	NumRegs: 1
// CHECK-NEXT: 	LaneMask: 00000000000003E8
// CHECK-NEXT: 	HasDisjunctSubRegs: 1
// CHECK-NEXT: 	CoveredBySubRegs: 1
// CHECK-NEXT: 	Allocatable: 1
// CHECK-NEXT: 	AllocationPriority: 0
// CHECK-NEXT: 	BaseClassOrder: None
// CHECK-NEXT: 	Regs: D0_D1_D2
// CHECK-NEXT: 	SubClasses: DTuplesRC3
// CHECK-NEXT: 	SuperClasses:
// CHECK-NEXT: SubRegIndex dsub:
// CHECK-NEXT: 	LaneMask: 0000000000000088
// CHECK-NEXT: 	AllSuperRegsCovered: 1
// CHECK-NEXT: 	Offset: { Default:0 }
// CHECK-NEXT: 	Size: { Default:64 }
// CHECK-NEXT: SubRegIndex dsub0:
// CHECK-NEXT: 	LaneMask: 0000000000000088
// CHECK-NEXT: 	AllSuperRegsCovered: 1
// CHECK-NEXT: 	Offset: { Default:0 }
// CHECK-NEXT: 	Size: { Default:64 }
// CHECK-NEXT: SubRegIndex dsub1:
// CHECK-NEXT: 	LaneMask: 0000000000000120
// CHECK-NEXT: 	AllSuperRegsCovered: 1
// CHECK-NEXT: 	Offset: { Default:0 }
// CHECK-NEXT: 	Size: { Default:64 }
// CHECK-NEXT: SubRegIndex dsub2:
// CHECK-NEXT: 	LaneMask: 0000000000000240
// CHECK-NEXT: 	AllSuperRegsCovered: 1
// CHECK-NEXT: 	Offset: { Default:0 }
// CHECK-NEXT: 	Size: { Default:64 }
// CHECK-NEXT: SubRegIndex dsub_hi:
// CHECK-NEXT: 	LaneMask: 0000000000000001
// CHECK-NEXT: 	AllSuperRegsCovered: 1
// CHECK-NEXT: 	Offset: { Default:64 }
// CHECK-NEXT: 	Size: { Default:64 }
// CHECK-NEXT: SubRegIndex qsub:
// CHECK-NEXT: 	LaneMask: 0000000000000002
// CHECK-NEXT: 	AllSuperRegsCovered: 1
// CHECK-NEXT: 	Offset: { Default:0 }
// CHECK-NEXT: 	Size: { Default:128 }
// CHECK-NEXT: SubRegIndex qsub_hi:
// CHECK-NEXT: 	LaneMask: 0000000000000004
// CHECK-NEXT: 	AllSuperRegsCovered: 1
// CHECK-NEXT: 	Offset: { Default:128 }
// CHECK-NEXT: 	Size: { Default:128 }
// CHECK-NEXT: SubRegIndex ssub:
// CHECK-NEXT: 	LaneMask: 0000000000000008
// CHECK-NEXT: 	AllSuperRegsCovered: 1
// CHECK-NEXT: 	Offset: { Default:0 }
// CHECK-NEXT: 	Size: { Default:32 }
// CHECK-NEXT: SubRegIndex ssub0:
// CHECK-NEXT: 	LaneMask: 0000000000000010
// CHECK-NEXT: 	AllSuperRegsCovered: 1
// CHECK-NEXT: 	Offset: { Default:0 }
// CHECK-NEXT: 	Size: { Default:32 }
// CHECK-NEXT: SubRegIndex ssub1:
// CHECK-NEXT: 	LaneMask: 0000000000000020
// CHECK-NEXT: 	AllSuperRegsCovered: 1
// CHECK-NEXT: 	Offset: { Default:0 }
// CHECK-NEXT: 	Size: { Default:32 }
// CHECK-NEXT: SubRegIndex ssub2:
// CHECK-NEXT: 	LaneMask: 0000000000000040
// CHECK-NEXT: 	AllSuperRegsCovered: 1
// CHECK-NEXT: 	Offset: { Default:0 }
// CHECK-NEXT: 	Size: { Default:32 }
// CHECK-NEXT: SubRegIndex ssub_hi:
// CHECK-NEXT: 	LaneMask: 0000000000000080
// CHECK-NEXT: 	AllSuperRegsCovered: 1
// CHECK-NEXT: 	Offset: { Default:32 }
// CHECK-NEXT: 	Size: { Default:32 }
// CHECK-NEXT: SubRegIndex dsub1_then_ssub_hi:
// CHECK-NEXT: 	LaneMask: 0000000000000100
// CHECK-NEXT: 	AllSuperRegsCovered: 1
// CHECK-NEXT: 	Offset: { Default:32 }
// CHECK-NEXT: 	Size: { Default:32 }
// CHECK-NEXT: SubRegIndex dsub2_then_ssub_hi:
// CHECK-NEXT: 	LaneMask: 0000000000000200
// CHECK-NEXT: 	AllSuperRegsCovered: 1
// CHECK-NEXT: 	Offset: { Default:32 }
// CHECK-NEXT: 	Size: { Default:32 }
// CHECK-NEXT: SubRegIndex ssub_ssub1:
// CHECK-NEXT: 	LaneMask: 0000000000000028
// CHECK-NEXT: 	AllSuperRegsCovered: 1
// CHECK-NEXT: 	Offset: { Default:65535 }
// CHECK-NEXT: 	Size: { Default:64 }
// CHECK-NEXT: SubRegIndex dsub0_dsub1:
// CHECK-NEXT: 	LaneMask: 00000000000001A8
// CHECK-NEXT: 	AllSuperRegsCovered: 1
// CHECK-NEXT: 	Offset: { Default:65535 }
// CHECK-NEXT: 	Size: { Default:128 }
// CHECK-NEXT: SubRegIndex dsub1_dsub2:
// CHECK-NEXT: 	LaneMask: 0000000000000360
// CHECK-NEXT: 	AllSuperRegsCovered: 1
// CHECK-NEXT: 	Offset: { Default:65535 }
// CHECK-NEXT: 	Size: { Default:128 }
// CHECK-NEXT: SubRegIndex ssub_ssub1_ssub2:
// CHECK-NEXT: 	LaneMask: 0000000000000068
// CHECK-NEXT: 	AllSuperRegsCovered: 1
// CHECK-NEXT: 	Offset: { Default:65535 }
// CHECK-NEXT: 	Size: { Default:96 }
// CHECK-NEXT: SubRegIndex ssub1_ssub2:
// CHECK-NEXT: 	LaneMask: 0000000000000060
// CHECK-NEXT: 	AllSuperRegsCovered: 1
// CHECK-NEXT: 	Offset: { Default:65535 }
// CHECK-NEXT: 	Size: { Default:64 }
// CHECK-NEXT: SubRegIndex ssub0_ssub1:
// CHECK-NEXT: 	LaneMask: 0000000000000030
// CHECK-NEXT: 	AllSuperRegsCovered: 1
// CHECK-NEXT: 	Offset: { Default:65535 }
// CHECK-NEXT: 	Size: { Default:64 }
// CHECK-NEXT: Register D0:
// CHECK-NEXT: 	CostPerUse: 0
// CHECK-NEXT: 	CoveredBySubregs: 1
// CHECK-NEXT: 	HasDisjunctSubRegs: 1
// CHECK-NEXT: 	SubReg ssub = S0
// CHECK-NEXT: 	SubReg ssub_hi = S0_HI
// CHECK-NEXT: Register D1:
// CHECK-NEXT: 	CostPerUse: 0
// CHECK-NEXT: 	CoveredBySubregs: 1
// CHECK-NEXT: 	HasDisjunctSubRegs: 1
// CHECK-NEXT: 	SubReg ssub = S1
// CHECK-NEXT: 	SubReg ssub_hi = S1_HI
// CHECK-NEXT: Register D2:
// CHECK-NEXT: 	CostPerUse: 0
// CHECK-NEXT: 	CoveredBySubregs: 1
// CHECK-NEXT: 	HasDisjunctSubRegs: 1
// CHECK-NEXT: 	SubReg ssub = S2
// CHECK-NEXT: 	SubReg ssub_hi = S2_HI
// CHECK-NEXT: Register Q0:
// CHECK-NEXT: 	CostPerUse: 0
// CHECK-NEXT: 	CoveredBySubregs: 1
// CHECK-NEXT: 	HasDisjunctSubRegs: 1
// CHECK-NEXT: 	SubReg dsub = D0
// CHECK-NEXT: 	SubReg dsub_hi = D0_HI
// CHECK-NEXT: 	SubReg ssub = S0
// CHECK-NEXT: 	SubReg ssub_hi = S0_HI
// CHECK-NEXT: Register Q1:
// CHECK-NEXT: 	CostPerUse: 0
// CHECK-NEXT: 	CoveredBySubregs: 1
// CHECK-NEXT: 	HasDisjunctSubRegs: 1
// CHECK-NEXT: 	SubReg dsub = D1
// CHECK-NEXT: 	SubReg dsub_hi = D1_HI
// CHECK-NEXT: 	SubReg ssub = S1
// CHECK-NEXT: 	SubReg ssub_hi = S1_HI
// CHECK-NEXT: Register Q2:
// CHECK-NEXT: 	CostPerUse: 0
// CHECK-NEXT: 	CoveredBySubregs: 1
// CHECK-NEXT: 	HasDisjunctSubRegs: 1
// CHECK-NEXT: 	SubReg dsub = D2
// CHECK-NEXT: 	SubReg dsub_hi = D2_HI
// CHECK-NEXT: 	SubReg ssub = S2
// CHECK-NEXT: 	SubReg ssub_hi = S2_HI
// CHECK-NEXT: Register S0:
// CHECK-NEXT: 	CostPerUse: 0
// CHECK-NEXT: 	CoveredBySubregs: 0
// CHECK-NEXT: 	HasDisjunctSubRegs: 0
// CHECK-NEXT: Register S1:
// CHECK-NEXT: 	CostPerUse: 0
// CHECK-NEXT: 	CoveredBySubregs: 0
// CHECK-NEXT: 	HasDisjunctSubRegs: 0
// CHECK-NEXT: Register S2:
// CHECK-NEXT: 	CostPerUse: 0
// CHECK-NEXT: 	CoveredBySubregs: 0
// CHECK-NEXT: 	HasDisjunctSubRegs: 0
// CHECK-NEXT: Register D0_HI:
// CHECK-NEXT: 	CostPerUse: 0
// CHECK-NEXT: 	CoveredBySubregs: 0
// CHECK-NEXT: 	HasDisjunctSubRegs: 0
// CHECK-NEXT: Register D1_HI:
// CHECK-NEXT: 	CostPerUse: 0
// CHECK-NEXT: 	CoveredBySubregs: 0
// CHECK-NEXT: 	HasDisjunctSubRegs: 0
// CHECK-NEXT: Register D2_HI:
// CHECK-NEXT: 	CostPerUse: 0
// CHECK-NEXT: 	CoveredBySubregs: 0
// CHECK-NEXT: 	HasDisjunctSubRegs: 0
// CHECK-NEXT: Register S0_HI:
// CHECK-NEXT: 	CostPerUse: 0
// CHECK-NEXT: 	CoveredBySubregs: 0
// CHECK-NEXT: 	HasDisjunctSubRegs: 0
// CHECK-NEXT: Register S1_HI:
// CHECK-NEXT: 	CostPerUse: 0
// CHECK-NEXT: 	CoveredBySubregs: 0
// CHECK-NEXT: 	HasDisjunctSubRegs: 0
// CHECK-NEXT: Register S2_HI:
// CHECK-NEXT: 	CostPerUse: 0
// CHECK-NEXT: 	CoveredBySubregs: 0
// CHECK-NEXT: 	HasDisjunctSubRegs: 0
// CHECK-NEXT: Register D0_D1:
// CHECK-NEXT: 	CostPerUse: 0
// CHECK-NEXT: 	CoveredBySubregs: 1
// CHECK-NEXT: 	HasDisjunctSubRegs: 1
// CHECK-NEXT: 	SubReg dsub0 = D0
// CHECK-NEXT: 	SubReg dsub1 = D1
// CHECK-NEXT: 	SubReg ssub = S0
// CHECK-NEXT: 	SubReg ssub1 = S1
// CHECK-NEXT: 	SubReg ssub_hi = S0_HI
// CHECK-NEXT: 	SubReg dsub1_then_ssub_hi = S1_HI
// CHECK-NEXT: 	SubReg ssub_ssub1 = S0_S1
// CHECK-NEXT: Register D1_D2:
// CHECK-NEXT: 	CostPerUse: 0
// CHECK-NEXT: 	CoveredBySubregs: 1
// CHECK-NEXT: 	HasDisjunctSubRegs: 1
// CHECK-NEXT: 	SubReg dsub0 = D1
// CHECK-NEXT: 	SubReg dsub1 = D2
// CHECK-NEXT: 	SubReg ssub = S1
// CHECK-NEXT: 	SubReg ssub1 = S2
// CHECK-NEXT: 	SubReg ssub_hi = S1_HI
// CHECK-NEXT: 	SubReg dsub1_then_ssub_hi = S2_HI
// CHECK-NEXT: 	SubReg ssub_ssub1 = S1_S2
// CHECK-NEXT: Register D0_D1_D2:
// CHECK-NEXT: 	CostPerUse: 0
// CHECK-NEXT: 	CoveredBySubregs: 1
// CHECK-NEXT: 	HasDisjunctSubRegs: 1
// CHECK-NEXT: 	SubReg dsub0 = D0
// CHECK-NEXT: 	SubReg dsub1 = D1
// CHECK-NEXT: 	SubReg dsub2 = D2
// CHECK-NEXT: 	SubReg ssub = S0
// CHECK-NEXT: 	SubReg ssub1 = S1
// CHECK-NEXT: 	SubReg ssub2 = S2
// CHECK-NEXT: 	SubReg ssub_hi = S0_HI
// CHECK-NEXT: 	SubReg dsub1_then_ssub_hi = S1_HI
// CHECK-NEXT: 	SubReg dsub2_then_ssub_hi = S2_HI
// CHECK-NEXT: 	SubReg ssub_ssub1 = S0_S1
// CHECK-NEXT: 	SubReg dsub0_dsub1 = D0_D1
// CHECK-NEXT: 	SubReg dsub1_dsub2 = D1_D2
// CHECK-NEXT: 	SubReg ssub_ssub1_ssub2 = S0_S1_S2
// CHECK-NEXT: 	SubReg ssub1_ssub2 = S1_S2
// CHECK-NEXT: Register S0_S1:
// CHECK-NEXT: 	CostPerUse: 0
// CHECK-NEXT: 	CoveredBySubregs: 1
// CHECK-NEXT: 	HasDisjunctSubRegs: 1
// CHECK-NEXT: 	SubReg ssub0 = S0
// CHECK-NEXT: 	SubReg ssub1 = S1
// CHECK-NEXT: Register S1_S2:
// CHECK-NEXT: 	CostPerUse: 0
// CHECK-NEXT: 	CoveredBySubregs: 1
// CHECK-NEXT: 	HasDisjunctSubRegs: 1
// CHECK-NEXT: 	SubReg ssub0 = S1
// CHECK-NEXT: 	SubReg ssub1 = S2
// CHECK-NEXT: Register S0_S1_S2:
// CHECK-NEXT: 	CostPerUse: 0
// CHECK-NEXT: 	CoveredBySubregs: 1
// CHECK-NEXT: 	HasDisjunctSubRegs: 1
// CHECK-NEXT: 	SubReg ssub0 = S0
// CHECK-NEXT: 	SubReg ssub1 = S1
// CHECK-NEXT: 	SubReg ssub2 = S2
// CHECK-NEXT: 	SubReg ssub1_ssub2 = S1_S2
// CHECK-NEXT: 	SubReg ssub0_ssub1 = S0_S1
