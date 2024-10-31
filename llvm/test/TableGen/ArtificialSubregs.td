// RUN: not llvm-tblgen -gen-register-info -register-info-debug -I %p/../../include %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK
include "llvm/Target/Target.td"

// CHECK: error: No SubRegIndex for S0_S1 in D0_D1_D2
// CHECK-NEXT: def DTuples3 : RegisterTuples<[dsub0, dsub1, dsub2],

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
