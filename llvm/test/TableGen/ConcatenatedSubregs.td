// RUN: llvm-tblgen -gen-register-info -register-info-debug -I %p/../../include %s -o /dev/null 2>&1 | FileCheck %s
// Checks that tablegen correctly and completely infers subregister relations.
include "llvm/Target/Target.td"

class MyReg<string n, list<Register> subregs = []>
  : Register<n> {
  let Namespace = "Test";
  let SubRegs = subregs;
  let CoveredBySubRegs = 1;
}
class MyClass<int size, list<ValueType> types, dag registers>
  : RegisterClass<"Test", types, size, registers> {
  let Size = size;
}

// Register Example:
// D0_D1 -- D0 (sub0) -- S0 (ssub0)
//       \            \- S1 (ssub1)
//        \ D1 (sub1) -- S2 (ssub2)
//                    \- S3 (ssub3)

def sub0 : SubRegIndex<32>;
def sub1 : SubRegIndex<32, 32>;
def sub2 : SubRegIndex<32, 64>;

def ssub0 : SubRegIndex<-1>;
def ssub1 : SubRegIndex<16, 16>;
def ssub2 : ComposedSubRegIndex<sub1, ssub0>;
def ssub3 : ComposedSubRegIndex<sub1, ssub1>;
def ssub4 : ComposedSubRegIndex<sub2, ssub0>;

def S0 : MyReg<"s0">;
def S1 : MyReg<"s1">;
def S2 : MyReg<"s2">;
def S3 : MyReg<"s3">;
def S4 : MyReg<"s4">;
def S5 : MyReg<"s5">;
def S6 : MyReg<"s6">;
def S7 : MyReg<"s7">;
def S8 : MyReg<"s8">;
def S9 : MyReg<"s9">;
def S10 : MyReg<"s10">;
def S11 : MyReg<"s11">;
def S12 : MyReg<"s12">;
def S13 : MyReg<"s13">;
def S14 : MyReg<"s14">;
def S15 : MyReg<"s15">;
def SRegs : MyClass<16, [i16], (sequence "S%u", 0, 15)>;

let SubRegIndices = [ssub0, ssub1] in {
def D0 : MyReg<"d0", [S0, S1]>;
def D1 : MyReg<"d1", [S2, S3]>;
def D2 : MyReg<"d2", [S4, S5]>;
def D3 : MyReg<"d3", [S6, S7]>;
def D4 : MyReg<"d4", [S8, S9]>;
def D5 : MyReg<"d5", [S10, S11]>;
def D6 : MyReg<"d6", [S12, S13]>;
def D7 : MyReg<"d7", [S14, S15]>;
}

def DRegs : MyClass<32, [i32], (sequence "D%u", 0, 7)>;

def Dtup2regs : RegisterTuples<[sub0, sub1],
                           [(shl DRegs, 0), (shl DRegs, 1)]>;
def Dtup2 : MyClass<64, [untyped], (add Dtup2regs)>;

def Stup2_odds_regs : RegisterTuples<[ssub0, ssub1],
                                     [(decimate (shl SRegs, 1), 2),
                                      (decimate (shl SRegs, 2), 2)]>;
def Stup2 : MyClass<32, [untyped], (interleave DRegs, Stup2_odds_regs)>;

def Stup5 : RegisterTuples<[ssub0, ssub1, ssub2, ssub3, ssub4], [
                            (shl SRegs, 0),
                            (shl SRegs, 1),
                            (shl SRegs, 2),
                            (shl SRegs, 3),
                            (shl SRegs, 4)
                           ]>;


def TestTarget : Target;

// CHECK-LABEL: RegisterClass SRegs:
// CHECK: CoveredBySubRegs: 1
// CHECK: Regs: S0 S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 S11 S12 S13 S14 S15

// CHECK-LABEL: RegisterClass Stup2:
// CHECK: CoveredBySubRegs: 1
// CHECK: Regs: D0 D1 D2 D3 D4 D5 D6 D7 S1_S2 S3_S4 S5_S6 S7_S8 S9_S10 S11_S12 S13_S14
// CHECK-LABEL: RegisterClass DRegs:

// CHECK-LABEL: SubRegIndex ssub1:
// CHECK:        Offset, Size: 16, 16
// CHECK-LABEL: SubRegIndex sub0:
// CHECK-LABEL: SubRegIndex sub1:
// CHECK-LABEL: SubRegIndex sub2:
// Check inferred indexes:
// CHECK-LABEL: SubRegIndex ssub1_ssub2:
// CHECK:         Offset, Size: 16, 65535
// CHECK-LABEL: SubRegIndex ssub3_ssub4:
// CHECK-LABEL: SubRegIndex ssub0_ssub1_ssub2_ssub3:
// CHECK:         Offset, Size: 65535, 65535
// CHECK-LABEL: SubRegIndex ssub1_ssub2_ssub3_ssub4:

// Check that all subregs are generated on some examples
// CHECK-LABEL: Register D0:
// CHECK: HasDisjunctSubRegs: 1
// CHECK-NEXT: SubReg ssub0 = S0
// CHECK-NEXT: SubReg ssub1 = S1

// CHECK-LABEL: Register S9_S10_S11_S12_S13:
// CHECK: HasDisjunctSubRegs: 1
// CHECK-NEXT: SubReg ssub0 = S9
// CHECK-NEXT: SubReg ssub1 = S10
// CHECK-NEXT: SubReg ssub2 = S11
// CHECK-NEXT: SubReg ssub3 = S12
// CHECK-NEXT: SubReg ssub4 = S13
// CHECK-NEXT: SubReg sub0 = S9_S10
// CHECK-NEXT: SubReg sub1 = S11_S12
// CHECK-NEXT: SubReg ssub1_ssub2 = D5
// CHECK-NEXT: SubReg ssub3_ssub4 = D6
// CHECK-NEXT: SubReg ssub1_ssub2_ssub3_ssub4 = D5_D6

// CHECK-LABEL: Register S10_S11_S12_S13_S14:
// CHECK: HasDisjunctSubRegs: 1
// CHECK-NEXT: SubReg ssub0 = S10
// CHECK-NEXT: SubReg ssub1 = S11
// CHECK-NEXT: SubReg ssub2 = S12
// CHECK-NEXT: SubReg ssub3 = S13
// CHECK-NEXT: SubReg ssub4 = S14
// CHECK-NEXT: SubReg sub0 = D5
// CHECK-NEXT: SubReg sub1 = D6
// CHECK-NEXT: SubReg ssub1_ssub2 = S11_S12
// CHECK-NEXT: SubReg ssub3_ssub4 = S13_S14
// CHECK-NEXT: SubReg ssub0_ssub1_ssub2_ssub3 = D5_D6
