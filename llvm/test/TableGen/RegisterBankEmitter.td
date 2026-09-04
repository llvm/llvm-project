// RUN: llvm-tblgen -gen-register-bank -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def MyTarget : Target;
def R0 : Register<"r0">;
let Size = 32 in {
  def ClassA : RegisterClass<"MyTarget", [i32], 32, (add R0)>;
  def ClassB : RegisterClass<"MyTarget", [i1], 32, (add ClassA)>;
}

// CHECK: enum PartialMappingIdx
// CHECK:   PMI_GPR1,
// CHECK:   PMI_GPR32,
// CHECK:   PMI_TestSubReg1,
// CHECK:   PMI_TestSubReg8,
// CHECK:   PMI_TestSubReg8_8,
// CHECK:   PMI_TestSubReg16,
// CHECK: const RegisterBankInfo::PartialMapping PartialMappings
// CHECK: {0, 32, GPRRegBank},
// CHECK: {0, 1, TestSubRegRegBank},
// CHECK: {4, 4, TestSubRegRegBank},
// CHECK: {0, 8, TestSubRegRegBank},
// CHECK: {8, 8, TestSubRegRegBank},
// CHECK: {0, 16, TestSubRegRegBank},
def r_hi : SubRegIndex<8, 8>;
def r_lo : SubRegIndex<8, 0>;
def R1_hi : Register<"r1.hi">;
def R1_lo : Register<"r1.lo">;
def R1 : Register<"r1"> {
  let SubRegs = [R1_hi, R1_lo];
  let SubRegIndices = [r_hi, r_lo];
}
def ClassSubRegs : RegisterClass<"MyTarget", [i16], 16, (add R1)>;
let ExtraPartMappings = [[1], [4, 4]] in
def TestSubRegRegBank : RegisterBank<"TestSubReg", [ClassSubRegs]>;

// CHECK: GPRRegBankCoverageData
// CHECK: MyTarget::ClassARegClassID
// CHECK: MyTarget::ClassBRegClassID
def GPRRegBank : RegisterBank<"GPR", [ClassA]>;
