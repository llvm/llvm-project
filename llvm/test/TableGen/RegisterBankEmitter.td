// RUN: llvm-tblgen -gen-register-bank -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def MyTarget : Target;
def R0 : Register<"r0">;
let Size = 32 in {
  def ClassA : RegisterClass<"MyTarget", [i32], 32, (add R0)>;
  def ClassB : RegisterClass<"MyTarget", [i1], 32, (add ClassA)>;
}

// CHECK: GPRRegBankCoverageData
// CHECK: MyTarget::ClassARegClassID
// CHECK: MyTarget::ClassBRegClassID
def GPRRegBank : RegisterBank<"GPR", [ClassA]>;
