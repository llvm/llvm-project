// RUN: not llvm-tblgen -gen-register-bank -I %p/../../include %s 2>&1 | FileCheck %s

include "llvm/Target/Target.td"

def MyTarget : Target;
def R0 : Register<"r0">;
let Size = 32 in {
  def ClassA : RegisterClass<"MyTarget", [i32], 32, (add R0)>;
  def ClassB : RegisterClass<"MyTarget", [i1], 32, (add ClassA)>;
}

def GPRRegBank : RegisterBank<"GPR", [ClassA]>;

// CHECK: type1 is not mapped in rules
def : RegBankInstrMapping<[G_SHL], [
  RegBankMappingRule<[RegBankTypeMapping<GPRRegBank, type0>]>
]>;
