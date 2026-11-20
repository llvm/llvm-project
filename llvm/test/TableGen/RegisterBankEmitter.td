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

// CHECK: {0, 8, BreakDownBank},
// CHECK: {8, 8, BreakDownBank},
// CHECK: {16, 8, BreakDownBank},
// CHECK: {24, 8, BreakDownBank},
// CHECK: {0, 16, BreakDownBank},
// CHECK: {16, 16, BreakDownBank},
// CHECK: {0, 32, BreakDownBank},
def BreakDownBank : RegisterBank<"GPR", [ClassA]> {
  let ExtraPartialMappings=[
    PartialMapping<16>,
    PartialMapping<8>,PartialMapping<8,8>,
    PartialMapping<16, 16>,
    PartialMapping<8,16>,PartialMapping<8,24>
  ];
}
