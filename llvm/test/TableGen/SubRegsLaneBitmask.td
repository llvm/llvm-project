// RUN: not llvm-tblgen -gen-register-info -register-info-debug -I %p/../../include %s -o - 2>&1  | FileCheck %s
include "llvm/Target/Target.td"
def TestTarget : Target;

foreach Index = 0...64 in {
  def sub#Index : SubRegIndex<32, !shl(Index, 5)>;
}

def A : Register<"">;
def TestRC : RegisterClass<"Test", [i32], 0, (add A)>;

// CHECK: error: Ran out of lanemask bits to represent subregister sub64
