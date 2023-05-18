// RUN: llvm-tblgen -gen-register-info -I %p/../../include -I %p/Common %s | FileCheck %s

// Checks the CostPerUse value for the registers.
include "llvm/Target/Target.td"

let Namespace = "MyTarget" in {

  foreach Index = 0-3 in {
    // Adds register cost value 1.
    let CostPerUse = [1] in {
      def S#Index : Register <"s"#Index>;
    }
  }

  // CostPerUse by default to 0.
  def D0 : Register<"d0">;
  def D1 : Register<"d1">;

} // Namespace = "MyTarget"


def SRegs : RegisterClass<"MyTarget", [i32], 32, (sequence "S%u", 0, 3)>;
def DRegs : RegisterClass<"MyTarget", [i32], 32, (sequence "D%u", 0, 1)>;

def MyTarget : Target;

// CHECK:  static const uint8_t CostPerUseTable[] = {
// CHECK-NEXT:  0, 0, 0, 1, 1, 1, 1, };

// CHECK:  static const bool InAllocatableClassTable[] = {
// CHECK-NEXT:  false, true, true, true, true, true, true, };

// CHECK:  static const TargetRegisterInfoDesc MyTargetRegInfoDesc = { // Extra Descriptors
// CHECK-NEXT:  CostPerUseTable, 1, InAllocatableClassTable};

// CHECK:  TargetRegisterInfo(&MyTargetRegInfoDesc, RegisterClasses, RegisterClasses+2,
