// RUN: llvm-tblgen --gen-register-info -I %p/../../include %s 2>&1 | FileCheck %s
// RUN: not llvm-tblgen --gen-register-info -I %p/../../include -DERROR %s 2>&1 | FileCheck -check-prefix=ERROR %s

// Check that there is no assertion when specifying unsupported
// CopyCost values on register classes. Check that negative CopyCost
// values are saturated to 255.

include "llvm/Target/Target.td"

// CHECK: extern const MCRegisterClassStorage<{{[0-9, ]+}}> MyTargetMCRegisterClassStorage = {
// CHECK-NEXT: {
// CHECK-NEXT:   { {{[^,]+}}, {{[^,]+}}, 0, 32, 2, {{[0-9]+}}, MyTarget::GPR32RegClassID, 1, true, false },
// CHECK-NEXT:   { {{[^,]+}}, {{[^,]+}}, 6, 32, 1, {{[0-9]+}}, MyTarget::SPECIAL_CLASSRegClassID, 255, true, false },
// CHECK-NEXT: },

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def R0 : Register<"r0"> { let Namespace = "MyTarget"; }
def R1 : Register<"r1"> { let Namespace = "MyTarget"; }
def SPECIAL : Register<"special"> { let Namespace = "MyTarget"; }

// ERROR: :[[@LINE+1]]:5: error: 'CopyCost' must be an 8-bit value
def GPR32 : RegisterClass<"MyTarget", [i32], 32, (add R0, R1)> {
#ifdef ERROR
  let CopyCost = 256;
#endif
}

def SPECIAL_CLASS : RegisterClass<"MyTarget", [i32], 32, (add SPECIAL)> {
  let CopyCost = -1;
}
