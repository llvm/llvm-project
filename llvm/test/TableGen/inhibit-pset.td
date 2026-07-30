// RUN: llvm-tblgen -gen-register-info -I %p/../../include -I %p/Common %s | FileCheck %s

include "reg-with-subregs-common.td"

// CHECK-DAG: GPR32_AND_XR32RegClassID =
// CHECK-DAG: XR32RegClassID =

def X0 : Register <"x0">;

// CHECK-LABEL: getRegPressureSetName(unsigned Idx) const {
// CHECK-NEXT:    static const char *PressureNameTable[] = {
// CHECK-NEXT:      "GPR32",
// CHECK-NEXT:    };
// CHECK-NEXT:    return PressureNameTable[Idx];
// CHECK-NEXT:  }

// CHECK:      unsigned TestTargetGenRegisterInfo::
// CHECK-NEXT: getRegPressureSetLimit(const MachineFunction &MF, unsigned Idx) const {
// CHECK-NEXT:  static const uint16_t PressureLimitTable[] = {
// CHECK-NEXT:    {{[0-9]+}},        // 0: GPR32
// CHECK-NEXT:  };
// CHECK-NEXT:  return PressureLimitTable[Idx];
// CHECK-NEXT:}

// CHECK:      static const int RCSetsTable[] = {
// CHECK-NEXT:   /* 0 */ 0, -1,
// CHECK-NEXT: };

def XR32 : RegisterClass<"TestTarget", [i32], 32, (add X0)> {
  let GeneratePressureSet = 0;
}

def GPR32_AND_XR32 : RegisterClass<"TestTarget", [i32], 32, (add GPR32, X0)>;
