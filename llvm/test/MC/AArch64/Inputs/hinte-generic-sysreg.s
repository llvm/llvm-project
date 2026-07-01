msr S0_0_C2_C0_0, x0
// CHECK: msr S0_0_C2_C0_0, x0 // encoding: [0x00,0x20,0x00,0xd5]
// CHECK-ERROR: error: expected writable system register or pstate

mrs x0, S0_0_C2_C0_0
// CHECK: mrs x0, S0_0_C2_C0_0 // encoding: [0x00,0x20,0x20,0xd5]
// CHECK-ERROR: error: expected readable system register
