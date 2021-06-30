// RUN: not llvm-mc -triple aarch64-arm-none-eabi -mattr +rme -show-encoding %s 2> %t | FileCheck %s
// RUN: FileCheck --check-prefix=CHECK-ERROR %s < %t
// RUN: not llvm-mc -triple aarch64-arm-none-eabi -show-encoding %s 2> %t | FileCheck --check-prefix=CHECK-NO-RME %s
// RUN: FileCheck --check-prefix=CHECK-NO-RME-ERROR %s < %t

msr MFAR_EL3, x0
msr GPCCR_EL3, x0
msr GPTBR_EL3, x0
mrs x0, MFAR_EL3
mrs x0, GPCCR_EL3
mrs x0, GPTBR_EL3
// CHECK: msr MFAR_EL3,  x0   // encoding: [0xa0,0x60,0x1e,0xd5]
// CHECK: msr GPCCR_EL3, x0   // encoding: [0xc0,0x21,0x1e,0xd5]
// CHECK: msr GPTBR_EL3, x0   // encoding: [0x80,0x21,0x1e,0xd5]
// CHECK: mrs x0, MFAR_EL3    // encoding: [0xa0,0x60,0x3e,0xd5]
// CHECK: mrs x0, GPCCR_EL3   // encoding: [0xc0,0x21,0x3e,0xd5]
// CHECK: mrs x0, GPTBR_EL3   // encoding: [0x80,0x21,0x3e,0xd5]
// CHECK-NO-RME-ERROR: [[@LINE-12]]:5: error: expected writable system register
// CHECK-NO-RME-ERROR: [[@LINE-12]]:5: error: expected writable system register
// CHECK-NO-RME-ERROR: [[@LINE-12]]:5: error: expected writable system register
// CHECK-NO-RME-ERROR: [[@LINE-12]]:9: error: expected readable system register
// CHECK-NO-RME-ERROR: [[@LINE-12]]:9: error: expected readable system register
// CHECK-NO-RME-ERROR: [[@LINE-12]]:9: error: expected readable system register

tlbi rpaos, x0
tlbi rpalos, x0
tlbi paallos
tlbi paall
// CHECK: tlbi rpaos, x0      // encoding: [0x60,0x84,0x0e,0xd5]
// CHECK: tlbi rpalos, x0     // encoding: [0xe0,0x84,0x0e,0xd5]
// CHECK: tlbi paallos        // encoding: [0x9f,0x81,0x0e,0xd5]
// CHECK: tlbi paall          // encoding: [0x9f,0x87,0x0e,0xd5]
// CHECK-NO-RME-ERROR: [[@LINE-8]]:6: error: TLBI RPAOS requires: rme
// CHECK-NO-RME-ERROR: [[@LINE-8]]:6: error: TLBI RPALOS requires: rme
// CHECK-NO-RME-ERROR: [[@LINE-8]]:6: error: TLBI PAALLOS requires: rme
// CHECK-NO-RME-ERROR: [[@LINE-8]]:6: error: TLBI PAALL requires: rme

tlbi RPAOS
tlbi RPALOS
tlbi PAALLOS, x25
tlbi PAALL, x25
// CHECK-ERROR: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi RPAOS
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op requires a register
// CHECK-ERROR-NEXT:         tlbi RPALOS
// CHECK-ERROR-NEXT:              ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op does not use a register
// CHECK-ERROR-NEXT:         tlbi PAALLOS, x25
// CHECK-ERROR-NEXT:                       ^
// CHECK-ERROR-NEXT: error: specified {{TLBI|tlbi}} op does not use a register
// CHECK-ERROR-NEXT:         tlbi PAALL, x25
// CHECK-ERROR-NEXT:                    ^
// CHECK-NO-RME-ERROR: [[@LINE-16]]:6: error: TLBI RPAOS requires: rme
// CHECK-NO-RME-ERROR: [[@LINE-16]]:6: error: TLBI RPALOS requires: rme
// CHECK-NO-RME-ERROR: [[@LINE-16]]:6: error: TLBI PAALLOS requires: rme
// CHECK-NO-RME-ERROR: [[@LINE-16]]:6: error: TLBI PAALL requires: rme

sys #6, c8, c4, #3
sys #6, c8, c4, #7
sys #6, c8, c1, #4
sys #6, c8, c7, #4
// CHECK: tlbi rpaos
// CHECK: tlbi rpalos
// CHECK: tlbi paallos
// CHECK: tlbi paall
// CHECK-NO-RME: sys #6, c8, c4, #3
// CHECK-NO-RME: sys #6, c8, c4, #7
// CHECK-NO-RME: sys #6, c8, c1, #4
// CHECK-NO-RME: sys #6, c8, c7, #4
