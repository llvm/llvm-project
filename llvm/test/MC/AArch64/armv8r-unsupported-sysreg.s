// RUN: llvm-mc -triple aarch64 -mattr=+v8a -show-encoding < %s | FileCheck %s
// RUN: not llvm-mc -triple aarch64 -mattr=+v8r -show-encoding < %s 2>&1 |\
// RUN:   FileCheck --check-prefix=CHECK-ERROR %s

// CHECK:       msr	TTBR0_EL2, x3          // encoding: [0x03,0x20,0x1c,0xd5]
// CHECK-NEXT: 	mrs	x3, TTBR0_EL2          // encoding: [0x03,0x20,0x3c,0xd5]
// CHECK-NEXT: 	msr	VTTBR_EL2, x3          // encoding: [0x03,0x21,0x1c,0xd5]
// CHECK-NEXT: 	mrs	x3, VTTBR_EL2          // encoding: [0x03,0x21,0x3c,0xd5]
// CHECK-NEXT: 	msr	VSTTBR_EL2, x3         // encoding: [0x03,0x26,0x1c,0xd5]
// CHECK-NEXT: 	mrs	x3, VSTTBR_EL2         // encoding: [0x03,0x26,0x3c,0xd5]

msr TTBR0_EL2, x3
mrs x3, TTBR0_EL2
msr VTTBR_EL2, x3
mrs x3, VTTBR_EL2
msr VSTTBR_EL2, x3
mrs x3, VSTTBR_EL2

// CHECK-ERROR:      {{.*}}: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr TTBR0_EL2, x3
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: {{.*}}: error: expected readable system register
// CHECK-ERROR-NEXT:         mrs x3, TTBR0_EL2
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: {{.*}}: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr VTTBR_EL2, x3
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: {{.*}}: error: expected readable system register
// CHECK-ERROR-NEXT:         mrs x3, VTTBR_EL2
// CHECK-ERROR-NEXT:                 ^
// CHECK-ERROR-NEXT: {{.*}}: error: expected writable system register or pstate
// CHECK-ERROR-NEXT:         msr VSTTBR_EL2, x3
// CHECK-ERROR-NEXT:             ^
// CHECK-ERROR-NEXT: {{.*}}: error: expected readable system register
// CHECK-ERROR-NEXT:         mrs x3, VSTTBR_EL2
// CHECK-ERROR-NEXT:                 ^
