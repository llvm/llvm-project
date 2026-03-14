// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid addressing modes.

adr z0.s, [z0.s, z0.d]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: adr z0.s, [z0.s, z0.d]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

adr z0.s, [z0.s, z0.s, lsl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected #imm after shift specifier
// CHECK-NEXT: adr z0.s, [z0.s, z0.s, lsl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

adr z0.s, [z0.s, z0.s, lsl #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].s'
// CHECK-NEXT: adr z0.s, [z0.s, z0.s, lsl #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

adr z0.s, [z0.s, z0.s, uxtw]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].s'
// CHECK-NEXT: adr z0.s, [z0.s, z0.s, uxtw]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

adr z0.s, [z0.s, z0.s, sxtw]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].s'
// CHECK-NEXT: adr z0.s, [z0.s, z0.s, sxtw]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

adr z0.d, [z0.d, z0.s]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: adr z0.d, [z0.d, z0.s]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

adr z0.d, [z0.d, z0.s, uxtw]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: adr z0.d, [z0.d, z0.s, uxtw]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

adr z0.d, [z0.d, z0.s, sxtw]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: adr z0.d, [z0.d, z0.s, sxtw]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

adr z0.d, [z0.d, z0.d, lsl #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].d, (lsl|uxtw|sxtw) #3'
// CHECK-NEXT: adr z0.d, [z0.d, z0.d, lsl #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

adr z0.d, [z0.d, z0.d, uxtw #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].d, (lsl|uxtw|sxtw) #3'
// CHECK-NEXT: adr z0.d, [z0.d, z0.d, uxtw #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

adr z0.d, [z0.d, z0.d, sxtw #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].d, (lsl|uxtw|sxtw) #3'
// CHECK-NEXT: adr z0.d, [z0.d, z0.d, sxtw #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
adr     z0.d, [z0.d, z0.d, sxtw #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: adr     z0.d, [z0.d, z0.d, sxtw #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
adr     z0.d, [z0.d, z0.d, sxtw #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: adr     z0.d, [z0.d, z0.d, sxtw #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
