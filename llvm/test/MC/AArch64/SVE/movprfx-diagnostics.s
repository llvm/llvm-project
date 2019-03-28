// RUN: not llvm-mc -triple=aarch64-none-linux-gnu -show-encoding -mattr=+sve  2>&1 < %s | FileCheck %s

// ------------------------------------------------------------------------- //
// Type suffix on unpredicated movprfx

movprfx z0.b, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: movprfx z0.b, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.b, z1.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: movprfx z0.b, z1.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z1.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected register without element width suffix
// CHECK-NEXT: movprfx z0, z1.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Different destination register (unary)

movprfx z0, z1
abs z2.d, p0/m, z2.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx writing to a different destination
// CHECK-NEXT: abs z2.d, p0/m, z2.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Different destination register (binary)

movprfx z0, z1
add z2.d, p0/m, z2.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx writing to a different destination
// CHECK-NEXT: add z2.d, p0/m, z2.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Different destination register (wide element)

movprfx z0, z1
asr z2.s, p0/m, z2.s, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx writing to a different destination
// CHECK-NEXT: asr z2.s, p0/m, z2.s, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Different destination register (ternary)

movprfx z0, z1
mla z3.d, p0/m, z1.d, z2.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx writing to a different destination
// CHECK-NEXT: mla z3.d, p0/m, z1.d, z2.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Destination used in other operand (unary)

movprfx z0, z1
abs z0.d, p0/m, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx and destination also used as non-destructive source
// CHECK-NEXT: abs z0.d, p0/m, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.d, p0/z, z1.d
cpy z0.d, p0/m, d0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx and destination also used as non-destructive source
// CHECK-NEXT: cpy z0.d, p0/m, d0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.d, p0/z, z1.d
mov z0.d, p0/m, d0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx and destination also used as non-destructive source
// CHECK-NEXT: mov z0.d, p0/m, d0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Destination used in other operand (binary)

movprfx z0, z1
add z0.d, p0/m, z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx and destination also used as non-destructive source
// CHECK-NEXT: add z0.d, p0/m, z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Destination used in other operand (wide element)

movprfx z0, z1
asr z0.s, p0/m, z0.s, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx and destination also used as non-destructive source
// CHECK-NEXT: asr z0.s, p0/m, z0.s, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Destination used in other operand (ternary)

movprfx z0, z1
mla z0.d, p0/m, z0.d, z2.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx and destination also used as non-destructive source
// CHECK-NEXT: mla z0.d, p0/m, z0.d, z2.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Different general predicate (unary)

movprfx z0.d, p0/m, z1.d
abs z0.d, p1/m, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx using a different general predicate
// CHECK-NEXT: abs z0.d, p1/m, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Different general predicate (binary)

movprfx z0.d, p0/m, z1.d
add z0.d, p1/m, z0.d, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx using a different general predicate
// CHECK-NEXT: add z0.d, p1/m, z0.d, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Different general predicate (wide element)

movprfx z0.d, p0/m, z1.d
asr z0.s, p1/m, z0.s, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx using a different general predicate
// CHECK-NEXT: asr z0.s, p1/m, z0.s, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Different general predicate (ternary)

movprfx z0.d, p0/m, z1.d
mla z0.d, p1/m, z1.d, z2.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx using a different general predicate
// CHECK-NEXT: mla z0.d, p1/m, z1.d, z2.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Different element size (unary)

movprfx z0.s, p0/m, z1.s
abs z0.d, p0/m, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx with a different element size
// CHECK-NEXT: abs z0.d, p0/m, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Different element size (binary)

movprfx z0.s, p0/m, z1.s
add z0.d, p0/m, z0.d, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx with a different element size
// CHECK-NEXT: add z0.d, p0/m, z0.d, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Different element size (wide element)

movprfx z0.d, p0/m, z1.d
asr z0.s, p0/m, z0.s, z1.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx with a different element size
// CHECK-NEXT: asr z0.s, p0/m, z0.s, z1.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Different element size (ternary)

movprfx z0.s, p0/m, z1.s
mla z0.d, p0/m, z1.d, z2.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx with a different element size
// CHECK-NEXT: mla z0.d, p0/m, z1.d, z2.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Predicated movprfx with non-predicated instruction.

movprfx z0.d, p0/m, z1.d
add z0.d, z0.d, #1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: add z0.d, z0.d, #1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Ensure we don't try to apply a prefix to subsequent instructions (upon failure)

movprfx z0, z1
add z0.d, z1.d, z2.d
add z0.d, z1.d, z2.d
// CHECK: [[@LINE-2]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: add z0.d, z1.d, z2.d
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:
// CHECK: add z0.d, z1.d, z2.d
