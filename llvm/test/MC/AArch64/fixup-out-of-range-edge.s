// RUN: not llvm-mc -triple aarch64 -filetype obj < %s -o /dev/null 2>&1 | FileCheck %s

// COM: Edge case testing for branches and ADR[P]
// CHECK-LABEL: :{{[0-9]+}}:{{[0-9]+}}: error: fixup value out of range
// CHECK-NEXT:  adr x0, adr_lower
adr_lower:
  adr x0, adr_lower-(1<<20)-1

// CHECK-LABEL: :{{[0-9]+}}:{{[0-9]+}}: error: fixup value out of range
// CHECK-NEXT:  adr x0, adr_upper
adr_upper:
  adr x0, adr_upper+(1<<20)

// CHECK-LABEL: :{{[0-9]+}}:{{[0-9]+}}: error: fixup value out of range
// CHECK-NEXT: b b_lower
b_lower:
  b b_lower-(1<<27)-4

// CHECK-LABEL: :{{[0-9]+}}:{{[0-9]+}}: error: fixup value out of range
// CHECK-NEXT: b b_upper
b_upper:
  b b_upper+(1<<27)

// CHECK-LABEL: :{{[0-9]+}}:{{[0-9]+}}: error: fixup value out of range
// CHECK-NEXT: beq beq_lower
beq_lower:
  beq beq_lower-(1<<20)-4

// CHECK-LABEL: :{{[0-9]+}}:{{[0-9]+}}: error: fixup value out of range
// CHECK-NEXT: beq beq_upper
beq_upper:
  beq beq_upper+(1<<20)

// CHECK-LABEL: :{{[0-9]+}}:{{[0-9]+}}: error: fixup value out of range
// CHECK-NEXT: ldr x0, ldr_lower
ldr_lower:
  ldr x0, ldr_lower-(1<<20)-4

// CHECK-LABEL: :{{[0-9]+}}:{{[0-9]+}}: error: fixup value out of range
// CHECK-NEXT: ldr x0, ldr_upper
ldr_upper:
  ldr x0, ldr_upper+(1<<20)

// CHECK-LABEL: :{{[0-9]+}}:{{[0-9]+}}: error: fixup value out of range
// CHECK-NEXT: tbz x0, #1, tbz_lower
tbz_lower:
  tbz x0, #1, tbz_lower-(1<<15)-4

// CHECK-LABEL: :{{[0-9]+}}:{{[0-9]+}}: error: fixup value out of range
// CHECK-NEXT: tbz x0, #1, tbz_upper
tbz_upper:
  tbz x0, #1, tbz_upper+(1<<15)

