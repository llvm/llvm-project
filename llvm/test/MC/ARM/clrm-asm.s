// RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -show-encoding < %s 2>%t \
// RUN: | FileCheck --check-prefix=CHECK %s
// RUN:   FileCheck --check-prefix=ERROR < %t %s

// CHECK: clrm            {r0, r1, r2, r3} @ encoding: [0x9f,0xe8,0x0f,0x00]
clrm {r0, r1, r2, r3}

// CHECK: clrm            {r1, r2, r3, r4} @ encoding: [0x9f,0xe8,0x1e,0x00]
// ERROR-NOT: register list not in ascending order
clrm {r3, r4, r1, r2}

// CHECK: clrm            {r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, lr, apsr} @ encoding: [0x9f,0xe8,0xff,0xdf]
clrm {r0-r12, lr, apsr}

// CHECK: clrm            {lr, apsr} @ encoding: [0x9f,0xe8,0x00,0xc0]
clrm {apsr, lr}

// CHECK: clrm            {r0, r1, apsr}  @ encoding: [0x9f,0xe8,0x03,0x80]
clrm {apsr, r1, r0}

// CHECK: clrm            {r0, r1, r2, r3, r4, lr, apsr} @ encoding: [0x9f,0xe8,0x1f,0xc0]
clrm {r0-r4, apsr, lr}

// ERROR: invalid register in register list. Valid registers are r0-r12, lr/r14 and APSR.
clrm {sp}

// ERROR: invalid register in register list. Valid registers are r0-r12, lr/r14 and APSR.
clrm {r13}

// ERROR: invalid register in register list. Valid registers are r0-r12, lr/r14 and APSR.
clrm {r0-r12, sp}
