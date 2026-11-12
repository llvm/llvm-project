@ RUN: not llvm-mc -triple=armv7a-eabi < %s 2>&1 | FileCheck %s

ldr r4, [s1, #12]
@ CHECK: [[@LINE-1]]{{.*}}error: invalid operand for instruction

ldr r4, [d2, #12]
@ CHECK: [[@LINE-1]]{{.*}}error: invalid operand for instruction

ldr r4, [q3, #12]
@ CHECK: [[@LINE-1]]{{.*}} invalid operand for instruction
ldr r4, [cpsr, #12]
@ CHECK: [[@LINE-1]]{{.*}} invalid operand for instruction
ldr r4, [r1, s12]
@ CHECK: [[@LINE-1]]{{.*}} invalid operand for instruction
ldr r4, [r1, d12]
@ CHECK: [[@LINE-1]]{{.*}} invalid operand for instruction
ldr r4, [r1, q12]
@ CHECK: [[@LINE-1]]{{.*}} invalid operand for instruction
ldr r4, [r1, cpsr]
@ CHECK: [[@LINE-1]]{{.*}} invalid operand for instruction
ldr r4, [r3], s12
@ CHECK: [[@LINE-1]]{{.*}} invalid operand for instruction
ldr r4, [r3], d12
@ CHECK: [[@LINE-1]]{{.*}} invalid operand for instruction
ldr r4, [r3], q12
@ CHECK: [[@LINE-1]]{{.*}} invalid operand for instruction
ldr r4, [r3], cpsr
@ CHECK: [[@LINE-1]]{{.*}} invalid operand for instruction
add r3, r0, s1, lsl #2
@ CHECK: [[@LINE-1]]{{.*}} invalid operand for instruction
add r3, r0, d1, lsl #2
@ CHECK: [[@LINE-1]]{{.*}} invalid operand for instruction
add r3, r0, q1, lsl #2
@ CHECK: [[@LINE-1]]{{.*}} invalid operand for instruction
add r3, r0, cpsr, lsl #2
@ CHECK: [[@LINE-1]]{{.*}} invalid operand for instruction
add r3, r0, r1, lsl s6
@ CHECK: [[@LINE-1]]{{.*}} invalid operand for instruction
add r3, r0, r1, lsl d6
@ CHECK: [[@LINE-1]]{{.*}} invalid operand for instruction
add r3, r0, r1, lsl q6
@ CHECK: [[@LINE-1]]{{.*}} invalid operand for instruction
add r3, r0, r1, lsl cpsr
@ CHECK: [[@LINE-1]]{{.*}} invalid operand for instruction
ldrd r2, r3, [s4]
@ CHECK: [[@LINE-1]]{{.*}} invalid operand for instruction
ldrd r2, r3, [r4, s5]
@ CHECK: [[@LINE-1]]{{.*}} invalid operand for instruction
ldrd r2, r3, [r4], s5
@ CHECK: [[@LINE-1]]{{.*}} invalid operand for instruction
