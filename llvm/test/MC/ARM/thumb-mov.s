// RUN: not llvm-mc -triple=thumbv7 -show-encoding < %s 2>&1 | FileCheck --check-prefix=CHECK --check-prefix=CHECK-V7 %s
// RUN: not llvm-mc -triple=thumbv8 -show-encoding < %s 2>&1 | FileCheck --check-prefix=CHECK --check-prefix=CHECK-V8 %s

        // Tests to check handling of sp and pc in thumb mov instructions. We
        // have to be careful about the order of things, as stdout/stderr
        // buffering means the errors appear before the non-error output, so
        // we have to put all the error checks at the top.

        // First check instructions that are never valid. These are thumb2
        // instructions that uses pc

        // t2MOVr selected because no thumb1 movs that can access high regs
        movs pc, r0
        movs r0, pc
        movs pc, pc
// CHECK: error: invalid operand for instruction
// CHECK-NEXT: movs pc, r0
// CHECK: error: invalid operand for instruction
// CHECK-NEXT: movs r0, pc
// CHECK: error: invalid operand for instruction
// CHECK-NEXT: movs pc, pc

        // mov.w selects t2MOVr
        mov.w pc, r0
        mov.w r0, pc
        mov.w pc, pc
// CHECK: error: invalid operand for instruction
// CHECK-NEXT: mov.w pc, r0
// CHECK: error: invalid operand for instruction
// CHECK-NEXT: mov.w r0, pc
// CHECK: error: invalid operand for instruction
// CHECK-NEXT: mov.w pc, pc

        // movs.w selects t2MOVr
        movs.w pc, r0
        movs.w r0, pc
        movs.w pc, pc
// CHECK: error: invalid operand for instruction
// CHECK-NEXT: movs.w pc, r0
// CHECK: error: invalid operand for instruction
// CHECK-NEXT: movs.w r0, pc
// CHECK: error: invalid operand for instruction
// CHECK-NEXT: movs.w pc, pc


        // Now check instructions that are invalid before ARMv8 due to SP usage

        movs sp, r0
        movs r0, sp
        movs sp, sp
// CHECK-V7: error: instruction variant requires ARMv8 or later
// CHECK-V7-NEXT: movs sp, r0
// CHECK-V7: error: instruction variant requires ARMv8 or later
// CHECK-V7-NEXT: movs r0, sp
// CHECK-V7: error: instruction variant requires ARMv8 or later
// CHECK-V7-NEXT: movs sp, sp
// CHECK-V8: movs.w sp, r0            @ encoding: [0x5f,0xea,0x00,0x0d]
// CHECK-V8: movs.w r0, sp            @ encoding: [0x5f,0xea,0x0d,0x00]
// CHECK-V8: movs.w sp, sp            @ encoding: [0x5f,0xea,0x0d,0x0d]

        mov.w sp, sp
// CHECK-V7: error: instruction variant requires ARMv8 or later
// CHECK-V7-NEXT: mov.w sp, sp
// CHECK-V8: mov.w sp, sp             @ encoding: [0x4f,0xea,0x0d,0x0d]

        movs.w sp, r0
        movs.w r0, sp
        movs.w sp, sp
// CHECK-V7: error: instruction variant requires ARMv8 or later
// CHECK-V7-NEXT: movs.w sp, r0
// CHECK-V7: error: instruction variant requires ARMv8 or later
// CHECK-V7-NEXT: movs.w r0, sp
// CHECK-V7: error: instruction variant requires ARMv8 or later
// CHECK-V7-NEXT: movs.w sp, sp
// CHECK-V8: movs.w sp, r0            @ encoding: [0x5f,0xea,0x00,0x0d]
// CHECK-V8: movs.w r0, sp            @ encoding: [0x5f,0xea,0x0d,0x00]
// CHECK-V8: movs.w sp, sp            @ encoding: [0x5f,0xea,0x0d,0x0d]


        // Now instructions that are always valid

        // mov selects tMOVr, where sp and pc are allowed
        mov sp, r0
        mov r0, sp
        mov sp, sp
        mov pc, r0
        mov r0, pc
        mov pc, pc
// CHECK: mov sp, r0                  @ encoding: [0x85,0x46]
// CHECK: mov r0, sp                  @ encoding: [0x68,0x46]
// CHECK: mov sp, sp                  @ encoding: [0xed,0x46]
// CHECK: mov pc, r0                  @ encoding: [0x87,0x46]
// CHECK: mov r0, pc                  @ encoding: [0x78,0x46]
// CHECK: mov pc, pc                  @ encoding: [0xff,0x46]

        // sp allowed in non-flags-setting t2MOVr
        mov.w sp, r0
        mov.w r0, sp
// CHECK: mov.w sp, r0                @ encoding: [0x4f,0xea,0x00,0x0d]
// CHECK: mov.w r0, sp                @ encoding: [0x4f,0xea,0x0d,0x00]
