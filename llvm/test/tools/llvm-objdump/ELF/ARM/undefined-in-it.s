@RUN: llvm-mc -triple arm-none-eabi -mcpu=cortex-m33 -filetype=obj %s | llvm-objdump -d --mcpu=cortex-m3 - | FileCheck %s

@ Check that instructions that are disassembled as <undefined> within an IT
@ block advance the IT state. This prevents the IT state spilling over into
@ the next instruction.

@ The vldmiaeq instruction is disassembled as <undefined> with
@ -mcpu=cortex-m3 as this does not have a fpu.
.text
.fpu fp-armv8
.thumb
 ite eq
 vldmiaeq r0!, {s16-s31}
 addne    r0, r0, r0
 add      r1, r1, r1

 itet eq
 vldmiaeq r0!, {s16-s31}
 vldmiane r0!, {s16-s31}
 vldmiaeq r0!, {s16-s31}
 add      r0, r0, r0
 add      r1, r1, r1
 add      r2, r2, r2

 it eq
 vldmiaeq r0!, {s16-s31}

 it ne
 addne      r0, r0, r0

@ CHECK:             0: bf0c          ite     eq
@ CHECK-NEXT:        2: ecb0 8a10     <unknown>
@ CHECK-NEXT:        6: 1800          addne   r0, r0, r0
@ CHECK-NEXT:        8: 4409          add     r1, r1
@ CHECK-NEXT:        a: bf0a          itet    eq
@ CHECK-NEXT:        c: ecb0 8a10     <unknown>
@ CHECK-NEXT:       10: ecb0 8a10     <unknown>
@ CHECK-NEXT:       14: ecb0 8a10     <unknown>
@ CHECK-NEXT:       18: 4400          add     r0, r0
@ CHECK-NEXT:       1a: 4409          add     r1, r1
@ CHECK-NEXT:       1c: 4412          add     r2, r2
@ CHECK-NEXT:       1e: bf08          it      eq
@ CHECK-NEXT:       20: ecb0 8a10     <unknown>
@ CHECK-NEXT:       24: bf18          it      ne
@ CHECK-NEXT:       26: 1800          addne   r0, r0, r0
