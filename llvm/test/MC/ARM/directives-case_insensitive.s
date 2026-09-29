// RUN: llvm-mc -triple armv7-eabi -filetype asm -o - %s 2>&1 | FileCheck %s

.WORD 0x12345678
# CHECK: .long 305419896

.SHORT 0x1234
# CHECK: .short 4660

.HWORD 0x3456
# CHECK: .short 13398

.ARM
# CHECK: .code 32

.THUMB_FUNC

.CODE 32
# CHECK: .code 32

.SYNTAX unified

foo .REQ r5
.UNREQ foo

.FNSTART
# CHECK: .fnstart
.CANTUNWIND
# CHECK: .cantunwind
.FNEND
# CHECK: .fnend

.FNSTART
# CHECK: .fnstart
.UNWIND_RAW 4, 0xb1, 0x01
# CHECK: .unwind_raw 4, 0xb1, 0x1
.PERSONALITY  __gxx_personality_v0
# CHECK: .personality __gxx_personality_v0
.HANDLERDATA
# CHECK: .handlerdata
.FNEND
# CHECK: .fnend

.FNSTART
# CHECK: .fnstart
.MOVSP r7
# CHECK: .movsp r7
.PERSONALITYINDEX 0
# CHECK: .personalityindex 0
.PAD #16
# CHECK: .pad #16
.SETFP r11, sp, #8
# CHECK: .setfp r11, sp, #8
.SAVE   {r4, r5, r11, lr}
# CHECK: .save  {r4, r5, r11, lr}
.VSAVE  {d0}
# CHECK: .vsave {d0}
.FNEND
# CHECK: .fnend

.LTORG

.POOL

.EVEN
# CHECK: .p2align 1

.ALIGN 2
# CHECK: .p2align 2

.ARCH armv8-a
# CHECK: .arch  armv8-a
.ARCH_EXTENSION crc

.CPU cortex-a8
# CHECK: .cpu cortex-a8
.EABI_ATTRIBUTE Tag_CPU_name, "cortex-a9"
# CHECK: .cpu cortex-a9

.THUMB_SET bar, 1
# CHECK: .thumb_set  bar, 1

.INST 0x87654321
# CHECK: .inst 0x87654321
.THUMB
# CHECK: .code 16
.INST.N 0xCAFE
# CHECK: .inst.n 0xcafe
.INST.W 0x44445555
# CHECK: .inst.w 0x44445555

.FPU neon
# CHECK: .fpu neon

.TLSDESCSEQ variable
# CHECK: .tlsdescseq  variable

.OBJECT_ARCH armv8
# CHECK: .object_arch armv8-a

