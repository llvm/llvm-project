# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax,+c,+m %s -o - | \
# RUN: llvm-objdump -dr --mattr=+c,+m - | FileCheck %s

        .text
        .option relax
        .balign 4
        .global _start
        .type _start, @function
# This R_RISCV_ALIGN for .balign 4
# CHECK: R_RISCV_ALIGN *ABS*+0x2
_start:
        lui a0, %hi(foo)
        addi a0, a0, %lo(foo)
	mul a0, a1, a4
        .option push

        .option norelax
        .option norvc
# This R_RISCV_ALIGN for .balign 8, we should emit that even
# norelax is set, because the code before this point might relax,
# and size may changed, so that we need to align this again at linker
# time.
# Also padding pad should be +6 rather than +4 here, because we have enabled
# RVC before, and linker may relax instructions to RVC instructions,
# That will cause 4 byte padding might not be enough to fix the alignment.
# CHECK: R_RISCV_ALIGN *ABS*+0x6
        .balign 8
SHOULD_ALIGN_8_HERE:
        .word 0x12345678

        .option pop

foo:
        ret
