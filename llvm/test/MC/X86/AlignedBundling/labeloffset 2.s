# RUN: llvm-mc -triple=i686-linux -filetype=obj %s -o - | \
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn -r - | FileCheck %s
# RUN: llvm-mc -triple=i686-nacl -filetype=obj %s -o - | \
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn -r - | FileCheck %s
# RUN: llvm-mc -triple=i686-nacl -filetype=obj -mc-relax-all %s -o - | \
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn -r - | FileCheck %s

        .bundle_align_mode 5
        .text
        .globl  main
        .align  32, 0x90
        .type   main,@function
main:                                   # @main
# CHECK-LABEL: <main>:
# Call + pop sequence for determining the PIC base.
        .bundle_lock align_to_end
        calll   .L0$pb
        .bundle_unlock
.L0$pb:
        popl    %eax
# CHECK: 20: popl
# 26 bytes of instructions between the pop and the use of the pic base symbol.
        movl    $3, 2(%ebx, %ebx)
        movl    $3, 2(%ebx, %ebx)
        movl    $3, 2(%ebx, %ebx)
        hlt
        hlt
# CHECK: nop
.Ltmp0:
        addl    (.Ltmp0-.L0$pb), %eax
# The addl has bundle padding to push it from 0x3b to 0x40.
# The difference between the labels should be 0x20 (0x40-0x20) not 0x1b
# (0x3b-0x20)
# CHECK: 40: addl 32, %eax
        popl    %ecx
        jmp     *%ecx


# Also make sure it works with a non-relaxable instruction (cmp vs add)
# and for 2 adjacent labels that both point to the correct instruction
        .section .text.bar, "ax"
        .globl  bar
        .align  32, 0x90
        .type   bar,@function
bar:
# CHECK-LABEL: bar:
        .bundle_lock align_to_end
        calll   .L1$pb
        .bundle_unlock
.L1$pb:
        popl %eax
# CHECK: 20: popl
# 26 bytes of instructions between the pop and the use of the pic base symbol.
        movl    $3, 2(%ebx, %ebx)
        movl    $3, 2(%ebx, %ebx)
        movl    $3, 2(%ebx, %ebx)
        hlt
        hlt
# CHECK: nop
.Ltmp1:
.Ltmp2:
        cmpl    %eax, .Ltmp1
# CHECK: 40: cmpl %eax, 64
        cmpl     %eax, (.Ltmp2-.L1$pb)
# CHECK: 46: cmpl %eax, 32
        popl    %ecx
        jmp *%ecx


# Switch sections in the middle of a function
        .section .text.foo, "ax"
        .globl  foo
        .align  32, 0x90
        .type   foo,@function
# CHECK-LABEL: foo:
foo:
        inc %eax
tmp3:
        .rodata
        .type   obj,@object
        .comm   obj,4,4
        .section .text.foo
        inc %eax
# CHECK: <tmp3>:
# CHECK-NEXT: 1: incl
