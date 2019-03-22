# RUN: llvm-mc -triple=powerpc64le-unknown-linux -filetype=obj %s -o %t.o
# RUN: llvm-objdump -d %t.o | FileCheck %s

# RUN: llvm-mc -triple=powerpc64-unknown-linux -filetype=obj %s -o %t.o
# RUN: llvm-objdump -d %t.o | FileCheck %s

# RUN: llvm-mc -triple=powerpc-unknown-linux -filetype=obj %s -o %t.o
# RUN: llvm-objdump -d %t.o | FileCheck %s

# CHECK: 0000000000000000 callee_back:
# CHECK: 18: {{.*}} bl .-24
# CHECK: 20: {{.*}} bl .+16
# CHECK: 0000000000000030 callee_forward:

        .text
        .global caller
        .type caller,@function
        .type callee_forward,@function
        .type callee_back,@function

        .p2align 4
callee_back:
        li 3, 55
        blr

        .p2align 4
caller:
.Lgep:
        addis 2, 12, .TOC.-.Lgep@ha
        addi 2, 2, .TOC.-.Lgep@l
.Llep:
        .localentry caller, .Llep-.Lgep
        bl callee_back
        mr 31, 3
        bl callee_forward
        add 3, 3, 31
        blr

        .p2align 4
callee_forward:
        li 3, 66
        blr

