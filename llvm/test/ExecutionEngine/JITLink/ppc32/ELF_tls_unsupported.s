# RUN: llvm-mc -triple=powerpc-unknown-linux-gnu -filetype=obj %s -o %t.o
# RUN: not llvm-jitlink -noexec %t.o 2>&1 | FileCheck %s

        .text
        .globl main
main:
        lwz 3, tls_var@got@tprel(30)
        blr

        .section .tbss,"awT",@nobits
tls_var:
        .space 4

# CHECK: Unsupported PPC32 relocation R_PPC_GOT_TPREL16
