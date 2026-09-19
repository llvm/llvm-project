# RUN: llvm-mc -triple=mipsel-unknown-linux-gnu -mcpu=mips32r2 -filetype=obj %s -o %t.o
# RUN: not llvm-jitlink -noexec %t.o 2>&1 | FileCheck %s

        .text
        .globl main
main:
        lw $2, %gottprel(tls_var)($gp)
        jr $ra
        nop

        .section .tbss,"awT",@nobits
tls_var:
        .space 4

# CHECK: initial/local-exec MIPS TLS relocations are unsupported
