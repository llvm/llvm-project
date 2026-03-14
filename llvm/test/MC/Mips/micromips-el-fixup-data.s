# RUN: llvm-mc %s -triple=mipsel-unknown-linux -mcpu=mips32r2 \
# RUN:   -mattr=+micromips 2>&1 -filetype=obj > %t.o
# RUN: llvm-objdump %t.o -d | FileCheck %s

# Check that fixup data is written in the microMIPS specific little endian
# byte order.

    .text
    .globl  main
    .align  2
    .type   main,@function
    .set    micromips
    .set    nomips16
    .ent    main
main:
    addiu   $sp, $sp, -16
    bnez    $9, lab1

# CHECK:    09 b4 03 00    bnez $9, 0xe <lab1>

    addu    $zero, $zero, $zero
lab1:
    jr  $ra
    addiu   $sp, $sp, 16
    .end    main
