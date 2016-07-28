@ RUN: llvm-mc -triple thumbv7-windows-itanium -filetype obj -o - %s \
@ RUN:   | llvm-readobj -s - | FileCheck %s

    .syntax unified
    .thumb

    .text

    .global function
    .thumb_func
function:
    bx lr

@ CHECK: Section
@ CHECK-DAG: IMAGE_SCN_CNT_CODE
@ CHECK-DAG: IMAGE_SCN_MEM_16BIT
