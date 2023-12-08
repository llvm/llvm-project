// REQUIRES: aarch64-registered-target
// RUN: llvm-mc -filetype=obj -triple aarch64-windows %s -o - \
// RUN:   | not --crash llvm-readobj --unwind - 2>&1 | FileCheck %s

// Older versions of LLVM had a bug where we would accidentally
// truncate the number of epilogue scopes to a uint8_t; make
// sure this doesn't happen.
//
// We expect the llvm-readobj invocation to fail because the
// xdata section is truncated (to reduce the size of the testcase).

// CHECK: EpilogueScopes: 256

.section .pdata,"dr"
        .long "?func@@YAHXZ"@IMGREL
        .long "$unwind$func@@YAHXZ"@IMGREL

        .text
        .globl  "?func@@YAHXZ"
        .p2align        3
"?func@@YAHXZ":
        ret

.section        .xdata,"dr"
"$unwind$func@@YAHXZ":
.long 0x00000000, 0x02010100, 0x09000000, 0x0A000000
