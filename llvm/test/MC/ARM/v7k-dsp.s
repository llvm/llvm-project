@ RUN: llvm-mc -triple thumbv7k-apple-watchos2.0 %s | FileCheck %s

@ CHECK: usad8 r2, r1, r4
    usad8 r2, r1, r4
