@ RUN: not llvm-mc -triple=armv7-apple-darwin < %s 2>&1 | FileCheck %s

ldrd r6, r7 [r2, #15]

@ CHECK: error: immediate value expected for vector index
