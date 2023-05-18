; RUN: not llvm-mc -triple arm64-apple-ios8.0 %s -o /dev/null 2>&1 | FileCheck %s

        fmov s0, #-0.0
; CHECK: error: expected compatible register or floating-point constant

        fmov d0, #-0.0
; CHECK: error: expected compatible register or floating-point constant

