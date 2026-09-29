; RUN: llc -o - %s -mtriple=aarch64-linux-gnu | FileCheck %s

; ModuleID = 'compiler-ident.c'
target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; CHECK: .ident  "some LLVM version"

!llvm.ident = !{!0}

!0 = !{!"some LLVM version"}

