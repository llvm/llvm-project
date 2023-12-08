; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-linux-gnu < %s | FileCheck %s

; CHECK: .section  .bss.X,"awR",@nobits
; CHECK: .weak X
; CHECK-LABEL: X:
; CHECK: .long 0
; CHECK: .size X, 4

@X = weak global i32 0          ; <ptr> [#uses=1]
@.str = internal constant [4 x i8] c"t.c\00", section "llvm.metadata"          ; <ptr> [#uses=1]
@llvm.used = appending global [1 x ptr] [ ptr @X ], section "llvm.metadata"       ; <ptr> [#uses=0]

