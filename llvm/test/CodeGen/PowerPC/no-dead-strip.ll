; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-linux-gnu < %s | FileCheck %s

; CHECK: .section  .bss,"aw",@nobits
; CHECK: .weak X
; CHECK-LABEL: X:
; CHECK: .long 0
; CHECK: .size X, 4

@X = weak global i32 0          ; <i32*> [#uses=1]
@.str = internal constant [4 x i8] c"t.c\00", section "llvm.metadata"          ; <[4 x i8]*> [#uses=1]
@llvm.used = appending global [1 x i8*] [ i8* bitcast (i32* @X to i8*) ], section "llvm.metadata"       ; <[1 x i8*]*> [#uses=0]

