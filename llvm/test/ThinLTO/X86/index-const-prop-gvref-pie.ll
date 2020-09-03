;; The same as index-const-prop-gvref.ll, except for PIE.
; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/index-const-prop-gvref.ll -o %t2.bc
; RUN: llvm-lto2 run -save-temps %t2.bc -r=%t2.bc,b,pl -r=%t2.bc,a,pl \
; RUN:   %t1.bc -r=%t1.bc,main,plx -r=%t1.bc,a, -r=%t1.bc,b, -o %t3
; RUN: llvm-dis %t3.2.3.import.bc -o - | FileCheck %s --check-prefix=DEST

;; For PIE, keep dso_local for declarations to enable direct access.
; DEST:      @b = external dso_local global i32*
; DEST-NEXT: @a = available_externally dso_local global i32 42, align 4

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = external global i32
@b = external global i32*

define i32 @main() {
  %p = load i32*, i32** @b, align 8
  store i32 33, i32* %p, align 4
  %v = load i32, i32* @a, align 4
  ret i32 %v
}

!llvm.module.flags = !{!0}

!0 = !{i32 7, !"PIE Level", i32 2}
!1 = !{i32 7, !"PIC Level", i32 2}
