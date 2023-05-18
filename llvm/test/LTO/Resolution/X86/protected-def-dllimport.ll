; RUN: llvm-as --disable-verify -o %t0.bc %s

; RUN: llvm-lto2 run -r %t0.bc,__imp_f,l \
; RUN:               -r %t0.bc,g,p \
; RUN:               -r %t0.bc,main,x \
; RUN:               -save-temps -o %t1 %t0.bc
; RUN: llvm-dis %t1.0.0.preopt.bc -o - | FileCheck %s
source_filename = "test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@llvm.used = appending global [2 x ptr] [ ptr @main, ptr @f ], section "llvm.metadata"

$g = comdat any
@g = global i8 42, comdat, !type !0

; CHECK: declare protected ptr @f()
define protected dllimport i8* @f() {
  ret i8* @g
}

define available_externally i32 @main() {
  %1 = call i8* @f()
  %2 = ptrtoint i8* %1 to i32
  ret i32 %2
}

!llvm.module.flags = !{!1}

!0 = !{i32 0, !"typeid"}
!1 = !{i32 1, !"ThinLTO", i32 0}
