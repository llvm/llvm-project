; RUN: opt -thinlto-bc -o %t0.bc %s
; RUN: llvm-lto2 run -r %t0.bc,__imp_f,l \
; RUN:               -r %t0.bc,g,p \
; RUN:               -r %t0.bc,g,l \
; RUN:               -r %t0.bc,e,l \
; RUN:               -r %t0.bc,main,x \
; RUN:               -save-temps -o %t1 %t0.bc
; RUN: llvm-dis %t1.1.3.import.bc -o - | FileCheck %s
source_filename = "test.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$g = comdat any
@g = global i8 42, comdat, !type !0

; CHECK: define
; CHECK-NOT: dllimport
; CHECK-SAME: @f
define available_externally dllimport i8* @f() {
  ret i8* @g
}

define i8* @e() {
  ret i8* @g
}

define i32 @main() {
  %1 = call i8* @f()
  %2 = ptrtoint i8* %1 to i32
  ret i32 %2
}
!0 = !{i32 0, !"typeid"}
