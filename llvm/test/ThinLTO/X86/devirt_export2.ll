; REQUIRES: x86-registered-target

; Test that devirtualized local targets are not promoted to external when
; using Regular LTO.

; Regular LTO WPD.
; RUN: opt -module-summary -o %t9.o %s
; RUN: opt -module-summary -o %t10.o %p/Inputs/devirt_export2.ll
; RUN: llvm-lto2 run %t9.o %t10.o -devirtualize-speculatively -save-temps -pass-remarks=. \
; RUN:   -o %t11 \
; RUN:   -r=%t9.o,test,px \
; RUN:   -r=%t10.o,_ZTV1A,px \
; RUN:   2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t11.0.4.opt.bc -o - | FileCheck %s

; REMARK: devirtualized a call to _ZN1A1fEv

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @test(ptr %obj) {
entry:
  ; CHECK: define{{.*}} @test(
  ; CHECK: @_ZN1A1fEv
  %0 = load ptr, ptr %obj
  %1 = call i1 @llvm.public.type.test(ptr %0, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %1)
  %2 = load ptr, ptr %0
  %3 = call i32 %2(ptr %obj)
  ret i32 %3
}

; CHECK: define internal{{.*}} @_ZN1A1fEv(

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ThinLTO", i32 0}
