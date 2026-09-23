; REQUIRES: x86-registered-target

;; Check that successful devirtualization to an alias does NOT allow constant propagation
;; when the alias or aliasee are interposable.

;; Generate split module with summary for hybrid Thin/Regular LTO WPD.
; RUN: opt -passes=assign-guid -thinlto-bc -thinlto-split-lto-unit -o %t.o %s

; RUN: llvm-lto2 run %t.o -save-temps \
; RUN:   -whole-program-visibility \
; RUN:   -o %t2 \
; RUN:   -r=%t.o,test_interposable_aliasee,px \
; RUN:   -r=%t.o,_ZTV2D, \
; RUN:   -r=%t.o,_ZTV2D,px \
; RUN:   -r=%t.o,_ZN2D1mEi,px \
; RUN:   -r=%t.o,_ZN2D1mEiAlias,px \
; RUN:   -r=%t.o,_ZN2D1mEiAlias, \
; RUN:   -r=%t.o,test_interposable_alias,px \
; RUN:   -r=%t.o,_ZTV3D, \
; RUN:   -r=%t.o,_ZTV3D,px \
; RUN:   -r=%t.o,_ZN3D1mEi,px \
; RUN:   -r=%t.o,_ZN3D1mEiAlias,px \
; RUN:   -r=%t.o,_ZN3D1mEiAlias,
; RUN: llvm-dis %t2.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.D = type { ptr }

@_ZTV2D = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr poison, ptr @_ZN2D1mEiAlias] }, !type !4

define weak i32 @_ZN2D1mEi(ptr %this, i32 %a) {
   ret i32 0
}

@_ZN2D1mEiAlias = hidden unnamed_addr alias i32 (ptr, i32), ptr @_ZN2D1mEi

; CHECK-IR-LABEL: define i32 @test_interposable_aliasee(
define i32 @test_interposable_aliasee(ptr %obj2, i32 %a) {
entry:
  %vtable2 = load ptr, ptr %obj2
  %p2 = call i1 @llvm.type.test(ptr %vtable2, metadata !"_ZTS2D")
  call void @llvm.assume(i1 %p2)
  %fptr33 = load ptr, ptr %vtable2, align 8

  ;; CHECK-IR: tail call i32 @_ZN2D1mEiAlias
  %call4 = tail call i32 %fptr33(ptr nonnull %obj2, i32 %a)
  ret i32 %call4
}


@_ZTV3D = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr poison, ptr @_ZN3D1mEiAlias] }, !type !5

define i32 @_ZN3D1mEi(ptr %this, i32 %a) {
   ret i32 0
}

@_ZN3D1mEiAlias = weak alias i32 (ptr, i32), ptr @_ZN3D1mEi

; CHECK-IR-LABEL: define i32 @test_interposable_alias(
define i32 @test_interposable_alias(ptr %obj2, i32 %a) {
entry:
  %vtable2 = load ptr, ptr %obj2
  %p2 = call i1 @llvm.type.test(ptr %vtable2, metadata !"_ZTS3D")
  call void @llvm.assume(i1 %p2)
  %fptr33 = load ptr, ptr %vtable2, align 8

  ;; CHECK-IR: tail call i32 @_ZN3D1mEiAlias
  %call4 = tail call i32 %fptr33(ptr nonnull %obj2, i32 %a)
  ret i32 %call4
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!4 = !{i64 16, !"_ZTS2D"}
!5 = !{i64 16, !"_ZTS3D"}
