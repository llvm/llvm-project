; REQUIRES: x86-registered-target

;; Check that successful devirtualization to an alias allows constant propagation
;; and attribute deduction for the caller.

;; Generate split module with summary for hybrid Thin/Regular LTO WPD.
; RUN: opt -passes=assign-guid -thinlto-bc -thinlto-split-lto-unit -o %t.o %s

; RUN: llvm-lto2 run %t.o -save-temps \
; RUN:   -whole-program-visibility \
; RUN:   -o %t2 \
; RUN:   -r=%t.o,test,px \
; RUN:   -r=%t.o,_ZTV1D, \
; RUN:   -r=%t.o,_ZTV1D,px \
; RUN:   -r=%t.o,_ZN1D1mEi,px \
; RUN:   -r=%t.o,_ZN1D1mEiAlias,px \
; RUN:   -r=%t.o,_ZN1D1mEiAlias,
; RUN: llvm-dis %t2.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

@_ZTV1D = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr poison, ptr @_ZN1D1mEiAlias] }, !type !3

;; The aliasee has NO optnone or noinline, permitting IPSCCP to fold ret i32 0.
define i32 @_ZN1D1mEi(ptr %this, i32 %a) {
   ret i32 0
}

@_ZN1D1mEiAlias = hidden unnamed_addr alias i32 (ptr, i32), ptr @_ZN1D1mEi

; CHECK-IR-LABEL: define noundef i32 @test
define i32 @test(ptr %obj2, i32 %a) {
entry:
  %vtable2 = load ptr, ptr %obj2
  %p2 = call i1 @llvm.type.test(ptr %vtable2, metadata !"_ZTS1D")
  call void @llvm.assume(i1 %p2)

  %fptr33 = load ptr, ptr %vtable2, align 8

  ;; Check that the call was devirtualized and folded to 0, granting @test noundef.
  ;; CHECK-IR-NOT: tail call
  ;; CHECK-IR: ret i32 0
  %call4 = tail call i32 %fptr33(ptr nonnull %obj2, i32 %a)
  ret i32 %call4
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!3 = !{i64 16, !"_ZTS1D"}
