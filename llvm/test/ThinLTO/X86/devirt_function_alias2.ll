; REQUIRES: x86-registered-target

;; Check for successful devirtualization when vtable contains an alias,
;; and there is a single implementation.

;; Test pure ThinLTO

;; Generate unsplit module with summary for ThinLTO index-based WPD.
; RUN: opt -thinlto-bc -o %t1.o %s

;; Check that we have properly recorded the alias in the vtable summary.
; RUN: llvm-dis -o - %t1.o | FileCheck %s --check-prefix SUMMARY
; SUMMARY: gv: (name: "_ZTV1D", {{.*}} vTableFuncs: ((virtFunc: ^[[ALIAS:([0-9]+)]], offset: 16))
; SUMMARY: ^[[ALIAS]] = gv: (name: "_ZN1D1mEiAlias"

; RUN: llvm-lto2 run %t1.o -save-temps -pass-remarks=. \
; RUN:   -whole-program-visibility \
; RUN:   -wholeprogramdevirt-print-index-based \
; RUN:   -o %t2 \
; RUN:   -r=%t1.o,test,px \
; RUN:   -r=%t1.o,_ZTV1D,px \
; RUN:   -r=%t1.o,_ZN1D1mEi,px \
; RUN:   -r=%t1.o,_ZN1D1mEiAlias,px \
; RUN:   2>&1 | FileCheck %s --check-prefix=REMARK --check-prefix=PRINT
; RUN: llvm-dis %t2.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR1

; PRINT-DAG: Devirtualized call to {{.*}} (_ZN1D1mEiAlias)
; REMARK-DAG: single-impl: devirtualized a call to _ZN1D1mEiAlias

;; Test hybrid Thin/Regular LTO

;; Generate split module with summary for hybrid Thin/Regular LTO WPD.
; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t3.o %s

; RUN: llvm-lto2 run %t3.o -save-temps -pass-remarks=. \
; RUN:   -whole-program-visibility \
; RUN:   -o %t4 \
; RUN:   -r=%t3.o,test,px \
; RUN:   -r=%t3.o,_ZTV1D, \
; RUN:   -r=%t3.o,_ZTV1D,px \
; RUN:   -r=%t3.o,_ZN1D1mEi,px \
; RUN:   -r=%t3.o,_ZN1D1mEiAlias,px \
; RUN:   -r=%t3.o,_ZN1D1mEiAlias, \
; RUN:   2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t4.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR1

;; Test Regular LTO
; RUN: opt -o %t5.o %s
; RUN: llvm-lto2 run %t5.o -save-temps -pass-remarks=. \
; RUN:   -whole-program-visibility \
; RUN:   -o %t6 \
; RUN:   -r=%t5.o,test,px \
; RUN:   -r=%t5.o,_ZTV1D,px \
; RUN:   -r=%t5.o,_ZN1D1mEi,px \
; RUN:   -r=%t5.o,_ZN1D1mEiAlias,px \
; RUN:   2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t6.0.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR1

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.D = type { ptr }

@_ZTV1D = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr undef, ptr @_ZN1D1mEiAlias] }, !type !3

define i32 @_ZN1D1mEi(ptr %this, i32 %a) {
   ret i32 0;
}

@_ZN1D1mEiAlias = unnamed_addr alias i32 (ptr, i32), ptr @_ZN1D1mEi

; CHECK-IR1-LABEL: define i32 @test
define i32 @test(ptr %obj2, i32 %a) {
entry:
  %vtable2 = load ptr, ptr %obj2
  %p2 = call i1 @llvm.type.test(ptr %vtable2, metadata !"_ZTS1D")
  call void @llvm.assume(i1 %p2)

  %fptr33 = load ptr, ptr %vtable2, align 8

  ;; Check that the call was devirtualized.
  ;; CHECK-IR1: %call4 = tail call i32 @_ZN1D1mEi
  %call4 = tail call i32 %fptr33(ptr nonnull %obj2, i32 %a)
  ret i32 %call4
}
; CHECK-IR1-LABEL: ret i32
; CHECK-IR1-LABEL: }

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!3 = !{i64 16, !"_ZTS1D"}
