; RUN: rm -rf %t && split-file %s %t
; REQUIRES: x86-registered-target

;; Check for successful devirtualization across ThinLTO modules when vtable 
;; contains an alias. Ensures the ThinLTO index step properly connects the alias 
;; to the aliasee, allowing importing and devirtualization in the caller's backend.

; RUN: opt -passes=assign-guid -thinlto-bc -o %t_caller.o %t/caller.ll
; RUN: opt -passes=assign-guid -thinlto-bc -o %t_callee.o %t/callee.ll

; RUN: llvm-lto2 run %t_caller.o %t_callee.o -save-temps -pass-remarks=. \
; RUN:   -whole-program-visibility \
; RUN:   -wholeprogramdevirt-print-index-based \
; RUN:   -o %t_out \
; RUN:   -r=%t_caller.o,test_interposable_aliasee,px \
; RUN:   -r=%t_caller.o,test_interposable_alias,px \
; RUN:   -r=%t_callee.o,_ZTV2D,px \
; RUN:   -r=%t_callee.o,_ZN2D1mEi,px \
; RUN:   -r=%t_callee.o,_ZN2D1mEiAlias,px \
; RUN:   -r=%t_callee.o,_ZTV3D,px \
; RUN:   -r=%t_callee.o,_ZN3D1mEi,px \
; RUN:   -r=%t_callee.o,_ZN3D1mEiAlias,px \
; RUN:   2>&1 | FileCheck %s --check-prefix=REMARK --check-prefix=PRINT
; RUN: llvm-dis %t_out.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR

; PRINT-DAG: Devirtualized call to {{.*}} (_ZN2D1mEiAlias)
; PRINT-DAG: Devirtualized call to {{.*}} (_ZN3D1mEiAlias)
; REMARK-DAG: single-impl: devirtualized a call to _ZN2D1mEiAlias
; REMARK-DAG: single-impl: devirtualized a call to _ZN3D1mEiAlias

;--- caller.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.D = type { ptr }

; CHECK-IR-LABEL: define i32 @test_interposable_aliasee(
define i32 @test_interposable_aliasee(ptr %obj2, i32 %a) {
entry:
  %vtable2 = load ptr, ptr %obj2
  %p2 = call i1 @llvm.type.test(ptr %vtable2, metadata !"_ZTS2D")
  call void @llvm.assume(i1 %p2)
  %fptr33 = load ptr, ptr %vtable2, align 8
  ;; CHECK-IR: %call4 = tail call i32 @_ZN2D1mEiAlias
  %call4 = tail call i32 %fptr33(ptr nonnull %obj2, i32 %a)
  ret i32 %call4
}

; CHECK-IR-LABEL: define i32 @test_interposable_alias(
define i32 @test_interposable_alias(ptr %obj2, i32 %a) {
entry:
  %vtable2 = load ptr, ptr %obj2
  %p2 = call i1 @llvm.type.test(ptr %vtable2, metadata !"_ZTS3D")
  call void @llvm.assume(i1 %p2)
  %fptr33 = load ptr, ptr %vtable2, align 8
  ;; CHECK-IR: %call4 = tail call i32 @_ZN3D1mEiAlias
  %call4 = tail call i32 %fptr33(ptr nonnull %obj2, i32 %a)
  ret i32 %call4
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

;--- callee.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.D = type { ptr }

@_ZTV2D = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr poison, ptr @_ZN2D1mEiAlias] }, !type !4
define weak i32 @_ZN2D1mEi(ptr %this, i32 %a) #0 {
   ret i32 0;
}
@_ZN2D1mEiAlias = hidden unnamed_addr alias i32 (ptr, i32), ptr @_ZN2D1mEi
!4 = !{i64 16, !"_ZTS2D"}

@_ZTV3D = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr poison, ptr @_ZN3D1mEiAlias] }, !type !5
define i32 @_ZN3D1mEi(ptr %this, i32 %a) #0 {
   ret i32 0;
}
@_ZN3D1mEiAlias = weak alias i32 (ptr, i32), ptr @_ZN3D1mEi
!5 = !{i64 16, !"_ZTS3D"}

attributes #0 = { noinline optnone }
