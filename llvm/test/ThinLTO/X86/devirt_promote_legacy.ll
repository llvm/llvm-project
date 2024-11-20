; REQUIRES: x86-registered-target

; Test devirtualization requiring promotion of local targets, where the
; promotion is required by one devirtualization and needs to be updated
; for a second devirtualization in the defining module as a post-pass
; update.

; Generate unsplit module with summary for ThinLTO index-based WPD.
; RUN: opt -thinlto-bc -o %t3.o %s
; RUN: opt -thinlto-bc -o %t4.o %p/Inputs/devirt_promote.ll

; RUN: llvm-lto -thinlto-action=run %t3.o %t4.o --thinlto-save-temps=%t5. \
; RUN:   -whole-program-visibility \
; RUN:   --pass-remarks=. \
; RUN:   --exported-symbol=test \
; RUN:   --exported-symbol=test2 \
; RUN:   --exported-symbol=_ZTV1B 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t5.0.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR1
; RUN: llvm-dis %t5.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR2

; We should devirt call to _ZN1A1nEi once in importing module and once
; in original (exporting) module.
; REMARK-COUNT-2: single-impl: devirtualized a call to _ZN1A1nEi.llvm.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.A = type { ptr }

; CHECK-IR1-LABEL: define i32 @test
define i32 @test(ptr %obj, i32 %a) {
entry:
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr ptr, ptr %vtable, i32 1
  %fptr1 = load ptr, ptr %fptrptr, align 8

  ; Check that the call was devirtualized.
  ; CHECK-IR1: = tail call i32 @_ZN1A1nEi
  %call = tail call i32 %fptr1(ptr nonnull %obj, i32 %a)

  ret i32 %call
}
; CHECK-IR1-LABEL: ret i32
; CHECK-IR1-LABEL: }

; CHECK-IR2: define noundef i32 @test2
; Check that the call was devirtualized.
; CHECK-IR2: = tail call i32 @_ZN1A1nEi

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

attributes #0 = { noinline optnone }
