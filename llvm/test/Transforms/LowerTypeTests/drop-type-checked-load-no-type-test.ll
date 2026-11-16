;; A checked load is dropped even with no @llvm.type.test anywhere in the
;; module, and that alone triggers the vcall_visibility metadata erasure.
; RUN: opt -S -passes="drop-type-tests<all>" < %s | FileCheck %s
; RUN: opt -S -passes="fatlto-cleanup" < %s | FileCheck %s --check-prefix=CLEANUP
; RUN: opt -S -passes="fatlto-pre-link<O2>" < %s | FileCheck %s --check-prefix=FATLTO

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; CHECK: @vt = internal constant [1 x ptr] [ptr @vfunc]
; CHECK-NOT: vcall_visibility
; CLEANUP: @vt = internal constant [1 x ptr] [ptr @vfunc]
; CLEANUP-NOT: vcall_visibility
; FATLTO: @vt = internal constant [1 x ptr] [ptr @vfunc]
; FATLTO-NOT: vcall_visibility
@vt = internal constant [1 x ptr] [ptr @vfunc], !type !0, !vcall_visibility !1

; CHECK: define internal void @vfunc()
; CLEANUP: define internal void @vfunc()
; FATLTO: define internal void @vfunc()
define internal void @vfunc() {
  ret void
}

declare {ptr, i1} @llvm.type.checked.load(ptr, i32, metadata)
declare void @use(ptr)

; CHECK-LABEL: define ptr @f(
; CHECK-NEXT: %[[SLOT:.*]] = getelementptr i8, ptr %vtable, i64 0
; CHECK-NEXT: %[[FP:.*]] = load ptr, ptr %[[SLOT]], align 8
; CHECK-NEXT: %[[S0:.*]] = insertvalue { ptr, i1 } { ptr poison, i1 true }, ptr %[[FP]], 0
; CHECK-NOT: call {{.*}}@llvm.type.checked.load(
; CLEANUP-LABEL: define ptr @f(
; CLEANUP-NEXT: %[[SLOT:.*]] = getelementptr i8, ptr %vtable, i64 0
; CLEANUP-NEXT: %[[FP:.*]] = load ptr, ptr %[[SLOT]], align 8
; CLEANUP-NEXT: %[[S0:.*]] = insertvalue { ptr, i1 } { ptr poison, i1 true }, ptr %[[FP]], 0
; CLEANUP-NOT: call {{.*}}@llvm.type.checked.load(
; FATLTO-LABEL: define noundef nonnull ptr @f(
; FATLTO-NEXT: %vfunc = load ptr, ptr %vtable, align 8
; FATLTO-NEXT: tail call void @use(ptr %vfunc)
; FATLTO-NEXT: ret ptr @vt
define ptr @f(ptr %vtable) {
  %pair = call {ptr, i1} @llvm.type.checked.load(ptr %vtable, i32 0, metadata !"typeid")
  %fptr = extractvalue {ptr, i1} %pair, 0
  call void @use(ptr %fptr)
  ret ptr @vt
}

!llvm.module.flags = !{!2}
!0 = !{i64 0, !"typeid"}
!1 = !{i64 2}
!2 = !{i32 1, !"Virtual Function Elim", i32 1}
