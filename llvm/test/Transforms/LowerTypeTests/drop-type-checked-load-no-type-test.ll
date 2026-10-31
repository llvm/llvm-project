;; A checked load is dropped even with no @llvm.type.test anywhere in the
;; module, and that alone triggers the vcall_visibility metadata erasure.
; RUN: opt -S -passes="drop-type-tests<all>" < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; CHECK: @vt = constant [2 x ptr] zeroinitializer
; CHECK-NOT: vcall_visibility
@vt = constant [2 x ptr] [ptr null, ptr null], !vcall_visibility !0

declare {ptr, i1} @llvm.type.checked.load(ptr, i32, metadata)
declare void @use(ptr)

; CHECK-LABEL: define void @f(
; CHECK-NEXT: %[[SLOT:.*]] = getelementptr i8, ptr %vtable, i32 0
; CHECK-NEXT: %[[FP:.*]] = load ptr, ptr %[[SLOT]], align 8
; CHECK-NEXT: %[[S0:.*]] = insertvalue { ptr, i1 } { ptr poison, i1 true }, ptr %[[FP]], 0
; CHECK-NOT: call {{.*}}@llvm.type.checked.load(
define void @f(ptr %vtable) {
  %pair = call {ptr, i1} @llvm.type.checked.load(ptr %vtable, i32 0, metadata !"typeid")
  %fptr = extractvalue {ptr, i1} %pair, 0
  call void @use(ptr %fptr)
  ret void
}

!0 = !{i64 0}
