; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

target datalayout = "e-p:64:64"

; CHECK-LABEL: Function: trunc_offset
; CHECK: NoAlias: i32* %p, i32* %q
define void @trunc_offset(ptr %base, i64 %x) {
  %next = add i64 %x, 1
  %i = trunc i64 %x to i32
  %j = trunc i64 %next to i32
  %p = getelementptr i32, ptr %base, i32 %i
  %q = getelementptr i32, ptr %base, i32 %j
  store i32 0, ptr %p
  store i32 0, ptr %q
  ret void
}

; CHECK-LABEL: Function: trunc_wrap
; CHECK: MayAlias: i32* %p, i32* %q
define void @trunc_wrap(ptr %base, i64 %x) {
  %next = add i64 %x, 4294967296
  %i = trunc i64 %x to i32
  %j = trunc i64 %next to i32
  %p = getelementptr i32, ptr %base, i32 %i
  %q = getelementptr i32, ptr %base, i32 %j
  store i32 0, ptr %p
  store i32 0, ptr %q
  ret void
}

; Preserve the inequality proof for i and i + distance when the common index
; is an explicit truncation. Motivated by IntervalMap in LiveDebugVariables.
; CHECK-LABEL: Function: trunc_variable_distance
; CHECK: NoAlias: i8* %dst, i8* %src
define void @trunc_variable_distance(ptr %base, i64 %iv, i32 %distance) {
  %nz = icmp ne i32 %distance, 0
  call void @llvm.assume(i1 %nz)
  %next = add nsw i64 %iv, -1
  %idx = trunc i64 %next to i32
  %masked = zext i32 %idx to i64
  %dstidx = add i32 %distance, %idx
  %dstidx64 = zext i32 %dstidx to i64
  %src = getelementptr inbounds [24 x i8], ptr %base, i64 %masked, i64 8
  %dst = getelementptr inbounds [24 x i8], ptr %base, i64 %dstidx64, i64 8
  %a = load i8, ptr %src
  %b = load i8, ptr %dst
  ret void
}

; Truncation can still be decomposed when it cancels an extension.
; CHECK-LABEL: Function: cancel_trunc
; CHECK: MustAlias: i8* %p, i8* %q
define void @cancel_trunc(ptr %base, i32 %x) {
  %wide = zext i32 %x to i40
  %tagged = or disjoint i40 %wide, 4294967296
  %narrow = trunc i40 %tagged to i32
  %i = zext i32 %x to i64
  %j = zext i32 %narrow to i64
  %p = getelementptr i8, ptr %base, i64 %i
  %q = getelementptr i8, ptr %base, i64 %j
  %a = load i8, ptr %p
  %b = load i8, ptr %q
  ret void
}

declare void @llvm.assume(i1)
