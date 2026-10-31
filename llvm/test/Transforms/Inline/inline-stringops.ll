; RUN: opt -passes=inline -S %s | FileCheck %s

; Opted-in callee inlined into unmarked caller: caller must inherit the attribute.
define internal void @callee_memset(ptr %p) #0 {
entry:
  call void @llvm.memset.p0.i64(ptr align 1 %p, i8 0, i64 512, i1 false)
  ret void
}

define void @caller_unmarked(ptr %p) {
; CHECK-LABEL: define void @caller_unmarked(
; CHECK-SAME: ptr [[P:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr align 1 [[P]], i8 0, i64 512, i1 false)
; CHECK-NEXT:    ret void
entry:
  call void @callee_memset(ptr %p)
  ret void
}

; Unmarked caller with its own memset inlining an opted-in callee:
; Caller inherits the attribute via setOR<InlineMemsetAttr>.
define void @caller_with_own_memset(ptr %p1, ptr %p2) {
; CHECK-LABEL: define void @caller_with_own_memset(
; CHECK-SAME: ptr [[P1:%.*]], ptr [[P2:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr align 1 [[P1]], i8 0, i64 1024, i1 false)
; CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr align 1 [[P2]], i8 0, i64 512, i1 false)
; CHECK-NEXT:    ret void
entry:
  call void @llvm.memset.p0.i64(ptr align 1 %p1, i8 0, i64 1024, i1 false)
  call void @callee_memset(ptr %p2)
  ret void
}

; Non-inlined opted-in callee: unmarked caller must NOT inherit the attribute.
define void @callee_noinline(ptr %p) noinline #0 {
entry:
  call void @llvm.memset.p0.i64(ptr align 1 %p, i8 0, i64 512, i1 false)
  ret void
}

define void @caller_noinline(ptr %p) {
; CHECK-LABEL: define void @caller_noinline(
; CHECK-SAME: ptr [[P:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @callee_noinline(ptr [[P]])
; CHECK-NEXT:    ret void
entry:
  call void @callee_noinline(ptr %p)
  ret void
}

; Unmarked callee inlined into unmarked caller: neither has attribute.
define internal void @callee_unmarked(ptr %p) {
entry:
  call void @llvm.memset.p0.i64(ptr align 1 %p, i8 0, i64 64, i1 false)
  ret void
}

define void @caller_unmarked_callee(ptr %p) {
; CHECK-LABEL: define void @caller_unmarked_callee(
; CHECK-SAME: ptr [[P:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr align 1 [[P]], i8 0, i64 64, i1 false)
; CHECK-NEXT:    ret void
entry:
  call void @callee_unmarked(ptr %p)
  ret void
}

; Callee with "inline-memset"="false" inlined into unmarked caller:
; Caller must NOT inherit the attribute.
define internal void @callee_false_memset(ptr %p) #1 {
entry:
  call void @llvm.memset.p0.i64(ptr align 1 %p, i8 0, i64 128, i1 false)
  ret void
}

define void @caller_false_callee(ptr %p) {
; CHECK-LABEL: define void @caller_false_callee(
; CHECK-SAME: ptr [[P:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr align 1 [[P]], i8 0, i64 128, i1 false)
; CHECK-NEXT:    ret void
entry:
  call void @callee_false_memset(ptr %p)
  ret void
}

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)

attributes #0 = { "inline-memset"="true" }
attributes #1 = { "inline-memset"="false" }

; CHECK: attributes #[[ATTR0]] = { "inline-memset"="true" }
