; RUN: opt -passes=attributor -S < %s | FileCheck %s
; RUN: opt -passes=attributor-cgscc -S < %s | FileCheck %s

declare void @use(ptr)

; Parameter attributes such as align and nonnull produce poison when their
; constraints are violated. Unless the argument is also noundef (or otherwise
; rejects poison), these attributes do not describe the original value.

define void @align_without_noundef(ptr %p) {
; CHECK-LABEL: define void @align_without_noundef(
; CHECK-SAME: ptr [[P:%.*]]) {
; CHECK-NEXT:    call void @use(ptr align 4 [[P]])
; CHECK-NEXT:    store i8 1, ptr [[P]], align 1
; CHECK-NEXT:    ret void
  call void @use(ptr align 4 %p)
  store i8 1, ptr %p, align 1
  ret void
}

define void @align_with_noundef(ptr %p) {
; CHECK-LABEL: define void @align_with_noundef(
; CHECK-SAME: ptr noundef align 4 [[P:%.*]]) {
; CHECK-NEXT:    call void @use(ptr noundef align 4 [[P]])
; CHECK-NEXT:    store i8 1, ptr [[P]], align 4
; CHECK-NEXT:    ret void
  call void @use(ptr noundef align 4 %p)
  store i8 1, ptr %p, align 1
  ret void
}

define i1 @nonnull_without_noundef(ptr %p) {
; CHECK-LABEL: define i1 @nonnull_without_noundef(
; CHECK-SAME: ptr [[P:%.*]]) {
; CHECK-NEXT:    call void @use(ptr nonnull [[P]])
; CHECK-NEXT:    [[CMP:%.*]] = icmp ne ptr [[P]], null
; CHECK-NEXT:    ret i1 [[CMP]]
  call void @use(ptr nonnull %p)
  %cmp = icmp ne ptr %p, null
  ret i1 %cmp
}

define i1 @nonnull_with_noundef(ptr %p) {
; CHECK-LABEL: define noundef i1 @nonnull_with_noundef(
; CHECK-SAME: ptr noundef nonnull [[P:%.*]]) {
; CHECK-NEXT:    call void @use(ptr noundef nonnull [[P]])
; CHECK-NEXT:    ret i1 true
  call void @use(ptr noundef nonnull %p)
  %cmp = icmp ne ptr %p, null
  ret i1 %cmp
}
