; RUN: opt -passes=pre-isel-intrinsic-lowering -S -o - %s | FileCheck %s

; Small constant (32 bytes) with "inline-memset" should convert to llvm.memset.inline
define void @optin_small_const(ptr %p) #0 {
; CHECK-LABEL: define void @optin_small_const(
; CHECK-SAME: ptr [[P:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.memset.inline.p0.i64(ptr align 1 [[P]], i8 0, i64 32, i1 false)
; CHECK-NEXT:    ret void
entry:
  call void @llvm.memset.p0.i64(ptr align 1 %p, i8 0, i64 32, i1 false)
  ret void
}

; Large constant (512 bytes) with "inline-memset" should expand to loop
define void @optin_large_const(ptr %p) #0 {
; CHECK-LABEL: define void @optin_large_const(
; CHECK-SAME: ptr [[P:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %[[LOOP:.*]]
; CHECK:       [[LOOP]]:
; CHECK:         store i8 0, ptr
; CHECK:         icmp ult i64 {{.*}}, 512
; CHECK:         br i1 {{.*}}, label %[[LOOP]], label %[[EXIT:.*]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    ret void
entry:
  call void @llvm.memset.p0.i64(ptr align 1 %p, i8 0, i64 512, i1 false)
  ret void
}

; Variable length with "inline-memset" should expand to loop
define void @optin_var(ptr %p, i64 %len) #0 {
; CHECK-LABEL: define void @optin_var(
; CHECK-SAME: ptr [[P:%.*]], i64 [[LEN:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp ne i64 [[LEN]], 0
; CHECK-NEXT:    br i1 [[CMP]], label %[[VAR_LOOP:.*]], label %[[VAR_EXIT:.*]]
; CHECK:       [[VAR_LOOP]]:
; CHECK:         store i8 0, ptr
; CHECK:         icmp ult i64 {{.*}}, [[LEN]]
; CHECK:         br i1 {{.*}}, label %[[VAR_LOOP]], label %[[VAR_EXIT]]
; CHECK:       [[VAR_EXIT]]:
; CHECK-NEXT:    ret void
entry:
  call void @llvm.memset.p0.i64(ptr align 1 %p, i8 0, i64 %len, i1 false)
  ret void
}

; Small constant in unmarked function should remain untouched
define void @unmarked_small_const(ptr %p) {
; CHECK-LABEL: define void @unmarked_small_const(
; CHECK-SAME: ptr [[P:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr align 1 [[P]], i8 0, i64 32, i1 false)
; CHECK-NEXT:    ret void
entry:
  call void @llvm.memset.p0.i64(ptr align 1 %p, i8 0, i64 32, i1 false)
  ret void
}

; Large constant in unmarked function should remain untouched
define void @unmarked_large_const(ptr %p) {
; CHECK-LABEL: define void @unmarked_large_const(
; CHECK-SAME: ptr [[P:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr align 1 [[P]], i8 0, i64 512, i1 false)
; CHECK-NEXT:    ret void
entry:
  call void @llvm.memset.p0.i64(ptr align 1 %p, i8 0, i64 512, i1 false)
  ret void
}

; Variable length in unmarked function should remain untouched
define void @unmarked_var(ptr %p, i64 %len) {
; CHECK-LABEL: define void @unmarked_var(
; CHECK-SAME: ptr [[P:%.*]], i64 [[LEN:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr align 1 [[P]], i8 0, i64 [[LEN]], i1 false)
; CHECK-NEXT:    ret void
entry:
  call void @llvm.memset.p0.i64(ptr align 1 %p, i8 0, i64 %len, i1 false)
  ret void
}

; Function with "inline-memset"="false" should NOT transform memset
define void @optout_explicit_false(ptr %p) #1 {
; CHECK-LABEL: define void @optout_explicit_false(
; CHECK-SAME: ptr [[P:%.*]]) #[[ATTR1:[0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr align 1 [[P]], i8 0, i64 512, i1 false)
; CHECK-NEXT:    ret void
entry:
  call void @llvm.memset.p0.i64(ptr align 1 %p, i8 0, i64 512, i1 false)
  ret void
}

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)

attributes #0 = { "inline-memset"="true" }
attributes #1 = { "inline-memset"="false" }

; CHECK: attributes #[[ATTR0]] = { "inline-memset"="true" }
; CHECK: attributes #[[ATTR1]] = { "inline-memset"="false" }
