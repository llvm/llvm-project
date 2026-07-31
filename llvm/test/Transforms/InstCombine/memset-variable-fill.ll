; RUN: opt -passes=instcombine -S < %s | FileCheck %s

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
declare void @llvm.memset.p1.i64(ptr addrspace(1) nocapture writeonly, i8, i64, i1 immarg)
declare void @llvm.memset.element.unordered.atomic.p0.i64(ptr nocapture writeonly, i8, i64, i32 immarg)

define void @variable_fill_len1(ptr %dst, i8 %value) {
; CHECK-LABEL: define void @variable_fill_len1(
; CHECK-SAME: ptr [[DST:%.*]], i8 [[VALUE:%.*]]) {
; CHECK-NEXT:    store i8 [[VALUE]], ptr [[DST]], align 1
; CHECK-NEXT:    ret void
  call void @llvm.memset.p0.i64(ptr align 1 %dst, i8 %value, i64 1, i1 false)
  ret void
}

define void @variable_fill_len1_volatile(ptr %dst, i8 %value) {
; CHECK-LABEL: define void @variable_fill_len1_volatile(
; CHECK-SAME: ptr [[DST:%.*]], i8 [[VALUE:%.*]]) {
; CHECK-NEXT:    store volatile i8 [[VALUE]], ptr [[DST]], align 1
; CHECK-NEXT:    ret void
  call void @llvm.memset.p0.i64(ptr align 1 %dst, i8 %value, i64 1, i1 true)
  ret void
}

define void @variable_fill_len1_align8(ptr %dst, i8 %value) {
; CHECK-LABEL: define void @variable_fill_len1_align8(
; CHECK-SAME: ptr [[DST:%.*]], i8 [[VALUE:%.*]]) {
; CHECK-NEXT:    store i8 [[VALUE]], ptr [[DST]], align 8
; CHECK-NEXT:    ret void
  call void @llvm.memset.p0.i64(ptr align 8 %dst, i8 %value, i64 1, i1 false)
  ret void
}

define void @variable_fill_len1_addrspace(ptr addrspace(1) %dst, i8 %value) {
; CHECK-LABEL: define void @variable_fill_len1_addrspace(
; CHECK-SAME: ptr addrspace(1) [[DST:%.*]], i8 [[VALUE:%.*]]) {
; CHECK-NEXT:    store i8 [[VALUE]], ptr addrspace(1) [[DST]], align 1
; CHECK-NEXT:    ret void
  call void @llvm.memset.p1.i64(ptr addrspace(1) align 1 %dst, i8 %value, i64 1, i1 false)
  ret void
}

define void @variable_fill_len1_atomic(ptr %dst, i8 %value) {
; CHECK-LABEL: define void @variable_fill_len1_atomic(
; CHECK-SAME: ptr [[DST:%.*]], i8 [[VALUE:%.*]]) {
; CHECK-NEXT:    store atomic i8 [[VALUE]], ptr [[DST]] unordered, align 1
; CHECK-NEXT:    ret void
  call void @llvm.memset.element.unordered.atomic.p0.i64(ptr align 1 %dst, i8 %value, i64 1, i32 1)
  ret void
}

define void @variable_fill_len2(ptr %dst, i8 %value) {
; CHECK-LABEL: define void @variable_fill_len2(
; CHECK-SAME: ptr [[DST:%.*]], i8 [[VALUE:%.*]]) {
; CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr {{.*}}[[DST]], i8 [[VALUE]], i64 2, i1 false)
; CHECK-NEXT:    ret void
  call void @llvm.memset.p0.i64(ptr align 1 %dst, i8 %value, i64 2, i1 false)
  ret void
}
