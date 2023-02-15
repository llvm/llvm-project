; RUN: opt -mtriple=riscv32 -data-layout="e-m:e-p:32:32" -S -codegenprepare < %s \
; RUN:   | FileCheck %s '-D#NEW_ALIGNMENT=4'
; RUN: opt -mtriple=riscv64 -data-layout="e-m:e-p:64:64" -S -codegenprepare < %s \
; RUN:   | FileCheck %s '-D#NEW_ALIGNMENT=8'

@str = private unnamed_addr constant [45 x i8] c"THIS IS A LONG STRING THAT SHOULD BE ALIGNED\00", align 1


declare void @use(ptr %arg)


; CHECK: @[[STR:[a-zA-Z0-9_$"\\.-]+]] = private unnamed_addr constant [45 x i8] c"THIS IS A LONG STRING THAT SHOULD BE ALIGNED\00", align [[#NEW_ALIGNMENT]]

define void @foo() {
; CHECK-LABEL: @foo(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[DST:%.*]] = alloca [45 x i8], align [[#NEW_ALIGNMENT]]
; CHECK-NEXT:    tail call void @llvm.memcpy.p0.p0.i32(ptr align [[#NEW_ALIGNMENT]] [[DST]], ptr align [[#NEW_ALIGNMENT]] dereferenceable(31) @str, i32 31, i1 false)
; CHECK-NEXT:    ret void

entry:
  %dst = alloca [45 x i8], align 1
  tail call void @llvm.memcpy.p0i8.p0i8.i32(ptr align 1 %dst, ptr align 1 dereferenceable(31) @str, i32 31, i1 false)
  ret void
}

; negative test - check that we don't align objects that are too small
define void @no_align(ptr %src) {
; CHECK-LABEL: @no_align(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[DST:%.*]] = alloca [3 x i8], align 1
; CHECK-NEXT:    tail call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[DST]], ptr align 1 [[SRC:%.*]], i32 31, i1 false)
; CHECK-NEXT:    ret void
;
entry:
  %dst = alloca [3 x i8], align 1
  tail call void @llvm.memcpy.p0i8.p0i8.i32(ptr align 1 %dst, ptr %src, i32 31, i1 false)
  ret void
}

; negative test - check that minsize requires at least 8 byte object size
define void @no_align_minsize(ptr %src) minsize {
; CHECK-LABEL: @no_align_minsize(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[DST:%.*]] = alloca [7 x i8], align 1
; CHECK-NEXT:    tail call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[DST]], ptr align 1 [[SRC:%.*]], i32 31, i1 false)
; CHECK-NEXT:    ret void
;
entry:
  %dst = alloca [7 x i8], align 1
  tail call void @llvm.memcpy.p0i8.p0i8.i32(ptr align 1 %dst, ptr %src, i32 31, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1)
