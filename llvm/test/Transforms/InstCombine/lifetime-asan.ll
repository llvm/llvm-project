; RUN: opt < %s -instcombine -S | FileCheck %s

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)
declare void @foo(i8* nocapture)

define void @asan() sanitize_address {
entry:
  ; CHECK-LABEL: @asan(
  %text = alloca i8, align 1

  call void @llvm.lifetime.start.p0i8(i64 1, i8* %text)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %text)
  ; CHECK: call void @llvm.lifetime.start
  ; CHECK-NEXT: call void @llvm.lifetime.end

  call void @foo(i8* %text) ; Keep alloca alive

  ret void
}


define void @no_asan() {
entry:
  ; CHECK-LABEL: @no_asan(
  %text = alloca i8, align 1

  call void @llvm.lifetime.start.p0i8(i64 1, i8* %text)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %text)
  ; CHECK-NO: call void @llvm.lifetime

  call void @foo(i8* %text) ; Keep alloca alive

  ret void
}
