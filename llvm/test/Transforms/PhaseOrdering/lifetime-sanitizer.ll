; RUN: opt < %s -O0 -S | FileCheck %s --check-prefixes=CHECK,NOOPT
; RUN: opt < %s -O1 -S | FileCheck %s --check-prefixes=CHECK,OPT
; RUN: opt < %s -O2 -S | FileCheck %s --check-prefixes=CHECK,OPT
; RUN: opt < %s -O3 -S | FileCheck %s --check-prefixes=CHECK,OPT
; RUN: opt < %s -passes='default<O0>' -S | FileCheck %s --check-prefixes=CHECK,NOOPT
; RUN: opt < %s -passes='default<O1>' -S | FileCheck %s --check-prefixes=CHECK,OPT
; RUN: opt < %s -passes='default<O2>' -S | FileCheck %s --check-prefixes=CHECK,OPT
; RUN: opt < %s -passes="default<O3>" -S | FileCheck %s --check-prefixes=CHECK,OPT

declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)
declare void @foo(ptr nocapture)

define void @asan() sanitize_address {
entry:
  ; CHECK-LABEL: @asan(
  %text = alloca i8, align 1

  call void @llvm.lifetime.start.p0(i64 1, ptr %text)
  call void @llvm.lifetime.end.p0(i64 1, ptr %text)
  ; CHECK: call void @llvm.lifetime.start
  ; CHECK-NEXT: call void @llvm.lifetime.end

  call void @foo(ptr %text) ; Keep alloca alive

  ret void
}

define void @hwasan() sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @hwasan(
  %text = alloca i8, align 1

  call void @llvm.lifetime.start.p0(i64 1, ptr %text)
  call void @llvm.lifetime.end.p0(i64 1, ptr %text)
  ; CHECK: call void @llvm.lifetime.start
  ; CHECK-NEXT: call void @llvm.lifetime.end

  call void @foo(ptr %text) ; Keep alloca alive

  ret void
}

define void @msan() sanitize_memory {
entry:
  ; CHECK-LABEL: @msan(
  %text = alloca i8, align 1

  call void @llvm.lifetime.start.p0(i64 1, ptr %text)
  call void @llvm.lifetime.end.p0(i64 1, ptr %text)
  ; CHECK: call void @llvm.lifetime.start
  ; CHECK-NEXT: call void @llvm.lifetime.end

  call void @foo(ptr %text) ; Keep alloca alive

  ret void
}

define void @no_asan() {
entry:
  ; CHECK-LABEL: @no_asan(
  %text = alloca i8, align 1

  call void @llvm.lifetime.start.p0(i64 1, ptr %text)
  call void @llvm.lifetime.end.p0(i64 1, ptr %text)
  ; OPT-NOT:    call void @llvm.lifetime
  ; NOOPT:      call void @llvm.lifetime.start
  ; NOOPT-NEXT: call void @llvm.lifetime.end

  call void @foo(ptr %text) ; Keep alloca alive

  ret void
}
