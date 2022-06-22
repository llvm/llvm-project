; RUN: opt < %s -passes=sroa -S | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux"

; Make sure we properly handle allocas where the allocated
; size overflows a uint32_t. This specific constant results in
; the size in bits being 32 after truncation to a 32-bit int.
; CHECK-LABEL: fn1
; CHECK-NEXT: ret void
define void @fn1() {
  %a = alloca [1073741825 x i32], align 16
  call void @llvm.lifetime.end.p0(i64 4294967300, ptr %a)
  ret void
}

declare void @llvm.lifetime.end.p0(i64, ptr nocapture)
