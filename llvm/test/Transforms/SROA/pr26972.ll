; RUN: opt < %s -sroa -S | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux"

; Make sure we properly handle allocas where the allocated
; size overflows a uint32_t. This specific constant results in
; the size in bits being 32 after truncation to a 32-bit int.
; CHECK-LABEL: fn1
; CHECK-NEXT: ret void
define void @fn1() {
  %a = alloca [1073741825 x i32], align 16
  %t0 = bitcast [1073741825 x i32]* %a to i8*
  call void @llvm.lifetime.end.p0i8(i64 4294967300, i8* %t0)
  ret void
}

declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)
