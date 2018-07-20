; RUN: llc < %s -asm-verbose=false | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%0 = type { %1, i64, %2 }
%1 = type { i8* }
%2 = type { i64, [8 x i8] }

@0 = internal constant [10 x i8] c"asdf jkl;\00", align 1

; Memcpy lowering should emit stores of immediates containing string data from
; the correct offsets.

; CHECK-LABEL: foo:
; CHECK: movb  $0, 6(%rdi)
; CHECK: movw  $15212, 4(%rdi)
; CHECK: movl  $1802117222, (%rdi)
define void @foo(i8* %tmp2) {
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tmp2, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @0, i64 0, i64 3), i64 7, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1)
