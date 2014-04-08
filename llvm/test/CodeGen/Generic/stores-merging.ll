; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%structTy = type { i8, i32, i32 }

@e = common global %structTy zeroinitializer, align 4

; CHECK-LABEL: f
define void @f() {
entry:

; CHECK:   movabsq	$528280977409, %rax
; CHECK:   movq    %rax, e+4(%rip)
; CHECK:   movl    $456, e+8(%rip)

  store i32 1, i32* getelementptr inbounds (%structTy* @e, i64 0, i32 1), align 4
  store i32 123, i32* getelementptr inbounds (%structTy* @e, i64 0, i32 2), align 4
  store i32 456, i32* getelementptr inbounds (%structTy* @e, i64 0, i32 2), align 4
  ret void
}

