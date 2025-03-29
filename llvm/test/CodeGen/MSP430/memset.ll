; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16"
target triple = "msp430---elf"

@buf = external global ptr

; Function Attrs: nounwind
define void @test() nounwind {
entry:
; CHECK-LABEL: test:
  %0 = load ptr, ptr @buf, align 2
; CHECK: mov &buf, r12
; CHECK-NEXT: mov #5, r13
; CHECK-NEXT: mov #128, r14
; CHECK-NEXT: call #memset
  call void @llvm.memset.p0.i16(ptr %0, i8 5, i16 128, i1 false)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.memset.p0.i16(ptr nocapture, i8, i16, i1) nounwind

