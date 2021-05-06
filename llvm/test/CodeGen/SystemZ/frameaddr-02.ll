; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test lowering of @llvm.frameaddress with packed-stack.

; With back chain
attributes #0 = { nounwind "packed-stack" "backchain" "use-soft-float"="true" }
define i8* @fp0() #0 {
entry:
; CHECK-LABEL: fp0:
; CHECK:      la   %r2, 152(%r15)
; CHECK-NEXT: br   %r14
  %0 = tail call i8* @llvm.frameaddress(i32 0)
  ret i8* %0
}

define i8* @fp0f() #0 {
entry:
; CHECK-LABEL: fp0f:
; CHECK:      lgr	%r1, %r15
; CHECK-NEXT: aghi	%r15, -16
; CHECK-NEXT: stg	%r1, 152(%r15)
; CHECK-NEXT: la	%r2, 168(%r15)
; CHECK-NEXT: aghi	%r15, 16
; CHECK-NEXT: br	%r14
  %0 = alloca i64, align 8
  %1 = tail call i8* @llvm.frameaddress(i32 0)
  ret i8* %1
}

; Without back chain

attributes #1 = { nounwind "packed-stack" }
define i8* @fp1() #1 {
entry:
; CHECK-LABEL: fp1:
; CHECK:      la   %r2, 152(%r15)
; CHECK-NEXT: br   %r14
  %0 = tail call i8* @llvm.frameaddress(i32 0)
  ret i8* %0
}

; No saved registers: returning address of unused slot where backcahin would
; have been located.
define i8* @fp1f() #1 {
entry:
; CHECK-LABEL: fp1f:
; CHECK:      aghi	%r15, -16
; CHECK-NEXT: la	%r2, 168(%r15)
; CHECK-NEXT: aghi	%r15, 16
; CHECK-NEXT: br	%r14
  %0 = alloca i64, align 8
  %1 = tail call i8* @llvm.frameaddress(i32 0)
  ret i8* %1
}

; Saved registers: returning address for first saved GPR.
declare void @foo(i8* %Arg)
define i8* @fp2() #1 {
entry:
; CHECK-LABEL: fp2:
; CHECK:      stmg      %r14, %r15, 144(%r15)
; CHECK-NEXT: aghi	%r15, -16
; CHECK-NEXT: la	%r2, 168(%r15)
; CHECK-NEXT: brasl     %r14, foo@PLT
; CHECK-NEXT: la	%r2, 168(%r15)
; CHECK-NEXT: lmg       %r14, %r15, 160(%r15)
; CHECK-NEXT: br	%r14
  %0 = tail call i8* @llvm.frameaddress(i32 0)
  call void @foo(i8* %0);
  ret i8* %0
}

declare i8* @llvm.frameaddress(i32) nounwind readnone
