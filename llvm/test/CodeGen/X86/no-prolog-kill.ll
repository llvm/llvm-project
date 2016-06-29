; RUN: llc -verify-machineinstrs -o - %s | FileCheck %s
target triple = "x86_64--"

; This function gets a AL live-in and at same time saves+restores RAX. We must
; not add a kill flag to the "PUSHQ %rax" or the machine verifier will complain.
; CHECK-LABEL: test:
; CHECK: pushq %rax
; CHECK: testb %al, %al
; CHECK: je .LBB
define void @test(i64 %a, i8* %b, ...)  {
entry:
  %bar = alloca i8
  call void @llvm.va_start(i8* %bar)
  call void @llvm.eh.unwind.init()
  call void @llvm.eh.return.i64(i64 %a, i8* %b)
  unreachable
}

declare void @llvm.eh.return.i64(i64, i8*)
declare void @llvm.eh.unwind.init()
declare void @llvm.va_start(i8*)
