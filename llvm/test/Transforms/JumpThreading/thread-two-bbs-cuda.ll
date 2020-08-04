; RUN: opt < %s -jump-threading -S -verify | FileCheck %s

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

$wrapped_tid = comdat any

$foo = comdat any

define i32 @wrapped_tid() #0 comdat align 32 {
  %1 = call i32 @tid()
  ret i32 %1
}

declare void @llvm.nvvm.barrier0() #1

; We had a bug where we duplicated basic blocks containing convergent
; functions like @llvm.nvvm.barrier0 below.  Verify that we don't do
; that.
define void @foo() local_unnamed_addr #2 comdat align 32 {
; CHECK-LABEL: @foo
  %1 = call i32 @tid()
  %2 = urem i32 %1, 7
  br label %3

3:
  %4 = icmp eq i32 %1, 0
  br i1 %4, label %5, label %6

5:
  call void @bar()
  br label %6

6:
; CHECK: call void @llvm.nvvm.barrier0()
; CHECK-NOT: call void @llvm.nvvm.barrier0()
  call void @llvm.nvvm.barrier0()
  %7 = icmp eq i32 %2, 0
  br i1 %7, label %11, label %8

8:
  %9 = icmp ult i32 %1, 49
  br i1 %9, label %10, label %11

10:
  call void @llvm.trap()
  unreachable

11:
  br label %3
}

declare i32 @tid() #2

declare void @bar()

declare void @llvm.trap() #3

attributes #1 = { convergent }
attributes #2 = { readnone }
attributes #3 = { noreturn }
attributes #4 = { convergent }
