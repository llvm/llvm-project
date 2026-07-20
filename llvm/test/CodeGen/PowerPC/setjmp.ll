; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -verify-machineinstrs | FileCheck %s

; Verify that @llvm.eh.sjlj.setjmp stores the special registers
; (FP, IP, SP, TOC, BP) into the buffer.

@buf = internal global [5 x ptr] zeroinitializer, align 8

declare i32 @llvm.eh.sjlj.setjmp(ptr) nounwind

define i32 @setjmp_test() nounwind "frame-pointer"="all" {
  %r = call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  ret i32 %r
}

; CHECK-LABEL: setjmp_test:
; CHECK:       addis [[SCRATCH:[0-9]+]], 2, buf@toc@ha
; CHECK:       addi [[BUFADDR:[0-9]+]], [[SCRATCH]], buf@toc@l
; CHECK-DAG:   std 31, 0([[BUFADDR]])
; CHECK-DAG:   std 1, 16([[BUFADDR]])
; CHECK-DAG:   std 2, 24([[BUFADDR]])
; CHECK-DAG:   std **BASE POINTER**, 32([[BUFADDR]])
; CHECK:       mflr [[IPREG:[0-9]+]]
; CHECK:       std [[IPREG]], 8({{[0-9]+}})
