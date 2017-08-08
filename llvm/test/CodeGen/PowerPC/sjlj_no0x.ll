; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=a2 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 -verify-machineinstrs | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; Function Attrs: noinline nounwind
define void @_Z23BuiltinLongJmpFunc1_bufv() #0 {
entry:
  call void @llvm.eh.sjlj.longjmp(i8* bitcast (void ()* @_Z23BuiltinLongJmpFunc1_bufv to i8*))
  unreachable

; CHECK: @_Z23BuiltinLongJmpFunc1_bufv
; CHECK: addis [[REG:[0-9]+]], 2, .LC0@toc@ha
; CHECK: ld 31, 0([[REG]])
; CHECK: ld [[REG2:[0-9]+]], 8([[REG]])
; CHECK-DAG: ld 1, 16([[REG]])
; CHECK-DAG: ld 30, 32([[REG]])
; CHECK-DAG: ld 2, 24([[REG]])
; CHECK-DAG: mtctr [[REG2]]
; CHECK: bctr

return:                                           ; No predecessors!
  ret void
}

; Function Attrs: noreturn nounwind
declare void @llvm.eh.sjlj.longjmp(i8*) #1
