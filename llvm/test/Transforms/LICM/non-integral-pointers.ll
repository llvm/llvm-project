; RUN: opt -passes=licm -S < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-ni:1"

declare void @use_i64(i64 %0)
declare void @use_p1(ptr addrspace(1) %0)
declare i1 @cond()

define void @dont_hoist_ptrtoint(ptr addrspace(1) %p) {
; CHECK-LABEL: @dont_hoist_ptrtoint
; CHECK-LABEL: loop
; CHECK:         ptrtoint
entry:
  br label %loop

loop:
  %p.int = ptrtoint ptr addrspace(1) %p to i64
  call void @use_i64(i64 %p.int)
  br label %loop
}

define void @dont_hoist_inttoptr(i64 %p.int) {
; CHECK-LABEL: @dont_hoist_inttoptr
; CHECK-LABEL: loop
; CHECK:         inttoptr
entry:
  br label %loop

loop:
  %p = inttoptr i64 %p.int to ptr addrspace(1)
  call void @use_p1(ptr addrspace(1) %p)
  br label %loop
}

define i64 @dont_sink_ptrtoint(ptr addrspace(1) %p) {
; CHECK-LABEL: @dont_sink_ptrtoint
; CHECK-LABEL: loop
; CHECK:         ptrtoint
; CHECK-LABEL: exit
entry:
  br label %loop

loop:
  %p.int = ptrtoint ptr addrspace(1) %p to i64
  %c = call i1 @cond()
  br i1 %c, label %loop, label %exit

exit:
  ret i64 %p.int
}

define ptr addrspace(1) @dont_sink_inttoptr(i64 %p.int) {
; CHECK-LABEL: @dont_sink_inttoptr
; CHECK-LABEL: loop
; CHECK:         inttoptr
; CHECK-LABEL: exit
entry:
  br label %loop

loop:
  %p = inttoptr i64 %p.int to ptr addrspace(1)
  %c = call i1 @cond()
  br i1 %c, label %loop, label %exit

exit:
  ret ptr addrspace(1) %p
}
