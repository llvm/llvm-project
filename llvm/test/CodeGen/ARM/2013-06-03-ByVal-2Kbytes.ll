; RUN: llc < %s -mcpu=cortex-a15 | FileCheck %s
; RUN: llc < %s -mcpu=cortex-a15 -stop-after=finalize-isel -o %t.mir
; RUN: llc %t.mir -mcpu=cortex-a15 -start-after=finalize-isel -o - | FileCheck %s
; ModuleID = 'attri_16.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32-S64"
target triple = "armv4t--linux-gnueabihf"

%big_struct0 = type { [517 x i32] }
%big_struct1 = type { [516 x i32] }

;CHECK-LABEL: f:
define void @f(ptr %p0, ptr %p1) {

;CHECK: sub sp, sp, #8
;CHECK: sub sp, sp, #2048
;CHECK: bl callme0
  call void @callme0(ptr byval(%big_struct0) %p0)

;CHECK: add sp, sp, #8
;CHECK: add sp, sp, #2048
;CHECK: sub sp, sp, #2048
;CHECK: bl callme1
  call void @callme1(ptr byval(%big_struct1) %p1)

;CHECK: add sp, sp, #2048

  ret void
}

declare void @callme0(ptr byval(%big_struct0))
declare void @callme1(ptr byval(%big_struct1))

