; Check that llc does not abort, which happened due to incorrect MIR.
; RUN: llc -O2 -mtriple=hexagon -insert-max-ifmap=1 < %s
; RUN: llc -O2 -mtriple=hexagon -insert-max-ifmap=2 < %s
; RUN: llc -O2 -mtriple=hexagon -insert-max-ifmap=3 < %s
; RUN: llc -O2 -mtriple=hexagon -insert-max-ifmap=4 < %s
; RUN: llc -O2 -mtriple=hexagon -insert-max-ifmap=5 < %s

; Look for this symptom, in case llc does not check invalid IR.
; CHECK-NOT: insert(%14,%5,#5,#5)

; RUN: llc -O2 -mtriple=hexagon -insert-max-ifmap=1 -debug-only=hexinsert -stop-after hexinsert < %s 2>&1 | FileCheck %s
; RUN: llc -O2 -mtriple=hexagon -insert-max-ifmap=2 -debug-only=hexinsert -stop-after hexinsert < %s 2>&1 | FileCheck %s
; RUN: llc -O2 -mtriple=hexagon -insert-max-ifmap=3 -debug-only=hexinsert -stop-after hexinsert < %s 2>&1 | FileCheck %s
; RUN: llc -O2 -mtriple=hexagon -insert-max-ifmap=4 -debug-only=hexinsert -stop-after hexinsert < %s 2>&1 | FileCheck %s
; RUN: llc -O2 -mtriple=hexagon -insert-max-ifmap=5 -debug-only=hexinsert -stop-after hexinsert < %s 2>&1 | FileCheck %s

; REQUIRES: asserts

define i32 @f(i32 %0, i32 %1, i32 %2) {
entry:
  switch i32 %0, label %common.ret1 [
    i32 8907, label %3
    i32 4115, label %6
  ]

common.ret1:
  %common.ret1.op = phi i32 [ %5, %3 ], [ %526, %6 ], [ 0, %entry ]
  ret i32 %common.ret1.op

3:
  %4 = shl i32 %2, 5
  %5 = and i32 %4, 992
  br label %common.ret1

6:
  %7 = shl i32 %0, 10
  %8 = and i32 %7, 7168
  %9 = shl i32 %0, 5
  %10 = and i32 %9, 992
  %11 = or i32 %10, %8
  %12 = and i32 %0, 1
  %13 = or i32 %11, %12
  %14 = shl i32 %1, 1
  %15 = and i32 %14, 2031616
  %526 = or i32 %13, %15
  br label %common.ret1
}
