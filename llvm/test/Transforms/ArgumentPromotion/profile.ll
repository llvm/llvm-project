; RUN: opt -argpromotion -mem2reg -S < %s | FileCheck %s
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

; Checks if !prof metadata is corret in deadargelim.

define void @caller() #0 {
  %x = alloca i32
  store i32 42, i32* %x
  call void @promote_i32_ptr(i32* %x), !prof !0
; CHECK: call void @promote_i32_ptr(i32 42), !prof ![[PROF:[0-9]]]
  ret void
}

define internal void @promote_i32_ptr(i32* %xp) {
  %x = load i32, i32* %xp
  call void @use_i32(i32 %x)
  ret void
}

declare void @use_i32(i32)

; CHECK: ![[PROF]] = !{!"branch_weights", i32 30}
!0 = !{!"branch_weights", i32 30}
