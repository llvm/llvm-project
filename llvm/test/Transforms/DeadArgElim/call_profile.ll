; RUN: opt -passes=deadargelim -S < %s | FileCheck %s

; Checks if !prof metadata is corret in deadargelim.

define void @caller() #0 {
; CHECK: call void @test_vararg.argelim(), !prof ![[PROF:[0-9]]]
; CHECK: call void @test.argelim(), !prof ![[PROF]]
  call void (i32, ...) @test_vararg(i32 1), !prof !0
  call void @test(i32 1), !prof !0
  ret void
}

define internal void @test_vararg(i32, ...) #1 {
  ret void
}

define internal void @test(i32 %a) #1 {
  ret void
}

; CHECK:![[PROF]] = !{!"branch_weights", i32 30}
!0 = !{!"branch_weights", i32 30}
