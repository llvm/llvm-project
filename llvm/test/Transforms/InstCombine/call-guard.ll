; RUN: opt < %s -instcombine -S | FileCheck %s

declare void @llvm.experimental.guard(i1, ...)

define void @test_guard_adjacent_same_cond(i1 %A) {
; CHECK-LABEL: @test_guard_adjacent_same_cond(
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %A) [ "deopt"() ]
; CHECK-NEXT:    ret void
  call void(i1, ...) @llvm.experimental.guard( i1 %A )[ "deopt"() ]
  call void(i1, ...) @llvm.experimental.guard( i1 %A )[ "deopt"() ]
  call void(i1, ...) @llvm.experimental.guard( i1 %A )[ "deopt"() ]
  call void(i1, ...) @llvm.experimental.guard( i1 %A )[ "deopt"() ]
  call void(i1, ...) @llvm.experimental.guard( i1 %A )[ "deopt"() ]
  call void(i1, ...) @llvm.experimental.guard( i1 %A )[ "deopt"() ]
  call void(i1, ...) @llvm.experimental.guard( i1 %A )[ "deopt"() ]
  call void(i1, ...) @llvm.experimental.guard( i1 %A )[ "deopt"() ]
  call void(i1, ...) @llvm.experimental.guard( i1 %A )[ "deopt"() ]
  call void(i1, ...) @llvm.experimental.guard( i1 %A )[ "deopt"() ]
  ret void
}

define void @test_guard_adjacent_diff_cond(i1 %A, i1 %B, i1 %C) {
; CHECK-LABEL: @test_guard_adjacent_diff_cond(
; CHECK-NEXT:    %1 = and i1 %A, %B
; CHECK-NEXT:    %2 = and i1 %1, %C
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %2, i32 123) [ "deopt"() ]
; CHECK-NEXT:    ret void
  call void(i1, ...) @llvm.experimental.guard( i1 %A, i32 123 )[ "deopt"() ]
  call void(i1, ...) @llvm.experimental.guard( i1 %B, i32 456 )[ "deopt"() ]
  call void(i1, ...) @llvm.experimental.guard( i1 %C, i32 789 )[ "deopt"() ]
  ret void
}
