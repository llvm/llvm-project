; RUN: opt < %s -instcombine -instcombine-infinite-loop-threshold=2 -S | FileCheck %s
; RUN: opt < %s -instcombine -S -debugify-each | FileCheck %s
; RUN: opt < %s -passes=instcombine -S -debugify-each | FileCheck %s

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

; This version tests for the common form where the conditions are
; between the guards
define void @test_guard_adjacent_diff_cond2(i32 %V1, i32 %V2) {
; CHECK-LABEL: @test_guard_adjacent_diff_cond2(
; CHECK-NEXT:    %1 = and i32 %V1, %V2
; CHECK-NEXT:    %2 = icmp slt i32 %1, 0
; CHECK-NEXT:    %and = and i32 %V1, 255
; CHECK-NEXT:    %C = icmp ult i32 %and, 129
; CHECK-NEXT:    %3 = and i1 %2, %C
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %3, i32 123) [ "deopt"() ]
; CHECK-NEXT:    ret void
  %A = icmp slt i32 %V1, 0
  call void(i1, ...) @llvm.experimental.guard( i1 %A, i32 123 )[ "deopt"() ]
  %B = icmp slt i32 %V2, 0
  call void(i1, ...) @llvm.experimental.guard( i1 %B, i32 456 )[ "deopt"() ]
  %and = and i32 %V1, 255
  %C = icmp sle i32 %and, 128
  call void(i1, ...) @llvm.experimental.guard( i1 %C, i32 789 )[ "deopt"() ]
  ret void
}

; Might not be legal to hoist the load above the first guard since the
; guard might control dereferenceability
define void @negative_load(i32 %V1, i32* %P) {
; CHECK-LABEL: @negative_load
; CHECK:    @llvm.experimental.guard
; CHECK:    @llvm.experimental.guard
  %A = icmp slt i32 %V1, 0
  call void(i1, ...) @llvm.experimental.guard( i1 %A, i32 123 )[ "deopt"() ]
  %V2 = load i32, i32* %P
  %B = icmp slt i32 %V2, 0
  call void(i1, ...) @llvm.experimental.guard( i1 %B, i32 456 )[ "deopt"() ]
  ret void
}

define void @deref_load(i32 %V1, i32* dereferenceable(4) align 4 %P) {
; CHECK-LABEL: @deref_load
; CHECK-NEXT:  %V2 = load i32, i32* %P, align 4
; CHECK-NEXT:  %1 = and i32 %V2, %V1
; CHECK-NEXT:  %2 = icmp slt i32 %1, 0
; CHECK-NEXT:  call void (i1, ...) @llvm.experimental.guard(i1 %2, i32 123) [ "deopt"() ]
  %A = icmp slt i32 %V1, 0
  call void(i1, ...) @llvm.experimental.guard( i1 %A, i32 123 )[ "deopt"() ]
  %V2 = load i32, i32* %P
  %B = icmp slt i32 %V2, 0
  call void(i1, ...) @llvm.experimental.guard( i1 %B, i32 456 )[ "deopt"() ]
  ret void
}

; The divide might fault above the guard
define void @negative_div(i32 %V1, i32 %D) {
; CHECK-LABEL: @negative_div
; CHECK:    @llvm.experimental.guard
; CHECK:    @llvm.experimental.guard
  %A = icmp slt i32 %V1, 0
  call void(i1, ...) @llvm.experimental.guard( i1 %A, i32 123 )[ "deopt"() ]
  %V2 = udiv i32 %V1, %D 
  %B = icmp slt i32 %V2, 0
  call void(i1, ...) @llvm.experimental.guard( i1 %B, i32 456 )[ "deopt"() ]
  ret void
}

; Highlight the limit of the window in a case which would otherwise be mergable
define void @negative_window(i32 %V1, i32 %a, i32 %b, i32 %c, i32 %d) {
; CHECK-LABEL: @negative_window
; CHECK:    @llvm.experimental.guard
; CHECK:    @llvm.experimental.guard
  %A = icmp slt i32 %V1, 0
  call void(i1, ...) @llvm.experimental.guard( i1 %A, i32 123 )[ "deopt"() ]
  %V2 = add i32 %a, %b
  %V3 = add i32 %V2, %c
  %V4 = add i32 %V3, %d
  %B = icmp slt i32 %V4, 0
  call void(i1, ...) @llvm.experimental.guard( i1 %B, i32 456 )[ "deopt"() ]
  ret void
}

