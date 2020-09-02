; RUN: opt -globalopt < %s -S -o - | FileCheck %s

; When simplifying users of a global variable, the pass could incorrectly
; return false if there were still some uses left, and no further optimizations
; was done. This was caught by the pass return status check that is hidden
; under EXPENSIVE_CHECKS.

@GV1 = internal unnamed_addr global i64 1, align 8

; CHECK: @GV1 = internal unnamed_addr global i64 1, align 8

define void @test1() local_unnamed_addr {
; CHECK-LABEL: @test1
; CHECK-NEXT: %val = load atomic i8
; CHECK-NEXT: ret void

  %val = load atomic i8, i8* bitcast (i64* @GV1 to i8*) acquire, align 8
  ret void
}

define i64 @test2() local_unnamed_addr {
; CHECK-LABEL: @test2
; CHECK-NEXT: ret i64 1

  %val = load atomic i64, i64* @GV1 acquire, align 8
  ret i64 %val
}
