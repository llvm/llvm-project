; REQUIRES: asserts
; Test debug counter with multiple ranges using colon separators
; (DebugCounter uses colon separators to avoid conflicts with cl::CommaSeparated)

; RUN: opt -passes=dce -S -debug-counter=dce-transform=1:3:5 < %s | FileCheck %s --check-prefix=CHECK-COLON
; RUN: opt -passes=dce -S -debug-counter=dce-transform=1-2:4:6-7 < %s | FileCheck %s --check-prefix=CHECK-MIXED-COLON

; Test that with debug counters on, we can selectively apply transformations
; using different range syntaxes. All variants should produce the same result.

; Original function has 8 dead instructions that DCE can eliminate
define void @test() {
  %dead1 = add i32 1, 2
  %dead2 = add i32 3, 4  
  %dead3 = add i32 5, 6
  %dead4 = add i32 7, 8
  %dead5 = add i32 9, 10
  %dead6 = add i32 11, 12
  %dead7 = add i32 13, 14
  %dead8 = add i32 15, 16
  ret void
}

; Test colon separator: apply transformations 1, 3, 5 (eliminate dead2, dead4, dead6)
; CHECK-COLON-LABEL: @test
; CHECK-COLON-NEXT: %dead1 = add i32 1, 2
; CHECK-COLON-NEXT: %dead3 = add i32 5, 6
; CHECK-COLON-NEXT: %dead5 = add i32 9, 10
; CHECK-COLON-NEXT: %dead7 = add i32 13, 14
; CHECK-COLON-NEXT: %dead8 = add i32 15, 16
; CHECK-COLON-NEXT: ret void

; Test mixed ranges with colon: apply transformations 1-2, 4, 6-7 (eliminate dead2, dead3, dead5, dead7, dead8)
; CHECK-MIXED-COLON-LABEL: @test
; CHECK-MIXED-COLON-NEXT: %dead1 = add i32 1, 2
; CHECK-MIXED-COLON-NEXT: %dead4 = add i32 7, 8
; CHECK-MIXED-COLON-NEXT: %dead6 = add i32 11, 12
; CHECK-MIXED-COLON-NEXT: ret void
