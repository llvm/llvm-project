; REQUIRES: asserts
; Test debug counter with multiple ranges

; RUN: opt -passes=dce -S -debug-counter=dce-transform=1:3:5 < %s | FileCheck %s --check-prefix=CHECK-SINGLE
; RUN: opt -passes=dce -S -debug-counter=dce-transform=1-2:4:6-7 < %s | FileCheck %s --check-prefix=CHECK-MIXED
; RUN: opt -passes=dce -S -debug-counter=dce-transform=1-7 < %s | FileCheck %s --check-prefix=CHECK-ALL
; RUN: opt -passes=dce -S -debug-counter=dce-transform=100 < %s | FileCheck %s --check-prefix=CHECK-NONE
; RUN: opt -passes=dce -S -debug-counter=dce-transform=7 < %s | FileCheck %s --check-prefix=CHECK-LAST
; RUN: opt -passes=dce -S -debug-counter=dce-transform=1 < %s | FileCheck %s --check-prefix=CHECK-FIRST

; Test error cases - these should produce error messages but not crash
; RUN: not opt -passes=dce -S -debug-counter=dce-transform=invalid 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR-INVALID
; RUN: not opt -passes=dce -S -debug-counter=dce-transform=5-2 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR-BACKWARDS
; RUN: not opt -passes=dce -S -debug-counter=dce-transform=1:3:2 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR-UNORDERED

; Test that with debug counters on, we can selectively apply transformations
; using different range specifications and edge cases.

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

; Test single values: apply transformations 1, 3, 5 (eliminate dead2, dead4, dead6)
; CHECK-SINGLE-LABEL: @test
; CHECK-SINGLE-NEXT: %dead1 = add i32 1, 2
; CHECK-SINGLE-NEXT: %dead3 = add i32 5, 6
; CHECK-SINGLE-NEXT: %dead5 = add i32 9, 10
; CHECK-SINGLE-NEXT: %dead7 = add i32 13, 14
; CHECK-SINGLE-NEXT: %dead8 = add i32 15, 16
; CHECK-SINGLE-NEXT: ret void

; Test mixed ranges: apply transformations 1-2, 4, 6-7 (eliminate dead2, dead3, dead5, dead7, dead8)
; CHECK-MIXED-LABEL: @test
; CHECK-MIXED-NEXT: %dead1 = add i32 1, 2
; CHECK-MIXED-NEXT: %dead4 = add i32 7, 8
; CHECK-MIXED-NEXT: %dead6 = add i32 11, 12
; CHECK-MIXED-NEXT: ret void

; Test all range: apply transformations 1-7 (eliminate all dead instructions except dead1)
; CHECK-ALL-LABEL: @test
; CHECK-ALL-NEXT: %dead1 = add i32 1, 2
; CHECK-ALL-NEXT: ret void

; Test out of range: apply transformation 100 (eliminate nothing, counter too high)
; CHECK-NONE-LABEL: @test
; CHECK-NONE-NEXT: %dead1 = add i32 1, 2
; CHECK-NONE-NEXT: %dead2 = add i32 3, 4
; CHECK-NONE-NEXT: %dead3 = add i32 5, 6
; CHECK-NONE-NEXT: %dead4 = add i32 7, 8
; CHECK-NONE-NEXT: %dead5 = add i32 9, 10
; CHECK-NONE-NEXT: %dead6 = add i32 11, 12
; CHECK-NONE-NEXT: %dead7 = add i32 13, 14
; CHECK-NONE-NEXT: %dead8 = add i32 15, 16
; CHECK-NONE-NEXT: ret void

; Test last transformation: apply transformation 7 (eliminate dead8)
; CHECK-LAST-LABEL: @test
; CHECK-LAST-NEXT: %dead1 = add i32 1, 2
; CHECK-LAST-NEXT: %dead2 = add i32 3, 4
; CHECK-LAST-NEXT: %dead3 = add i32 5, 6
; CHECK-LAST-NEXT: %dead4 = add i32 7, 8
; CHECK-LAST-NEXT: %dead5 = add i32 9, 10
; CHECK-LAST-NEXT: %dead6 = add i32 11, 12
; CHECK-LAST-NEXT: %dead7 = add i32 13, 14
; CHECK-LAST-NEXT: ret void

; Test first transformation: apply transformation 1 (eliminate dead2)
; CHECK-FIRST-LABEL: @test
; CHECK-FIRST-NEXT: %dead1 = add i32 1, 2
; CHECK-FIRST-NEXT: %dead3 = add i32 5, 6
; CHECK-FIRST-NEXT: %dead4 = add i32 7, 8
; CHECK-FIRST-NEXT: %dead5 = add i32 9, 10
; CHECK-FIRST-NEXT: %dead6 = add i32 11, 12
; CHECK-FIRST-NEXT: %dead7 = add i32 13, 14
; CHECK-FIRST-NEXT: %dead8 = add i32 15, 16
; CHECK-FIRST-NEXT: ret void

; Error case checks
; CHECK-ERROR-INVALID: Invalid range format: 'invalid'
; CHECK-ERROR-BACKWARDS: Invalid range: 5 >= 2
; CHECK-ERROR-UNORDERED: Expected ranges to be in increasing order: 2 <= 3
