; REQUIRES: asserts

; Test debug counter with multiple intervals
; RUN: opt -passes=dce -S -debug-counter=dce-transform=0 < %s | FileCheck %s --check-prefix=CHECK-ZERO
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
; RUN: not opt -passes=dce -S -debug-counter=dce-transform=abc-def 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR-NON-NUMERIC
; RUN: not opt -passes=dce -S -debug-counter=dce-transform=1-abc 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR-MIXED
; RUN: not opt -passes=dce -S -debug-counter=dce-transform=1:2:3:2:4 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR-COMPLEX-UNORDERED
; RUN: not opt -passes=dce -S -debug-counter=dce-transform=1--5 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR-DOUBLE-DASH
; RUN: not opt -passes=dce -S -debug-counter=dce-transform=-5 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR-NEGATIVE
; RUN: not opt -passes=dce -S -debug-counter=dce-transform= 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR-EMPTY
; RUN: not opt -passes=dce -S -debug-counter=dce-transform=1:1:1 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR-DUPLICATE

; Test that with debug counters on, we can selectively apply transformations
; using different interval specifications. Also check that we catch errors during parsing.

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

; Test zero: eliminate transformation 0
; CHECK-ZERO-LABEL: @test  
; CHECK-ZERO-NEXT: %dead2 = add i32 3, 4
; CHECK-ZERO-NEXT: %dead3 = add i32 5, 6
; CHECK-ZERO-NEXT: %dead4 = add i32 7, 8
; CHECK-ZERO-NEXT: %dead5 = add i32 9, 10
; CHECK-ZERO-NEXT: %dead6 = add i32 11, 12
; CHECK-ZERO-NEXT: %dead7 = add i32 13, 14
; CHECK-ZERO-NEXT: %dead8 = add i32 15, 16
; CHECK-ZERO-NEXT: ret void

; Test single values: apply transformations 1, 3, 5 (eliminate dead2, dead4, dead6)
; CHECK-SINGLE-LABEL: @test
; CHECK-SINGLE-NEXT: %dead1 = add i32 1, 2
; CHECK-SINGLE-NEXT: %dead3 = add i32 5, 6
; CHECK-SINGLE-NEXT: %dead5 = add i32 9, 10
; CHECK-SINGLE-NEXT: %dead7 = add i32 13, 14
; CHECK-SINGLE-NEXT: %dead8 = add i32 15, 16
; CHECK-SINGLE-NEXT: ret void

; Test mixed intervals: apply transformations 1-2, 4, 6-7 (eliminate dead2, dead3, dead5, dead7, dead8)
; CHECK-MIXED-LABEL: @test
; CHECK-MIXED-NEXT: %dead1 = add i32 1, 2
; CHECK-MIXED-NEXT: %dead4 = add i32 7, 8
; CHECK-MIXED-NEXT: %dead6 = add i32 11, 12
; CHECK-MIXED-NEXT: ret void

; Test all interval: apply transformations 1-7 (eliminate all dead instructions except dead1)
; CHECK-ALL-LABEL: @test
; CHECK-ALL-NEXT: %dead1 = add i32 1, 2
; CHECK-ALL-NEXT: ret void

; Test out of interval: apply transformation 100 (eliminate nothing, counter too high)
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

; Error case checks - test comprehensive error handling
; CHECK-ERROR-INVALID: DebugCounter Error: Invalid interval format: 'invalid'
; CHECK-ERROR-BACKWARDS: DebugCounter Error: Invalid interval: 5 >= 2
; CHECK-ERROR-UNORDERED: DebugCounter Error: Expected intervals to be in increasing order: 2 <= 3
; CHECK-ERROR-NON-NUMERIC: DebugCounter Error: Invalid interval format: 'abc-def'
; CHECK-ERROR-MIXED: DebugCounter Error: Invalid interval format: '1-abc'
; CHECK-ERROR-COMPLEX-UNORDERED: DebugCounter Error: Expected intervals to be in increasing order: 2 <= 3
; CHECK-ERROR-DOUBLE-DASH: DebugCounter Error: Invalid interval format: '1--5'
; CHECK-ERROR-NEGATIVE: DebugCounter Error: Invalid interval format: '-5'
; CHECK-ERROR-EMPTY: DebugCounter Error: dce-transform= does not have an = in it
; CHECK-ERROR-DUPLICATE: DebugCounter Error: Expected intervals to be in increasing order: 1 <= 1
