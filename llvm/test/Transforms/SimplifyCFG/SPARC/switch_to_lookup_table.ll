; RUN: opt < %s -simplifycfg -S -mtriple=sparc-unknown-unknown | FileCheck %s

; Check that switches are not turned into lookup tables, as this is not
; considered profitable on the target.

define i32 @f(i32 %c) nounwind uwtable readnone {
entry:
  switch i32 %c, label %sw.default [
    i32 42, label %return
    i32 43, label %sw.bb1
    i32 44, label %sw.bb2
    i32 45, label %sw.bb3
    i32 46, label %sw.bb4
    i32 47, label %sw.bb5
    i32 48, label %sw.bb6
  ]

sw.bb1: br label %return
sw.bb2: br label %return
sw.bb3: br label %return
sw.bb4: br label %return
sw.bb5: br label %return
sw.bb6: br label %return
sw.default: br label %return
return:
  %retval.0 = phi i32 [ 15, %sw.default ], [ 1, %sw.bb6 ], [ 62, %sw.bb5 ], [ 27, %sw.bb4 ], [ -1, %sw.bb3 ], [ 0, %sw.bb2 ], [ 123, %sw.bb1 ], [ 55, %entry ]
  ret i32 %retval.0

; CHECK-LABEL: @f(
; CHECK-NOT: getelementptr
; CHECK: switch i32 %c
}
