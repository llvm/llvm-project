; RUN: opt < %s -passes='print<phi-values>' -disable-output 2>&1 | FileCheck %s

; This test has a phi with a large number of incoming values that are all the
; same phi, and that phi depends on this phi. This is to check that phi values
; analysis doesn't repeatedly add a phis values to itself until it segfaults.

; CHECK-LABEL: PHI Values for function: fn
define void @fn(i8* %arg) {
entry:
  br label %for.body

for.body:
; CHECK: PHI %phi1 has values:
; CHECK-DAG: i8* %arg
; CHECK-DAG: i8* undef
  %phi1 = phi i8* [ %arg, %entry ], [ %phi2, %end ]
  switch i32 undef, label %end [
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
    i32 4, label %bb4
    i32 5, label %bb5
    i32 6, label %bb6
    i32 7, label %bb7
    i32 8, label %bb8
    i32 9, label %bb9
    i32 10, label %bb10
    i32 11, label %bb11
    i32 12, label %bb12
    i32 13, label %bb13
  ]

bb1:
  br label %end

bb2:
  br label %end

bb3:
  br label %end

bb4:
  br label %end

bb5:
  br label %end

bb6:
  br label %end

bb7:
  br label %end

bb8:
  br label %end

bb9:
  br label %end

bb10:
  br label %end

bb11:
  br label %end

bb12:
  br label %end

bb13:
  br label %end

end:
; CHECK: PHI %phi2 has values:
; CHECK-DAG: i8* %arg
; CHECK-DAG: i8* undef
  %phi2 = phi i8* [ %phi1, %for.body ], [ %phi1, %bb1 ], [ %phi1, %bb2 ], [ %phi1, %bb3 ], [ %phi1, %bb4 ], [ %phi1, %bb5 ], [ %phi1, %bb6 ], [ %phi1, %bb7 ], [ undef, %bb8 ], [ %phi1, %bb9 ], [ %phi1, %bb10 ], [ %phi1, %bb11 ], [ %phi1, %bb12 ], [ %phi1, %bb13 ]
  br label %for.body
}
