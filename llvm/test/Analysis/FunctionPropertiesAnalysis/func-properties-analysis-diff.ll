; REQUIRES: asserts
; RUN: opt -stats -enable-detailed-function-properties -O3 -disable-output < %s 2>&1 | FileCheck %s

; CHECK-DAG: 4 func-properties-stats - Number of basic blocks (before optimizations)
; CHECK-DAG: 5 func-properties-stats - Number of instructions (of all types) (before optimizations)
; CHECK-DAG: 4 func-properties-stats - Number of basic block successors (before optimizations)
; CHECK-DAG: 1 func-properties-stats - Number of basic blocks (after optimizations)
; CHECK-DAG: 1 func-properties-stats - Number of instructions (of all types) (after optimizations)
; CHECK-NOT: func-properties-stats - Number of basic block successors (after optimizations)

define i32 @test_cfg() {
entry:
  ; This branch is trivially resolvable
  br i1 true, label %then, label %else

then:
  br label %end

else:
  br label %end

end:
  %phi = phi i32 [ 1, %then ], [ 2, %else ]
  ret i32 %phi
}