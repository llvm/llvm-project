; Check that zeroed branch weights do not crash or otherwise break basic
; LoopUnroll behavior when it tries to compute a probability from them.

; RUN: opt < %s -S -unroll-count=2 -passes='loop-unroll' 2>&1 | FileCheck %s

define void @test() {
entry:
  br label %loop

loop:
  br i1 false, label %end, label %loop, !prof !0

end:
  ret void
}

!0 = !{!"branch_weights", i32 0, i32 0}

; CHECK: define void @test() {
; CHECK: entry:
; CHECK:   br label %loop
; CHECK: loop:
; CHECK:   br i1 false, label %end, label %loop.1, !prof !0
; CHECK: loop.1:
; CHECK:   br i1 false, label %end, label %loop, !prof !0, !llvm.loop !1
; CHECK-NOT: loop.2
; CHECK: end:
; CHECK:   ret void
; CHECK: }
; CHECK: !0 = !{!"branch_weights", i32 0, i32 0}
