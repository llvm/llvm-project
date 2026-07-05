; RUN: opt -passes=jump-threading -S %s | FileCheck %s

; Check that we thread arg2neg -> checkpos -> end.
;
; LazyValueInfo would previously fail to analyze the value of %arg in arg2neg
; because its predecessing blocks (checkneg) hadn't been processed yet (PR21238)

; CHECK-LABEL: @test_jump_threading
; CHECK: arg2neg:
; CHECK-NEXT: br i1 %arg1, label %end, label %checkpos.thread
; CHECK: checkpos.thread:
; CHECK-NEXT: br label %end

define i32 @test_jump_threading(i1 %arg1, i32 %arg2) {
checkneg:
  %cmp = icmp slt i32 %arg2, 0
  br i1 %cmp, label %arg2neg, label %checkpos

arg2neg:
  br i1 %arg1, label %end, label %checkpos

checkpos:
  %cmp2 = icmp sgt i32 %arg2, 0
  br i1 %cmp2, label %arg2pos, label %end

arg2pos:
  br label %end

end:
  %0 = phi i32 [ 1, %arg2neg ], [ 2, %checkpos ], [ 3, %arg2pos ]
  ret i32 %0
}


; arg2neg has an edge back to itself. If LazyValueInfo is not careful when
; visiting predecessors, it could get into an infinite loop.

; CHECK-LABEL: test_infinite_loop

define i32 @test_infinite_loop(i1 %arg1, i32 %arg2) {
checkneg:
  %cmp = icmp slt i32 %arg2, 0
  br i1 %cmp, label %arg2neg, label %checkpos

arg2neg:
  br i1 %arg1, label %arg2neg, label %checkpos

checkpos:
  %cmp2 = icmp sgt i32 %arg2, 0
  br i1 %cmp2, label %arg2pos, label %end

arg2pos:
  br label %end

end:
  %0 = phi i32 [ 2, %checkpos ], [ 3, %arg2pos ]
  ret i32 %0
}
