; RUN: opt < %s -disable-output "-passes=print<da>" -da-enable-dependence-test=banerjee-miv 2>&1 \
; RUN:     | FileCheck %s

;
; for (int i = 0; i < 1; i++)
;   for (int j = 0; j < 1; j++) {
;     A[i + j] = 0;
;     A[i + j] = 1;
;   }
;
; A backedge-taken count is the maximum normalized iteration index. For a
; single-iteration loop it is 0, so the < and > direction domains are empty.
;
define void @banerjee_single_iteration(ptr %A) {
; CHECK-LABEL: 'banerjee_single_iteration'
; CHECK-NEXT:  Src: store i8 0, ptr %gep.0, align 1 --> Dst: store i8 0, ptr %gep.0, align 1
; CHECK-NEXT:    da analyze - none!
; CHECK-NEXT:  Src: store i8 0, ptr %gep.0, align 1 --> Dst: store i8 1, ptr %gep.1, align 1
; CHECK-NEXT:    da analyze - output [0 0|<]!
; CHECK-NEXT:  Src: store i8 1, ptr %gep.1, align 1 --> Dst: store i8 1, ptr %gep.1, align 1
; CHECK-NEXT:    da analyze - none!
;
entry:
  br label %loop.i

loop.i:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %loop.i.latch ]
  br label %loop.j

loop.j:
  %j = phi i64 [ 0, %loop.i ], [ %j.inc, %loop.j ]
  %offset = add nsw i64 %i, %j
  %gep.0 = getelementptr i8, ptr %A, i64 %offset
  store i8 0, ptr %gep.0, align 1
  %gep.1 = getelementptr i8, ptr %A, i64 %offset
  store i8 1, ptr %gep.1, align 1
  %j.inc = add i64 %j, 1
  %ec.j = icmp eq i64 %j.inc, 1
  br i1 %ec.j, label %loop.i.latch, label %loop.j

loop.i.latch:
  %i.inc = add i64 %i, 1
  %ec.i = icmp eq i64 %i.inc, 1
  br i1 %ec.i, label %exit, label %loop.i

exit:
  ret void
}
