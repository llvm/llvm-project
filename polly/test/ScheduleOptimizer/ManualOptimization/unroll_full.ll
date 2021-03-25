; RUN: opt %loadPolly -polly-opt-isl -analyze < %s | FileCheck %s --match-full-lines
;
; Full unroll of a loop with 5 iterations.
;
define void @func(double* noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, 5
  br i1 %j.cmp, label %body, label %exit

    body:
      store double 42.0, double* %A
      br label %inc

inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for, !llvm.loop !2

exit:
  br label %return

return:
  ret void
}


!2 = distinct !{!2, !4, !5}
!4 = !{!"llvm.loop.unroll.enable", i1 true}
!5 = !{!"llvm.loop.unroll.full"}


; CHECK-LABEL: Printing analysis 'Polly - Optimize schedule of SCoP' for region: 'for => return' in function 'func':
; CHECK:       domain: "{ Stmt_body[i0] : 0 <= i0 <= 4 }"
; CHECK:         sequence:
; CHECK-NEXT:      - filter: "{ Stmt_body[0] }"
; CHECK-NEXT:      - filter: "{ Stmt_body[1] }"
; CHECK-NEXT:      - filter: "{ Stmt_body[2] }"
; CHECK-NEXT:      - filter: "{ Stmt_body[3] }"
; CHECK-NEXT:      - filter: "{ Stmt_body[4] }"
