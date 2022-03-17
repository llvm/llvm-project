; RUN: opt %loadPolly -polly-print-opt-isl -disable-output < %s | FileCheck %s --match-full-lines
;
; Apply two loop transformations. First partial, then full unrolling.
;
define void @func(double* noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, 12
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


!2 = distinct !{!2, !4, !5, !6}
!4 = !{!"llvm.loop.unroll.enable", i1 true}
!5 = !{!"llvm.loop.unroll.count", i4 4}
!6 = !{!"llvm.loop.unroll.followup_unrolled", !7}

!7 = distinct !{!7, !8, !9}
!8 = !{!"llvm.loop.unroll.enable", i1 true}
!9 = !{!"llvm.loop.unroll.full"}


; CHECK-LABEL: Printing analysis 'Polly - Optimize schedule of SCoP' for region: 'for => return' in function 'func':
; CHECK: domain: "{ Stmt_body[i0] : 0 <= i0 <= 11 }"
; CHECK    sequence:
; CHECK:   - filter: "{ Stmt_body[i0] : 0 <= i0 <= 3 }"
; CHECK        sequence:
; CHECK:       - filter: "{ Stmt_body[0] }"
; CHECK:       - filter: "{ Stmt_body[i0] : (-1 + i0) mod 4 = 0 }"
; CHECK:       - filter: "{ Stmt_body[i0] : (2 + i0) mod 4 = 0 }"
; CHECK:       - filter: "{ Stmt_body[i0] : (1 + i0) mod 4 = 0 }"
; CHECK    sequence:
; CHECK:   - filter: "{ Stmt_body[i0] : 4 <= i0 <= 7 }"
; CHECK        sequence:
; CHECK:       - filter: "{ Stmt_body[4] }"
; CHECK:       - filter: "{ Stmt_body[i0] : (-1 + i0) mod 4 = 0 }"
; CHECK:       - filter: "{ Stmt_body[i0] : (2 + i0) mod 4 = 0 }"
; CHECK:       - filter: "{ Stmt_body[i0] : (1 + i0) mod 4 = 0 }"
; CHECK    sequence:
; CHECK:   - filter: "{ Stmt_body[i0] : 8 <= i0 <= 11 }"
; CHECK        sequence:
; CHECK:       - filter: "{ Stmt_body[8] }"
; CHECK:       - filter: "{ Stmt_body[i0] : (-1 + i0) mod 4 = 0 }"
; CHECK:       - filter: "{ Stmt_body[i0] : (2 + i0) mod 4 = 0 }"
; CHECK:       - filter: "{ Stmt_body[i0] : (1 + i0) mod 4 = 0 }"
