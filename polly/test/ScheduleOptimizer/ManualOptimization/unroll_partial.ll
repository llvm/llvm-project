; RUN: opt %loadPolly -polly-opt-isl -polly-pragma-based-opts=1 -analyze < %s | FileCheck %s --match-full-lines
; RUN: opt %loadPolly -polly-opt-isl -polly-pragma-based-opts=0 -analyze < %s | FileCheck %s --check-prefix=OFF --match-full-lines
;
; Partial unroll by a factor of 4.
;
define void @func(i32 %n, double* noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
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


!2 = distinct !{!2, !5}
!5 = !{!"llvm.loop.unroll.count", i4 4}


; CHECK-LABEL: Printing analysis 'Polly - Optimize schedule of SCoP' for region: 'for => return' in function 'func':
; CHECK:       domain: "[n] -> { Stmt_body[i0] : 0 <= i0 < n }"
; CHECK:         schedule: "[n] -> [{ Stmt_body[i0] -> [(i0 - (i0) mod 4)] }]"
; CHECK:           sequence:
; CHECK-NEXT:      - filter: "[n] -> { Stmt_body[i0] : (i0) mod 4 = 0 }"
; CHECK-NEXT:      - filter: "[n] -> { Stmt_body[i0] : (-1 + i0) mod 4 = 0 }"
; CHECK-NEXT:      - filter: "[n] -> { Stmt_body[i0] : (2 + i0) mod 4 = 0 }"
; CHECK-NEXT:      - filter: "[n] -> { Stmt_body[i0] : (1 + i0) mod 4 = 0 }"


; OFF-LABEL: Printing analysis 'Polly - Optimize schedule of SCoP' for region: 'for => return' in function 'func':
; OFF-NEXT:  Calculated schedule:
; OFF-NEXT:    n/a
