; RUN: opt %loadPolly -polly-reschedule=0 -polly-pragma-based-opts=1 -polly-print-opt-isl -disable-output < %s | FileCheck %s --match-full-lines --check-prefix=ON
; RUN: opt %loadPolly -polly-reschedule=0 -polly-pragma-based-opts=0 -polly-print-opt-isl -disable-output < %s | FileCheck %s --match-full-lines --check-prefix=OFF
;
define void @func(i32 %n, ptr noalias nonnull %A, ptr noalias nonnull %B) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      store double 42.0, ptr %A
      %c = fadd double 21.0, 21.0
      store double %c, ptr %B
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
!5 = !{!"llvm.loop.distribute.enable"}


; ON: Printing analysis 'Polly - Optimize schedule of SCoP' for region: 'for => return' in function 'func':
; ON:      Calculated schedule:
; ON-NEXT: domain: "[n] -> { Stmt_body[i0] : 0 <= i0 < n; Stmt_body_b[i0] : 0 <= i0 < n }"
; ON-NEXT: child:
; ON-NEXT:   sequence:
; ON-NEXT:   - filter: "[n] -> { Stmt_body[i0] : 0 <= i0 < n }"
; ON-NEXT:     child:
; ON-NEXT:       schedule: "[n] -> [{ Stmt_body[i0] -> [(i0)] }]"
; ON-NEXT:   - filter: "[n] -> { Stmt_body_b[i0] : 0 <= i0 < n }"
; ON-NEXT:     child:
; ON-NEXT:       schedule: "[n] -> [{ Stmt_body_b[i0] -> [(i0)] }]"


; OFF-LABEL: Printing analysis 'Polly - Optimize schedule of SCoP' for region: 'for => return' in function 'func':
; OFF-NEXT:  Calculated schedule:
; OFF-NEXT:    n/a

