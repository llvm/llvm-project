; RUN: opt %loadPolly -polly-print-opt-isl -disable-output < %s | FileCheck %s -match-full-lines
;
; Check that the disable_nonforced metadata is honored; optimization
; heuristics/rescheduling must not be applied.
;
define void @func(i32 %n, ptr noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %inner.for, label %exit


    inner.for:
      %i = phi i32 [0, %for], [%i.inc, %inner.inc]
      br label %bodyA


        bodyA:
          %mul = mul nuw nsw i32 %j, 128
          %add = add nuw nsw i32 %mul, %i
          %A_idx = getelementptr inbounds double, ptr %A, i32 %add
          store double 0.0, ptr %A_idx
          br label %inner.inc


    inner.inc:
      %i.inc = add nuw nsw i32 %i, 1
      %i.cmp = icmp slt i32 %i.inc, 128
      br i1 %i.cmp, label %inner.for, label %inner.exit

    inner.exit:
       br label %inc, !llvm.loop !2


inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for, !llvm.loop !2

exit:
  br label %return

return:
  ret void
}


!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.disable_nonforced"}


; n/a indicates no new schedule was computed
;
; CHECK-LABEL: Printing analysis 'Polly - Optimize schedule of SCoP' for region: 'for => return' in function 'func':
; CHECK-NEXT:  Calculated schedule:
; CHECK-NEXT:    n/a
