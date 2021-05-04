; RUN: opt %loadPolly -polly-opt-isl -polly-pragma-based-opts=1 -analyze < %s | FileCheck %s --match-full-lines
;
; Override unroll metadata with llvm.loop.unroll.disable.
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


!2 = distinct !{!2, !3, !4}
!3 = !{!"llvm.loop.unroll.count", i32 4}
!4 = !{!"llvm.loop.unroll.disable"}


; CHECK-LABEL: Printing analysis 'Polly - Optimize schedule of SCoP' for region: 'for => return' in function 'func':
; CHECK-NEXT:  Calculated schedule:
; CHECK-NEXT:    n/a
