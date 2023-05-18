; RUN: opt < %s -passes=loop-unroll,instcombine -S | FileCheck %s

; This test is a worst-case scenario for bitreversal/byteswap detection.
; After loop unrolling (the unrolled loop is unreadably large so it has been kept
; rolled here), we have a binary tree of OR operands (as bitreversal detection
; looks straight through shifts):
;
;  OR
;  | \
;  |  LSHR
;  | /
;  OR
;  | \
;  |  LSHR
;  | /
;  OR
;
; This results in exponential runtime. The loop here is 32 iterations which will
; totally hang if we don't deal with this case cleverly.

@b = common global i32 0, align 4

; CHECK: define i32 @fn1
define i32 @fn1() #0 {
entry:
  %b.promoted = load i32, ptr @b, align 4, !tbaa !2
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %or4 = phi i32 [ %b.promoted, %entry ], [ %or, %for.body ]
  %i.03 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %shr = lshr i32 %or4, 1
  %or = or i32 %shr, %or4
  %inc = add nuw nsw i32 %i.03, 1
  %exitcond = icmp eq i32 %inc, 32
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  store i32 %or, ptr @b, align 4, !tbaa !2
  ret i32 undef
}

attributes #0 = { norecurse nounwind ssp uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="core2" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+ssse3" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 3.8.0"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
