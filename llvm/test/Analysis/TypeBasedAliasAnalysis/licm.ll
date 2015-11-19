; RUN: opt -tbaa -licm -S < %s | FileCheck %s

; LICM should be able to hoist the address load out of the loop
; by using TBAA information.

; CHECK: @foo
; CHECK:      entry:
; CHECK-NEXT:   %tmp3 = load double*, double** @P
; CHECK-NEXT:   br label %for.body

@P = common global double* null

define void @foo(i64 %n) nounwind {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.07 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %tmp3 = load double*, double** @P, !tbaa !1
  %scevgep = getelementptr double, double* %tmp3, i64 %i.07
  %tmp4 = load double, double* %scevgep, !tbaa !2
  %mul = fmul double %tmp4, 2.300000e+00
  store double %mul, double* %scevgep, !tbaa !2
  %inc = add i64 %i.07, 1
  %exitcond = icmp eq i64 %inc, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

!0 = !{!"root", null}
!1 = !{!6, !6, i64 0}
!2 = !{!7, !7, i64 0}

; LICM shouldn't hoist anything here.

; CHECK: @bar
; CHECK: loop:
; CHECK: load
; CHECK: store
; CHECK: load
; CHECK: store
; CHECK: br label %loop

define void @bar(i8** %p) nounwind {
entry:
  %q = bitcast i8** %p to i8*
  br label %loop

loop:
  %tmp51 = load i8*, i8** %p, !tbaa !4
  store i8* %tmp51, i8** %p
  %tmp40 = load i8, i8* %q, !tbaa !5
  store i8 %tmp40, i8* %q
  br label %loop
}

!3 = !{!"pointer", !8}
!4 = !{!8, !8, i64 0}
!5 = !{!9, !9, i64 0}
!6 = !{!"pointer", !0}
!7 = !{!"double", !0}
!8 = !{!"char", !9}
!9 = !{!"root", null}
