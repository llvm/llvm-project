; RUN: opt < %s -passes="print<scalar-evolution>,loop-vectorize" --verify-scev -force-vector-width=4 -force-vector-interleave=1 -S | FileCheck %s

; CHECK-LABEL: @main(
; CHECK: vector.body
define i32 @main(i32 %.pre) {
entry:
  br label %for.body

for.body:
  %g.019 = phi i16 [ 0, %entry ], [ %dec7, %for.body ]
  %and = and i32 %.pre, 40
  %0 = sub i32 0, %and
  %dec7 = add i16 %g.019, 1
  %cmp.not = icmp eq i16 %dec7, 0
  br i1 %cmp.not, label %for.inc16, label %for.body

for.inc16:
  %1 = phi i32 [ %inc, %for.inc16 ], [ 0, %for.body ]
  %inc = add i32 %1, 1
  %add12 = add i32 %0, %1
  br label %for.inc16
}

; CHECK-LABEL: @pr66616(
; CHECK: vector.body
define void @pr66616(ptr %ptr) {
entry:
  br label %loop.1

loop.1:
  %iv.1 = phi i8 [ 0, %entry ], [ %inc, %loop.1 ]
  %load = load i32, ptr %ptr, align 4
  %add3 = add i32 %load, 1
  %inc = add i8 %iv.1, 1
  %cond1 = icmp eq i8 %inc, 0
  br i1 %cond1, label %preheader, label %loop.1

preheader:
  br label %loop.2

loop.2:
  %iv.2 = phi i32 [ %iv.2.i, %loop.2 ], [ %add3, %preheader ]
  %iv.3 = phi ptr [ %iv.3.i, %loop.2 ], [ %ptr, %preheader ]
  %iv.2.i = add i32 %iv.2, 1
  %iv.3.i = getelementptr i8, ptr %iv.3, i64 1
  %cond2 = icmp eq i32 %iv.2, 0
  br i1 %cond2, label %exit, label %loop.2

exit:
  ret void
}
