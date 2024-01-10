; RUN: opt < %s -passes="require<scalar-evolution>,print<scalar-evolution>,loop-vectorize" --verify-scev -force-vector-interleave=2 -force-vector-width=8 -S | FileCheck %s

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
