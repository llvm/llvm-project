; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s

define i32 @foo(i64* nocapture %perm, i32 %n) {
entry:
  br label %body

body:
; CHECK-LABEL: foo:
; CHECK: pslldq  $8, %xmm0
  %vec.ind = phi <2 x i64> [ <i64 0, i64 1>, %entry ], [ <i64 2, i64 3>, %body ]
  %l13 = extractelement <2 x i64> %vec.ind, i32 %n
  %l14 = getelementptr inbounds i64, i64* %perm, i64 %l13
  %l15 = bitcast i64* %l14 to <2 x i64>*
  store <2 x i64> %vec.ind, <2 x i64>* %l15, align 8
  %niter.ncmp.3 = icmp eq i64 %l13, 0
  br i1 %niter.ncmp.3, label %exit, label %body

exit:
  ret i32 %n

}

