; RUN: opt < %s -indvars -S | FileCheck %s
; Test indvars' ability to hoist new sext created by WidenIV.
; From ffbench.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
define internal double @fourn(double* %data, i32 %x, i32 %y, i32 %n) nounwind {
; CHECK: entry:
; CHECK: sext
; CHECK: sext
entry:
  br label %for.body

; CHECK: for.body:
; CHECK-NOT: sext
; CHECK: br
for.body:
  %i2.115 = phi i32 [ 0, %entry ], [ %add249, %for.body ]
  %add174 = add nsw i32 %i2.115, %x
  %idxprom177 = sext i32 %add174 to i64
  %arrayidx179 = getelementptr inbounds double* %data, i64 %idxprom177
  %tmp180 = load double* %arrayidx179, align 8
  %add249 = add nsw i32 %i2.115, %y
  %cmp168 = icmp sgt i32 %add249, %n
  br i1 %cmp168, label %exit, label %for.body

exit:
  ret double %tmp180
}
