; This should be run without alias analysis enabled.
;RUN: opt %loadPolly -polly-scops  -disable-output < %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"

define i32 @main() nounwind {
entry:
  %t.02.reg2mem = alloca float
  br label %entry.split

entry.split:                                      ; preds = %entry
  store float 0.000000e+00, ptr %t.02.reg2mem
  br label %for.body

for.body:                                         ; preds = %for.body, %entry.split
  %j.01 = phi i32 [ 0, %entry.split ], [ %inc1, %for.body ]
  %t.02.reload = load float, ptr %t.02.reg2mem
  %inc = fadd float %t.02.reload, 1.000000e+00
  %inc1 = add nsw i32 %j.01, 1
  %exitcond = icmp eq i32 %inc1, 5000001
  store float %inc, ptr %t.02.reg2mem
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %conv = fptosi float %inc to i32
  ret i32 %conv
}
