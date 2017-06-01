; RUN: opt < %s -basicaa -disable-verify -slp-vectorizer -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin13.3.0"

define void @_foo(double %p1, double %p2, double %p3) #0 {
entry:
  %tab1 = alloca [256 x i32], align 16
  %tab2 = alloca [256 x i32], align 16
  br label %bb1


bb1:
  %mul19 = fmul double %p1, 1.638400e+04
  %mul20 = fmul double %p3, 1.638400e+04
  %add = fadd double %mul20, 8.192000e+03
  %mul21 = fmul double %p2, 1.638400e+04
  ; The SLPVectorizer crashed when scheduling this block after it inserted an
  ; insertelement instruction (during vectorizing the for.body block) at this position.
  br label %for.body

for.body:
  %indvars.iv266 = phi i64 [ 0, %bb1 ], [ %indvars.iv.next267, %for.body ]
  %t.0259 = phi double [ 0.000000e+00, %bb1 ], [ %add27, %for.body ]
  %p3.addr.0258 = phi double [ %add, %bb1 ], [ %add28, %for.body ]
  %vecinit.i.i237 = insertelement <2 x double> undef, double %t.0259, i32 0
  %x13 = tail call i32 @_xfn(<2 x double> %vecinit.i.i237) #2
  %arrayidx = getelementptr inbounds [256 x i32], [256 x i32]* %tab1, i64 0, i64 %indvars.iv266
  store i32 %x13, i32* %arrayidx, align 4, !tbaa !4
  %vecinit.i.i = insertelement <2 x double> undef, double %p3.addr.0258, i32 0
  %x14 = tail call i32 @_xfn(<2 x double> %vecinit.i.i) #2
  %arrayidx26 = getelementptr inbounds [256 x i32], [256 x i32]* %tab2, i64 0, i64 %indvars.iv266
  store i32 %x14, i32* %arrayidx26, align 4, !tbaa !4
  %add27 = fadd double %mul19, %t.0259
  %add28 = fadd double %mul21, %p3.addr.0258
  %indvars.iv.next267 = add nuw nsw i64 %indvars.iv266, 1
  %exitcond = icmp eq i64 %indvars.iv.next267, 256
  br i1 %exitcond, label %return, label %for.body

return:
  ret void
}

declare i32 @_xfn(<2 x double>) #4

!3 = !{!"int", !4, i64 0}
!4 = !{!3, !3, i64 0}
