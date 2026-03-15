; RUN: opt < %s -passes='print<loop-cache-cost>' -disable-output

; Ensure no crash happens after PR #164798

target datalayout = "p21:32:16"

define i16 @f() {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %i.02 = phi i16 [ 0, %entry ], [ %inc8, %for.cond.cleanup3 ]
  %idxprom = zext i16 %i.02 to i32
  %arrayidx = getelementptr [18 x i16], ptr addrspace(21) null, i32 %idxprom
  br label %for.body4

for.cond.cleanup:
  ret i16 0

for.cond.cleanup3:
  %inc8 = add i16 %i.02, 1
  %exitcond3.not = icmp eq i16 %inc8, 0
  br i1 %exitcond3.not, label %for.cond.cleanup, label %for.cond1.preheader

for.body4:
  %j.01 = phi i16 [ 0, %for.cond1.preheader ], [ %inc.2, %for.body4 ]
  %idxprom5 = zext i16 %j.01 to i32
  %arrayidx6 = getelementptr i16, ptr addrspace(21) %arrayidx, i32 %idxprom5
  store i16 0, ptr addrspace(21) %arrayidx6, align 1
  %inc.2 = add i16 %j.01, 1
  %exitcond.not.2 = icmp eq i16 %inc.2, 18
  br i1 %exitcond.not.2, label %for.cond.cleanup3, label %for.body4
}
