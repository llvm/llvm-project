; RUN: opt < %s -loop-reduce -S | FileCheck %s
; CHECK-NOT: {{inttoptr|ptrtoint}}
; CHECK: scevgep
; CHECK-NOT: {{inttoptr|ptrtoint}}
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128-n32:64"

; Indvars shouldn't need inttoptr/ptrtoint to expand an address here.

define void @foo(i8* %p) nounwind {
entry:
  br i1 true, label %bb.nph, label %for.end

for.cond:
  %phitmp = icmp slt i64 %inc, 20
  br i1 %phitmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:
  br label %for.end

bb.nph:
  br label %for.body

for.body:
  %storemerge1 = phi i64 [ %inc, %for.cond ], [ 0, %bb.nph ]
  %call = tail call i64 @bar() nounwind
  %call2 = tail call i64 @car() nounwind
  %conv = trunc i64 %call2 to i8
  %conv3 = sext i8 %conv to i64
  %add = add nsw i64 %call, %storemerge1
  %add4 = add nsw i64 %add, %conv3
  %arrayidx = getelementptr inbounds i8* %p, i64 %add4
  store i8 0, i8* %arrayidx
  %inc = add nsw i64 %storemerge1, 1
  br label %for.cond

for.end:
  ret void
}

declare i64 @bar()

declare i64 @car()
