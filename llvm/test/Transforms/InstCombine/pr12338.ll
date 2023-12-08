; RUN: opt < %s -passes=instcombine -S | FileCheck %s

define void @entry() nounwind {
entry:
  br label %for.cond

; CHECK: br label %for.cond
for.cond:
  %local = phi <1 x i32> [ <i32 0>, %entry ], [ %phi2, %cond.end47 ]
  %phi3 = sub <1 x i32> zeroinitializer, %local
  br label %cond.end

cond.false:
  br label %cond.end

cond.end:
  %cond = phi <1 x i32> [ %phi3, %for.cond ], [ undef, %cond.false ]
  br label %cond.end47

cond.end47:
  %sum = add <1 x i32> %cond, <i32 92>
  %phi2 = sub <1 x i32> zeroinitializer, %sum
  br label %for.cond
}
