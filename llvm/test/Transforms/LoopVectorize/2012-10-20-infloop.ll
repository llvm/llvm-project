; RUN: opt < %s  -loop-vectorize -force-vector-width=4 -dce

; Check that we don't fall into an infinite loop.
define void @test() nounwind {
entry:
 br label %for.body

for.body:
 %0 = phi i32 [ 1, %entry ], [ 0, %for.body ]
 br label %for.body
}



define void @test2() nounwind {
entry:
 br label %for.body

for.body:                                         ; preds = %for.body, %entry
 %indvars.iv47 = phi i64 [ 0, %entry ], [ %indvars.iv.next48, %for.body ]
 %0 = phi i32 [ 1, %entry ], [ 0, %for.body ]
 %indvars.iv.next48 = add i64 %indvars.iv47, 1
 br i1 undef, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
 unreachable
}
