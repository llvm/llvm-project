; RUN: opt -passes=gvn < %s

define void @__memdep_requires_dominator_tree(ptr nocapture %bufUInt, ptr nocapture %pattern) nounwind {
entry:
  br label %for.body

for.exit:                                         ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %i.01 = phi i32 [ 0, %entry ], [ %tmp8.7, %for.body ]
  %arrayidx = getelementptr i32, ptr %bufUInt, i32 %i.01
  %arrayidx5 = getelementptr i32, ptr %pattern, i32 %i.01
  %tmp6 = load i32, ptr %arrayidx5, align 4
  store i32 %tmp6, ptr %arrayidx, align 4
  %tmp8.7 = add i32 %i.01, 8
  %cmp.7 = icmp ult i32 %tmp8.7, 1024
  br i1 %cmp.7, label %for.body, label %for.exit
}
