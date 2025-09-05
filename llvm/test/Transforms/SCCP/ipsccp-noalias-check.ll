; RUN: opt < %s -passes='ipsccp,inline' -S | FileCheck %s

; This test looks if the store from callee after inlining gets the alias.scope
; If IPSCCP ignores noalias %p ptr and replaces %add.ptr5's GEP %p with @arr,
; the store will not receive an alias.scope as noalias attribute is lost.
@arr = global [100 x i32] zeroinitializer, align 16

define void @caller(i32 noundef %c, i32 noundef %d) #0 {
; COM: Check that store has both !alias.scope and !noalias
; CHECK-LABEL: for.body.i:
; CHECK:  store {{.*}}!alias.scope{{.*}}!noalias
entry:
  %idx.ext = sext i32 %d to i64
  %add.ptr = getelementptr inbounds i32, ptr @arr, i64 %idx.ext
  %add.ptr4 = getelementptr inbounds i32, ptr %add.ptr, i64 %idx.ext
  call void @callee(ptr noundef @arr, ptr noundef %add.ptr, ptr noundef %add.ptr4, i32 noundef %d)
  ret void
}

define void @callee(ptr noalias noundef %p, ptr noalias noundef %q, ptr noalias noundef %r, i32 noundef %len) #1 align 2 {
entry:
  br label %for.cond
for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %i.0, %len
  br i1 %cmp, label %for.body, label %for.end
for.body:                                         ; preds = %for.cond
  %idx.ext = sext i32 %i.0 to i64
  %add.ptr = getelementptr inbounds i32, ptr %q, i64 %idx.ext
  %0 = load i32, ptr %add.ptr, align 4
  %add.ptr3 = getelementptr inbounds i32, ptr %r, i64 %idx.ext
  %1 = load i32, ptr %add.ptr3, align 4
  %mul = mul nsw i32 %0, %1
  %add = add nsw i32 %mul, %i.0
  %add.ptr5 = getelementptr inbounds i32, ptr %p, i64 %idx.ext
  store i32 %add, ptr %add.ptr5, align 4
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

attributes #0 = { mustprogress uwtable}
attributes #1 = { mustprogress nounwind uwtable }


