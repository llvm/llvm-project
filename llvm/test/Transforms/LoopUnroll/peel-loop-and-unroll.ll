; RUN: not opt -loop-unroll -unroll-peel-count=2 -unroll-count=2 -S < %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Cannot specify both explicit peel count and explicit unroll count

@a = global [8 x i32] zeroinitializer, align 16

define void @test1() {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [8 x i32], [8 x i32]* @a, i64 0, i64 %indvars.iv
  %0 = trunc i64 %indvars.iv to i32
  store i32 %0, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 8
  br i1 %exitcond, label %for.body, label %for.exit

for.exit:                        ; preds = %for.body
  ret void
}
