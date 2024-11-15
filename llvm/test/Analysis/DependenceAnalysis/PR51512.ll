; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa

; Check that the testcase does not crash the compiler.
; https://github.com/llvm/llvm-project/issues/51512

define void @func_1() {
entry:
  %l_83.i.i = alloca [2 x [5 x i32]], align 1
  br label %for.cond857.preheader.i.i

for.cond857.preheader.i.i:                        ; preds = %cleanup.cont1138.i.i, %entry
  %l_89.08.i.i = phi i32 [ 0, %entry ], [ %add1140.i.i, %cleanup.cont1138.i.i ]
  %0 = trunc i32 %l_89.08.i.i to i16
  %1 = add i16 %0, 3
  %arrayidx916.i.i = getelementptr inbounds [2 x [5 x i32]], [2 x [5 x i32]]* %l_83.i.i, i16 0, i16 %0, i16 %1
  br label %for.body860.i.i

for.body860.i.i:                                  ; preds = %for.body860.i.i, %for.cond857.preheader.i.i
  %l_74.07.i.i = phi i32 [ 0, %for.cond857.preheader.i.i ], [ %add964.i.i, %for.body860.i.i ]
  store i32 undef, i32* %arrayidx916.i.i, align 1
  %2 = trunc i32 %l_74.07.i.i to i16
  %arrayidx962.i.i = getelementptr inbounds [2 x [5 x i32]], [2 x [5 x i32]]* %l_83.i.i, i16 0, i16 %2, i16 %1
  store i32 0, i32* %arrayidx962.i.i, align 1
  %add964.i.i = add nuw nsw i32 %l_74.07.i.i, 1
  br i1 false, label %for.body860.i.i, label %cleanup.cont1138.i.i

cleanup.cont1138.i.i:                             ; preds = %for.body860.i.i
  %add1140.i.i = add nuw nsw i32 %l_89.08.i.i, 1
  %cmp602.i.i = icmp eq i32 %l_89.08.i.i, 0
  br i1 %cmp602.i.i, label %for.cond857.preheader.i.i, label %for.cond1480.i.i.preheader

for.cond1480.i.i.preheader:                       ; preds = %cleanup.cont1138.i.i
  unreachable
}
