; RUN: opt %loadNPMPolly '-passes=print<polly-detect>' -disable-output < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @f(ptr %A, i64 %N, i64 %M) nounwind {
entry:
  fence seq_cst
  br label %for.i

for.i:
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %for.i ]
  %scevgep = getelementptr i64, ptr %A, i64 %indvar
  store i64 %indvar, ptr %scevgep
  %indvar.next = add nsw i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %N
  br i1 %exitcond, label %next, label %for.i

next:
  fence seq_cst
  br label %for.j

for.j:
  %indvar.j = phi i64 [ %indvar, %next ], [ %indvar.j.next, %for.j ]
  %scevgep.j = getelementptr i64, ptr %A, i64 %indvar.j
  store i64 %indvar.j, ptr %scevgep.j
  fence seq_cst
  %indvar.j.next = add nsw i64 %indvar.j, 1
  %exitcond.j = icmp eq i64 %indvar.j.next, %M
  br i1 %exitcond.j, label %return, label %for.j

return:
  fence seq_cst
  ret void
}
