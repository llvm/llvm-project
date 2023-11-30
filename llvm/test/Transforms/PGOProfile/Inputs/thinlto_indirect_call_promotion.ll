; ModuleID = 'lib.bc'
source_filename = "lib.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@calleeAddrs = dso_local local_unnamed_addr global [2 x ptr] [ptr @_ZL7callee0v, ptr @_ZL7callee1v], align 16

define internal void @_ZL7callee0v() {
entry:
  ret void
}

define internal void @_ZL7callee1v() {
entry:
  ret void
}

define dso_local void @_Z11global_funcv() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp ult i32 %i.0, 5
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.body:                                         ; preds = %for.cond
  %rem = and i32 %i.0, 1
  %idxprom = zext nneg i32 %rem to i64
  %arrayidx = getelementptr inbounds [2 x ptr], ptr @calleeAddrs, i64 0, i64 %idxprom
  %0 = load ptr, ptr %arrayidx ;, align 8, !tbaa !5
  call void %0()
  %inc = add nuw nsw i32 %i.0, 1
  br label %for.cond ;, !llvm.loop !9
}