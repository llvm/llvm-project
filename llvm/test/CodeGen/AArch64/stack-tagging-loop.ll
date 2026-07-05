; We set a low dom-tree-reachability-max-bbs-to-explore to check whether the
; loop analysis is working. Without skipping over the loop, we would need more
; than 4 BB to reach end from entry.

; RUN: opt -S -dom-tree-reachability-max-bbs-to-explore=4 -aarch64-stack-tagging %s -o - | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

define dso_local void @foo(i1 %x, i32 %n) sanitize_memtag {
entry:
  %c = alloca [1024 x i8], align 1
  call void @llvm.lifetime.start.p0(i64 1024, ptr nonnull %c)
  %cmp2.not = icmp eq i32 %n, 0
  br i1 %x, label %entry2, label %noloop

entry2:
  br i1 %cmp2.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
; CHECK-LABEL: for.cond.cleanup:
; CHECK: call{{.*}}settag
; CHECK: call{{.*}}lifetime.end
  call void @llvm.lifetime.end.p0(i64 1024, ptr nonnull %c)
  call void @bar(ptr noundef nonnull inttoptr (i64 120 to ptr))
  br label %end

for.body:                                         ; preds = %entry, %for.body
  %i.03 = phi i32 [ %inc, %for.body2 ], [ 0, %entry2 ]
  call void @bar(ptr noundef nonnull %c) #3
  br label %for.body2

for.body2:
  %inc = add nuw nsw i32 %i.03, 1
  %cmp = icmp ult i32 %inc, %n
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !13

noloop:
; CHECK-LABEL: noloop:
; CHECK: call{{.*}}settag
; CHECK: call{{.*}}lifetime.end
  call void @llvm.lifetime.end.p0(i64 1024, ptr nonnull %c)
  br label %end

end:
; CHECK-LABEL: end:
; CHECK-NOT: call{{.*}}settag
  ret void
}

; Function Attrs: argmemonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #0
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #0

declare dso_local void @bar(ptr noundef)

attributes #0 = { argmemonly mustprogress nocallback nofree nosync nounwind willreturn }

!13 = distinct !{!13, !14}
!14 = !{!"llvm.loop.mustprogress"}
