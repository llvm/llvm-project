; RUN: opt < %s -passes=loop-vectorize -disable-output -debug-only=loop-vectorize 2>&1 | FileCheck %s
; REQUIRES: asserts
target triple = "aarch64"

; CHECK: Cost of 1 for VF 2: EMIT vp<{{.*}}> = first-order splice
; CHECK: Cost of 3 for VF vscale x 2: EMIT vp<{{.*}}> = first-order splice

define void @foo(ptr noalias %in, ptr noalias %out, i64 %n) "target-features"="+sve" {
entry:
  %load.prev = load i64, ptr %in, align 8
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %prev = phi i64 [ %load.prev, %entry ], [ %load.cur, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx.load = getelementptr inbounds nuw i64, ptr %in, i64 %indvars.iv.next
  %load.cur = load i64, ptr %arrayidx.load
  %add = add i64 %load.cur, %prev
  %arrayidx.store = getelementptr inbounds nuw i64, ptr %out, i64 %indvars.iv
  store i64 %add, ptr %arrayidx.store
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !0

for.cond.cleanup:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.unroll.disable"}
