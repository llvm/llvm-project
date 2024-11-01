; RUN: opt -S -passes=loop-vectorize -mattr=+sve -mtriple aarch64-unknown-linux-gnu -force-vector-width=2 -pass-remarks-analysis=loop-vectorize -pass-remarks-missed=loop-vectorize < %s 2>%t | FileCheck %s
; RUN: FileCheck %s --check-prefix=CHECK-REMARKS < %t

; CHECK-REMARKS: UserVF ignored because of invalid costs.
; CHECK-REMARKS: Instruction with invalid costs prevented vectorization at VF=(vscale x 1, vscale x 2): alloca
; CHECK-REMARKS: Instruction with invalid costs prevented vectorization at VF=(vscale x 1): store
define void @alloca(i32** %vla, i64 %N) {
; CHECK-LABEL: @alloca(
; CHECK-NOT: <vscale x

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %alloca = alloca i32, align 16
  %arrayidx = getelementptr inbounds i32*, i32** %vla, i64 %iv
  store i32* %alloca, i32** %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  call void @foo(i32** nonnull %vla)
  ret void
}

declare void @foo(i32**)

!0 = !{!0, !1}
!1 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
