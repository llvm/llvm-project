; REQUIRES: asserts
; RUN: opt < %s -passes='loop-vectorize' -force-target-supports-scalable-vectors=true -enable-epilogue-vectorization -epilogue-vectorization-force-VF=2 --debug-only=loop-vectorize -S -scalable-vectorization=on 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64-v256:256:256-v512:512:512"

; Currently we cannot handle scalable vectorization factors.
; CHECK: LV: Checking a loop in 'f1'
; CHECK: LEV: Epilogue vectorization factor is forced.
; CHECK: Epilogue Loop VF:2, Epilogue Loop UF:1

define void @f1(ptr %A) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %A, i64 %iv
  store i8 1, ptr %arrayidx, align 1
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp ne i64 %iv.next, 1024
  br i1 %exitcond, label %for.body, label %exit, !llvm.loop !0

exit:
  ret void
}

!0 = !{!0, !1}
!1 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
