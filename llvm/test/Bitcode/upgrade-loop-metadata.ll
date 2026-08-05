; Test to make sure loop vectorizer metadata is automatically upgraded.
;
; RUN: llvm-dis < %s.bc | FileCheck %s
; RUN: verify-uselistorder < %s.bc

define void @_Z28loop_with_vectorize_metadatav() {
entry:
  %i = alloca i32, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 16
  br i1 %cmp, label %for.body, label %for.end, !llvm.loop !1

for.body:                                         ; preds = %for.cond
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %1 = load i32, i32* %i, align 4
  %inc = add nsw i32 %1, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; CHECK: !{!"llvm.loop.interleave.count", i32 4}
; CHECK: !{!"llvm.loop.vectorize.width", i32 8}
; CHECK: !{!"llvm.loop.vectorize.enable", i1 true}

!0 = metadata !{metadata !"clang version 3.5.0 (trunk 211528)"}
!1 = metadata !{metadata !1, metadata !2, metadata !3, metadata !4, metadata !4}
!2 = metadata !{metadata !"llvm.vectorizer.unroll", i32 4}
!3 = metadata !{metadata !"llvm.vectorizer.width", i32 8}
!4 = metadata !{metadata !"llvm.vectorizer.enable", i1 true}
