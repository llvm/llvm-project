; Test to make sure loop vectorizer metadata is automatically upgraded.
;
; Run using opt as well to ensure that the metadata is upgraded when parsing
; assembly.
;
; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: opt -S < %s | FileCheck %s
; RUN: verify-uselistorder %s

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

!0 = !{!"clang version 3.5.0 (trunk 211528)"}
!1 = !{!1, !2, !3, !4, !4}
!2 = !{!"llvm.vectorizer.unroll", i32 4}
!3 = !{!"llvm.vectorizer.width", i32 8}
!4 = !{!"llvm.vectorizer.enable", i1 true}
