; RUN: opt %loadNPMPolly '-passes=print<polly-detect>' -polly-process-unprofitable=false -disable-output -polly-detect-full-functions  < %s 2>&1 | FileCheck %s

; Verify if a simple function with basic block not part of loop doesn't crash with polly-process-unprofitable=false and polly-detect-full-functions flags.

; CHECK: Detected Scops in Function foo

define void @foo() {
  br label %1

1:                                                ; preds = %1, %0
  br i1 false, label %2, label %1

2:                                                ; preds = %1
  %3 = load ptr, ptr null, align 8
  store ptr null, ptr null, align 8
  ret void
}
