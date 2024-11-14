; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa 2>&1 \
; RUN: | FileCheck %s

; The dependence test does not handle array accesses of different sizes: i32 and i64.
; Bug 16183 - https://github.com/llvm/llvm-project/issues/16183
; CHECK-LABEL: bug16183_alias
; CHECK: da analyze - confused!

define i64 @bug16183_alias(i32* nocapture %A) {
entry:
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 1
  store i32 2, ptr %arrayidx, align 4
  %0 = load i64, ptr %A, align 8
  ret i64 %0
}
