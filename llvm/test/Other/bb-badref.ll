; RUN: opt -passes=loop-instsimplify -print-after-all -disable-output -S < %s 2>&1 | FileCheck %s

; loop-instsimplify dumps individual basic blocks as part of a loop,
; not a function.  Verify that the non-entry basic block is labeled as
; "1", not "<badref>".

; CHECK-NOT: <badref>

define void @foo() {
  br label %1

1:
  br label %1
}
