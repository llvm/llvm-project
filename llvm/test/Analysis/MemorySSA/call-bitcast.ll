; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s
;
; Ensures that MemorySSA leverages the ground truth of the function being
; called even if the call and function signatures don't match.

declare i1 @opaque_true(i1) nounwind readonly

define i1 @foo(ptr %ptr, i1 %cond) {
  %cond_wide = zext i1 %cond to i32
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: call i32 @opaque_true(i32 %cond_wide)
  %cond_hidden_wide = call i32 @opaque_true(i32 %cond_wide)
  %cond_hidden = trunc i32 %cond_hidden_wide to i1
  ret i1 %cond_hidden
}
