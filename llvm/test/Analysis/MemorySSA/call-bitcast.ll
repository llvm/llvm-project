; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s
;
; Ensures that MemorySSA leverages the ground truth of the function being called when wrapped in a bitcast.

declare i1 @opaque_true(i1) nounwind readonly

define i1 @foo(i32* %ptr, i1 %cond) {
  %cond_wide = zext i1 %cond to i32
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: call i32 bitcast
  %cond_hidden_wide = call i32 bitcast (i1 (i1)* @opaque_true to i32 (i32)*)(i32 %cond_wide)
  %cond_hidden = trunc i32 %cond_hidden_wide to i1
  ret i1 %cond_hidden
}
