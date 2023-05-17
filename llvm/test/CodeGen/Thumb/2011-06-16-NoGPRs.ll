; RUN: llc < %s
;
; This test would crash because isel creates a GPR register for the return
; value from f1. The register is only used by tBLXr_r9 which accepts a full GPR
; register, but we cannot have live GPRs in thumb mode because we don't know how
; to spill them.
;
; <rdar://problem/9624323>
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv6-apple-darwin10"

%0 = type opaque

declare ptr @f1(ptr, ptr) optsize
declare ptr @f2(ptr, ptr, ...)

define internal void @f(ptr %self, ptr %_cmd, ptr %inObjects, ptr %inIndexes) optsize ssp {
entry:
  %call14 = tail call ptr (ptr, ptr) @f1(ptr undef, ptr %_cmd) optsize
  tail call void %call14(ptr %self, ptr %_cmd, ptr %inObjects, ptr %inIndexes) optsize
  tail call void @f2(ptr %self, ptr undef, i32 2, ptr %inIndexes, ptr undef) optsize
  ret void
}
