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

declare i8* (i8*, i8*, ...)* @f1(i8*, i8*) optsize
declare i8* @f2(i8*, i8*, ...)

define internal void @f(i8* %self, i8* %_cmd, %0* %inObjects, %0* %inIndexes) optsize ssp {
entry:
  %call14 = tail call i8* (i8*, i8*, ...)* (i8*, i8*)* @f1(i8* undef, i8* %_cmd) optsize
  %0 = bitcast i8* (i8*, i8*, ...)* %call14 to void (i8*, i8*, %0*, %0*)*
  tail call void %0(i8* %self, i8* %_cmd, %0* %inObjects, %0* %inIndexes) optsize
  tail call void bitcast (i8* (i8*, i8*, ...)* @f2 to void (i8*, i8*, i32, %0*, %0*)*)(i8* %self, i8* undef, i32 2, %0* %inIndexes, %0* undef) optsize
  ret void
}
