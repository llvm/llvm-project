; RUN: llc < %s -mcpu=core2
; PR7375
;
; This function contains a block (while.cond) with a lonely RFP use that is
; not a kill. We still need an FP_REG_KILL for that block since the register
; allocator will insert a reload.
;
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

define void @_ZN7QVectorIdE4fillERKdi(ptr nocapture %t) nounwind ssp align 2 {
entry:
  %tmp2 = load double, ptr %t                         ; <double> [#uses=1]
  br i1 undef, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  br i1 undef, label %if.end, label %bb.nph

while.cond:                                       ; preds = %bb.nph, %while.cond
  store double %tmp2, ptr undef
  br i1 undef, label %if.end, label %while.cond

bb.nph:                                           ; preds = %if.then
  br label %while.cond

if.end:                                           ; preds = %while.cond, %if.then, %entry
  ret void
}
