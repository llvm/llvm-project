; RUN: llc < %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin9.0.0"

%0 = type { i32, i1 }

declare %0 @llvm.umul.with.overflow.i32(i32, i32) nounwind readnone

define linkonce_odr hidden void @_ZN2js5QueueINS_7SlotMap8SlotInfoEE6ensureEj(ptr nocapture %this, i32 %size) nounwind align 2 {
  br i1 undef, label %14, label %1

; <label>:1                                       ; preds = %0
  br i1 undef, label %2, label %3

; <label>:2                                       ; preds = %1
  br label %3

; <label>:3                                       ; preds = %2, %1
  br i1 undef, label %13, label %4

; <label>:4                                       ; preds = %3
  %5 = tail call %0 @llvm.umul.with.overflow.i32(i32 undef, i32 16)
  %6 = extractvalue %0 %5, 1
  %7 = extractvalue %0 %5, 0
  %.op = add i32 %7, 7
  %.op.op = and i32 %.op, -8
  %8 = select i1 %6, i32 0, i32 %.op.op
  br i1 undef, label %10, label %9

; <label>:9                                       ; preds = %4
  br label %_ZnamRN7nanojit9AllocatorE.exit

; <label>:10                                      ; preds = %4
  %11 = tail call ptr @_ZN7nanojit9Allocator9allocSlowEmb(ptr undef, i32 %8, i1 zeroext false) nounwind
  br label %_ZnamRN7nanojit9AllocatorE.exit

_ZnamRN7nanojit9AllocatorE.exit:                  ; preds = %10, %9
  br i1 false, label %._crit_edge, label %.lr.ph

.lr.ph:                                           ; preds = %_ZnamRN7nanojit9AllocatorE.exit
  br label %12

; <label>:12                                      ; preds = %12, %.lr.ph
  br i1 undef, label %._crit_edge, label %12

._crit_edge:                                      ; preds = %12, %_ZnamRN7nanojit9AllocatorE.exit
  br label %14

; <label>:13                                      ; preds = %3
  br label %14

; <label>:14                                      ; preds = %13, %._crit_edge, %0
  ret void
}

declare ptr @_ZN7nanojit9Allocator9allocSlowEmb(ptr, i32, i1 zeroext)
