; RUN: llc < %s -mtriple=thumbv7s-apple-ios6.0.0 -verify-machineinstrs

; Check to make sure the tail-call return at the end doesn't use a
; callee-saved register. Register hinting from t2LDRDri was getting this
; wrong. The intervening call will force allocation to try a high register
; first, so the hint will attempt to fire, but must be rejected due to
; not being in the allocation order for the tcGPR register class.
; The machine instruction verifier will make sure that all actually worked
; out the way it's supposed to.

%"myclass" = type { %struct.foo }
%struct.foo = type { i32, [40 x i8] }

define hidden void @func(ptr %Data) nounwind ssp {
  %1 = getelementptr inbounds i8, ptr %Data, i32 12
  tail call void @abc(ptr %1) nounwind
  tail call void @def(ptr %1) nounwind
  %2 = getelementptr inbounds i8, ptr %Data, i32 8
  %3 = load ptr, ptr %2, align 4
  tail call void @ghi(ptr %3) nounwind
  %4 = load ptr, ptr %Data, align 4
  %5 = getelementptr inbounds i8, ptr %Data, i32 4
  %6 = load ptr, ptr %5, align 4
  %7 = icmp eq ptr %Data, null
  br i1 %7, label %10, label %8

; <label>:12                                      ; preds = %0
  %9 = tail call ptr @jkl(ptr %1) nounwind
  tail call void @mno(ptr %Data) nounwind
  br label %10

; <label>:14                                      ; preds = %8, %0
  tail call void %4(ptr %6) nounwind
  ret void
}

declare void @mno(ptr)

declare void @def(ptr)

declare void @abc(ptr)

declare void @ghi(ptr)

declare ptr @jkl(ptr) nounwind
