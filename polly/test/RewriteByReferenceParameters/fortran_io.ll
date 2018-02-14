; RUN: opt %loadPolly -polly-rewrite-byref-params -S < %s \
; RUN: | FileCheck %s


; Verify that we rewrite the read-only by-reference into a separate alloca slot.
; This is useful in case %j3 is an induction variable, which should be promoted
; by -mem2reg into a register.

; CHECK: define void @foo(%struct.__st_parameter_dt* %p) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %polly_byref_alloca_j3 = alloca i32
; CHECK-NEXT:   %j3 = alloca i32, align 4
; CHECK-NEXT:   %tmp = bitcast i32* %j3 to i8*
; CHECK-NEXT:   br label %bb

; CHECK: bb:                                               ; preds = %entry
; CHECK-NEXT:   %polly_byref_load_j3 = load i32, i32* %j3
; CHECK-NEXT:   store i32 %polly_byref_load_j3, i32* %polly_byref_alloca_j3
; CHECK-NEXT:   %polly_byref_cast_j3 = bitcast i32* %polly_byref_alloca_j3 to i8*
; CHECK-NEXT:   call void @_gfortran_transfer_integer_write(%struct.__st_parameter_dt* %p, i8* %polly_byref_cast_j3, i32 4)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct.__st_parameter_dt = type { }

declare void @_gfortran_transfer_integer_write(%struct.__st_parameter_dt*, i8*, i32)

define void @foo(%struct.__st_parameter_dt* %p) {
entry:
  %j3 = alloca i32, align 4
  %tmp = bitcast i32* %j3 to i8*
  br label %bb

bb:
  call void @_gfortran_transfer_integer_write(%struct.__st_parameter_dt* %p, i8* %tmp, i32 4)
  ret void
}
