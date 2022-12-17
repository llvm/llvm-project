; RUN: opt < %s -S -mtriple=amdgcn-- -passes=loop-unroll | FileCheck %s

; Check the handling of amdgpu.loop.unroll.threshold metadata which can be used to
; set the default threshold for a loop. This metadata overrides both the AMDGPU
; default, and any value specified by the amdgpu-unroll-threshold function attribute
; (which sets a threshold for all loops in the function).

; Check that the loop in unroll_default is not fully unrolled using the default
; unroll threshold
; CHECK-LABEL: @unroll_default
; CHECK: entry:
; CHECK: br i1 %cmp
; CHECK: ret void

@in = internal unnamed_addr global i32* null, align 8
@out = internal unnamed_addr global i32* null, align 8

define void @unroll_default() {
entry:
  br label %do.body

do.body:                                          ; preds = %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %do.body ]
  %v1 = load i64, i64* bitcast (i32** @in to i64*), align 8
  store i64 %v1, i64* bitcast (i32** @out to i64*), align 8
  %inc = add nsw i32 %i.0, 1
  %cmp = icmp slt i32 %inc, 100
  br i1 %cmp, label %do.body, label %do.end

do.end:                                           ; preds = %do.body
  ret void
}

; Check that the same loop in unroll_full is fully unrolled when the default
; unroll threshold is increased by use of the amdgpu.loop.unroll.threshold metadata
; CHECK-LABEL: @unroll_full
; CHECK: entry:
; CHECK-NOT: br i1 %cmp
; CHECK: ret void

define void @unroll_full() {
entry:
  br label %do.body

do.body:                                          ; preds = %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %do.body ]
  %v1 = load i64, i64* bitcast (i32** @in to i64*), align 8
  store i64 %v1, i64* bitcast (i32** @out to i64*), align 8
  %inc = add nsw i32 %i.0, 1
  %cmp = icmp slt i32 %inc, 100
  br i1 %cmp, label %do.body, label %do.end, !llvm.loop !1

do.end:                                           ; preds = %do.body
  ret void
}

; Check that the same loop in override_no_unroll is not unrolled when a high default
; unroll threshold specified using the amdgpu-unroll-threshold function attribute
; is overridden by a low threshold using the amdgpu.loop.unroll.threshold metadata

; CHECK-LABEL: @override_no_unroll
; CHECK: entry:
; CHECK: br i1 %cmp
; CHECK: ret void

define void @override_no_unroll() #0 {
entry:
  br label %do.body

do.body:                                          ; preds = %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %do.body ]
  %v1 = load i64, i64* bitcast (i32** @in to i64*), align 8
  store i64 %v1, i64* bitcast (i32** @out to i64*), align 8
  %inc = add nsw i32 %i.0, 1
  %cmp = icmp slt i32 %inc, 100
  br i1 %cmp, label %do.body, label %do.end, !llvm.loop !3

do.end:                                           ; preds = %do.body
  ret void
}

; Check that the same loop in override_unroll is fully unrolled when a low default
; unroll threshold specified using the amdgpu-unroll-threshold function attribute
; is overridden by a high threshold using the amdgpu.loop.unroll.threshold metadata

; CHECK-LABEL: @override_unroll
; CHECK: entry:
; CHECK-NOT: br i1 %cmp
; CHECK: ret void

define void @override_unroll() #1 {
entry:
  br label %do.body

do.body:                                          ; preds = %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %do.body ]
  %v1 = load i64, i64* bitcast (i32** @in to i64*), align 8
  store i64 %v1, i64* bitcast (i32** @out to i64*), align 8
  %inc = add nsw i32 %i.0, 1
  %cmp = icmp slt i32 %inc, 100
  br i1 %cmp, label %do.body, label %do.end, !llvm.loop !1

do.end:                                           ; preds = %do.body
  ret void
}

attributes #0 = { "amdgpu-unroll-threshold"="1000" }
attributes #1 = { "amdgpu-unroll-threshold"="100" }

!1 = !{!1, !2}
!2 = !{!"amdgpu.loop.unroll.threshold", i32 1000}
!3 = !{!3, !4}
!4 = !{!"amdgpu.loop.unroll.threshold", i32 100}
