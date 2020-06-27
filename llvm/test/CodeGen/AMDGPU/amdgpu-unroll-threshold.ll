; RUN: opt < %s -S -mtriple=amdgcn-- -basic-aa -loop-unroll | FileCheck %s

; Check that the loop in unroll_default is not fully unrolled using the default
; unroll threshold
; CHECK-LABEL: @unroll_default
; CHECK: entry:
; CHECK: br i1 %cmp
; CHECK: ret void

; Check that the same loop in unroll_full is fully unrolled when the default
; unroll threshold is increased by use of the amdgpu-unroll-threshold attribute
; CHECK-LABEL: @unroll_full
; CHECK: entry:
; CHECK-NOT: br i1 %cmp
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

define void @unroll_full() #0 {
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

attributes #0 = { "amdgpu-unroll-threshold"="1000" }
