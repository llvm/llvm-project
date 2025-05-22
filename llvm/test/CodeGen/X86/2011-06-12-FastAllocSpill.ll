; REQUIRES: asserts
; RUN: llc < %s -O0 -frame-pointer=all -relocation-model=pic -stats 2>&1 | FileCheck %s
;
; This test should not cause any spilling with RAFast.
;
; CHECK: Number of copies coalesced
; CHECK-NOT: Number of stores added
;
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

%0 = type { i64, i64, ptr, ptr }
%1 = type opaque
%2 = type opaque
%3 = type <{ ptr, i32, i32, ptr, ptr, i64 }>
%4 = type { ptr, i32, i32, ptr, ptr, i64 }
%5 = type { i64, i64 }
%6 = type { ptr, i32, i32, ptr, ptr }

@0 = external hidden constant %0

define hidden void @f() ssp {
bb:
  %tmp5 = alloca i64, align 8
  %tmp6 = alloca ptr, align 8
  %tmp7 = alloca %3, align 8
  store i64 0, ptr %tmp5, align 8
  br label %bb8

bb8:                                              ; preds = %bb23, %bb
  %tmp15 = getelementptr inbounds %3, ptr %tmp7, i32 0, i32 4
  store ptr @0, ptr %tmp15
  store ptr %tmp7, ptr %tmp6, align 8
  %tmp17 = load ptr, ptr %tmp6, align 8
  %tmp19 = getelementptr inbounds %6, ptr %tmp17, i32 0, i32 3
  %tmp21 = load ptr, ptr %tmp19
  call void %tmp21(ptr %tmp17)
  br label %bb23

bb23:                                             ; preds = %bb8
  %tmp24 = load i64, ptr %tmp5, align 8
  %tmp25 = add i64 %tmp24, 1
  store i64 %tmp25, ptr %tmp5, align 8
  %tmp26 = icmp ult i64 %tmp25, 10
  br i1 %tmp26, label %bb8, label %bb27

bb27:                                             ; preds = %bb23
  ret void
}
