; RUN: opt -opaque-pointers=0 -passes=inline -S < %s | FileCheck %s
; RUN: opt -opaque-pointers=0 -passes='cgscc(inline)' -S < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare void @llvm.lifetime.start.p0i8(i64, i8*)
declare void @llvm.lifetime.end.p0i8(i64, i8*)

define void @helper_both_markers() {
  %a = alloca i8
  ; Size in llvm.lifetime.start / llvm.lifetime.end differs from
  ; allocation size. We should use the former.
  call void @llvm.lifetime.start.p0i8(i64 2, i8* %a)
  call void @llvm.lifetime.end.p0i8(i64 2, i8* %a)
  ret void
}

define void @test_both_markers() {
; CHECK-LABEL: @test_both_markers(
; CHECK: llvm.lifetime.start.p0i8(i64 2
; CHECK-NEXT: llvm.lifetime.end.p0i8(i64 2
  call void @helper_both_markers()
; CHECK-NEXT: llvm.lifetime.start.p0i8(i64 2
; CHECK-NEXT: llvm.lifetime.end.p0i8(i64 2
  call void @helper_both_markers()
; CHECK-NEXT: ret void
  ret void
}

;; Without this, the inliner will simplify out @test_no_marker before adding
;; any lifetime markers.
declare void @use(i8* %a)

define void @helper_no_markers() {
  %a = alloca i8 ; Allocation size is 1 byte.
  call void @use(i8* %a)
  ret void
}

;; We can't use CHECK-NEXT because there's an extra call void @use in between.
;; Instead, we use CHECK-NOT to verify that there are no other lifetime calls.
define void @test_no_marker() {
; CHECK-LABEL: @test_no_marker(
; CHECK-NOT: lifetime
; CHECK: llvm.lifetime.start.p0i8(i64 1
; CHECK-NOT: lifetime
; CHECK: llvm.lifetime.end.p0i8(i64 1
  call void @helper_no_markers()
; CHECK-NOT: lifetime
; CHECK: llvm.lifetime.start.p0i8(i64 1
; CHECK-NOT: lifetime
; CHECK: llvm.lifetime.end.p0i8(i64 1
  call void @helper_no_markers()
; CHECK-NOT: lifetime
; CHECK: ret void
  ret void
}

define void @helper_two_casts() {
  %a = alloca i32
  %b = bitcast i32* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %b)
  %c = bitcast i32* %a to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %c)
  ret void
}

define void @test_two_casts() {
; CHECK-LABEL: @test_two_casts(
; CHECK-NOT: lifetime
; CHECK: llvm.lifetime.start.p0i8(i64 4
; CHECK-NOT: lifetime
; CHECK: llvm.lifetime.end.p0i8(i64 4
  call void @helper_two_casts()
; CHECK-NOT: lifetime
; CHECK: llvm.lifetime.start.p0i8(i64 4
; CHECK-NOT: lifetime
; CHECK: llvm.lifetime.end.p0i8(i64 4
  call void @helper_two_casts()
; CHECK-NOT: lifetime
; CHECK: ret void
  ret void
}

define void @helper_arrays_alloca() {
  %a = alloca [10 x i32], align 16
  %1 = bitcast [10 x i32]* %a to i8*
  call void @use(i8* %1)
  ret void
}

define void @test_arrays_alloca() {
; CHECK-LABEL: @test_arrays_alloca(
; CHECK-NOT: lifetime
; CHECK: llvm.lifetime.start.p0i8(i64 40,
; CHECK-NOT: lifetime
; CHECK: llvm.lifetime.end.p0i8(i64 40,
  call void @helper_arrays_alloca()
; CHECK-NOT: lifetime
; CHECK: ret void
  ret void
}

%swift.error = type opaque

define void @helper_swifterror_alloca() {
entry:
  %swifterror = alloca swifterror %swift.error*, align 8
  store %swift.error* null, %swift.error** %swifterror, align 8
  ret void
}

define void @test_swifterror_alloca() {
; CHECK-LABEL: @test_swifterror_alloca(
; CHECK-NOT: lifetime
  call void @helper_swifterror_alloca()
; CHECK: ret void
  ret void
}
