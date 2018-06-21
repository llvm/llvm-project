; RUN: opt < %s -sroa -S | FileCheck %s
target datalayout = "e-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-f80:128-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind

define void @no_split_on_non_byte_width(i32) {
; This tests that allocas are not split into slices that are not byte width multiple
  %arg = alloca i32 , align 8
  store i32 %0, i32* %arg
  br label %load_i32

load_i32:
; CHECK-LABEL: load_i32:
; CHECK-NOT: bitcast {{.*}} to i1
; CHECK-NOT: zext i1
  %r0 = load i32, i32* %arg
  br label %load_i1

load_i1:
; CHECK-LABEL: load_i1:
; CHECK: bitcast {{.*}} to i1
  %p1 = bitcast i32* %arg to i1*
  %t1 = load i1, i1* %p1
  ret void
}

; PR18726: Check that we use memcpy and memset to fill out padding when we have
; a slice with a simple single type whose store size is smaller than the slice
; size.

%union.Foo = type { x86_fp80, i64, i64 }

@foo_copy_source = external constant %union.Foo
@i64_sink = global i64 0

define void @memcpy_fp80_padding() {
  %x = alloca %union.Foo

  ; Copy from a global.
  %x_i8 = bitcast %union.Foo* %x to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %x_i8, i8* bitcast (%union.Foo* @foo_copy_source to i8*), i32 32, i32 16, i1 false)

  ; Access a slice of the alloca to trigger SROA.
  %mid_p = getelementptr %union.Foo, %union.Foo* %x, i32 0, i32 1
  %elt = load i64, i64* %mid_p
  store i64 %elt, i64* @i64_sink
  ret void
}
; CHECK-LABEL: define void @memcpy_fp80_padding
; CHECK: alloca x86_fp80
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i32
; CHECK: load i64, i64* getelementptr inbounds (%union.Foo, %union.Foo* @foo_copy_source, i64 0, i32 1)
; CHECK: load i64, i64* getelementptr inbounds (%union.Foo, %union.Foo* @foo_copy_source, i64 0, i32 2)

define void @memset_fp80_padding() {
  %x = alloca %union.Foo

  ; Set to all ones.
  %x_i8 = bitcast %union.Foo* %x to i8*
  call void @llvm.memset.p0i8.i32(i8* %x_i8, i8 -1, i32 32, i32 16, i1 false)

  ; Access a slice of the alloca to trigger SROA.
  %mid_p = getelementptr %union.Foo, %union.Foo* %x, i32 0, i32 1
  %elt = load i64, i64* %mid_p
  store i64 %elt, i64* @i64_sink
  ret void
}
; CHECK-LABEL: define void @memset_fp80_padding
; CHECK: alloca x86_fp80
; CHECK: call void @llvm.memset.p0i8.i32(i8* %{{.*}}, i8 -1, i32 16, i32 16, i1 false)
; CHECK: store i64 -1, i64* @i64_sink

%S.vec3float = type { float, float, float }
%U.vec3float = type { <4 x float> }

declare i32 @memcpy_vec3float_helper(%S.vec3float*)

define i32 @memcpy_vec3float_widening(%S.vec3float* %x) {
; CHECK-LABEL: @memcpy_vec3float_widening(
; PR18726: Check that SROA does not rewrite a 12-byte memcpy into a 16-byte
; vector store, hence accidentally putting gibberish onto the stack.
entry:
  ; Create a temporary variable %tmp1 and copy %x[0] into it
  %tmp1 = alloca %S.vec3float, align 4
  %0 = bitcast %S.vec3float* %tmp1 to i8*
  %1 = bitcast %S.vec3float* %x to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %0, i8* %1, i32 12, i32 4, i1 false)

  ; The following block does nothing; but appears to confuse SROA
  %unused1 = bitcast %S.vec3float* %tmp1 to %U.vec3float*
  %unused2 = getelementptr inbounds %U.vec3float, %U.vec3float* %unused1, i32 0, i32 0
  %unused3 = load <4 x float>, <4 x float>* %unused2, align 1

  ; Create a second temporary and copy %tmp1 into it
  %tmp2 = alloca %S.vec3float, align 4
  %2 = bitcast %S.vec3float* %tmp2 to i8*
  %3 = bitcast %S.vec3float* %tmp1 to i8*
; CHECK: alloca
; CHECK-NOT: store <4 x float>
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %2, i8* %3, i32 12, i32 4, i1 false)

  %result = call i32 @memcpy_vec3float_helper(%S.vec3float* %tmp2)
  ret i32 %result
; CHECK: ret i32 %result
}
