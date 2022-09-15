; RUN: opt < %s -msan-instrumentation-with-call-threshold=0 -S -passes=msan 2>&1 | FileCheck %s --implicit-check-not="call void @__msan_w" --implicit-check-not="call void @__msan_m"

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define <4 x i32> @test1(<4 x i32> %vec, i1 %idx, i32 %x) sanitize_memory {
  %vec1 = insertelement <4 x i32> %vec, i32 %x, i1 %idx
  ret <4 x i32> %vec1
}
; CHECK-LABEL: @test1(
; CHECK:         %[[A:.*]] = zext i1 {{.*}} to i8
; CHECK:         call void @__msan_maybe_warning_1(i8 zeroext %[[A]], i32 zeroext 0)
; CHECK:         ret <4 x i32>

define <4 x i32> @test2(<4 x i32> %vec, i2 %idx, i32 %x) sanitize_memory {
  %vec1 = insertelement <4 x i32> %vec, i32 %x, i2 %idx
  ret <4 x i32> %vec1
}
; CHECK-LABEL: @test2(
; CHECK:         %[[A:.*]] = zext i2 {{.*}} to i8
; CHECK:         call void @__msan_maybe_warning_1(i8 zeroext %[[A]], i32 zeroext 0)
; CHECK:         ret <4 x i32>

define <4 x i32> @test8(<4 x i32> %vec, i8 %idx, i32 %x) sanitize_memory {
  %vec1 = insertelement <4 x i32> %vec, i32 %x, i8 %idx
  ret <4 x i32> %vec1
}
; CHECK-LABEL: @test8(
; zext i8 -> i8 unnecessary.
; CHECK:         call void @__msan_maybe_warning_1(i8 zeroext %{{.*}}, i32 zeroext 0)
; CHECK:         ret <4 x i32>

define <4 x i32> @test9(<4 x i32> %vec, i9 %idx, i32 %x) sanitize_memory {
  %vec1 = insertelement <4 x i32> %vec, i32 %x, i9 %idx
  ret <4 x i32> %vec1
}
; CHECK-LABEL: @test9(
; CHECK:         %[[A:.*]] = zext i9 {{.*}} to i16
; CHECK:         call void @__msan_maybe_warning_2(i16 zeroext %[[A]], i32 zeroext 0)
; CHECK:         ret <4 x i32>

define <4 x i32> @test16(<4 x i32> %vec, i16 %idx, i32 %x) sanitize_memory {
  %vec1 = insertelement <4 x i32> %vec, i32 %x, i16 %idx
  ret <4 x i32> %vec1
}
; CHECK-LABEL: @test16(
; CHECK:         call void @__msan_maybe_warning_2(i16 zeroext %{{.*}}, i32 zeroext 0)
; CHECK:         ret <4 x i32>

define <4 x i32> @test17(<4 x i32> %vec, i17 %idx, i32 %x) sanitize_memory {
  %vec1 = insertelement <4 x i32> %vec, i32 %x, i17 %idx
  ret <4 x i32> %vec1
}
; CHECK-LABEL: @test17(
; CHECK:         %[[A:.*]] = zext i17 {{.*}} to i32
; CHECK:         call void @__msan_maybe_warning_4(i32 zeroext %[[A]], i32 zeroext 0)
; CHECK:         ret <4 x i32>

define <4 x i32> @test42(<4 x i32> %vec, i42 %idx, i32 %x) sanitize_memory {
  %vec1 = insertelement <4 x i32> %vec, i32 %x, i42 %idx
  ret <4 x i32> %vec1
}
; CHECK-LABEL: @test42(
; CHECK:         %[[A:.*]] = zext i42 {{.*}} to i64
; CHECK:         call void @__msan_maybe_warning_8(i64 zeroext %[[A]], i32 zeroext 0)
; CHECK:         ret <4 x i32>

define <4 x i32> @test64(<4 x i32> %vec, i64 %idx, i32 %x) sanitize_memory {
  %vec1 = insertelement <4 x i32> %vec, i32 %x, i64 %idx
  ret <4 x i32> %vec1
}
; CHECK-LABEL: @test64(
; CHECK:         call void @__msan_maybe_warning_8(i64 zeroext %{{.*}}, i32 zeroext 0)
; CHECK:         ret <4 x i32>

; Type size too large => inline check.
define <4 x i32> @test65(<4 x i32> %vec, i65 %idx, i32 %x) sanitize_memory {
  %vec1 = insertelement <4 x i32> %vec, i32 %x, i65 %idx
  ret <4 x i32> %vec1
}
; CHECK-LABEL: @test65(
; CHECK:         call void @__msan_warning_noreturn
; CHECK:         ret <4 x i32>

define <4 x i32> @testUndef(<4 x i32> %vec, i32 %x) sanitize_memory {
  %vec1 = insertelement <4 x i32> %vec, i32 undef, i32 undef
  ret <4 x i32> %vec1
}
; CHECK-LABEL: @testUndef(
; CHECK:         call void @__msan_warning_noreturn
; CHECK:         ret <4 x i32>
