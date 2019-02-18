; Test frame descriptors
;
; RUN: opt < %s -hwasan -S | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

declare void @use32(i32*, i64*)

define void @test_alloca() sanitize_hwaddress {
entry:
  %XYZ = alloca i32, align 4
  %ABC = alloca i64, align 4
  call void @use32(i32* nonnull %XYZ, i64 *nonnull %ABC)
  ret void
}

; CHECK: @[[STR:[0-9]*]] = private unnamed_addr constant [15 x i8] c"4 XYZ; 8 ABC; \00", align 1
; CHECK: private constant { void ()*, [15 x i8]* } { void ()* @test_alloca, [15 x i8]* @[[STR]] }, section "__hwasan_frames", comdat($test_alloca)

; CHECK-LABEL: @test_alloca(
; CHECK: ret void

; CHECK-LABEL: @hwasan.module_ctor
; CHECK: call void @__hwasan_init_frames(i8* @__start___hwasan_frames, i8* @__stop___hwasan_frames)
; CHECK: ret void

