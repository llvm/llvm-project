; RUN: llc -mtriple=armv7-none-linux-gnueabi < %s | FileCheck %s

@var_v2i8 = global <2 x i8> zeroinitializer
@var_v4i8 = global <4 x i8> zeroinitializer

@var_v2i16 = global <2 x i16> zeroinitializer
@var_v4i16 = global <4 x i16> zeroinitializer

@var_v2i32 = global <2 x i32> zeroinitializer
@var_v4i32 = global <4 x i32> zeroinitializer

@var_v2i64 = global <2 x i64> zeroinitializer

define void @test_v2i8tov2i32() {
; CHECK-LABEL: test_v2i8tov2i32:

  %i8val = load <2 x i8>, ptr @var_v2i8

  %i32val = sext <2 x i8> %i8val to <2 x i32>
  store <2 x i32> %i32val, ptr @var_v2i32
; CHECK: vld1.16 {d[[LOAD:[0-9]+]][0]}, [{{r[0-9]+}}:16]
; CHECK: vmovl.s8 {{q[0-9]+}}, d[[LOAD]]
; CHECK: vmovl.s16 {{q[0-9]+}}, {{d[0-9]+}}

  ret void
}

define void @test_v2i8tov2i64() {
; CHECK-LABEL: test_v2i8tov2i64:

  %i8val = load <2 x i8>, ptr @var_v2i8

  %i64val = sext <2 x i8> %i8val to <2 x i64>
  store <2 x i64> %i64val, ptr @var_v2i64
; CHECK: vld1.16 {d{{[0-9]+}}[0]}, [{{r[0-9]+}}:16]
; CHECK: vmovl.s8 {{q[0-9]+}}, d[[LOAD]]
; CHECK: vmovl.s16 {{q[0-9]+}}, {{d[0-9]+}}
; CHECK: vmovl.s32 {{q[0-9]+}}, {{d[0-9]+}}

;  %i64val = sext <2 x i8> %i8val to <2 x i64>
;  store <2 x i64> %i64val, ptr @var_v2i64

  ret void
}

define void @test_v4i8tov4i16() {
; CHECK-LABEL: test_v4i8tov4i16:

  %i8val = load <4 x i8>, ptr @var_v4i8

  %i16val = sext <4 x i8> %i8val to <4 x i16>
  store <4 x i16> %i16val, ptr @var_v4i16
; CHECK: vld1.32 {d[[LOAD:[0-9]+]][0]}, [{{r[0-9]+}}:32]
; CHECK: vmovl.s8 {{q[0-9]+}}, d[[LOAD]]
; CHECK-NOT: vmovl.s16

  ret void
; CHECK: bx lr
}

define void @test_v4i8tov4i32() {
; CHECK-LABEL: test_v4i8tov4i32:

  %i8val = load <4 x i8>, ptr @var_v4i8

  %i16val = sext <4 x i8> %i8val to <4 x i32>
  store <4 x i32> %i16val, ptr @var_v4i32
; CHECK: vld1.32 {d[[LOAD:[0-9]+]][0]}, [{{r[0-9]+}}:32]
; CHECK: vmovl.s8 {{q[0-9]+}}, d[[LOAD]]
; CHECK: vmovl.s16 {{q[0-9]+}}, {{d[0-9]+}}

  ret void
}

define void @test_v2i16tov2i32() {
; CHECK-LABEL: test_v2i16tov2i32:

  %i16val = load <2 x i16>, ptr @var_v2i16

  %i32val = sext <2 x i16> %i16val to <2 x i32>
  store <2 x i32> %i32val, ptr @var_v2i32
; CHECK: vld1.32 {d[[LOAD:[0-9]+]][0]}, [{{r[0-9]+}}:32]
; CHECK: vmovl.s16 {{q[0-9]+}}, d[[LOAD]]
; CHECK-NOT: vmovl

  ret void
; CHECK: bx lr
}

define void @test_v2i16tov2i64() {
; CHECK-LABEL: test_v2i16tov2i64:

  %i16val = load <2 x i16>, ptr @var_v2i16

  %i64val = sext <2 x i16> %i16val to <2 x i64>
  store <2 x i64> %i64val, ptr @var_v2i64
; CHECK: vld1.32 {d[[LOAD:[0-9]+]][0]}, [{{r[0-9]+}}:32]
; CHECK: vmovl.s16 {{q[0-9]+}}, d[[LOAD]]
; CHECK: vmovl.s32 {{q[0-9]+}}, d[[LOAD]]

  ret void
}
