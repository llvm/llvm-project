; RUN: llc < %s -march=nvptx64 -mcpu=sm_35 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_35 | %ptxas-verify %}


; CHECK-LABEL: test_v2i8
; CHECK-DAG:        ld.param.u16    [[A:%rs[0-9]+]], [test_v2i8_param_0];
; CHECK-DAG:        cvt.s16.s8      [[E0:%rs[0-9]+]], [[A]];
; CHECK-DAG:        shr.s16         [[E1:%rs[0-9]+]], [[A]], 8;
define i16  @test_v2i8(i16 %a) {
  %v = bitcast i16 %a to <2 x i8>
  %r0 = extractelement <2 x i8> %v, i64 0
  %r1 = extractelement <2 x i8> %v, i64 1
  %r0i = sext i8 %r0 to i16
  %r1i = sext i8 %r1 to i16
  %r01 = add i16 %r0i, %r1i
  ret i16 %r01
}

; CHECK-LABEL: test_v4i8
; CHECK:            ld.param.u32    [[R:%r[0-9]+]], [test_v4i8_param_0];
; CHECK-DAG:        bfe.s32         [[R0:%r[0-9]+]], [[R]], 0, 8;
; CHECK-DAG:        cvt.s8.s32      [[E0:%rs[0-9]+]], [[R0]];
; CHECK-DAG:        bfe.s32         [[R1:%r[0-9]+]], [[R]], 8, 8;
; CHECK-DAG:        cvt.s8.s32      [[E1:%rs[0-9]+]], [[R1]];
; CHECK-DAG:        bfe.s32         [[R2:%r[0-9]+]], [[R]], 16, 8;
; CHECK-DAG:        cvt.s8.s32      [[E2:%rs[0-9]+]], [[R2]];
; CHECK-DAG:        bfe.s32         [[R3:%r[0-9]+]], [[R]], 24, 8;
; CHECK-DAG:        cvt.s8.s32      [[E3:%rs[0-9]+]], [[R3]];
define i16  @test_v4i8(i32 %a) {
  %v = bitcast i32 %a to <4 x i8>
  %r0 = extractelement <4 x i8> %v, i64 0
  %r1 = extractelement <4 x i8> %v, i64 1
  %r2 = extractelement <4 x i8> %v, i64 2
  %r3 = extractelement <4 x i8> %v, i64 3
  %r0i = sext i8 %r0 to i16
  %r1i = sext i8 %r1 to i16
  %r2i = sext i8 %r2 to i16
  %r3i = sext i8 %r3 to i16
  %r01 = add i16 %r0i, %r1i
  %r23 = add i16 %r2i, %r3i
  %r = add i16 %r01, %r23
  ret i16 %r
}

; CHECK-LABEL: test_v4i8_s32
; CHECK:            ld.param.u32    [[R:%r[0-9]+]], [test_v4i8_s32_param_0];
; CHECK-DAG:        bfe.s32         [[R0:%r[0-9]+]], [[R]], 0, 8;
; CHECK-DAG:        bfe.s32         [[R1:%r[0-9]+]], [[R]], 8, 8;
; CHECK-DAG:        bfe.s32         [[R2:%r[0-9]+]], [[R]], 16, 8;
; CHECK-DAG:        bfe.s32         [[R3:%r[0-9]+]], [[R]], 24, 8;
; CHECK-DAG:        add.s32         [[R01:%r[0-9]+]], [[R0]], [[R1]]
; CHECK-DAG:        add.s32         [[R23:%r[0-9]+]], [[R2]], [[R3]]
; CHECK-DAG:        add.s32         [[R0123:%r[0-9]+]], [[R01]], [[R23]]
define i32  @test_v4i8_s32(i32 %a) {
  %v = bitcast i32 %a to <4 x i8>
  %r0 = extractelement <4 x i8> %v, i64 0
  %r1 = extractelement <4 x i8> %v, i64 1
  %r2 = extractelement <4 x i8> %v, i64 2
  %r3 = extractelement <4 x i8> %v, i64 3
  %r0i = sext i8 %r0 to i32
  %r1i = sext i8 %r1 to i32
  %r2i = sext i8 %r2 to i32
  %r3i = sext i8 %r3 to i32
  %r01 = add i32 %r0i, %r1i
  %r23 = add i32 %r2i, %r3i
  %r = add i32 %r01, %r23
  ret i32 %r
}

; CHECK-LABEL: test_v4i8_u32
; CHECK:            ld.param.u32    [[R:%r[0-9]+]], [test_v4i8_u32_param_0];
; CHECK-DAG:        bfe.u32         [[R0:%r[0-9]+]], [[R]], 0, 8;
; CHECK-DAG:        bfe.u32         [[R1:%r[0-9]+]], [[R]], 8, 8;
; CHECK-DAG:        bfe.u32         [[R2:%r[0-9]+]], [[R]], 16, 8;
; CHECK-DAG:        bfe.u32         [[R3:%r[0-9]+]], [[R]], 24, 8;
; CHECK-DAG:        add.s32         [[R01:%r[0-9]+]], [[R0]], [[R1]]
; CHECK-DAG:        add.s32         [[R23:%r[0-9]+]], [[R2]], [[R3]]
; CHECK-DAG:        add.s32         [[R0123:%r[0-9]+]], [[R01]], [[R23]]
define i32  @test_v4i8_u32(i32 %a) {
  %v = bitcast i32 %a to <4 x i8>
  %r0 = extractelement <4 x i8> %v, i64 0
  %r1 = extractelement <4 x i8> %v, i64 1
  %r2 = extractelement <4 x i8> %v, i64 2
  %r3 = extractelement <4 x i8> %v, i64 3
  %r0i = zext i8 %r0 to i32
  %r1i = zext i8 %r1 to i32
  %r2i = zext i8 %r2 to i32
  %r3i = zext i8 %r3 to i32
  %r01 = add i32 %r0i, %r1i
  %r23 = add i32 %r2i, %r3i
  %r = add i32 %r01, %r23
  ret i32 %r
}



; CHECK-LABEL: test_v8i8
; CHECK:       ld.param.u64    [[R:%rd[0-9]+]], [test_v8i8_param_0];
; CHECK-DAG:        cvt.u32.u64     [[R00:%r[0-9]+]], [[R]];
; CHECK-DAG:        { .reg .b32 tmp; mov.b64 {tmp, [[R01:%r[0-9]+]]}, [[R]]; }
; CHECK-DAG:        bfe.s32         [[R1:%r[0-9]+]], [[R00]], 0, 8;
; CHECK-DAG:        cvt.s8.s32      [[E1:%rs[0-9]+]], [[R1]];
; CHECK-DAG:        bfe.s32         [[R2:%r[0-9]+]], [[R00]], 8, 8;
; CHECK-DAG:        cvt.s8.s32      [[E2:%rs[0-9]+]], [[R2]];
; CHECK-DAG:        bfe.s32         [[R3:%r[0-9]+]], [[R00]], 16, 8;
; CHECK-DAG:        cvt.s8.s32      [[E3:%rs[0-9]+]], [[R3]];
; CHECK-DAG:        bfe.s32         [[R4:%r[0-9]+]], [[R00]], 24, 8;
; CHECK-DAG:        cvt.s8.s32      [[E4:%rs[0-9]+]], [[R4]];
; CHECK-DAG:        bfe.s32         [[R5:%r[0-9]+]], [[R01]], 0, 8;
; CHECK-DAG:        cvt.s8.s32      [[E5:%rs[0-9]+]], [[R5]];
; CHECK-DAG:        bfe.s32         [[R6:%r[0-9]+]], [[R01]], 8, 8;
; CHECK-DAG:        cvt.s8.s32      [[E6:%rs[0-9]+]], [[R6]];
; CHECK-DAG:        bfe.s32         [[R7:%r[0-9]+]], [[R01]], 16, 8;
; CHECK-DAG:        cvt.s8.s32      [[E7:%rs[0-9]+]], [[R7]];
; CHECK-DAG:        bfe.s32         [[R8:%r[0-9]+]], [[R01]], 24, 8;
; CHECK-DAG:        cvt.s8.s32      [[E8:%rs[0-9]+]], [[R8]];

define i16  @test_v8i8(i64 %a) {
  %v = bitcast i64 %a to <8 x i8>
  %r0 = extractelement <8 x i8> %v, i64 0
  %r1 = extractelement <8 x i8> %v, i64 1
  %r2 = extractelement <8 x i8> %v, i64 2
  %r3 = extractelement <8 x i8> %v, i64 3
  %r4 = extractelement <8 x i8> %v, i64 4
  %r5 = extractelement <8 x i8> %v, i64 5
  %r6 = extractelement <8 x i8> %v, i64 6
  %r7 = extractelement <8 x i8> %v, i64 7
  %r0i = sext i8 %r0 to i16
  %r1i = sext i8 %r1 to i16
  %r2i = sext i8 %r2 to i16
  %r3i = sext i8 %r3 to i16
  %r4i = sext i8 %r4 to i16
  %r5i = sext i8 %r5 to i16
  %r6i = sext i8 %r6 to i16
  %r7i = sext i8 %r7 to i16
  %r01 = add i16 %r0i, %r1i
  %r23 = add i16 %r2i, %r3i
  %r45 = add i16 %r4i, %r5i
  %r67 = add i16 %r6i, %r7i
  %r0123 = add i16 %r01, %r23
  %r4567 = add i16 %r45, %r67
  %r = add i16 %r0123, %r4567
  ret i16 %r
}
