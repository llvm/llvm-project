; RUN: llc -verify-machineinstrs -o - %s -mtriple=arm64 | FileCheck %s

@var32_0 = global i32 0
@var32_1 = global i32 0
@var64_0 = global i64 0
@var64_1 = global i64 0

define void @rorv_i64() {
; CHECK-LABEL: rorv_i64:
    %val0_tmp = load i64, ptr @var64_0
    %val1_tmp = load i64, ptr @var64_1
    %val2_tmp = sub i64 64, %val1_tmp
    %val3_tmp = shl i64 %val0_tmp, %val2_tmp
    %val4_tmp = lshr i64 %val0_tmp, %val1_tmp
    %val5_tmp = or i64 %val3_tmp, %val4_tmp
; CHECK: {{ror|rorv}} {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
    store volatile i64 %val5_tmp, ptr @var64_0
    ret void
}

define void @asrv_i64() {
; CHECK-LABEL: asrv_i64:
    %val0_tmp = load i64, ptr @var64_0
    %val1_tmp = load i64, ptr @var64_1
    %val4_tmp = ashr i64 %val0_tmp, %val1_tmp
; CHECK: {{asr|asrv}} {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
    store volatile i64 %val4_tmp, ptr @var64_1
    ret void
}

define void @lsrv_i64() {
; CHECK-LABEL: lsrv_i64:
    %val0_tmp = load i64, ptr @var64_0
    %val1_tmp = load i64, ptr @var64_1
    %val4_tmp = lshr i64 %val0_tmp, %val1_tmp
; CHECK: {{lsr|lsrv}} {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
    store volatile i64 %val4_tmp, ptr @var64_0
    ret void
}

define void @lslv_i64() {
; CHECK-LABEL: lslv_i64:
    %val0_tmp = load i64, ptr @var64_0
    %val1_tmp = load i64, ptr @var64_1
    %val4_tmp = shl i64 %val0_tmp, %val1_tmp
; CHECK: {{lsl|lslv}} {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
    store volatile i64 %val4_tmp, ptr @var64_1
    ret void
}

define void @udiv_i64() {
; CHECK-LABEL: udiv_i64:
    %val0_tmp = load i64, ptr @var64_0
    %val1_tmp = load i64, ptr @var64_1
    %val4_tmp = udiv i64 %val0_tmp, %val1_tmp
; CHECK: udiv	{{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
    store volatile i64 %val4_tmp, ptr @var64_0
    ret void
}

define void @sdiv_i64() {
; CHECK-LABEL: sdiv_i64:
    %val0_tmp = load i64, ptr @var64_0
    %val1_tmp = load i64, ptr @var64_1
    %val4_tmp = sdiv i64 %val0_tmp, %val1_tmp
; CHECK: sdiv	{{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
    store volatile i64 %val4_tmp, ptr @var64_1
    ret void
}


define void @lsrv_i32() {
; CHECK-LABEL: lsrv_i32:
    %val0_tmp = load i32, ptr @var32_0
    %val1_tmp = load i32, ptr @var32_1
    %val2_tmp = add i32 1, %val1_tmp
    %val4_tmp = lshr i32 %val0_tmp, %val2_tmp
; CHECK: {{lsr|lsrv}} {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
    store volatile i32 %val4_tmp, ptr @var32_0
    ret void
}

define void @lslv_i32() {
; CHECK-LABEL: lslv_i32:
    %val0_tmp = load i32, ptr @var32_0
    %val1_tmp = load i32, ptr @var32_1
    %val2_tmp = add i32 1, %val1_tmp
    %val4_tmp = shl i32 %val0_tmp, %val2_tmp
; CHECK: {{lsl|lslv}} {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
    store volatile i32 %val4_tmp, ptr @var32_1
    ret void
}

define void @rorv_i32() {
; CHECK-LABEL: rorv_i32:
    %val0_tmp = load i32, ptr @var32_0
    %val6_tmp = load i32, ptr @var32_1
    %val1_tmp = add i32 1, %val6_tmp
    %val2_tmp = sub i32 32, %val1_tmp
    %val3_tmp = shl i32 %val0_tmp, %val2_tmp
    %val4_tmp = lshr i32 %val0_tmp, %val1_tmp
    %val5_tmp = or i32 %val3_tmp, %val4_tmp
; CHECK: {{ror|rorv}} {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
    store volatile i32 %val5_tmp, ptr @var32_0
    ret void
}

define void @asrv_i32() {
; CHECK-LABEL: asrv_i32:
    %val0_tmp = load i32, ptr @var32_0
    %val1_tmp = load i32, ptr @var32_1
    %val2_tmp = add i32 1, %val1_tmp
    %val4_tmp = ashr i32 %val0_tmp, %val2_tmp
; CHECK: {{asr|asrv}} {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
    store volatile i32 %val4_tmp, ptr @var32_1
    ret void
}

define void @sdiv_i32() {
; CHECK-LABEL: sdiv_i32:
    %val0_tmp = load i32, ptr @var32_0
    %val1_tmp = load i32, ptr @var32_1
    %val4_tmp = sdiv i32 %val0_tmp, %val1_tmp
; CHECK: sdiv	{{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
    store volatile i32 %val4_tmp, ptr @var32_1
    ret void
}

define void @udiv_i32() {
; CHECK-LABEL: udiv_i32:
    %val0_tmp = load i32, ptr @var32_0
    %val1_tmp = load i32, ptr @var32_1
    %val4_tmp = udiv i32 %val0_tmp, %val1_tmp
; CHECK: udiv	{{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
    store volatile i32 %val4_tmp, ptr @var32_0
    ret void
}

; The point of this test is that we may not actually see (shl GPR32:$Val, (zext GPR32:$Val2))
; in the DAG (the RHS may be natively 64-bit), but we should still use the lsl instructions.
define i32 @test_lsl32() {
; CHECK-LABEL: test_lsl32:

  %val = load i32, ptr @var32_0
  %ret = shl i32 1, %val
; CHECK: {{lsl|lslv}} {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}

  ret i32 %ret
}

define i32 @test_lsr32() {
; CHECK-LABEL: test_lsr32:

  %val = load i32, ptr @var32_0
  %ret = lshr i32 1, %val
; CHECK: {{lsr|lsrv}} {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}

  ret i32 %ret
}

define i32 @test_asr32(i32 %in) {
; CHECK-LABEL: test_asr32:

  %val = load i32, ptr @var32_0
  %ret = ashr i32 %in, %val
; CHECK: {{asr|asrv}} {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}

  ret i32 %ret
}
