; RUN: llc -verify-machineinstrs -o - %s -mtriple=arm64-apple-ios7.0 | FileCheck %s

define i32 @test_madd32(i32 %val0, i32 %val1, i32 %val2) {
; CHECK-LABEL: test_madd32:
  %mid = mul i32 %val1, %val2
  %res = add i32 %val0, %mid
; CHECK: madd {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  ret i32 %res
}

define i64 @test_madd64(i64 %val0, i64 %val1, i64 %val2) {
; CHECK-LABEL: test_madd64:
  %mid = mul i64 %val1, %val2
  %res = add i64 %val0, %mid
; CHECK: madd {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
  ret i64 %res
}

define i32 @test_msub32(i32 %val0, i32 %val1, i32 %val2) {
; CHECK-LABEL: test_msub32:
  %mid = mul i32 %val1, %val2
  %res = sub i32 %val0, %mid
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  ret i32 %res
}

define i64 @test_msub64(i64 %val0, i64 %val1, i64 %val2) {
; CHECK-LABEL: test_msub64:
  %mid = mul i64 %val1, %val2
  %res = sub i64 %val0, %mid
; CHECK: msub {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
  ret i64 %res
}

define i64 @test_smaddl(i64 %acc, i32 %val1, i32 %val2) {
; CHECK-LABEL: test_smaddl:
  %ext1 = sext i32 %val1 to i64
  %ext2 = sext i32 %val2 to i64
  %prod = mul i64 %ext1, %ext2
  %res = add i64 %acc, %prod
; CHECK: smaddl {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{x[0-9]+}}
  ret i64 %res
}

define i64 @test_smsubl(i64 %acc, i32 %val1, i32 %val2) {
; CHECK-LABEL: test_smsubl:
  %ext1 = sext i32 %val1 to i64
  %ext2 = sext i32 %val2 to i64
  %prod = mul i64 %ext1, %ext2
  %res = sub i64 %acc, %prod
; CHECK: smsubl {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{x[0-9]+}}
  ret i64 %res
}

define i64 @test_umaddl(i64 %acc, i32 %val1, i32 %val2) {
; CHECK-LABEL: test_umaddl:
  %ext1 = zext i32 %val1 to i64
  %ext2 = zext i32 %val2 to i64
  %prod = mul i64 %ext1, %ext2
  %res = add i64 %acc, %prod
; CHECK: umaddl {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{x[0-9]+}}
  ret i64 %res
}

define i64 @test_umsubl(i64 %acc, i32 %val1, i32 %val2) {
; CHECK-LABEL: test_umsubl:
  %ext1 = zext i32 %val1 to i64
  %ext2 = zext i32 %val2 to i64
  %prod = mul i64 %ext1, %ext2
  %res = sub i64 %acc, %prod
; CHECK: umsubl {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{x[0-9]+}}
  ret i64 %res
}

define i64 @test_smulh(i64 %lhs, i64 %rhs) {
; CHECK-LABEL: test_smulh:
  %ext1 = sext i64 %lhs to i128
  %ext2 = sext i64 %rhs to i128
  %res = mul i128 %ext1, %ext2
  %high = lshr i128 %res, 64
  %val = trunc i128 %high to i64
; CHECK: smulh {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
  ret i64 %val
}

define i64 @test_umulh(i64 %lhs, i64 %rhs) {
; CHECK-LABEL: test_umulh:
  %ext1 = zext i64 %lhs to i128
  %ext2 = zext i64 %rhs to i128
  %res = mul i128 %ext1, %ext2
  %high = lshr i128 %res, 64
  %val = trunc i128 %high to i64
; CHECK: umulh {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
  ret i64 %val
}

define i32 @test_mul32(i32 %lhs, i32 %rhs) {
; CHECK-LABEL: test_mul32:
  %res = mul i32 %lhs, %rhs
; CHECK: mul {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  ret i32 %res
}

define i64 @test_mul64(i64 %lhs, i64 %rhs) {
; CHECK-LABEL: test_mul64:
  %res = mul i64 %lhs, %rhs
; CHECK: mul {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
  ret i64 %res
}

define i32 @test_mneg32(i32 %lhs, i32 %rhs) {
; CHECK-LABEL: test_mneg32:
  %prod = mul i32 %lhs, %rhs
  %res = sub i32 0, %prod
; CHECK: mneg {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  ret i32 %res
}

define i64 @test_mneg64(i64 %lhs, i64 %rhs) {
; CHECK-LABEL: test_mneg64:
  %prod = mul i64 %lhs, %rhs
  %res = sub i64 0, %prod
; CHECK: mneg {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
  ret i64 %res
}

define i64 @test_smull(i32 %lhs, i32 %rhs) {
; CHECK-LABEL: test_smull:
  %ext1 = sext i32 %lhs to i64
  %ext2 = sext i32 %rhs to i64
  %res = mul i64 %ext1, %ext2
; CHECK: smull {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  ret i64 %res
}

define i64 @test_umull(i32 %lhs, i32 %rhs) {
; CHECK-LABEL: test_umull:
  %ext1 = zext i32 %lhs to i64
  %ext2 = zext i32 %rhs to i64
  %res = mul i64 %ext1, %ext2
; CHECK: umull {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  ret i64 %res
}

define i64 @test_smnegl(i32 %lhs, i32 %rhs) {
; CHECK-LABEL: test_smnegl:
  %ext1 = sext i32 %lhs to i64
  %ext2 = sext i32 %rhs to i64
  %prod = mul i64 %ext1, %ext2
  %res = sub i64 0, %prod
; CHECK: smnegl {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  ret i64 %res
}

define i64 @test_umnegl(i32 %lhs, i32 %rhs) {
; CHECK-LABEL: test_umnegl:
  %ext1 = zext i32 %lhs to i64
  %ext2 = zext i32 %rhs to i64
  %prod = mul i64 %ext1, %ext2
  %res = sub i64 0, %prod
; CHECK: umnegl {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  ret i64 %res
}

@a = common global i32 0, align 4
@b = common global i32 0, align 4
@c = common global i32 0, align 4

define void @test_mneg(){
; CHECK-LABEL: test_mneg:
  %1 = load i32, ptr @a, align 4
  %2 = load i32, ptr @b, align 4
  %3 = sub i32 0, %1
  %4 = mul i32 %2, %3
  store i32 %4, ptr @c, align 4
; CHECK: mneg {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  ret void
}
