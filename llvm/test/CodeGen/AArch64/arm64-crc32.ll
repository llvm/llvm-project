; RUN: llc -mtriple=arm64-eabi -mattr=+crc -o - %s | FileCheck %s
; RUN: llc -mtriple=arm64-eabi -mattr=+v8r -o - %s | FileCheck %s
; RUN: llc -mtriple=arm64-eabi -mcpu=cortex-a53 -mattr=+crc -o - %s | FileCheck %s

define i32 @test_crc32b(i32 %cur, i8 %next) {
; CHECK-LABEL: test_crc32b:
; CHECK: crc32b w0, w0, w1
  %bits = zext i8 %next to i32
  %val = call i32 @llvm.aarch64.crc32b(i32 %cur, i32 %bits)
  ret i32 %val
}

define i32 @test_crc32h(i32 %cur, i16 %next) {
; CHECK-LABEL: test_crc32h:
; CHECK: crc32h w0, w0, w1
  %bits = zext i16 %next to i32
  %val = call i32 @llvm.aarch64.crc32h(i32 %cur, i32 %bits)
  ret i32 %val
}

define i32 @test_crc32w(i32 %cur, i32 %next) {
; CHECK-LABEL: test_crc32w:
; CHECK: crc32w w0, w0, w1
  %val = call i32 @llvm.aarch64.crc32w(i32 %cur, i32 %next)
  ret i32 %val
}

define i32 @test_crc32x(i32 %cur, i64 %next) {
; CHECK-LABEL: test_crc32x:
; CHECK: crc32x w0, w0, x1
  %val = call i32 @llvm.aarch64.crc32x(i32 %cur, i64 %next)
  ret i32 %val
}

define i32 @test_crc32cb(i32 %cur, i8 %next) {
; CHECK-LABEL: test_crc32cb:
; CHECK: crc32cb w0, w0, w1
  %bits = zext i8 %next to i32
  %val = call i32 @llvm.aarch64.crc32cb(i32 %cur, i32 %bits)
  ret i32 %val
}

define i32 @test_crc32ch(i32 %cur, i16 %next) {
; CHECK-LABEL: test_crc32ch:
; CHECK: crc32ch w0, w0, w1
  %bits = zext i16 %next to i32
  %val = call i32 @llvm.aarch64.crc32ch(i32 %cur, i32 %bits)
  ret i32 %val
}

define i32 @test_crc32cw(i32 %cur, i32 %next) {
; CHECK-LABEL: test_crc32cw:
; CHECK: crc32cw w0, w0, w1
  %val = call i32 @llvm.aarch64.crc32cw(i32 %cur, i32 %next)
  ret i32 %val
}

define i32 @test_crc32cx(i32 %cur, i64 %next) {
; CHECK-LABEL: test_crc32cx:
; CHECK: crc32cx w0, w0, x1
  %val = call i32 @llvm.aarch64.crc32cx(i32 %cur, i64 %next)
  ret i32 %val
}

declare i32 @llvm.aarch64.crc32b(i32, i32)
declare i32 @llvm.aarch64.crc32h(i32, i32)
declare i32 @llvm.aarch64.crc32w(i32, i32)
declare i32 @llvm.aarch64.crc32x(i32, i64)

declare i32 @llvm.aarch64.crc32cb(i32, i32)
declare i32 @llvm.aarch64.crc32ch(i32, i32)
declare i32 @llvm.aarch64.crc32cw(i32, i32)
declare i32 @llvm.aarch64.crc32cx(i32, i64)
