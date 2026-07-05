; RUN: llc -mtriple=thumbv7 -mattr=+crc -o - %s | FileCheck %s
; RUN: llc -mtriple=thumbv8 -o - %s | FileCheck %s
; RUN: llc -mtriple=armv7 -mattr=+crc -o - %s | FileCheck %s
; RUN: llc -mtriple=armv8 -o - %s | FileCheck %s

define i32 @test_crc32b(i32 %cur, i8 %next) {
; CHECK-LABEL: test_crc32b:
; CHECK: crc32b r0, r0, r1
  %bits = zext i8 %next to i32
  %val = call i32 @llvm.arm.crc32b(i32 %cur, i32 %bits)
  ret i32 %val
}

define i32 @test_crc32h(i32 %cur, i16 %next) {
; CHECK-LABEL: test_crc32h:
; CHECK: crc32h r0, r0, r1
  %bits = zext i16 %next to i32
  %val = call i32 @llvm.arm.crc32h(i32 %cur, i32 %bits)
  ret i32 %val
}

define i32 @test_crc32w(i32 %cur, i32 %next) {
; CHECK-LABEL: test_crc32w:
; CHECK: crc32w r0, r0, r1
  %val = call i32 @llvm.arm.crc32w(i32 %cur, i32 %next)
  ret i32 %val
}

define i32 @test_crc32cb(i32 %cur, i8 %next) {
; CHECK-LABEL: test_crc32cb:
; CHECK: crc32cb r0, r0, r1
  %bits = zext i8 %next to i32
  %val = call i32 @llvm.arm.crc32cb(i32 %cur, i32 %bits)
  ret i32 %val
}

define i32 @test_crc32ch(i32 %cur, i16 %next) {
; CHECK-LABEL: test_crc32ch:
; CHECK: crc32ch r0, r0, r1
  %bits = zext i16 %next to i32
  %val = call i32 @llvm.arm.crc32ch(i32 %cur, i32 %bits)
  ret i32 %val
}

define i32 @test_crc32cw(i32 %cur, i32 %next) {
; CHECK-LABEL: test_crc32cw:
; CHECK: crc32cw r0, r0, r1
  %val = call i32 @llvm.arm.crc32cw(i32 %cur, i32 %next)
  ret i32 %val
}


declare i32 @llvm.arm.crc32b(i32, i32)
declare i32 @llvm.arm.crc32h(i32, i32)
declare i32 @llvm.arm.crc32w(i32, i32)
declare i32 @llvm.arm.crc32x(i32, i64)

declare i32 @llvm.arm.crc32cb(i32, i32)
declare i32 @llvm.arm.crc32ch(i32, i32)
declare i32 @llvm.arm.crc32cw(i32, i32)
declare i32 @llvm.arm.crc32cx(i32, i64)
