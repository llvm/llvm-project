; RUN: opt -S -demanded-bits -analyze < %s | FileCheck %s
; RUN: opt -S -disable-output -passes="print<demanded-bits>" < %s 2>&1 | FileCheck %s

; CHECK-DAG: DemandedBits: 0xff000000 for   %1 = or i32 %x, 1
; CHECK-DAG: DemandedBits: 0xff for   %2 = call i32 @llvm.bitreverse.i32(i32 %1)
; CHECK-DAG: DemandedBits: 0xff for   %3 = trunc i32 %2 to i8
define i8 @test_bswap(i32 %x) {
  %1 = or i32 %x, 1
  %2 = call i32 @llvm.bswap.i32(i32 %1)
  %3 = trunc i32 %2 to i8
  ret i8 %3
}
declare i32 @llvm.bswap.i32(i32)

; CHECK-DAG: DemandedBits: 0xff000000 for   %1 = or i32 %x, 1
; CHECK-DAG: DemandedBits: 0xff for   %2 = call i32 @llvm.bswap.i32(i32 %1)
; CHECK-DAG: DemandedBits: 0xff for   %3 = trunc i32 %2 to i8
define i8 @test_bitreverse(i32 %x) {
  %1 = or i32 %x, 1
  %2 = call i32 @llvm.bitreverse.i32(i32 %1)
  %3 = trunc i32 %2 to i8
  ret i8 %3
}
declare i32 @llvm.bitreverse.i32(i32)

; Funnel shifts
declare i32 @llvm.fshl.i32(i32, i32, i32)
declare i33 @llvm.fshr.i33(i33, i33, i33)

; CHECK-DAG: DemandedBits: 0xff for   %x2 = or i32 %x, 1
; CHECK-DAG: DemandedBits: 0xff000000 for   %y2 = or i32 %y, 1
; CHECK-DAG: DemandedBits: 0xffff for   %z = call i32 @llvm.fshl.i32(i32 %x2, i32 %y2, i32 8)
; CHECK-DAG: DemandedBits: 0xffffffff for   %r = and i32 %z, 65535
define i32 @test_fshl(i32 %x, i32 %y) {
  %x2 = or i32 %x, 1
  %y2 = or i32 %y, 1
  %z = call i32 @llvm.fshl.i32(i32 %x2, i32 %y2, i32 8)
  %r = and i32 %z, 65535
  ret i32 %r
}

; CHECK-DAG: DemandedBits: 0xff for   %x2 = or i33 %x, 1
; CHECK-DAG: DemandedBits: 0x1fe000000 for   %y2 = or i33 %y, 1
; CHECK-DAG: DemandedBits: 0xffff for   %z = call i33 @llvm.fshr.i33(i33 %x2, i33 %y2, i33 25)
; CHECK-DAG: DemandedBits: 0x1ffffffff for   %r = and i33 %z, 65535
define i33 @test_fshr(i33 %x, i33 %y) {
  %x2 = or i33 %x, 1
  %y2 = or i33 %y, 1
  %z = call i33 @llvm.fshr.i33(i33 %x2, i33 %y2, i33 25)
  %r = and i33 %z, 65535
  ret i33 %r
}

; CHECK-DAG: DemandedBits: 0xffff for   %x2 = or i32 %x, 1
; CHECK-DAG: DemandedBits: 0x0 for   %y2 = or i32 %y, 1
; CHECK-DAG: DemandedBits: 0xffff for   %z = call i32 @llvm.fshl.i32(i32 %x2, i32 %y2, i32 0)
; CHECK-DAG: DemandedBits: 0xffffffff for   %r = and i32 %z, 65535
define i32 @test_fshl_zero_shift(i32 %x, i32 %y) {
  %x2 = or i32 %x, 1
  %y2 = or i32 %y, 1
  %z = call i32 @llvm.fshl.i32(i32 %x2, i32 %y2, i32 0)
  %r = and i32 %z, 65535
  ret i32 %r
}

; CHECK-DAG: DemandedBits: 0x0 for   %x2 = or i33 %x, 1
; CHECK-DAG: DemandedBits: 0xffff for   %y2 = or i33 %y, 1
; CHECK-DAG: DemandedBits: 0xffff for   %z = call i33 @llvm.fshr.i33(i33 %x2, i33 %y2, i33 33)
; CHECK-DAG: DemandedBits: 0x1ffffffff for   %r = and i33 %z, 65535
define i33 @test_fshr_full_shift(i33 %x, i33 %y) {
  %x2 = or i33 %x, 1
  %y2 = or i33 %y, 1
  %z = call i33 @llvm.fshr.i33(i33 %x2, i33 %y2, i33 33)
  %r = and i33 %z, 65535
  ret i33 %r
}

; CHECK-DAG: DemandedBits: 0xffffffff for   %x2 = or i32 %x, 1
; CHECK-DAG: DemandedBits: 0xffffffff for   %y2 = or i32 %y, 1
; CHECK-DAG: DemandedBits: 0x1f for   %z2 = or i32 %z, 1
; CHECK-DAG: DemandedBits: 0xffff for   %f = call i32 @llvm.fshl.i32(i32 %x2, i32 %y2, i32 %z2)
; CHECK-DAG: DemandedBits: 0xffffffff for   %r = and i32 %f, 65535
define i32 @test_fshl_pow2_bitwidth(i32 %x, i32 %y, i32 %z) {
  %x2 = or i32 %x, 1
  %y2 = or i32 %y, 1
  %z2 = or i32 %z, 1
  %f = call i32 @llvm.fshl.i32(i32 %x2, i32 %y2, i32 %z2)
  %r = and i32 %f, 65535
  ret i32 %r
}

; CHECK-DAG: DemandedBits: 0x1ffffffff for   %x2 = or i33 %x, 1
; CHECK-DAG: DemandedBits: 0x1ffffffff for   %y2 = or i33 %y, 1
; CHECK-DAG: DemandedBits: 0x1ffffffff for   %z2 = or i33 %z, 1
; CHECK-DAG: DemandedBits: 0xffff for   %f = call i33 @llvm.fshr.i33(i33 %x2, i33 %y2, i33 %z2)
; CHECK-DAG: DemandedBits: 0x1ffffffff for   %r = and i33 %f, 65535
define i33 @test_fshr_non_pow2_bitwidth(i33 %x, i33 %y, i33 %z) {
  %x2 = or i33 %x, 1
  %y2 = or i33 %y, 1
  %z2 = or i33 %z, 1
  %f = call i33 @llvm.fshr.i33(i33 %x2, i33 %y2, i33 %z2)
  %r = and i33 %f, 65535
  ret i33 %r
}
