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

