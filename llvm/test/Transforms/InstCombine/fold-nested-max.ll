; RUN: opt -passes=instcombine -S < %s | FileCheck %s

define i64 @test1(i64 %x) {
  %1 = call i64 @llvm.umax.i64(i64 %x, i64 4)
  %2 = shl i64 %1, 1
  %3 = call i64 @llvm.umax.i64(i64 %2, i64 16)
  ret i64 %3
}

; CHECK-LABEL: @test1
; CHECK: shl i64 %x, 1
; CHECK: call i64 @llvm.umax.i64(i64 {{.*}}, i64 16)