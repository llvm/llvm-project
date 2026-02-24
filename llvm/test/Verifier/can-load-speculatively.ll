; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare i1 @llvm.can.load.speculatively.p0(ptr, i64)

; Test that constant size must be a positive power of 2

define i1 @test_size_zero(ptr %ptr) {
; CHECK: llvm.can.load.speculatively size must be a positive power of 2
; CHECK-NEXT: %res = call i1 @llvm.can.load.speculatively.p0(ptr %ptr, i64 0)
  %res = call i1 @llvm.can.load.speculatively.p0(ptr %ptr, i64 0)
  ret i1 %res
}

define i1 @test_non_power_of_2(ptr %ptr) {
; CHECK: llvm.can.load.speculatively size must be a positive power of 2
; CHECK-NEXT: %res = call i1 @llvm.can.load.speculatively.p0(ptr %ptr, i64 3)
  %res = call i1 @llvm.can.load.speculatively.p0(ptr %ptr, i64 3)
  ret i1 %res
}
