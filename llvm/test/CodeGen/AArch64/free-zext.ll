; RUN: llc -verify-machineinstrs -o - %s -mtriple=arm64-apple-ios7.0 | FileCheck %s

define i64 @test_free_zext(ptr %a, ptr %b) {
; CHECK-LABEL: test_free_zext:
; CHECK-DAG: ldrb w[[A:[0-9]+]], [x0]
; CHECK: ldrh w[[B:[0-9]+]], [x1]
; CHECK: add x0, x[[B]], x[[A]]
  %1 = load i8, ptr %a, align 1
  %conv = zext i8 %1 to i64
  %2 = load i16, ptr %b, align 2
  %conv1 = zext i16 %2 to i64
  %add = add nsw i64 %conv1, %conv
  ret i64 %add
}

define void @test_free_zext2(ptr %ptr, ptr %dst1, ptr %dst2) {
; CHECK-LABEL: test_free_zext2:
; CHECK: ldrh w[[A:[0-9]+]], [x0]
; CHECK-NOT: and x
; CHECK: str w[[A]], [x1]
; CHECK: str x[[A]], [x2]
  %load = load i32, ptr %ptr, align 8
  %load16 = and i32 %load, 65535
  %load64 = zext i32 %load16 to i64
  store i32 %load16, ptr %dst1, align 4
  store i64 %load64, ptr %dst2, align 8
  ret void
}

; Test for CodeGenPrepare::optimizeLoadExt(): simple case: two loads
; feeding a phi that zext's each loaded value.
define i32 @test_free_zext3(ptr %ptr, ptr %ptr2, ptr %dst, i32 %c) {
; CHECK-LABEL: test_free_zext3:
bb1:
; CHECK: ldrh [[REG:w[0-9]+]]
; CHECK-NOT: and {{w[0-9]+}}, [[REG]], #0xffff
  %tmp1 = load i32, ptr %ptr, align 4
  %cmp = icmp ne i32 %c, 0
  br i1 %cmp, label %bb2, label %bb3
bb2:
; CHECK: ldrh [[REG2:w[0-9]+]]
; CHECK-NOT: and {{w[0-9]+}}, [[REG2]], #0xffff
  %tmp2 = load i32, ptr %ptr2, align 4
  br label %bb3
bb3:
  %tmp3 = phi i32 [ %tmp1, %bb1 ], [ %tmp2, %bb2 ]
; CHECK-NOT: and {{w[0-9]+}}, {{w[0-9]+}}, #0xffff
  %tmpand = and i32 %tmp3, 65535
  ret i32 %tmpand
}

; Test for CodeGenPrepare::optimizeLoadExt(): check case of zext-able
; load feeding a phi in the same block.
define void @test_free_zext4(ptr %ptr, ptr %ptr2, ptr %dst) {
; CHECK-LABEL: test_free_zext4:
; CHECK: ldrh [[REG:w[0-9]+]]
; TODO: fix isel to remove final and XCHECK-NOT: and {{w[0-9]+}}, {{w[0-9]+}}, #0xffff
; CHECK: ldrh [[REG:w[0-9]+]]
bb1:
  %load1 = load i32, ptr %ptr, align 4
  br label %loop
loop:
  %phi = phi i32 [ %load1, %bb1 ], [ %load2, %loop ]
  %and = and i32 %phi, 65535
  store i32 %and, ptr %dst, align 4
  %load2 = load i32, ptr %ptr2, align 4
  %cmp = icmp ne i32 %and, 0
  br i1 %cmp, label %loop, label %end
end:
  ret void
}
