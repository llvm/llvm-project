; RUN: opt %s -passes=instcombine -mtriple=riscv64-unknown-linux-gnu -S | FileCheck %s

declare signext i32 @memcmp(ptr, ptr, i64)

; Make sure we use signext attribute for the bcmp result.
define signext i32 @test_bcmp(ptr %mem1, ptr %mem2, i64 %size) {
; CHECK-LABEL: define {{[^@]+}}@test_bcmp(
; CHECK-NEXT:    [[BCMP:%.*]] = call i32 @bcmp(ptr [[MEM1:%.*]], ptr [[MEM2:%.*]], i64 [[SIZE:%.*]])
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[BCMP]], 0
; CHECK-NEXT:    [[ZEXT:%.*]] = zext i1 [[CMP]] to i32
; CHECK-NEXT:    ret i32 [[ZEXT]]
;
  %call = call signext i32 @memcmp(ptr %mem1, ptr %mem2, i64 %size)
  %cmp = icmp eq i32 %call, 0
  %zext = zext i1 %cmp to i32
  ret i32 %zext
}

; CHECK: declare signext i32 @bcmp(ptr captures(none), ptr captures(none), i64)
