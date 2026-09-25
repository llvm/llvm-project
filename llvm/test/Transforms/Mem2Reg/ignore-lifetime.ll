; RUN: opt -passes=mem2reg -S -o - < %s | FileCheck %s

declare void @llvm.lifetime.start.p0(ptr nocapture %ptr)
declare void @llvm.lifetime.end.p0(ptr nocapture %ptr)

define void @test1() {
; CHECK: test1
; CHECK-NOT: alloca
  %A = alloca i32
  call void @llvm.lifetime.start.p0(ptr %A)
  store i32 1, ptr %A
  call void @llvm.lifetime.end.p0(ptr %A)
  ret void
}

define void @test2() {
; CHECK: test2
; CHECK-NOT: alloca
  %A = alloca {i8, i16}
  call void @llvm.lifetime.start.p0(ptr %A)
  store {i8, i16} zeroinitializer, ptr %A
  call void @llvm.lifetime.end.p0(ptr %A)
  ret void
}

; Verify that multiple single-block allocas with lifetime markers in the same
; block are promoted accurately when removeIntrinsicUsers runs as a pre-pass
; before LargeBlockInfo caches instruction indices.
define i32 @multiple_single_block_allocas_with_lifetime(i32 %x, i32 %y) {
; CHECK-LABEL: define i32 @multiple_single_block_allocas_with_lifetime(
; CHECK-SAME: i32 [[X:%.*]], i32 [[Y:%.*]]) {
; CHECK-NEXT:    [[SUM:%.*]] = add i32 [[X]], [[Y]]
; CHECK-NEXT:    ret i32 [[SUM]]
;
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr %a)
  store i32 %x, ptr %a, align 4
  %va = load i32, ptr %a, align 4
  call void @llvm.lifetime.end.p0(ptr %a)
  call void @llvm.lifetime.start.p0(ptr %b)
  store i32 %y, ptr %b, align 4
  %vb = load i32, ptr %b, align 4
  call void @llvm.lifetime.end.p0(ptr %b)
  %sum = add i32 %va, %vb
  ret i32 %sum
}
