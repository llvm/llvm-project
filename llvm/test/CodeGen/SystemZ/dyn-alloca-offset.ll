; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i64 @llvm.get.dynamic.area.offset.i64()

declare void @use(i64)

define void @f1() {
; CHECK-LABEL: f1
; CHECK: la %r2, 160
; CHECK: brasl %r14, use
; CHECK: br %r14
  %tmp = alloca i64, align 32
  %dynamic_area_offset = call i64 @llvm.get.dynamic.area.offset.i64()
  call void @use(i64 %dynamic_area_offset)
  ret void
}

define void @f2(i64 %arg) {
; CHECK-LABEL: f2
; CHECK: la %r2, 160(%r2)
; CHECK: brasl %r14, use
; CHECK: br %r14
  %tmp = alloca i64, align 32
  %dynamic_area_offset = call i64 @llvm.get.dynamic.area.offset.i64()
  %param = add i64 %dynamic_area_offset, %arg
  call void @use(i64 %param)
  ret void
}

declare void @eatsalot(i64, i64, i64, i64, i64, i64)

define void @f3() {
; CHECK-LABEL: f3
; CHECK: la %r2, 168
; CHECK: brasl %r14, use
; CHECK: br %r14
  %tmp = alloca i64, align 32
  call void @eatsalot(i64 0, i64 0, i64 0, i64 0, i64 0, i64 0)
  %dynamic_area_offset = call i64 @llvm.get.dynamic.area.offset.i64()
  call void @use(i64 %dynamic_area_offset)
  ret void
}
