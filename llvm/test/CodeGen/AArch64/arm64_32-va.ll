; RUN: llc -mtriple=arm64_32-apple-ios %s -o - | FileCheck %s

define void @test_va_copy(ptr %dst, ptr %src) {
; CHECK-LABEL: test_va_copy:
; CHECK: ldr [[PTR:w[0-9]+]], [x1]
; CHECK: str [[PTR]], [x0]

  call void @llvm.va_copy(ptr %dst, ptr %src)
  ret void
}

define void @test_va_start(i32, ...)  {
; CHECK-LABEL: test_va_start
; CHECK: add x[[LIST:[0-9]+]], sp, #16
; CHECK: str w[[LIST]],
  %slot = alloca ptr, align 4
  call void @llvm.va_start(ptr %slot)
  ret void
}

define void @test_va_start_odd([8 x i64], i32, ...) {
; CHECK-LABEL: test_va_start_odd:
; CHECK: add x[[LIST:[0-9]+]], sp, #20
; CHECK: str w[[LIST]],
  %slot = alloca ptr, align 4
  call void @llvm.va_start(ptr %slot)
  ret void
}

define ptr @test_va_arg(ptr %list) {
; CHECK-LABEL: test_va_arg:
; CHECK: ldr w[[LOC:[0-9]+]], [x0]
; CHECK: add [[NEXTLOC:w[0-9]+]], w[[LOC]], #4
; CHECK: str [[NEXTLOC]], [x0]
; CHECK: ldr w0, [x[[LOC]]]
  %res = va_arg ptr %list, ptr
  ret ptr %res
}

define ptr @really_test_va_arg(ptr %list, i1 %tst) {
; CHECK-LABEL: really_test_va_arg:
; CHECK: ldr w[[LOC:[0-9]+]], [x0]
; CHECK: add [[NEXTLOC:w[0-9]+]], w[[LOC]], #4
; CHECK: str [[NEXTLOC]], [x0]
; CHECK: ldr w[[VAARG:[0-9]+]], [x[[LOC]]]
; CHECK: csel x0, x[[VAARG]], xzr
  %tmp = va_arg ptr %list, ptr
  %res = select i1 %tst, ptr %tmp, ptr null
  ret ptr %res
}

declare void @llvm.va_start(ptr) 

declare void @llvm.va_copy(ptr, ptr)
