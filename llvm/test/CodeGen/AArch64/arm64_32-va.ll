; RUN: llc -mtriple=arm64_32-apple-ios %s -o - | FileCheck %s

define void @test_va_copy(i8* %dst, i8* %src) {
; CHECK-LABEL: test_va_copy:
; CHECK: ldr [[PTR:w[0-9]+]], [x1]
; CHECK: str [[PTR]], [x0]

  call void @llvm.va_copy(i8* %dst, i8* %src)
  ret void
}

define void @test_va_start(i32, ...)  {
; CHECK-LABEL: test_va_start
; CHECK: add x[[LIST:[0-9]+]], sp, #16
; CHECK: str w[[LIST]],
  %slot = alloca i8*, align 4
  %list = bitcast i8** %slot to i8*
  call void @llvm.va_start(i8* %list)
  ret void
}

define void @test_va_start_odd([8 x i64], i32, ...) {
; CHECK-LABEL: test_va_start_odd:
; CHECK: add x[[LIST:[0-9]+]], sp, #20
; CHECK: str w[[LIST]],
  %slot = alloca i8*, align 4
  %list = bitcast i8** %slot to i8*
  call void @llvm.va_start(i8* %list)
  ret void
}

define i8* @test_va_arg(i8** %list) {
; CHECK-LABEL: test_va_arg:
; CHECK: ldr w[[LOC:[0-9]+]], [x0]
; CHECK: add [[NEXTLOC:w[0-9]+]], w[[LOC]], #4
; CHECK: str [[NEXTLOC]], [x0]
; CHECK: ldr w0, [x[[LOC]]]
  %res = va_arg i8** %list, i8*
  ret i8* %res
}

define i8* @really_test_va_arg(i8** %list, i1 %tst) {
; CHECK-LABEL: really_test_va_arg:
; CHECK: ldr w[[LOC:[0-9]+]], [x0]
; CHECK: add [[NEXTLOC:w[0-9]+]], w[[LOC]], #4
; CHECK: str [[NEXTLOC]], [x0]
; CHECK: ldr w[[VAARG:[0-9]+]], [x[[LOC]]]
; CHECK: csel x0, x[[VAARG]], xzr
  %tmp = va_arg i8** %list, i8*
  %res = select i1 %tst, i8* %tmp, i8* null
  ret i8* %res
}

declare void @llvm.va_start(i8*) 

declare void @llvm.va_copy(i8*, i8*)
