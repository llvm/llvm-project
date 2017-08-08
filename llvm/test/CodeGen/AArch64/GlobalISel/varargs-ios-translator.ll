; RUN: llc -mtriple=aarch64-apple-ios -stop-after=instruction-select -global-isel -verify-machineinstrs %s -o - | FileCheck %s

define void @test_varargs_sentinel(i8* %list, i64, i64, i64, i64, i64, i64, i64,
                                   i32, ...) {
; CHECK-LABEL: name: test_varargs_sentinel
; CHECK: fixedStack:
; CHECK:   - { id: [[VARARGS_SLOT:[0-9]+]], type: default, offset: 8
; CHECK: body:
; CHECK:   [[LIST:%[0-9]+]] = COPY %x0
; CHECK:   [[VARARGS_AREA:%[0-9]+]] = ADDXri %fixed-stack.[[VARARGS_SLOT]], 0, 0
; CHECK:   STRXui [[VARARGS_AREA]], [[LIST]], 0 :: (store 8 into %ir.list, align 0)
  call void @llvm.va_start(i8* %list)
  ret void
}

declare void @llvm.va_start(i8*)
