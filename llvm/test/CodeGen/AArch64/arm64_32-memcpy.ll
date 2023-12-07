; RUN: llc -mtriple=arm64_32-apple-ios9.0 -o - %s | FileCheck %s

define i64 @test_memcpy(ptr %addr, ptr %src, i1 %tst) minsize {
; CHECK-LABEL: test_memcpy:
; CHECK: ldr [[VAL64:x[0-9]+]], [x0]
; [...]
; CHECK: and x0, [[VAL64]], #0xffffffff
; CHECK: bl _memcpy

  %val64 = load i64, ptr %addr
  br i1 %tst, label %true, label %false

true:
  ret i64 %val64

false:
  %val32 = trunc i64 %val64 to i32
  %val.ptr = inttoptr i32 %val32 to ptr
  call void @llvm.memcpy.p0.p0.i32(ptr %val.ptr, ptr %src, i32 128, i32 0, i1 1)
  ret i64 undef
}

define i64 @test_memmove(ptr %addr, ptr %src, i1 %tst) minsize {
; CHECK-LABEL: test_memmove:
; CHECK: ldr [[VAL64:x[0-9]+]], [x0]
; [...]
; CHECK: and x0, [[VAL64]], #0xffffffff
; CHECK: bl _memmove

  %val64 = load i64, ptr %addr
  br i1 %tst, label %true, label %false

true:
  ret i64 %val64

false:
  %val32 = trunc i64 %val64 to i32
  %val.ptr = inttoptr i32 %val32 to ptr
  call void @llvm.memmove.p0.p0.i32(ptr %val.ptr, ptr %src, i32 128, i32 0, i1 1)
  ret i64 undef
}

define i64 @test_memset(ptr %addr, ptr %src, i1 %tst) minsize {
; CHECK-LABEL: test_memset:
; CHECK: ldr [[VAL64:x[0-9]+]], [x0]
; [...]
; CHECK: and x0, [[VAL64]], #0xffffffff
; CHECK: bl _memset

  %val64 = load i64, ptr %addr
  br i1 %tst, label %true, label %false

true:
  ret i64 %val64

false:
  %val32 = trunc i64 %val64 to i32
  %val.ptr = inttoptr i32 %val32 to ptr
  call void @llvm.memset.p0.i32(ptr %val.ptr, i8 42, i32 256, i32 0, i1 1)
  ret i64 undef
}

declare void @llvm.memcpy.p0.p0.i32(ptr, ptr, i32, i32, i1)
declare void @llvm.memmove.p0.p0.i32(ptr, ptr, i32, i32, i1)
declare void @llvm.memset.p0.i32(ptr, i8, i32, i32, i1)

