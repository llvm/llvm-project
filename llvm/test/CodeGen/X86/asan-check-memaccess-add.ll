; RUN: llc < %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

define void @load1(i8* nocapture readonly %x) {
; CHECK:              pushq   %rax
; CHECK-NOT:          push    %rbp
; CHECK:              callq   __asan_check_load_add_1_[[REG1:.*]]
; CHECK:              callq   __asan_check_store_add_1_[[REG1]]
; CHECK-NOT:          pop     %rbp
; CHECK:              popq    %rax
  call void @llvm.asan.check.memaccess(i8* %x, i32 0)
  call void @llvm.asan.check.memaccess(i8* %x, i32 32)
  ret void
}

define void @load2(i16* nocapture readonly %x) {
; CHECK:              pushq   %rax
; CHECK-NOT:          push    %rbp
; CHECK:              callq   __asan_check_load_add_2_[[REG2:.*]]
; CHECK:              callq   __asan_check_store_add_2_[[REG2]]
; CHECK-NOT:          pop     %rbp
; CHECK:              popq    %rax
  %1 = ptrtoint i16* %x to i64
  %2 = bitcast i16* %x to i8*
  call void @llvm.asan.check.memaccess(i8* %2, i32 2)
  call void @llvm.asan.check.memaccess(i8* %2, i32 34)
  ret void
}

define void @load4(i32* nocapture readonly %x) {
; CHECK:              pushq   %rax
; CHECK-NOT:          push    %rbp
; CHECK:              callq   __asan_check_load_add_4_[[REG4:.*]]
; CHECK:              callq   __asan_check_store_add_4_[[REG4]]
; CHECK-NOT:          pop     %rbp
; CHECK:              popq    %rax
  %1 = ptrtoint i32* %x to i64
  %2 = bitcast i32* %x to i8*
  call void @llvm.asan.check.memaccess(i8* %2, i32 4)
  call void @llvm.asan.check.memaccess(i8* %2, i32 36)
  ret void
}
define void @load8(i64* nocapture readonly %x) {
; CHECK:              pushq   %rax
; CHECK-NOT:          push    %rbp
; CHECK:              callq   __asan_check_load_add_8_[[REG8:.*]]
; CHECK:              callq   __asan_check_store_add_8_[[REG8]]
; CHECK-NOT:          pop     %rbp
; CHECK:              popq    %rax
  %1 = ptrtoint i64* %x to i64
  %2 = bitcast i64* %x to i8*
  call void @llvm.asan.check.memaccess(i8* %2, i32 6)
  call void @llvm.asan.check.memaccess(i8* %2, i32 38)
  ret void
}

define void @load16(i128* nocapture readonly %x) {
; CHECK:              pushq   %rax
; CHECK-NOT:          push    %rbp
; CHECK:              callq   __asan_check_load_add_16_[[REG16:.*]]
; CHECK:              callq   __asan_check_store_add_16_[[REG16]]
; CHECK-NOT:          pop     %rbp
; CHECK:              popq    %rax
  %1 = ptrtoint i128* %x to i64
  %2 = bitcast i128* %x to i8*
  call void @llvm.asan.check.memaccess(i8* %2, i32 8)
  call void @llvm.asan.check.memaccess(i8* %2, i32 40)
  ret void
}

declare void @llvm.asan.check.memaccess(i8*, i32 immarg)
