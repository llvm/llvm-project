; RUN: opt -passes=asan %s -S | FileCheck %s

;; Punt AddressSanitizer::instrumentMemIntrinsics out for MemIntrinsics
;; that need write to unsupported registers on X86
;; PR124238: https://www.github.com/llvm/llvm-project/issues/124238

target triple = "x86_64-unknown-linux-gnu"

$.str.658906a285b7a0f82dabd9915e07848c = comdat any
@.str = internal constant { [2 x i8], [30 x i8] } { [2 x i8] c"x\00", [30 x i8] zeroinitializer }, comdat($.str.658906a285b7a0f82dabd9915e07848c), align 32
@0 = private alias { [2 x i8], [30 x i8] }, ptr @.str

define void @test_memcpy(i64 noundef %addr) sanitize_address #0 {
entry:
  %addr.addr = alloca i64, align 8
  store i64 %addr, ptr %addr.addr, align 8
  %0 = load i64, ptr %addr.addr, align 8
  %1 = inttoptr i64 %0 to ptr addrspace(257)
  call void @llvm.memcpy.p257.p0.i64(ptr addrspace(257) align 1 %1, ptr align 1 @.str, i64 1, i1 false)
; CHECK: llvm.memcpy
  %2 = load i64, ptr %addr.addr, align 8
  %3 = inttoptr i64 %2 to ptr addrspace(256)
  call void @llvm.memcpy.p256.p0.i64(ptr addrspace(256) align 1 %3, ptr align 1 @.str, i64 1, i1 false)
; CHECK: llvm.memcpy
  ret void
}

define void @test_memset(i64 noundef %addr) sanitize_address #0 {
entry:
  %addr.addr = alloca i64, align 8
  store i64 %addr, ptr %addr.addr, align 8
  %0 = load i64, ptr %addr.addr, align 8
  %1 = inttoptr i64 %0 to ptr addrspace(257)
  call void @llvm.memset.p257.i64(ptr addrspace(257) align 1 %1, i8 0, i64 1, i1 false)
; CHECK: llvm.memset
  %2 = load i64, ptr %addr.addr, align 8
  %3 = inttoptr i64 %2 to ptr addrspace(256)
  call void @llvm.memset.p256.i64(ptr addrspace(256) align 1 %3, i8 0, i64 1, i1 false)
; CHECK: llvm.memset
  ret void
}

define void @test_memmove(i64 noundef %addr) sanitize_address #0 {
entry:
  %addr.addr = alloca i64, align 8
  store i64 %addr, ptr %addr.addr, align 8
  %0 = load i64, ptr %addr.addr, align 8
  %1 = inttoptr i64 %0 to ptr addrspace(257)
  %2 = load i64, ptr %addr.addr, align 8
  %3 = inttoptr i64 %2 to ptr
  call void @llvm.memmove.p257.p0.i64(ptr addrspace(257) align 1 %1, ptr align 1 %3, i64 1, i1 false)
; CHECK: llvm.memmove
  %4 = load i64, ptr %addr.addr, align 8
  %5 = inttoptr i64 %4 to ptr addrspace(256)
  %6 = load i64, ptr %addr.addr, align 8
  %7 = inttoptr i64 %6 to ptr
  call void @llvm.memmove.p256.p0.i64(ptr addrspace(256) align 1 %5, ptr align 1 %7, i64 1, i1 false)
; CHECK: llvm.memmove
  ret void
}
