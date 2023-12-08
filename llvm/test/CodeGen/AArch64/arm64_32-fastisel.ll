; RUN: llc -mtriple=arm64_32-apple-ios -O0 -fast-isel %s -o - | FileCheck %s
@var = global ptr null

define void @test_store_release_ptr() {
; CHECK-LABEL: test_store_release_ptr
; CHECK: mov [[ZERO:w[0-9]+]], wzr
; CHECK: stlr [[ZERO]]
  store atomic ptr null, ptr @var release, align 4
  br label %next

next:
  ret void
}

declare [2 x i32] @callee()

define void @test_struct_return(ptr %addr) {
; CHECK-LABEL: test_struct_return:
; CHECK: bl _callee
; CHECK: x[[COPYX0:[0-9]+]], x0
; CHECK-DAG: lsr [[HI:x[0-9]+]], x[[COPYX0]], #32
; CHECK-DAG: str w[[COPYX0]]
  %res = call [2 x i32] @callee()
  %res.0 = extractvalue [2 x i32] %res, 0
  store i32 %res.0, ptr %addr
  %res.1 = extractvalue [2 x i32] %res, 1
  store i32 %res.1, ptr %addr
  ret void
}

define ptr @test_ret_ptr(i64 %in) {
; CHECK-LABEL: test_ret_ptr:
; CHECK: add [[TMP:x[0-9]]], x0, #1
; CHECK: and x0, [[TMP]], #0xffffffff

  %sum = add i64 %in, 1
  %res = inttoptr i64 %sum to ptr
  ret ptr %res
}

; Handled by SDAG because the struct confuses FastISel, which is fine.
define {ptr} @test_ret_ptr_struct(i64 %in) {
; CHECK-LABEL: test_ret_ptr_struct:
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, #1

  %sum = add i64 %in, 1
  %res.ptr = inttoptr i64 %sum to ptr
  %res = insertvalue {ptr} undef, ptr %res.ptr, 0
  ret {ptr} %res
}
