; RUN: llc -mtriple=arm64_32-apple-ios -O0 -fast-isel %s -o - | FileCheck %s
@var = global i8* null

define void @test_store_release_ptr() {
; CHECK-LABEL: test_store_release_ptr
; CHECK: mov [[ZERO:w[0-9]+]], wzr
; CHECK: stlr [[ZERO]]
  store atomic i8* null, i8** @var release, align 4
  br label %next

next:
  ret void
}

declare [2 x i32] @callee()

define void @test_struct_return(i32* %addr) {
; CHECK-LABEL: test_struct_return:
; CHECK: bl _callee
; CHECK-DAG: lsr [[HI:x[0-9]+]], x0, #32
; CHECK-DAG: str w0
  %res = call [2 x i32] @callee()
  %res.0 = extractvalue [2 x i32] %res, 0
  store i32 %res.0, i32* %addr
  %res.1 = extractvalue [2 x i32] %res, 1
  store i32 %res.1, i32* %addr
  ret void
}
