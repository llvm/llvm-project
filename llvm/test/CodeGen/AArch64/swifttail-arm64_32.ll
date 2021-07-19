; RUN: llc -mtriple=arm64_32-apple-watchos %s -o - | FileCheck %s

declare swifttailcc void @pointer_align_callee([8 x i64], i32, i32, i32, i8*)
define swifttailcc void @pointer_align_caller(i8* swiftasync %as, i8* %in) "frame-pointer"="all" {
; CHECK-LABEL: pointer_align_caller:
; CHECK: sub sp, sp, #48
; CHECK: mov [[TWO:w[0-9]+]], #2
; CHECK: mov [[ZERO_ONE:x[0-9]+]], #4294967296
; CHECK: stp [[TWO]], w0, [x29, #24]
; CHECK: str [[ZERO_ONE]], [x29, #16]
; CHECK: add sp, sp, #32
; CHECK: b _pointer_align_callee
  alloca i32
  musttail call swifttailcc void @pointer_align_callee([8 x i64] undef, i32 0, i32 1, i32 2, i8* %in)
  ret void
}
