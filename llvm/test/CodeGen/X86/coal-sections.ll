; RUN: llc < %s -mtriple x86_64-apple-darwin | FileCheck %s

; Check that *coal* sections are not emitted.

; CHECK: .section  __TEXT,__text,regular,pure_instructions{{$}}
; CHECK-NEXT: .globl  _foo

; CHECK: .section  __TEXT,__const{{$}}
; CHECK-NEXT: .globl  _a

; CHECK: .section  __DATA,__data{{$}}
; CHECK-NEXT: .globl  _b

@a = weak_odr constant [4 x i32] [i32 1, i32 2, i32 3, i32 4], align 16
@b = weak global i32 5, align 4
@g = common global ptr null, align 8

; Function Attrs: nounwind ssp uwtable
define weak ptr @foo() {
entry:
  store ptr @a, ptr @g, align 8
  ret ptr @b
}
