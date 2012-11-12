; RUN: llc < %s -O0 | FileCheck %s
target triple = "x86_64-apple-macosx"

declare x86_fp80 @x1(i32) nounwind
declare i32 @x2(x86_fp80, x86_fp80) nounwind

; Keep track of the return value.
; CHECK: test1
; CHECK: x1
; Pass arguments on the stack.
; CHECK-NEXT: movq %rsp, [[RCX:%r..]]
; Copy constant-pool value.
; CHECK-NEXT: fldl LCPI
; CHECK-NEXT: fstpt 16([[RCX]])
; Copy x1 return value.
; CHECK-NEXT: fstpt ([[RCX]])
; CHECK-NEXT: x2
define i32 @test1() nounwind uwtable ssp {
entry:
  %call = call x86_fp80 (...)* bitcast (x86_fp80 (i32)* @x1 to x86_fp80 (...)*)(i32 -1)
  %call1 = call i32 @x2(x86_fp80 %call, x86_fp80 0xK401EFFFFFFFF00000000)
  ret i32 %call1
}

