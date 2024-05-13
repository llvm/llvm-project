; RUN: llc -mtriple=arm64-darwin-unknown < %s | FileCheck %s

%T = type { i32, i32, i32, i32 }

; Test if the constant base address gets only materialized once.
define i32 @test1() nounwind {
; CHECK-LABEL:  test1
; CHECK:        mov  w8, #49152
; CHECK-NEXT:   movk  w8, #1039, lsl #16
; CHECK-NEXT:   ldp w9, w10, [x8, #4]
; CHECK:        ldr w8, [x8, #12]
  %at = inttoptr i64 68141056 to ptr
  %o1 = getelementptr %T, ptr %at, i32 0, i32 1
  %t1 = load i32, ptr %o1
  %o2 = getelementptr %T, ptr %at, i32 0, i32 2
  %t2 = load i32, ptr %o2
  %a1 = add i32 %t1, %t2
  %o3 = getelementptr %T, ptr %at, i32 0, i32 3
  %t3 = load i32, ptr %o3
  %a2 = add i32 %a1, %t3
  ret i32 %a2
}

