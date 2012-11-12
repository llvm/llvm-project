; RUN: llc < %s -relocation-model=static -march=x86 | FileCheck %s

; This should produce two shll instructions, not any lea's.

target triple = "i686-apple-darwin8"
@Y = weak global i32 0          ; <i32*> [#uses=1]
@X = weak global i32 0          ; <i32*> [#uses=2]


define void @fn1() {
; CHECK: fn1:
; CHECK-NOT: ret
; CHECK-NOT: lea
; CHECK: shll $3
; CHECK-NOT: lea
; CHECK: ret

  %tmp = load i32* @Y             ; <i32> [#uses=1]
  %tmp1 = shl i32 %tmp, 3         ; <i32> [#uses=1]
  %tmp2 = load i32* @X            ; <i32> [#uses=1]
  %tmp3 = or i32 %tmp1, %tmp2             ; <i32> [#uses=1]
  store i32 %tmp3, i32* @X
  ret void
}

define i32 @fn2(i32 %X, i32 %Y) {
; CHECK: fn2:
; CHECK-NOT: ret
; CHECK-NOT: lea
; CHECK: shll $3
; CHECK-NOT: lea
; CHECK: ret

  %tmp2 = shl i32 %Y, 3           ; <i32> [#uses=1]
  %tmp4 = or i32 %tmp2, %X                ; <i32> [#uses=1]
  ret i32 %tmp4
}

