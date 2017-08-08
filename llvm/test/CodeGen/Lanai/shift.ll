; RUN: llc < %s -march=lanai | FileCheck %s

; Test lowering of shifts.

define i32 @irs(i32 inreg %a) #0 {
entry:
  %shr = ashr i32 %a, 13
  ret i32 %shr
}
; CHECK-LABEL: irs
; CHECK: sha %r6, -0xd, %rv

define i32 @urs(i32 inreg %a) #0 {
entry:
  %shr = lshr i32 %a, 13
  ret i32 %shr
}
; CHECK-LABEL: urs
; CHECK: sh %r6, -0xd, %rv

define i32 @ls(i32 inreg %a) #0 {
entry:
  %shl = shl i32 %a, 13
  ret i32 %shl
}
; CHECK-LABEL: ls
; CHECK: sh %r6, 0xd, %rv

