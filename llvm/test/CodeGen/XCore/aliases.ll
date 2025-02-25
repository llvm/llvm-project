; RUN: llc < %s -mtriple=xcore | FileCheck %s
define void @a_val() nounwind {
  ret void
}
@b_val = constant i32 42, section ".cp.rodata"
@c_val = global i32 42

@a = alias void (), ptr @a_val
@b = alias i32, ptr @b_val
@c = alias i32, ptr @c_val

; CHECK-LABEL: a_addr:
; CHECK: ldap r11, a
; CHECK: retsp
define ptr @a_addr() nounwind {
entry:
  ret ptr @a
}

; CHECK-LABEL: b_addr:
; CHECK: ldaw r11, cp[b]
; CHECK: retsp
define ptr @b_addr() nounwind {
entry:
  ret ptr @b
}

; CHECK-LABEL: c_addr:
; CHECK: ldaw r0, dp[c]
; CHECK: retsp
define ptr @c_addr() nounwind {
entry:
  ret ptr @c
}
