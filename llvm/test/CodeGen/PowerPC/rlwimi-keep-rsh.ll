; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s
; Formerly dropped the RHS of %tmp6 when constructing rlwimi.
; 7346117

@foo = external global i32

define void @xxx(i32 %a, i32 %b, i32 %c, i32 %d) nounwind optsize {
; CHECK: xxx:
; CHECK: or
; CHECK: and
; CHECK: rlwimi
entry:
  %tmp0 = ashr i32 %d, 31
  %tmp1 = and i32 %tmp0, 255
  %tmp2 = xor i32 %tmp1, 255
  %tmp3 = ashr i32 %b, 31
  %tmp4 = ashr i32 %a, 4
  %tmp5 = or i32 %tmp3, %tmp4
  %tmp6 = and i32 %tmp2, %tmp5
  %tmp7 = shl i32 %c, 8
  %tmp8 = or i32 %tmp6, %tmp7
  store i32 %tmp8, i32* @foo, align 4
  br label %return

return:
  ret void
; CHECK: blr
}
