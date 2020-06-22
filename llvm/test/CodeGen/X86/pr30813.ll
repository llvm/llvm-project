; RUN: llc -mtriple=x86_64-linux-gnu -O0 %s -o - | FileCheck %s
; CHECK: patatino:
; CHECK:         .cfi_startproc
; CHECK:         movzwl  (%rax), %e[[REG0:[abcd]x]]
; CHECK:         movq    %r[[REG0]], ({{%r[abcd]x}})
; CHECK:         retq

define void @patatino() {
  %tmp = load i16, i16* undef, align 8
  %conv18098 = sext i16 %tmp to i64
  %and1 = and i64 %conv18098, -1
  %cmp = icmp ult i64 -1, undef
  %conv = sext i1 %cmp to i64
  %load1 = load i48, i48* undef, align 8
  %bf.cast18158 = sext i48 %load1 to i64
  %conv18159 = trunc i64 %bf.cast18158 to i32
  %conv18160 = sext i32 %conv18159 to i64
  %div18162 = udiv i64 %conv, %conv18160
  %and18163 = and i64 %conv18098, %div18162
  %shr18164 = lshr i64 %and1, %and18163
  %conv18165 = trunc i64 %shr18164 to i16
  %conv18166 = zext i16 %conv18165 to i64
  store i64 %conv18166, i64* undef, align 8
  store i48 undef, i48* undef, align 8
  ret void
}
